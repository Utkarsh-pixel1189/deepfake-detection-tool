"""
sequence_dataset.py

PyTorch Dataset for the LSTM+RNN temporal branch — loads cached
per-video embedding sequences (produced by extract_embeddings.py)
instead of individual frames.

Videos have different numbers of frames (some 10, some 50), so
sequences need to be padded to a common length within each batch.
This module handles that via a custom collate function.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import CATEGORY_LABELS


class VideoSequenceDataset(Dataset):
    """
    Each sample is one video: a (seq_len, embedding_dim) tensor of
    frame embeddings, its true sequence length (before padding), and
    its label (0 = real, 1 = fake).
    """

    def __init__(self, samples: list):
        """
        Args:
            samples: list of (embedding_path: Path, label: int) tuples.
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        embedding_path, label = self.samples[idx]
        embeddings = np.load(embedding_path)  # shape: (num_frames, embedding_dim)
        sequence = torch.from_numpy(embeddings).float()
        seq_len = sequence.shape[0]
        return sequence, seq_len, label


def collate_sequences(batch):
    """
    Custom collate function: pads all sequences in a batch to the
    length of the longest sequence in that batch, and returns the
    true lengths so the model can ignore the padding (via
    pack_padded_sequence in the LSTM/RNN branches).
    """
    sequences, seq_lens, labels = zip(*batch)

    max_len = max(seq_lens)
    embedding_dim = sequences[0].shape[1]

    padded = torch.zeros(len(sequences), max_len, embedding_dim)
    for i, seq in enumerate(sequences):
        padded[i, :seq.shape[0], :] = seq

    seq_lens = torch.tensor(seq_lens, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded, seq_lens, labels


def build_sequence_splits(
    embeddings_root: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Build train/val/test splits over cached video embeddings.

    Uses the SAME video-level split logic as dataset.py's
    build_dataset_splits, for consistency: a video that landed in the
    CNN's training set should also land in the temporal model's
    training set, not leak across splits between the two stages.

    Args:
        embeddings_root: path to the folder of cached embeddings
                           (output of extract_embeddings.py).
        val_ratio, test_ratio, seed: same meaning as in dataset.py.

    Returns:
        (train_samples, val_samples, test_samples) — each a list of
        (embedding_path, label) tuples.
    """
    embeddings_root = Path(embeddings_root)
    if (embeddings_root / "real").exists() and (embeddings_root / "fake").exists():
        return build_sequence_splits_cdf(embeddings_root, val_ratio, test_ratio, seed)

    rng = random.Random(seed)

    train_samples, val_samples, test_samples = [], [], []

    for category, label in CATEGORY_LABELS.items():
        category_dir = embeddings_root / category
        if not category_dir.exists():
            print(f"Warning: no embeddings found for category '{category}' at {category_dir}")
            continue

        video_files = sorted(category_dir.glob("*.npy"))
        video_ids = [f.stem for f in video_files]

        if not video_ids:
            continue

        rng.shuffle(video_ids)

        n_total = len(video_ids)
        n_val = max(1, int(n_total * val_ratio))
        n_test = max(1, int(n_total * test_ratio))
        n_train = n_total - n_val - n_test

        train_ids = video_ids[:n_train]
        val_ids = video_ids[n_train:n_train + n_val]
        test_ids = video_ids[n_train + n_val:]

        for video_id in train_ids:
            train_samples.append((category_dir / f"{video_id}.npy", label))
        for video_id in val_ids:
            val_samples.append((category_dir / f"{video_id}.npy", label))
        for video_id in test_ids:
            test_samples.append((category_dir / f"{video_id}.npy", label))

        print(f"{category}: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test videos")

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    print(f"\nTotal videos -> train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}")

    return train_samples, val_samples, test_samples


def build_sequence_splits_cdf(
    embeddings_root,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Celeb-DF-v3 layout version (embeddings_root/real/... + fake/<group>/<method>/...).

    Mirrors dataset.build_dataset_splits_cdf step for step (same category
    ordering, same seed, same per-category shuffle) so every video lands in
    the SAME split as in CNN training - no train/test leakage between the
    CNN stage and the temporal stage. Videos with zero usable frames are
    dropped AFTER splitting (they still took part in the shuffle, exactly
    as they did for the CNN).
    """
    embeddings_root = Path(embeddings_root)
    rng = random.Random(seed)
    train_samples, val_samples, test_samples = [], [], []

    def leaf_dirs(root):
        return [d for d in root.rglob("*") if d.is_dir() and any(f.suffix == ".npy" for f in d.iterdir())]

    category_dirs = [(d, 0) for d in leaf_dirs(embeddings_root / "real")]
    category_dirs += [(d, 1) for d in leaf_dirs(embeddings_root / "fake")]
    if not category_dirs:
        raise ValueError(f"No embedding folders found under {embeddings_root}.")

    dropped = 0
    for category_dir, label in sorted(category_dirs, key=lambda c: str(c[0])):
        video_ids = sorted(f.stem for f in category_dir.glob("*.npy"))
        rng.shuffle(video_ids)

        n_total = len(video_ids)
        n_val = max(1, int(n_total * val_ratio))
        n_test = max(1, int(n_total * test_ratio))
        n_train = n_total - n_val - n_test

        for ids, bucket in ((video_ids[:n_train], train_samples),
                            (video_ids[n_train:n_train + n_val], val_samples),
                            (video_ids[n_train + n_val:], test_samples)):
            for vid in ids:
                path = category_dir / f"{vid}.npy"
                if np.load(path, mmap_mode="r").shape[0] == 0:
                    dropped += 1
                    continue
                bucket.append((path, label))

        print(f"{category_dir.relative_to(embeddings_root)}: {n_train} train, {n_val} val, "
              f"{n_total - n_train - n_val} test videos")

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    print(f"\nTotal videos -> train: {len(train_samples)}, val: {len(val_samples)}, "
          f"test: {len(test_samples)} (dropped {dropped} empty videos)")
    return train_samples, val_samples, test_samples


if __name__ == "__main__":
    import sys

    embeddings_root = sys.argv[1] if len(sys.argv) > 1 else "embeddings"
    train_samples, val_samples, test_samples = build_sequence_splits(embeddings_root)

    if train_samples:
        ds = VideoSequenceDataset(train_samples)
        seq, seq_len, label = ds[0]
        print(f"\nSample check -> sequence shape: {tuple(seq.shape)}, seq_len: {seq_len}, label: {label}")