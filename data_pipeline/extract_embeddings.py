"""
extract_embeddings.py

Extracts frame-level feature embeddings from the trained Xception
checkpoint (the winning backbone from the CNN comparison) and caches
them to disk, grouped by video.

Why this exists: the LSTM + RNN temporal branches need SEQUENCES of
frame features per video, not independent frame predictions. Re-running
Xception on every frame, every epoch, during LSTM/RNN training would be
slow and wasteful — Xception's weights are frozen at this stage anyway
(we already selected it as the winning backbone), so its output for a
given frame never changes between epochs. Caching embeddings once,
here, means the temporal training loop only has to do cheap LSTM/RNN
computation on small cached vectors, not repeat the expensive CNN
forward pass every time.

Output format: one .npy file per video, containing a (num_frames, embedding_dim)
array, saved into a folder structure mirroring the original faces/ folder
(one subfolder per category, one file per video).

Usage:
    python extract_embeddings.py --faces_root faces --checkpoint checkpoints/xception_best.pth --output_dir embeddings
"""

import argparse
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from tqdm import tqdm

from dataset import CATEGORY_LABELS, IMAGE_EXTENSIONS, _find_leaf_category_dirs, get_eval_transform


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_feature_extractor(checkpoint_path: str, backbone_name: str, device: torch.device):
    """
    Load the trained backbone and strip off its classification head,
    so it outputs raw pooled features instead of a real/fake prediction.

    num_classes=0 tells timm to skip the final classifier layer
    entirely and return the pooled feature vector instead.
    """
    model = timm.create_model(backbone_name, pretrained=False, num_classes=0)

    state_dict = torch.load(checkpoint_path, map_location=device)
    # The saved checkpoint includes the classifier layer's weights (since it
    # was trained with num_classes=2); loading with strict=False ignores
    # that mismatched final layer and keeps everything else, since we only
    # want the feature-extraction layers, not the classifier we're removing.
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model


def extract_embeddings_for_video(video_dir: Path, model, transform, device: torch.device,
                                 batch_size: int = 32) -> np.ndarray:
    """
    Run every frame in one video's folder through the feature extractor,
    in filename order (so the sequence preserves temporal order), and
    stack the results into a single (num_frames, embedding_dim) array.
    """
    frame_paths = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not frame_paths:
        return np.empty((0,))

    # Batched forward pass (chunks of `batch_size` frames) - same numbers as
    # one-frame-at-a-time because the model is in eval mode (BatchNorm uses
    # running stats, dropout off), but far faster on GPU.
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(frame_paths), batch_size):
            chunk = frame_paths[start:start + batch_size]
            batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in chunk]).to(device)
            embeddings.append(model(batch).cpu().numpy())  # (chunk, embedding_dim)

    return np.concatenate(embeddings, axis=0)  # shape: (num_frames, embedding_dim)


def main():
    parser = argparse.ArgumentParser(description="Extract and cache CNN embeddings for LSTM/RNN training.")
    parser.add_argument("--faces_root", required=True, help="Path to the 'faces' folder (output of mtcnn_align.py).")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained backbone checkpoint (.pth).")
    parser.add_argument("--backbone", default="xception", help="Must match the architecture the checkpoint was trained with.")
    parser.add_argument("--output_dir", required=True, help="Directory to save cached embeddings into.")
    args = parser.parse_args()

    faces_root = Path(args.faces_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    model = load_feature_extractor(args.checkpoint, args.backbone, device)
    transform = get_eval_transform()  # no augmentation — we want stable, consistent embeddings

    # Build the list of (category_dir, relative_output_subpath) to process.
    # FF++ layout: categories sit directly under faces_root (CATEGORY_LABELS;
    # faceshifter_holdout is intentionally excluded). Celeb-DF-v3 layout
    # (real/ + fake/): categories are auto-discovered and the real/ + fake/
    # folder structure is mirrored in the output, so sequence_dataset.py can
    # reproduce the exact same video-level split the CNN was trained with.
    if (faces_root / "real").exists() and (faces_root / "fake").exists():
        print("Detected Celeb-DF-v3-style layout (real/ + fake/)")
        category_dirs = _find_leaf_category_dirs(faces_root / "real") + _find_leaf_category_dirs(faces_root / "fake")
        jobs = [(d, d.relative_to(faces_root)) for d in sorted(category_dirs, key=lambda d: str(d))]
    else:
        jobs = [(faces_root / c, Path(c)) for c in CATEGORY_LABELS]

    for category_dir, rel in jobs:
        if not category_dir.exists():
            print(f"Warning: category folder not found, skipping: {category_dir}")
            continue

        category_output_dir = output_dir / rel
        category_output_dir.mkdir(parents=True, exist_ok=True)

        video_dirs = sorted(p for p in category_dir.iterdir() if p.is_dir())
        print(f"\n{rel}: extracting embeddings for {len(video_dirs)} videos...")

        for video_dir in tqdm(video_dirs, desc=f"  {rel}", unit="video"):
            output_path = category_output_dir / f"{video_dir.name}.npy"
            if output_path.exists():
                continue  # already done — allows resuming an interrupted run

            embeddings = extract_embeddings_for_video(video_dir, model, transform, device)
            if embeddings.size == 0:
                # Video had every frame discarded during alignment. Still write a
                # placeholder so the folder listing (and therefore the seeded
                # train/val/test shuffle) matches the CNN's split exactly;
                # sequence_dataset.py drops empty videos after splitting.
                embeddings = np.zeros((0, model.num_features), dtype=np.float32)
            np.save(output_path, embeddings)

    print("\nDone. Embeddings cached to:", output_dir)


if __name__ == "__main__":
    main()