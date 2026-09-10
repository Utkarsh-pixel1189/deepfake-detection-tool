"""
dataset.py

PyTorch Dataset for the deepfake detection CNN, plus train/val/test
split logic over the aligned face crops produced by mtcnn_align.py.

Expected folder layout (one folder per category, as produced by the
pipeline so far):

    faces/
        original/<video_id>/frame_00000.jpg, ...
        deepfakes/<video_id>/frame_00000.jpg, ...
        face2face/<video_id>/...
        faceswap/<video_id>/...
        neuraltextures/<video_id>/...
        faceshifter_holdout/<video_id>/...   <- NOT included here on purpose,
                                                  this is the Week 7 generalization
                                                  holdout and must never appear
                                                  in train/val/test splits.

Labels: "original" -> 0 (real), every manipulated category -> 1 (fake).

Splitting is done at the VIDEO level, not the frame level. Frames from
the same video are highly correlated (near-duplicate faces a fraction
of a second apart), so splitting at the frame level would leak near-
identical images across train/val/test and inflate validation/test
accuracy artificially. Splitting by video_id avoids that.
"""

import io
import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Category -> label. 0 = real, 1 = fake.
# faceshifter_holdout is intentionally excluded — it must never be split
# into train/val/test, it is reserved entirely for the Week 7
# generalization test on an unseen manipulation technique.
CATEGORY_LABELS = {
    "original": 0,
    "deepfakes": 1,
    "face2face": 1,
    "faceswap": 1,
    "neuraltextures": 1,
}

# Standard ImageNet normalization stats, since all three backbones
# (Xception/EfficientNet/ResNet) were pretrained on ImageNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomJPEGCompression:
    """
    Re-compresses the image as JPEG at a randomly chosen quality level,
    simulating what happens when a video is re-shared/re-uploaded
    multiple times on social media — each re-share typically applies
    another lossy compression pass, which can wash out the subtle
    pixel-level artifacts deepfake detectors rely on. Training on
    randomly degraded compression levels (not just the dataset's
    original c23 quality) is intended to make the model rely on more
    compression-robust cues, closer to what real-world sharing produces.
    """

    def __init__(self, quality_range=(30, 90)):
        self.quality_range = quality_range

    def __call__(self, image: Image.Image) -> Image.Image:
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomDownscaleUpscale:
    """
    Randomly shrinks the image then resizes it back up, simulating
    low-resolution footage or aggressive social-media re-encoding
    (many platforms downscale video before serving it).
    """

    def __init__(self, scale_range=(0.4, 1.0)):
        self.scale_range = scale_range

    def __call__(self, image: Image.Image) -> Image.Image:
        original_size = image.size
        scale = random.uniform(*self.scale_range)
        if scale >= 0.999:
            return image
        small_size = (max(1, int(original_size[0] * scale)), max(1, int(original_size[1] * scale)))
        image = image.resize(small_size, Image.BILINEAR)
        return image.resize(original_size, Image.BILINEAR)


def get_train_transform():
    """Light augmentation for training: helps the model generalize
    rather than memorize specific frames.

    NOTE: a stronger geometric/color augmentation variant (RandomResizedCrop,
    wider rotation, stronger color jitter, RandomErasing) was tested as
    polishing Step 3 and found to REDUCE downstream temporal model
    accuracy. Reverted back to this lighter version.

    ADDED (real-world robustness experiment): RandomJPEGCompression and
    RandomDownscaleUpscale, applied with 40% probability each. Unlike
    the earlier geometric augmentation experiment, this targets a
    specific, measured weakness — the model's cross-dataset accuracy
    on Celeb-DF (68.00%) and generalization to an unseen manipulation
    technique (37.39% on faceshifter_holdout) both suggest sensitivity
    to exact compression characteristics. This is a different
    hypothesis than the geometric augmentation that didn't help, so
    it's being tested independently rather than assumed to fail too.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomApply([RandomJPEGCompression()], p=0.4),
        transforms.RandomApply([RandomDownscaleUpscale()], p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform():
    """No augmentation for validation/test — we want a stable,
    reproducible evaluation, not randomized inputs."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class FaceDataset(Dataset):
    """
    A simple (image_path, label) dataset over aligned face crops.
    """

    def __init__(self, samples: list, transform=None):
        """
        Args:
            samples: list of (image_path: Path, label: int) tuples.
            transform: torchvision transform to apply to each image.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def _collect_video_ids(category_dir: Path) -> list:
    """Return every video-id subfolder inside one category's face folder."""
    if not category_dir.exists():
        return []
    return sorted(p.name for p in category_dir.iterdir() if p.is_dir())


def _frames_for_video(category_dir: Path, video_id: str, label: int) -> list:
    """Return all (frame_path, label) pairs for one video."""
    video_dir = category_dir / video_id
    return [
        (p, label) for p in sorted(video_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def load_extra_training_frames(extra_dir: str, label: int) -> list:
    """
    Loads every frame from every video folder under extra_dir as extra
    TRAINING-ONLY data, labeled with the given class.

    This is the mechanism for diversifying training data beyond
    FaceForensics++ (e.g. adding a separate, non-overlapping slice of
    Celeb-DF) WITHOUT touching the existing val/test split — val/test
    stay exactly as they were, so results remain comparable to every
    prior run, while train gets more diverse examples to learn from.

    Args:
        extra_dir: path to a folder of aligned face video subfolders,
                    all belonging to ONE class (all real, or all fake) —
                    same structure as mtcnn_align.py's output for a
                    single category.
        label: 0 for real, 1 for fake — applied to every frame found.

    Returns:
        List of (frame_path, label) tuples.
    """
    extra_path = Path(extra_dir)
    if not extra_path.exists():
        print(f"Warning: extra training dir not found: {extra_path}")
        return []

    video_ids = _collect_video_ids(extra_path)
    samples = []
    for video_id in video_ids:
        samples.extend(_frames_for_video(extra_path, video_id, label))

    print(f"Loaded {len(samples)} extra training frames from {extra_dir} (label={label})")
    return samples


def build_dataset_splits(
    faces_root: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Build train/val/test sample lists by splitting VIDEOS (not frames)
    within each category, then collecting all frames for the videos
    assigned to each split.

    Args:
        faces_root: path to the "faces" folder containing the category
                     subfolders (output of mtcnn_align.py). Should NOT
                     point at faceshifter_holdout — that folder is
                     handled separately, see module docstring.
        val_ratio: fraction of videos (per category) assigned to validation.
        test_ratio: fraction of videos (per category) assigned to test.
        seed: random seed, for reproducible splits across runs.

    Returns:
        (train_samples, val_samples, test_samples) — each a list of
        (image_path, label) tuples.
    """
    faces_root = Path(faces_root)
    rng = random.Random(seed)

    train_samples, val_samples, test_samples = [], [], []

    for category, label in CATEGORY_LABELS.items():
        category_dir = faces_root / category
        video_ids = _collect_video_ids(category_dir)

        if not video_ids:
            print(f"Warning: no videos found for category '{category}' at {category_dir}")
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
            train_samples.extend(_frames_for_video(category_dir, video_id, label))
        for video_id in val_ids:
            val_samples.extend(_frames_for_video(category_dir, video_id, label))
        for video_id in test_ids:
            test_samples.extend(_frames_for_video(category_dir, video_id, label))

        print(f"{category}: {n_train} train videos, {len(val_ids)} val videos, "
              f"{len(test_ids)} test videos")

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    print(f"\nTotal frames -> train: {len(train_samples)}, "
          f"val: {len(val_samples)}, test: {len(test_samples)}")

    return train_samples, val_samples, test_samples


if __name__ == "__main__":
    # Quick sanity check: run this directly to confirm splits build
    # correctly before wiring it into the training loop.
    import sys

    faces_root = sys.argv[1] if len(sys.argv) > 1 else "faces"
    train_samples, val_samples, test_samples = build_dataset_splits(faces_root)

    if train_samples:
        train_ds = FaceDataset(train_samples, transform=get_train_transform())
        image, label = train_ds[0]
        print(f"\nSample check -> image shape: {tuple(image.shape)}, label: {label}")
