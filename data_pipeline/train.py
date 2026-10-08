"""
train.py

Training loop for fine-tuning a pretrained CNN backbone (Xception/
EfficientNet/ResNet) for real/fake binary classification.

Usage:
    python train.py --faces_root faces --backbone xception --epochs 10

Resume from an interrupted run (e.g. a Kaggle/Colab quota cutoff or
session reset mid-training):
    python train.py --faces_root faces --backbone xception --epochs 10 \\
        --resume_from checkpoints/xception_best.pth --initial_best_accuracy 0.87

Works on both the FaceForensics++ layout and the Celeb-DF-v3 layout
(faces_root/real/... + faces_root/fake/...) — the right split function
is picked automatically based on faces_root's structure.
"""

import argparse
import random
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    FaceDataset,
    build_dataset_splits,
    build_dataset_splits_cdf,
    get_eval_transform,
    get_train_transform,
    load_extra_training_frames,
)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(backbone_name: str, device: torch.device, drop_rate: float = 0.3) -> nn.Module:
    """
    Load a pretrained backbone via timm and swap the classifier head
    for 2-class (real/fake) output.
    """
    model = timm.create_model(backbone_name, pretrained=True, num_classes=2, drop_rate=drop_rate)
    return model.to(device)


def run_one_epoch(model, dataloader, criterion, optimizer, device, is_training: bool,
                   phase_name: str = "train") -> dict:
    model.train() if is_training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    batch_times = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    progress_bar = tqdm(dataloader, desc=f"  {phase_name}", unit="batch", leave=False)

    with context:
        for images, labels in progress_bar:
            batch_start = time.time()
            images, labels = images.to(device), labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            batch_times.append(time.time() - batch_start)
            running_loss = total_loss / total
            running_acc = correct / total
            avg_batch_time = sum(batch_times) / len(batch_times)

            progress_bar.set_postfix({
                "loss": f"{running_loss:.4f}",
                "acc": f"{running_acc:.4f}",
                "s/batch": f"{avg_batch_time:.2f}",
            })

    return {
        "loss": total_loss / total if total > 0 else 0.0,
        "accuracy": correct / total if total > 0 else 0.0,
    }


def train_model(
    faces_root: str,
    backbone_name: str = "xception",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    drop_rate: float = 0.3,
    early_stopping_patience: int = 3,
    checkpoint_dir: str = "checkpoints",
    resume_from: str = None,
    initial_best_accuracy: float = 0.0,
    extra_real_dir: str = None,
    extra_fake_dir: str = None,
):
    """
    Full training run: builds splits, dataloaders, model, and runs the
    train/val loop, saving the best checkpoint (by validation accuracy)
    along the way.

    resume_from: optional path to an existing checkpoint (.pth) to load
    weights from before training starts. Added after repeatedly losing
    checkpoints to Kaggle/Colab session resets mid-project — lets a new
    session continue fine-tuning from a previously trained state instead
    of starting from the pretrained backbone's original weights.

    initial_best_accuracy: when resuming, pass the val accuracy the
    resumed checkpoint already achieved, so early stopping and "new
    best" logic compare against the real prior best rather than 0.0
    (which would make the very first epoch after resume look like an
    improvement even if it's actually worse than what you resumed from).
    Only meaningful when resuming onto the SAME val set the number was
    measured on — a number from a different val set (e.g. FF++'s) isn't
    comparable and should be left at the default 0.0 instead.

    Note: this resumes MODEL WEIGHTS only, not optimizer state (momentum,
    etc.) — the optimizer restarts fresh. For fine-tuning a few more
    epochs this is a reasonable simplification; it means the first
    resumed epoch may behave slightly differently than a true
    uninterrupted continuation, but avoids the extra complexity of
    saving/loading full training state for what is normally a short
    fine-tuning run.
    """
    device = get_device()
    print(f"Using device: {device}")

    # --- Data ---
    # Auto-detect which dataset layout faces_root uses. Celeb-DF-v3's
    # layout nests fake methods under group folders and sits real/fake
    # content under top-level real/ and fake/ folders; FaceForensics++'s
    # layout has its categories directly under faces_root with no such
    # split, so the two are distinguishable just by checking for those
    # two folders.
    faces_root_path = Path(faces_root)
    if (faces_root_path / "real").exists() and (faces_root_path / "fake").exists():
        print(f"Detected Celeb-DF-v3-style layout (real/ + fake/) under {faces_root}")
        train_samples, val_samples, test_samples = build_dataset_splits_cdf(faces_root)
    else:
        train_samples, val_samples, test_samples = build_dataset_splits(faces_root)

    # Diversify TRAINING data only — val/test stay exactly as they were,
    # so results remain comparable to every prior run on this same
    # test set, while the model gets exposure to a second dataset's
    # manipulation style during training.
    if extra_real_dir:
        train_samples.extend(load_extra_training_frames(extra_real_dir, label=0))
    if extra_fake_dir:
        train_samples.extend(load_extra_training_frames(extra_fake_dir, label=1))
    if extra_real_dir or extra_fake_dir:
        random.shuffle(train_samples)
        print(f"Training set after adding extra data: {len(train_samples)} frames")

    train_dataset = FaceDataset(train_samples, transform=get_train_transform())
    val_dataset = FaceDataset(val_samples, transform=get_eval_transform())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train batches per epoch: {len(train_loader)} | Val batches per epoch: {len(val_loader)}")

    # --- Model, loss, optimizer ---
    model = build_model(backbone_name, device, drop_rate=drop_rate)

    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            model.load_state_dict(torch.load(resume_path, map_location=device))
            print(f"Resumed weights from {resume_path}")
        else:
            print(f"Warning: --resume_from path not found ({resume_path}), starting from pretrained weights instead.")

    # --- Class weighting ---
    # Celeb-DF-v3's training frames are heavily imbalanced (~1:6 real:fake
    # at the frame level, observed on the 21-category aligned-face set).
    # Unweighted CrossEntropyLoss would bias the model toward predicting
    # "fake". Weight is inverse-frequency, computed from the actual
    # training split (after any --extra_*_dir additions), never val/test.
    label_counts = {0: 0, 1: 0}
    for _, label in train_samples:
        label_counts[label] += 1
    total_train = label_counts[0] + label_counts[1]
    if label_counts[0] > 0 and label_counts[1] > 0:
        class_weights = torch.tensor(
            [total_train / (2 * label_counts[0]), total_train / (2 * label_counts[1])],
            dtype=torch.float32,
            device=device,
        )
        print(f"Train class counts -> real: {label_counts[0]}, fake: {label_counts[1]} "
              f"| loss weights -> real: {class_weights[0]:.3f}, fake: {class_weights[1]:.3f}")
    else:
        class_weights = None
        print("Warning: one class is missing from the training split entirely; skipping class weighting.")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = initial_best_accuracy
    epochs_without_improvement = 0
    if initial_best_accuracy > 0:
        print(f"Starting from initial_best_accuracy={initial_best_accuracy:.4f} "
              f"(only strictly better epochs will be saved as a new best).")

    # --- Quick single-batch timing check ---
    print("Timing a single batch load (sanity check before full training)...")
    check_start = time.time()
    first_batch = next(iter(train_loader))
    load_time = time.time() - check_start
    print(f"First batch loaded in {load_time:.2f}s "
          f"({'looks fine' if load_time < 5 else 'this is slow — data loading may be the bottleneck, not the GPU'})")

    # --- Epoch loop ---
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        print(f"Epoch {epoch}/{epochs} starting...")
        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device,
                                       is_training=True, phase_name="train")
        val_metrics = run_one_epoch(model, val_loader, criterion, optimizer, device,
                                     is_training=False, phase_name="val")

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | "
            f"train_loss: {train_metrics['loss']:.4f} train_acc: {train_metrics['accuracy']:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} val_acc: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            epochs_without_improvement = 0
            checkpoint_path = checkpoint_dir / f"{backbone_name}_best.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  New best val accuracy ({best_val_accuracy:.4f}) — saved to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s) (best so far: {best_val_accuracy:.4f})")

            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping: val accuracy hasn't improved for "
                      f"{early_stopping_patience} consecutive epochs. Stopping at epoch {epoch}/{epochs}.")
                break

    print(f"\nTraining done. Best val accuracy: {best_val_accuracy:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained CNN backbone for deepfake detection.")
    parser.add_argument("--faces_root", required=True, help="Path to the 'faces' folder (output of mtcnn_align.py).")
    parser.add_argument("--backbone", default="xception", help="timm model name (default: xception).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--drop_rate", type=float, default=0.3)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--resume_from", default=None,
                         help="Path to an existing checkpoint to resume training from, instead of "
                              "starting from the pretrained backbone's original weights.")
    parser.add_argument("--initial_best_accuracy", type=float, default=0.0,
                         help="Val accuracy the resumed checkpoint already achieved, so early "
                              "stopping/best-checkpoint logic compares correctly.")
    parser.add_argument("--extra_real_dir", default=None,
                         help="Optional: path to an extra folder of aligned real-face videos "
                              "(e.g. a non-overlapping Celeb-DF slice) to ADD to training only — "
                              "val/test are unaffected, for diversifying training data.")
    parser.add_argument("--extra_fake_dir", default=None,
                         help="Optional: same as --extra_real_dir but for extra fake-face videos.")
    args = parser.parse_args()

    train_model(
        faces_root=args.faces_root,
        backbone_name=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        drop_rate=args.drop_rate,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        initial_best_accuracy=args.initial_best_accuracy,
        extra_real_dir=args.extra_real_dir,
        extra_fake_dir=args.extra_fake_dir,
    )


if __name__ == "__main__":
    main()