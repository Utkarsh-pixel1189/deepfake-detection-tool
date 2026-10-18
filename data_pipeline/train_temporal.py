"""
train_temporal.py

Training loop for the LSTM+RNN temporal fusion model, operating on
cached CNN embeddings (from extract_embeddings.py).

Usage:
    python train_temporal.py --embeddings_root embeddings --epochs 15

Resume from an interrupted run:
    python train_temporal.py --embeddings_root embeddings --epochs 15 \\
        --resume_from checkpoints/temporal_fusion_best.pth --initial_best_accuracy 0.89
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sequence_dataset import (
    VideoSequenceDataset,
    build_sequence_splits,
    build_sequence_splits_multi,
    collate_sequences,
)
from temporal_model import TemporalFusionModel, TemporalFusionAttentionModel


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_embedding_dim(sample_path: Path) -> int:
    sample = np.load(sample_path)
    return sample.shape[1]


def run_one_epoch(model, dataloader, criterion, optimizer, device, is_training: bool, phase_name: str) -> dict:
    model.train() if is_training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0, 0]
    class_total = [0, 0]

    context = torch.enable_grad() if is_training else torch.no_grad()
    progress_bar = tqdm(dataloader, desc=f"  {phase_name}", unit="batch", leave=False)

    with context:
        for sequences, seq_lens, labels in progress_bar:
            sequences, labels = sequences.to(device), labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(sequences, seq_lens)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            for c in (0, 1):
                mask = labels == c
                class_total[c] += mask.sum().item()
                class_correct[c] += ((predictions == labels) & mask).sum().item()

            progress_bar.set_postfix({
                "loss": f"{total_loss / total:.4f}",
                "acc": f"{correct / total:.4f}",
            })

    real_acc = class_correct[0] / class_total[0] if class_total[0] else 0.0
    fake_acc = class_correct[1] / class_total[1] if class_total[1] else 0.0
    return {
        "loss": total_loss / total if total > 0 else 0.0,
        "accuracy": correct / total if total > 0 else 0.0,
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "balanced_accuracy": (real_acc + fake_acc) / 2,
    }


def train_temporal_model(
    embeddings_root: str,
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    dropout: float = 0.4,
    early_stopping_patience: int = 5,
    lstm_hidden_size: int = 256,
    lstm_num_layers: int = 2,
    rnn_hidden_size: int = 256,
    rnn_num_layers: int = 1,
    checkpoint_dir: str = "checkpoints",
    checkpoint_name: str = "temporal_fusion_best.pth",
    resume_from: str = None,
    initial_best_accuracy: float = 0.0,
    use_attention: bool = False,
    extra_embeddings_roots: list = None,
    select_by: str = "balanced_accuracy",
):
    """
    Full training run for the LSTM+RNN fusion model.

    resume_from / initial_best_accuracy: same resume pattern as
    train.py — loads existing weights rather than starting from
    scratch, and seeds best_val_accuracy so early stopping/checkpoint
    logic compares against the real prior best rather than 0.0.
    Model architecture flags (lstm_hidden_size etc.) must match
    whatever the checkpoint being resumed was originally trained with.
    """
    device = get_device()
    print(f"Using device: {device}")

    if extra_embeddings_roots:
        # Joint multi-source training: embeddings_root plus every extra root.
        train_samples, val_samples, test_samples = build_sequence_splits_multi(
            [embeddings_root] + list(extra_embeddings_roots))
    else:
        train_samples, val_samples, test_samples = build_sequence_splits(embeddings_root)

    if not train_samples:
        raise ValueError(
            "No training samples found. Did extract_embeddings.py run successfully "
            "and produce .npy files under the given embeddings_root?"
        )

    embedding_dim = infer_embedding_dim(train_samples[0][0])
    print(f"Detected embedding dimension: {embedding_dim}")
    print(f"Architecture: LSTM(hidden={lstm_hidden_size}, layers={lstm_num_layers}) "
          f"+ RNN(hidden={rnn_hidden_size}, layers={rnn_num_layers}) "
          f"[{'attention pooling' if use_attention else 'final hidden state'}]")

    train_dataset = VideoSequenceDataset(train_samples)
    val_dataset = VideoSequenceDataset(val_samples)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_sequences, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_sequences, num_workers=2
    )

    print(f"Train batches per epoch: {len(train_loader)} | Val batches per epoch: {len(val_loader)}")

    ModelClass = TemporalFusionAttentionModel if use_attention else TemporalFusionModel
    model = ModelClass(
        embedding_dim=embedding_dim,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        rnn_hidden_size=rnn_hidden_size,
        rnn_num_layers=rnn_num_layers,
        dropout=dropout,
    ).to(device)

    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            model.load_state_dict(torch.load(resume_path, map_location=device))
            print(f"Resumed weights from {resume_path}")
        else:
            print(f"Warning: --resume_from path not found ({resume_path}), starting fresh instead.")

    # Inverse-frequency class weights from the TRAIN split only. Celeb-DF-v3 is
    # heavily fake-skewed at the video level, so unweighted loss would push the
    # model to just say "fake".
    n_real = sum(1 for _, l in train_samples if l == 0)
    n_fake = len(train_samples) - n_real
    if n_real > 0 and n_fake > 0:
        total = n_real + n_fake
        class_weights = torch.tensor([total / (2 * n_real), total / (2 * n_fake)],
                                     dtype=torch.float32, device=device)
        print(f"Train videos -> real: {n_real}, fake: {n_fake} | loss weights -> "
              f"real: {class_weights[0]:.3f}, fake: {class_weights[1]:.3f}")
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = initial_best_accuracy
    epochs_without_improvement = 0
    if initial_best_accuracy > 0:
        print(f"Starting from initial_best_accuracy={initial_best_accuracy:.4f}")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        print(f"Epoch {epoch}/{epochs} starting...")
        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device, True, "train")
        val_metrics = run_one_epoch(model, val_loader, criterion, optimizer, device, False, "val")

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | "
            f"train_loss: {train_metrics['loss']:.4f} train_acc: {train_metrics['accuracy']:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} val_acc: {val_metrics['accuracy']:.4f} | "
            f"val real_acc: {val_metrics['real_acc']:.4f} fake_acc: {val_metrics['fake_acc']:.4f} "
            f"balanced: {val_metrics['balanced_accuracy']:.4f}"
        )

        if val_metrics[select_by] > best_val_accuracy:
            best_val_accuracy = val_metrics[select_by]
            epochs_without_improvement = 0
            checkpoint_path = checkpoint_dir / checkpoint_name
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  New best val accuracy ({best_val_accuracy:.4f}) — saved to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s) (best so far: {best_val_accuracy:.4f})")

            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}/{epochs}.")
                break

    print(f"\nTraining done. Best val {select_by}: {best_val_accuracy:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the LSTM+RNN temporal fusion model.")
    parser.add_argument("--embeddings_root", required=True, help="Path to cached embeddings (output of extract_embeddings.py).")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--lstm_hidden_size", type=int, default=256)
    parser.add_argument("--lstm_num_layers", type=int, default=2)
    parser.add_argument("--rnn_hidden_size", type=int, default=256,
                         help="Default changed from 128 to 256 after tuning — this was the single "
                              "most effective architecture change tested.")
    parser.add_argument("--rnn_num_layers", type=int, default=1)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--checkpoint_name", default="temporal_fusion_best.pth",
                         help="Filename to save this run's best checkpoint as — give each "
                              "architecture variation a distinct name so you can compare "
                              "multiple saved checkpoints rather than overwriting one.")
    parser.add_argument("--resume_from", default=None,
                         help="Path to an existing checkpoint to resume training from.")
    parser.add_argument("--initial_best_accuracy", type=float, default=0.0,
                         help="Val accuracy the resumed checkpoint already achieved.")
    parser.add_argument("--use_attention", action="store_true",
                         help="Use attention pooling over all timesteps instead of just the "
                              "final hidden state (TemporalFusionAttentionModel instead of "
                              "TemporalFusionModel). This is an enhancement beyond the project "
                              "doc's original architecture spec.")
    parser.add_argument("--extra_embeddings_root", action="append", default=[],
                         help="Joint training: another embeddings root (any supported layout) merged "
                              "into training. Repeatable. Append ::K to repeat its train videos K times.")
    parser.add_argument("--select_by", default="balanced_accuracy", choices=["accuracy", "balanced_accuracy"],
                         help="Metric used to pick the best checkpoint (default balanced_accuracy).")
    args = parser.parse_args()

    train_temporal_model(
        embeddings_root=args.embeddings_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        early_stopping_patience=args.early_stopping_patience,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_num_layers=args.rnn_num_layers,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        resume_from=args.resume_from,
        initial_best_accuracy=args.initial_best_accuracy,
        use_attention=args.use_attention,
        extra_embeddings_roots=args.extra_embeddings_root,
        select_by=args.select_by,
    )


if __name__ == "__main__":
    main()