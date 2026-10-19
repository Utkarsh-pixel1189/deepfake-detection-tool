"""
train_image.py

Trains an image-level Xception real/fake classifier on aligned face crops
(output of prepare_image_faces.py), picks the best epoch on a held-out part
of the TRAINING data, and then tests ONCE on a separate external dataset.

Why the external test is kept separate: a model tested on images from the
same source it was trained on can look far better than it really is. The
external set (140k Real and Fake Faces, test split) is only touched after
training is finished.

Labels: real = 0, fake = 1 (same convention as the video model).

Usage (Kaggle GPU, from data_pipeline/):
    python train_image.py \\
        --train_real /kaggle/working/hf_faces_aligned/real \\
        --train_fake /kaggle/working/hf_faces_aligned/fake \\
        --test_real  /kaggle/working/test_faces_aligned/real \\
        --test_fake  /kaggle/working/test_faces_aligned/fake \\
        --out_dir /kaggle/working/image_model
"""

import argparse
import io
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
REAL, FAKE = 0, 1


def list_images(folder: str, label: int) -> list:
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return [(str(p), label) for p in paths]


class ImageDataset(Dataset):
    def __init__(self, items: list, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class RandomJPEG:
    """Re-encode at a random JPEG quality so compression level is not a class cue."""

    def __init__(self, qmin: int = 40, qmax: int = 95, p: float = 0.5):
        self.qmin, self.qmax, self.p = qmin, qmax, p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(self.qmin, self.qmax))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


# Softness and file size differed between classes in the HF training set
# (fakes softer and smaller), and the model learned that as a fake cue.
# Blur and recompression are applied to BOTH classes so neither one can be
# recognised by how soft or compressed it looks.
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
    RandomJPEG(qmin=40, qmax=95, p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def stratified_split(items: list, val_frac: float, seed: int):
    """Split so both classes keep the same proportion in train and val."""
    rng = random.Random(seed)
    train, val = [], []
    for label in (REAL, FAKE):
        group = [it for it in items if it[1] == label]
        rng.shuffle(group)
        n_val = int(round(len(group) * val_frac))
        val += group[:n_val]
        train += group[n_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def metrics(labels, probs, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)
    real_acc = float(((preds == 0) & (labels == 0)).sum() / (labels == 0).sum())
    fake_acc = float(((preds == 1) & (labels == 1)).sum() / (labels == 1).sum())
    return {
        "n": int(len(labels)),
        "accuracy": float((preds == labels).mean()),
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "balanced_accuracy": (real_acc + fake_acc) / 2,
        "auc": float(roc_auc_score(labels, probs)),
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        probs = torch.softmax(logits.float(), dim=1)[:, FAKE]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_real", required=True)
    ap.add_argument("--train_fake", required=True)
    ap.add_argument("--test_real", required=True)
    ap.add_argument("--test_fake", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--backbone", default="xception")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    all_train = list_images(args.train_real, REAL) + list_images(args.train_fake, FAKE)
    test_items = list_images(args.test_real, REAL) + list_images(args.test_fake, FAKE)
    train_items, val_items = stratified_split(all_train, args.val_frac, args.seed)
    print(f"training source: {len(all_train)} images -> train {len(train_items)}, val {len(val_items)}")
    print(f"external test:   {len(test_items)} images (not used until the end)")
    print(f"train classes: {dict(Counter(l for _, l in train_items))}")

    train_loader = DataLoader(ImageDataset(train_items, train_tf), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ImageDataset(val_items, eval_tf), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(ImageDataset(test_items, eval_tf), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = timm.create_model(args.backbone, pretrained=True, num_classes=2, drop_rate=0.3).to(device)

    # Class weights so the model can't win by predicting the bigger class.
    counts = Counter(l for _, l in train_items)
    total = sum(counts.values())
    weights = torch.tensor([total / (2 * counts[c]) for c in (REAL, FAKE)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_bal, best_epoch, history = -1.0, -1, []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        scheduler.step()
        v_labels, v_probs = predict(model, val_loader, device)
        vm = metrics(v_labels, v_probs)
        history.append({"epoch": epoch, "train_loss": loss, "val": vm})
        print(f"epoch {epoch}: train_loss {loss:.4f} | val acc {vm['accuracy']:.4f} "
              f"real {vm['real_acc']:.4f} fake {vm['fake_acc']:.4f} "
              f"balanced {vm['balanced_accuracy']:.4f} auc {vm['auc']:.4f}")
        if vm["balanced_accuracy"] > best_bal:
            best_bal, best_epoch = vm["balanced_accuracy"], epoch
            torch.save({"state_dict": model.state_dict(), "backbone": args.backbone,
                        "epoch": epoch, "val": vm}, out / "image_best.pth")
            print("  saved new best checkpoint")

    # Final test: load the best checkpoint and score the external set ONCE.
    ckpt = torch.load(out / "image_best.pth", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    t_labels, t_probs = predict(model, test_loader, device)
    tm = metrics(t_labels, t_probs)

    print("\n=== EXTERNAL TEST (140k test split, never used in training) ===")
    print(f"best epoch {best_epoch} (val balanced {best_bal:.4f})")
    print(f"accuracy {tm['accuracy']:.4f}  real {tm['real_acc']:.4f}  "
          f"fake {tm['fake_acc']:.4f}  balanced {tm['balanced_accuracy']:.4f}  auc {tm['auc']:.4f}")

    results = {"config": vars(args), "best_epoch": best_epoch, "history": history,
               "external_test": tm}
    (out / "image_results.json").write_text(json.dumps(results, indent=2))
    print(f"saved {out / 'image_best.pth'} and {out / 'image_results.json'}")


if __name__ == "__main__":
    main()