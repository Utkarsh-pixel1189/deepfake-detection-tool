"""
fetch_hf_subset.py

Stream a real/fake image dataset from the Hugging Face Hub and save only a
balanced SAMPLE of it to disk (no need to download the whole 20+ GB):

    out_dir/real/000001.png ...
    out_dir/fake/000001.png ...

All images are saved as lossless PNG for BOTH classes so the saved format
cannot become a real-vs-fake shortcut.

Usage (Kaggle notebook, internet ON):
    python fetch_hf_subset.py --repo OWNER/DATASET_NAME --split train \
        --per_class 6000 --out_dir /kaggle/working/hf_faces

If the label column / image column are not auto-detected, pass
--image_col and --label_col. If label names are not recognisable
('real'/'fake'), pass --real_value and --fake_value (e.g. 0 and 1).
"""

import argparse
from pathlib import Path


def resolve_label_values(feature, real_value, fake_value):
    """Return (real_value, fake_value) as they appear in the data."""
    if real_value is not None and fake_value is not None:
        return real_value, fake_value
    names = getattr(feature, "names", None)
    if names:
        low = [str(n).lower() for n in names]
        real_idx = next((i for i, n in enumerate(low) if "real" in n or "genuine" in n or "authentic" in n), None)
        fake_idx = next((i for i, n in enumerate(low) if "fake" in n or "deepfake" in n or "synthetic" in n or "ai" == n), None)
        if real_idx is not None and fake_idx is not None:
            print(f"Label names: {names} -> real={real_idx} ({names[real_idx]}), fake={fake_idx} ({names[fake_idx]})")
            return real_idx, fake_idx
    raise SystemExit("Could not tell which label is real/fake. Pass --real_value and --fake_value "
                     f"(label feature: {feature}).")


def collect(examples, image_col, label_col, real_value, fake_value, per_class, out_dir: Path, max_seen: int):
    """Consume an iterable of dicts, save up to per_class images per class. Returns counts."""
    counts = {"real": 0, "fake": 0}
    for cls in counts:
        (out_dir / cls).mkdir(parents=True, exist_ok=True)
    seen = 0
    for ex in examples:
        seen += 1
        label = ex[label_col]
        cls = "real" if label == real_value else "fake" if label == fake_value else None
        if cls is None or counts[cls] >= per_class:
            if seen >= max_seen or all(c >= per_class for c in counts.values()):
                break
            continue
        counts[cls] += 1
        ex[image_col].convert("RGB").save(out_dir / cls / f"{counts[cls]:06d}.png")
        if counts[cls] % 500 == 0:
            print(f"  saved {counts}")
        if all(c >= per_class for c in counts.values()) or seen >= max_seen:
            break
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Hugging Face dataset id, e.g. owner/name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--config", default=None)
    ap.add_argument("--image_col", default=None)
    ap.add_argument("--label_col", default=None)
    ap.add_argument("--real_value", default=None, type=int)
    ap.add_argument("--fake_value", default=None, type=int)
    ap.add_argument("--per_class", type=int, default=6000)
    ap.add_argument("--max_seen", type=int, default=200000, help="Stop after reading this many rows.")
    ap.add_argument("--shuffle_buffer", type=int, default=5000)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    from datasets import Image as HFImage, load_dataset

    ds = load_dataset(args.repo, args.config, split=args.split, streaming=True)
    feats = ds.features
    print("Columns:", {k: type(v).__name__ for k, v in feats.items()})
    image_col = args.image_col or next(k for k, v in feats.items() if isinstance(v, HFImage))
    label_col = args.label_col or next(k for k, v in feats.items() if k.lower() in ("label", "labels", "class", "target"))
    real_value, fake_value = resolve_label_values(feats[label_col], args.real_value, args.fake_value)
    ds = ds.shuffle(seed=42, buffer_size=args.shuffle_buffer)

    counts = collect(ds, image_col, label_col, real_value, fake_value, args.per_class, Path(args.out_dir), args.max_seen)
    print(f"\nDONE. Saved {counts} to {args.out_dir}")
    if min(counts.values()) < args.per_class:
        print("Note: one class has fewer images than requested - the stream ended or max_seen was reached.")


if __name__ == "__main__":
    main()
