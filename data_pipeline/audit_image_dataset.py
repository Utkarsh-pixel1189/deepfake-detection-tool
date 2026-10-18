"""
audit_image_dataset.py

Quick sanity audit of a real-vs-fake IMAGE dataset BEFORE training on it.
Looks for the things that silently ruin image detectors:

  1. Format / size shortcuts - if all real images are JPEGs of varying size and
     all fake images are 1024x1024 PNGs, a model can score 99% without ever
     looking at the face. Printed per class, with warnings.
  2. Label noise - identical files that appear in BOTH classes, and exact
     duplicates inside a class.
  3. A contact sheet (random images per class) so you can LOOK at the labels.

Usage:
    python audit_image_dataset.py --real /path/to/real --fake /path/to/fake \
        --name kaggle_faces --out_dir /kaggle/working/audit_kaggle

Folders are searched recursively. Up to --max_files images per class are
sampled (random, seed 42) so it stays fast on 30k+ datasets.
"""

import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(folder: str) -> list:
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(paths: list) -> dict:
    """Format / size statistics for a list of image paths."""
    exts, modes, widths, heights, sizes = Counter(), Counter(), [], [], []
    unreadable = 0
    for p in paths:
        try:
            with Image.open(p) as im:
                w, h = im.size
                modes[im.mode] += 1
            widths.append(w)
            heights.append(h)
            sizes.append(p.stat().st_size)
            exts[p.suffix.lower()] += 1
        except Exception:
            unreadable += 1
    n = max(1, len(widths))
    return {
        "n_read": len(widths), "unreadable": unreadable,
        "ext_share": {k: v / n for k, v in exts.items()},
        "mode_share": {k: v / n for k, v in modes.items()},
        "w_min": int(min(widths)) if widths else 0, "w_med": float(np.median(widths)) if widths else 0,
        "w_max": int(max(widths)) if widths else 0,
        "h_med": float(np.median(heights)) if heights else 0,
        "n_unique_sizes": len(set(zip(widths, heights))),
        "kb_med": float(np.median(sizes)) / 1024 if sizes else 0,
    }


def shortcut_warnings(real: dict, fake: dict) -> list:
    warns = []
    for ext in set(real["ext_share"]) | set(fake["ext_share"]):
        diff = abs(real["ext_share"].get(ext, 0) - fake["ext_share"].get(ext, 0))
        if diff > 0.5:
            warns.append(f"File type {ext} differs by {diff:.0%} between classes - a model can learn the format, not the face.")
    if real["w_med"] and fake["w_med"]:
        ratio = max(real["w_med"], fake["w_med"]) / max(1.0, min(real["w_med"], fake["w_med"]))
        if ratio > 1.5:
            warns.append(f"Median width differs {ratio:.1f}x (real {real['w_med']:.0f}px vs fake {fake['w_med']:.0f}px) - resolution shortcut.")
    for cls, d in (("real", real), ("fake", fake)):
        if d["n_read"] > 50 and d["n_unique_sizes"] == 1:
            warns.append(f"All {cls} images have exactly the same size - generator/export fingerprint.")
    kb_ratio = max(real["kb_med"], fake["kb_med"]) / max(0.1, min(real["kb_med"], fake["kb_med"]))
    if kb_ratio > 2.5:
        warns.append(f"Median file size differs {kb_ratio:.1f}x (real {real['kb_med']:.0f}KB vs fake {fake['kb_med']:.0f}KB) - compression shortcut.")
    return warns


def contact_sheet(paths: list, out_path: Path, n: int = 20, thumb: int = 160, cols: int = 10, seed: int = 42):
    rng = random.Random(seed)
    chosen = rng.sample(paths, min(n, len(paths)))
    rows = (len(chosen) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, max(1, rows) * thumb), (30, 30, 30))
    for i, p in enumerate(chosen):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb, thumb))
                sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
        except Exception:
            pass
    sheet.save(out_path)


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    all_real, all_fake = list_images(args.real), list_images(args.fake)
    print(f"[{args.name}] images found -> real: {len(all_real)}, fake: {len(all_fake)}")
    if not all_real or not all_fake:
        raise SystemExit("One of the classes is empty - check the folder paths.")

    real = rng.sample(all_real, min(args.max_files, len(all_real)))
    fake = rng.sample(all_fake, min(args.max_files, len(all_fake)))

    rd, fd = describe(real), describe(fake)
    for cls, d in (("REAL", rd), ("FAKE", fd)):
        print(f"\n{cls} (sampled {d['n_read']}, unreadable {d['unreadable']})")
        print(f"  types : { {k: f'{v:.0%}' for k, v in d['ext_share'].items()} }")
        print(f"  modes : { {k: f'{v:.0%}' for k, v in d['mode_share'].items()} }")
        print(f"  width : min {d['w_min']}  median {d['w_med']:.0f}  max {d['w_max']}   (distinct sizes: {d['n_unique_sizes']})")
        print(f"  file  : median {d['kb_med']:.0f} KB")

    warns = shortcut_warnings(rd, fd)
    print("\n=== SHORTCUT CHECK ===")
    print("\n".join(f"  WARNING: {w}" for w in warns) if warns else "  No obvious format/size shortcuts in the sample.")

    # Duplicates / label noise (hash the sampled files)
    hr = {}
    for p in real:
        hr.setdefault(md5(p), []).append(p)
    hf = {}
    for p in fake:
        hf.setdefault(md5(p), []).append(p)
    cross = set(hr) & set(hf)
    dup_real = sum(len(v) - 1 for v in hr.values())
    dup_fake = sum(len(v) - 1 for v in hf.values())
    print("\n=== LABEL NOISE CHECK (exact duplicates, sampled files) ===")
    print(f"  identical image in BOTH classes : {len(cross)}   <- any number above 0 means contradictory labels")
    print(f"  duplicates inside real          : {dup_real}")
    print(f"  duplicates inside fake          : {dup_fake}")
    if cross:
        print("  examples:")
        for h in list(cross)[:5]:
            print(f"    real={hr[h][0]}  fake={hf[h][0]}")

    contact_sheet(real, out_dir / f"{args.name}_real.png")
    contact_sheet(fake, out_dir / f"{args.name}_fake.png")
    print(f"\nContact sheets saved: {out_dir}/{args.name}_real.png and {args.name}_fake.png")
    print("OPEN BOTH and check by eye: do the 'real' images look real and the 'fake' ones synthetic?")


def main():
    ap = argparse.ArgumentParser(description="Audit a real/fake image dataset before training.")
    ap.add_argument("--real", required=True)
    ap.add_argument("--fake", required=True)
    ap.add_argument("--name", default="dataset")
    ap.add_argument("--out_dir", default="audit")
    ap.add_argument("--max_files", type=int, default=3000)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
