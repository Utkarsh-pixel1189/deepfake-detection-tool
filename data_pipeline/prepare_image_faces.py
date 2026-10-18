"""
prepare_image_faces.py

Turns a folder of still images into aligned face crops that match the VIDEO
pipeline's face format (MTCNN crop -> 224x224 -> JPEG), after first
normalising image size so resolution cannot act as a real-vs-fake shortcut.

Why: the audit showed all 'real' images are exactly 512x512 while 'fake'
images range from ~500 to 3000 px. Without this step a model can learn
"big photo = fake" instead of looking at the face. Steps here:
  1. Downscale any image whose longest side is above --max_side (default 512)
     with Lanczos resampling. Smaller images are left untouched.
  2. Detect faces with MTCNN (same confidence rule as mtcnn_align.py), keep
     the largest confident face, crop and resize to 224x224.
  3. Save as JPEG with PIL's default quality - exactly how the training
     faces for the video model were saved - for BOTH classes.
It also prints the median detected-face width per run, so you can compare
real vs fake: if those numbers are far apart, a size shortcut remains.

Usage (run once per class, from data_pipeline/):
    python prepare_image_faces.py --in_dir /kaggle/working/hf_faces/real \
        --out_dir /kaggle/working/hf_faces_aligned/real
    python prepare_image_faces.py --in_dir /kaggle/working/hf_faces/fake \
        --out_dir /kaggle/working/hf_faces_aligned/fake
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def downscale(img: Image.Image, max_side: int) -> Image.Image:
    """Shrink so the longest side is at most max_side (never enlarges)."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_side", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from facenet_pytorch import MTCNN
    from tqdm import tqdm

    from mtcnn_align import OUTPUT_SIZE, select_best_face

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(args.in_dir).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        paths = paths[: args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device)

    kept, no_face, errors, face_widths = 0, 0, 0, []
    for p in tqdm(paths, unit="img"):
        target = out_dir / f"{p.stem}.jpg"
        if target.exists():                       # resume support
            kept += 1
            continue
        try:
            img = downscale(Image.open(p).convert("RGB"), args.max_side)
            boxes, probs = mtcnn.detect(img)
            box = select_best_face(boxes, probs)
            if box is None:
                no_face += 1
                continue
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.width, x2), min(img.height, y2)
            face_widths.append(x2 - x1)
            img.crop((x1, y1, x2, y2)).resize(OUTPUT_SIZE).save(target)   # JPEG, PIL default quality
            kept += 1
        except Exception as e:
            errors += 1
            print(f"  error on {p.name}: {e}")

    print(f"\nDONE {args.in_dir}: kept {kept}, no face {no_face}, errors {errors}")
    if face_widths:
        fw = np.array(face_widths)
        print(f"Detected face width BEFORE resize (px): median {np.median(fw):.0f}, "
              f"10th pct {np.percentile(fw, 10):.0f}, 90th pct {np.percentile(fw, 90):.0f}")
        print("Compare this line between real and fake: similar numbers = no size shortcut left.")


if __name__ == "__main__":
    main()
