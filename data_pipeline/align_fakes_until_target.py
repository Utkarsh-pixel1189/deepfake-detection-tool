import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN

from prepare_image_faces import downscale
from mtcnn_align import OUTPUT_SIZE, select_best_face

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--target", type=int, default=6000)
ap.add_argument("--min_face", type=int, default=112)
ap.add_argument("--face_res", type=int, default=112)
ap.add_argument("--max_side", type=int, default=512)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
paths = sorted(p for p in Path(args.src).rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
random.Random(args.seed).shuffle(paths)
print(f"pool: {len(paths)} images")

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

kept, tried, widths = 0, 0, []
pbar = tqdm(total=args.target, unit="kept")
for p in paths:
    if kept >= args.target:
        break
    tried += 1
    try:
        img = downscale(Image.open(p).convert("RGB"), args.max_side)
        boxes, probs = mtcnn.detect(img)
        box = select_best_face(boxes, probs)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        if (x2 - x1) < args.min_face:
            continue
        face = img.crop((x1, y1, x2, y2))
        face = face.resize((args.face_res, args.face_res), Image.LANCZOS).resize(OUTPUT_SIZE, Image.BICUBIC)
        face.save(out / f"{p.stem}.jpg")
        widths.append(x2 - x1)
        kept += 1
        pbar.update(1)
    except Exception as e:
        print(f"error on {p.name}: {e}")
pbar.close()

print(f"\nDONE: kept {kept} of {tried} tried")
if widths:
    print(f"Detected face width BEFORE resize (px): median {np.median(widths):.0f}")
