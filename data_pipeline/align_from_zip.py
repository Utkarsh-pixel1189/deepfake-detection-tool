"""
align_from_zip.py

Aligns faces straight out of a dataset zip (nothing is extracted to disk),
same steps as training (MTCNN -> crop -> 112 -> 224 JPEG), but detecting
faces at full resolution (up to --max_side, default 2048) so faces in large
photos are not lost.

Goes through each label's images in random order until --target faces are
kept (or the label runs out). Safe to rerun: already saved faces are counted.

Labels come from either:
  zipdirs : folder names inside the zip containing "real" or "fake"
            (optionally only paths containing --only, e.g. /train/)
  zipcsv  : a CSV with image_name,label columns -> output folders label0, label1

Usage (from data_pipeline/):
  python align_from_zip.py zipdirs C:\\...\\archive.zip --out C:\\...\\archive_aligned --only /train/ --target 6000
  python align_from_zip.py zipcsv  C:\\...\\val_images.zip C:\\...\\val_labels.csv --out C:\\...\\d6_aligned --target 5000
"""

import argparse
import csv
import io
import random
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["zipdirs", "zipcsv", "parquet"])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=6000, help="Faces to keep per label.")
    ap.add_argument("--only", default=None, help="zipdirs: only use paths containing this text, e.g. /train/")
    ap.add_argument("--max_side", type=int, default=2048)
    ap.add_argument("--min_face", type=int, default=112)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Only align these labels (e.g. --labels real). Default: all labels.")
    args = ap.parse_args()

    groups = defaultdict(list)
    if args.mode == "parquet":
        # inputs[0] is a glob, e.g. C:\...\dataset5\*.parquet ; label from label_name (or label)
        import glob

        import pyarrow.parquet as pq

        tables = []
        for fi, f in enumerate(sorted(glob.glob(args.inputs[0]))):
            t = pq.read_table(f)
            lab_col = "label_name" if "label_name" in t.column_names else "label"
            tables.append(t.column("image"))
            for ri, lab in enumerate(t.column(lab_col).to_pylist()):
                groups[str(lab)].append(f"pq{fi}_{ri}")
        if not tables:
            raise SystemExit(f"no parquet files match {args.inputs[0]}")

        def read_bytes(key):
            fi, ri = key[2:].split("_")
            return tables[int(fi)][int(ri)].as_py()["bytes"]
    else:
        z = zipfile.ZipFile(args.inputs[0])
        read_bytes = z.read
    if args.mode == "zipdirs":
        for name in z.namelist():
            p = PurePosixPath(name)
            if p.suffix.lower() not in IMAGE_EXTS or (args.only and args.only not in name):
                continue
            parts = [q.lower() for q in p.parts[:-1]]
            label = "real" if "real" in parts else "fake" if "fake" in parts else None
            if label:
                groups[label].append(name)
    elif args.mode == "zipcsv":
        by_name = {PurePosixPath(n).name: n for n in z.namelist() if PurePosixPath(n).suffix.lower() in IMAGE_EXTS}
        with open(args.inputs[1], newline="") as fh:
            for row in csv.DictReader(fh):
                nm = row.get("image_name") or row.get("filename") or row.get("file")
                if nm in by_name:
                    groups[f"label{row['label']}"].append(by_name[nm])

    from openfake_faces import FaceAligner   # same alignment steps as training

    aligner = FaceAligner(max_side=args.max_side, min_face=args.min_face)
    rng = random.Random(args.seed)
    if args.labels:
        groups = {k: v for k, v in groups.items() if k in args.labels}
        if not groups:
            raise SystemExit(f"none of the labels {args.labels} found")
    for label in sorted(groups):
        names = sorted(groups[label])
        rng.shuffle(names)
        out = Path(args.out) / label
        out.mkdir(parents=True, exist_ok=True)
        kept = sum(1 for _ in out.glob("*.jpg"))
        tried, no_face, small, errors, widths = 0, 0, 0, 0, []
        print(f"{label}: {len(names)} images available, {kept} already saved, target {args.target}", flush=True)
        for name in names:
            if kept >= args.target:
                break
            target = out / (PurePosixPath(name).stem + ".jpg")
            if target.exists():
                continue
            tried += 1
            try:
                face, info = aligner(Image.open(io.BytesIO(read_bytes(name))).convert("RGB"))
            except Exception as e:
                errors += 1
                print(f"  error on {name}: {e}")
                continue
            if face is None:
                no_face += info == "no_face"
                small += info == "too_small"
                continue
            face.save(target)
            kept += 1
            widths.append(info)
            if kept % 500 == 0:
                print(f"  {label}: {kept} kept ({tried} tried)", flush=True)
        w = f"median {np.median(widths):.0f}" if widths else "-"
        print(f"DONE {label}: kept {kept} | tried {tried} | no face {no_face} | too small {small} | "
              f"errors {errors} | face width {w}", flush=True)


if __name__ == "__main__":
    main()
