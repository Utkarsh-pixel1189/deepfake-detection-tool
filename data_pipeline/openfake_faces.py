"""
openfake_faces.py

Builds an aligned face set from the OpenFake test split
(ComplexDataLab/OpenFake, core/test-*.parquet, about 68 GB) WITHOUT
downloading it to disk: each parquet shard is read in small batches over
HTTP, images with a usable face are aligned exactly like
prepare_image_faces.py, and everything else is discarded.

Output layout (one folder per generator, so each can be held out later):
    <out_dir>/real/<row_id>.jpg
    <out_dir>/fake_<model>/<row_id>.jpg

Quotas: up to --per_model faces per fake generator and --real_target real
faces. Rows whose quota is already full are skipped without face detection.
An optional prompt keyword filter skips rows that are unlikely to show a
person (it only decides which images get checked; every kept image still has
to pass the face detector).

Safe to rerun: images already saved are counted and skipped.

Usage (Kaggle GPU, from data_pipeline/):
    python openfake_faces.py --out_dir /kaggle/working/openfake_aligned
"""

import argparse
import io
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

REPO = "datasets/ComplexDataLab/OpenFake"
PERSON_WORDS = re.compile(
    r"\b(person|people|man|men|woman|women|girl|girls|boy|boys|child|children|kid|kids|face|faces|"
    r"portrait|selfie|guy|lady|ladies|player|actor|actress|politician|president|minister|couple|"
    r"family|teen|teenager|student|doctor|soldier|officer|worker|singer|athlete|bride|groom|"
    r"he|she|his|her|him|headshot|smiling|smile)\b", re.IGNORECASE)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")[:60] or "unknown"


class FaceAligner:
    """Same steps as prepare_image_faces.py: downscale, MTCNN, min face, 112 -> 224."""

    def __init__(self, max_side=512, min_face=112, face_res=112):
        import torch
        from facenet_pytorch import MTCNN

        from mtcnn_align import OUTPUT_SIZE, select_best_face
        from prepare_image_faces import downscale

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.select_best_face, self.downscale, self.output_size = select_best_face, downscale, OUTPUT_SIZE
        self.max_side, self.min_face, self.face_res = max_side, min_face, face_res

    def __call__(self, img):
        """Returns (face_224, face_width) or (None, reason)."""
        img = self.downscale(img, self.max_side)
        boxes, probs = self.mtcnn.detect(img)
        box = self.select_best_face(boxes, probs)
        if box is None:
            return None, "no_face"
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        if (x2 - x1) < self.min_face:
            return None, "too_small"
        face = img.crop((x1, y1, x2, y2))
        face = face.resize((self.face_res, self.face_res), Image.LANCZOS).resize(self.output_size, Image.BICUBIC)
        return face, float(x2 - x1)


def is_real(label) -> bool:
    return str(label).strip().lower() in {"real", "0", "authentic", "true_real"}


def main(argv=None, aligner=None, open_file=None, files=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--per_model", type=int, default=400, help="Max faces per fake generator.")
    ap.add_argument("--real_target", type=int, default=6000)
    ap.add_argument("--no_keyword_filter", action="store_true")
    ap.add_argument("--batch_rows", type=int, default=32)
    ap.add_argument("--min_face", type=int, default=112)
    ap.add_argument("--max_side", type=int, default=2048,
                    help="Detect faces at up to this resolution (full resolution keeps faces in large images).")
    args = ap.parse_args(argv)

    if files is None or open_file is None:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        files = sorted(fs.glob(f"{REPO}/core/{args.split}-*.parquet"))
        open_file = fs.open
    print(f"{len(files)} shards in split '{args.split}'")

    # Pass 1: read only label/model (small columns) to set the quotas.
    meta = Counter()
    for f in files:
        t = pq.read_table(open_file(f), columns=["label", "model"]).to_pydict()
        for lab, model in zip(t["label"], t["model"]):
            meta["real" if is_real(lab) else f"fake_{safe_name(model)}"] += 1
    quota = {k: (args.real_target if k == "real" else args.per_model) for k in meta}
    print("rows per class in this split -> quota:")
    for k in sorted(meta):
        print(f"  {k:40s} {meta[k]:6d} -> {quota[k]}")

    out = Path(args.out_dir)
    kept = Counter()
    for k in quota:
        d = out / k
        if d.exists():
            kept[k] = sum(1 for _ in d.glob("*.jpg"))
    if sum(kept.values()):
        print(f"resuming: {sum(kept.values())} faces already saved")

    if aligner is None:
        aligner = FaceAligner(max_side=args.max_side, min_face=args.min_face)
    seen, skipped_kw, reasons, widths = Counter(), 0, Counter(), {"real": [], "fake": []}

    def all_full():
        return all(kept[k] >= quota[k] for k in quota)

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None

    for fi, f in enumerate(files):
        if all_full():
            break
        pf = pq.ParquetFile(open_file(f))
        batches = pf.iter_batches(batch_size=args.batch_rows,
                                  columns=["image", "label", "model", "prompt", "row_id"])
        it = tqdm(batches, desc=f"shard {fi + 1}/{len(files)}", unit="batch") if tqdm else batches
        for batch in it:
            rows = batch.to_pydict()
            for img, lab, model, prompt, row_id in zip(rows["image"], rows["label"], rows["model"],
                                                       rows["prompt"], rows["row_id"]):
                key = "real" if is_real(lab) else f"fake_{safe_name(model)}"
                if kept[key] >= quota.get(key, 0):
                    continue
                if not args.no_keyword_filter and prompt and not PERSON_WORDS.search(str(prompt)):
                    skipped_kw += 1
                    continue
                target = out / key / f"{safe_name(row_id)}.jpg"
                if target.exists():
                    continue
                seen[key] += 1
                try:
                    pil = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
                    face, info = aligner(pil)
                except Exception as e:  # corrupt image etc.
                    reasons["error"] += 1
                    print(f"  error on {row_id}: {e}")
                    continue
                if face is None:
                    reasons[info] += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                face.save(target)
                kept[key] += 1
                widths["real" if key == "real" else "fake"].append(info)
            if all_full():
                break
        print(f"after shard {fi + 1}: real {kept['real']}/{quota.get('real', 0)}, "
              f"fake {sum(v for k, v in kept.items() if k != 'real')}/"
              f"{sum(v for k, v in quota.items() if k != 'real')}")

    print("\n=== DONE ===")
    print(f"skipped by keyword filter: {skipped_kw}; no face: {reasons['no_face']}; "
          f"too small: {reasons['too_small']}; errors: {reasons['error']}")
    for k in sorted(quota):
        print(f"  {k:40s} kept {kept[k]:5d} / quota {quota[k]:5d}  (checked {seen[k]})")
    for cls, w in widths.items():
        if w:
            print(f"face width BEFORE resize, {cls}: median {np.median(w):.0f}, "
                  f"10th {np.percentile(w, 10):.0f}, 90th {np.percentile(w, 90):.0f}")
    return kept


if __name__ == "__main__":
    main()
