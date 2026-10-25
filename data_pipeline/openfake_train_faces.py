"""
openfake_train_faces.py

Pulls face crops for specific generators (Grok, ChatGPT/GPT Image, Gemini Nano
Banana, Google Imagen) plus real Pexels photos from OpenFake's TRAINING split
(ComplexDataLab/OpenFake, core/train-*.parquet, ~3.35 TB) by STREAMING a random
subset of shards. Nothing large is written to disk: each ~96 MB block of rows is
read into memory, faces are aligned exactly like training (full-resolution
detection, 112 -> 224), and the block is discarded.

Only blocks that contain a wanted row are downloaded. Several shards are read
in parallel; face detection runs on the GPU in the main thread.

Progress (shards done, GB downloaded, MB/s, faces kept, estimated finish time)
is printed about once a minute, so it can be followed in the Kaggle Logs of a
"Save & Run All" background run.

Output:
    <out>/fake_<model>/<row_id>.jpg     one folder per generator
    <out>/real/<row_id>.jpg             Pexels real photos

Usage (Kaggle, Internet ON, GPU ON, HF_TOKEN logged in):
    python openfake_train_faces.py --out /kaggle/working/openfake_train_aligned --shards 40
"""

import argparse
import io
import queue
import random
import re
import threading
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

REPO = "datasets/ComplexDataLab/OpenFake"

# Faces to keep per generator. Rare ones (Grok, Nano Banana) are effectively uncapped.
DEFAULT_QUOTAS = {
    "grok-2-image-1212": 100000,
    "grok-imagine": 100000,
    "nano-banana-2": 100000,
    "gpt-image-1": 1500,
    "imagen-4.0": 500,
    "imagen-3.0-002": 200,
    "pexels": 600,           # real photos (big files, slow to check, so capped)
}


def safe(s) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")[:60] or "x"


def folder_for(model: str) -> str:
    return "real" if model == "pexels" else f"fake_{safe(model)}"


def fmt_time(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def main(argv=None, files=None, open_file=None, aligner=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=40, help="How many random training shards to stream.")
    ap.add_argument("--workers", type=int, default=8, help="Shards downloaded in parallel.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--min_face", type=int, default=112)
    ap.add_argument("--quota", action="append", default=[], help="Override, e.g. --quota gpt-image-1=2000")
    ap.add_argument("--split", default="train")
    args = ap.parse_args(argv)

    quota = dict(DEFAULT_QUOTAS)
    for q in args.quota:
        k, v = q.split("=", 1)
        quota[k] = int(v)

    if files is None:
        from huggingface_hub import HfFileSystem

        files = sorted(HfFileSystem().glob(f"{REPO}/core/{args.split}-*.parquet"))

        def open_file(f):
            return HfFileSystem().open(f, block_size=1 << 20)

    chosen = random.Random(args.seed).sample(files, min(args.shards, len(files)))
    n_workers = max(1, min(args.workers, len(chosen)))
    print(f"streaming {len(chosen)} of {len(files)} shards with {n_workers} parallel readers", flush=True)

    out = Path(args.out)
    kept = Counter()
    for model in quota:
        d = out / folder_for(model)
        if d.exists():
            kept[model] = sum(1 for _ in d.glob("*.jpg"))
    if sum(kept.values()):
        print(f"resuming: {sum(kept.values())} faces already saved", flush=True)

    def is_full(model):
        return kept[model] >= quota.get(model, 0)

    q = queue.Queue(maxsize=64)

    def decode(data):
        """Decode AND shrink in the reader threads (parallel), so the main thread
        only runs face detection. JPEGs are decoded straight at reduced size."""
        img = Image.open(io.BytesIO(data))
        if img.format == "JPEG":
            img.draft("RGB", (args.max_side, args.max_side))
        img = img.convert("RGB")
        if max(img.size) > args.max_side:          # same Lanczos shrink training used
            s = args.max_side / max(img.size)
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
        return img

    lock = threading.Lock()
    stats = {"bytes": 0, "shards_done": 0, "errors": 0, "rg_done": 0, "rg_per_shard": 0, "rg_skipped": 0}

    # Blocks fully handled in earlier runs are recorded here, so a restart does
    # not download them again.
    out.mkdir(parents=True, exist_ok=True)
    done_file = out / "_done_blocks.txt"
    done_blocks = set(done_file.read_text().split()) if done_file.exists() else set()
    if done_blocks:
        print(f"resuming: {len(done_blocks)} blocks already handled, they will be skipped", flush=True)

    def mark_done(key):
        with lock:
            with open(done_file, "a") as fh:
                fh.write(key + "\n")

    def reader(paths):
        for f in paths:
            name = str(f).split("/")[-1]
            try:
                pf = pq.ParquetFile(open_file(f), pre_buffer=True)
                has_row_id = "row_id" in pf.schema_arrow.names
                with lock:
                    stats["rg_per_shard"] = max(stats["rg_per_shard"], pf.metadata.num_row_groups)
                for rg in range(pf.metadata.num_row_groups):
                    key = f"{name}:{rg}"
                    with lock:
                        stats["rg_done"] += 1
                    if key in done_blocks:
                        with lock:
                            stats["rg_skipped"] += 1
                        continue
                    meta = pf.read_row_group(rg, columns=["label", "model"]).to_pydict()
                    rows = [i for i, (lab, m) in enumerate(zip(meta["label"], meta["model"]))
                            if m in quota and not is_full(m)
                            and ((str(lab).strip().lower() == "real") == (m == "pexels"))]
                    if not rows:
                        mark_done(key)
                        continue
                    cols = ["image", "row_id"] if has_row_id else ["image"]
                    t = pf.read_row_group(rg, columns=cols)
                    rgm = pf.metadata.row_group(rg)
                    size = sum(rgm.column(j).total_compressed_size for j in range(rgm.num_columns)
                               if rgm.column(j).path_in_schema.split(".")[0] in cols)
                    with lock:
                        stats["bytes"] += size
                    images = t.column("image")
                    ids = t.column("row_id") if has_row_id else None
                    for i in rows:
                        rid = ids[i].as_py() if ids is not None else f"{name}_{rg}_{i}"
                        model = meta["model"][i]
                        if is_full(model) or (out / folder_for(model) / f"{safe(rid)}.jpg").exists():
                            continue
                        try:
                            img = decode(images[i].as_py()["bytes"])
                        except Exception:
                            img = None
                        q.put((model, rid, img))
                    mark_done(key)
            except Exception as e:
                with lock:
                    stats["errors"] += 1
                print(f"  error reading {name}: {str(e)[:200]}", flush=True)
            with lock:
                stats["shards_done"] += 1
        q.put(None)

    threads = [threading.Thread(target=reader, args=(chosen[i::n_workers],), daemon=True) for i in range(n_workers)]
    for th in threads:
        th.start()

    if aligner is None:
        from openfake_faces import FaceAligner

        aligner = FaceAligner(max_side=args.max_side, min_face=args.min_face)
        # Faces narrower than min_face are thrown away anyway, so tell the detector
        # not to search for tiny faces: far fewer image scales, much faster.
        from facenet_pytorch import MTCNN

        aligner.mtcnn = MTCNN(keep_all=True, device=aligner.device,
                              min_face_size=max(20, int(args.min_face * 0.6)))
        print(f"face detection on {aligner.device}, images shrunk to {args.max_side}px", flush=True)

    t0 = last = time.time()
    finished, checked, no_face, small, bad = 0, Counter(), 0, 0, 0

    def report(final=False):
        el = time.time() - t0
        with lock:
            done, gb = stats["shards_done"], stats["bytes"] / 1e9
        speed = gb * 1000 / el if el > 0 else 0
        # Estimate from row groups (blocks) processed, so it works before any shard finishes.
        with lock:
            rg_done, rg_skip = stats["rg_done"], stats["rg_skipped"]
            rg_total = stats["rg_per_shard"] * len(chosen)
        work_done = rg_done - rg_skip                 # blocks actually processed this run
        eta = el / work_done * (rg_total - rg_done) if work_done and rg_total else float("nan")
        eta_txt = "-" if final else ("estimating..." if eta != eta else fmt_time(eta))
        kept_txt = ", ".join(f"{m} {kept[m]}" for m in quota)
        rate = sum(checked.values()) / el if el > 0 else 0
        print(f"[{fmt_time(el)}] blocks {rg_done}/{rg_total or '?'} | {gb:.1f} GB at {speed:.0f} MB/s | "
              f"faces checked {rate:.1f}/s | time left ~{eta_txt} | queue {q.qsize()}/64 | kept: {kept_txt}",
              flush=True)

    while finished < n_workers:
        try:
            item = q.get(timeout=30)
        except queue.Empty:
            item = "tick"
        if item is None:
            finished += 1
        elif item != "tick":
            model, rid, data = item
            target = out / folder_for(model) / f"{safe(rid)}.jpg"
            if not is_full(model) and not target.exists():
                checked[model] += 1
                try:
                    if data is None:
                        raise ValueError("unreadable image")
                    face, info = aligner(data)
                except Exception:
                    bad += 1
                    face, info = None, "error"
                if face is None:
                    no_face += info == "no_face"
                    small += info == "too_small"
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    face.save(target)
                    kept[model] += 1
        if time.time() - last >= 60:
            report()
            last = time.time()

    report(final=True)
    print("\n=== DONE ===")
    print(f"total time {fmt_time(time.time() - t0)} | no face {no_face} | too small {small} | "
          f"unreadable images {bad} | shard read errors {stats['errors']}")
    for m in quota:
        print(f"  {folder_for(m):30s} kept {kept[m]:5d}   (checked {checked[m]}, quota {quota[m]})")
    return kept


if __name__ == "__main__":
    main()
