"""
fetch_hf_zip_subset.py

For Hugging Face image datasets that are stored as ONE big .zip file
(e.g. prithivMLmods/Deepfake-vs-Real-60K = a single 24 GB zip), so
streaming with `datasets` cannot work and the labels come out wrong.

This reads the zip's table of contents over HTTP and downloads ONLY the
randomly chosen images (range requests) - not the whole 24 GB.

    out_dir/real/000001.png ...
    out_dir/fake/000001.png ...

Class is taken from the folder/file name (Fake / Real). All images are saved
as lossless PNG for BOTH classes so file format cannot become a shortcut.

Usage (Kaggle, internet ON, already logged in to Hugging Face):
    python fetch_hf_zip_subset.py --repo prithivMLmods/Deepfake-vs-Real-60K \
        --zip_name deepfake-vs-real-60k.zip --per_class 6000 --out_dir /kaggle/working/hf_faces

Check the "folder names found" table it prints before trusting the labels.
If you already have the zip on disk, pass --zip_path /path/to/file.zip instead.
"""

import argparse
import io
import random
import threading
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def classify(name: str):
    """'real' / 'fake' / None from the folder names (nearest folder first) or the file name."""
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    if not parts:
        return None
    candidates = list(reversed(parts[:-1])) + [Path(parts[-1]).stem]
    for comp in candidates:
        low = comp.lower()
        has_fake, has_real = "fake" in low, "real" in low
        if has_fake and not has_real:      # 'Fake', 'deepfake', ...
            return "fake"
        if has_real and not has_fake:      # 'Real', 'real_images', ...
            return "real"
        # a name with BOTH words (e.g. the dataset's own folder name) says nothing - keep looking
    return None


def open_zip(args):
    if args.zip_path:
        return zipfile.ZipFile(args.zip_path)
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()  # uses the token saved by huggingface_hub.login()
    path = f"datasets/{args.repo}/{args.zip_name}"
    try:
        handle = fs.open(path, "rb", block_size=1 << 16, cache_type="bytes")
    except TypeError:
        handle = fs.open(path, "rb")
    return zipfile.ZipFile(handle)


def run(args):
    out_dir = Path(args.out_dir)
    for cls in ("real", "fake"):
        (out_dir / cls).mkdir(parents=True, exist_ok=True)

    print("Reading the zip's table of contents (a few MB)...")
    zf = open_zip(args)
    infos = [i for i in zf.infolist() if not i.is_dir() and Path(i.filename).suffix.lower() in IMAGE_EXTS]
    print(f"{len(infos)} images inside the zip")

    by_class = {"real": [], "fake": []}
    unknown = 0
    folder_counter = Counter()
    for info in infos:
        cls = classify(info.filename)
        parent = "/".join(info.filename.split("/")[:-1][-2:])
        folder_counter[(parent, cls)] += 1
        if cls is None:
            unknown += 1
        else:
            by_class[cls].append(info)

    print("\nfolder names found (parent folder -> assigned class : count):")
    for (parent, cls), n in folder_counter.most_common(12):
        print(f"  {parent or '(root)':50s} -> {cls}: {n}")
    print(f"\nTotals -> real: {len(by_class['real'])}, fake: {len(by_class['fake'])}, unclassified: {unknown}")
    if not by_class["real"] or not by_class["fake"]:
        raise SystemExit("Could not find both classes from the folder names. Paste the table above to Claude.")

    rng = random.Random(42)
    tasks = []
    for cls, items in by_class.items():
        chosen = rng.sample(items, min(args.per_class, len(items)))
        tasks += [(info, cls, n + 1) for n, info in enumerate(chosen)]
    tasks.sort(key=lambda t: t[0].header_offset)       # sequential reads are faster
    chunks = [tasks[i::args.workers] for i in range(args.workers)]

    lock, done, failed = threading.Lock(), [0], [0]

    def work(chunk):
        local_zip = open_zip(args)
        for info, cls, idx in chunk:
            try:
                data = local_zip.read(info.filename)
                Image.open(io.BytesIO(data)).convert("RGB").save(out_dir / cls / f"{idx:06d}.png")
                ok = True
            except Exception as e:  # keep going: one bad file must not stop the batch
                ok = False
                print(f"  skipped {info.filename}: {e}")
            with lock:
                done[0] += ok
                failed[0] += (not ok)
                if done[0] % 500 == 0 and ok:
                    print(f"  saved {done[0]} / {len(tasks)}")

    print(f"\nDownloading {len(tasks)} images with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, chunks))
    print(f"\nDONE. Saved {done[0]} images ({failed[0]} failed) to {out_dir}")
    print({c: len(list((out_dir / c).glob('*.png'))) for c in ('real', 'fake')})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="prithivMLmods/Deepfake-vs-Real-60K")
    ap.add_argument("--zip_name", default="deepfake-vs-real-60k.zip")
    ap.add_argument("--zip_path", default=None, help="Use a zip already on disk instead of the Hub.")
    ap.add_argument("--per_class", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out_dir", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
