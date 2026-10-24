"""
count_openfake.py

Counts how many fake images each generator has in OpenFake's TRAINING split
(ComplexDataLab/OpenFake, core/train-*.parquet). Reads only the small
"label" and "model" columns, never the images, so it downloads only a small
fraction of the 3.4 TB dataset.

Usage:
    python count_openfake.py
Results are printed and saved to openfake_train_counts.csv.
"""

import csv
from collections import Counter

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem


def main():
    fs = HfFileSystem()
    files = sorted(fs.glob("datasets/ComplexDataLab/OpenFake/core/train-*.parquet"))
    print(len(files), "train shards", flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def read_one(f):
        # Each shard is a few small network reads, so many can run at once.
        return pq.read_table(HfFileSystem().open(f), columns=["label", "model"]).to_pydict()

    fakes, reals, done = Counter(), 0, 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(read_one, f): f for f in files}
        for fut in as_completed(futures):
            done += 1
            try:
                t = fut.result()
            except Exception as e:
                print(f"  could not read {futures[fut].split('/')[-1]}: {str(e)[:150]}", flush=True)
                continue
            for label, model in zip(t["label"], t["model"]):
                if str(label).strip().lower() == "real":
                    reals += 1
                else:
                    fakes[str(model)] += 1
            if done % 50 == 0:
                print(f"{done}/{len(files)} shards read", flush=True)

    print(f"\nreal rows: {reals}")
    print("fake rows per generator:")
    for model, n in fakes.most_common():
        print(f"  {model:40s} {n}")

    with open("openfake_train_counts.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "fake_rows"])
        w.writerows(fakes.most_common())
        w.writerow(["real", reals])
    print("\nsaved openfake_train_counts.csv")


if __name__ == "__main__":
    main()
