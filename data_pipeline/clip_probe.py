"""
clip_probe.py

Cheap, decisive experiment: does a FROZEN CLIP image encoder + a simple
linear classifier generalize across datasets / unseen manipulation methods
better than our fine-tuned Xception?

Why: the fine-tuned Xception scores ~96-97% on the Celeb-DF++ test split
but only ~51-57% on the unseen FaceShifter method and ~63-65% on the
FaceForensics++ test split (AUC 0.64-0.71). It learned the quirks of one
dataset. Published work on frozen CLIP features for fake detection reports
much better transfer to unseen generators; this script measures whether
that holds on OUR data, with the same splits and the same unseen sets, in
minutes (no fine-tuning).

What it does:
  1. Builds the same video-level train/val/test split of Celeb-DF++ used
     everywhere else (dataset.build_dataset_splits_cdf, seed 42).
  2. Extracts L2-normalised CLIP image embeddings for a few evenly spaced
     frames per video (cached to disk, safe to rerun).
  3. Trains a logistic regression on the Celeb-DF++ TRAIN frames only
     (class-balanced; C picked on VAL).
  4. Scores whole videos (mean of frame probabilities) on:
       - Celeb-DF++ test split                    (in-distribution)
       - each --holdout NAME=DIR (all fake) + the real test videos
       - the FaceForensics++ test split           (other dataset)
Nothing is tuned on any unseen set. Threshold is fixed at 0.5.

Usage (Kaggle, internet ON so the CLIP weights can download):
    python clip_probe.py \\
        --cdf_root /kaggle/input/datasets/setonirl/celeb-df-v3-faces \\
        --ffpp_root /kaggle/input/datasets/setonirl/deepfake-faces-ff-300/faces_local_300 \\
        --holdout FaceShifter=/kaggle/input/datasets/setonirl/deepfake-faces-ff-300/faces_local_300/faceshifter_holdout \\
        --cache /kaggle/working/clip_feats.npz --output_json /kaggle/working/clip_probe_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ----------------------------------------------------------------------------
# Sampling helpers (pure python / numpy)
# ----------------------------------------------------------------------------

def evenly_spaced(paths: list, k: int) -> list:
    if len(paths) <= k:
        return list(paths)
    idx = np.unique(np.linspace(0, len(paths) - 1, k).round().astype(int))
    return [paths[i] for i in idx]


def to_video_items(samples: list, k: int) -> list:
    """[(frame_path, label)] -> [(frame_path, label, video_key)], at most k evenly spaced frames per video."""
    groups = {}
    for p, l in samples:
        groups.setdefault(str(Path(p).parent), (l, []))[1].append(Path(p))
    items = []
    for video_key, (label, paths) in sorted(groups.items()):
        for p in evenly_spaced(sorted(paths), k):
            items.append((str(p), label, video_key))
    return items


def folder_items(folder: str, label: int, k: int, prefix: str) -> list:
    """<folder>/<video_id>/frame_*.jpg -> items, all with the same label."""
    items = []
    for video_dir in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
        frames = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for p in evenly_spaced(frames, k):
            items.append((str(p), label, f"{prefix}:{video_dir.name}"))
    return items


def video_scores(items: list, frame_probs) -> tuple:
    """Average frame probabilities per video -> (video_keys, labels, mean_probs)."""
    acc = {}
    for (_, label, key), p in zip(items, frame_probs):
        acc.setdefault(key, (label, []))[1].append(float(p))
    keys = sorted(acc)
    return keys, np.array([acc[k][0] for k in keys]), np.array([np.mean(acc[k][1]) for k in keys])


def metrics(labels, probs, threshold: float = 0.5) -> dict:
    labels, probs = np.asarray(labels), np.asarray(probs)
    preds = (probs >= threshold).astype(int)
    m = {"n_videos": int(len(labels)), "accuracy": float(accuracy_score(labels, preds))}
    if len(set(labels.tolist())) == 2:
        rec = float(((preds == 1) & (labels == 1)).sum() / (labels == 1).sum())
        spec = float(((preds == 0) & (labels == 0)).sum() / (labels == 0).sum())
        m.update(fake_recall=rec, real_specificity=spec, balanced_accuracy=(rec + spec) / 2,
                 auc_roc=float(roc_auc_score(labels, probs)))
    elif labels[0] == 1:
        m["detection_rate"] = m["accuracy"]
    return m


def fmt(m: dict) -> str:
    parts = [f"videos={m['n_videos']}", f"acc={m['accuracy']:.3f}"]
    for k in ("fake_recall", "real_specificity", "balanced_accuracy", "auc_roc", "detection_rate"):
        if k in m:
            parts.append(f"{k}={m[k]:.3f}")
    return "  ".join(parts)


# ----------------------------------------------------------------------------
# CLIP feature extraction (torch / transformers imported lazily)
# ----------------------------------------------------------------------------

def extract_features(paths: list, model_name: str, batch_size: int, cache_path: str) -> np.ndarray:
    cache = {}
    if cache_path and Path(cache_path).exists():
        z = np.load(cache_path, allow_pickle=False)
        cache = dict(zip(z["paths"].tolist(), z["feats"]))
        print(f"Loaded {len(cache)} cached feature vectors from {cache_path}")

    missing = [p for p in dict.fromkeys(paths) if p not in cache]
    if missing:
        import torch
        from PIL import Image
        from tqdm import tqdm
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        print(f"Extracting CLIP features for {len(missing)} frames with {model_name} on {device}...")
        processor = CLIPImageProcessor.from_pretrained(model_name)
        model = CLIPVisionModelWithProjection.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()

        def save():
            if cache_path:
                keys = list(cache.keys())
                np.savez(cache_path, paths=np.array(keys), feats=np.stack([cache[k] for k in keys]))

        for n, start in enumerate(tqdm(range(0, len(missing), batch_size), unit="batch")):
            chunk = missing[start:start + batch_size]
            images = [Image.open(p).convert("RGB") for p in chunk]
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device, dtype=dtype)
            with torch.no_grad():
                emb = model(pixel_values=pixels).image_embeds
            emb = torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy()
            for p, f in zip(chunk, emb):
                cache[p] = f
            if n % 150 == 149:
                save()
        save()

    return np.stack([cache[p] for p in paths])


# ----------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------

def run(args):
    from dataset import build_dataset_splits, build_dataset_splits_cdf

    train, val, test = build_dataset_splits_cdf(args.cdf_root)
    train_items = to_video_items(train, args.train_frames_per_video)
    val_items = to_video_items(val, args.eval_frames_per_video)
    test_items = to_video_items(test, args.eval_frames_per_video)
    real_test_items = [it for it in test_items if it[1] == 0]

    holdouts = {}
    for spec in args.holdout:
        name, path = spec.split("=", 1)
        holdouts[name] = folder_items(path, 1, args.eval_frames_per_video, f"holdout-{name}")

    ffpp_items = []
    if args.ffpp_root:
        _, _, ffpp_test = build_dataset_splits(args.ffpp_root)
        ffpp_items = to_video_items(ffpp_test, args.eval_frames_per_video)

    all_items = train_items + val_items + test_items + ffpp_items + [i for v in holdouts.values() for i in v]
    feats = extract_features([i[0] for i in all_items], args.model, args.batch_size, args.cache)
    feat_by_path = {it[0]: feats[k] for k, it in enumerate(all_items)}

    def X(items):
        return np.stack([feat_by_path[it[0]] for it in items])

    ytr = np.array([it[1] for it in train_items])
    best = None
    print("\nChoosing C on the validation videos (by balanced accuracy):")
    for C in (0.1, 1.0, 10.0, 100.0):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        clf.fit(X(train_items), ytr)
        _, vl, vp = video_scores(val_items, clf.predict_proba(X(val_items))[:, 1])
        m = metrics(vl, vp)
        print(f"  C={C:<6} {fmt(m)}")
        if best is None or m["balanced_accuracy"] > best[0]:
            best = (m["balanced_accuracy"], C, clf)
    _, C, clf = best
    print(f"-> using C={C}")

    def evaluate(name, items, results):
        _, labels, probs = video_scores(items, clf.predict_proba(X(items))[:, 1])
        results[name] = metrics(labels, probs)
        print(f"  {name:48s} {fmt(results[name])}")

    results = {"chosen_C": C, "model": args.model}
    print("\n=== Frozen CLIP + logistic regression (threshold 0.5, video-level) ===")
    evaluate("Celeb-DF++ test split (in-distribution)", test_items, results)
    for name, items in holdouts.items():
        evaluate(f"UNSEEN {name} alone (all fake)", items, results)
        evaluate(f"UNSEEN {name} + real test videos", real_test_items + items, results)
    if ffpp_items:
        evaluate("FaceForensics++ test split (other dataset)", ffpp_items, results)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"\nSaved results to {args.output_json}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Frozen CLIP + linear probe generalization experiment.")
    parser.add_argument("--cdf_root", required=True, help="Celeb-DF-v3 faces root (real/ + fake/ layout).")
    parser.add_argument("--ffpp_root", default=None, help="FF++ faces root (original/, deepfakes/, ...).")
    parser.add_argument("--holdout", action="append", default=[],
                        help="NAME=DIR of aligned-face video folders, all fake (repeatable).")
    parser.add_argument("--model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--train_frames_per_video", type=int, default=4)
    parser.add_argument("--eval_frames_per_video", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cache", default="clip_feats.npz")
    parser.add_argument("--output_json", default="clip_probe_results.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()