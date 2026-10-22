"""
loso_from_cache.py

Leave-one-source-out test from saved CLIP features (output of
extract_clip_features.py). Runs on CPU in seconds - no GPU, torch or
transformers needed, only numpy and scikit-learn.

For each source:
  1. train a logistic regression on all the OTHER sources
  2. choose C and the decision threshold on a validation split of those
     training sources (step 5 of the plan: threshold from several sources)
  3. test once on the held-out source, overall and per subfolder
     (e.g. how many fake_inpainting images were caught)

Usage (from the repo folder, on your PC):
    python data_pipeline/loso_from_cache.py --cache_dir C:/Users/stren/Documents/datasets/clip_features
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def scores(y, p, threshold=0.5) -> dict:
    pred = (p >= threshold).astype(int)
    fake_recall = float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    real_spec = float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "fake_recall": fake_recall, "real_specificity": real_spec,
            "balanced_accuracy": (fake_recall + real_spec) / 2, "auc": auc}


def best_threshold(y, p) -> float:
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        s = scores(y, p, t)["balanced_accuracy"]
        if s > best_s:
            best_t, best_s = float(t), s
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True, help="Folder with one <source>.npz per source.")
    ap.add_argument("--only", nargs="*", default=None, help="Optional subset of source names to use.")
    ap.add_argument("--out_json", default="loso_results.json")
    args = ap.parse_args()

    data = {}
    for f in sorted(Path(args.cache_dir).glob("*.npz")):
        if args.only and f.stem not in args.only:
            continue
        z = np.load(f, allow_pickle=False)
        data[f.stem] = (z["X"], z["y"], z["g"])
        print(f"loaded {f.stem}: {len(z['y'])} images "
              f"(real {int((z['y'] == 0).sum())}, fake {int((z['y'] == 1).sum())})")
    if len(data) < 2:
        raise SystemExit("Need at least two sources in --cache_dir.")

    results, summary = {}, []
    print("\n=== Leave-one-source-out (frozen CLIP + logistic regression) ===")
    for held in data:
        train_names = [n for n in data if n != held]
        X = np.concatenate([data[n][0] for n in train_names])
        y = np.concatenate([data[n][1] for n in train_names])
        X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

        best = None
        for C in (0.1, 1.0, 10.0):
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X_fit, y_fit)
            b = scores(y_val, clf.predict_proba(X_val)[:, 1])["balanced_accuracy"]
            if best is None or b > best[0]:
                best = (b, C, clf)
        _, C, clf = best
        thr = best_threshold(y_val, clf.predict_proba(X_val)[:, 1])

        Xt, yt, gt = data[held]
        pt = clf.predict_proba(Xt)[:, 1]
        r = {"train_sources": train_names, "C": C, "val_threshold": thr,
             "at_0.5": scores(yt, pt, 0.5), "at_val_threshold": scores(yt, pt, thr), "per_folder": {}}
        for folder in np.unique(gt):
            m = gt == folder
            lab = int(yt[m][0])
            pred = (pt[m] >= thr).astype(int)
            r["per_folder"][str(folder)] = {"label": "fake" if lab else "real", "n": int(m.sum()),
                                            "correct_rate": float((pred == lab).mean())}
        results[held] = r

        a, b = r["at_0.5"], r["at_val_threshold"]
        print(f"\nHELD OUT: {held}  (trained on {', '.join(train_names)}; C={C})")
        print(f"  AUC {a['auc']:.3f}")
        print(f"  @0.50  balanced {a['balanced_accuracy']:.3f}  fake recall {a['fake_recall']:.3f}  real {a['real_specificity']:.3f}")
        print(f"  @{thr:.2f}  balanced {b['balanced_accuracy']:.3f}  fake recall {b['fake_recall']:.3f}  real {b['real_specificity']:.3f}")
        for name, v in r["per_folder"].items():
            print(f"    {name:18s} ({v['label']}, n={v['n']}): correct {v['correct_rate']:.3f}")
        summary.append((held, a["auc"], b["balanced_accuracy"], thr))

    print("\n=== Summary ===")
    print(f"{'held out':12s} {'AUC':>6s} {'balanced@val_thr':>17s} {'thr':>5s}")
    for held, auc, bal, thr in summary:
        print(f"{held:12s} {auc:6.3f} {bal:17.3f} {thr:5.2f}")

    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
