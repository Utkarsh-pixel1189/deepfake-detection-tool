"""
train_clip_head.py

Trains the FINAL image classifier: a logistic regression on frozen CLIP
features, using ALL sources together (output of extract_clip_features.py).

  - 15% of every source is held out as validation (stratified per source),
    so the validation set contains every source and every fake type.
  - C and the decision threshold are chosen on that mixed validation set
    (step 5 of the plan).
  - Validation results are printed per source and per subfolder.
  - The final classifier is refit on all images and saved as a small .npz
    (weights, bias, threshold, CLIP model name). Scoring it needs only numpy:
        p_fake = sigmoid(features @ coef + intercept)

Usage:
    python train_clip_head.py --cache_dir /kaggle/working/clip_features \
        --out /kaggle/working/clip_head.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from loso_from_cache import best_threshold, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out", required=True, help="Where to save the classifier (.npz).")
    ap.add_argument("--exclude", nargs="*", default=[], help="Source names to leave out.")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model_name = None
    tr_X, tr_y, va_X, va_y, va_src, va_grp = [], [], [], [], [], []
    for f in sorted(Path(args.cache_dir).glob("*.npz")):
        if f.stem in args.exclude:
            continue
        z = np.load(f, allow_pickle=False)
        X, y, g = z["X"], z["y"], z["g"]
        if "model" in z.files:
            model_name = str(z["model"])
        val_mask = np.zeros(len(y), dtype=bool)
        for label in (0, 1):
            idx = np.where(y == label)[0].copy()
            rng.shuffle(idx)
            val_mask[idx[: int(round(len(idx) * args.val_frac))]] = True
        tr_X.append(X[~val_mask]); tr_y.append(y[~val_mask])
        va_X.append(X[val_mask]); va_y.append(y[val_mask])
        va_src += [f.stem] * int(val_mask.sum()); va_grp += g[val_mask].tolist()
        print(f"{f.stem:12s} train {int((~val_mask).sum()):6d}  val {int(val_mask.sum()):5d}")

    Xtr, ytr = np.concatenate(tr_X), np.concatenate(tr_y)
    Xva, yva = np.concatenate(va_X), np.concatenate(va_y)
    va_src, va_grp = np.array(va_src), np.array(va_grp)

    best = None
    for C in (0.1, 1.0, 10.0, 100.0):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Xtr, ytr)
        b = scores(yva, clf.predict_proba(Xva)[:, 1])["balanced_accuracy"]
        print(f"C={C:<6} val balanced {b:.3f}")
        if best is None or b > best[0]:
            best = (b, C, clf)
    _, C, clf = best
    pva = clf.predict_proba(Xva)[:, 1]
    thr = best_threshold(yva, pva)
    print(f"-> C={C}, threshold {thr:.2f}")

    report = {"C": C, "threshold": thr, "overall": scores(yva, pva, thr), "per_source": {}}
    print("\n=== Validation (held-out 15% of every source) ===")
    o = report["overall"]
    print(f"overall      AUC {o['auc']:.3f}  balanced {o['balanced_accuracy']:.3f}  "
          f"fake caught {o['fake_recall']:.3f}  real correct {o['real_specificity']:.3f}")
    for s in sorted(set(va_src.tolist())):
        m = va_src == s
        r = scores(yva[m], pva[m], thr)
        report["per_source"][s] = r
        print(f"{s:12s} AUC {r['auc']:.3f}  balanced {r['balanced_accuracy']:.3f}  "
              f"fake caught {r['fake_recall']:.3f}  real correct {r['real_specificity']:.3f}")
        for grp in sorted(set(va_grp[m].tolist())):
            mg = m & (va_grp == grp)
            lab = int(yva[mg][0])
            ok = float(((pva[mg] >= thr).astype(int) == lab).mean())
            print(f"    {grp:20s} ({'fake' if lab else 'real'}, n={int(mg.sum())}): correct {ok:.3f}")

    # Refit on everything with the chosen C; keep the validation threshold.
    final = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(
        np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva]))
    np.savez(args.out, coef=final.coef_[0].astype(np.float32), intercept=np.float32(final.intercept_[0]),
             threshold=np.float32(thr), clip_model=np.array(model_name or "openai/clip-vit-large-patch14"))
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved classifier to {args.out} (and validation report {Path(args.out).with_suffix('.json')})")


if __name__ == "__main__":
    main()
