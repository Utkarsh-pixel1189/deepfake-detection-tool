"""
loso_fake_type.py

Do the fake types inside one source teach the model to catch each other?

For each fake type in --source (default: dff, i.e. fake_inpainting,
fake_text2img, fake_insight), two runs are compared on the SAME test images:

  UNSEEN: train on every other source + this source's other fake types
          (none of the held-out type), then test on the held-out type.
  SEEN:   same, plus half of the held-out type's images in training.
          This is the reference: how well that type is caught once seen.

Test set in both runs = the other half of the held-out type + a held-out half
of this source's real images (never used in training), so no image or real
photo appears in both training and test.

If UNSEEN is close to SEEN, the fake types transfer to each other, and adding
any new diffusion source should help. If UNSEEN is far below SEEN, each
generator has to be seen in training.

Usage (Kaggle or anywhere with numpy + scikit-learn):
    python data_pipeline/loso_fake_type.py --cache_dir /kaggle/working/clip_features
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from loso_from_cache import best_threshold, scores


def fit_and_test(X, y, Xt, yt):
    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    best = None
    for C in (0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X_fit, y_fit)
        b = scores(y_val, clf.predict_proba(X_val)[:, 1])["balanced_accuracy"]
        if best is None or b > best[0]:
            best = (b, C, clf)
    _, C, clf = best
    thr = best_threshold(y_val, clf.predict_proba(X_val)[:, 1])
    pt = clf.predict_proba(Xt)[:, 1]
    return {"C": C, "val_threshold": thr, "at_0.5": scores(yt, pt, 0.5), "at_val_threshold": scores(yt, pt, thr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--source", default="dff", help="Source whose fake types are held out one at a time.")
    ap.add_argument("--out_json", default="fake_type_results.json")
    args = ap.parse_args()

    data = {}
    for f in sorted(Path(args.cache_dir).glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        data[f.stem] = (z["X"], z["y"], z["g"])
    if args.source not in data:
        raise SystemExit(f"{args.source}.npz not found in {args.cache_dir}")

    Xs, ys, gs = data[args.source]
    others = [n for n in data if n != args.source]
    X_other = np.concatenate([data[n][0] for n in others])
    y_other = np.concatenate([data[n][1] for n in others])

    rng = np.random.default_rng(42)
    real_idx = np.where(ys == 0)[0].copy()
    rng.shuffle(real_idx)
    real_train, real_test = real_idx[: len(real_idx) // 2], real_idx[len(real_idx) // 2:]

    fake_types = sorted(set(gs[ys == 1].tolist()))
    print(f"source {args.source}: real {len(real_idx)} (train {len(real_train)}, test {len(real_test)}), "
          f"fake types {fake_types}; other sources in training: {', '.join(others)}")

    results, summary = {}, []
    for held in fake_types:
        held_idx = np.where(gs == held)[0].copy()
        rng.shuffle(held_idx)
        held_train, held_test = held_idx[: len(held_idx) // 2], held_idx[len(held_idx) // 2:]
        other_fake = np.where((ys == 1) & (gs != held))[0]

        te = np.concatenate([real_test, held_test])
        Xt, yt = Xs[te], ys[te]

        r = {}
        for mode, extra in (("unseen", np.array([], dtype=int)), ("seen", held_train)):
            tr = np.concatenate([real_train, other_fake, extra])
            X = np.concatenate([X_other, Xs[tr]])
            y = np.concatenate([y_other, ys[tr]])
            r[mode] = fit_and_test(X, y, Xt, yt)
        results[held] = r

        print(f"\nHELD-OUT FAKE TYPE: {held}  (test: {len(held_test)} {held} + {len(real_test)} real)")
        for mode in ("unseen", "seen"):
            m = r[mode]
            a, b = m["at_0.5"], m["at_val_threshold"]
            print(f"  {mode:6s} AUC {a['auc']:.3f} | @{m['val_threshold']:.2f} fake caught {b['fake_recall']:.3f}  "
                  f"real correct {b['real_specificity']:.3f}  balanced {b['balanced_accuracy']:.3f}")
        summary.append((held, r["unseen"]["at_0.5"]["auc"], r["seen"]["at_0.5"]["auc"],
                        r["unseen"]["at_val_threshold"]["fake_recall"], r["seen"]["at_val_threshold"]["fake_recall"]))

    print("\n=== Summary: is each fake type caught without seeing it? ===")
    print(f"{'fake type':18s} {'AUC unseen':>10s} {'AUC seen':>9s} {'caught unseen':>14s} {'caught seen':>12s}")
    for held, au, as_, cu, cs in summary:
        print(f"{held:18s} {au:10.3f} {as_:9.3f} {cu:14.3f} {cs:12.3f}")

    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
