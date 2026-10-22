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


class TorchMLP:
    """768 -> 512 (ReLU) -> 1 classifier trained with PyTorch (GPU if available).

    Class-balanced loss, AdamW, early stopping on an internal 10% split.
    After fit(), coefs_/intercepts_ hold the weights in the same layout as
    scikit-learn's MLPClassifier, so the saved head is scored with plain numpy.
    """

    def __init__(self, weight_decay=1e-3, seed=42, hidden=512, lr=1e-3, batch=512, epochs=100, patience=6):
        self.wd, self.seed, self.hidden = weight_decay, seed, hidden
        self.lr, self.batch, self.epochs, self.patience = lr, batch, epochs, patience

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        idx = np.random.default_rng(self.seed).permutation(len(y))
        n_val = max(1, len(y) // 10)
        va, tr = idx[:n_val], idx[n_val:]

        Xt = torch.tensor(X, dtype=torch.float32, device=dev)
        yt = torch.tensor(y, dtype=torch.float32, device=dev)
        tr_t, va_t = torch.tensor(tr, device=dev), torch.tensor(va, device=dev)
        pos, neg = int((y[tr] == 1).sum()), int((y[tr] == 0).sum())
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1), device=dev))

        model = nn.Sequential(nn.Linear(X.shape[1], self.hidden), nn.ReLU(), nn.Linear(self.hidden, 1)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        best_loss, best_state, bad = float("inf"), None, 0
        for _ in range(self.epochs):
            model.train()
            perm = tr_t[torch.randperm(len(tr_t), device=dev)]
            for i in range(0, len(perm), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                loss_fn(model(Xt[b]).squeeze(1), yt[b]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(Xt[va_t]).squeeze(1), yt[va_t]).item()
            if vl < best_loss - 1e-4:
                best_loss, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        model.load_state_dict(best_state)
        model = model.cpu()
        self.coefs_ = [model[0].weight.detach().numpy().T.copy(), model[2].weight.detach().numpy().T.copy()]
        self.intercepts_ = [model[0].bias.detach().numpy().copy(), model[2].bias.detach().numpy().copy()]
        return self

    def predict_proba(self, X):
        h = np.maximum(X @ self.coefs_[0] + self.intercepts_[0], 0.0)
        p = 1.0 / (1.0 + np.exp(-(h @ self.coefs_[1] + self.intercepts_[1]).ravel()))
        return np.stack([1.0 - p, p], axis=1)
from sklearn.metrics import roc_auc_score

from loso_from_cache import best_threshold, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out", required=True, help="Where to save the classifier (.npz).")
    ap.add_argument("--exclude", nargs="*", default=[], help="Source names to leave out.")
    ap.add_argument("--head", choices=["linear", "mlp"], default="linear",
                    help="linear = logistic regression; mlp = one hidden layer of 512 units.")
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

    def make(param):
        if args.head == "mlp":
            # Small non-linear head on the GPU: one hidden layer, weight decay = param.
            return TorchMLP(weight_decay=param, seed=args.seed)
        return LogisticRegression(C=param, class_weight="balanced", max_iter=5000)

    def fit(model, X, y):
        return model.fit(X, y)

    grid = (1e-4, 1e-3, 1e-2) if args.head == "mlp" else (0.1, 1.0, 10.0, 100.0)
    pname = "alpha" if args.head == "mlp" else "C"
    best = None
    for param in grid:
        clf = fit(make(param), Xtr, ytr)
        b = scores(yva, clf.predict_proba(Xva)[:, 1])["balanced_accuracy"]
        print(f"{pname}={param:<7} val balanced {b:.3f}", flush=True)
        if best is None or b > best[0]:
            best = (b, param, clf)
    _, C, clf = best
    pva = clf.predict_proba(Xva)[:, 1]
    thr = best_threshold(yva, pva)
    print(f"-> {args.head} head, {pname}={C}, threshold {thr:.2f}")

    report = {"head": args.head, pname: C, "threshold": thr, "overall": scores(yva, pva, thr), "per_source": {}}
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

    # Refit on everything with the chosen setting; keep the validation threshold.
    final = fit(make(C), np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva]))
    common = dict(threshold=np.float32(thr), clip_model=np.array(model_name or "openai/clip-vit-large-patch14"),
                  kind=np.array(args.head))
    if args.head == "mlp":
        np.savez(args.out, W0=final.coefs_[0].astype(np.float32), b0=final.intercepts_[0].astype(np.float32),
                 W1=final.coefs_[1].astype(np.float32), b1=final.intercepts_[1].astype(np.float32), **common)
    else:
        np.savez(args.out, coef=final.coef_[0].astype(np.float32), intercept=np.float32(final.intercept_[0]),
                 **common)
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved classifier to {args.out} (and validation report {Path(args.out).with_suffix('.json')})")


if __name__ == "__main__":
    main()