"""
final_evaluation.py

One script that produces every number the project report needs, from
CACHED EMBEDDINGS (so it runs in minutes and needs no face images):

  1. In-distribution test set (Celeb-DF-v3 test split, same video-level
     split the CNN and temporal model never trained on):
        - CNN-only baseline   (CNN's own fc head, mean fake-prob over a video's frames)
        - CNN + LSTM + RNN    (the temporal fusion model)
     with a per-category breakdown (which manipulation methods are hard).
  2. Held-out manipulation methods never used in ANY training stage
     (e.g. LivePortrait / GHOST / SadTalker, FaceShifter): each is all-fake,
     so it is paired with the real test videos for full metrics, and its
     own detection rate is reported separately (the direct generalization number).
  3. Optional legacy FF++ test split (embeddings from the NEW CNN), to
     check the new model did not forget what the old one could do.

Embeddings for held-out sets come from extract_embeddings_holdout.py
(--holdout_root <folder of aligned-face video folders>).

Usage:
    python final_evaluation.py \\
        --embeddings_root embeddings_cdf \\
        --xception_checkpoint checkpoints_cdf/xception_best.pth \\
        --temporal_checkpoint checkpoints_cdf/temporal_attention_cdf.pth \\
        --holdout LivePortrait=embeddings_holdout/LivePortrait \\
        --holdout GHOST=embeddings_holdout/GHOST \\
        --holdout SadTalker=embeddings_holdout/SadTalker \\
        --holdout FaceShifter=embeddings_holdout/faceshifter \\
        --legacy_embeddings_root embeddings_ffpp_newcnn \\
        --output_json final_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


# ----------------------------------------------------------------------------
# Pure numpy / sklearn helpers (no torch needed)
# ----------------------------------------------------------------------------

def softmax_fake_prob(logits: np.ndarray) -> np.ndarray:
    """logits: (..., 2) -> P(fake) with a numerically stable softmax."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp[..., 1] / exp.sum(axis=-1)


def cnn_only_fake_probs(samples: list, fc_weight: np.ndarray, fc_bias: np.ndarray) -> np.ndarray:
    """
    CNN-only baseline at the VIDEO level: apply the CNN's own classifier
    head to each cached frame embedding, then average the frames'
    fake-probabilities. (Cached embeddings are the pooled features right
    before that head, so this reproduces the CNN's frame predictions.)
    """
    probs = []
    for path, _ in samples:
        emb = np.load(path)                       # (num_frames, 2048)
        logits = emb @ fc_weight.T + fc_bias      # (num_frames, 2)
        probs.append(softmax_fake_prob(logits).mean())
    return np.array(probs)


def compute_metrics(labels, fake_probs, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels)
    preds = (np.asarray(fake_probs) >= threshold).astype(int)
    m = {"n": int(len(labels)), "accuracy": float(accuracy_score(labels, preds))}
    if len(set(labels.tolist())) == 2:
        m["precision"] = float(precision_score(labels, preds))
        m["recall"] = float(recall_score(labels, preds))
        m["specificity"] = float(((preds == 0) & (labels == 0)).sum() / max((labels == 0).sum(), 1))
        m["balanced_accuracy"] = (m["recall"] + m["specificity"]) / 2
        m["f1"] = float(f1_score(labels, preds))
        m["auc_roc"] = float(roc_auc_score(labels, fake_probs))
    elif labels[0] == 1:
        m["detection_rate"] = m["accuracy"]       # all-fake set: fraction caught
    else:
        m["specificity"] = m["accuracy"]          # all-real set: fraction passed as real
    return m


def best_threshold(labels, fake_probs) -> float:
    """
    Threshold that maximizes BALANCED accuracy (mean of fake-recall and
    real-specificity) on the given set. Plain accuracy is misleading here
    because ~90% of the videos are fake. Ties go to the value closest to 0.5.
    Fit this on VALIDATION data only, then apply it to test/held-out sets.
    """
    labels = np.asarray(labels)
    probs = np.asarray(fake_probs)
    best, best_score = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 97):
        preds = (probs >= t).astype(int)
        rec = ((preds == 1) & (labels == 1)).sum() / max((labels == 1).sum(), 1)
        spec = ((preds == 0) & (labels == 0)).sum() / max((labels == 0).sum(), 1)
        score = (rec + spec) / 2
        if score > best_score + 1e-9 or (abs(score - best_score) <= 1e-9 and abs(t - 0.5) < abs(best - 0.5)):
            best, best_score = float(t), score
    return best


def per_category_breakdown(samples: list, fake_probs, root: Path, threshold: float = 0.5) -> dict:
    """Accuracy per category folder (e.g. fake/FaceSwap/SimSwap), keyed by path relative to root."""
    buckets = {}
    for (path, label), p in zip(samples, fake_probs):
        try:
            key = str(Path(path).parent.relative_to(root))
        except ValueError:
            key = str(Path(path).parent)
        buckets.setdefault(key, []).append(int((p >= threshold) == bool(label)))
    return {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in sorted(buckets.items())}


def load_fake_only_dir(embeddings_dir: str) -> list:
    """Every non-empty .npy in the folder, labeled fake (1)."""
    samples = []
    for f in sorted(Path(embeddings_dir).glob("*.npy")):
        arr = np.load(f, mmap_mode="r")
        if arr.ndim == 2 and arr.shape[0] > 0:
            samples.append((f, 1))
    return samples


def fmt(m: dict) -> str:
    parts = [f"n={m['n']}", f"acc={m['accuracy']:.4f}"]
    for key in ("precision", "recall", "specificity", "balanced_accuracy", "f1", "auc_roc", "detection_rate"):
        if key in m:
            parts.append(f"{key}={m[key]:.4f}")
    return "  ".join(parts)


# ----------------------------------------------------------------------------
# Torch parts
# ----------------------------------------------------------------------------

def load_cnn_head(xception_checkpoint: str):
    import torch
    state = torch.load(xception_checkpoint, map_location="cpu")
    return state["fc.weight"].numpy(), state["fc.bias"].numpy()


def load_temporal_model(checkpoint: str, embedding_dim: int, use_attention: bool, device):
    import torch
    from temporal_model import TemporalFusionAttentionModel, TemporalFusionModel
    cls = TemporalFusionAttentionModel if use_attention else TemporalFusionModel
    model = cls(embedding_dim=embedding_dim, lstm_hidden_size=256, lstm_num_layers=2,
                rnn_hidden_size=256, rnn_num_layers=1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def temporal_fake_probs(samples: list, model, device, batch_size: int = 16) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader
    from sequence_dataset import VideoSequenceDataset, collate_sequences

    loader = DataLoader(VideoSequenceDataset(samples), batch_size=batch_size, shuffle=False,
                        collate_fn=collate_sequences, num_workers=2)
    probs = []
    with torch.no_grad():
        for sequences, seq_lens, _ in loader:
            out = model(sequences.to(device), seq_lens)
            probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy().tolist())
    return np.array(probs)


# ----------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------

def evaluate_set(name: str, samples: list, cnn_fn, temporal_fn, results: dict, root: Path = None, thr: dict = None):
    labels = [l for _, l in samples]
    cnn_p, tmp_p = cnn_fn(samples), temporal_fn(samples)
    entry = {"cnn_only": compute_metrics(labels, cnn_p), "cnn_lstm_rnn": compute_metrics(labels, tmp_p)}
    print(f"\n=== {name} ===")
    print(f"  CNN-only                 : {fmt(entry['cnn_only'])}")
    print(f"  CNN+LSTM+RNN             : {fmt(entry['cnn_lstm_rnn'])}")
    if thr:
        entry["cnn_only_calibrated"] = compute_metrics(labels, cnn_p, thr["cnn"])
        entry["cnn_lstm_rnn_calibrated"] = compute_metrics(labels, tmp_p, thr["tmp"])
        print(f"  CNN-only @thr={thr['cnn']:.2f}       : {fmt(entry['cnn_only_calibrated'])}")
        print(f"  CNN+LSTM+RNN @thr={thr['tmp']:.2f}   : {fmt(entry['cnn_lstm_rnn_calibrated'])}")
    if root is not None:
        cat_cnn = per_category_breakdown(samples, cnn_p, root)
        cat_tmp = per_category_breakdown(samples, tmp_p, root)
        entry["per_category"] = {k: {"n": cat_cnn[k]["n"], "cnn_only_acc": cat_cnn[k]["accuracy"],
                                     "cnn_lstm_rnn_acc": cat_tmp[k]["accuracy"]} for k in cat_cnn}
        print(f"  {'category':36s} {'n':>4s} {'CNN-only':>9s} {'CNN+LSTM+RNN':>13s}")
        for k, v in entry["per_category"].items():
            print(f"  {k:36s} {v['n']:4d} {v['cnn_only_acc']:9.3f} {v['cnn_lstm_rnn_acc']:13.3f}")
    results[name] = entry
    return entry


def run(args):
    import torch
    from sequence_dataset import build_sequence_splits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fc_w, fc_b = load_cnn_head(args.xception_checkpoint)

    root = Path(args.embeddings_root)
    _, val_samples, test_samples = build_sequence_splits(args.embeddings_root)
    embedding_dim = np.load(test_samples[0][0], mmap_mode="r").shape[1]
    model = load_temporal_model(args.temporal_checkpoint, embedding_dim, not args.no_attention, device)

    cnn_fn = lambda s: cnn_only_fake_probs(s, fc_w, fc_b)
    tmp_fn = lambda s: temporal_fake_probs(s, model, device, args.batch_size)

    # Decision thresholds fitted on VALIDATION videos only (never test / held-out).
    val_labels = [l for _, l in val_samples]
    thr = {"cnn": best_threshold(val_labels, cnn_fn(val_samples)),
           "tmp": best_threshold(val_labels, tmp_fn(val_samples))}
    print(f"\nCalibrated thresholds (max balanced accuracy on val): "
          f"CNN-only={thr['cnn']:.2f}  CNN+LSTM+RNN={thr['tmp']:.2f}")

    results = {"calibrated_thresholds": thr}
    evaluate_set("IN-DISTRIBUTION TEST (Celeb-DF-v3 test split)", test_samples, cnn_fn, tmp_fn, results, root, thr)

    real_test = [(p, l) for p, l in test_samples if l == 0]
    all_holdout = []
    for spec in args.holdout:
        name, path = spec.split("=", 1)
        fakes = load_fake_only_dir(path)
        if not fakes:
            print(f"\nWarning: no usable embeddings in {path}, skipping {name}")
            continue
        all_holdout.extend(fakes)
        evaluate_set(f"HELD-OUT {name} (alone, all fake)", fakes, cnn_fn, tmp_fn, results, thr=thr)
        evaluate_set(f"HELD-OUT {name} + real test videos", real_test + fakes, cnn_fn, tmp_fn, results, thr=thr)

    if len(args.holdout) > 1 and all_holdout:
        evaluate_set("ALL HELD-OUT METHODS + real test videos", real_test + all_holdout, cnn_fn, tmp_fn, results, thr=thr)

    if args.legacy_embeddings_root:
        _, _, legacy_test = build_sequence_splits(args.legacy_embeddings_root)
        evaluate_set("LEGACY FF++ TEST SPLIT (new CNN embeddings)", legacy_test, cnn_fn, tmp_fn, results,
                     Path(args.legacy_embeddings_root), thr)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"\nSaved all results to {args.output_json}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Full evaluation of CNN-only vs CNN+LSTM+RNN on cached embeddings.")
    parser.add_argument("--embeddings_root", required=True, help="Celeb-DF-v3 embeddings (extract_embeddings.py output).")
    parser.add_argument("--xception_checkpoint", required=True, help="CNN checkpoint (for the CNN-only baseline head).")
    parser.add_argument("--temporal_checkpoint", required=True)
    parser.add_argument("--no_attention", action="store_true",
                        help="Only if the temporal checkpoint was trained WITHOUT --use_attention.")
    parser.add_argument("--holdout", action="append", default=[],
                        help="NAME=EMBEDDINGS_DIR (repeatable). All videos in the folder are fake.")
    parser.add_argument("--legacy_embeddings_root", default=None,
                        help="Optional FF++-layout embeddings extracted with the NEW CNN.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_json", default="final_results.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()