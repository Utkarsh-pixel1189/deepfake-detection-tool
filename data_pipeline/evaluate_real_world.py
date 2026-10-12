"""
evaluate_real_world.py

Runs the FULL pipeline (frames -> MTCNN face crop -> Xception -> LSTM+RNN)
on a small folder of real-life videos and tells you WHY it gets them
right or wrong, not just whether it does.

For every video it records two scores side by side:
  - CNN-only : mean P(fake) over the video's frames, using the CNN's own head
  - Temporal : P(fake) from the LSTM+RNN fusion model (what predict.py returns)

Folder layout (label comes from the top-level folder name; any subfolders
under real/ or fake/ are treated as groups, e.g. fake/ai_generated,
fake/face_swap_app, so you can see which kinds of fakes are missed):

    real_world_test/
        real/            phone clips, interviews, normal footage ...
        fake/
            ai_generated/    fully generated video (Sora/Veo/Kling style)
            deepfake/        face swaps / lip-sync / reenactment
            ...

Usage (from data_pipeline/):
    python evaluate_real_world.py --videos_root real_world_test \\
        --xception_checkpoint xception_best.pth \\
        --temporal_checkpoint temporal_attention_cdf.pth \\
        --output_csv real_world_results.csv

How to read the summary:
  - AUC near 0.5  -> the scores carry no signal on these videos. This is a
    DOMAIN GAP (the model has never seen this kind of fake); moving the
    threshold cannot fix it - the fix is different training data / scope.
  - AUC high (0.85+) but detection/specificity poor at 0.5 -> the scores
    separate the classes but the cutoff is wrong for this domain; a
    cutoff chosen on a separate real-world set can help.
  - "no_face" videos are reported separately: the model cannot judge them.
"""

import argparse
import csv
import shutil
import tempfile
from pathlib import Path

import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def softmax_fake_prob(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp[..., 1] / exp.sum(axis=-1)


def collect_videos(videos_root: Path) -> list:
    """[(path, label, group)] - label 0 for real/, 1 for fake/; group = subfolder path below the label folder."""
    items = []
    for label_name, label in (("real", 0), ("fake", 1)):
        base = videos_root / label_name
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                rel = p.parent.relative_to(base)
                items.append((p, label, str(rel) if str(rel) != "." else "(ungrouped)"))
    return items


def summarize(rows: list) -> dict:
    """rows: dicts with label, group, status, cnn_fake_prob, temporal_fake_prob. Pure numpy/sklearn."""
    from sklearn.metrics import roc_auc_score

    ok = [r for r in rows if r["status"] == "ok"]
    out = {"n_total": len(rows), "n_scored": len(ok), "n_no_face": len(rows) - len(ok)}
    for key, name in (("cnn_fake_prob", "cnn_only"), ("temporal_fake_prob", "temporal")):
        labels = np.array([r["label"] for r in ok])
        probs = np.array([r[key] for r in ok], dtype=float)
        m = {}
        if (labels == 0).any():
            m["specificity_at_0.5"] = float(((probs < 0.5) & (labels == 0)).sum() / (labels == 0).sum())
            m["mean_p_fake_on_real"] = float(probs[labels == 0].mean())
        if (labels == 1).any():
            m["detection_at_0.5"] = float(((probs >= 0.5) & (labels == 1)).sum() / (labels == 1).sum())
            m["mean_p_fake_on_fake"] = float(probs[labels == 1].mean())
            m["by_group"] = {}
            for g in sorted({r["group"] for r in ok if r["label"] == 1}):
                gp = np.array([r[key] for r in ok if r["label"] == 1 and r["group"] == g], dtype=float)
                m["by_group"][g] = {"n": int(len(gp)), "detected_at_0.5": float((gp >= 0.5).mean()),
                                    "mean_p_fake": float(gp.mean())}
        if len(set(labels.tolist())) == 2:
            m["auc_roc"] = float(roc_auc_score(labels, probs))
        out[name] = m
    return out


def print_summary(s: dict):
    print("\n" + "=" * 64)
    print(f"Videos: {s['n_total']}  scored: {s['n_scored']}  no face detected: {s['n_no_face']}")
    for name in ("cnn_only", "temporal"):
        m = s[name]
        print(f"\n--- {name} ---")
        for k in ("specificity_at_0.5", "detection_at_0.5", "mean_p_fake_on_real", "mean_p_fake_on_fake", "auc_roc"):
            if k in m:
                print(f"  {k:22s}: {m[k]:.3f}")
        for g, v in m.get("by_group", {}).items():
            print(f"  fake group {g:20s} n={v['n']:3d}  caught={v['detected_at_0.5']:.2f}  mean P(fake)={v['mean_p_fake']:.2f}")
        auc = m.get("auc_roc")
        if auc is not None:
            if auc < 0.65:
                print("  -> AUC near chance: no usable signal on these videos (domain gap, not a threshold problem).")
            elif auc >= 0.85:
                print("  -> Scores separate real from fake; a better cutoff for this domain could help.")
            else:
                print("  -> Partial signal: weak separation, expect errors in both directions.")
    print("=" * 64)


def run(args):
    import timm
    import torch
    from facenet_pytorch import MTCNN
    from PIL import Image

    from dataset import get_eval_transform
    from extract_frames import extract_frames_from_video
    from mtcnn_align import align_face_in_image
    from predict import load_temporal_model, load_xception_feature_extractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xception = load_xception_feature_extractor(args.xception_checkpoint, device)
    state = torch.load(args.xception_checkpoint, map_location="cpu")
    fc_w, fc_b = state["fc.weight"].numpy(), state["fc.bias"].numpy()
    temporal = load_temporal_model(args.temporal_checkpoint, fc_w.shape[1], device)
    mtcnn = MTCNN(keep_all=True, device=device)
    transform = get_eval_transform()

    videos = collect_videos(Path(args.videos_root))
    if not videos:
        raise SystemExit(f"No videos found under {args.videos_root}/real and /fake")
    print(f"Found {len(videos)} videos ({sum(1 for v in videos if v[1] == 0)} real, "
          f"{sum(1 for v in videos if v[1] == 1)} fake)")

    rows = []
    for i, (path, label, group) in enumerate(videos, 1):
        row = {"video": str(path), "label": label, "group": group, "status": "ok", "frames_used": 0,
               "frames_discarded": 0, "cnn_fake_prob": "", "cnn_min_frame": "", "cnn_max_frame": "",
               "temporal_fake_prob": ""}
        tmp = Path(tempfile.mkdtemp(prefix="rw_eval_"))
        try:
            frames_root = tmp / "frames"
            frames_root.mkdir()
            extract_frames_from_video(path, frames_root, args.fps)
            frame_dir = frames_root / path.stem
            frame_paths = sorted(frame_dir.iterdir()) if frame_dir.exists() else []

            faces_dir = tmp / "faces"
            faces_dir.mkdir()
            faces = []
            for fp in frame_paths:
                face = align_face_in_image(fp, mtcnn)
                if face is None:
                    row["frames_discarded"] += 1
                    continue
                # Same JPEG save + reload step as training / predict.py
                out = faces_dir / fp.name
                face.save(out)
                faces.append(Image.open(out).convert("RGB"))

            row["frames_used"] = len(faces)
            if not faces:
                row["status"] = "no_face"
            else:
                with torch.no_grad():
                    batch = torch.stack([transform(f) for f in faces]).to(device)
                    emb = xception(batch).cpu().numpy()                       # (T, 2048)
                    frame_p = softmax_fake_prob(emb @ fc_w.T + fc_b)          # CNN head per frame
                    seq = torch.from_numpy(emb).float().unsqueeze(0).to(device)
                    logits = temporal(seq, torch.tensor([emb.shape[0]]))
                    t_p = torch.softmax(logits, dim=1)[0, 1].item()
                row["cnn_fake_prob"] = float(frame_p.mean())
                row["cnn_min_frame"] = float(frame_p.min())
                row["cnn_max_frame"] = float(frame_p.max())
                row["temporal_fake_prob"] = float(t_p)
        except Exception as e:  # keep going - one bad file must not stop the batch
            row["status"] = f"error: {e}"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        rows.append(row)
        tag = "REAL" if label == 0 else "FAKE"
        if row["status"] == "ok":
            print(f"[{i}/{len(videos)}] {tag:4s} {group:16s} CNN={row['cnn_fake_prob']:.2f} "
                  f"temporal={row['temporal_fake_prob']:.2f}  frames={row['frames_used']}  {path.name}")
        else:
            print(f"[{i}/{len(videos)}] {tag:4s} {group:16s} {row['status']}  {path.name}")

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-video results saved to {args.output_csv}")

    scored = [dict(r, cnn_fake_prob=float(r["cnn_fake_prob"]), temporal_fake_prob=float(r["temporal_fake_prob"]))
              for r in rows if r["status"] == "ok"]
    unscored = [r for r in rows if r["status"] != "ok"]
    print_summary(summarize(scored + unscored))


def main():
    parser = argparse.ArgumentParser(description="Evaluate the full pipeline on a folder of real-life videos.")
    parser.add_argument("--videos_root", required=True, help="Folder containing real/ and fake/ subfolders of videos.")
    parser.add_argument("--xception_checkpoint", required=True)
    parser.add_argument("--temporal_checkpoint", required=True)
    parser.add_argument("--fps", type=float, default=1.0, help="Frame sampling rate (must match training; default 1.0).")
    parser.add_argument("--output_csv", default="real_world_results.csv")
    run(parser.parse_args())


if __name__ == "__main__":
    main()