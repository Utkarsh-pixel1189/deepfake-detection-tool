"""
predict_image_clip.py

Scores still images as real or AI-generated/deepfake using:
  face alignment (same as training) -> frozen CLIP ViT-L/14 -> logistic
  regression head trained by train_clip_head.py.

Images with no usable face are reported as "NO FACE FOUND", never as real.

Usage:
  Score some images:
    python predict_image_clip.py --head clip_head.npz --images a.jpg b.png
  Score a whole folder:
    python predict_image_clip.py --head clip_head.npz --folder photos/
  Real-world test (Akash, step 4): a folder with real/ and fake/ subfolders.
  Writes a per-image CSV and prints accuracy per subfolder:
    python predict_image_clip.py --head clip_head.npz --eval_dir real_world_test/ --csv results.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from predict_image import jpeg_roundtrip

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_FACE = 112     # same minimum face width as training
SMALL_FACE = 64    # smaller faces are scored only as a low-confidence fallback
MAX_FACES = 5      # group photos: score up to the 5 largest faces, report the highest P(fake)
DET_PROB = 0.90    # MTCNN confidence needed to count as a face


class ClipFaceScorer:
    def __init__(self, head_path: str, device=None):
        from facenet_pytorch import MTCNN
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        head = np.load(head_path, allow_pickle=False)
        self.kind = str(head["kind"]) if "kind" in head.files else "linear"
        if self.kind == "mlp":
            self.W0, self.b0 = head["W0"], head["b0"]
            self.W1, self.b1 = head["W1"], head["b1"]
        else:
            self.coef = head["coef"].astype(np.float32)
            self.intercept = float(head["intercept"])
        self.threshold = float(head["threshold"])
        model_name = str(head["clip_model"])

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.clip = CLIPVisionModelWithProjection.from_pretrained(model_name, torch_dtype=self.dtype)
        self.clip.to(self.device).eval()

    def find_faces(self, img):
        """All usable faces in the image, as 224x224 crops (largest first, at most MAX_FACES).

        Same preprocessing as training (shrink, MTCNN, crop, 112 -> 224). Tries
        512 px first (training size), then 2048 px for big photos and group shots.
        If only small faces (SMALL_FACE..112 px) exist, they are used too, but the
        result is flagged low_confidence: they had to be enlarged, so they are
        softer than anything the model saw in training.
        Returns (crops, widths, low_confidence)."""
        from mtcnn_align import OUTPUT_SIZE
        from prepare_image_faces import downscale

        for max_side in (512, 2048):
            small = downscale(img, max_side)
            boxes, probs = self.mtcnn.detect(small)
            if boxes is None:
                continue
            cands = []
            for b, p in zip(boxes, probs):
                if p is None or p < DET_PROB:
                    continue
                x1, y1 = max(0.0, float(b[0])), max(0.0, float(b[1]))
                x2, y2 = min(float(small.width), float(b[2])), min(float(small.height), float(b[3]))
                cands.append((x2 - x1, (x1, y1, x2, y2)))
            big = [c for c in cands if c[0] >= MIN_FACE]
            if big:
                chosen, low = big, False
            elif max_side == 2048 and any(c[0] >= SMALL_FACE for c in cands):
                chosen, low = [c for c in cands if c[0] >= SMALL_FACE], True
            else:
                continue
            chosen = sorted(chosen, key=lambda c: -c[0])[:MAX_FACES]
            crops = [small.crop(box).resize((MIN_FACE, MIN_FACE), Image.LANCZOS).resize(OUTPUT_SIZE, Image.BICUBIC)
                     for _, box in chosen]
            return crops, [w for w, _ in chosen], low
        return [], [], False

    @torch.no_grad()
    def face_probs(self, faces):
        """P(fake) for a list of 224x224 face crops."""
        faces = [jpeg_roundtrip(f) for f in faces]   # training faces were saved as JPEG
        pixels = self.processor(images=faces, return_tensors="pt")["pixel_values"].to(self.device, dtype=self.dtype)
        emb = self.clip(pixel_values=pixels).image_embeds
        emb = torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy()
        return [self.head_prob(e) for e in emb]

    def score_details(self, source):
        """source: path or PIL image. Returns a dict:
        p_fake (max over faces, or None if no usable face), faces (width + p_fake each),
        low_confidence (True if only small faces were found)."""
        img = Image.open(source).convert("RGB") if isinstance(source, (str, Path)) else source.convert("RGB")
        crops, widths, low = self.find_faces(img)
        if not crops:
            return {"p_fake": None, "faces": [], "low_confidence": False}
        probs = self.face_probs(crops)
        return {"p_fake": max(probs), "low_confidence": low,
                "faces": [{"width_px": round(w), "p_fake": p} for w, p in zip(widths, probs)]}

    def score(self, source):
        """P(fake) in [0, 1] (highest over the faces in the image), or None if no usable face."""
        return self.score_details(source)["p_fake"]

    def head_prob(self, emb: np.ndarray) -> float:
        """P(fake) from one L2-normalised CLIP embedding."""
        if self.kind == "mlp":
            h = np.maximum(emb @ self.W0 + self.b0, 0.0)
            z = float((h @ self.W1 + self.b1).ravel()[0])
        else:
            z = float(emb @ self.coef + self.intercept)
        return float(1.0 / (1.0 + np.exp(-z)))


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="clip_head.npz from train_clip_head.py")
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--folder", default=None)
    ap.add_argument("--eval_dir", default=None, help="Folder with real/ and fake/ subfolders.")
    ap.add_argument("--csv", default=None, help="Write per-image results here.")
    ap.add_argument("--threshold", type=float, default=None, help="Override the saved threshold.")
    args = ap.parse_args()

    items = [(Path(p), "") for p in args.images]
    if args.folder:
        items += [(p, "") for p in list_images(args.folder)]
    if args.eval_dir:
        for label in ("real", "fake"):
            d = Path(args.eval_dir) / label
            if d.exists():
                items += [(p, label) for p in list_images(d)]
    if not items:
        ap.error("give --images, --folder or --eval_dir")

    scorer = ClipFaceScorer(args.head)
    thr = args.threshold if args.threshold is not None else scorer.threshold
    print(f"threshold {thr:.2f} (P(fake) at or above this = FAKE)")

    rows = []
    for p, true in items:
        d = scorer.score_details(p)
        s = d["p_fake"]
        verdict = "NO FACE" if s is None else ("FAKE" if s >= thr else "REAL")
        n_faces = len(d["faces"])
        rows.append({"file": str(p), "true_label": true, "p_fake": "" if s is None else f"{s:.4f}",
                     "verdict": verdict, "faces": n_faces, "low_confidence": d["low_confidence"]})
        extra = f"  ({n_faces} faces{', small faces - low confidence' if d['low_confidence'] else ''})" \
            if n_faces > 1 or d["low_confidence"] else ""
        print(f"{p.name}: " + ("NO FACE FOUND (not scored)" if s is None else f"P(fake)={s:.3f} -> {verdict}")
              + extra + (f"  [true: {true}]" if true else ""))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "true_label", "p_fake", "verdict", "faces", "low_confidence"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nsaved {args.csv}")

    labelled = [r for r in rows if r["true_label"]]
    if labelled:
        print("\n=== Accuracy (images with a face) ===")
        for label in ("real", "fake"):
            group = [r for r in labelled if r["true_label"] == label]
            scored = [r for r in group if r["verdict"] != "NO FACE"]
            if group:
                ok = sum(r["verdict"].lower() == label for r in scored)
                acc = ok / len(scored) if scored else float("nan")
                print(f"  {label}: {ok}/{len(scored)} correct ({acc:.1%}); no face found in {len(group) - len(scored)}")


if __name__ == "__main__":
    main()