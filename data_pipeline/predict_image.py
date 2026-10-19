"""
predict_image.py

Scores still images as real/fake with the trained image model.

PROVISIONAL: do not use in the app until the image model has passed the
external test (Fake-Vs-Real-Faces Hard). The preprocessing below must match
prepare_image_faces.py and the training run exactly, so keep the two in sync.

Preprocessing, in the same order as training:
  1. downscale the image so its longest side is at most 512 px
  2. MTCNN finds the largest confident face (select_best_face)
  3. crop, shrink to 112 px, then enlarge to 224 px (equal sharpness for all faces)
  4. JPEG round-trip at PIL's default quality (training faces were saved this way)
  5. ImageNet normalisation, then the model's P(fake)

If no usable face is found, the image is NOT scored as real. The result is None
so the caller can say "no face found" instead.

Usage (from data_pipeline/):
    python predict_image.py --checkpoint /path/to/image_best.pth --images a.jpg b.png
    python predict_image.py --checkpoint /path/to/image_best.pth --folder /path/to/images
"""

import argparse
import io
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms

from mtcnn_align import OUTPUT_SIZE, select_best_face
from prepare_image_faces import downscale

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FAKE = 1

eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def align_face(img: Image.Image, mtcnn, max_side: int = 512, min_face: int = 112,
               face_res: int = 112):
    """Same steps as prepare_image_faces.py. Returns a 224x224 PIL image, or None."""
    img = downscale(img, max_side)
    boxes, probs = mtcnn.detect(img)
    box = select_best_face(boxes, probs)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if (x2 - x1) < min_face:
        return None
    face = img.crop((x1, y1, x2, y2))
    face = face.resize((face_res, face_res), Image.LANCZOS).resize(OUTPUT_SIZE, Image.BICUBIC)
    return face


def jpeg_roundtrip(img: Image.Image) -> Image.Image:
    """Save and reload as JPEG at PIL's default quality, as the training faces were saved."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class ImageScorer:
    def __init__(self, checkpoint: str, device: torch.device = None):
        from facenet_pytorch import MTCNN

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.model = timm.create_model(ckpt.get("backbone", "xception"), pretrained=False, num_classes=2)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def score(self, source):
        """source: a file path or a PIL image. Returns P(fake) in [0, 1], or None if no face."""
        if isinstance(source, (str, Path)):
            img = Image.open(source).convert("RGB")
        else:
            img = source.convert("RGB")
        face = align_face(img, self.mtcnn)
        if face is None:
            return None
        face = jpeg_roundtrip(face)
        x = eval_tf(face).unsqueeze(0).to(self.device)
        logits = self.model(x)
        return float(torch.softmax(logits.float(), dim=1)[0, FAKE])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--folder", default=None)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="PROVISIONAL. Do not treat 0.5 as final until it is chosen on validation data.")
    args = ap.parse_args()

    paths = [Path(p) for p in args.images]
    if args.folder:
        paths += sorted(p for p in Path(args.folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        ap.error("give --images or --folder")

    scorer = ImageScorer(args.checkpoint)
    for p in paths:
        s = scorer.score(p)
        if s is None:
            print(f"{p}: NO FACE FOUND (not scored)")
        else:
            label = "FAKE" if s >= args.threshold else "REAL"
            print(f"{p}: P(fake)={s:.3f} -> {label}")


if __name__ == "__main__":
    main()

