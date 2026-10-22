"""
extract_clip_features.py

Extracts frozen CLIP image features for each face source and saves one .npz
file per source. Run this on Kaggle (GPU), download the output folder, then
run loso_from_cache.py on your own computer (CPU is fine there).

Each .npz holds:
  X  - L2-normalised CLIP image embeddings (float32)
  y  - labels (0 = real, 1 = fake)
  g  - name of the subfolder each image came from (e.g. fake_inpainting)

Usage (Kaggle, internet ON so the CLIP weights can download):
    python extract_clip_features.py --sources /kaggle/working/sources.json \
        --out_dir /kaggle/working/clip_features
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(folder: str) -> list:
    return sorted(str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


class ImageFolderDataset(Dataset):
    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True, help='JSON file: {"name": [[folder, label], ...]}')
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = json.loads(Path(args.sources).read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = CLIPImageProcessor.from_pretrained(args.model)
    model = CLIPVisionModelWithProjection.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    print(f"model {args.model} on {device}")

    for name, spec in sources.items():
        paths, labels, groups = [], [], []
        for folder, label in spec:
            imgs = list_images(folder)
            paths += imgs
            labels += [int(label)] * len(imgs)
            groups += [Path(folder).name] * len(imgs)
            print(f"  {name}: {Path(folder).name} ({'fake' if label else 'real'}) {len(imgs)} images")

        target = out_dir / f"{name}.npz"
        if target.exists():
            z = np.load(target, allow_pickle=False)
            if len(z["y"]) == len(paths):
                print(f"{name}: already done ({len(paths)} images), skipping")
                continue

        loader = DataLoader(ImageFolderDataset(paths, processor), batch_size=args.batch,
                            shuffle=False, num_workers=args.num_workers)
        feats = []
        for pixels in tqdm(loader, desc=name, unit="batch"):
            with torch.no_grad():
                emb = model(pixel_values=pixels.to(device, dtype=dtype)).image_embeds
            feats.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy())

        np.savez(target, X=np.concatenate(feats).astype(np.float32),
                 y=np.array(labels), g=np.array(groups), model=np.array(args.model))
        print(f"{name}: saved {len(paths)} features to {target}")

    print("DONE")


if __name__ == "__main__":
    main()
