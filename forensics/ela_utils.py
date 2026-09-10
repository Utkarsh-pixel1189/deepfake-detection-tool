"""
ela_utils.py

Error Level Analysis (ELA): flags possible tampering by recompressing an
image at a known JPEG quality and measuring the difference from the
original. Regions that were edited after the last save tend to show a
different error level than untouched regions.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageChops


def run_ela(image_path, quality: int = 90, output_dir=None) -> dict:
    """
    Run Error Level Analysis on an image.

    Args:
        image_path: path to the input image.
        quality: JPEG quality to recompress at (default 90).
        output_dir: where to save the ELA difference image. Defaults to
                     the same folder as the input image.

    Returns:
        Dictionary with an anomaly score (higher = more suspicious) and
        the path to the saved ELA difference image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"File not found: {image_path}")

    output_dir = Path(output_dir) if output_dir else image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    original = Image.open(image_path).convert("RGB")

    # Recompress at a known quality and reload
    resaved_path = output_dir / f"{image_path.stem}_resaved_temp.jpg"
    original.save(resaved_path, "JPEG", quality=quality)
    resaved = Image.open(resaved_path)

    # Pixel-level difference between original and recompressed version
    diff = ImageChops.difference(original, resaved)
    diff_array = np.array(diff)

    # Scale the difference so it's visible in the saved image
    max_diff = diff_array.max() if diff_array.size else 0
    scale = 255.0 / max_diff if max_diff != 0 else 1.0
    ela_visual = (diff_array * scale).astype(np.uint8)

    ela_image_path = output_dir / f"{image_path.stem}_ela.jpg"
    Image.fromarray(ela_visual).save(ela_image_path)

    # Simple anomaly score: mean brightness of the difference map.
    # Higher = more inconsistent compression history = more suspicious.
    anomaly_score = float(np.mean(diff_array))

    resaved_path.unlink(missing_ok=True)  # clean up temp file

    return {
        "anomaly_score": round(anomaly_score, 2),
        "ela_image_path": str(ela_image_path),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python ela_utils.py <image_path>")
        sys.exit(1)

    result = run_ela(sys.argv[1])
    print(result)