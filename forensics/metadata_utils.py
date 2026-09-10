"""
metadata_utils.py

Extracts metadata from an image/video file using ExifTool.
Requires ExifTool to be installed on the system: brew install exiftool
"""

import json
import subprocess
from pathlib import Path


def extract_metadata(file_path) -> dict:
    """
    Extract metadata from a file using ExifTool.

    Args:
        file_path: path to the image or video file.

    Returns:
        Dictionary of metadata fields.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        result = subprocess.run(
            ["exiftool", "-j", str(file_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ExifTool not found. Install it with: brew install exiftool"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ExifTool failed: {e.stderr}")

    metadata_list = json.loads(result.stdout)
    # exiftool -j returns a list with one dict per file; we passed one file
    return metadata_list[0] if metadata_list else {}


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python metadata_utils.py <file_path>")
        sys.exit(1)

    metadata = extract_metadata(sys.argv[1])
    print(json.dumps(metadata, indent=2))