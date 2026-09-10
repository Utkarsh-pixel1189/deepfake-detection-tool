"""
hash_utils.py

Computes SHA-256 hash of a file for integrity verification.
"""

import hashlib
from pathlib import Path


def compute_sha256(file_path) -> str:
    """
    Compute the SHA-256 hash of a file.

    Args:
        file_path: path to the file to hash.

    Returns:
        Hex string of the SHA-256 hash.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in chunks so large video files don't get loaded into memory at once
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python hash_utils.py <file_path>")
        sys.exit(1)

    result = compute_sha256(sys.argv[1])
    print(f"SHA-256: {result}")