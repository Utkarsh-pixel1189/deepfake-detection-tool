"""
forensics_pipeline.py

Orchestrator: runs all four forensic checks on a single uploaded file
and returns one combined result. This is the function the Streamlit
app and the model pipeline will call.
"""

from pathlib import Path

from ela_utils import run_ela
from hash_utils import compute_sha256
from malware_scan import scan_file
from metadata_utils import extract_metadata

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def run_forensics(file_path) -> dict:
    """
    Run the full forensics suite on a file.

    Args:
        file_path: path to the uploaded file (image or video).

    Returns:
        Combined dictionary with hash, metadata, ELA (images only), and
        malware scan results. Any individual check that fails is reported
        with an "_error" field rather than crashing the whole pipeline.
    """
    file_path = Path(file_path)
    results = {"file_name": file_path.name}

    try:
        results["sha256"] = compute_sha256(file_path)
    except Exception as e:
        results["sha256"] = None
        results["sha256_error"] = str(e)

    try:
        results["metadata"] = extract_metadata(file_path)
    except Exception as e:
        results["metadata"] = None
        results["metadata_error"] = str(e)

    if file_path.suffix.lower() in IMAGE_EXTENSIONS:
        try:
            results["ela"] = run_ela(file_path)
        except Exception as e:
            results["ela"] = None
            results["ela_error"] = str(e)
    else:
        results["ela"] = None
        results["ela_note"] = "ELA only runs on static images, skipped for this file type."

    try:
        results["malware_scan"] = scan_file(file_path)
    except Exception as e:
        results["malware_scan"] = None
        results["malware_scan_error"] = str(e)

    return results


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("Usage: python forensics_pipeline.py <file_path>")
        sys.exit(1)

    output = run_forensics(sys.argv[1])
    print(json.dumps(output, indent=2, default=str))