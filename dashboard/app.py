"""
app.py
Streamlit dashboard for the Deepfake Detection Tool.
Forensics + confidence fusion visualization. Model verdict/Grad-CAM
sections activate once Charlie's model is integrated (Week 6).

Run with: streamlit run dashboard/app.py
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "forensics"))
sys.path.insert(0, str(PROJECT_ROOT / "fusion"))

from forensics_pipeline import run_forensics  # noqa: E402
from confidence_fusion import fuse_confidence  # noqa: E402

st.set_page_config(page_title="Deepfake Forensics Dashboard", layout="wide")

st.title("🔍 Deepfake & Fake Media Detection — Forensics Module")
st.caption("Forensic analysis (hashing, metadata, ELA, malware scan) + confidence fusion preview. Model predictions arrive in a later phase.")

uploaded_files = st.file_uploader(
    "Upload one or more images/videos to analyze",
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
    accept_multiple_files=True,
)


def render_result(uploaded_file, results: dict):
    st.subheader(f"Results for `{uploaded_file.name}`")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔑 File Hash (SHA-256)")
        if results.get("sha256"):
            st.code(results["sha256"], language="text")
        else:
            st.warning(f"Hash failed: {results.get('sha256_error', 'unknown error')}")

    with col2:
        st.markdown("### 🛡️ Malware Scan")
        scan = results.get("malware_scan")
        if scan:
            if scan["status"] == "clean":
                st.success(f"Clean — {scan['details']}")
            elif scan["status"] == "infected":
                st.error(f"Infected! — {scan['details']}")
            else:
                st.warning(f"Scan error — {scan['details']}")
        else:
            st.warning(f"Malware scan failed: {results.get('malware_scan_error', 'unknown error')}")

    st.markdown("### 📋 Metadata")
    if results.get("metadata"):
        with st.expander("View full metadata", expanded=False):
            st.json(results["metadata"])
    else:
        st.warning(f"Metadata extraction failed: {results.get('metadata_error', 'unknown error')}")

    st.markdown("### 🧪 Error Level Analysis")
    if results.get("ela"):
        ela = results["ela"]
        ela_col1, ela_col2 = st.columns([1, 1])
        with ela_col1:
            st.image(ela["ela_image_path"], caption="ELA difference map")
        with ela_col2:
            st.metric("Anomaly score", ela["anomaly_score"])
            st.caption("Higher = more inconsistent compression history = more suspicious.")
    elif results.get("ela_note"):
        st.info(results["ela_note"])
    else:
        st.warning(f"ELA failed: {results.get('ela_error', 'unknown error')}")

    # Forensics-only fusion preview — model_confidence=None until Week 6
    st.markdown("### ⚖️ Combined Confidence (preview)")
    fusion = fuse_confidence(results, model_confidence=None, model_verdict=None)
    st.info(
        f"Fused verdict: **{fusion['fused_verdict']}** (score: {fusion['fused_score']}) — "
        "forensics-only signal for now; model confidence will be added once integrated."
    )

    with st.expander("Raw forensics output (JSON)"):
        st.json(results)
    with st.expander("Raw fusion output (JSON)"):
        st.json(fusion)


if uploaded_files:
    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp.name)

        try:
            with st.spinner(f"Running forensic analysis on {uploaded_file.name}..."):
                results = run_forensics(tmp_path)
            render_result(uploaded_file, results)
        except Exception as e:
            # forensics_pipeline already catches per-check errors internally;
            # this guards against something totally unreadable (e.g. a
            # corrupted upload) from killing the rest of the batch.
            st.error(f"Unexpected error analyzing `{uploaded_file.name}`: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)

        st.divider()
else:
    st.info("Upload one or more files above to begin.")