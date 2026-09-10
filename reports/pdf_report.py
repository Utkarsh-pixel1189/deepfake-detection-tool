"""
pdf_report.py
Auto-generates a PDF evidence report combining forensics results,
model verdict, and confidence fusion score.

SCAFFOLD — the forensics section is fully wired to forensics_pipeline's
real output schema. The model verdict / Grad-CAM section renders a
placeholder until Charlie's model exists (Week 6); the fusion section
uses fusion/confidence_fusion.py's real output whenever it's passed in.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from fpdf import FPDF
from fpdf.enums import XPos, YPos


class EvidenceReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Deepfake Detection - Evidence Report",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.ln(2)

    def key_value(self, key: str, value: str):
        """
        Key on its own bold line, value on the line(s) below at full page
        width. Stacking avoids width-from-offset issues that break on long
        unbroken strings like SHA-256 hashes.
        """
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, f"{key}:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, str(value))
        self.ln(1)


def generate_pdf_report(
    forensics_result: dict,
    model_result: Optional[dict] = None,
    fusion_result: Optional[dict] = None,
    output_path: str = "evidence_report.pdf",
) -> str:
    """
    Args:
        forensics_result: dict from forensics_pipeline.run_forensics() — required.
        model_result: {"verdict": ..., "confidence": ..., "gradcam_image_path": ...}
            or None (Week 6+ only).
        fusion_result: output of fusion.confidence_fusion.fuse_confidence(), or None.
        output_path: where to write the PDF.

    Returns:
        output_path, for convenience chaining.
    """
    pdf = EvidenceReport()
    pdf.add_page()

    pdf.section_title("File Identification")
    pdf.key_value("File name", forensics_result.get("file_name", "N/A"))
    pdf.key_value(
        "SHA-256",
        forensics_result.get("sha256") or f"Error: {forensics_result.get('sha256_error', 'unknown')}",
    )
    pdf.ln(4)

    pdf.section_title("Malware Scan")
    scan = forensics_result.get("malware_scan")
    if scan:
        pdf.key_value("Status", scan.get("status", "N/A"))
        pdf.key_value("Details", scan.get("details", "N/A"))
    else:
        pdf.key_value("Status", f"Error: {forensics_result.get('malware_scan_error', 'unknown')}")
    pdf.ln(4)

    pdf.section_title("Metadata Summary")
    metadata = forensics_result.get("metadata")
    if metadata:
        for field in ["FileType", "FileSize", "ImageWidth", "ImageHeight", "CreateDate", "Artist", "Encoder"]:
            if field in metadata:
                pdf.key_value(field, metadata[field])
    else:
        pdf.key_value("Status", f"Error: {forensics_result.get('metadata_error', 'unknown')}")
    pdf.ln(4)

    pdf.section_title("Error Level Analysis")
    ela = forensics_result.get("ela")
    if ela:
        pdf.key_value("Anomaly score", ela.get("anomaly_score", "N/A"))
        ela_image_path = ela.get("ela_image_path")
        if ela_image_path and Path(ela_image_path).exists():
            pdf.ln(2)
            pdf.image(ela_image_path, w=80)
    elif forensics_result.get("ela_note"):
        pdf.key_value("Status", forensics_result["ela_note"])
    else:
        pdf.key_value("Status", f"Error: {forensics_result.get('ela_error', 'unknown')}")
    pdf.ln(4)

    pdf.section_title("Model Verdict")
    if model_result:
        pdf.key_value("Verdict", model_result.get("verdict", "N/A"))
        pdf.key_value("Confidence", model_result.get("confidence", "N/A"))
        gradcam_path = model_result.get("gradcam_image_path")
        if gradcam_path and Path(gradcam_path).exists():
            pdf.ln(2)
            pdf.image(gradcam_path, w=80)
    else:
        pdf.key_value("Status", "Not yet available - model integration pending (Week 6).")
    pdf.ln(4)

    pdf.section_title("Combined Confidence Score")
    if fusion_result:
        pdf.key_value("Fused verdict", fusion_result.get("fused_verdict", "N/A"))
        pdf.key_value("Fused score", fusion_result.get("fused_score", "N/A"))
    else:
        pdf.key_value("Status", "Not yet available.")

    pdf.output(output_path)
    return output_path


if __name__ == "__main__":
    dummy_forensics = {
        "file_name": "sample.jpg",
        "sha256": "a7413f2b2c946964c3cc407dc553d5b6a5d5d09ce8adb4e32ee86edec359f242",
        "metadata": {"FileType": "JPEG", "FileSize": "561 kB", "ImageWidth": 1715, "ImageHeight": 2572},
        "ela": {"anomaly_score": 0.18, "ela_image_path": "test_ela.jpg"},
        "malware_scan": {"status": "clean", "details": "sample.jpg: OK"},
    }
    path = generate_pdf_report(dummy_forensics, output_path="test_report.pdf")
    print(f"Report written to: {path}")