"""
confidence_fusion.py
Combines forensic anomaly signals with the model's prediction confidence
into one composite verdict + score.

INTERFACE DESIGN — the function signature and return schema are final.
The fusion math below (weights, normalization) is a placeholder so the
function is callable and testable today. Replace it with properly
calibrated weights once real anomaly-score and model-confidence
distributions exist (Week 7) — e.g. by sweeping weights against a
validation set instead of guessing.
"""

from typing import Optional


def fuse_confidence(
    forensic_result: dict,
    model_confidence: Optional[float] = None,
    model_verdict: Optional[str] = None,
) -> dict:
    """
    Combine forensic signals and model confidence into one fused verdict.

    Args:
        forensic_result: dict from forensics_pipeline.run_forensics().
            Currently only ela.anomaly_score feeds the fusion — this is
            where metadata red flags or malware status get folded in
            as the logic matures.
        model_confidence: model's confidence the file is "fake" (0.0-1.0).
            None until Charlie's model is integrated (Week 6).
        model_verdict: model's raw predicted label ("real" | "fake").
            None until Week 6.

    Returns:
        Dict with:
            - fused_score: float (0.0-1.0), composite suspicion score
            - fused_verdict: "real" | "fake" | "inconclusive"
            - components: raw inputs that went into the fusion, kept for
              transparency in the dashboard and PDF report
    """
    ela = forensic_result.get("ela") or {}
    anomaly_score = ela.get("anomaly_score")

    components = {
        "ela_anomaly_score": anomaly_score,
        "model_confidence": model_confidence,
        "model_verdict": model_verdict,
    }

    # --- PLACEHOLDER LOGIC — replace once real score distributions exist ---
    if model_confidence is None:
        # No model output yet — forensics-only signal, never a firm verdict.
        forensic_component = min((anomaly_score or 0) / 50.0, 1.0)
        fused_score = round(forensic_component, 4)
        fused_verdict = "inconclusive"
    else:
        forensic_component = min((anomaly_score or 0) / 50.0, 1.0)
        fused_score = round(0.7 * model_confidence + 0.3 * forensic_component, 4)
        fused_verdict = "fake" if fused_score >= 0.5 else "real"

    return {
        "fused_score": fused_score,
        "fused_verdict": fused_verdict,
        "components": components,
    }


if __name__ == "__main__":
    dummy_forensics = {"ela": {"anomaly_score": 0.18}}

    print("Forensics-only (no model yet):")
    print(fuse_confidence(dummy_forensics))

    print("\nWith dummy model output:")
    print(fuse_confidence(dummy_forensics, model_confidence=0.91, model_verdict="fake"))