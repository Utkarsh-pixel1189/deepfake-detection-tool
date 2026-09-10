"""
metrics.py
Evaluation harness for model predictions using scikit-learn.

SKELETON — fully implemented and runnable today, but exercised only
with dummy data until Charlie's model produces real predictions.
Once Week 6 integration happens, swap the dummy arrays in __main__
for real y_true/y_pred/y_proba and this module needs no changes.
"""

from typing import Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_predictions(y_true, y_pred, y_proba: Optional[np.ndarray] = None) -> dict:
    """
    Compute the full evaluation suite for a set of predictions.

    Args:
        y_true: ground truth labels (0 = real, 1 = fake), array-like.
        y_pred: predicted labels (0 or 1), array-like.
        y_proba: predicted probability of the "fake" class, needed for
            AUC-ROC. Optional — auc_roc will be None if omitted.

    Returns:
        Dict with accuracy, precision, recall, f1, auc_roc (if available),
        and the raw confusion matrix.
    """
    results = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc": None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None:
        try:
            results["auc_roc"] = round(roc_auc_score(y_true, y_proba), 4)
        except ValueError as e:
            # e.g. only one class present in y_true — AUC undefined
            results["auc_roc_error"] = str(e)

    return results


def full_report(y_true, y_pred) -> str:
    """Human-readable per-class report — useful for logs or the PDF report."""
    return classification_report(y_true, y_pred, target_names=["real", "fake"], zero_division=0)


if __name__ == "__main__":
    # Dummy data so this runs and is verifiable right now, before real
    # predictions exist. Replace with real arrays once Charlie's model
    # produces predictions (Week 6+).
    dummy_y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    dummy_y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    dummy_y_proba = np.array([0.1, 0.8, 0.4, 0.2, 0.9, 0.3, 0.7, 0.85, 0.6, 0.15])

    metrics = evaluate_predictions(dummy_y_true, dummy_y_pred, dummy_y_proba)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nFull classification report:")
    print(full_report(dummy_y_true, dummy_y_pred))