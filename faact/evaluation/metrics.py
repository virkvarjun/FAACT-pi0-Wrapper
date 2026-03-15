"""Offline and online evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)


def compute_offline_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """AUROC, AUPRC, F1, Brier, ECE."""
    pred = (probs >= threshold).astype(int)
    metrics: dict[str, float] = {
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(np.mean((probs - y_true) ** 2)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, probs))
        metrics["auprc"] = float(average_precision_score(y_true, probs))
    return metrics


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() * np.abs(probs[mask].mean() - y_true[mask].mean())
    return float(ece / len(probs))
