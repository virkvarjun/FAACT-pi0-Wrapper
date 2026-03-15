"""Calibration utilities: temperature scaling, threshold sweep."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def temperature_scale(
    logits: np.ndarray,
    probs: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Find optimal temperature for scaling (1D search)."""
    best_t = 1.0
    best_ece = float("inf")
    for t in np.linspace(0.5, 3.0, 26):
        scaled = 1.0 / (1.0 + np.exp(-logits / t))
        ece = _compute_ece(scaled, y_true, n_bins=10)
        if ece < best_ece:
            best_ece = ece
            best_t = t
    return best_t


def _compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += mask.sum() * np.abs(avg_conf - avg_acc)
    return ece / len(probs)


def threshold_sweep(
    probs: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Sweep thresholds and compute F1, precision, recall."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    results = []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        results.append({"threshold": float(t), "precision": prec, "recall": rec, "f1": f1})
    best = max(results, key=lambda r: r["f1"])
    return {"sweep": results, "best_threshold": best["threshold"], "best_f1": best["f1"]}
