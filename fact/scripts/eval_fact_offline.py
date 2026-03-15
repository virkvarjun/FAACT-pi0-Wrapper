#!/usr/bin/env python
"""Offline evaluation: AUROC, AUPRC, F1, ECE, Brier."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fact.fact.models import FactMLP
from fact.training.calibrate import _compute_ece

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean((probs - y_true) ** 2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/fact_dataset.npz")
    p.add_argument("--checkpoint", default="checkpoints/fact/best_model.pt")
    p.add_argument("--output", default="eval/offline_metrics.json")
    args = p.parse_args()

    data = np.load(args.dataset)
    X, y = data["X"], data["y"]
    if len(X) == 0:
        logger.warning("Empty dataset")
        return

    with open(Path(args.checkpoint).parent / "config.json") as f:
        config = json.load(f)
    model = FactMLP(input_dim=config["input_dim"])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float()).numpy()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

    metrics = {}
    if len(np.unique(y)) > 1:
        metrics["auroc"] = float(roc_auc_score(y, probs))
        metrics["auprc"] = float(average_precision_score(y, probs))
    pred = (probs >= 0.5).astype(int)
    metrics["f1"] = float(f1_score(y, pred, zero_division=0))
    metrics["brier"] = float(brier_score(probs, y))
    metrics["ece"] = float(_compute_ece(probs, y))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
