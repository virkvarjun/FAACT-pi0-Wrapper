#!/usr/bin/env python
"""Calibrate FAACT model: threshold sweep, ECE."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from faact.predictor.models import FactMLP
from faact.training.calibrate import temperature_scale, threshold_sweep, _compute_ece

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/fact_dataset.npz")
    p.add_argument("--checkpoint", default="checkpoints/fact/best_model.pt")
    p.add_argument("--output", default="checkpoints/fact/calibration.json")
    args = p.parse_args()

    data = np.load(args.dataset)
    X, y = data["X"], data["y"]
    if len(X) == 0:
        logger.warning("Empty dataset, skipping calibration")
        return

    with open(Path(args.checkpoint).parent / "config.json") as f:
        config = json.load(f)
    model = FactMLP(input_dim=config["input_dim"])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float()).numpy()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

    t = temperature_scale(logits, probs, y)
    scaled_probs = 1.0 / (1.0 + np.exp(-logits / t))
    sweep = threshold_sweep(scaled_probs, y)
    ece = _compute_ece(scaled_probs, y)
    result = {
        "temperature": t,
        "ece": ece,
        "threshold_sweep": sweep,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Calibration saved to %s: T=%.3f ECE=%.4f best_thr=%.3f", args.output, t, ece, sweep["best_threshold"])


if __name__ == "__main__":
    main()
