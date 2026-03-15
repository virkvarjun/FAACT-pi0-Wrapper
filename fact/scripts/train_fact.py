#!/usr/bin/env python
"""Train FACT model on chunk-level dataset."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fact.training.trainer import train_fact

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/fact_dataset.npz")
    p.add_argument("--output_dir", default="checkpoints/fact")
    p.add_argument("--model_type", default="mlp", choices=["mlp", "temporal"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None, help="cuda, cpu, or mps")
    args = p.parse_args()

    data = np.load(args.dataset)
    X, y = data["X"], data["y"]
    result = train_fact(
        X, y,
        model_type=args.model_type,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device or None,
    )
    logger.info("Training done: %s", result)


if __name__ == "__main__":
    main()
