#!/usr/bin/env python
"""Build FACT training dataset from rollout logs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fact.data.dataset_builder import build_chunk_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_path", required=True)
    p.add_argument("--output_path", default="data/fact_dataset.npz")
    p.add_argument("--failure_horizon_k", type=int, default=5)
    p.add_argument("--feature_key", default="action_chunk_mean")
    args = p.parse_args()

    X, y = build_chunk_dataset(
        args.rollout_path,
        failure_horizon_k=args.failure_horizon_k,
        feature_key=args.feature_key,
    )
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_path, X=X, y=y)
    logger.info("Saved X %s y %s to %s", X.shape, y.shape, args.output_path)


if __name__ == "__main__":
    main()
