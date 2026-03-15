#!/usr/bin/env python
"""Collect rollout data from backbone policy for FAACT training."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from faact.backbone.stub import StubBackboneWrapper
from faact.data.rollout_logger import RolloutLogger
from faact.data.schemas import ChunkRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="data/rollouts")
    p.add_argument("--run_id", default="rollout")
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--checkpoint", default="", help="PI05 checkpoint (empty => stub)")
    p.add_argument("--task", default="pick up the object")
    args = p.parse_args()

    if args.checkpoint:
        try:
            from faact.backbone.pi05_wrapper import Pi05PolicyWrapper
            backbone = Pi05PolicyWrapper(args.checkpoint)
        except ImportError:
            logger.warning("lerobot not available, using stub")
            backbone = StubBackboneWrapper()
    else:
        backbone = StubBackboneWrapper()

    rl = RolloutLogger(args.output_dir, run_id=args.run_id)

    for ep in range(args.num_episodes):
        rl.start_episode(episode_id=ep, task_id=args.task)
        backbone.reset(task_spec=args.task)

        # Simulated rollout: 5 chunks per episode
        for c in range(5):
            obs = {
                "pixels": {"main": np.zeros((224, 224, 3), dtype=np.uint8)},
                "agent_pos": np.zeros(14, dtype=np.float32),
            }
            proposal = backbone.propose_chunk(obs, return_features=True)
            feat = proposal.features
            action_mean = proposal.actions.mean(axis=0)
            if feat and "action_chunk_mean" in feat.raw:
                action_mean = np.asarray(feat.raw["action_chunk_mean"], dtype=np.float32)
            rec = ChunkRecord(
                episode_id=ep,
                chunk_index=c,
                step_index=c,
                timestep=c * backbone.chunk_size,
                action_chunk=proposal.actions,
                action_chunk_mean=action_mean,
                y_fail_within_k_chunks=0,
                y_episode_fail=0,
                task_id=args.task,
            )
            rl.log_chunk(rec)

        rl.end_episode(success=True, metadata={"simulated": True})

    path = rl.save()
    logger.info("Saved to %s", path)


if __name__ == "__main__":
    main()
