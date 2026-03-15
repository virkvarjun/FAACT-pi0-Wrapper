#!/usr/bin/env python
"""Run wrapped policy with FACT governor in the loop."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fact.backbone.stub import StubBackboneWrapper
from fact.fact.models import FactMLP
from fact.governor import Governor, GovernorDecision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone_checkpoint", default="")
    p.add_argument("--fact_checkpoint", default="checkpoints/fact/best_model.pt")
    p.add_argument("--risk_threshold", type=float, default=0.5)
    p.add_argument("--task", default="pick up the object")
    p.add_argument("--num_steps", type=int, default=10)
    args = p.parse_args()

    if args.backbone_checkpoint:
        try:
            from fact.backbone.pi05_wrapper import Pi05PolicyWrapper
            backbone = Pi05PolicyWrapper(args.backbone_checkpoint)
        except ImportError:
            backbone = StubBackboneWrapper()
    else:
        backbone = StubBackboneWrapper()

    governor = Governor(threshold=args.risk_threshold)
    with open(Path(args.fact_checkpoint).parent / "config.json") as f:
        import json
        config = json.load(f)
    fact = FactMLP(input_dim=config["input_dim"])
    fact.load_state_dict(torch.load(args.fact_checkpoint, map_location="cpu"))
    fact.eval()

    backbone.reset(task_spec=args.task)
    step = 0
    while step < args.num_steps:
        obs = {"pixels": {"main": np.zeros((224, 224, 3), dtype=np.uint8)}, "agent_pos": np.zeros(14)}
        proposal = backbone.propose_chunk(obs, return_features=True)
        feat = proposal.features.raw.get("action_chunk_mean", np.zeros(14)) if proposal.features else np.zeros(14)
        x = torch.from_numpy(np.asarray(feat, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            risk = torch.sigmoid(fact(x)).item()
        dec = governor.decide(risk)
        if dec.decision == GovernorDecision.EXECUTE:
            logger.info("Step %d: execute chunk risk=%.3f", step, risk)
            step += len(proposal.actions)
        else:
            logger.info("Step %d: reject risk=%.3f replan", step, risk)
            backbone.reset(task_spec=args.task)
    logger.info("Done %d steps", step)


if __name__ == "__main__":
    main()
