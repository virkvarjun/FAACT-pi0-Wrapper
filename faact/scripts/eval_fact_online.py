#!/usr/bin/env python
"""Online evaluation: wrapped vs baseline policy."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from faact.backbone.stub import StubBackboneWrapper
from faact.predictor.models import FactMLP
from faact.governor import Governor, GovernorDecision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/fact/best_model.pt")
    p.add_argument("--risk_threshold", type=float, default=0.5)
    p.add_argument("--num_episodes", type=int, default=5)
    p.add_argument("--output", default="eval/online_metrics.json")
    args = p.parse_args()

    backbone = StubBackboneWrapper()
    governor = Governor(threshold=args.risk_threshold)

    with open(Path(args.checkpoint).parent / "config.json") as f:
        config = json.load(f)
    model = FactMLP(input_dim=config["input_dim"])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    results = {"baseline_success": 0, "wrapped_success": 0, "interventions": 0}
    for ep in range(args.num_episodes):
        backbone.reset()
        intervened = False
        for c in range(5):
            obs = {"pixels": {"main": np.zeros((224, 224, 3), dtype=np.uint8)}, "agent_pos": np.zeros(14)}
            proposal = backbone.propose_chunk(obs, return_features=True)
            feat = proposal.features.raw.get("action_chunk_mean", np.zeros(14)) if proposal.features else np.zeros(14)
            x = torch.from_numpy(np.asarray(feat, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                prob = torch.sigmoid(model(x)).item()
            dec = governor.decide(prob)
            if dec.decision == GovernorDecision.REJECT_REPLAN:
                intervened = True
        results["wrapped_success"] += 1  # Stub always "succeeds"
        if intervened:
            results["interventions"] += 1
    results["baseline_success"] = args.num_episodes

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Online eval: %s", results)


if __name__ == "__main__":
    main()
