#!/usr/bin/env python
"""Collect rollout data from backbone policy for FAACT training."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from faact.data.rollout_logger import RolloutLogger
from faact.data.schemas import ChunkRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def make_env(task: str, env_type: str, max_steps: int | None = None):
    """Create gym env (e.g. gym_aloha/AlohaTransferCube-v0)."""
    import gymnasium as gym

    gym_kwargs = {"obs_type": "pixels_agent_pos", "render_mode": "rgb_array"}
    if max_steps is not None:
        gym_kwargs["max_episode_steps"] = max_steps

    gym_id = f"gym_{env_type}/{task}"
    return gym.make(gym_id, **gym_kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description="Collect FAACT rollout data from backbone policy")
    p.add_argument("--output_dir", default="data/rollouts")
    p.add_argument("--run_id", default="rollout")
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--backbone", choices=["act", "pi05", "stub"], default="stub",
                   help="Backbone policy (act=LeRobot ACT, pi05=π₀.5, stub=synthetic)")
    p.add_argument("--checkpoint", default="",
                   help="Path to backbone checkpoint (required for act/pi05)")
    p.add_argument("--task", default="pick up the object",
                   help="Task spec for stub; or env task ID when --env_task/--env_type set")
    p.add_argument("--env_task", default="",
                   help="Gym task ID (e.g. AlohaTransferCube-v0). If set, run real rollouts.")
    p.add_argument("--env_type", default="aloha",
                   help="Env package (aloha, pusht, etc.)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--failure_horizon_k", type=int, default=5,
                   help="K for failure_within_k label")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Resolve backbone
    if args.backbone == "act":
        if not args.checkpoint:
            p.error("--checkpoint required for backbone=act")
        try:
            from faact.backbone import ACTPolicyWrapper
            backbone = ACTPolicyWrapper(args.checkpoint, device=args.device)
        except ImportError as e:
            logger.error("ACTPolicyWrapper requires lerobot. %s", e)
            sys.exit(1)
    elif args.backbone == "pi05":
        if not args.checkpoint:
            p.error("--checkpoint required for backbone=pi05")
        try:
            from faact.backbone import Pi05PolicyWrapper
            backbone = Pi05PolicyWrapper(args.checkpoint)
        except ImportError:
            logger.warning("lerobot not available, falling back to stub")
            from faact.backbone import StubBackboneWrapper
            backbone = StubBackboneWrapper()
    else:
        from faact.backbone import StubBackboneWrapper
        backbone = StubBackboneWrapper()

    rl = RolloutLogger(args.output_dir, run_id=args.run_id)

    use_real_env = bool(args.env_task)

    if use_real_env:
        try:
            importlib.import_module(f"gym_{args.env_type}")
        except ModuleNotFoundError as e:
            logger.error("Env package 'gym_%s' not found. pip install gym-%s", args.env_type, args.env_type)
            sys.exit(1)

        env = make_env(args.env_task, args.env_type, args.max_steps)
        task_id = args.env_task
    else:
        env = None
        task_id = args.task

    success_count = 0
    for ep in range(args.num_episodes):
        rl.start_episode(episode_id=ep, task_id=task_id)
        backbone.reset(task_spec=task_id)

        if use_real_env:
            obs, info = env.reset(seed=args.seed + ep)
            max_steps = env.spec.max_episode_steps or 400
            if args.max_steps:
                max_steps = args.max_steps
            current_chunk = None
            chunk_step_idx = 0
            n_action_steps = backbone.chunk_size
            step = 0
            success = False
            failure_step = None

            while step < max_steps:
                need_new_chunk = (current_chunk is None) or (chunk_step_idx >= n_action_steps)
                if need_new_chunk:
                    proposal = backbone.propose_chunk(obs, return_features=True)
                    current_chunk = proposal.actions
                    current_features = proposal.features
                    chunk_step_idx = 0
                    chunk_index = step // n_action_steps
                    timestep = step

                    action_mean = current_chunk.mean(axis=0).astype(np.float32)
                    raw_feat = {}
                    if current_features and current_features.raw:
                        raw_feat = dict(current_features.raw)
                        if "decoder_mean" in raw_feat:
                            action_mean = np.asarray(raw_feat["decoder_mean"], dtype=np.float32)
                            raw_feat["action_chunk_mean"] = action_mean

                    rec = ChunkRecord(
                        episode_id=ep,
                        chunk_index=chunk_index,
                        step_index=chunk_index,
                        timestep=timestep,
                        action_chunk=current_chunk,
                        action_chunk_mean=action_mean,
                        observation_embedding=current_features.observation_embedding if current_features else None,
                        raw_features=raw_feat,
                        y_fail_within_k_chunks=0,
                        y_episode_fail=0,
                        task_id=task_id,
                    )
                    rl.log_chunk(rec)

                action = current_chunk[chunk_step_idx]
                obs, reward, terminated, truncated, info = env.step(action)
                chunk_step_idx += 1
                step += 1
                done = terminated or truncated

                if info.get("is_success", False):
                    success = True
                if done and not success:
                    failure_step = step

                if done:
                    break

            if success:
                success_count += 1

            rl.end_episode(
                success=success,
                failure_step=failure_step,
                metadata={
                    "real_env": True,
                    "backbone": args.backbone,
                    "checkpoint": args.checkpoint or "",
                },
            )
        else:
            # Stub: simulated 5 chunks per episode
            for c in range(5):
                obs = {
                    "pixels": {"main": np.zeros((224, 224, 3), dtype=np.uint8)},
                    "agent_pos": np.zeros(14, dtype=np.float32),
                }
                proposal = backbone.propose_chunk(obs, return_features=True)
                feat = proposal.features
                action_mean = proposal.actions.mean(axis=0).astype(np.float32)
                raw_feat = {}
                if feat and feat.raw and "action_chunk_mean" in feat.raw:
                    action_mean = np.asarray(feat.raw["action_chunk_mean"], dtype=np.float32)
                    raw_feat = dict(feat.raw)
                rec = ChunkRecord(
                    episode_id=ep,
                    chunk_index=c,
                    step_index=c,
                    timestep=c * backbone.chunk_size,
                    action_chunk=proposal.actions,
                    action_chunk_mean=action_mean,
                    raw_features=raw_feat,
                    y_fail_within_k_chunks=0,
                    y_episode_fail=0,
                    task_id=task_id,
                )
                rl.log_chunk(rec)

            rl.end_episode(success=True, metadata={"simulated": True})

    if use_real_env and env:
        env.close()

    path = rl.save(backbone_checkpoint=args.checkpoint or "", task_ids=[task_id])
    n_success = success_count if use_real_env else args.num_episodes
    logger.info("Saved to %s (%d episodes, %d success)", path, args.num_episodes, n_success)


if __name__ == "__main__":
    main()
