"""π₀.5 policy wrapper for FAACT backbone."""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from faact.backbone.base import BackboneFeatures, ChunkProposal, BackbonePolicyWrapper

logger = logging.getLogger(__name__)

# LeRobot imports - optional for when lerobot is installed
try:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    PI05Policy = None
    make_pre_post_processors = None


class Pi05PolicyWrapper(BackbonePolicyWrapper):
    """Wrapper around LeRobot PI05Policy for chunk-level inference.

    Uses observation + action chunk as feature fallback when internal
    backbone features are not accessible.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        task_default: str = "pick up the object",
        norm_stats_path: str | Path | None = None,
    ) -> None:
        if not LEROBOT_AVAILABLE:
            raise ImportError(
                "lerobot is required for Pi05PolicyWrapper. "
                "Install from Research/lerobot: pip install -e path/to/Research/lerobot"
            )
        path = Path(checkpoint_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        self.device = device
        self.task_default = task_default
        self._step_index = 0
        self._action_queue: deque = deque()

        policy = PI05Policy.from_pretrained(
            str(path),
            local_files_only=True,
        )
        policy.to(device)
        policy.eval()

        preprocessor_overrides = {"device_processor": {"device": device}}
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(path),
            preprocessor_overrides=preprocessor_overrides,
            dataset_stats=None,  # Will use pretrained norm stats from checkpoint
        )

        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor

        # chunk_size from policy config
        self._chunk_size = policy.config.n_action_steps

    def reset(self, task_spec: str | None = None) -> None:
        """Reset policy state for new episode."""
        self._step_index = 0
        self._action_queue.clear()
        if hasattr(self._policy, "_action_queue"):
            self._policy._action_queue.clear()
        self._task = task_spec or self.task_default

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def _obs_to_batch(
        self,
        obs: dict[str, np.ndarray | dict[str, np.ndarray]],
        task: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Convert env observation to PI05 batch format.

        Expected obs keys:
            - pixels: dict[cam_name, (H,W,C) uint8] or single array
            - agent_pos or state: (state_dim,) float32
        """
        task = task or self._task
        batch: dict[str, Any] = {}

        # Images: LeRobot uses observation.images.<key>
        pixels = obs.get("pixels") or (obs.get("observation") or {}).get("images", {})
        if isinstance(pixels, dict):
            for key, img in pixels.items():
                arr = np.asarray(img, dtype=np.uint8)
                if arr.ndim == 3:
                    arr = arr[np.newaxis, ...]
                t = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
                batch[f"observation.images.{key}"] = t
        elif pixels is not None:
            arr = np.asarray(pixels)
            if arr.ndim == 3:
                arr = arr[np.newaxis, ...]
            t = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
            batch["observation.images.main"] = t

        # State: LeRobot uses observation.state
        state = obs.get("agent_pos") or obs.get("state") or (obs.get("observation") or {}).get("state")
        if state is not None:
            s = np.asarray(state, dtype=np.float32)
            if s.ndim == 1:
                s = s[np.newaxis, :]
            batch["observation.state"] = torch.from_numpy(s)

        # Task for PI05 tokenizer
        batch["task"] = [task]

        return self._preprocessor(batch)

    def propose_chunk(
        self,
        obs: dict[str, np.ndarray | dict[str, np.ndarray]],
        context: dict[str, Any] | None = None,
        return_features: bool = False,
    ) -> ChunkProposal:
        """Propose action chunk from π₀.5."""
        task = (context or {}).get("task", self._task)
        batch = self._obs_to_batch(obs, task=task)

        with torch.inference_mode():
            actions = self._policy.predict_action_chunk(batch)
            actions = actions[:, : self._chunk_size]

        # Postprocess each action in chunk to env scale
        actions_processed = []
        for i in range(actions.shape[1]):
            single = actions[:, i, :]  # (B, action_dim)
            out = self._postprocessor(single)
            arr = out.cpu().numpy() if torch.is_tensor(out) else out
            actions_processed.append(arr)
        chunk_np = np.stack(actions_processed, axis=0)  # (chunk_size, B, action_dim)
        if chunk_np.ndim == 3:
            chunk_np = chunk_np[:, 0, :]  # (chunk_size, action_dim)
        chunk_np = np.asarray(chunk_np, dtype=np.float32)

        features = None
        if return_features:
            features = BackboneFeatures(
                raw={
                    "action_chunk_flattened": chunk_np.flatten(),
                    "action_chunk_mean": chunk_np.mean(axis=0),
                }
            )

        proposal = ChunkProposal(
            actions=chunk_np,
            step_index=self._step_index,
            features=features,
        )
        self._step_index += 1
        return proposal
