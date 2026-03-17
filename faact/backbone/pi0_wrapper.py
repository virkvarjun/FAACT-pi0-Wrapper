"""PI0 policy wrapper for FAACT backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from faact.backbone.base import BackboneFeatures, ChunkProposal, BackbonePolicyWrapper
from faact.backbone.features import derive_action_features

try:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    PI0Policy = None
    make_pre_post_processors = None


def _preprocess_obs(obs: dict, task_desc: str | None = None):
    """Convert gym observations into the flat LeRobot batch format for PI0."""
    result: dict[str, torch.Tensor] = {}

    if "pixels" in obs:
        pixels = obs["pixels"]
        if isinstance(pixels, dict):
            for key, img in pixels.items():
                img_t = torch.from_numpy(np.asarray(img))
                if img_t.ndim == 3:
                    img_t = img_t.unsqueeze(0)
                target_key = key if str(key).startswith("observation.images.") else f"observation.images.{key}"
                result[target_key] = img_t.permute(0, 3, 1, 2).float() / 255.0
        else:
            img_t = torch.from_numpy(np.asarray(pixels))
            if img_t.ndim == 3:
                img_t = img_t.unsqueeze(0)
            result["observation.images.main"] = img_t.permute(0, 3, 1, 2).float() / 255.0

    if "agent_pos" in obs:
        state = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))
        if state.ndim == 1:
            state = state.unsqueeze(0)
        result["observation.state"] = state

    if task_desc:
        result["task"] = task_desc if task_desc.endswith("\n") else f"{task_desc}\n"
    return result


def _map_pixels_to_expected_keys(
    obs: dict,
    expected_image_keys: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Map generic env camera observations onto the image keys expected by PI0."""
    pixels = obs.get("pixels")
    if pixels is None:
        return {}

    if isinstance(pixels, dict):
        available = {str(key): np.asarray(value, dtype=np.uint8) for key, value in pixels.items()}
    else:
        available = {"main": np.asarray(pixels, dtype=np.uint8)}

    if not available:
        return {}

    if not expected_image_keys:
        return available

    aliases = {
        "main": ["main", "top", "base_0_rgb"],
        "base_0_rgb": ["base_0_rgb", "top", "main"],
        "left_wrist_0_rgb": ["left_wrist_0_rgb", "left_wrist", "wrist_left", "main", "top"],
        "right_wrist_0_rgb": ["right_wrist_0_rgb", "right_wrist", "wrist_right", "main", "top"],
    }
    fallback = next(iter(available.values()))
    mapped: dict[str, np.ndarray] = {}
    for key in expected_image_keys:
        preferred = aliases.get(key, [key, "main", "top"])
        chosen = next((available[name] for name in preferred if name in available), fallback)
        mapped[key] = chosen
    return mapped


class PI0PolicyWrapper(BackbonePolicyWrapper):
    """Wrapper around LeRobot PI0Policy for chunk-level inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        task_default: str | None = None,
    ) -> None:
        if not LEROBOT_AVAILABLE:
            raise ImportError(
                "lerobot with PI0 support is required for PI0PolicyWrapper. "
                "Install the local lerobot environment before using this wrapper."
            )

        path = Path(checkpoint_path).resolve()
        ckpt_str = str(path) if path.exists() else checkpoint_path

        self.device = device
        self.task_default = task_default
        self._task = task_default
        self._step_index = 0

        policy = PI0Policy.from_pretrained(
            pretrained_name_or_path=ckpt_str,
            local_files_only=path.exists(),
        )
        policy.to(device)
        policy.eval()

        preprocessor_overrides = {"device_processor": {"device": device}}
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=ckpt_str,
            preprocessor_overrides=preprocessor_overrides,
        )

        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._chunk_size = getattr(policy.config, "n_action_steps", getattr(policy.config, "chunk_size", 1))
        input_features = getattr(policy.config, "input_features", {}) or {}
        self._image_feature_keys = [key for key in input_features if str(key).startswith("observation.images.")]

    def reset(self, task_spec: str | None = None) -> None:
        self._step_index = 0
        self._task = task_spec or self.task_default
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def propose_chunk(
        self,
        obs: dict[str, np.ndarray | dict[str, np.ndarray]],
        context: dict[str, Any] | None = None,
        return_features: bool = False,
    ) -> ChunkProposal:
        task_desc = (context or {}).get("task") or self._task
        obs_for_policy = dict(obs)
        obs_for_policy["pixels"] = _map_pixels_to_expected_keys(obs, self._image_feature_keys)
        obs_t = _preprocess_obs(obs_for_policy, task_desc=task_desc)
        batch = self._preprocessor(obs_t)

        with torch.inference_mode():
            actions = self._policy.predict_action_chunk(batch)

        actions = actions[:, : self._chunk_size]
        actions_list = []
        for idx in range(actions.shape[1]):
            single = actions[:, idx, :]
            out = self._postprocessor(single)
            arr = out.cpu().numpy() if torch.is_tensor(out) else out
            actions_list.append(arr)

        chunk_np = np.stack(actions_list, axis=0)
        if chunk_np.ndim == 3:
            chunk_np = chunk_np[:, 0, :]
        chunk_np = np.asarray(chunk_np, dtype=np.float32)

        features = None
        if return_features:
            raw = {
                "action_chunk_mean": chunk_np.mean(axis=0).astype(np.float32),
            }
            raw.update(derive_action_features(chunk_np, chunk_step_idx=0))
            features = BackboneFeatures(
                action_chunk_embedding=raw["action_chunk_mean"],
                raw=raw,
            )

        proposal = ChunkProposal(
            actions=chunk_np,
            step_index=self._step_index,
            features=features,
        )
        self._step_index += 1
        return proposal
