"""ACT policy wrapper for FAACT backbone (bridge until PI05)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from faact.backbone.base import BackboneFeatures, ChunkProposal, BackbonePolicyWrapper

logger = logging.getLogger(__name__)

try:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    ACTPolicy = None
    make_pre_post_processors = None


def _preprocess_obs(obs: dict) -> dict[str, torch.Tensor]:
    """Convert gym obs to ACT format: pixels (1,C,H,W) float [0,1], agent_pos -> state."""
    result = {}
    if "pixels" in obs:
        pixels = obs["pixels"]
        if isinstance(pixels, dict):
            for key, img in pixels.items():
                img_t = torch.from_numpy(np.asarray(img))
                if img_t.ndim == 3:
                    img_t = img_t.unsqueeze(0)
                img_t = img_t.permute(0, 3, 1, 2).float() / 255.0
                result[f"observation.images.{key}"] = img_t
        else:
            img_t = torch.from_numpy(np.asarray(pixels))
            if img_t.ndim == 3:
                img_t = img_t.unsqueeze(0)
            img_t = img_t.permute(0, 3, 1, 2).float() / 255.0
            result["observation.images.main"] = img_t
    if "agent_pos" in obs:
        state = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))
        if state.ndim == 1:
            state = state.unsqueeze(0)
        result["observation.state"] = state
    return result


def _features_to_numpy(features: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Extract decoder_mean, encoder_latent_token from ACT features."""
    result = {}
    for key, val in features.items():
        v = val.detach().cpu()
        if key == "encoder_out":
            result["encoder_latent_token"] = v[:, 0, :].squeeze(0).numpy()
        elif key == "decoder_out":
            result["decoder_mean"] = v.mean(dim=1).squeeze(0).numpy()
        elif key == "latent_sample":
            result["latent_sample"] = v.squeeze(0).numpy()
        else:
            result[key] = v.squeeze(0).numpy()
    return result


class ACTPolicyWrapper(BackbonePolicyWrapper):
    """Wrapper around LeRobot ACT for chunk-level inference with decoder_mean features."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ) -> None:
        if not LEROBOT_AVAILABLE:
            raise ImportError(
                "lerobot is required for ACTPolicyWrapper. "
                "Install: pip install -e path/to/Research/lerobot"
            )
        path = Path(checkpoint_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        self.device = device
        self._step_index = 0

        policy = ACTPolicy.from_pretrained(str(path), local_files_only=True)
        policy.to(device)
        policy.eval()

        preprocessor_overrides = {"device_processor": {"device": device}}
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(path),
            preprocessor_overrides=preprocessor_overrides,
        )

        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._chunk_size = policy.config.n_action_steps

    def reset(self, task_spec: str | None = None) -> None:
        self._step_index = 0
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
        obs_t = _preprocess_obs(obs)
        batch = self._preprocessor(obs_t)

        with torch.inference_mode():
            if hasattr(self._policy, "predict_action_chunk_with_features"):
                actions, features = self._policy.predict_action_chunk_with_features(batch)
            else:
                actions = self._policy.predict_action_chunk(batch)
                features = {}

        actions = actions[:, : self._chunk_size]
        # Postprocess each action to env scale
        actions_list = []
        for i in range(actions.shape[1]):
            single = actions[:, i, :]
            out = self._postprocessor(single)
            arr = out.cpu().numpy() if torch.is_tensor(out) else out
            actions_list.append(arr)
        chunk_np = np.stack(actions_list, axis=0)
        if chunk_np.ndim == 3:
            chunk_np = chunk_np[:, 0, :]
        chunk_np = np.asarray(chunk_np, dtype=np.float32)

        features_out = None
        if return_features and features:
            feat_np = _features_to_numpy(features)
            decoder_mean = feat_np.get("decoder_mean")
            features_out = BackboneFeatures(
                observation_embedding=feat_np.get("encoder_latent_token"),
                action_chunk_embedding=decoder_mean,
                raw={k: v for k, v in feat_np.items()},
            )
            if features_out.raw and "action_chunk_mean" not in features_out.raw and decoder_mean is not None:
                features_out.raw["action_chunk_mean"] = decoder_mean

        proposal = ChunkProposal(
            actions=chunk_np,
            step_index=self._step_index,
            features=features_out,
        )
        self._step_index += 1
        return proposal
