"""ACT policy wrapper for FAACT backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from faact.backbone.base import BackboneFeatures, ChunkProposal, BackbonePolicyWrapper
from faact.backbone.features import derive_action_features, tensor_features_to_numpy

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

def _predict_act_with_features(policy, obs_processed, use_dropout: bool = False):
    """Predict an ACT chunk and expose a stable feature payload."""
    if hasattr(policy, "predict_action_chunk_with_features") and not use_dropout:
        return policy.predict_action_chunk_with_features(obs_processed)

    from lerobot.utils.constants import OBS_IMAGES

    batch = dict(obs_processed)
    if getattr(policy, "config", None) and getattr(policy.config, "image_features", None):
        batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]

    was_training = policy.training
    if use_dropout:
        policy.train()

    with torch.inference_mode():
        try:
            out = policy.model(batch, return_features=True)
        except TypeError:
            out = None
        else:
            actions = out[0]
            features = out[2] if len(out) >= 3 else {}
            if use_dropout and not was_training:
                policy.eval()
            return actions, features

        captured = {}

        def make_hook(name):
            def hook(_module, _inputs, output):
                captured[name] = output.detach()

            return hook

        handles = []
        if hasattr(policy.model, "encoder"):
            handles.append(policy.model.encoder.register_forward_hook(make_hook("encoder_out")))
        if hasattr(policy.model, "decoder"):
            handles.append(policy.model.decoder.register_forward_hook(make_hook("decoder_out")))

        try:
            actions = policy.model(batch)[0]
        finally:
            for handle in handles:
                handle.remove()
            if use_dropout and not was_training:
                policy.eval()

        batch_size = actions.shape[0]
        cfg = getattr(policy.model, "config", policy.config)
        latent_dim = getattr(cfg, "latent_dim", 32)
        dim_model = getattr(cfg, "dim_model", 512)

        def _transpose_if_sequence(value):
            if value is not None and value.dim() == 3:
                return value.transpose(0, 1)
            return value

        enc = _transpose_if_sequence(captured.get("encoder_out"))
        dec = _transpose_if_sequence(captured.get("decoder_out"))
        features = {
            "latent_sample": torch.zeros(batch_size, latent_dim, device=actions.device, dtype=actions.dtype),
            "encoder_out": enc
            if enc is not None
            else torch.zeros(batch_size, 1, dim_model, device=actions.device, dtype=actions.dtype),
            "decoder_out": dec
            if dec is not None
            else torch.zeros(batch_size, actions.shape[1], dim_model, device=actions.device, dtype=actions.dtype),
        }
        return actions, features


class ACTPolicyWrapper(BackbonePolicyWrapper):
    """Wrapper around LeRobot ACT for chunk-level inference."""

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
        use_dropout = bool((context or {}).get("use_dropout", False))

        actions, features = _predict_act_with_features(self._policy, batch, use_dropout=use_dropout)
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
        if return_features:
            feat_np = tensor_features_to_numpy(features)
            feat_np.update(derive_action_features(chunk_np, chunk_step_idx=0))
            decoder_mean = feat_np.get("decoder_mean")
            features_out = BackboneFeatures(
                observation_embedding=feat_np.get("encoder_latent_token"),
                action_chunk_embedding=decoder_mean,
                raw=feat_np,
            )
            if features_out.raw is not None:
                features_out.raw.setdefault("action_chunk_mean", chunk_np.mean(axis=0).astype(np.float32))

        proposal = ChunkProposal(
            actions=chunk_np,
            step_index=self._step_index,
            features=features_out,
        )
        self._step_index += 1
        return proposal
