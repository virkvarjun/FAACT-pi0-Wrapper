"""Shared feature extraction helpers for FAACT backbones."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

ACTION_PREFIX_STEPS = 10


def chunk_to_numpy(action_chunk: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a chunk tensor/array to `(chunk_size, action_dim)` float32."""
    chunk = action_chunk.detach().cpu().numpy() if isinstance(action_chunk, torch.Tensor) else np.asarray(action_chunk)
    if chunk.ndim == 3:
        chunk = chunk[0]
    return chunk.astype(np.float32, copy=False)


def tensor_features_to_numpy(features: dict[str, torch.Tensor] | None) -> dict[str, np.ndarray]:
    """Convert common policy feature tensors to a standard numpy schema."""
    if not features:
        return {}

    result: dict[str, np.ndarray] = {}
    for key, val in features.items():
        v = val.detach().cpu()
        if key == "encoder_out":
            result["encoder_latent_token"] = v[:, 0, :].squeeze(0).numpy()
        elif key == "decoder_out":
            result["decoder_mean"] = v.mean(dim=1).squeeze(0).numpy()
        elif key == "latent_sample":
            result["latent_sample"] = v.squeeze(0).numpy()
        elif key == "action_chunk_mean":
            result["action_chunk_mean"] = v.squeeze(0).numpy()
        else:
            result[key] = v.squeeze(0).numpy()
    return result


def derive_action_features(
    action_chunk: torch.Tensor | np.ndarray | None,
    chunk_step_idx: int = 0,
    prefix_steps: int = ACTION_PREFIX_STEPS,
) -> dict[str, np.ndarray]:
    """Derive boundary-time and in-flight action-prefix features from a chunk."""
    if action_chunk is None:
        return {}

    chunk = chunk_to_numpy(action_chunk)
    if len(chunk) == 0:
        return {}

    prefix_end = min(prefix_steps, len(chunk))
    prefix = chunk[:prefix_end]
    start_idx = int(np.clip(chunk_step_idx, 0, len(chunk) - 1))
    remaining = chunk[start_idx : start_idx + prefix_steps]
    if len(remaining) == 0:
        remaining = chunk[:1]
    if len(remaining) < prefix_steps:
        pad = np.repeat(remaining[-1:], prefix_steps - len(remaining), axis=0)
        remaining = np.concatenate([remaining, pad], axis=0)

    return {
        "action_first": chunk[0],
        f"action_prefix_mean_{prefix_steps}": prefix.mean(axis=0),
        f"action_prefix_flat_{prefix_steps}": prefix.reshape(-1),
        "action_remaining_first": remaining[0],
        f"action_remaining_prefix_mean_{prefix_steps}": remaining.mean(axis=0),
        f"action_remaining_prefix_flat_{prefix_steps}": remaining.reshape(-1),
    }


def merge_feature_dicts(
    raw_features: dict[str, Any] | None,
    action_chunk: torch.Tensor | np.ndarray | None,
    chunk_step_idx: int = 0,
) -> dict[str, np.ndarray]:
    """Combine raw backbone features with standard action-derived features."""
    result: dict[str, np.ndarray] = {}
    if raw_features:
        for key, value in raw_features.items():
            if value is None:
                continue
            result[key] = np.asarray(value, dtype=np.float32)
    result.update(derive_action_features(action_chunk, chunk_step_idx=chunk_step_idx))
    return result
