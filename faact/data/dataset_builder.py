"""Build FAACT training dataset from rollout logs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from faact.data.schemas import ChunkRecord, EpisodeRecord

logger = logging.getLogger(__name__)


def load_rollout_log(path: str | Path) -> list[dict[str, Any]]:
    """Load episodes from jsonl rollout log."""
    path = Path(path)
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def _chunk_dict_to_record(d: dict[str, Any], episode_id: int) -> ChunkRecord:
    """Rebuild ChunkRecord from serialized dict."""
    action_chunk_mean = None
    if "action_chunk_mean" in d:
        action_chunk_mean = np.array(d["action_chunk_mean"], dtype=np.float32)
    action_chunk = None
    if "action_chunk" in d:
        action_chunk = np.array(d["action_chunk"], dtype=np.float32)
    obs_emb = None
    if "observation_embedding" in d:
        obs_emb = np.array(d["observation_embedding"], dtype=np.float32)
    raw = {k[5:]: np.array(v) for k, v in d.items() if k.startswith("feat_")}
    return ChunkRecord(
        episode_id=episode_id,
        chunk_index=d["chunk_index"],
        step_index=d["step_index"],
        timestep=d["timestep"],
        action_chunk_mean=action_chunk_mean,
        action_chunk=action_chunk,
        observation_embedding=obs_emb,
        raw_features=raw,
        y_fail_within_k_chunks=d.get("y_fail_within_k_chunks", 0),
        y_episode_fail=d.get("y_episode_fail", 0),
        y_intervention_good=d.get("y_intervention_good", 0),
        task_id=d.get("task_id", ""),
        scene_id=d.get("scene_id", ""),
        intervention_triggered=d.get("intervention_triggered", False),
        replan_attempt=d.get("replan_attempt", 0),
    )


def build_chunk_dataset(
    rollout_path: str | Path,
    failure_horizon_k: int = 5,
    feature_key: str = "action_chunk_mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) arrays for FAACT training.

    For each chunk, X = feature vector, y = 1 if failure within next K chunks.

    Args:
        rollout_path: Path to .jsonl rollout log.
        failure_horizon_k: Label y=1 if failure occurs within next K chunks.
        feature_key: Key to use for features (action_chunk_mean, observation_embedding, etc.).

    Returns:
        X: (n_samples, feature_dim) float32
        y: (n_samples,) int 0/1
    """
    episodes = load_rollout_log(rollout_path)
    X_list = []
    y_list = []

    for ep in episodes:
        success = ep["success"]
        chunks_data = ep.get("chunks", [])
        failure_step = ep.get("failure_step")
        n_chunks = len(chunks_data)

        for i, c in enumerate(chunks_data):
            rec = _chunk_dict_to_record(c, ep["episode_id"])
            feat = None
            if feature_key == "action_chunk_mean" and rec.action_chunk_mean is not None:
                feat = rec.action_chunk_mean
            elif feature_key == "observation_embedding" and rec.observation_embedding is not None:
                feat = rec.observation_embedding
            elif feature_key.startswith("feat_") and feature_key[5:] in rec.raw_features:
                feat = rec.raw_features[feature_key[5:]]
            if feat is None:
                continue

            # Label: failure within K chunks from this chunk
            y = 0
            if not success and failure_step is not None and n_chunks > 0:
                steps_per_chunk = max(1, ep["total_steps"] // n_chunks)
                fail_chunk_idx = failure_step // steps_per_chunk
                if 0 <= fail_chunk_idx - i <= failure_horizon_k:
                    y = 1

            X_list.append(feat)
            y_list.append(y)

    if not X_list:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y
