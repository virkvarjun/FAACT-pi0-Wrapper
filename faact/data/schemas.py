"""Trajectory and rollout data schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ChunkRecord:
    """Per-chunk record for training FAACT."""

    episode_id: int
    chunk_index: int
    step_index: int
    timestep: int
    # Features for FAACT
    observation_embedding: np.ndarray | None = None
    action_chunk: np.ndarray | None = None
    action_chunk_mean: np.ndarray | None = None
    raw_features: dict[str, np.ndarray] = field(default_factory=dict)
    # Labels
    y_fail_within_k_chunks: int = 0
    y_episode_fail: int = 0
    y_intervention_good: int = 0
    # Metadata
    task_id: str = ""
    scene_id: str = ""
    intervention_triggered: bool = False
    replan_attempt: int = 0


@dataclass
class EpisodeRecord:
    """Per-episode record."""

    episode_id: int
    task_id: str
    scene_id: str
    success: bool
    failure_step: int | None
    total_steps: int
    chunk_records: list[ChunkRecord] = field(default_factory=list)
    interventions: list[int] = field(default_factory=list)
    replans: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutLog:
    """Full rollout log for an experiment run."""

    run_id: str
    backbone_checkpoint: str
    task_ids: list[str]
    episodes: list[EpisodeRecord] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
