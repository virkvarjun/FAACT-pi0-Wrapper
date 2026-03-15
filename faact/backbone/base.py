"""Base interface for backbone policy wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BackboneFeatures:
    """Optional internal features from the backbone policy."""

    observation_embedding: np.ndarray | None = None
    action_chunk_embedding: np.ndarray | None = None
    raw: dict[str, np.ndarray] | None = None


@dataclass
class ChunkProposal:
    """Proposed action chunk from the backbone."""

    actions: np.ndarray  # (chunk_size, action_dim)
    step_index: int
    features: BackboneFeatures | None = None


class BackbonePolicyWrapper(ABC):
    """Abstract interface for backbone policy inference."""

    @abstractmethod
    def reset(self, task_spec: str | None = None) -> None:
        """Reset policy state (e.g. action queue) for a new episode."""
        pass

    @abstractmethod
    def propose_chunk(
        self,
        obs: dict[str, np.ndarray | dict[str, np.ndarray]],
        context: dict[str, Any] | None = None,
        return_features: bool = False,
    ) -> ChunkProposal:
        """Propose an action chunk given current observation.

        Args:
            obs: Observation dict (images, state, etc.).
            context: Optional context (task, scene_id, etc.).
            return_features: If True, populate ChunkProposal.features if available.

        Returns:
            ChunkProposal with actions and optional features.
        """
        pass

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Number of actions per chunk."""
        pass
