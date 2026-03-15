"""Stub backbone for when π₀.5 / LeRobot is unavailable."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from fact.backbone.base import BackboneFeatures, ChunkProposal, BackbonePolicyWrapper

logger = logging.getLogger(__name__)


class StubBackboneWrapper(BackbonePolicyWrapper):
    """Placeholder backbone when PI05/LeRobot is not installed.

    Returns zero actions. Use for testing the FACT pipeline without a real policy.
    """

    def __init__(
        self,
        chunk_size: int = 50,
        action_dim: int = 14,
    ) -> None:
        self._chunk_size = chunk_size
        self._action_dim = action_dim
        self._step_index = 0
        logger.warning(
            "Using StubBackboneWrapper (zero actions). "
            "Install lerobot for real PI05: pip install -e path/to/Research/lerobot"
        )

    def reset(self, task_spec: str | None = None) -> None:
        self._step_index = 0

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def propose_chunk(
        self,
        obs: dict[str, np.ndarray | dict[str, np.ndarray]],
        context: dict[str, Any] | None = None,
        return_features: bool = False,
    ) -> ChunkProposal:
        chunk = np.zeros((self._chunk_size, self._action_dim), dtype=np.float32)
        features = None
        if return_features:
            features = BackboneFeatures(
                raw={"action_chunk_mean": chunk.mean(axis=0), "stub": np.array([1.0])}
            )
        proposal = ChunkProposal(
            actions=chunk,
            step_index=self._step_index,
            features=features,
        )
        self._step_index += 1
        return proposal
