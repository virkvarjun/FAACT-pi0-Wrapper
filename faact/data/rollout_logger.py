"""Rollout logging for trajectory collection."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from faact.data.schemas import ChunkRecord, EpisodeRecord, RolloutLog

logger = logging.getLogger(__name__)


def _serialize_chunk(chunk: ChunkRecord) -> dict[str, Any]:
    """Serialize chunk to JSON-serializable dict."""
    d: dict[str, Any] = {
        "episode_id": chunk.episode_id,
        "chunk_index": chunk.chunk_index,
        "step_index": chunk.step_index,
        "timestep": chunk.timestep,
        "y_fail_within_k_chunks": chunk.y_fail_within_k_chunks,
        "y_episode_fail": chunk.y_episode_fail,
        "y_intervention_good": chunk.y_intervention_good,
        "task_id": chunk.task_id,
        "scene_id": chunk.scene_id,
        "intervention_triggered": chunk.intervention_triggered,
        "replan_attempt": chunk.replan_attempt,
    }
    if chunk.action_chunk_mean is not None:
        d["action_chunk_mean"] = chunk.action_chunk_mean.tolist()
    if chunk.action_chunk is not None:
        d["action_chunk"] = chunk.action_chunk.tolist()
    if chunk.observation_embedding is not None:
        d["observation_embedding"] = chunk.observation_embedding.tolist()
    for k, v in chunk.raw_features.items():
        if isinstance(v, np.ndarray):
            d[f"feat_{k}"] = v.tolist()
    return d


def _serialize_episode(ep: EpisodeRecord) -> dict[str, Any]:
    d: dict[str, Any] = {
        "episode_id": ep.episode_id,
        "task_id": ep.task_id,
        "scene_id": ep.scene_id,
        "success": ep.success,
        "failure_step": ep.failure_step,
        "total_steps": ep.total_steps,
        "interventions": ep.interventions,
        "replans": ep.replans,
        "metadata": ep.metadata,
        "chunks": [_serialize_chunk(c) for c in ep.chunk_records],
    }
    return d


class RolloutLogger:
    """Log rollouts to disk for later FAACT training."""

    def __init__(
        self,
        output_dir: str | Path,
        run_id: str = "rollout",
        save_format: str = "jsonl",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.save_format = save_format
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._current_episode: EpisodeRecord | None = None
        self._episodes: list[EpisodeRecord] = []

    def start_episode(
        self,
        episode_id: int,
        task_id: str,
        scene_id: str = "",
    ) -> None:
        """Start a new episode."""
        self._current_episode = EpisodeRecord(
            episode_id=episode_id,
            task_id=task_id,
            scene_id=scene_id,
            success=False,
            failure_step=None,
            total_steps=0,
            chunk_records=[],
            interventions=[],
        )

    def log_chunk(
        self,
        chunk_record: ChunkRecord,
    ) -> None:
        """Append a chunk record to current episode."""
        if self._current_episode is None:
            raise RuntimeError("Call start_episode before log_chunk")
        self._current_episode.chunk_records.append(chunk_record)
        self._current_episode.total_steps += len(chunk_record.action_chunk) if chunk_record.action_chunk is not None else 0
        if chunk_record.intervention_triggered:
            self._current_episode.interventions.append(chunk_record.chunk_index)

    def end_episode(
        self,
        success: bool,
        failure_step: int | None = None,
        replans: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Finalize current episode."""
        if self._current_episode is None:
            raise RuntimeError("Call start_episode before end_episode")
        self._current_episode.success = success
        self._current_episode.failure_step = failure_step
        self._current_episode.replans = replans
        if metadata:
            self._current_episode.metadata.update(metadata)
        self._episodes.append(self._current_episode)
        self._current_episode = None

    def save(
        self,
        rollout_log: RolloutLog | None = None,
    ) -> Path:
        """Persist logged episodes to disk."""
        if rollout_log is None:
            rollout_log = RolloutLog(
                run_id=self.run_id,
                backbone_checkpoint="",
                task_ids=[],
                episodes=self._episodes,
            )

        out_path = self.output_dir / f"{self.run_id}.jsonl"
        with open(out_path, "w") as f:
            for ep in rollout_log.episodes:
                f.write(json.dumps(_serialize_episode(ep)) + "\n")

        # Also save summary
        summary_path = self.output_dir / f"{self.run_id}_summary.json"
        summary = {
            "run_id": rollout_log.run_id,
            "backbone_checkpoint": rollout_log.backbone_checkpoint,
            "n_episodes": len(rollout_log.episodes),
            "n_success": sum(1 for e in rollout_log.episodes if e.success),
            "n_chunks": sum(len(e.chunk_records) for e in rollout_log.episodes),
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Saved rollout to %s (%d episodes)", out_path, len(rollout_log.episodes))
        return out_path
