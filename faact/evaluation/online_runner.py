"""Shared online episode runner for FAACT-wrapped policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from faact.backbone.base import BackbonePolicyWrapper
from faact.backbone.features import chunk_to_numpy, merge_feature_dicts
from failure_prediction.interfaces import InterventionPolicy, RiskScorer
from failure_prediction.utils.success_inference import infer_episode_outcome

try:
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
except ImportError:
    ACTTemporalEnsembler = None


FrameCaptureFn = Callable[[Any, dict | None], np.ndarray]


@dataclass
class EpisodeRunnerConfig:
    mode: str = "baseline"
    num_candidate_chunks: int = 5
    obs_noise_std: float = 0.03
    switch_margin: float = 0.0
    replan_interval: int | None = None
    candidate_source: str = "obs_noise"
    action_noise_std: float = 0.05
    action_noise_prefix_steps: int = 10
    task_desc: str | None = None
    score_every_step: bool = False
    temporal_ensemble_coeff: float | None = None
    cooldown_steps: int = 0
    max_interventions_per_episode: int | None = None
    boundary_only_intervention: bool = False
    min_candidate_l2_to_baseline: float = 0.0


def add_obs_noise(obs_dict: dict, noise_std: float = 0.03, rng: np.random.Generator | None = None) -> dict:
    """Add pixel-space noise to observations for candidate diversification."""
    if rng is None:
        rng = np.random.default_rng()

    out = dict(obs_dict)
    if "pixels" not in out:
        return out

    scale = 255.0 * noise_std
    if isinstance(out["pixels"], dict):
        out["pixels"] = {
            key: np.clip(
                np.asarray(value, dtype=np.float32)
                + rng.normal(0.0, scale, np.asarray(value).shape).astype(np.float32),
                0,
                255,
            ).astype(np.uint8)
            for key, value in out["pixels"].items()
        }
    else:
        arr = np.asarray(out["pixels"], dtype=np.float32)
        out["pixels"] = np.clip(arr + rng.normal(0.0, scale, arr.shape), 0, 255).astype(np.uint8)
    return out


def add_action_noise(
    action_chunk: np.ndarray,
    noise_std: float = 0.05,
    prefix_steps: int = 10,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Perturb only the early action prefix to search for local corrections."""
    if rng is None:
        rng = np.random.default_rng()
    candidate = np.asarray(action_chunk, dtype=np.float32).copy()
    prefix = min(prefix_steps, candidate.shape[0])
    if prefix <= 0:
        return candidate
    candidate[:prefix] += rng.normal(0.0, noise_std, size=candidate[:prefix].shape).astype(np.float32)
    return candidate


def compute_candidate_diversity(
    baseline_chunk: np.ndarray,
    candidate_chunks: list[np.ndarray],
) -> dict[str, float | list[float]]:
    """Measure how much candidate chunks differ from baseline and each other."""
    if not candidate_chunks:
        return {
            "candidate_l2_to_baseline": [],
            "candidate_l2_to_baseline_mean": 0.0,
            "candidate_l2_to_baseline_min": 0.0,
            "candidate_l2_to_baseline_max": 0.0,
            "candidate_pairwise_l2_mean": 0.0,
            "candidate_pairwise_l2_max": 0.0,
        }

    baseline_flat = baseline_chunk.reshape(-1)
    dists_to_baseline = [
        float(np.linalg.norm(candidate.reshape(-1) - baseline_flat)) for candidate in candidate_chunks
    ]

    pairwise_dists = []
    for idx, first in enumerate(candidate_chunks):
        first_flat = first.reshape(-1)
        for second in candidate_chunks[idx + 1 :]:
            pairwise_dists.append(float(np.linalg.norm(first_flat - second.reshape(-1))))

    return {
        "candidate_l2_to_baseline": dists_to_baseline,
        "candidate_l2_to_baseline_mean": float(np.mean(dists_to_baseline)),
        "candidate_l2_to_baseline_min": float(np.min(dists_to_baseline)),
        "candidate_l2_to_baseline_max": float(np.max(dists_to_baseline)),
        "candidate_pairwise_l2_mean": float(np.mean(pairwise_dists)) if pairwise_dists else 0.0,
        "candidate_pairwise_l2_max": float(np.max(pairwise_dists)) if pairwise_dists else 0.0,
    }


def _proposal_context(task_desc: str | None = None, use_dropout: bool = False) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if task_desc:
        context["task"] = task_desc
    if use_dropout:
        context["use_dropout"] = True
    return context


def _make_temporal_ensembler(config: EpisodeRunnerConfig, backbone: BackbonePolicyWrapper):
    if config.temporal_ensemble_coeff is None or ACTTemporalEnsembler is None:
        return None
    return ACTTemporalEnsembler(config.temporal_ensemble_coeff, backbone.chunk_size)


def _candidate_mode(config: EpisodeRunnerConfig, candidate_idx: int) -> str:
    if config.candidate_source != "hybrid":
        return config.candidate_source
    modes = ["obs_noise", "action_noise", "obs_noise_dropout"]
    return modes[candidate_idx % len(modes)]


def _float_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _float_max(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


def run_episode(
    env,
    backbone: BackbonePolicyWrapper,
    rng: np.random.Generator | None = None,
    risk_scorer: RiskScorer | None = None,
    intervention_policy: InterventionPolicy | None = None,
    config: EpisodeRunnerConfig | None = None,
    capture_frames: bool = False,
    frame_fn: FrameCaptureFn | None = None,
) -> tuple[dict[str, Any], list[np.ndarray] | None]:
    """Run a single episode using the shared FAACT runtime."""
    config = config or EpisodeRunnerConfig()
    if rng is None:
        rng = np.random.default_rng()

    backbone.reset(task_spec=config.task_desc)
    raw_obs, _info = env.reset(seed=int(rng.integers(0, 2**31)))
    frames = [frame_fn(env, raw_obs)] if capture_frames and frame_fn is not None else None

    current_chunk: np.ndarray | None = None
    current_features_raw: dict[str, np.ndarray] = {}
    chunk_step_idx = 0
    last_risk_score = None
    last_alarmed = False
    last_intervention_step = None
    max_ep_steps = env.spec.max_episode_steps or 400
    effective_replan_interval = (
        backbone.chunk_size
        if config.replan_interval is None
        else max(1, min(config.replan_interval, backbone.chunk_size))
    )
    temporal_ensembler = _make_temporal_ensembler(config, backbone)

    episode_rewards = []
    episode_successes = []
    episode_dones = []
    episode_terminated = []
    episode_truncated = []
    interventions = []
    alarms = []
    alarm_events = []
    step_scores = [] if risk_scorer is not None else None

    done = False
    step = 0
    while not done and step < max_ep_steps:
        need_new_chunk = (
            temporal_ensembler is not None
            or current_chunk is None
            or chunk_step_idx >= effective_replan_interval
        )

        if need_new_chunk:
            proposal = backbone.propose_chunk(
                raw_obs,
                context=_proposal_context(task_desc=config.task_desc),
                return_features=True,
            )
            current_chunk = chunk_to_numpy(proposal.actions)
            current_features_raw = dict(proposal.features.raw) if proposal.features and proposal.features.raw else {}
            chunk_step_idx = 0

        runtime_features = merge_feature_dicts(current_features_raw, current_chunk, chunk_step_idx=chunk_step_idx)
        should_score = config.score_every_step or need_new_chunk
        if risk_scorer is not None and should_score:
            last_risk_score = risk_scorer.predict_step(runtime_features)
            decision = None
            if intervention_policy is not None:
                decision = intervention_policy.should_interrupt(
                    risk_score=last_risk_score,
                    step=step,
                    need_new_chunk=need_new_chunk,
                    accepted_interventions_so_far=sum(
                        1 for item in interventions if item.get("accepted", False)
                    ),
                    last_intervention_step=last_intervention_step,
                )
                last_alarmed = bool(decision.should_interrupt)
            else:
                last_alarmed = False

            if step_scores is not None:
                step_scores.append(
                    {
                        "step": step,
                        "risk_prob": last_risk_score.prob if last_risk_score is not None else None,
                        "risk_logit": last_risk_score.logit if last_risk_score is not None else None,
                        "at_chunk_boundary": need_new_chunk,
                        "chunk_step_idx": chunk_step_idx,
                        "alarmed": last_alarmed,
                        "decision_reason": decision.reason if decision is not None else "",
                        "decision_confidence": decision.confidence if decision is not None else 0.0,
                    }
                )

            if last_alarmed:
                alarm_events.append(
                    {
                        "step": step,
                        "risk_prob": last_risk_score.prob if last_risk_score is not None else None,
                        "risk_logit": last_risk_score.logit if last_risk_score is not None else None,
                        "at_chunk_boundary": need_new_chunk,
                        "chunk_step_idx": chunk_step_idx,
                        "decision_reason": decision.reason if decision is not None else "",
                        "decision_confidence": decision.confidence if decision is not None else 0.0,
                        "decision_details": decision.details if decision is not None else None,
                    }
                )

            if config.mode == "intervention" and last_alarmed and current_chunk is not None:
                baseline_chunk_np = current_chunk.copy()
                candidates = []
                candidate_chunk_nps = []
                candidate_sources = []
                for candidate_idx in range(config.num_candidate_chunks):
                    candidate_features_raw: dict[str, np.ndarray] = {}
                    candidate_mode = _candidate_mode(config, candidate_idx)
                    candidate_sources.append(candidate_mode)
                    if candidate_mode == "action_noise":
                        candidate_chunk = add_action_noise(
                            current_chunk,
                            noise_std=config.action_noise_std,
                            prefix_steps=config.action_noise_prefix_steps,
                            rng=rng,
                        )
                    else:
                        use_obs_noise = candidate_mode in {"obs_noise", "obs_noise_dropout"}
                        use_dropout = candidate_mode in {"dropout", "obs_noise_dropout"}
                        noisy_obs = (
                            add_obs_noise(raw_obs, noise_std=config.obs_noise_std, rng=rng)
                            if use_obs_noise
                            else raw_obs
                        )
                        proposal_cand = backbone.propose_chunk(
                            noisy_obs,
                            context=_proposal_context(task_desc=config.task_desc, use_dropout=use_dropout),
                            return_features=True,
                        )
                        candidate_chunk = chunk_to_numpy(proposal_cand.actions)
                        candidate_features_raw = (
                            dict(proposal_cand.features.raw) if proposal_cand.features and proposal_cand.features.raw else {}
                        )

                    feat_cand = merge_feature_dicts(candidate_features_raw, candidate_chunk, chunk_step_idx=0)
                    candidate_score = risk_scorer.predict_step(feat_cand)
                    if candidate_score is None:
                        continue

                    candidate_np = np.asarray(candidate_chunk, dtype=np.float32)
                    candidates.append((candidate_np, candidate_features_raw, candidate_score.prob, candidate_mode))
                    candidate_chunk_nps.append(candidate_np)

                candidate_risks = [candidate[2] for candidate in candidates]
                diversity = compute_candidate_diversity(baseline_chunk_np, candidate_chunk_nps)
                best_idx = int(np.argmin(candidate_risks)) if candidate_risks else -1
                baseline_risk = last_risk_score.prob if last_risk_score is not None else None
                best_risk = candidate_risks[best_idx] if candidate_risks else baseline_risk
                best_candidate_delta = (
                    best_risk - baseline_risk
                    if best_risk is not None and baseline_risk is not None
                    else None
                )
                best_candidate_l2 = (
                    float(diversity["candidate_l2_to_baseline"][best_idx])
                    if candidate_risks and best_idx >= 0
                    else 0.0
                )
                meets_margin = (
                    best_candidate_delta is not None
                    and best_candidate_delta <= -config.switch_margin
                )
                meets_diversity = best_candidate_l2 >= config.min_candidate_l2_to_baseline
                accepted = bool(candidate_risks) and meets_margin and meets_diversity
                rejection_reason = ""
                if not candidate_risks:
                    rejection_reason = "no_scored_candidates"
                elif best_candidate_delta is None:
                    rejection_reason = "missing_risk_delta"
                elif not meets_diversity:
                    rejection_reason = "insufficient_diversity"
                elif not meets_margin:
                    rejection_reason = "insufficient_improvement"

                if accepted:
                    best_chunk, best_features_raw, _, best_source = candidates[best_idx]
                    current_chunk = best_chunk
                    current_features_raw = best_features_raw
                    chunk_step_idx = 0
                    last_intervention_step = step
                else:
                    best_source = candidates[best_idx][3] if candidate_risks and best_idx >= 0 else ""

                interventions.append(
                    {
                        "step": step,
                        "attempted": True,
                        "accepted": accepted,
                        "rejection_reason": "" if accepted else rejection_reason,
                        "at_chunk_boundary": need_new_chunk,
                        "baseline_kept": not accepted,
                        "alarm_reason": decision.reason if decision is not None else "",
                        "alarm_confidence": decision.confidence if decision is not None else 0.0,
                        "alarm_details": decision.details if decision is not None else None,
                        "baseline_risk": baseline_risk,
                        "chosen_risk": best_risk if accepted else baseline_risk,
                        "best_candidate_risk": best_risk,
                        "risk_delta": (
                            best_candidate_delta
                            if accepted and best_candidate_delta is not None
                            else 0.0
                        ),
                        "best_candidate_risk_delta": best_candidate_delta,
                        "best_candidate_beats_baseline": (
                            best_candidate_delta is not None and best_candidate_delta < 0.0
                        ),
                        "best_candidate_meets_margin": bool(meets_margin),
                        "best_candidate_meets_diversity": bool(meets_diversity),
                        "best_candidate_l2_to_baseline": best_candidate_l2,
                        "candidate_sources": [candidate[3] for candidate in candidates],
                        "chosen_source": best_source if accepted else "",
                        "switch_margin": config.switch_margin,
                        "min_candidate_l2_to_baseline": config.min_candidate_l2_to_baseline,
                        "n_candidates": len(candidates),
                        "candidate_risks": candidate_risks,
                        "chosen_idx": best_idx if accepted else -1,
                        **diversity,
                    }
                )

        if risk_scorer is not None:
            alarms.append(last_alarmed)

        if temporal_ensembler is not None:
            ensemble_in = torch.from_numpy(np.asarray(current_chunk, dtype=np.float32)).unsqueeze(0)
            action = temporal_ensembler.update(ensemble_in)
            action_np = action.detach().cpu().numpy()
            if action_np.ndim == 2:
                action_np = action_np[0]
        else:
            action_np = np.asarray(current_chunk[chunk_step_idx], dtype=np.float32)

        raw_obs, reward, terminated, truncated, info = env.step(action_np)
        if frames is not None and frame_fn is not None:
            frames.append(frame_fn(env, raw_obs))

        success_this_step = bool(info.get("is_success", False))
        done = terminated or truncated

        episode_rewards.append(float(reward))
        episode_successes.append(success_this_step)
        episode_dones.append(done)
        episode_terminated.append(terminated)
        episode_truncated.append(truncated)

        chunk_step_idx += 1
        step += 1

    outcome = infer_episode_outcome(
        rewards=np.array(episode_rewards),
        successes=np.array(episode_successes),
        dones=np.array(episode_dones),
        terminated=np.array(episode_terminated),
        truncated=np.array(episode_truncated),
        env_name=env.unwrapped.spec.id if hasattr(env, "unwrapped") else "",
    )

    accepted_interventions = [item for item in interventions if item.get("accepted", False)]
    rejected_interventions = [item for item in interventions if not item.get("accepted", False)]
    better_candidate_attempts = [
        item for item in interventions if item.get("best_candidate_beats_baseline", False)
    ]
    alarm_risks = [
        float(event["risk_prob"])
        for event in alarm_events
        if event.get("risk_prob") is not None
    ]
    accepted_deltas = [
        float(item["risk_delta"])
        for item in accepted_interventions
        if item.get("risk_delta") is not None
    ]
    best_candidate_deltas = [
        float(item["best_candidate_risk_delta"])
        for item in interventions
        if item.get("best_candidate_risk_delta") is not None
    ]
    episode_summary = {
        "n_alarm_steps": len(alarm_events),
        "n_boundary_alarms": sum(1 for event in alarm_events if event.get("at_chunk_boundary", False)),
        "n_mid_chunk_alarms": sum(1 for event in alarm_events if not event.get("at_chunk_boundary", False)),
        "n_accepted_interventions": len(accepted_interventions),
        "n_rejected_interventions": len(rejected_interventions),
        "accepted_intervention_rate": (
            len(accepted_interventions) / len(interventions) if interventions else 0.0
        ),
        "better_candidate_available_rate": (
            len(better_candidate_attempts) / len(interventions) if interventions else 0.0
        ),
        "mean_alarm_risk": _float_mean(alarm_risks),
        "max_alarm_risk": _float_max(alarm_risks),
        "mean_accepted_risk_delta": _float_mean(accepted_deltas),
        "mean_best_candidate_delta": _float_mean(best_candidate_deltas),
        "mean_best_candidate_l2_to_baseline": _float_mean(
            [
                float(item["best_candidate_l2_to_baseline"])
                for item in interventions
                if item.get("best_candidate_l2_to_baseline") is not None
            ]
        ),
        "rejection_reason_counts": {
            reason: sum(1 for item in rejected_interventions if item.get("rejection_reason") == reason)
            for reason in sorted({item.get("rejection_reason", "") for item in rejected_interventions})
            if reason
        },
    }

    result = {
        "success": outcome["success"],
        "terminal_step": outcome["terminal_step"],
        "episode_length": step,
        "interventions": interventions,
        "n_interventions": sum(1 for item in interventions if item.get("accepted", False)),
        "n_intervention_attempts": len(interventions),
        "replan_interval": effective_replan_interval,
        "temporal_ensemble_coeff": config.temporal_ensemble_coeff,
        "alarms": alarms,
        "alarm_events": alarm_events,
        "step_scores": step_scores,
        "episode_summary": episode_summary,
    }
    return result, frames
