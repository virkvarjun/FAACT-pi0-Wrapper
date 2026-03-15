# FACT: Failure-Aware Chunk Transformer

PhD-level research prototype: a failure-aware / risk-aware supervisory wrapper around π₀.5 (Physical Intelligence backbone).

## Overview

FACT monitors candidate action chunks from π₀.5, predicts imminent failure risk, decides whether to continue/intervene/replan, and logs the full pipeline for research experiments.

## Architecture

- **backbone/**: π₀.5 policy wrapper (`Pi05PolicyWrapper`), unified `propose_chunk(obs)` interface
- **fact/**: Failure predictor (MLP baseline, temporal transformer)
- **governor/**: Runtime supervisor (threshold-based execute/reject)
- **data/**: Rollout logging, dataset builder for FACT training
- **training/**: Train FACT, calibration (temperature scaling, threshold sweep)
- **evaluation/**: Offline (AUROC, ECE, F1) and online metrics

## Setup

```bash
cd fact
pip install -e .

# For π₀.5 (LeRobot): install from Research
pip install -e /path/to/Research/lerobot
```

## Quick Start

### 1. Collect rollouts (stub without PI05 checkpoint)

```bash
python -m fact.scripts.collect_rollouts --output_dir data/rollouts --num_episodes 20
```

### 2. Build dataset

```bash
python -m fact.scripts.build_fact_dataset \
  --rollout_path data/rollouts/rollout.jsonl \
  --output_path data/fact_dataset.npz \
  --failure_horizon_k 5
```

### 3. Train FACT

```bash
python -m fact.scripts.train_fact \
  --dataset data/fact_dataset.npz \
  --output_dir checkpoints/fact \
  --epochs 50
```

### 4. Calibrate & evaluate

```bash
python -m fact.scripts.calibrate_fact
python -m fact.scripts.eval_fact_offline
```

### 5. Run wrapped policy

```bash
python -m fact.scripts.run_wrapped_policy \
  --fact_checkpoint checkpoints/fact/best_model.pt \
  --risk_threshold 0.5
```

## With π₀.5 checkpoint

```bash
# Collect from real PI05
python -m fact.scripts.collect_rollouts \
  --checkpoint /path/to/pi05_checkpoint \
  --output_dir data/rollouts
```

## Current limitations

- **PI05 internal features**: π₀.5 does not expose intermediate embeddings. FACT uses `action_chunk_mean` as the primary feature. Future: add hooks or `return_features` to PI05 for richer inputs.
- **Simulation**: Stub backbone returns zeros; use gym-aloha or similar for real rollouts.
- **Calibration**: Basic temperature scaling; conformal methods not yet implemented.
- **Recovery**: Governor only supports reject-and-replan; no fallback primitives.

## Research extensions

- Integrate with gym-aloha / DROID for real rollouts
- Add observation embeddings (e.g. ResNet features) as FACT input
- Conformal risk calibration
- Multi-candidate chunk scoring (re-sample N, pick lowest risk)
