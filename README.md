# FAACT: Failure-Aware Action Chunk Transformer

PhD-level research prototype: a failure-aware / risk-aware supervisory wrapper around π₀.5 (Physical Intelligence backbone).

**Path:** If your cwd is `Research/`, use `cd faact` (NOT `cd Research/faact`).

## Overview

FAACT monitors candidate action chunks from π₀.5, predicts imminent failure risk, decides whether to continue/intervene/replan, and logs the full pipeline for research experiments.

## Architecture

- **backbone/**: π₀.5 policy wrapper (`Pi05PolicyWrapper`), unified `propose_chunk(obs)` interface
- **predictor/**: Failure predictor (MLP baseline, temporal transformer)
- **governor/**: Runtime supervisor (threshold-based execute/reject)
- **data/**: Rollout logging, dataset builder for FAACT training
- **training/**: Train FAACT, calibration (temperature scaling, threshold sweep)
- **evaluation/**: Offline (AUROC, ECE, F1) and online metrics

## Setup

From the **Research** directory (parent of faact):

```bash
cd faact
pip install -e .

# For π₀.5 (LeRobot): install from Research
pip install -e ../lerobot
```

## Quick Start

**All commands from the `faact/` directory** (after `cd faact`):

```bash
cd faact
pip install -e .
python -m faact.scripts.collect_rollouts --num_episodes 20
```

Or run the full pipeline:

```bash
cd faact
./scripts/run_full_pipeline.sh
```

### 1. Collect rollouts (stub without PI05 checkpoint)

```bash
cd faact
python -m faact.scripts.collect_rollouts --num_episodes 20
```

### 2. Build dataset

```bash
python -m faact.scripts.build_fact_dataset \
  --rollout_path data/rollouts/rollout.jsonl \
  --output_path data/fact_dataset.npz \
  --failure_horizon_k 5
```

### 3. Train FAACT

```bash
python -m faact.scripts.train_fact \
  --dataset data/fact_dataset.npz \
  --output_dir checkpoints/fact \
  --epochs 50
```

### 4. Calibrate & evaluate

```bash
python -m faact.scripts.calibrate_fact
python -m faact.scripts.eval_fact_offline
```

### 5. Run wrapped policy

```bash
python -m faact.scripts.run_wrapped_policy \
  --fact_checkpoint checkpoints/fact/best_model.pt \
  --risk_threshold 0.5
```

## With π₀.5 checkpoint

```bash
# Collect from real PI05
python -m faact.scripts.collect_rollouts \
  --checkpoint /path/to/pi05_checkpoint \
  --output_dir data/rollouts
```

## Current limitations

- **PI05 internal features**: π₀.5 does not expose intermediate embeddings. FAACT uses `action_chunk_mean` as the primary feature. Future: add hooks or `return_features` to PI05 for richer inputs.
- **Simulation**: Stub backbone returns zeros; use gym-aloha or similar for real rollouts.
- **Calibration**: Basic temperature scaling; conformal methods not yet implemented.
- **Recovery**: Governor only supports reject-and-replan; no fallback primitives.

## Research extensions

- Integrate with gym-aloha / DROID for real rollouts
- Add observation embeddings (e.g. ResNet features) as FAACT input
- Conformal risk calibration
- Multi-candidate chunk scoring (re-sample N, pick lowest risk)
