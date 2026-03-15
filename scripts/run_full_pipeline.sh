#!/bin/bash
# Run full FAACT pipeline from the faact/ directory.
# Usage: ./scripts/run_full_pipeline.sh
# Or from Research: cd faact && ./scripts/run_full_pipeline.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Installing faact package ==="
pip install -e .

echo "=== 1. Collect rollouts ==="
python -m faact.scripts.collect_rollouts --num_episodes 20

echo "=== 2. Build dataset ==="
python -m faact.scripts.build_fact_dataset --rollout_path data/rollouts/rollout.jsonl

echo "=== 3. Train FAACT ==="
python -m faact.scripts.train_fact --epochs 50

echo "=== 4. Calibrate ==="
python -m faact.scripts.calibrate_fact

echo "=== 5. Offline eval ==="
python -m faact.scripts.eval_fact_offline

echo "=== 6. Run wrapped policy ==="
python -m faact.scripts.run_wrapped_policy --risk_threshold 0.5

echo "Done."
