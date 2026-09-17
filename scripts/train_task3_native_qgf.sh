#!/usr/bin/env bash
set -euo pipefail

# Matched native QGF control for train_task3_context_q.sh.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${1:-1}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/exp/task3_native_qgf}"
DATASET_DIR="${OGBENCH_DATA_DIR:?Set OGBENCH_DATA_DIR to the OGBench root}"

cd "$ROOT"
exec python main.py \
  --agent=agents/qgf.py \
  --env_name=cube-triple-play-singletask-task3-v0 \
  --ogbench_dataset_dir="${DATASET_DIR}/cube-triple-play-100m-v0/" \
  --seed="$SEED" \
  --save_dir="$SAVE_ROOT" \
  --offline_steps=500000 \
  --agent.batch_size=1024 \
  --agent.action_chunking=True \
  --agent.horizon_length=5 \
  --agent.discount=0.999 \
  --agent.actor_hidden_dims='(1024,1024,1024,1024)' \
  --agent.value_network_kwargs.hidden_dims='(1024,1024,1024,1024)' \
  --eval_interval=100000 \
  --save_interval=100000 \
  --guidance_weights=0.0,0.004,0.008,0.01,0.02,0.04,0.06,0.08 \
  --wandb_offline=true
