#!/usr/bin/env bash
set -euo pipefail

# Matched native control for train_task3_context_q_adapter.sh.
# It starts from the same native QGF checkpoint, freezes the flow actor, and
# continues only native Q/V updates for the same 30k globally paired batches.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${1:-1}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/exp/task3_native_qv_continue}"
DATASET_DIR="${OGBENCH_DATA_DIR:?Set OGBENCH_DATA_DIR to the OGBench root}"
NATIVE_CHECKPOINT="${NATIVE_CHECKPOINT:?Set NATIVE_CHECKPOINT to a native QGF run directory}"
NATIVE_EPOCH="${NATIVE_EPOCH:-500000}"
FINETUNE_STEPS="${FINETUNE_STEPS:-30000}"
OFFLINE_STEPS="$((NATIVE_EPOCH + FINETUNE_STEPS))"
MUJOCO_PYTHONPATH="${MUJOCO_PYTHONPATH:-/home/lrh/qgf-native/mujoco_alt_381}"

if [[ ! -f "${NATIVE_CHECKPOINT}/params_${NATIVE_EPOCH}.pkl" ]]; then
  echo "Missing native checkpoint: ${NATIVE_CHECKPOINT}/params_${NATIVE_EPOCH}.pkl" >&2
  exit 1
fi

cd "$ROOT"
PYTHONPATH="$MUJOCO_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" exec python main.py \
  --agent=agents/qgf_qv_finetune.py \
  --env_name=cube-triple-play-singletask-task3-v0 \
  --ogbench_dataset_dir="${DATASET_DIR}/cube-triple-play-100m-v0/" \
  --seed="$SEED" \
  --save_dir="$SAVE_ROOT" \
  --restore_path="$NATIVE_CHECKPOINT" \
  --restore_epoch="$NATIVE_EPOCH" \
  --offline_steps="$OFFLINE_STEPS" \
  --dataset_replace_interval=1000 \
  --deterministic_batch_indices=true \
  --agent.batch_size=1024 \
  --agent.action_chunking=True \
  --agent.horizon_length=5 \
  --agent.discount=0.999 \
  --agent.actor_hidden_dims='(1024,1024,1024,1024)' \
  --agent.value_network_kwargs.hidden_dims='(1024,1024,1024,1024)' \
  --eval_interval=0 \
  --save_interval=5000 \
  --guidance_weights=0.0,0.004,0.008,0.01,0.02,0.04,0.06,0.08 \
  --wandb_offline=true \
  --wandb_project=context-qgf-adapter \
  --wandb_run_group="task3_native_qv_continue_seed$(printf '%02d' "$SEED")"
