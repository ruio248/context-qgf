#!/usr/bin/env bash
set -euo pipefail

# Checkpoint-preserving Context-Q adapter finetuning for Task3.
#
# Example:
#   OGBENCH_DATA_DIR=/data/ogbench \
#   NATIVE_CHECKPOINT=/path/to/native/run \
#   ./scripts/train_task3_context_q_adapter.sh 1
#
# This starts from native params_500000.pkl, freezes the policy and native Q/V
# backbone, and optimizes only ContextInput projections plus the context encoder
# for 30k matched offline updates.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${1:-1}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/exp/task3_context_q_adapter}"
DATASET_DIR="${OGBENCH_DATA_DIR:?Set OGBENCH_DATA_DIR to the OGBench root}"
NATIVE_CHECKPOINT="${NATIVE_CHECKPOINT:?Set NATIVE_CHECKPOINT to a native QGF run directory}"
NATIVE_EPOCH="${NATIVE_EPOCH:-500000}"
FINETUNE_STEPS="${FINETUNE_STEPS:-30000}"
OFFLINE_STEPS="$((NATIVE_EPOCH + FINETUNE_STEPS))"
MUJOCO_PYTHONPATH="${MUJOCO_PYTHONPATH:-/home/lrh/qgf-native/mujoco_alt_381}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${NATIVE_CHECKPOINT}/params_${NATIVE_EPOCH}.pkl" ]]; then
  echo "Missing native checkpoint: ${NATIVE_CHECKPOINT}/params_${NATIVE_EPOCH}.pkl" >&2
  exit 1
fi

cd "$ROOT"
PYTHONPATH="$MUJOCO_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" exec "$PYTHON_BIN" main.py \
  --agent=agents/context_qgf_adapter.py \
  --env_name=cube-triple-play-singletask-task3-v0 \
  --ogbench_dataset_dir="${DATASET_DIR}/cube-triple-play-100m-v0/" \
  --seed="$SEED" \
  --save_dir="$SAVE_ROOT" \
  --native_adapter_restore_path="$NATIVE_CHECKPOINT" \
  --native_adapter_restore_epoch="$NATIVE_EPOCH" \
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
  --wandb_run_group="task3_context_q_adapter_seed$(printf '%02d' "$SEED")" \
  --agent.context_include_reward=true \
  --agent.context_length=20 \
  --agent.min_context_transitions=20 \
  --agent.context_value_init_scale=0.0
