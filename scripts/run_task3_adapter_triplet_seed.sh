#!/usr/bin/env bash
set -euo pipefail

# Run the two matched 30k continuations for one source seed, then emit the
# requested three-arm actor-only MC and closed-loop result.  The baseline arm
# is the untouched native 500k checkpoint, so it needs no training process.
#
# Example on new_server_4090:
#   PYTHON_BIN=/home/lrh/qgf-native/venv/bin/python \
#   SAVE_ROOT=/data/lrh/context-q-adapter-ft \
#   bash scripts/run_task3_adapter_triplet_seed.sh 1 0 1 2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${1:?Provide seed: 1, 2, or 3}"
ADAPTER_GPU="${2:-0}"
QV_GPU="${3:-1}"
EVAL_GPU="${4:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${OGBENCH_DATA_DIR:-/home/lrh/qgf-native/data}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/exp/task3_adapter_triplet}"
FINETUNE_STEPS="${FINETUNE_STEPS:-30000}"
NATIVE_EPOCH="${NATIVE_EPOCH:-500000}"
FINAL_EPOCH="$((NATIVE_EPOCH + FINETUNE_STEPS))"
ALPHA="${ALPHA:-0.04}"
MC_EPISODES="${MC_EPISODES:-10}"
CLOSED_LOOP_EPISODES="${CLOSED_LOOP_EPISODES:-30}"
MC_ROLLOUTS="${MC_ROLLOUTS:-8}"
MUJOCO_PYTHONPATH="${MUJOCO_PYTHONPATH:-/home/lrh/qgf-native/mujoco_alt_381}"

case "$SEED" in
  1)
    NATIVE_CHECKPOINT="/home/lrh/qgf-native/artifacts/runs/native_qgf_cube_triple_task3_3seed/seed01/qgf-native-baseline/qgf-native-recovery-task3-seed01-resume200k/qgf-native-recovery-task3-seed01-resume200k_qgf_cube-triple-play-task3_seed01_c24aba51" ;;
  2)
    NATIVE_CHECKPOINT="/home/lrh/qgf-native/artifacts/runs/native_qgf_cube_triple_task3_3seed/seed02/recovery_400k_to_500k_noeval_20260908T0010Z/qgf-native-baseline/qgf-native-recovery-task3-seed02-resume400k-noeval/qgf-native-recovery-task3-seed02-resume400k-noeval_qgf_cube-triple-play-task3_seed02_c24aba51" ;;
  3)
    NATIVE_CHECKPOINT="/home/lrh/qgf-native/artifacts/runs/native_qgf_cube_triple_task3_3seed/seed03/qgf-native-baseline/qgf-native-recovery-task3-seed03-resume200k/qgf-native-recovery-task3-seed03-resume200k_qgf_cube-triple-play-task3_seed03_c24aba51" ;;
  *) echo "Seed must be 1, 2, or 3" >&2; exit 2 ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python binary is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$NATIVE_CHECKPOINT/params_${NATIVE_EPOCH}.pkl" ]]; then
  echo "Missing native checkpoint: $NATIVE_CHECKPOINT/params_${NATIVE_EPOCH}.pkl" >&2
  exit 1
fi
if [[ ! -d "$DATASET_DIR/cube-triple-play-100m-v0" ]]; then
  echo "Missing Task3 dataset: $DATASET_DIR/cube-triple-play-100m-v0" >&2
  exit 1
fi

RUN_ROOT="$SAVE_ROOT/seed$(printf '%02d' "$SEED")"
ADAPTER_ROOT="$RUN_ROOT/context_adapter"
QV_ROOT="$RUN_ROOT/native_qv_continuation"
EVAL_ROOT="$RUN_ROOT/evaluation"
LOG_ROOT="$RUN_ROOT/logs"
mkdir -p "$LOG_ROOT"

find_final_checkpoint() {
  local root="$1"
  find "$root" -type f -name "params_${FINAL_EPOCH}.pkl" -printf '%h\n' 2>/dev/null | sort | tail -n 1
}

adapter_checkpoint="$(find_final_checkpoint "$ADAPTER_ROOT")"
qv_checkpoint="$(find_final_checkpoint "$QV_ROOT")"

if [[ -z "$adapter_checkpoint" ]]; then
  if find "$ADAPTER_ROOT" -type f -name "params_${NATIVE_EPOCH}.pkl" -print -quit 2>/dev/null | grep -q .; then
    echo "Adapter has a prior initialization snapshot but no ${FINAL_EPOCH} checkpoint; use a new SAVE_ROOT rather than overwriting it." >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="$ADAPTER_GPU" PYTHON_BIN="$PYTHON_BIN" \
    OGBENCH_DATA_DIR="$DATASET_DIR" SAVE_ROOT="$ADAPTER_ROOT" \
    NATIVE_CHECKPOINT="$NATIVE_CHECKPOINT" NATIVE_EPOCH="$NATIVE_EPOCH" \
    FINETUNE_STEPS="$FINETUNE_STEPS" MUJOCO_PYTHONPATH="$MUJOCO_PYTHONPATH" \
    bash "$ROOT/scripts/train_task3_context_q_adapter.sh" "$SEED" \
    >"$LOG_ROOT/train_context_adapter.log" 2>&1 &
  adapter_pid=$!
else
  adapter_pid=""
fi

if [[ -z "$qv_checkpoint" ]]; then
  CUDA_VISIBLE_DEVICES="$QV_GPU" PYTHON_BIN="$PYTHON_BIN" \
    OGBENCH_DATA_DIR="$DATASET_DIR" SAVE_ROOT="$QV_ROOT" \
    NATIVE_CHECKPOINT="$NATIVE_CHECKPOINT" NATIVE_EPOCH="$NATIVE_EPOCH" \
    FINETUNE_STEPS="$FINETUNE_STEPS" MUJOCO_PYTHONPATH="$MUJOCO_PYTHONPATH" \
    bash "$ROOT/scripts/train_task3_native_qv_continue.sh" "$SEED" \
    >"$LOG_ROOT/train_native_qv.log" 2>&1 &
  qv_pid=$!
else
  qv_pid=""
fi

if [[ -n "$adapter_pid" ]]; then wait "$adapter_pid"; fi
if [[ -n "$qv_pid" ]]; then wait "$qv_pid"; fi
adapter_checkpoint="$(find_final_checkpoint "$ADAPTER_ROOT")"
qv_checkpoint="$(find_final_checkpoint "$QV_ROOT")"
if [[ -z "$adapter_checkpoint" || -z "$qv_checkpoint" ]]; then
  echo "A continuation failed; see $LOG_ROOT" >&2
  exit 1
fi
if [[ -e "$EVAL_ROOT" ]]; then
  echo "Evaluation output already exists: $EVAL_ROOT" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
MUJOCO_GL=egl \
PYTHONPATH="$MUJOCO_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" "$ROOT/experiments/evaluate_task3_adapter_triplet.py" \
  --env-name=cube-triple-play-singletask-task3-v0 \
  --frozen-native-checkpoint="$NATIVE_CHECKPOINT" \
  --native-qv-checkpoint="$qv_checkpoint" \
  --context-adapter-checkpoint="$adapter_checkpoint" \
  --frozen-native-epoch="$NATIVE_EPOCH" \
  --native-qv-epoch="$FINAL_EPOCH" \
  --context-adapter-epoch="$FINAL_EPOCH" \
  --output-dir="$EVAL_ROOT" \
  --guidance-weight="$ALPHA" \
  --continuation-guidance-weight=0 \
  --mc-episodes="$MC_EPISODES" \
  --closed-loop-episodes="$CLOSED_LOOP_EPISODES" \
  --mc-rollouts="$MC_ROLLOUTS" \
  >"$LOG_ROOT/evaluate_triplet.log" 2>&1

echo "Finished: $EVAL_ROOT/result.json" >&2
