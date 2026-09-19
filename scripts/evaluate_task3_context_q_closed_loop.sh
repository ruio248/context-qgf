#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   NATIVE_CHECKPOINT=/path/to/native/run \
#   CONTEXT_CHECKPOINT=/path/to/context/run \
#   ./scripts/evaluate_task3_context_q_closed_loop.sh /path/to/output
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:?Provide an output directory}"
NATIVE_CHECKPOINT="${NATIVE_CHECKPOINT:?Set NATIVE_CHECKPOINT}"
CONTEXT_CHECKPOINT="${CONTEXT_CHECKPOINT:?Set CONTEXT_CHECKPOINT}"
NATIVE_EPOCH="${NATIVE_EPOCH:-${EPOCH:-500000}}"
CONTEXT_EPOCH="${CONTEXT_EPOCH:-${EPOCH:-500000}}"
MUJOCO_PYTHONPATH="${MUJOCO_PYTHONPATH:-/home/lrh/qgf-native/mujoco_alt_381}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT"
MULTICCD_ARGS=()
if [[ "${DISABLE_MULTICCD:-0}" == "1" ]]; then
  MULTICCD_ARGS+=(--disable-multiccd)
fi
ACTOR_ARGS=()
if [[ "${CONTEXT_ACTOR_SOURCE:-context}" == "native" ]]; then
  ACTOR_ARGS+=(--context-actor-source=native)
fi

PYTHONPATH="$MUJOCO_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" experiments/evaluate_task3_closed_loop.py \
  --env-name=cube-triple-play-singletask-task3-v0 \
  --native-checkpoint="$NATIVE_CHECKPOINT" \
  --context-checkpoint="$CONTEXT_CHECKPOINT" \
  --native-epoch="$NATIVE_EPOCH" \
  --context-epoch="$CONTEXT_EPOCH" \
  --output-dir="$OUTPUT_DIR" \
  --guidance-weight="${ALPHA:-0.04}" \
  --episodes="${EPISODES:-30}" \
  "${MULTICCD_ARGS[@]}" \
  "${ACTOR_ARGS[@]}"
