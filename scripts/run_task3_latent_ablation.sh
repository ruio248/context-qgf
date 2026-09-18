#!/usr/bin/env bash
set -euo pipefail

# Compare posterior-mean and posterior-sample latent inference while holding
# the actor fixed to the native QGF actor.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:?Provide an output root}"
NATIVE_CHECKPOINT="${NATIVE_CHECKPOINT:?Set NATIVE_CHECKPOINT}"
CONTEXT_CHECKPOINT="${CONTEXT_CHECKPOINT:?Set CONTEXT_CHECKPOINT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MUJOCO_PYTHONPATH="${MUJOCO_PYTHONPATH:-/home/lrh/qgf-native/mujoco_alt_381}"

cd "$ROOT"
for mode in mean sample; do
  PYTHONPATH="$MUJOCO_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" experiments/evaluate_task3_closed_loop.py \
    --env-name=cube-triple-play-singletask-task3-v0 \
    --native-checkpoint="$NATIVE_CHECKPOINT" \
    --context-checkpoint="$CONTEXT_CHECKPOINT" \
    --epoch="${EPOCH:-500000}" \
    --output-dir="$OUTPUT_ROOT/latent_$mode" \
    --guidance-weight="${ALPHA:-0.04}" \
    --episodes="${EPISODES:-30}" \
    --context-actor-source=native \
    --latent-mode="$mode"
done
