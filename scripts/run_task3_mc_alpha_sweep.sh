#!/usr/bin/env bash
set -euo pipefail

# Paired MC calibration sweep over guidance weights.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:?Provide an output root}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MUJOCO_PYTHONPATH="${MUJOCO_PYTHONPATH:-/home/lrh/qgf-native/mujoco_alt_381}"
ALPHAS_STRING="${ALPHAS:-0 0.004 0.008 0.01 0.02 0.04 0.06 0.08 0.1 0.12}"
EPISODES="${EPISODES:-10}"
MC_ROLLOUTS="${MC_ROLLOUTS:-8}"
QUERY_TRANSITIONS="${QUERY_TRANSITIONS:-20,40,60}"
CHUNK_SIZE="${CHUNK_SIZE:-2}"

read -r -a ALPHAS <<< "$ALPHAS_STRING"
mkdir -p "$OUTPUT_ROOT/logs"

native_checkpoint_for_seed() {
  case "$1" in
    1)
      echo "/home/lrh/qgf-native/artifacts/runs/native_qgf_cube_triple_task3_3seed/seed01/qgf-native-baseline/qgf-native-recovery-task3-seed01-resume200k/qgf-native-recovery-task3-seed01-resume200k_qgf_cube-triple-play-task3_seed01_c24aba51"
      ;;
    2)
      echo "/home/lrh/qgf-native/artifacts/runs/native_qgf_cube_triple_task3_3seed/seed02/recovery_400k_to_500k_noeval_20260908T0010Z/qgf-native-baseline/qgf-native-recovery-task3-seed02-resume400k-noeval/qgf-native-recovery-task3-seed02-resume400k-noeval_qgf_cube-triple-play-task3_seed02_c24aba51"
      ;;
    3)
      echo "/home/lrh/qgf-native/artifacts/runs/native_qgf_cube_triple_task3_3seed/seed03/qgf-native-baseline/qgf-native-recovery-task3-seed03-resume200k/qgf-native-recovery-task3-seed03-resume200k_qgf_cube-triple-play-task3_seed03_c24aba51"
      ;;
  esac
}

context_checkpoint_for_seed() {
  local seed="$1"
  local group="task3_context_q_seed$(printf '%02d' "$seed")"
  find "$ROOT/exp/task3_context_q_from_scratch" \
    -type f \
    -name "params_500000.pkl" \
    -path "*/${group}/*" \
    -printf '%h\n' \
    | sort \
    | tail -n 1
}

alpha_label() {
  echo "$1" | sed 's/\./p/g'
}

for ((offset = 0; offset < ${#ALPHAS[@]}; offset += CHUNK_SIZE)); do
  pids=()
  gpu_index=0
  for ((index = offset; index < offset + CHUNK_SIZE && index < ${#ALPHAS[@]}; index++)); do
    alpha="${ALPHAS[$index]}"
    label="$(alpha_label "$alpha")"
    for seed in 1 2 3; do
      native_checkpoint="$(native_checkpoint_for_seed "$seed")"
      context_checkpoint="$(context_checkpoint_for_seed "$seed")"
      output="$OUTPUT_ROOT/alpha_${label}/seed0${seed}"
      log="$OUTPUT_ROOT/logs/alpha_${label}_seed0${seed}.log"
      if [[ -f "$output/_SUCCESS" ]]; then
        continue
      fi
      nohup env \
        PYTHONPATH="$MUJOCO_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
        CUDA_VISIBLE_DEVICES="$gpu_index" \
        MUJOCO_GL=egl \
        "$PYTHON_BIN" "$ROOT/experiments/evaluate_task3_mc.py" \
          --env-name=cube-triple-play-singletask-task3-v0 \
          --native-checkpoint="$native_checkpoint" \
          --context-checkpoint="$context_checkpoint" \
          --epoch=500000 \
          --output-dir="$output" \
          --guidance-weight="$alpha" \
          --episodes="$EPISODES" \
          --query-transitions="$QUERY_TRANSITIONS" \
          --mc-rollouts="$MC_ROLLOUTS" \
        >"$log" 2>&1 &
      pids+=("$!")
      gpu_index=$((gpu_index + 1))
    done
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done

echo "MC_ALPHA_SWEEP_DONE"
