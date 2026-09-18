#!/usr/bin/env bash
set -euo pipefail

# Clean Task3 Context-Q comparison runner.
#
# It reuses the three existing native QGF checkpoints, trains Context-Q from
# scratch for each requested seed, runs the paired MC-return calibration
# evaluator for that seed, and finally writes a small aggregate summary.
#
# Example:
#   OGBENCH_DATA_DIR=/home/lrh/qgf-native/data \
#   PYTHON_BIN=/home/lrh/qgf-native/venv/bin/python \
#   SEEDS="1 2 3" \
#   bash scripts/run_task3_clean_compare.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OGBENCH_DATA_DIR="${OGBENCH_DATA_DIR:-/home/lrh/qgf-native/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/lrh/qgf-native/venv/bin/python}"
SEEDS="${SEEDS:-1 2 3}"
EPOCH="${EPOCH:-500000}"
ALPHA="${ALPHA:-0.04}"
EPISODES="${EPISODES:-10}"
QUERY_TRANSITIONS="${QUERY_TRANSITIONS:-20,40,60}"
MC_ROLLOUTS="${MC_ROLLOUTS:-8}"
DISABLE_MULTICCD="${DISABLE_MULTICCD:-0}"
CONTEXT_SAVE_ROOT="${CONTEXT_SAVE_ROOT:-${ROOT}/exp/task3_context_q_from_scratch}"
MC_SAVE_ROOT="${MC_SAVE_ROOT:-${ROOT}/exp/task3_mc_clean}"
LOG_DIR="${LOG_DIR:-${ROOT}/exp/task3_clean_compare_logs}"
GPU_OFFSET="${GPU_OFFSET:-0}"
GPU_COUNT="${GPU_COUNT:-8}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
MUJOCO_PYTHONPATH="${MUJOCO_PYTHONPATH:-/home/lrh/qgf-native/mujoco_alt_381}"

mkdir -p "$LOG_DIR"
mkdir -p "$MC_SAVE_ROOT"

if [[ ! -d "${OGBENCH_DATA_DIR}/cube-triple-play-100m-v0" ]]; then
    echo "Missing OGBench dataset directory: ${OGBENCH_DATA_DIR}/cube-triple-play-100m-v0" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python binary not executable: $PYTHON_BIN" >&2
    exit 1
fi

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
        *)
            echo "Unsupported seed: $1" >&2
            return 1
            ;;
    esac
}

context_checkpoint_for_seed() {
    local seed="$1"
    local group="task3_context_q_seed$(printf '%02d' "$seed")"
    local checkpoint=""
    if [[ -d "$CONTEXT_SAVE_ROOT" ]]; then
        checkpoint="$(
            {
                find "$CONTEXT_SAVE_ROOT" \
                    -type f \
                    -name "params_${EPOCH}.pkl" \
                    -path "*/${group}/*" \
                    -printf '%T@ %h\n' \
                    | sort -n \
                    | tail -n 1 \
                    | cut -d' ' -f2-
            } || true
        )"
    fi
    printf '%s\n' "$checkpoint"
}

MULTICCD_ARGS=()
if [[ "$DISABLE_MULTICCD" == "1" ]]; then
    MULTICCD_ARGS+=(--disable-multiccd)
fi

native_checkpoints=()
context_checkpoints=()
train_pids=()

for seed in $SEEDS; do
    seed_index="$((seed - 1))"
    seed_padded="$(printf '%02d' "$seed")"
    native_checkpoint="$(native_checkpoint_for_seed "$seed")"
    if [[ ! -f "$native_checkpoint/params_${EPOCH}.pkl" ]]; then
        echo "Missing native QGF checkpoint: $native_checkpoint/params_${EPOCH}.pkl" >&2
        exit 1
    fi
    native_checkpoints["$seed_index"]="$native_checkpoint"
    context_checkpoints["$seed_index"]="$(context_checkpoint_for_seed "$seed")"
    gpu_index="$(( (seed - 1 + GPU_OFFSET) % GPU_COUNT ))"

    if [[ -z "${context_checkpoints[$seed_index]}" ]]; then
        train_log="$LOG_DIR/train_context_seed${seed_padded}.log"
        echo "Training Context-Q seed ${seed} on GPU ${gpu_index}" >&2
        (
            cd "$ROOT"
            CUDA_VISIBLE_DEVICES="$gpu_index" \
            MUJOCO_GL="$MUJOCO_GL" \
            PYTHONPATH="$MUJOCO_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
            "$PYTHON_BIN" main.py \
                --agent=agents/context_qgf.py \
                --env_name=cube-triple-play-singletask-task3-v0 \
                --ogbench_dataset_dir="${OGBENCH_DATA_DIR}/cube-triple-play-100m-v0/" \
                --seed="$seed" \
                --save_dir="$CONTEXT_SAVE_ROOT" \
                --offline_steps=500000 \
                --agent.batch_size=1024 \
                --agent.action_chunking=True \
                --agent.horizon_length=5 \
                --agent.discount=0.999 \
                --agent.actor_hidden_dims='(1024,1024,1024,1024)' \
                --agent.value_network_kwargs.hidden_dims='(1024,1024,1024,1024)' \
                --eval_interval=100000 \
                --eval_vecenv_size=1 \
                --save_interval=100000 \
                --guidance_weights=0.0,0.004,0.008,0.01,0.02,0.04,0.06,0.08 \
                --wandb_offline=true \
                --wandb_project=context-qgf-clean \
                --wandb_run_group="task3_context_q_seed${seed_padded}" \
                --agent.context_include_reward=true \
                --agent.context_length=20 \
                --agent.min_context_transitions=20
        ) >"$train_log" 2>&1 &
        train_pids["$seed_index"]=$!
    else
        echo "Reusing Context-Q seed ${seed}: ${context_checkpoints[$seed_index]}" >&2
        train_pids["$seed_index"]=""
    fi
done

training_failed=0
for seed in $SEEDS; do
    seed_index="$((seed - 1))"
    if [[ -n "${train_pids[$seed_index]}" ]]; then
        if ! wait "${train_pids[$seed_index]}"; then
            echo "Context-Q training failed for seed ${seed}; see $LOG_DIR/train_context_seed$(printf '%02d' "$seed").log" >&2
            training_failed=1
        fi
    fi
done
if [[ "$training_failed" -ne 0 ]]; then
    exit 1
fi

for seed in $SEEDS; do
    seed_index="$((seed - 1))"
    seed_padded="$(printf '%02d' "$seed")"
    native_checkpoint="${native_checkpoints[$seed_index]}"
    context_checkpoint="${context_checkpoints[$seed_index]}"
    gpu_index="$(( (seed - 1 + GPU_OFFSET) % GPU_COUNT ))"

    if [[ -z "$context_checkpoint" ]]; then
        context_checkpoint="$(context_checkpoint_for_seed "$seed")"
        if [[ -z "$context_checkpoint" ]]; then
            echo "Context-Q training finished without a params_${EPOCH}.pkl checkpoint for seed ${seed}; see $LOG_DIR/train_context_seed${seed_padded}.log" >&2
            exit 1
        fi
    fi

    mc_output="$MC_SAVE_ROOT/mc_seed${seed_padded}"
    if [[ -f "$mc_output/result.json" && -f "$mc_output/_SUCCESS" ]]; then
        echo "Reusing MC result seed ${seed}: $mc_output" >&2
    elif [[ -e "$mc_output" ]]; then
        echo "Partial MC output exists; move it aside before rerunning: $mc_output" >&2
        exit 1
    else
        eval_log="$LOG_DIR/eval_mc_seed${seed_padded}.log"
        echo "Running MC calibration seed ${seed} on GPU ${gpu_index}" >&2
        (
            cd "$ROOT"
            CUDA_VISIBLE_DEVICES="$gpu_index" \
            MUJOCO_GL="$MUJOCO_GL" \
            PYTHONPATH="$MUJOCO_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
            "$PYTHON_BIN" experiments/evaluate_task3_mc.py \
                --env-name=cube-triple-play-singletask-task3-v0 \
                --native-checkpoint="$native_checkpoint" \
                --context-checkpoint="$context_checkpoint" \
                --epoch="$EPOCH" \
                --output-dir="$mc_output" \
                --guidance-weight="$ALPHA" \
                --episodes="$EPISODES" \
                --query-transitions="$QUERY_TRANSITIONS" \
                --mc-rollouts="$MC_ROLLOUTS" \
                "${MULTICCD_ARGS[@]}"
        ) >"$eval_log" 2>&1
    fi
done

"$PYTHON_BIN" - "$MC_SAVE_ROOT" $SEEDS <<'PY'
import csv
import json
import os
import statistics
import sys

mc_root = sys.argv[1]
seeds = [int(seed) for seed in sys.argv[2:]]
rows = []
for seed in seeds:
    seed_dir = os.path.join(mc_root, f"mc_seed{seed:02d}")
    result_path = os.path.join(seed_dir, "result.json")
    if not os.path.isfile(result_path):
        raise RuntimeError(f"Missing MC result: {result_path}")
    with open(result_path) as stream:
        result = json.load(stream)
    rows.append(
        {
            "seed": seed,
            "native_mae": result["native"]["mae"],
            "context_mae": result["context"]["mae"],
            "native_rmse": result["native"]["rmse"],
            "context_rmse": result["context"]["rmse"],
            "delta_mc_mean": result["delta_mc"]["mean"],
            "delta_mc_ci95": result["delta_mc"]["ci95"],
            "query_count": result["query_count"],
        }
    )

aggregate_dir = os.path.join(mc_root, "aggregate")
os.makedirs(aggregate_dir, exist_ok=True)
aggregate = {
    "seeds": seeds,
    "delta_mc_mean_across_seeds": statistics.mean(
        row["delta_mc_mean"] for row in rows
    ),
    "native_mae_mean": statistics.mean(row["native_mae"] for row in rows),
    "context_mae_mean": statistics.mean(row["context_mae"] for row in rows),
    "native_rmse_mean": statistics.mean(row["native_rmse"] for row in rows),
    "context_rmse_mean": statistics.mean(row["context_rmse"] for row in rows),
}
with open(os.path.join(aggregate_dir, "result.json"), "w") as stream:
    json.dump({"aggregate": aggregate, "per_seed": rows}, stream, indent=2, sort_keys=True)
    stream.write("\n")

with open(os.path.join(aggregate_dir, "summary.csv"), "w", newline="") as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "seed",
            "native_mae",
            "context_mae",
            "native_rmse",
            "context_rmse",
            "delta_mc_mean",
            "delta_mc_ci95",
            "query_count",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(json.dumps(aggregate, indent=2, sort_keys=True))
PY

echo "Finished. Aggregate: ${MC_SAVE_ROOT}/aggregate/result.json" >&2
