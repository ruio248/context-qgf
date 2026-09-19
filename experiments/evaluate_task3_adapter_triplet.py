#!/usr/bin/env python3
"""Fair three-arm Task3 evaluation for checkpoint-preserving Context-Q.

The arms are a frozen native QGF checkpoint, a native Q/V-only continuation,
and a Context-Q adapter.  All three have the same frozen behavior actor.  MC
queries use a shared frozen-native prefix and independently test each arm's
guided *first* chunk; every continuation afterwards uses the unguided frozen
native actor.  Thus the MC return attributes a difference to the selected
first action instead of recursively mixing in a different Q-guidance policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from experiments.evaluate_task3_closed_loop import (
    make_paired_env,
    paired_bootstrap,
    rollout_episode,
)
from experiments.evaluate_task3_mc import (
    action_key,
    base_action,
    bookkeeping_state,
    continuation_return,
    execute_chunk,
    load_checkpoint,
    make_env,
    physics_state,
    restore_bookkeeping,
    restore_physics_state,
    tree_sha256,
)
from utils.context import pad_context_numpy, transition_token_numpy


ARM_NAMES = ("frozen_native", "native_qv_continuation", "context_adapter")


def parse_ints(value):
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="cube-triple-play-singletask-task3-v0")
    parser.add_argument("--frozen-native-checkpoint", required=True)
    parser.add_argument("--native-qv-checkpoint", required=True)
    parser.add_argument("--context-adapter-checkpoint", required=True)
    parser.add_argument("--frozen-native-epoch", type=int, default=500_000)
    parser.add_argument("--native-qv-epoch", type=int, default=530_000)
    parser.add_argument("--context-adapter-epoch", type=int, default=530_000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--guidance-weight", type=float, default=0.04)
    parser.add_argument(
        "--continuation-guidance-weight",
        type=float,
        default=0.0,
        help="Must remain 0 for actor-only MC continuation evaluation.",
    )
    parser.add_argument("--mc-episodes", type=int, default=10)
    parser.add_argument("--closed-loop-episodes", type=int, default=30)
    parser.add_argument("--query-transitions", type=parse_ints, default=(20, 40, 60))
    parser.add_argument("--mc-rollouts", type=int, default=8)
    parser.add_argument("--episode-seed-base", type=int, default=72_000_000)
    parser.add_argument("--action-seed-base", type=int, default=172_000_003)
    parser.add_argument("--continuation-seed-base", type=int, default=272_000_003)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--max-transitions", type=int, default=1_000)
    parser.add_argument("--disable-multiccd", action="store_true")
    return parser.parse_args()


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def adapter_action(agent, observation, key, alpha, history):
    context, mask = pad_context_numpy(
        history,
        int(agent.config["context_length"]),
        int(agent.config["context_token_dim"]),
    )
    flat = agent.sample_actions(
        jnp.asarray(observation),
        seed=key,
        guidance_weight=float(alpha),
        context=jnp.asarray(context),
        context_mask=jnp.asarray(mask),
        deterministic_latent=True,
    )
    return np.asarray(flat, dtype=np.float32).reshape(
        int(agent.config["horizon_length"]), int(agent.config["action_dim"])
    )


def _episode_bootstrap(rows, field, draws, seed):
    values_by_episode = {}
    for row in rows:
        values_by_episode.setdefault(row["episode"], []).append(row[field])
    values = np.asarray(
        [np.mean(values_by_episode[key]) for key in sorted(values_by_episode)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(draws)]
    )
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "episodes": int(len(values)),
        "draws": int(draws),
    }


def load_arms(args, observation, action):
    frozen, frozen_flags = load_checkpoint(
        args.frozen_native_checkpoint,
        args.frozen_native_epoch,
        observation,
        action,
        contextual=False,
    )
    qv, qv_flags = load_checkpoint(
        args.native_qv_checkpoint,
        args.native_qv_epoch,
        observation,
        action,
        contextual=False,
    )
    adapter, adapter_flags = load_checkpoint(
        args.context_adapter_checkpoint,
        args.context_adapter_epoch,
        observation,
        action,
        contextual=True,
    )
    if jax.tree_util.tree_structure(frozen.policy.params) != jax.tree_util.tree_structure(
        qv.policy.params
    ) or jax.tree_util.tree_structure(frozen.policy.params) != jax.tree_util.tree_structure(
        adapter.policy.params
    ):
        raise ValueError("The three arms do not share an actor parameter structure")
    return (frozen, qv, adapter), (frozen_flags, qv_flags, adapter_flags)


def evaluate_mc(args, frozen, qv, adapter):
    environment = make_env(args)
    rows = []
    discount = float(frozen.config["discount"])
    normalization = adapter.config.get("context_normalization", None)
    include_reward = bool(adapter.config.get("context_include_reward", True))

    for episode in range(args.mc_episodes):
        seed = args.episode_seed_base + episode
        observation, _ = environment.reset(seed=seed, options={"task_id": None})
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        history = []
        transition = 0
        chunk_index = 0
        done = False
        while not done and transition < args.max_transitions:
            key = action_key(args.action_seed_base, episode, chunk_index)
            # Shared prefix is frozen-native guidance so every arm is queried
            # from exactly the same state/history distribution.
            prefix = base_action(frozen, observation, key, args.guidance_weight)
            if transition in args.query_transitions:
                state, state_kind = physics_state(environment)
                bookkeeping = bookkeeping_state(environment)
                chunks = {
                    "frozen_native": prefix,
                    "native_qv_continuation": base_action(
                        qv, observation, key, args.guidance_weight
                    ),
                    "context_adapter": adapter_action(
                        adapter, observation, key, args.guidance_weight, history
                    ),
                }
                row = {"episode": episode, "transition": transition}
                for arm_name, chunk in chunks.items():
                    returns = [
                        continuation_return(
                            environment,
                            frozen,
                            state,
                            state_kind,
                            bookkeeping,
                            observation,
                            chunk,
                            discount,
                            args.continuation_guidance_weight,
                            args.continuation_seed_base,
                            episode,
                            transition,
                            args.max_transitions - transition,
                            rollout_index,
                        )
                        for rollout_index in range(args.mc_rollouts)
                    ]
                    row[f"{arm_name}_mc_return_mean"] = float(np.mean(returns))
                    row[f"{arm_name}_mc_return_std"] = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
                restore_physics_state(environment, state, state_kind)
                restore_bookkeeping(bookkeeping)
                rows.append(row)

            transitions = execute_chunk(environment, prefix)
            for command, (next_observation, reward, done, _) in zip(prefix, transitions):
                history.append(
                    transition_token_numpy(
                        observation,
                        command,
                        reward,
                        next_observation,
                        done,
                        normalization=normalization,
                        include_reward=include_reward,
                    )
                )
                history = history[-int(adapter.config["context_length"]) :]
                observation = next_observation
                transition += 1
                if done:
                    break
            chunk_index += 1
    environment.close()
    if not rows:
        raise RuntimeError("No MC query points collected; check query transitions")
    return rows


def evaluate_closed_loop(args, frozen, qv, adapter):
    environment = make_paired_env(args)
    rows = []
    for episode in range(args.closed_loop_episodes):
        seed = (args.episode_seed_base + episode) % (2**32)
        results = {
            "frozen_native": rollout_episode(
                environment, frozen, contextual=False, alpha=args.guidance_weight,
                episode_index=episode, episode_seed=seed,
                action_seed_base=args.action_seed_base,
            ),
            "native_qv_continuation": rollout_episode(
                environment, qv, contextual=False, alpha=args.guidance_weight,
                episode_index=episode, episode_seed=seed,
                action_seed_base=args.action_seed_base,
            ),
            "context_adapter": rollout_episode(
                environment, adapter, contextual=True, alpha=args.guidance_weight,
                episode_index=episode, episode_seed=seed,
                action_seed_base=args.action_seed_base,
            ),
        }
        row = {"episode": episode, "episode_seed": seed}
        for arm_name, result in results.items():
            for key, value in result.items():
                row[f"{arm_name}_{key}"] = value
        rows.append(row)
    environment.close()
    return rows


def main():
    args = parse_args()
    if args.continuation_guidance_weight != 0.0:
        raise ValueError("Actor-only MC requires --continuation-guidance-weight=0")
    if min(args.mc_episodes, args.closed_loop_episodes, args.mc_rollouts) <= 0:
        raise ValueError("Episode and rollout counts must be positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    bootstrap_env = make_env(args)
    observation, _ = bootstrap_env.reset(
        seed=args.episode_seed_base, options={"task_id": None}
    )
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    action = np.zeros(bootstrap_env.action_space.shape, dtype=np.float32)
    bootstrap_env.close()
    (frozen, qv, adapter), flags = load_arms(args, observation, action)
    before = {
        "frozen": tree_sha256(frozen.target_critic.params),
        "qv": tree_sha256(qv.target_critic.params),
        "adapter": tree_sha256(adapter.target_critic.params),
        "encoder": tree_sha256(adapter.context_encoder.params),
    }

    mc_rows = evaluate_mc(args, frozen, qv, adapter)
    closed_rows = evaluate_closed_loop(args, frozen, qv, adapter)
    after = {
        "frozen": tree_sha256(frozen.target_critic.params),
        "qv": tree_sha256(qv.target_critic.params),
        "adapter": tree_sha256(adapter.target_critic.params),
        "encoder": tree_sha256(adapter.context_encoder.params),
    }
    if before != after:
        raise AssertionError("Evaluation changed a checkpoint parameter")

    mc_summary = {
        arm: _episode_bootstrap(mc_rows, f"{arm}_mc_return_mean", args.bootstrap_draws, 100 + index)
        for index, arm in enumerate(ARM_NAMES)
    }
    closed_summary = {
        arm: {
            "success": float(np.mean([row[f"{arm}_success"] for row in closed_rows])),
            "return": float(np.mean([row[f"{arm}_return"] for row in closed_rows])),
            "length": float(np.mean([row[f"{arm}_length"] for row in closed_rows])),
        }
        for arm in ARM_NAMES
    }
    comparisons = {}
    for arm in ARM_NAMES[1:]:
        comparisons[arm] = {
            "success_minus_frozen": paired_bootstrap(
                [row[f"{arm}_success"] - row["frozen_native_success"] for row in closed_rows],
                args.bootstrap_draws, 200 + len(comparisons),
            ),
            "return_minus_frozen": paired_bootstrap(
                [row[f"{arm}_return"] - row["frozen_native_return"] for row in closed_rows],
                args.bootstrap_draws, 300 + len(comparisons),
            ),
        }
    result = {
        "protocol": {
            "env_name": args.env_name,
            "guidance_weight": args.guidance_weight,
            "mc_query_prefix": "frozen native QGF guided policy with shared seeds",
            "mc_first_action": "each arm's own Q-guided actor action",
            "mc_continuation": "frozen native actor only (guidance weight exactly 0)",
            "mc_episodes": args.mc_episodes,
            "closed_loop_episodes": args.closed_loop_episodes,
            "query_transitions": list(args.query_transitions),
            "mc_rollouts": args.mc_rollouts,
            "disable_multiccd": bool(args.disable_multiccd),
            "checkpoints": {
                "frozen_native": {"path": args.frozen_native_checkpoint, "epoch": args.frozen_native_epoch, "training_seed": flags[0]["seed"]},
                "native_qv_continuation": {"path": args.native_qv_checkpoint, "epoch": args.native_qv_epoch, "training_seed": flags[1]["seed"]},
                "context_adapter": {"path": args.context_adapter_checkpoint, "epoch": args.context_adapter_epoch, "training_seed": flags[2]["seed"]},
            },
        },
        "mc_return": mc_summary,
        "closed_loop": closed_summary,
        "closed_loop_delta_vs_frozen": comparisons,
        "parameter_immutable": True,
    }
    atomic_json(output / "result.json", result)
    for filename, rows in (("mc_per_query.csv", mc_rows), ("closed_loop_per_episode.csv", closed_rows)):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "_SUCCESS").touch()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
