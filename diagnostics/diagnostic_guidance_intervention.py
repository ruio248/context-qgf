#!/usr/bin/env python3
"""Compare the actual actions produced by native and Context Q guidance.

At each paired query point we generate one chunk with native Q guidance and one
with Context-Q guidance using the same initial action-noise key.  Both chunks
are then evaluated with the same native continuation policy and the same
continuation noise sequence.
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

from diagnostics.common import build_fixed_actor_context_agent
from experiments.evaluate_task3_closed_loop import action_key, make_paired_env
from experiments.evaluate_task3_mc import (
    base_action,
    bookkeeping_state,
    execute_chunk,
    load_checkpoint,
    physics_state,
    restore_bookkeeping,
    restore_physics_state,
)
from experiments.evaluate_task3_mc_matrix import sample_continuation_action
from utils.context import pad_context_numpy, transition_token_numpy
from utils.evaluation import flatten


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-checkpoint", required=True)
    parser.add_argument("--context-checkpoint", required=True)
    parser.add_argument("--epoch", type=int, default=500_000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-name", default="cube-triple-play-singletask-task3-v0")
    parser.add_argument("--guidance-weight", type=float, default=0.04)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--query-transitions",
        type=lambda value: [int(item) for item in value.split(",")],
        default=(20, 40, 60, 80),
    )
    parser.add_argument("--mc-rollouts", type=int, default=8)
    parser.add_argument("--episode-seed-base", type=int, default=72_000_000)
    parser.add_argument("--action-seed-base", type=int, default=172_000_003)
    parser.add_argument("--continuation-seed-base", type=int, default=272_000_003)
    parser.add_argument("--max-transitions", type=int, default=1_000)
    parser.add_argument("--disable-multiccd", action="store_true")
    return parser.parse_args()


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def continuation_outcome(
    environment,
    agent,
    *,
    state,
    state_kind,
    bookkeeping,
    observation,
    first_chunk,
    history,
    discount,
    alpha,
    seed_base,
    episode,
    query_transition,
    max_transitions,
    rollout_index,
    normalization,
    include_reward,
):
    restore_physics_state(environment, state, state_kind)
    restore_bookkeeping(bookkeeping)

    current = np.asarray(observation, dtype=np.float32)
    current_history = list(history)
    total = 0.0
    power = 1.0
    steps = 0
    chunk_index = 0
    done = False
    final_info = {}
    chunk = np.asarray(first_chunk, dtype=np.float32)

    while not done and steps < max_transitions:
        transitions = execute_chunk(environment, chunk)
        for command, (next_observation, reward, done, info) in zip(chunk, transitions):
            total += power * reward
            power *= discount
            steps += 1
            current_history.append(
                transition_token_numpy(
                    current,
                    command,
                    reward,
                    next_observation,
                    done,
                    normalization=normalization,
                    include_reward=include_reward,
                )
            )
            current = next_observation
            final_info = info
            if done or steps >= max_transitions:
                break
        if done or steps >= max_transitions:
            break

        chunk_index += 1
        key = jax.random.fold_in(
            jax.random.PRNGKey(
                (
                    int(seed_base)
                    + 100_003 * int(episode)
                    + 1_009 * int(query_transition)
                    + 97_003 * int(rollout_index)
                )
                % (2**32)
            ),
            chunk_index,
        )
        chunk = sample_continuation_action(
            agent,
            current,
            key,
            alpha,
            contextual=False,
            history=current_history,
        )

    success = float(flatten(final_info).get("success", 0.0))
    return total, success


def continuation_bootstrap(values, draws, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    resampled = np.asarray([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(draws)
    ])
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(np.quantile(resampled, 0.025)),
            float(np.quantile(resampled, 0.975)),
        ],
        "draws": int(draws),
    }


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    environment = make_paired_env(args)
    observation, _ = environment.reset(
        seed=args.episode_seed_base, options={"task_id": None}
    )
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    action = np.zeros(environment.action_space.shape, dtype=np.float32)

    native, _ = load_checkpoint(
        args.native_checkpoint, args.epoch, observation, action, contextual=False
    )
    context, _ = load_checkpoint(
        args.context_checkpoint, args.epoch, observation, action, contextual=True
    )
    hybrid = build_fixed_actor_context_agent(native, context)
    discount = float(context.config["discount"])
    minimum = int(context.config["min_context_transitions"])
    normalization = context.config.get("context_normalization", None)
    include_reward = bool(context.config.get("context_include_reward", True))

    rows = []
    query_deltas = []
    for episode_index in range(args.episodes):
        episode_seed = (args.episode_seed_base + episode_index) % (2**32)
        environment.unwrapped._paired_action_space.seed(episode_seed)
        episode_observation, _ = environment.reset(
            seed=episode_seed, options={"task_id": None}
        )
        current = np.asarray(episode_observation, dtype=np.float32).reshape(-1)
        history = []
        transition = 0
        chunk_index = 0
        done = False

        while not done and transition <= max(args.query_transitions):
            key = action_key(args.action_seed_base, episode_index, chunk_index)
            base_commands = base_action(native, current, key, args.guidance_weight)

            if transition in args.query_transitions and len(history) >= minimum:
                state, state_kind = physics_state(environment)
                bookkeeping = bookkeeping_state(environment)
                context_tokens, context_mask = pad_context_numpy(
                    history,
                    int(context.config["context_length"]),
                    int(context.config["context_token_dim"]),
                )
                native_chunk = np.asarray(
                    native.sample_actions(
                        jnp.asarray(current),
                        seed=key,
                        guidance_weight=args.guidance_weight,
                    ),
                    dtype=np.float32,
                ).reshape(
                    int(native.config["horizon_length"]),
                    int(native.config["action_dim"]),
                )
                context_chunk = np.asarray(
                    hybrid.sample_actions(
                        jnp.asarray(current),
                        seed=key,
                        guidance_weight=args.guidance_weight,
                        context=jnp.asarray(context_tokens),
                        context_mask=jnp.asarray(context_mask),
                        deterministic_latent=True,
                    ),
                    dtype=np.float32,
                ).reshape(
                    int(hybrid.config["horizon_length"]),
                    int(hybrid.config["action_dim"]),
                )

                native_returns = []
                context_returns = []
                native_successes = []
                context_successes = []
                for rollout_index in range(args.mc_rollouts):
                    native_return, native_success = continuation_outcome(
                        environment,
                        native,
                        state=state,
                        state_kind=state_kind,
                        bookkeeping=bookkeeping,
                        observation=current,
                        first_chunk=native_chunk,
                        history=history,
                        discount=discount,
                        alpha=args.guidance_weight,
                        seed_base=args.continuation_seed_base,
                        episode=episode_index,
                        query_transition=transition,
                        max_transitions=args.max_transitions - transition,
                        rollout_index=rollout_index,
                        normalization=normalization,
                        include_reward=include_reward,
                    )
                    context_return, context_success = continuation_outcome(
                        environment,
                        native,
                        state=state,
                        state_kind=state_kind,
                        bookkeeping=bookkeeping,
                        observation=current,
                        first_chunk=context_chunk,
                        history=history,
                        discount=discount,
                        alpha=args.guidance_weight,
                        seed_base=args.continuation_seed_base,
                        episode=episode_index,
                        query_transition=transition,
                        max_transitions=args.max_transitions - transition,
                        rollout_index=rollout_index,
                        normalization=normalization,
                        include_reward=include_reward,
                    )
                    native_returns.append(native_return)
                    context_returns.append(context_return)
                    native_successes.append(native_success)
                    context_successes.append(context_success)

                restore_physics_state(environment, state, state_kind)
                restore_bookkeeping(bookkeeping)
                native_mean = float(np.mean(native_returns))
                context_mean = float(np.mean(context_returns))
                row = {
                    "episode": episode_index,
                    "transition": transition,
                    "native_return": native_mean,
                    "context_return": context_mean,
                    "return_delta": context_mean - native_mean,
                    "native_success": float(np.mean(native_successes)),
                    "context_success": float(np.mean(context_successes)),
                    "success_delta": float(
                        np.mean(context_successes) - np.mean(native_successes)
                    ),
                }
                rows.append(row)
                query_deltas.append(row["return_delta"])

            transitions = execute_chunk(environment, base_commands)
            for command_index, (next_observation, reward, done, _) in enumerate(
                transitions
            ):
                history.append(
                    transition_token_numpy(
                        current,
                        base_commands[command_index],
                        reward,
                        next_observation,
                        done,
                        normalization=normalization,
                        include_reward=include_reward,
                    )
                )
                current = next_observation
                transition += 1
                if done:
                    break
            chunk_index += 1

    result = {
        "protocol": {
            "env_name": args.env_name,
            "alpha": args.guidance_weight,
            "episodes": args.episodes,
            "query_transitions": args.query_transitions,
            "mc_rollouts": args.mc_rollouts,
            "continuation": "shared native QGF continuation and noise",
        },
        "return_delta": continuation_bootstrap(query_deltas, 20_000, 20260923),
        "success_delta": float(np.mean([row["success_delta"] for row in rows])),
        "queries": len(rows),
    }
    atomic_json(output / "result.json", result)
    with (output / "queries.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "_SUCCESS").touch()
    environment.close()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
