#!/usr/bin/env python3
"""Measure Q-gradient scale differences at paired query points."""

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

from experiments.checkpoint_protocol import (
    add_paired_checkpoint_epochs,
    paired_checkpoint_epochs,
)
from experiments.evaluate_task3_closed_loop import action_key, make_paired_env
from experiments.evaluate_task3_mc import (
    base_action,
    execute_chunk,
    load_checkpoint,
)
from utils.context import (
    context_is_ready_numpy,
    pad_context_numpy,
    transition_token_numpy,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-checkpoint", required=True)
    parser.add_argument("--context-checkpoint", required=True)
    add_paired_checkpoint_epochs(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-name", default="cube-triple-play-singletask-task3-v0")
    parser.add_argument("--guidance-weight", type=float, default=0.04)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--query-transitions",
        type=lambda value: [int(item) for item in value.split(",")],
        default=(20, 40, 60, 80),
    )
    parser.add_argument("--episode-seed-base", type=int, default=72_000_000)
    parser.add_argument("--action-seed-base", type=int, default=172_000_003)
    parser.add_argument("--max-transitions", type=int, default=1_000)
    parser.add_argument("--disable-multiccd", action="store_true")
    return parser.parse_args()


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def aggregate_percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.quantile(values, 0.5)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
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

    native_epoch, context_epoch = paired_checkpoint_epochs(args)
    native, _ = load_checkpoint(
        args.native_checkpoint, native_epoch, observation, action, contextual=False
    )
    context, _ = load_checkpoint(
        args.context_checkpoint, context_epoch, observation, action, contextual=True
    )
    minimum = int(context.config["min_context_transitions"])
    normalization = context.config.get("context_normalization", None)
    include_reward = bool(context.config.get("context_include_reward", True))

    rows = []
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
            commands = base_action(native, current, key, args.guidance_weight)

            if transition in args.query_transitions and len(history) >= minimum:
                context_tokens, context_mask = pad_context_numpy(
                    history,
                    int(context.config["context_length"]),
                    int(context.config["context_token_dim"]),
                )
                mean, _ = context.infer_posterior(
                    jnp.asarray(context_tokens)[None],
                    jnp.asarray(context_mask)[None],
                )
                mean = np.asarray(mean, dtype=np.float32)
                ready = context_is_ready_numpy(
                    context_mask,
                    int(context.config["min_context_transitions"]),
                    dtype=np.float32,
                )

                full_action_dim = int(native.config["action_dim"]) * int(
                    native.config["horizon_length"]
                )
                noisy_action = jax.random.normal(key, (full_action_dim,))
                time_value = 0.0
                velocity = native.policy(
                    jnp.asarray(current)[None],
                    noisy_action[None],
                    jnp.full((1,), time_value, dtype=jnp.float32),
                )[0]
                approx_action = jnp.clip(
                    noisy_action
                    + (1.0 - time_value)
                    * jax.lax.stop_gradient(velocity),
                    -1.0,
                    1.0,
                )

                def native_q_gradient(candidate):
                    values = native.target_critic(
                        jnp.asarray(current)[None],
                        candidate[None],
                    )
                    return native._aggregate_q(values)[0]

                def context_q_gradient(candidate):
                    values = context.target_critic(
                        jnp.asarray(current)[None],
                        candidate[None],
                        jnp.asarray(mean),
                        jnp.asarray(ready),
                    )
                    return context._aggregate_q(values)[0]

                g_native = jax.grad(native_q_gradient)(approx_action)
                g_context = jax.grad(context_q_gradient)(approx_action)

                norm_native = float(jnp.linalg.norm(g_native))
                norm_context = float(jnp.linalg.norm(g_context))
                velocity_norm = float(jnp.linalg.norm(velocity))
                eps = 1e-8
                ratio = norm_context / (norm_native + eps)
                rho_native = args.guidance_weight * norm_native / (velocity_norm + eps)
                rho_context = args.guidance_weight * norm_context / (velocity_norm + eps)
                cosine = float(
                    jnp.dot(g_native, g_context)
                    / (norm_native * norm_context + eps)
                )
                saturation = float(
                    np.mean(np.abs(np.asarray(approx_action)) > 0.999)
                )
                rows.append(
                    {
                        "episode": episode_index,
                        "transition": transition,
                        "norm_native": norm_native,
                        "norm_context": norm_context,
                        "gradient_norm_ratio": ratio,
                        "rho_native": rho_native,
                        "rho_context": rho_context,
                        "rho_ratio": rho_context / (rho_native + eps),
                        "cosine": cosine,
                        "action_saturation": saturation,
                    }
                )

            transitions = execute_chunk(environment, commands)
            for command_index, (next_observation, reward, done, _) in enumerate(
                transitions
            ):
                history.append(
                    transition_token_numpy(
                        current,
                        commands[command_index],
                        reward,
                        next_observation,
                        done,
                        normalization=normalization,
                        include_reward=include_reward,
                    )
                )
                history = history[-int(context.config["context_length"]) :]
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
            "same_initial_action_noise": True,
            "same_action_approx": True,
        },
        "gradient_norm_ratio": aggregate_percentiles(
            [row["gradient_norm_ratio"] for row in rows]
        ),
        "rho_native": aggregate_percentiles([row["rho_native"] for row in rows]),
        "rho_context": aggregate_percentiles([row["rho_context"] for row in rows]),
        "rho_ratio": aggregate_percentiles([row["rho_ratio"] for row in rows]),
        "cosine": aggregate_percentiles([row["cosine"] for row in rows]),
        "action_saturation": aggregate_percentiles(
            [row["action_saturation"] for row in rows]
        ),
        "query_count": len(rows),
    }
    atomic_json(output / "result.json", result)
    with (output / "gradients.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "_SUCCESS").touch()
    environment.close()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
