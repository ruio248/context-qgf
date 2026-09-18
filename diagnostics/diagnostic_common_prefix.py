#!/usr/bin/env python3
"""Compare native versus Context-Q critic after a shared native prefix."""

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
    bookkeeping_state,
    load_checkpoint,
    physics_state,
    restore_bookkeeping,
    restore_physics_state,
)
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
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--prefix-steps", type=int, default=20)
    parser.add_argument("--episode-seed-base", type=int, default=72_000_000)
    parser.add_argument("--action-seed-base", type=int, default=172_000_003)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--max-transitions", type=int, default=1_000)
    parser.add_argument("--disable-multiccd", action="store_true")
    return parser.parse_args()


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def success_from_info(info):
    flat = flatten(info)
    return float(flat.get("success", 0.0))


def sample_chunk(agent, observation, key, alpha, *, contextual, history):
    if contextual:
        context_tokens, context_mask = pad_context_numpy(
            history,
            int(agent.config["context_length"]),
            int(agent.config["context_token_dim"]),
        )
        flat = agent.sample_actions(
            jnp.asarray(observation),
            seed=key,
            guidance_weight=float(alpha),
            context=jnp.asarray(context_tokens),
            context_mask=jnp.asarray(context_mask),
            deterministic_latent=True,
        )
    else:
        flat = agent.sample_actions(
            jnp.asarray(observation),
            seed=key,
            guidance_weight=float(alpha),
        )
    return np.asarray(flat, dtype=np.float32).reshape(
        int(agent.config["horizon_length"]),
        int(agent.config["action_dim"]),
    )


def run_until(
    environment,
    agent,
    *,
    contextual,
    observation,
    history,
    episode_index,
    action_seed_base,
    alpha,
    chunk_index,
    target_steps,
    max_transitions,
    normalization,
    include_reward,
):
    """Run until target_steps or termination; return state and accumulators."""

    total_return = 0.0
    steps = 0
    done = False
    final_info = {}
    current = np.asarray(observation, dtype=np.float32)
    current_history = list(history)

    while not done and steps < target_steps:
        chunk = sample_chunk(
            agent,
            current,
            action_key(action_seed_base, episode_index, chunk_index),
            alpha,
            contextual=contextual,
            history=current_history,
        )
        for command in chunk:
            next_observation, reward, terminated, truncated, info = environment.step(
                command
            )
            next_observation = np.asarray(
                next_observation, dtype=np.float32
            ).reshape(-1)
            done = bool(terminated or truncated)
            total_return += float(reward)
            steps += 1
            current_history.append(
                transition_token_numpy(
                    current,
                    command,
                    reward,
                    next_observation,
                    done,
                    normalization=normalization if contextual else None,
                    include_reward=include_reward,
                )
            )
            current = next_observation
            final_info = info
            if done or steps >= target_steps:
                break
        chunk_index += 1
        if steps >= max_transitions:
            break

    return {
        "observation": current,
        "history": current_history,
        "return": total_return,
        "length": steps,
        "done": done,
        "info": final_info,
        "chunk_index": chunk_index,
    }


def run_branch(
    environment,
    agent,
    *,
    contextual,
    observation,
    history,
    episode_index,
    action_seed_base,
    alpha,
    chunk_index,
    max_transitions,
    normalization,
    include_reward,
):
    result = run_until(
        environment,
        agent,
        contextual=contextual,
        observation=observation,
        history=history,
        episode_index=episode_index,
        action_seed_base=action_seed_base,
        alpha=alpha,
        chunk_index=chunk_index,
        target_steps=max_transitions,
        max_transitions=max_transitions,
        normalization=normalization,
        include_reward=include_reward,
    )
    return result


def paired_bootstrap(values, draws, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws_array = np.asarray([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(draws)
    ])
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(np.quantile(draws_array, 0.025)),
            float(np.quantile(draws_array, 0.975)),
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
    normalization = context.config.get("context_normalization", None)
    include_reward = bool(context.config.get("context_include_reward", True))

    rows = []
    for episode_index in range(args.episodes):
        episode_seed = (args.episode_seed_base + episode_index) % (2**32)
        environment.unwrapped._paired_action_space.seed(episode_seed)
        episode_observation, _ = environment.reset(
            seed=episode_seed, options={"task_id": None}
        )
        episode_observation = np.asarray(
            episode_observation, dtype=np.float32
        ).reshape(-1)

        prefix = run_until(
            environment,
            native,
            contextual=False,
            observation=episode_observation,
            history=[],
            episode_index=episode_index,
            action_seed_base=args.action_seed_base,
            alpha=args.guidance_weight,
            chunk_index=0,
            target_steps=args.prefix_steps,
            max_transitions=args.max_transitions,
            normalization=normalization,
            include_reward=include_reward,
        )

        if prefix["done"]:
            native_branch = prefix
            context_branch = prefix
        else:
            state, state_kind = physics_state(environment)
            bookkeeping = bookkeeping_state(environment)

            native_branch = run_branch(
                environment,
                native,
                contextual=False,
                observation=prefix["observation"],
                history=prefix["history"],
                episode_index=episode_index,
                action_seed_base=args.action_seed_base,
                alpha=args.guidance_weight,
                chunk_index=prefix["chunk_index"],
                max_transitions=args.max_transitions - prefix["length"],
                normalization=normalization,
                include_reward=include_reward,
            )

            restore_physics_state(environment, state, state_kind)
            restore_bookkeeping(bookkeeping)

            context_branch = run_branch(
                environment,
                hybrid,
                contextual=True,
                observation=prefix["observation"],
                history=prefix["history"],
                episode_index=episode_index,
                action_seed_base=args.action_seed_base,
                alpha=args.guidance_weight,
                chunk_index=prefix["chunk_index"],
                max_transitions=args.max_transitions - prefix["length"],
                normalization=normalization,
                include_reward=include_reward,
            )

        total_native_return = prefix["return"] + native_branch["return"]
        total_context_return = prefix["return"] + context_branch["return"]
        native_success = success_from_info(native_branch["info"])
        context_success = success_from_info(context_branch["info"])
        rows.append(
            {
                "episode": episode_index,
                "prefix_length": prefix["length"],
                "native_success": native_success,
                "context_success": context_success,
                "success_delta": context_success - native_success,
                "native_return": total_native_return,
                "context_return": total_context_return,
                "return_delta": total_context_return - total_native_return,
            }
        )

    result = {
        "protocol": {
            "env_name": args.env_name,
            "alpha": args.guidance_weight,
            "episodes": args.episodes,
            "prefix_steps": args.prefix_steps,
            "prefix_policy": "native QGF",
            "post_prefix_context": "native actor + context critic and encoder",
        },
        "native": {
            "success": float(np.mean([row["native_success"] for row in rows])),
            "return": float(np.mean([row["native_return"] for row in rows])),
        },
        "context": {
            "success": float(np.mean([row["context_success"] for row in rows])),
            "return": float(np.mean([row["context_return"] for row in rows])),
        },
        "success_delta": paired_bootstrap(
            [row["success_delta"] for row in rows],
            args.bootstrap_draws,
            20260921,
        ),
        "return_delta": paired_bootstrap(
            [row["return_delta"] for row in rows],
            args.bootstrap_draws,
            20260922,
        ),
    }
    atomic_json(output / "result.json", result)
    with (output / "per_episode.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "_SUCCESS").touch()
    environment.close()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
