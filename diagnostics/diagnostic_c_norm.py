#!/usr/bin/env python3
"""C-norm ablation: keep Context gradient direction but match native norm."""

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
from experiments.evaluate_task3_mc import load_checkpoint
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


@jax.jit
def sample_cnorm_chunk(
    native_agent,
    context_agent,
    observation,
    key,
    alpha,
    context_tokens,
    context_mask,
    mean,
    ready,
):
    context_tokens = jnp.asarray(context_tokens)[None]
    context_mask = jnp.asarray(context_mask)[None]
    mean = jnp.asarray(mean)
    ready = jnp.asarray(ready)

    horizon = int(context_agent.config["horizon_length"])
    action_dim = int(context_agent.config["action_dim"])
    full_action_dim = horizon * action_dim
    denoise_steps = int(context_agent.config["denoise_steps"])
    action = jax.random.normal(key, (1, full_action_dim))
    dt = 1.0 / denoise_steps

    def step(current_action, time_index):
        time = jnp.full((1,), time_index / denoise_steps)
        velocity = context_agent.policy(
            jnp.asarray(observation)[None],
            current_action,
            time,
        )
        approx_action = jnp.clip(
            current_action
            + (1.0 - time[:, None])
            * jax.lax.stop_gradient(velocity),
            -1.0,
            1.0,
        )

        def native_q_fn(candidate):
            return native_agent._aggregate_q(
                native_agent.target_critic(
                    jnp.asarray(observation)[None],
                    candidate,
                )
            )[0]

        def context_q_fn(candidate):
            return context_agent._aggregate_q(
                context_agent.target_critic(
                    jnp.asarray(observation)[None],
                    candidate,
                    mean,
                    ready,
                )
            )[0]

        g_native = jax.grad(native_q_fn)(approx_action)
        g_context = jax.grad(context_q_fn)(approx_action)
        eps = 1e-8
        norm_native = jnp.linalg.norm(g_native, axis=-1, keepdims=True)
        norm_context = jnp.linalg.norm(g_context, axis=-1, keepdims=True)
        g_context = jnp.where(
            norm_context > eps,
            g_context * norm_native / jnp.maximum(norm_context, eps),
            g_context,
        )
        return current_action + (velocity + alpha * g_context) * dt, None

    actions, _ = jax.lax.scan(
        step,
        action,
        jnp.arange(denoise_steps),
        length=denoise_steps,
    )
    return jnp.clip(actions, -1.0, 1.0)[0]


def rollout_episode(
    environment,
    native_agent,
    context_agent,
    *,
    contextual,
    alpha,
    episode_index,
    episode_seed,
    action_seed_base,
    max_transitions,
):
    environment.unwrapped._paired_action_space.seed(episode_seed)
    observation, _ = environment.reset(
        seed=episode_seed, options={"task_id": None}
    )
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    history = []
    episode_return = 0.0
    episode_length = 0
    chunk_index = 0
    final_info = {}
    done = False

    while not done and episode_length < max_transitions:
        key = action_key(action_seed_base, episode_index, chunk_index)
        if contextual:
            context_tokens, context_mask = pad_context_numpy(
                history,
                int(context_agent.config["context_length"]),
                int(context_agent.config["context_token_dim"]),
            )
            mean, _ = context_agent.infer_posterior(
                jnp.asarray(context_tokens)[None],
                jnp.asarray(context_mask)[None],
            )
            mean = np.asarray(mean, dtype=np.float32)
            ready = np.asarray([1.0], dtype=np.float32)
            commands = np.asarray(
                sample_cnorm_chunk(
                native_agent,
                context_agent,
                observation,
                key,
                alpha,
                context_tokens,
                context_mask,
                mean,
                ready,
                ),
                dtype=np.float32,
            ).reshape(
                int(context_agent.config["horizon_length"]),
                int(context_agent.config["action_dim"]),
            )
        else:
            flat = native_agent.sample_actions(
                jnp.asarray(observation),
                seed=key,
                guidance_weight=float(alpha),
            )
            commands = np.asarray(flat, dtype=np.float32).reshape(
                int(native_agent.config["horizon_length"]),
                int(native_agent.config["action_dim"]),
            )

        for command in commands:
            command = np.clip(command, -1.0, 1.0).astype(np.float32)
            next_observation, reward, terminated, truncated, info = environment.step(
                command
            )
            next_observation = np.asarray(
                next_observation, dtype=np.float32
            ).reshape(-1)
            done = bool(terminated or truncated)
            episode_return += float(reward)
            episode_length += 1
            final_info = info
            if contextual:
                history.append(
                    transition_token_numpy(
                        observation,
                        command,
                        reward,
                        next_observation,
                        done,
                        normalization=context_agent.config.get(
                            "context_normalization", None
                        ),
                        include_reward=bool(
                            context_agent.config.get("context_include_reward", True)
                        ),
                    )
                )
            observation = next_observation
            if done:
                break
        chunk_index += 1

    success = float(flatten(final_info).get("success", 0.0))
    return success, episode_return, episode_length


def paired_bootstrap(values, draws, seed):
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

    rows = []
    for episode_index in range(args.episodes):
        episode_seed = (args.episode_seed_base + episode_index) % (2**32)
        native_result = rollout_episode(
            environment,
            native,
            hybrid,
            contextual=False,
            alpha=args.guidance_weight,
            episode_index=episode_index,
            episode_seed=episode_seed,
            action_seed_base=args.action_seed_base,
            max_transitions=args.max_transitions,
        )
        context_result = rollout_episode(
            environment,
            native,
            hybrid,
            contextual=True,
            alpha=args.guidance_weight,
            episode_index=episode_index,
            episode_seed=episode_seed,
            action_seed_base=args.action_seed_base,
            max_transitions=args.max_transitions,
        )
        rows.append(
            {
                "episode": episode_index,
                "native_success": native_result[0],
                "context_success": context_result[0],
                "success_delta": context_result[0] - native_result[0],
                "native_return": native_result[1],
                "context_return": context_result[1],
                "return_delta": context_result[1] - native_result[1],
            }
        )

    result = {
        "protocol": {
            "env_name": args.env_name,
            "alpha": args.guidance_weight,
            "episodes": args.episodes,
            "guidance_gradient": "Context direction, native per-step norm",
            "actor": "native QGF actor",
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
            20260924,
        ),
        "return_delta": paired_bootstrap(
            [row["return_delta"] for row in rows],
            args.bootstrap_draws,
            20260925,
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
