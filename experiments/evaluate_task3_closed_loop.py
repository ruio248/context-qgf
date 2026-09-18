#!/usr/bin/env python3
"""Paired Task3 closed-loop success evaluation for native QGF and Context-Q.

Each episode is rolled out twice with the same reset seed and the same
per-chunk action-noise keys.  Unlike the MC evaluator, this measures the
actual closed-loop policy success and return of both agents.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from envs.env_utils import EpisodeMonitor
from envs.ogbench_utils import make_ogbench_env_and_datasets
from experiments.evaluate_task3_mc import load_checkpoint, tree_sha256
from utils.context import pad_context_numpy, transition_token_numpy
from utils.evaluation import flatten


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="cube-triple-play-singletask-task3-v0")
    parser.add_argument("--native-checkpoint", required=True)
    parser.add_argument("--context-checkpoint", required=True)
    parser.add_argument("--epoch", type=int, default=500_000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--guidance-weight", type=float, default=0.04)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--episode-seed-base", type=int, default=72_000_000)
    parser.add_argument("--action-seed-base", type=int, default=172_000_003)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260918)
    parser.add_argument("--disable-multiccd", action="store_true")
    parser.add_argument(
        "--context-actor-source",
        choices=["context", "native"],
        default="context",
        help=(
            "Use the Context-Q actor, or copy the native QGF actor into the "
            "Context-Q agent to isolate the Q-conditioning effect."
        ),
    )
    return parser.parse_args()


def atomic_json(path: Path, value):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def action_key(base: int, episode: int, chunk: int):
    integer = (int(base) + 10_007 * int(episode)) % (2**32)
    return jax.random.fold_in(jax.random.PRNGKey(integer), int(chunk))


def make_paired_env(args):
    environment = make_ogbench_env_and_datasets(args.env_name, env_only=True)
    environment = EpisodeMonitor(
        environment,
        filter_regexes=[".*privileged.*", ".*proprio.*"],
    )
    if args.disable_multiccd:
        environment.unwrapped.model.opt.disableflags |= int(
            mujoco.mjtDisableBit.mjDSBL_MULTICCD
        )

    action_space = environment.unwrapped.action_space
    base_class = type(environment.unwrapped)
    paired_class = type(
        f"PairedReset{base_class.__name__}",
        (base_class,),
        {"action_space": property(lambda self: self._paired_action_space)},
    )
    environment.unwrapped._paired_action_space = action_space
    environment.unwrapped.__class__ = paired_class
    return environment


def nested_success(info):
    flat = flatten(info)
    if "success" in flat:
        return float(flat["success"])
    raise RuntimeError(f"Environment info does not contain success: {sorted(flat)}")


def rollout_episode(
    environment,
    agent,
    *,
    contextual,
    alpha,
    episode_index,
    episode_seed,
    action_seed_base,
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

    horizon = int(agent.config["horizon_length"])
    action_dim = int(agent.config["action_dim"])
    include_reward = bool(agent.config.get("context_include_reward", True))
    normalization = agent.config.get("context_normalization", None)

    while not done:
        key = action_key(action_seed_base, episode_index, chunk_index)
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

        commands = np.asarray(flat, dtype=np.float32).reshape(horizon, action_dim)
        for command in commands:
            command = np.clip(command, -1.0, 1.0).astype(np.float32)
            next_observation, reward, terminated, truncated, info = environment.step(
                command
            )
            next_observation = np.asarray(
                next_observation, dtype=np.float32
            ).reshape(-1)

            history.append(
                transition_token_numpy(
                    observation,
                    command,
                    reward,
                    next_observation,
                    bool(terminated or truncated),
                    normalization=normalization if contextual else None,
                    include_reward=include_reward,
                )
            )
            history = history[-int(agent.config.get("context_length", 0)) :]

            observation = next_observation
            episode_return += float(reward)
            episode_length += 1
            final_info = info
            done = bool(terminated or truncated)
            if done:
                break

        chunk_index += 1

    return {
        "success": nested_success(final_info),
        "return": episode_return,
        "length": episode_length,
    }


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

    native, native_flags = load_checkpoint(
        args.native_checkpoint, args.epoch, observation, action, contextual=False
    )
    context, context_flags = load_checkpoint(
        args.context_checkpoint, args.epoch, observation, action, contextual=True
    )
    if args.context_actor_source == "native":
        if jax.tree_util.tree_structure(native.policy.params) != jax.tree_util.tree_structure(
            context.policy.params
        ):
            raise ValueError("Native and Context-Q actor parameter trees differ")
        context = context.replace(
            policy=context.policy.replace(params=native.policy.params)
        )

    before = {
        "native": tree_sha256(native.target_critic.params),
        "context": tree_sha256(context.target_critic.params),
        "encoder": tree_sha256(context.context_encoder.params),
    }

    rows = []
    for episode_index in range(args.episodes):
        episode_seed = (args.episode_seed_base + episode_index) % (2**32)
        native_result = rollout_episode(
            environment,
            native,
            contextual=False,
            alpha=args.guidance_weight,
            episode_index=episode_index,
            episode_seed=episode_seed,
            action_seed_base=args.action_seed_base,
        )
        context_result = rollout_episode(
            environment,
            context,
            contextual=True,
            alpha=args.guidance_weight,
            episode_index=episode_index,
            episode_seed=episode_seed,
            action_seed_base=args.action_seed_base,
        )

        rows.append(
            {
                "episode": episode_index,
                "episode_seed": episode_seed,
                "native_success": native_result["success"],
                "context_success": context_result["success"],
                "success_delta": context_result["success"] - native_result["success"],
                "native_return": native_result["return"],
                "context_return": context_result["return"],
                "return_delta": context_result["return"] - native_result["return"],
                "native_length": native_result["length"],
                "context_length": context_result["length"],
            }
        )

    after = {
        "native": tree_sha256(native.target_critic.params),
        "context": tree_sha256(context.target_critic.params),
        "encoder": tree_sha256(context.context_encoder.params),
    }
    if before != after:
        raise AssertionError("evaluation changed checkpoint parameters")

    result = {
        "protocol": {
            "env_name": args.env_name,
            "alpha": args.guidance_weight,
            "episodes": args.episodes,
            "episode_seed_base": args.episode_seed_base,
            "action_seed_base": args.action_seed_base,
            "paired_seed_protocol": (
                "native and context share episode reset seeds and per-chunk "
                "action-noise keys for every episode"
            ),
            "disable_multiccd": bool(args.disable_multiccd),
            "context_actor_source": args.context_actor_source,
            "native_flags_seed": native_flags["seed"],
            "context_flags_seed": context_flags["seed"],
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
            args.bootstrap_seed,
        ),
        "return_delta": paired_bootstrap(
            [row["return_delta"] for row in rows],
            args.bootstrap_draws,
            args.bootstrap_seed + 1,
        ),
        "paired_episodes": len(rows),
        "parameter_immutable": True,
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
