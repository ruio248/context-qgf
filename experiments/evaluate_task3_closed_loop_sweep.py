#!/usr/bin/env python3
"""Paired Task3 closed-loop alpha sweep for native QGF vs Context-Q."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import numpy as np

from experiments.evaluate_task3_closed_loop import (
    atomic_json,
    make_paired_env,
    paired_bootstrap,
    rollout_episode,
)
from experiments.checkpoint_protocol import (
    add_paired_checkpoint_epochs,
    paired_checkpoint_epochs,
)
from experiments.evaluate_task3_mc import load_checkpoint, tree_sha256


def parse_floats(value):
    return [float(item) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="cube-triple-play-singletask-task3-v0")
    parser.add_argument("--native-checkpoint", required=True)
    parser.add_argument("--context-checkpoint", required=True)
    add_paired_checkpoint_epochs(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--guidance-weights",
        type=parse_floats,
        default=(0.0, 0.004, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08),
    )
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
    )
    return parser.parse_args()


def alpha_label(alpha):
    return f"{alpha:.6f}".rstrip("0").rstrip(".").replace(".", "p") or "0"


def run_alpha(
    environment,
    native,
    native_flags,
    context,
    context_flags,
    args,
    alpha,
):
    rows = []
    for episode_index in range(args.episodes):
        episode_seed = (args.episode_seed_base + episode_index) % (2**32)
        native_result = rollout_episode(
            environment,
            native,
            contextual=False,
            alpha=alpha,
            episode_index=episode_index,
            episode_seed=episode_seed,
            action_seed_base=args.action_seed_base,
        )
        context_result = rollout_episode(
            environment,
            context,
            contextual=True,
            alpha=alpha,
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

    return {
        "protocol": {
            "env_name": args.env_name,
            "alpha": alpha,
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
    }, rows


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
    native, native_flags = load_checkpoint(
        args.native_checkpoint, native_epoch, observation, action, contextual=False
    )
    context, context_flags = load_checkpoint(
        args.context_checkpoint, context_epoch, observation, action, contextual=True
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

    summary_rows = []
    for alpha in args.guidance_weights:
        label = f"alpha_{alpha_label(alpha)}"
        alpha_dir = output / label
        alpha_dir.mkdir()
        result, rows = run_alpha(
            environment,
            native,
            native_flags,
            context,
            context_flags,
            args,
            alpha,
        )
        atomic_json(alpha_dir / "result.json", result)
        with (alpha_dir / "per_episode.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (alpha_dir / "_SUCCESS").touch()
        summary_rows.append(
            {
                "alpha": alpha,
                "native_success": result["native"]["success"],
                "context_success": result["context"]["success"],
                "success_delta": result["success_delta"]["mean"],
                "success_delta_ci95": result["success_delta"]["ci95"],
                "native_return": result["native"]["return"],
                "context_return": result["context"]["return"],
                "return_delta": result["return_delta"]["mean"],
                "return_delta_ci95": result["return_delta"]["ci95"],
            }
        )

    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    atomic_json(output / "result.json", {
        "protocol": {
            "env_name": args.env_name,
            "native_epoch": native_epoch,
            "context_epoch": context_epoch,
            "episodes": args.episodes,
            "guidance_weights": args.guidance_weights,
            "episode_seed_base": args.episode_seed_base,
            "action_seed_base": args.action_seed_base,
            "context_actor_source": args.context_actor_source,
            "disable_multiccd": bool(args.disable_multiccd),
        },
        "per_alpha": summary_rows,
    })

    after = {
        "native": tree_sha256(native.target_critic.params),
        "context": tree_sha256(context.target_critic.params),
        "encoder": tree_sha256(context.context_encoder.params),
    }
    if before != after:
        raise AssertionError("evaluation changed checkpoint parameters")
    environment.close()
    print(json.dumps({"summary": summary_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
