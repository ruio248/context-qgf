#!/usr/bin/env python3
"""Paired Task3 closed-loop success evaluation for native QGF and Context-Q.

Every episode is rolled out twice under the same reset seed and the same
initial action-noise seed: once with the native QGF agent and once with the
Context-Q agent.  The only intended difference is the context conditioning
path.  This is the policy-level counterpart of the MC calibration evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.evaluate_task3_mc import load_checkpoint, make_env, tree_sha256
from utils.evaluation import flatten, run_episodes


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
    return parser.parse_args()


def atomic_json(path: Path, value):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def action_seed_for_episode(base: int, episode: int) -> int:
    return (int(base) + 10_007 * int(episode)) % (2**32)


def rollout_episode(agent, env, *, episode_seed, action_seed, guidance_weight):
    trajectories, _, returns, lengths = run_episodes(
        agent,
        env,
        guidance_weight=guidance_weight,
        episode_seed=episode_seed,
        action_seed=action_seed,
        rejection_sampling=1,
    )
    final_info = trajectories[0]["info"][-1]
    flat_info = flatten(final_info)
    if "success" not in flat_info:
        raise RuntimeError(f"Environment info does not contain success: {sorted(flat_info)}")
    return (
        float(flat_info["success"]),
        float(returns[0]),
        int(lengths[0]),
    )


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

    environment = make_env(args)
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

    before = {
        "native": tree_sha256(native.target_critic.params),
        "context": tree_sha256(context.target_critic.params),
        "encoder": tree_sha256(context.context_encoder.params),
    }

    rows = []
    for episode in range(args.episodes):
        episode_seed = (args.episode_seed_base + episode) % (2**32)
        action_seed = action_seed_for_episode(args.action_seed_base, episode)
        native_success, native_return, native_length = rollout_episode(
            native,
            environment,
            episode_seed=episode_seed,
            action_seed=action_seed,
            guidance_weight=args.guidance_weight,
        )
        context_success, context_return, context_length = rollout_episode(
            context,
            environment,
            episode_seed=episode_seed,
            action_seed=action_seed,
            guidance_weight=args.guidance_weight,
        )
        rows.append(
            {
                "episode": episode,
                "episode_seed": episode_seed,
                "action_seed": action_seed,
                "native_success": native_success,
                "context_success": context_success,
                "success_delta": context_success - native_success,
                "native_return": native_return,
                "context_return": context_return,
                "return_delta": context_return - native_return,
                "native_length": native_length,
                "context_length": context_length,
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
                "native and context share episode reset seeds and initial "
                "action-noise seeds for every episode"
            ),
            "disable_multiccd": bool(args.disable_multiccd),
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
