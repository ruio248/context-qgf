#!/usr/bin/env python3
"""Check real-checkpoint equivalence for a zero-context Context-Q agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import flax
import jax
import jax.numpy as jnp
import numpy as np

from diagnostics.common import (
    copy_matching_params,
    max_abs_diff,
    zero_context_inputs,
)
from experiments.evaluate_task3_mc import load_checkpoint
from utils.evaluation import run_episodes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-checkpoint", required=True)
    parser.add_argument("--context-checkpoint", required=True)
    parser.add_argument("--epoch", type=int, default=500_000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-name", default="cube-triple-play-singletask-task3-v0")
    parser.add_argument("--episode-seed", type=int, default=72_000_000)
    parser.add_argument("--action-seed", type=int, default=172_000_003)
    parser.add_argument("--disable-multiccd", action="store_true")
    return parser.parse_args()


def make_zero_context_agent(native, context):
    hybrid = context.replace(
        policy=context.policy.replace(
            params=copy_matching_params(context.policy.params, native.policy.params)
        ),
        critic=context.critic.replace(
            params=zero_context_inputs(
                copy_matching_params(context.critic.params, native.critic.params)
            )
        ),
        target_critic=context.target_critic.replace(
            params=zero_context_inputs(
                copy_matching_params(
                    context.target_critic.params,
                    native.target_critic.params,
                )
            )
        ),
        value=context.value.replace(
            params=zero_context_inputs(
                copy_matching_params(context.value.params, native.value.params)
            )
        ),
    )
    config = dict(hybrid.config)
    config["min_context_transitions"] = 10**9
    return hybrid.replace(config=flax.core.FrozenDict(config))


def grad_action(agent, observation, action, *, contextual, mean, ready):
    if contextual:
        def q_fn(candidate):
            return agent._aggregate_q(
                agent.target_critic(
                    jnp.asarray(observation)[None],
                    candidate[None],
                    jnp.asarray(mean),
                    jnp.asarray(ready),
                )
            )[0]
    else:
        def q_fn(candidate):
            return agent._aggregate_q(
                agent.target_critic(
                    jnp.asarray(observation)[None],
                    candidate[None],
                )
            )[0]
    return jax.grad(q_fn)(jnp.asarray(action))


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    from experiments.evaluate_task3_closed_loop import make_paired_env

    environment = make_paired_env(args)
    observation, _ = environment.reset(
        seed=args.episode_seed, options={"task_id": None}
    )
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    action = np.zeros(environment.action_space.shape, dtype=np.float32)

    native, _ = load_checkpoint(
        args.native_checkpoint, args.epoch, observation, action, contextual=False
    )
    context, _ = load_checkpoint(
        args.context_checkpoint, args.epoch, observation, action, contextual=True
    )
    hybrid = make_zero_context_agent(native, context)

    rng = np.random.default_rng(20260919)
    observations = rng.normal(size=(4, observation.shape[0])).astype(np.float32)
    actions = rng.normal(
        size=(4, int(native.config["action_dim"]) * int(native.config["horizon_length"]))
    ).astype(np.float32)
    latent = np.zeros((4, int(context.config["latent_dim"])), dtype=np.float32)
    ready = np.zeros((4,), dtype=np.float32)

    native_q = native.target_critic(
        jnp.asarray(observations), jnp.asarray(actions)
    )
    hybrid_q = hybrid.target_critic(
        jnp.asarray(observations),
        jnp.asarray(actions),
        jnp.asarray(latent),
        jnp.asarray(ready),
    )
    q_diff = float(jnp.max(jnp.abs(native_q - hybrid_q)))

    native_grad = grad_action(
        native,
        observations[0],
        actions[0],
        contextual=False,
        mean=latent[:1],
        ready=ready[:1],
    )
    hybrid_grad = grad_action(
        hybrid,
        observations[0],
        actions[0],
        contextual=True,
        mean=latent[:1],
        ready=ready[:1],
    )
    grad_diff = float(jnp.max(jnp.abs(native_grad - hybrid_grad)))

    seed = jax.random.PRNGKey(args.action_seed)
    native_alpha_zero = native.sample_actions(
        jnp.asarray(observation), seed=seed, guidance_weight=0.0
    )
    hybrid_alpha_zero = hybrid.sample_actions(
        jnp.asarray(observation),
        seed=seed,
        guidance_weight=0.0,
        context=jnp.zeros((20, int(hybrid.config["context_token_dim"]))),
        context_mask=jnp.zeros((20,)),
        deterministic_latent=True,
    )
    zero_action_diff = float(jnp.max(jnp.abs(native_alpha_zero - hybrid_alpha_zero)))

    native_alpha_004 = native.sample_actions(
        jnp.asarray(observation), seed=seed, guidance_weight=0.04
    )
    hybrid_alpha_004 = hybrid.sample_actions(
        jnp.asarray(observation),
        seed=seed,
        guidance_weight=0.04,
        context=jnp.zeros((20, int(hybrid.config["context_token_dim"]))),
        context_mask=jnp.zeros((20,)),
        deterministic_latent=True,
    )
    action_diff = float(jnp.max(jnp.abs(native_alpha_004 - hybrid_alpha_004)))

    native_traj, _, native_returns, native_lengths = run_episodes(
        native,
        environment,
        guidance_weight=0.04,
        episode_seed=args.episode_seed,
        action_seed=args.action_seed,
    )
    hybrid_traj, _, hybrid_returns, hybrid_lengths = run_episodes(
        hybrid,
        environment,
        guidance_weight=0.04,
        episode_seed=args.episode_seed,
        action_seed=args.action_seed,
    )
    native_info = native_traj[0]["info"][-1]
    hybrid_info = hybrid_traj[0]["info"][-1]

    result = {
        "q_max_abs_diff": q_diff,
        "action_gradient_max_abs_diff": grad_diff,
        "alpha_zero_action_max_abs_diff": zero_action_diff,
        "alpha_004_action_max_abs_diff": action_diff,
        "rollout": {
            "native_return": float(native_returns[0]),
            "hybrid_return": float(hybrid_returns[0]),
            "native_length": int(native_lengths[0]),
            "hybrid_length": int(hybrid_lengths[0]),
            "native_success": float(native_info.get("success", 0.0)),
            "hybrid_success": float(hybrid_info.get("success", 0.0)),
        },
        "zero_context_hybrid_is_equivalent": (
            q_diff < 1e-5
            and grad_diff < 1e-5
            and zero_action_diff < 1e-5
            and action_diff < 1e-5
        ),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (output / "_SUCCESS").touch()
    environment.close()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
