#!/usr/bin/env python3
"""Paired Task3 MC-return calibration audit for from-scratch QGF models.

The native and Context-Q checkpoints are queried at identical state/action
points.  Each point branches from one saved MuJoCo integration state, executes
the same first action chunk, and uses the same frozen native QGF continuation
policy for every Monte-Carlo replicate.  The primary statistic is

    Delta_MC = |Q_context - G_MC| - |Q_native - G_MC|.

Negative values favor Context-Q calibration; this is a value-calibration test,
not a claim about closed-loop guidance success.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from agents.context_qgf import ContextQGFAgent
from agents.qgf import QGFAgent
from envs.env_utils import EpisodeMonitor
from envs.ogbench_utils import make_ogbench_env_and_datasets
from utils.context import pad_context_numpy, transition_token_numpy
from utils.flax_utils import restore_agent


def parse_ints(value):
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="cube-triple-play-singletask-task3-v0")
    parser.add_argument("--native-checkpoint", required=True)
    parser.add_argument("--context-checkpoint", required=True)
    parser.add_argument("--epoch", type=int, default=500_000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--guidance-weight", type=float, default=0.04)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--query-transitions", type=parse_ints, default=(20, 40, 60))
    parser.add_argument("--mc-rollouts", type=int, default=8)
    parser.add_argument("--episode-seed-base", type=int, default=72_000_000)
    parser.add_argument("--action-seed-base", type=int, default=172_000_003)
    parser.add_argument("--continuation-seed-base", type=int, default=272_000_003)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--max-transitions", type=int, default=1_000)
    parser.add_argument("--disable-multiccd", action="store_true")
    return parser.parse_args()


def tree_sha256(tree):
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_checkpoint(path, epoch, observation, action, *, contextual):
    flags = json.loads((Path(path) / "flags.json").read_text())
    # Importing get_config through the module keeps this script compatible with
    # ml_collections ConfigDict serialization in main.py.
    from agents.context_qgf import get_config as get_context_config
    from agents.qgf import get_config as get_qgf_config

    config = get_context_config() if contextual else get_qgf_config()
    for key, value in flags["agent"].items():
        config[key] = value
    config.agent_name = "context_qgf" if contextual else "qgf"
    agent_cls = ContextQGFAgent if contextual else QGFAgent
    agent = agent_cls.create(
        int(flags["seed"]), observation[None], action[None], config
    )
    return restore_agent(agent, path, epoch), flags


def make_env(args):
    environment = make_ogbench_env_and_datasets(args.env_name, env_only=True)
    environment = EpisodeMonitor(
        environment, filter_regexes=[".*privileged.*", ".*proprio.*"]
    )
    if args.disable_multiccd:
        environment.unwrapped.model.opt.disableflags |= int(
            mujoco.mjtDisableBit.mjDSBL_MULTICCD
        )
    return environment


def action_key(base, episode, chunk):
    return jax.random.fold_in(
        jax.random.PRNGKey(
            (int(base) + 10_007 * int(episode)) % (2**32)
        ),
        int(chunk),
    )


def base_action(agent, observation, key, alpha):
    flat = agent.sample_actions(
        jnp.asarray(observation), seed=key, guidance_weight=float(alpha)
    )
    horizon = int(agent.config.get("horizon_length", 1))
    action_dim = int(agent.config["action_dim"])
    return np.asarray(flat, dtype=np.float32).reshape(horizon, action_dim)


_BOOKKEEPING_KEYS = (
    "episode_length", "reward_sum", "total_timesteps", "_elapsed_steps",
    "checked_step", "_prev_qpos", "_prev_qvel", "_prev_ob_info",
    "_success", "_success_timing", "_n_steps", "_reset_next_step", "_dirty",
)


def physics_state(environment):
    kind = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(environment.unwrapped.model, kind))
    mujoco.mj_getState(environment.unwrapped.model, environment.unwrapped.data, state, kind)
    return state, kind


def restore_physics_state(environment, state, kind):
    mujoco.mj_setState(environment.unwrapped.model, environment.unwrapped.data, state, kind)
    mujoco.mj_forward(environment.unwrapped.model, environment.unwrapped.data)


def bookkeeping_state(environment):
    records = []
    node = environment
    while True:
        records.append((
            node,
            {
                key: copy.deepcopy(getattr(node, key))
                for key in _BOOKKEEPING_KEYS
                if hasattr(node, key)
            },
        ))
        if not hasattr(node, "env"):
            return records
        node = node.env


def restore_bookkeeping(records):
    for node, values in records:
        for key, value in values.items():
            setattr(node, key, copy.deepcopy(value))


def execute_chunk(environment, commands):
    transitions = []
    for command in commands:
        next_observation, reward, terminated, truncated, info = environment.step(command)
        done = bool(terminated or truncated)
        transitions.append((
            np.asarray(next_observation, dtype=np.float32).reshape(-1),
            float(reward), done, info,
        ))
        if done:
            break
    return transitions


def continuation_return(
    environment, native_agent, state, state_kind, bookkeeping, observation,
    first_chunk, discount, alpha, seed_base, episode, query_transition,
    max_transitions, rollout_index,
):
    restore_physics_state(environment, state, state_kind)
    restore_bookkeeping(bookkeeping)
    current = np.asarray(observation, dtype=np.float32)
    total = 0.0
    power = 1.0
    steps = 0
    chunk_index = 0
    done = False
    chunk = first_chunk
    while not done and steps < max_transitions:
        transitions = execute_chunk(environment, chunk)
        for next_observation, reward, done, _ in transitions:
            total += power * reward
            power *= discount
            current = next_observation
            steps += 1
            if done or steps >= max_transitions:
                break
        chunk_index += 1
        if not done and steps < max_transitions:
            key = jax.random.fold_in(
                jax.random.PRNGKey(
                    (
                        int(seed_base)
                        + 100_003 * episode
                        + 1_009 * query_transition
                        + 97_003 * rollout_index
                    )
                    % (2**32)
                ),
                chunk_index,
            )
            chunk = base_action(native_agent, current, key, alpha)
    return total


def q_native(agent, observation, action):
    values = agent.target_critic(
        jnp.asarray(observation)[None], jnp.asarray(action)[None]
    )
    return float(np.asarray(agent._aggregate_q(values)).reshape(-1)[0])


def q_context(agent, observation, action, mean, ready):
    values = agent.target_critic(
        jnp.asarray(observation)[None],
        jnp.asarray(action)[None],
        jnp.asarray(mean),
        jnp.asarray(ready),
    )
    return float(np.asarray(agent._aggregate_q(values)).reshape(-1)[0])


def metrics(rows, field):
    prediction = np.asarray([row[field] for row in rows], dtype=np.float64)
    target = np.asarray([row["mc_return_mean"] for row in rows], dtype=np.float64)
    error = prediction - target
    return {
        "count": len(rows),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "pearson": float(np.corrcoef(prediction, target)[0, 1])
        if len(rows) > 1 and prediction.std() > 1e-12 and target.std() > 1e-12
        else None,
    }


def delta_bootstrap(rows, draws, seed):
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)
    episode_delta = np.asarray([
        np.mean([
            abs(item["context_q"] - item["mc_return_mean"])
            - abs(item["native_q"] - item["mc_return_mean"])
            for item in by_episode[episode]
        ])
        for episode in sorted(by_episode)
    ])
    rng = np.random.default_rng(seed)
    draws_array = np.asarray([
        rng.choice(episode_delta, size=len(episode_delta), replace=True).mean()
        for _ in range(draws)
    ])
    return {
        "mean": float(episode_delta.mean()),
        "ci95": [float(np.quantile(draws_array, 0.025)), float(np.quantile(draws_array, 0.975))],
        "draws": int(draws),
        "definition": "|Q_context-G_MC| - |Q_native-G_MC|; negative favors Context-Q",
    }


def main():
    args = parse_args()
    if args.episodes <= 0 or args.mc_rollouts <= 0:
        raise ValueError("episodes and mc-rollouts must be positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    environment = make_env(args)
    observation, _ = environment.reset(seed=args.episode_seed_base, options={"task_id": None})
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    action = np.zeros(environment.action_space.shape, dtype=np.float32)
    native, native_flags = load_checkpoint(
        args.native_checkpoint, args.epoch, observation, action, contextual=False
    )
    context, context_flags = load_checkpoint(
        args.context_checkpoint, args.epoch, observation, action, contextual=True
    )
    discount = float(context.config["discount"])
    minimum = int(context.config["min_context_transitions"])
    normalization = context.config.get("context_normalization", None)
    include_reward = bool(context.config.get("context_include_reward", True))
    rows = []

    before = {
        "native": tree_sha256(native.target_critic.params),
        "context": tree_sha256(context.target_critic.params),
        "encoder": tree_sha256(context.context_encoder.params),
    }
    for episode in range(args.episodes):
        episode_seed = args.episode_seed_base + episode
        observation, _ = environment.reset(seed=episode_seed, options={"task_id": None})
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        history = []
        transition = 0
        chunk_index = 0
        done = False
        while not done and transition < args.max_transitions:
            key = action_key(args.action_seed_base, episode, chunk_index)
            commands = base_action(native, observation, key, args.guidance_weight)
            if transition in args.query_transitions and len(history) >= minimum:
                state, state_kind = physics_state(environment)
                bookkeeping = bookkeeping_state(environment)
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
                ready = np.asarray([1.0], dtype=np.float32)
                flat_action = commands.reshape(-1)
                returns = [
                    continuation_return(
                        environment, native, state, state_kind, bookkeeping,
                        observation, commands, discount, args.guidance_weight,
                        args.continuation_seed_base, episode, transition,
                        args.max_transitions - transition, rollout_index,
                    )
                    for rollout_index in range(args.mc_rollouts)
                ]
                restore_physics_state(environment, state, state_kind)
                restore_bookkeeping(bookkeeping)
                mc_mean = float(np.mean(returns))
                rows.append({
                    "episode": episode,
                    "transition": transition,
                    "native_q": q_native(native, observation, flat_action),
                    "context_q": q_context(context, observation, flat_action, mean, ready),
                    "mc_return_mean": mc_mean,
                    "mc_return_std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
                })

            transitions = execute_chunk(environment, commands)
            for command_index, (next_observation, reward, done, _) in enumerate(
                transitions
            ):
                history.append(
                    transition_token_numpy(
                        observation,
                        commands[command_index],
                        reward,
                        next_observation,
                        done,
                        normalization=normalization,
                        include_reward=include_reward,
                    )
                )
                history = history[-int(context.config["context_length"]):]
                observation = next_observation
                transition += 1
                if done:
                    break
            chunk_index += 1

    after = {
        "native": tree_sha256(native.target_critic.params),
        "context": tree_sha256(context.target_critic.params),
        "encoder": tree_sha256(context.context_encoder.params),
    }
    if before != after:
        raise AssertionError("evaluation changed checkpoint parameters")
    if not rows:
        raise RuntimeError("no query points collected")
    result = {
        "protocol": {
            "env_name": args.env_name,
            "episodes": args.episodes,
            "query_transitions": list(args.query_transitions),
            "mc_rollouts": args.mc_rollouts,
            "hidden_gain": 1.0,
            "history_reward_input": include_reward,
            "continuation": "frozen native QGF with the same alpha and seeds",
            "mc_target": "discounted continuation return, not Q-star",
            "disable_multiccd": bool(args.disable_multiccd),
            "native_flags_seed": native_flags["seed"],
            "context_flags_seed": context_flags["seed"],
        },
        "native": metrics(rows, "native_q"),
        "context": metrics(rows, "context_q"),
        "delta_mc": delta_bootstrap(rows, args.bootstrap_draws, 20260917),
        "query_count": len(rows),
        "parameter_immutable": True,
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
