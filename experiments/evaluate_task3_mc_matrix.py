#!/usr/bin/env python3
"""Two-by-two Task3 MC calibration matrix for native QGF and Context-Q.

At each paired query point this script computes:

    G_native  = frozen native QGF continuation return
    G_context = Context-Q continuation return
    Q_native  = native critic value
    Q_context = Context-Q critic value

and reports the error matrix Q_x versus G_y.
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
import numpy as np

from experiments.evaluate_task3_mc import (
    action_key,
    base_action,
    bookkeeping_state,
    execute_chunk,
    load_checkpoint,
    make_env,
    parse_ints,
    physics_state,
    restore_bookkeeping,
    restore_physics_state,
    tree_sha256,
)
from experiments.checkpoint_protocol import (
    add_paired_checkpoint_epochs,
    paired_checkpoint_epochs,
)
from utils.context import (
    context_is_ready_numpy,
    pad_context_numpy,
    transition_token_numpy,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="cube-triple-play-singletask-task3-v0")
    parser.add_argument("--native-checkpoint", required=True)
    parser.add_argument("--context-checkpoint", required=True)
    add_paired_checkpoint_epochs(parser)
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


def atomic_json(path: Path, value):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sample_continuation_action(agent, observation, key, alpha, *, contextual, history):
    if not contextual:
        flat = agent.sample_actions(
            jnp.asarray(observation),
            seed=key,
            guidance_weight=float(alpha),
        )
    else:
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
    horizon = int(agent.config.get("horizon_length", 1))
    action_dim = int(agent.config["action_dim"])
    return np.asarray(flat, dtype=np.float32).reshape(horizon, action_dim)


def continuation_return(
    environment,
    agent,
    *,
    contextual,
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
    chunk = np.asarray(first_chunk, dtype=np.float32)

    while not done and steps < max_transitions:
        transitions = execute_chunk(environment, chunk)
        for command, (next_observation, reward, done, _) in zip(chunk, transitions):
            total += power * reward
            power *= discount
            steps += 1
            if contextual:
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
                current_history = current_history[-int(agent.config["context_length"]) :]
            current = next_observation
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
            contextual=contextual,
            history=current_history,
        )

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


def metrics(rows, prediction_field, target_field):
    prediction = np.asarray([row[prediction_field] for row in rows], dtype=np.float64)
    target = np.asarray([row[target_field] for row in rows], dtype=np.float64)
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


def delta_bootstrap(rows, prediction_a, prediction_b, target_field, draws, seed):
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(row)
    episode_delta = np.asarray([
        np.mean([
            abs(item[prediction_a] - item[target_field])
            - abs(item[prediction_b] - item[target_field])
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
        "ci95": [
            float(np.quantile(draws_array, 0.025)),
            float(np.quantile(draws_array, 0.975)),
        ],
        "draws": int(draws),
        "definition": f"|{prediction_a}-{target_field}| - |{prediction_b}-{target_field}|",
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

    native_epoch, context_epoch = paired_checkpoint_epochs(args)
    native, native_flags = load_checkpoint(
        args.native_checkpoint, native_epoch, observation, action, contextual=False
    )
    context, context_flags = load_checkpoint(
        args.context_checkpoint, context_epoch, observation, action, contextual=True
    )
    discount = float(context.config["discount"])
    minimum = int(context.config["min_context_transitions"])
    normalization = context.config.get("context_normalization", None)
    include_reward = bool(context.config.get("context_include_reward", True))

    before = {
        "native": tree_sha256(native.target_critic.params),
        "context": tree_sha256(context.target_critic.params),
        "encoder": tree_sha256(context.context_encoder.params),
    }
    rows = []

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
                ready = context_is_ready_numpy(context_mask, minimum, dtype=np.float32)

                flat_action = commands.reshape(-1)
                native_returns = []
                context_returns = []
                for rollout_index in range(args.mc_rollouts):
                    native_returns.append(
                        continuation_return(
                            environment,
                            native,
                            contextual=False,
                            state=state,
                            state_kind=state_kind,
                            bookkeeping=bookkeeping,
                            observation=observation,
                            first_chunk=commands,
                            history=history,
                            discount=discount,
                            alpha=args.guidance_weight,
                            seed_base=args.continuation_seed_base,
                            episode=episode,
                            query_transition=transition,
                            max_transitions=args.max_transitions - transition,
                            rollout_index=rollout_index,
                            normalization=normalization,
                            include_reward=include_reward,
                        )
                    )
                    context_returns.append(
                        continuation_return(
                            environment,
                            context,
                            contextual=True,
                            state=state,
                            state_kind=state_kind,
                            bookkeeping=bookkeeping,
                            observation=observation,
                            first_chunk=commands,
                            history=history,
                            discount=discount,
                            alpha=args.guidance_weight,
                            seed_base=args.continuation_seed_base,
                            episode=episode,
                            query_transition=transition,
                            max_transitions=args.max_transitions - transition,
                            rollout_index=rollout_index,
                            normalization=normalization,
                            include_reward=include_reward,
                        )
                    )
                restore_physics_state(environment, state, state_kind)
                restore_bookkeeping(bookkeeping)
                rows.append(
                    {
                        "episode": episode,
                        "transition": transition,
                        "native_q": q_native(native, observation, flat_action),
                        "context_q": q_context(context, observation, flat_action, mean, ready),
                        "mc_native_return_mean": float(np.mean(native_returns)),
                        "mc_context_return_mean": float(np.mean(context_returns)),
                        "mc_native_return_std": float(np.std(native_returns, ddof=1))
                        if len(native_returns) > 1
                        else 0.0,
                        "mc_context_return_std": float(np.std(context_returns, ddof=1))
                        if len(context_returns) > 1
                        else 0.0,
                    }
                )

            transitions = execute_chunk(environment, commands)
            for command_index, (next_observation, reward, done, _) in enumerate(transitions):
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
                history = history[-int(context.config["context_length"]) :]
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
            "alpha": args.guidance_weight,
            "native_epoch": native_epoch,
            "context_epoch": context_epoch,
            "native_continuation": "frozen native QGF with the same alpha and seeds",
            "context_continuation": "Context-Q with causal history and the same alpha and seeds",
            "disable_multiccd": bool(args.disable_multiccd),
            "native_flags_seed": native_flags["seed"],
            "context_flags_seed": context_flags["seed"],
        },
        "native_continuation": {
            "native_q": metrics(rows, "native_q", "mc_native_return_mean"),
            "context_q": metrics(rows, "context_q", "mc_native_return_mean"),
            "delta_context_minus_native": delta_bootstrap(
                rows,
                "context_q",
                "native_q",
                "mc_native_return_mean",
                args.bootstrap_draws,
                20260919,
            ),
        },
        "context_continuation": {
            "native_q": metrics(rows, "native_q", "mc_context_return_mean"),
            "context_q": metrics(rows, "context_q", "mc_context_return_mean"),
            "delta_context_minus_native": delta_bootstrap(
                rows,
                "context_q",
                "native_q",
                "mc_context_return_mean",
                args.bootstrap_draws,
                20260920,
            ),
        },
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
