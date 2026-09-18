#!/usr/bin/env python3
"""Check whether Q improves candidate-action ordering versus MC returns."""

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
from experiments.evaluate_task3_mc_matrix import continuation_return
from utils.context import pad_context_numpy, transition_token_numpy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-checkpoint", required=True)
    parser.add_argument("--context-checkpoint", required=True)
    parser.add_argument("--epoch", type=int, default=500_000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-name", default="cube-triple-play-singletask-task3-v0")
    parser.add_argument("--guidance-weight", type=float, default=0.04)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--query-transitions", type=lambda s: [int(x) for x in s.split(",")], default=(20, 40, 60))
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--mc-rollouts", type=int, default=4)
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


def pairwise_accuracy(predictions, actuals):
    correct = 0
    total = 0
    for i in range(len(predictions)):
        for j in range(len(predictions)):
            if i == j or np.isclose(actuals[i], actuals[j]):
                continue
            total += 1
            if (predictions[i] > predictions[j]) == (actuals[i] > actuals[j]):
                correct += 1
    return correct, total


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
    native_correct = 0
    context_correct = 0
    total_pairs = 0

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
            commands = base_action(
                native,
                current,
                action_key(args.action_seed_base, episode_index, chunk_index),
                args.guidance_weight,
            )

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

                base_key = action_key(
                    args.action_seed_base, episode_index, chunk_index
                )
                candidate_q_native = []
                candidate_q_context = []
                candidate_mc = []
                candidate_actions = []
                for candidate_index in range(args.candidates):
                    key = jax.random.fold_in(base_key, 10_000 + candidate_index)
                    flat = native.sample_actions(
                        jnp.asarray(current),
                        seed=key,
                        guidance_weight=args.guidance_weight,
                    )
                    candidate = np.asarray(flat, dtype=np.float32).reshape(
                        int(native.config["horizon_length"]),
                        int(native.config["action_dim"]),
                    )
                    candidate_actions.append(candidate)
                    candidate_q_native.append(
                        q_native(native, current, candidate.reshape(-1))
                    )
                    candidate_q_context.append(
                        q_context(
                            hybrid,
                            current,
                            candidate.reshape(-1),
                            mean,
                            ready,
                        )
                    )
                    returns = []
                    for rollout_index in range(args.mc_rollouts):
                        returns.append(
                            continuation_return(
                                environment,
                                native,
                                contextual=False,
                                state=state,
                                state_kind=state_kind,
                                bookkeeping=bookkeeping,
                                observation=current,
                                first_chunk=candidate,
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
                        )
                    candidate_mc.append(float(np.mean(returns)))
                restore_physics_state(environment, state, state_kind)
                restore_bookkeeping(bookkeeping)

                n_correct, n_total = pairwise_accuracy(
                    candidate_q_native, candidate_mc
                )
                c_correct, c_total = pairwise_accuracy(
                    candidate_q_context, candidate_mc
                )
                native_correct += n_correct
                context_correct += c_correct
                total_pairs += n_total
                rows.append(
                    {
                        "episode": episode_index,
                        "transition": transition,
                        "candidates": args.candidates,
                        "native_pairwise_correct": n_correct,
                        "context_pairwise_correct": c_correct,
                        "pairs": n_total,
                        "native_q": candidate_q_native,
                        "context_q": candidate_q_context,
                        "mc_return": candidate_mc,
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
            "candidates": args.candidates,
            "mc_rollouts": args.mc_rollouts,
            "continuation": "native QGF",
        },
        "native_pairwise_accuracy": float(native_correct / total_pairs)
        if total_pairs
        else None,
        "context_pairwise_accuracy": float(context_correct / total_pairs)
        if total_pairs
        else None,
        "pairs": total_pairs,
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
