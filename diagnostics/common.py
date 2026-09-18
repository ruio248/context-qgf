"""Shared helpers for Context-Q diagnostic scripts."""

from __future__ import annotations

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util


def copy_matching_params(destination, source):
    """Copy leaves from source whose flattened paths and shapes match."""

    destination_flat = traverse_util.flatten_dict(destination)
    source_flat = traverse_util.flatten_dict(source)
    merged = {}
    for path, value in destination_flat.items():
        if path in source_flat and np.shape(value) == np.shape(source_flat[path]):
            merged[path] = source_flat[path]
        else:
            merged[path] = value
    return flax.core.freeze(traverse_util.unflatten_dict(merged))


def zero_context_inputs(params):
    """Zero all ContextInput leaves in a parameter tree."""

    flat = traverse_util.flatten_dict(params)
    return flax.core.freeze(
        traverse_util.unflatten_dict(
            {
                path: jnp.zeros_like(value) if "ContextInput" in path else value
                for path, value in flat.items()
            }
        )
    )


def max_abs_diff(left, right):
    left_flat = traverse_util.flatten_dict(left)
    right_flat = traverse_util.flatten_dict(right)
    if set(left_flat) != set(right_flat):
        raise ValueError("parameter trees have different paths")
    maximum = 0.0
    for path, value in left_flat.items():
        maximum = max(
            maximum,
            float(jnp.max(jnp.abs(value - right_flat[path]))),
        )
    return maximum


def flatten_and_normalize(tree):
    leaves = [jnp.ravel(value) for value in jax.tree_util.tree_leaves(tree)]
    flat = jnp.concatenate(leaves) if leaves else jnp.zeros((0,))
    norm = jnp.linalg.norm(flat)
    return flat, norm


def tree_cosine(left, right):
    left_flat, left_norm = flatten_and_normalize(left)
    right_flat, right_norm = flatten_and_normalize(right)
    denominator = left_norm * right_norm
    if float(denominator) <= 1e-12:
        return 0.0
    return float(jnp.dot(left_flat, right_flat) / denominator)


def build_fixed_actor_context_agent(native, context):
    """Use native actor parameters inside a Context-Q agent."""

    return context.replace(
        policy=context.policy.replace(
            params=copy_matching_params(context.policy.params, native.policy.params)
        )
    )
