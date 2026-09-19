"""Utilities for checkpoint-preserving Context-Q adapter finetuning.

The Context-Q MLP intentionally gives its ordinary layers the same parameter
names as the native QGF MLP.  These helpers use that contract to copy a native
checkpoint into the contextual architecture while leaving the newly added
``ContextInput`` projections at exactly zero.
"""

from __future__ import annotations

import flax
import jax.numpy as jnp
import numpy as np
import optax
from flax import traverse_util


def _flatten(tree):
    return traverse_util.flatten_dict(tree)


def _match_container(reference, value):
    """Keep optimizer label trees structurally identical to parameter trees."""

    if isinstance(reference, flax.core.FrozenDict):
        return flax.core.freeze(value)
    return value


def copy_matching_params(destination, source):
    """Copy same-path, same-shape source leaves into ``destination``.

    ``destination`` may contain Context-Q-only parameters.  They deliberately
    remain untouched so that callers can initialize them separately.
    """

    destination_flat = _flatten(destination)
    source_flat = _flatten(source)
    merged = {}
    for path, value in destination_flat.items():
        source_value = source_flat.get(path)
        if source_value is not None and np.shape(value) == np.shape(source_value):
            merged[path] = source_value
        else:
            merged[path] = value
    return _match_container(destination, traverse_util.unflatten_dict(merged))


def copy_exact_params(destination, source, *, component):
    """Copy a parameter tree only when every path and shape agrees."""

    destination_flat = _flatten(destination)
    source_flat = _flatten(source)
    if set(destination_flat) != set(source_flat):
        missing = sorted(set(source_flat) - set(destination_flat))
        unexpected = sorted(set(destination_flat) - set(source_flat))
        raise ValueError(
            f"{component} parameter paths differ; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    shape_mismatches = [
        path
        for path, value in destination_flat.items()
        if np.shape(value) != np.shape(source_flat[path])
    ]
    if shape_mismatches:
        raise ValueError(
            f"{component} parameter shapes differ at {shape_mismatches[:3]}"
        )
    return source


def assert_source_is_covered(destination, source, *, component):
    """Ensure every native leaf has an identically shaped contextual target."""

    destination_flat = _flatten(destination)
    source_flat = _flatten(source)
    missing = [
        path
        for path, value in source_flat.items()
        if path not in destination_flat
        or np.shape(destination_flat[path]) != np.shape(value)
    ]
    if missing:
        raise ValueError(
            f"{component} cannot be transferred; unmatched native leaves: "
            f"{missing[:3]}"
        )


def zero_context_inputs(params):
    """Set every ContextInput parameter to exactly zero."""

    return _match_container(
        params,
        traverse_util.unflatten_dict(
            {
                path: jnp.zeros_like(value) if "ContextInput" in path else value
                for path, value in _flatten(params).items()
            }
        )
    )


def context_adapter_labels(params):
    """Label ContextInput leaves as trainable and all native leaves frozen."""

    return _match_container(
        params,
        traverse_util.unflatten_dict(
            {
                path: "adapter" if "ContextInput" in path else "frozen"
                for path in _flatten(params)
            }
        )
    )


def make_context_adapter_optimizer(params, learning_rate):
    """Adam for adapter leaves and an exact zero transform for native leaves."""

    return optax.multi_transform(
        {
            "adapter": optax.adam(learning_rate=learning_rate),
            "frozen": optax.set_to_zero(),
        },
        context_adapter_labels(params),
    )


def target_update_context_adapters(model, target_model, tau):
    """Polyak-update ContextInput leaves while keeping native target leaves fixed.

    The native online critic and target critic may already differ at the source
    checkpoint.  Updating every target leaf would silently move the supposedly
    frozen native target backbone toward the online critic during adapter
    finetuning.  Only new adapter leaves are allowed to track their online
    counterparts here.
    """

    model_flat = _flatten(model.params)
    target_flat = _flatten(target_model.params)
    if set(model_flat) != set(target_flat):
        raise ValueError("Online and target Context-Q parameter paths differ")
    updated = {
        path: value * tau + target_flat[path] * (1 - tau)
        if "ContextInput" in path
        else target_flat[path]
        for path, value in model_flat.items()
    }
    params = _match_container(
        target_model.params, traverse_util.unflatten_dict(updated)
    )
    return target_model.replace(params=params)
