"""Strict provenance and initialization checks for Context-Q adapter transfer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util


# These govern the new fine-tuning optimizer, not how the source checkpoint is
# interpreted.  Every architecture, inference, and TD-target semantic is
# checked exactly before parameters are transferred.
FINETUNE_OVERRIDE_KEYS = frozenset(
    {"agent_name", "batch_size", "bc_lr", "critic_lr", "value_lr", "tau"}
)
TRANSFER_FLAG_KEYS = ("env_name", "reward_scale", "reward_bias", "sparse")


def _jsonable(value):
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256_file(path):
    """Return a streaming SHA-256 digest without loading a checkpoint at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_file(checkpoint_dir, epoch):
    path = Path(checkpoint_dir) / f"params_{int(epoch)}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


def load_native_qgf_config(checkpoint_dir):
    """Read a native QGF run's immutable creation config from ``flags.json``."""

    path = Path(checkpoint_dir) / "flags.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Native adapter transfer requires source provenance file: {path}"
        )
    flags = json.loads(path.read_text())
    agent_flags = flags.get("agent")
    if not isinstance(agent_flags, dict):
        raise ValueError(f"{path} has no serialized agent configuration")
    source_agent_name = agent_flags.get("agent_name")
    if source_agent_name != "qgf":
        raise ValueError(
            "Adapter transfer requires a native qgf source checkpoint; "
            f"flags.json reports agent_name={source_agent_name!r}"
        )

    from agents.qgf import get_config as get_qgf_config

    config = get_qgf_config()
    for key, value in agent_flags.items():
        config[key] = value
    config["agent_name"] = "qgf"
    return config, flags


def assert_transfer_compatible(source_config, destination_config, source_flags, destination_flags):
    """Fail early when a source QGF means something different after transfer."""

    source = _jsonable(source_config)
    destination = _jsonable(destination_config)
    mismatches = {}
    for key, source_value in source.items():
        if key in FINETUNE_OVERRIDE_KEYS or key == "action_dim":
            continue
        if key not in destination:
            mismatches[f"agent.{key}"] = {"source": source_value, "destination": "<missing>"}
        elif destination[key] != source_value:
            mismatches[f"agent.{key}"] = {
                "source": source_value,
                "destination": destination[key],
            }
    for key in TRANSFER_FLAG_KEYS:
        if key in source_flags and _jsonable(source_flags[key]) != _jsonable(destination_flags[key]):
            mismatches[key] = {
                "source": _jsonable(source_flags[key]),
                "destination": _jsonable(destination_flags[key]),
            }
    if mismatches:
        detail = json.dumps(mismatches, sort_keys=True)
        raise ValueError(
            "Native checkpoint and adapter run have incompatible semantics: " + detail
        )
    return {
        "allowed_finetune_overrides": sorted(FINETUNE_OVERRIDE_KEYS),
        "source_agent_config": source,
        "destination_agent_config": destination,
        "semantic_mismatches": {},
    }


def source_backbone_finetune_config(source_config, destination_config, *, agent_name):
    """Retain the source model semantics while applying allowed new optimizer knobs."""

    config = dict(source_config)
    for key in FINETUNE_OVERRIDE_KEYS:
        if key in destination_config:
            config[key] = destination_config[key]
    config["agent_name"] = agent_name
    return config


def _tree_sha256(tree, *, include=None):
    digest = hashlib.sha256()
    flattened = traverse_util.flatten_dict(tree)
    for path in sorted(flattened):
        if include is not None and not include(path):
            continue
        value = np.ascontiguousarray(np.asarray(flattened[path]))
        digest.update(repr(path).encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _max_abs_difference(left, right):
    return float(jnp.max(jnp.abs(jnp.asarray(left) - jnp.asarray(right))))


def adapter_initialization_audit(native, adapter, example_batch, *, source_checkpoint, source_epoch, compatibility):
    """Verify zero-adapter equivalence on a ready history before any update."""

    observation = jnp.asarray(example_batch["observations"][:1])
    action = jnp.asarray(example_batch["actions"][:1])
    if bool(adapter.config.get("action_chunking", False)):
        action = jnp.concatenate(
            [action] * int(adapter.config["horizon_length"]), axis=-1
        )
    length = int(adapter.config["context_length"])
    token_dim = int(adapter.config["context_token_dim"])
    minimum = int(adapter.config["min_context_transitions"])
    context = jnp.reshape(
        jnp.arange(length * token_dim, dtype=observation.dtype), (1, length, token_dim)
    ) / max(length * token_dim, 1)
    mask = jnp.zeros((1, length), dtype=observation.dtype)
    mask = mask.at[:, max(0, length - minimum) :].set(1.0)
    mean, _ = adapter.infer_posterior(context, mask)
    ready = (jnp.sum(mask, axis=-1) >= minimum).astype(observation.dtype)

    native_q = native._aggregate_q(native.target_critic(observation, action))[0]
    adapter_q = adapter.q_values(observation, action, mean, ready)[0]

    native_grad = jax.grad(
        lambda candidate: native._aggregate_q(
            native.target_critic(observation, candidate)
        ).sum()
    )(action)
    adapter_grad = jax.grad(
        lambda candidate: adapter.q_values(observation, candidate, mean, ready).sum()
    )(action)
    key = jax.random.PRNGKey(20260920)
    native_alpha_zero = native.sample_actions(
        observation, seed=key, guidance_weight=0.0
    )
    adapter_alpha_zero = adapter.sample_actions(
        observation,
        seed=key,
        guidance_weight=0.0,
        context=context,
        context_mask=mask,
        deterministic_latent=True,
    )
    native_alpha = native.sample_actions(observation, seed=key, guidance_weight=0.04)
    adapter_alpha = adapter.sample_actions(
        observation,
        seed=key,
        guidance_weight=0.04,
        context=context,
        context_mask=mask,
        deterministic_latent=True,
    )

    adapter_param_trees = (
        adapter.critic.params,
        adapter.target_critic.params,
        adapter.value.params,
    )
    flat = [traverse_util.flatten_dict(params) for params in adapter_param_trees]
    context_abs_max = max(
        (
            float(jnp.max(jnp.abs(value)))
            for tree in flat
            for path, value in tree.items()
            if "ContextInput" in path
        ),
        default=0.0,
    )
    differences = {
        "target_q_max_abs": _max_abs_difference(native_q, adapter_q),
        "target_q_gradient_max_abs": _max_abs_difference(native_grad, adapter_grad),
        "guided_action_alpha_0_max_abs": _max_abs_difference(
            native_alpha_zero, adapter_alpha_zero
        ),
        "guided_action_alpha_0p04_max_abs": _max_abs_difference(
            native_alpha, adapter_alpha
        ),
    }
    tolerance = 2e-6
    failures = {key: value for key, value in differences.items() if value > tolerance}
    non_adapter = lambda path: "ContextInput" not in path
    parameter_sha256 = {
        "native_policy": _tree_sha256(native.policy.params),
        "adapter_policy": _tree_sha256(adapter.policy.params),
        "native_critic": _tree_sha256(native.critic.params),
        "adapter_critic_frozen_leaves": _tree_sha256(
            adapter.critic.params, include=non_adapter
        ),
        "native_target_critic": _tree_sha256(native.target_critic.params),
        "adapter_target_critic_frozen_leaves": _tree_sha256(
            adapter.target_critic.params, include=non_adapter
        ),
        "native_value": _tree_sha256(native.value.params),
        "adapter_value_frozen_leaves": _tree_sha256(
            adapter.value.params, include=non_adapter
        ),
    }
    copied_hashes_match = (
        parameter_sha256["native_policy"] == parameter_sha256["adapter_policy"]
        and parameter_sha256["native_critic"]
        == parameter_sha256["adapter_critic_frozen_leaves"]
        and parameter_sha256["native_target_critic"]
        == parameter_sha256["adapter_target_critic_frozen_leaves"]
        and parameter_sha256["native_value"]
        == parameter_sha256["adapter_value_frozen_leaves"]
    )
    if context_abs_max != 0.0 or failures or not copied_hashes_match:
        raise AssertionError(
            "Adapter initialization is not native-equivalent: "
            + json.dumps(
                {
                    "context_input_abs_max": context_abs_max,
                    "copied_hashes_match": copied_hashes_match,
                    **differences,
                }
            )
        )

    source_file = checkpoint_file(source_checkpoint, source_epoch)
    return {
        "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "source_epoch": int(source_epoch),
        "source_checkpoint_file": str(source_file.resolve()),
        "source_checkpoint_sha256": sha256_file(source_file),
        "source_flags_sha256": sha256_file(Path(source_checkpoint) / "flags.json"),
        "compatibility": compatibility,
        "ready_history_transitions": int(np.asarray(mask).sum()),
        "context_input_abs_max": context_abs_max,
        "equivalence_tolerance": tolerance,
        "equivalence_max_abs": differences,
        "copied_hashes_match": copied_hashes_match,
        "parameter_sha256": parameter_sha256,
    }
