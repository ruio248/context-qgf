"""Causal PEARL-style context inference utilities.

The deployable token is ``[state, commanded_action, reward, state_delta, done]``.
Only completed transitions may be placed in a history.  In particular, the
reward at decision ``t`` first becomes visible to the posterior at ``t + 1``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class CausalPearlEncoder(nn.Module):
    """Order-sensitive diagonal-Gaussian posterior over a causal history.

    An all-masked history returns the prior ``N(0, I)`` exactly.  Masked token
    values never affect the carried GRU state.
    """

    latent_dim: int
    token_hidden_dims: Sequence[int] = (256, 256)
    gru_hidden_dim: int = 128
    min_log_variance: float = -8.0
    max_log_variance: float = 4.0

    @nn.compact
    def __call__(self, context, mask):
        context = jnp.asarray(context)
        mask = jnp.asarray(mask, dtype=context.dtype)
        if context.ndim < 3:
            raise ValueError("context must have shape (..., length, token_dim)")
        if mask.shape != context.shape[:-1]:
            raise ValueError(
                f"mask shape {mask.shape} must match context prefix {context.shape[:-1]}"
            )

        features = context
        for index, width in enumerate(self.token_hidden_dims):
            features = nn.Dense(width, name=f"TokenDense_{index}")(features)
            features = nn.gelu(features)

        carry = jnp.zeros(
            (*context.shape[:-2], self.gru_hidden_dim), dtype=context.dtype
        )
        cell = nn.GRUCell(self.gru_hidden_dim, name="GRU")
        for timestep in range(context.shape[-2]):
            candidate, _ = cell(carry, features[..., timestep, :])
            valid = mask[..., timestep, None]
            carry = valid * candidate + (1.0 - valid) * carry

        mean = nn.Dense(
            self.latent_dim,
            bias_init=nn.initializers.zeros,
            name="MeanHead",
        )(carry)
        log_variance = nn.Dense(
            self.latent_dim,
            bias_init=nn.initializers.zeros,
            name="LogVarianceHead",
        )(carry)
        log_variance = jnp.clip(
            log_variance, self.min_log_variance, self.max_log_variance
        )

        ready = jnp.any(mask > 0, axis=-1, keepdims=True)
        mean = jnp.where(ready, mean, jnp.zeros_like(mean))
        log_variance = jnp.where(
            ready, log_variance, jnp.zeros_like(log_variance)
        )
        return mean, log_variance


def sample_latent(mean, log_variance, rng, *, deterministic=False):
    """Sample a posterior latent with the reparameterization trick."""

    if deterministic:
        return mean
    noise = jax.random.normal(rng, mean.shape)
    return mean + jnp.exp(0.5 * log_variance) * noise


def kl_to_standard_normal(mean, log_variance):
    """Return per-example ``KL(q(z|C) || N(0, I))``."""

    return 0.5 * jnp.sum(
        jnp.exp(log_variance) + jnp.square(mean) - 1.0 - log_variance,
        axis=-1,
    )


def context_is_ready(mask, reference, min_count=1):
    """Return a batch-shaped gate for histories with enough transitions."""

    return (jnp.sum(mask, axis=-1) >= min_count).astype(reference.dtype)


def context_is_ready_numpy(mask, min_count=1, *, dtype=np.float32):
    """NumPy counterpart of :func:`context_is_ready` for rollout diagnostics.

    Keeping the diagnostic gate in one place prevents an accidental
    ``ready=1`` from querying a contextual critic before its configured
    minimum number of completed transitions is available.
    """

    mask = np.asarray(mask)
    return (np.sum(mask, axis=-1) >= min_count).astype(dtype)


def _normalization_arrays(normalization: Mapping, dtype):
    return {
        key: np.asarray(value, dtype=dtype)
        for key, value in normalization.items()
    }


def transition_token_numpy(
    observation,
    action,
    reward,
    next_observation,
    done,
    *,
    normalization=None,
    include_reward=True,
):
    """Construct one causal transition token for dataset and rollout code."""

    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    next_observation = np.asarray(next_observation, dtype=np.float32).reshape(-1)
    reward = np.asarray(reward, dtype=np.float32).reshape(1)
    done = np.asarray(done, dtype=np.float32).reshape(1)
    delta = next_observation - observation

    if normalization is not None:
        stats = _normalization_arrays(normalization, np.float32)
        observation = (observation - stats["observation_mean"]) / stats[
            "observation_std"
        ]
        action = (action - stats["action_mean"]) / stats["action_std"]
        reward = (reward - stats["reward_mean"]) / stats["reward_std"]
        delta = (delta - stats["delta_mean"]) / stats["delta_std"]
    if not include_reward:
        reward = np.zeros_like(reward)

    return np.concatenate(
        [observation, action, reward, delta, done], axis=-1
    ).astype(np.float32, copy=False)


def pad_context_numpy(history, context_length, token_dim):
    """Left-pad the most recent causal tokens and return tokens plus mask."""

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    tokens = np.zeros((context_length, token_dim), dtype=np.float32)
    mask = np.zeros((context_length,), dtype=np.float32)
    recent = list(history[-context_length:])
    if recent:
        stacked = np.stack(recent).astype(np.float32, copy=False)
        if stacked.shape != (len(recent), token_dim):
            raise ValueError(
                f"history has shape {stacked.shape}; expected ({len(recent)}, {token_dim})"
            )
        tokens[-len(recent) :] = stacked
        mask[-len(recent) :] = 1.0
    return tokens, mask
