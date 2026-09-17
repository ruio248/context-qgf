"""PEARL-style Context-Q trained jointly from the first offline update.

The flow actor is the unmodified QGF behavior-cloning policy.  Only Q and V
are conditioned on a stochastic latent inferred from completed transitions.
There is no pretrained-backbone or adapter stage in this implementation.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.qgf import QGFAgent
from agents.qgf import get_config as get_qgf_config
from utils.activation import get_activation
from utils.context import (
    CausalPearlEncoder,
    context_is_ready,
    kl_to_standard_normal,
    sample_latent,
)
from utils.flax_utils import TrainState, expectile_loss, target_update
from utils.networks import ActorFlowField, ContextValue


class ContextQGFAgent(QGFAgent):
    """QGF whose critic/value use a causal PEARL-style task belief."""

    support_context = True

    context_encoder: TrainState

    def _posterior(
        self,
        context,
        context_mask,
        rng,
        *,
        context_params=None,
        deterministic=False,
    ):
        mean, log_variance = self.context_encoder(
            context,
            context_mask,
            params=context_params,
        )
        latent = sample_latent(
            mean,
            log_variance,
            rng,
            deterministic=deterministic,
        )
        ready = context_is_ready(
            context_mask,
            mean,
            min_count=int(self.config["min_context_transitions"]),
        )
        return latent, mean, log_variance, ready

    def _batch_posterior(
        self,
        batch,
        rng,
        *,
        context_params=None,
        next_context=False,
        deterministic=False,
    ):
        prefix = "next_" if next_context else ""
        context_key = f"{prefix}context"
        mask_key = f"{prefix}context_mask"
        if context_key not in batch or mask_key not in batch:
            raise KeyError(
                "Context-Q batches require context, context_mask, "
                "next_context, and next_context_mask"
            )
        return self._posterior(
            batch[context_key],
            batch[mask_key],
            rng,
            context_params=context_params,
            deterministic=deterministic,
        )

    def infer_posterior(self, context, context_mask):
        """Expose posterior statistics for evaluation and audits."""

        return self.context_encoder(context, context_mask)

    def critic_loss(
        self,
        batch,
        critic_params=None,
        context_params=None,
        rng=None,
    ):
        """Contextual IQL TD loss plus the PEARL information bottleneck."""

        if rng is None:
            rng = self.rng
        current_rng, next_rng = jax.random.split(rng)
        batch_actions, next_obs, rewards, masks, valid_w = self._get_flat_batch(batch)
        latent, mean, log_variance, ready = self._batch_posterior(
            batch,
            current_rng,
            context_params=context_params,
            deterministic=bool(self.config["deterministic_context_training"]),
        )
        next_latent, _, _, next_ready = self._batch_posterior(
            batch,
            next_rng,
            context_params=context_params,
            next_context=True,
            deterministic=bool(self.config["deterministic_context_training"]),
        )

        horizon = int(self.config.get("horizon_length", 1))
        next_value = self.value(next_obs, None, next_latent, next_ready)
        target_q = rewards + (self.config["discount"] ** horizon) * masks * next_value
        target_q = jax.lax.stop_gradient(target_q)
        qs = self.critic(
            batch["observations"],
            batch_actions,
            latent,
            ready,
            params=critic_params,
        )
        td_loss = (((qs - target_q[None]) ** 2) * valid_w).mean()
        kl = kl_to_standard_normal(mean, log_variance).mean()
        loss = td_loss + self.config["context_kl_weight"] * kl
        return loss, {
            "critic_loss": td_loss,
            "context_kl": kl,
            "q": qs[0].mean(),
            "posterior_std": jnp.exp(0.5 * log_variance).mean(),
            "context_ready_fraction": ready.mean(),
        }

    def value_loss(
        self,
        batch,
        value_params=None,
        context_params=None,
        rng=None,
    ):
        """Contextual IQL expectile value loss with a stopped latent."""

        if rng is None:
            rng = self.rng
        batch_actions, _, _, _, valid_w = self._get_flat_batch(batch)
        latent, _, _, ready = self._batch_posterior(
            batch,
            rng,
            context_params=context_params,
            deterministic=bool(self.config["deterministic_context_training"]),
        )
        latent = jax.lax.stop_gradient(latent)
        ready = jax.lax.stop_gradient(ready)
        qs = self.target_critic(
            batch["observations"], batch_actions, latent, ready
        )
        q = self._aggregate_q(qs)
        value = self.value(
            batch["observations"],
            None,
            latent,
            ready,
            params=value_params,
        )
        loss = (expectile_loss(q - value, self.config["expectile"]) * valid_w).mean()
        return loss, {"value_loss": loss, "v": value.mean()}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Validation loss compatible with the native training driver."""

        if rng is None:
            rng = self.rng
        policy_rng = jax.random.fold_in(rng, 0)
        critic_rng = jax.random.fold_in(rng, 1)
        value_rng = jax.random.fold_in(rng, 2)
        policy_params = self.policy.params if grad_params is None else grad_params
        policy_loss, policy_info = self.policy_loss(
            batch, policy_params=policy_params, rng=policy_rng
        )
        critic_loss, critic_info = self.critic_loss(
            batch,
            context_params=self.context_encoder.params,
            rng=critic_rng,
        )
        value_loss, value_info = self.value_loss(
            batch,
            context_params=self.context_encoder.params,
            rng=value_rng,
        )
        info = {
            **{f"policy/{key}": value for key, value in policy_info.items()},
            **{f"critic/{key}": value for key, value in critic_info.items()},
            **{f"value/{key}": value for key, value in value_info.items()},
        }
        return policy_loss + critic_loss + value_loss, info

    @jax.jit
    def update(self, batch):
        """Jointly train actor, contextual Q/V, and encoder from scratch."""

        # Preserve native QGF's actor RNG stream exactly.  This makes a QGF and
        # Context-Q run with the same seed and sampled indices share the actor.
        new_rng, policy_rng = jax.random.split(self.rng, 2)
        critic_rng = jax.random.fold_in(self.rng, 1)
        value_rng = jax.random.fold_in(self.rng, 2)

        new_policy, policy_info = self.policy.apply_loss_fn(
            loss_fn=lambda params: self.policy_loss(
                batch, policy_params=params, rng=policy_rng
            )
        )

        def critic_context_loss(critic_params, context_params):
            return self.critic_loss(
                batch,
                critic_params=critic_params,
                context_params=context_params,
                rng=critic_rng,
            )

        (_, critic_info), (critic_grads, context_grads) = jax.value_and_grad(
            critic_context_loss,
            argnums=(0, 1),
            has_aux=True,
        )(self.critic.params, self.context_encoder.params)
        new_critic = self.critic.apply_gradients(grads=critic_grads)
        new_context_encoder = self.context_encoder.apply_gradients(
            grads=context_grads
        )
        new_target_critic = target_update(
            self.critic, self.target_critic, self.config["tau"]
        )

        new_value, value_info = self.value.apply_loss_fn(
            loss_fn=lambda params: self.value_loss(
                batch,
                value_params=params,
                context_params=self.context_encoder.params,
                rng=value_rng,
            )
        )
        return self.replace(
            rng=new_rng,
            policy=new_policy,
            critic=new_critic,
            target_critic=new_target_critic,
            value=new_value,
            context_encoder=new_context_encoder,
        ), {**policy_info, **critic_info, **value_info}

    def q_values(self, observations, actions, latent, context_ready):
        """Return the aggregated target-Q used by guidance and MC audits."""

        return self._aggregate_q(
            self.target_critic(observations, actions, latent, context_ready)
        )

    @partial(
        jax.jit,
        static_argnames=["rejection_sampling", "deterministic_latent"],
    )
    def sample_actions(
        self,
        observations: jnp.ndarray,
        *,
        seed: Any,
        guidance_weight: float = 1.0,
        rejection_sampling: int = 1,
        context=None,
        context_mask=None,
        latent=None,
        context_ready=None,
        deterministic_latent: bool = True,
    ) -> jnp.ndarray:
        """Run native QGF denoising with one latent fixed for the full scan."""

        has_batch_dim = observations.ndim == 2
        observations = observations if has_batch_dim else observations[None]
        base_batch_size = observations.shape[0]

        if latent is None and context is not None:
            context = context if context.ndim == 3 else context[None]
            context_mask = (
                context_mask if context_mask.ndim == 2 else context_mask[None]
            )
            latent, _, _, inferred_ready = self._posterior(
                context,
                context_mask,
                jax.random.fold_in(seed, 1),
                deterministic=deterministic_latent,
            )
            context_ready = inferred_ready
        elif latent is None:
            latent = jnp.zeros(
                (base_batch_size, self.config["latent_dim"]),
                dtype=observations.dtype,
            )
            context_ready = jnp.zeros(
                (base_batch_size,), dtype=observations.dtype
            )
        else:
            latent = latent if latent.ndim == 2 else latent[None]
            if context_ready is None:
                context_ready = jnp.ones(
                    (latent.shape[0],), dtype=observations.dtype
                )
            elif context_ready.ndim == 0:
                context_ready = jnp.broadcast_to(
                    context_ready, (latent.shape[0],)
                )

        if rejection_sampling > 1:
            observations = jnp.repeat(observations, rejection_sampling, axis=0)
            latent = jnp.repeat(latent, rejection_sampling, axis=0)
            context_ready = jnp.repeat(
                context_ready, rejection_sampling, axis=0
            )

        horizon = int(self.config.get("horizon_length", 1))
        action_dim = int(self.config["action_dim"])
        full_action_dim = action_dim * (
            horizon if self.config.get("action_chunking", False) else 1
        )
        action = jax.random.normal(
            seed, (observations.shape[0], full_action_dim)
        )
        dt = 1.0 / self.config["denoise_steps"]
        approximation = self.config["denoised_action_approx"]
        apply_jacobian = self.config["apply_jacobian"]

        def step(current_action, time_index):
            time = jnp.full(
                (current_action.shape[0],),
                time_index / self.config["denoise_steps"],
            )
            time_column = time[..., None]
            velocity = self.policy(observations, current_action, time)
            if approximation == "noisy":
                action_approx = current_action
            elif approximation == "one_euler_step_approx":
                action_approx = jnp.clip(
                    current_action
                    + (1 - time_column) * jax.lax.stop_gradient(velocity),
                    -1,
                    1,
                )
            else:
                raise ValueError(
                    f"Unsupported denoised_action_approx: {approximation!r}"
                )

            def q_fn(candidate_action):
                return self.q_values(
                    observations, candidate_action, latent, context_ready
                ).sum()

            q_gradient = jax.grad(q_fn)(jax.lax.stop_gradient(action_approx))
            if apply_jacobian:
                if approximation != "one_euler_step_approx":
                    raise ValueError(
                        "apply_jacobian requires one_euler_step_approx"
                    )

                def map_single(action_i, observation_i, time_i):
                    velocity_i = self.policy(
                        observation_i[None], action_i[None], time_i
                    )[0]
                    return jnp.clip(
                        action_i + (1 - time_i[0]) * velocity_i, -1, 1
                    )

                jacobian = jax.vmap(jax.jacrev(map_single, argnums=0))(
                    current_action, observations, time_column
                )
                q_gradient = jnp.einsum("bi,bij->bj", q_gradient, jacobian)
            return current_action + (
                velocity + guidance_weight * q_gradient
            ) * dt, None

        actions, _ = jax.lax.scan(
            step,
            action,
            jnp.arange(self.config["denoise_steps"]),
            length=self.config["denoise_steps"],
        )
        actions = jnp.clip(actions, -1, 1)
        if rejection_sampling > 1:
            q = self.q_values(
                observations, actions, latent, context_ready
            ).reshape((base_batch_size, rejection_sampling))
            actions = actions.reshape(
                (base_batch_size, rejection_sampling, *actions.shape[1:])
            )
            actions = actions[jnp.arange(base_batch_size), jnp.argmax(q, axis=1)]
        return actions if has_batch_dim else actions[0]

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        if ex_observations.ndim != 2:
            raise ValueError("Context-Q currently supports vector observations only")

        # Match QGF's split exactly for actor/Q/V, and derive the additional
        # encoder key without perturbing the agent RNG returned by QGF.create.
        rng = jax.random.PRNGKey(seed)
        rng, policy_key, critic_key, value_key = jax.random.split(rng, 4)
        context_key = jax.random.fold_in(rng, 1)

        config = dict(config)
        action_dim = ex_actions.shape[-1]
        horizon = int(config.get("horizon_length", 1))
        ex_full_actions = (
            jnp.concatenate([ex_actions] * horizon, axis=-1)
            if config.get("action_chunking", False)
            else ex_actions
        )
        full_action_dim = ex_full_actions.shape[-1]
        config["action_dim"] = action_dim
        config["context_token_dim"] = (
            2 * ex_observations.shape[-1] + action_dim + 2
        )

        activation = get_activation(config["activation"])
        mlp_kwargs = {
            "activation": activation,
            "layer_norm": config["use_layer_norm"],
        }
        policy_def = ActorFlowField(
            config["actor_hidden_dims"],
            full_action_dim,
            mlp_kwargs=mlp_kwargs,
        )
        policy_params = policy_def.init(
            policy_key,
            ex_observations,
            ex_full_actions,
            jnp.zeros(ex_actions.shape[0]),
        )["params"]
        policy = TrainState.create(
            policy_def,
            policy_params,
            tx=optax.adam(learning_rate=config["bc_lr"]),
        )

        value_kwargs = {
            **config["value_network_kwargs"],
            "activation": activation,
        }
        latent = jnp.zeros((ex_actions.shape[0], config["latent_dim"]))
        ready = jnp.zeros((ex_actions.shape[0],))
        critic_def = ContextValue(
            latent_dim=config["latent_dim"],
            network_class=config["value_network_class"],
            network_kwargs=value_kwargs,
            num_ensembles=config["num_qs"],
            context_kernel_init_scale=config["context_value_init_scale"],
        )
        critic_params = critic_def.init(
            critic_key,
            ex_observations,
            ex_full_actions,
            latent,
            ready,
        )["params"]
        critic = TrainState.create(
            critic_def,
            critic_params,
            tx=optax.adam(learning_rate=config["critic_lr"]),
        )
        target_critic = TrainState.create(critic_def, critic_params)

        value_def = ContextValue(
            latent_dim=config["latent_dim"],
            network_class=config["value_network_class"],
            network_kwargs=value_kwargs,
            num_ensembles=1,
            context_kernel_init_scale=config["context_value_init_scale"],
        )
        value_params = value_def.init(
            value_key, ex_observations, None, latent, ready
        )["params"]
        value = TrainState.create(
            value_def,
            value_params,
            tx=optax.adam(learning_rate=config["value_lr"]),
        )

        context_def = CausalPearlEncoder(
            latent_dim=config["latent_dim"],
            token_hidden_dims=config["context_hidden_dims"],
            gru_hidden_dim=config["context_gru_hidden_dim"],
            min_log_variance=config["context_min_log_variance"],
            max_log_variance=config["context_max_log_variance"],
        )
        ex_context = jnp.zeros(
            (
                ex_observations.shape[0],
                config["context_length"],
                config["context_token_dim"],
            )
        )
        ex_mask = jnp.zeros(
            (ex_observations.shape[0], config["context_length"])
        )
        context_params = context_def.init(
            context_key, ex_context, ex_mask
        )["params"]
        context_encoder = TrainState.create(
            context_def,
            context_params,
            tx=optax.adam(learning_rate=config["context_lr"]),
        )
        return cls(
            rng=rng,
            policy=policy,
            critic=critic,
            target_critic=target_critic,
            value=value,
            context_encoder=context_encoder,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    """Task-agnostic Context-Q defaults; Task3 overrides live in scripts/."""

    config = get_qgf_config()
    config["agent_name"] = "context_qgf"
    config["context_length"] = 20
    config["min_context_transitions"] = 20
    config["latent_dim"] = 8
    config["context_hidden_dims"] = (256, 256)
    config["context_gru_hidden_dim"] = 128
    config["context_lr"] = 3e-4
    config["context_kl_weight"] = 1e-2
    config["context_min_log_variance"] = -8.0
    config["context_max_log_variance"] = 4.0
    config["context_value_init_scale"] = 1e-2
    config["context_include_reward"] = True
    config["context_normalization"] = None
    config["deterministic_context_training"] = False
    config["deterministic_context_eval"] = True
    return ml_collections.ConfigDict(config)
