"""Checkpoint-preserving Context-Q adapter finetuning.

This agent starts from a native QGF checkpoint.  The flow actor and all native
Q/V parameters are frozen; only the ContextInput projections and the causal
posterior encoder are optimized.  Exact-zero ContextInput initialization means
the contextual model initially has the same Q values and action gradients as
the native checkpoint even when a context history is ready.
"""

from __future__ import annotations

import jax
import ml_collections

from agents.context_qgf import ContextQGFAgent
from agents.context_qgf import get_config as get_context_config
from utils.context_adapter import (
    assert_source_is_covered,
    copy_exact_params,
    copy_matching_params,
    make_context_adapter_optimizer,
    target_update_context_adapters,
    zero_context_inputs,
)
from utils.flax_utils import TrainState


class ContextQGFAdapterAgent(ContextQGFAgent):
    """Context-Q whose only trainable Q/V parameters are ContextInput adapters."""

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = dict(config)
        # This is a semantic invariant of adapter finetuning, not a tunable
        # small-random initialization.  It preserves native Q/gradients at t=0.
        config["context_value_init_scale"] = 0.0
        agent = super().create(seed, ex_observations, ex_actions, config)

        critic_tx = make_context_adapter_optimizer(
            agent.critic.params, config["critic_lr"]
        )
        value_tx = make_context_adapter_optimizer(
            agent.value.params, config["value_lr"]
        )
        return agent.replace(
            critic=TrainState.create(
                agent.critic.model_def, agent.critic.params, tx=critic_tx
            ),
            value=TrainState.create(
                agent.value.model_def, agent.value.params, tx=value_tx
            ),
        )

    def initialize_from_native(self, native_agent):
        """Transfer a compatible native QGF checkpoint into this agent.

        Optimizer states are intentionally fresh for the newly introduced
        adapter parameters.  The native actor never receives an update in this
        agent, so its optimizer state is irrelevant and is not transferred.
        """

        policy_params = copy_exact_params(
            self.policy.params, native_agent.policy.params, component="policy"
        )
        assert_source_is_covered(
            self.critic.params, native_agent.critic.params, component="critic"
        )
        assert_source_is_covered(
            self.target_critic.params,
            native_agent.target_critic.params,
            component="target critic",
        )
        assert_source_is_covered(
            self.value.params, native_agent.value.params, component="value"
        )

        policy = self.policy.replace(params=policy_params)
        critic_params = zero_context_inputs(
            copy_matching_params(self.critic.params, native_agent.critic.params)
        )
        target_critic_params = zero_context_inputs(
            copy_matching_params(
                self.target_critic.params, native_agent.target_critic.params
            )
        )
        value_params = zero_context_inputs(
            copy_matching_params(self.value.params, native_agent.value.params)
        )

        critic_tx = make_context_adapter_optimizer(
            critic_params, self.config["critic_lr"]
        )
        value_tx = make_context_adapter_optimizer(
            value_params, self.config["value_lr"]
        )
        return self.replace(
            policy=policy,
            critic=TrainState.create(self.critic.model_def, critic_params, tx=critic_tx),
            target_critic=TrainState.create(
                self.target_critic.model_def, target_critic_params
            ),
            value=TrainState.create(self.value.model_def, value_params, tx=value_tx),
        )

    @jax.jit
    def update(self, batch):
        """Update only ContextInput parameters and the causal encoder."""

        new_rng, _ = jax.random.split(self.rng, 2)
        critic_rng = jax.random.fold_in(self.rng, 1)
        value_rng = jax.random.fold_in(self.rng, 2)

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
        # Preserve every native target-Q leaf from the checkpoint.  Only the
        # newly added adapter leaves track the just-updated online critic.
        new_target_critic = target_update_context_adapters(
            new_critic, self.target_critic, self.config["tau"]
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
            critic=new_critic,
            target_critic=new_target_critic,
            value=new_value,
            context_encoder=new_context_encoder,
        ), {**critic_info, **value_info}


def get_config():
    config = get_context_config()
    config["agent_name"] = "context_qgf_adapter"
    config["context_value_init_scale"] = 0.0
    return ml_collections.ConfigDict(config)
