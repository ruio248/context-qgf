import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

from agents.context_qgf import ContextQGFAgent
from agents.context_qgf import get_config as get_context_config
from agents.qgf import QGFAgent
from utils.context import (
    CausalPearlEncoder,
    kl_to_standard_normal,
    transition_token_numpy,
)


def small_config():
    config = get_context_config()
    config.actor_hidden_dims = (32, 32)
    config.value_network_kwargs.hidden_dims = (32, 32)
    config.batch_size = 4
    config.context_hidden_dims = (16,)
    config.context_gru_hidden_dim = 12
    config.context_length = 3
    config.min_context_transitions = 1
    config.latent_dim = 4
    config.denoise_steps = 3
    config.horizon_length = 1
    config.action_chunking = False
    return config


def tree_l1(left, right, *, exclude=()):
    left_flat = traverse_util.flatten_dict(left)
    right_flat = traverse_util.flatten_dict(right)
    total = 0.0
    for path, value in left_flat.items():
        if any(marker in path for marker in exclude):
            continue
        total += float(jnp.sum(jnp.abs(value - right_flat[path])))
    return total


class CausalPearlEncoderTest(unittest.TestCase):
    def test_empty_context_is_exact_standard_normal(self):
        encoder = CausalPearlEncoder(
            latent_dim=3, token_hidden_dims=(8,), gru_hidden_dim=6
        )
        context = jnp.zeros((2, 5, 7))
        mask = jnp.zeros((2, 5))
        params = encoder.init(jax.random.PRNGKey(0), context, mask)
        mean, log_variance = encoder.apply(params, context, mask)
        np.testing.assert_array_equal(np.asarray(mean), np.zeros((2, 3)))
        np.testing.assert_array_equal(
            np.asarray(log_variance), np.zeros((2, 3))
        )
        np.testing.assert_array_equal(
            np.asarray(kl_to_standard_normal(mean, log_variance)), np.zeros(2)
        )

    def test_masked_future_tokens_cannot_change_posterior(self):
        encoder = CausalPearlEncoder(
            latent_dim=3, token_hidden_dims=(8,), gru_hidden_dim=6
        )
        prefix = np.arange(21, dtype=np.float32).reshape(1, 3, 7)
        context_a = jnp.asarray(
            np.concatenate([prefix, np.zeros((1, 2, 7), np.float32)], axis=1)
        )
        context_b = jnp.asarray(
            np.concatenate(
                [prefix, np.full((1, 2, 7), 999.0, np.float32)], axis=1
            )
        )
        mask = jnp.array([[1.0, 1.0, 1.0, 0.0, 0.0]])
        params = encoder.init(jax.random.PRNGKey(1), context_a, mask)
        posterior_a = encoder.apply(params, context_a, mask)
        posterior_b = encoder.apply(params, context_b, mask)
        np.testing.assert_allclose(posterior_a[0], posterior_b[0], atol=1e-7)
        np.testing.assert_allclose(posterior_a[1], posterior_b[1], atol=1e-7)

    def test_reward_is_a_completed_transition_feature(self):
        token = transition_token_numpy(
            observation=np.array([1.0, 2.0]),
            action=np.array([0.5]),
            reward=3.0,
            next_observation=np.array([2.0, 4.0]),
            done=False,
        )
        # [state(2), action(1), reward(1), state_delta(2), done(1)]
        np.testing.assert_allclose(
            token, np.array([1.0, 2.0, 0.5, 3.0, 1.0, 2.0, 0.0])
        )


class ContextQGFAgentTest(unittest.TestCase):
    def setUp(self):
        self.config = small_config()
        self.observations = jnp.zeros((4, 5), dtype=jnp.float32)
        self.actions = jnp.zeros((4, 2), dtype=jnp.float32)
        self.base = QGFAgent.create(
            7, self.observations, self.actions, self.config
        )
        self.context_agent = ContextQGFAgent.create(
            7, self.observations, self.actions, self.config
        )

    def batch(self):
        rng = np.random.default_rng(3)
        token_dim = int(self.context_agent.config["context_token_dim"])
        return {
            "observations": rng.normal(size=(4, 5)).astype(np.float32),
            "actions": rng.normal(size=(4, 1, 2)).astype(np.float32),
            "next_observations": rng.normal(size=(4, 1, 5)).astype(np.float32),
            "rewards": rng.normal(size=(4, 1)).astype(np.float32),
            "masks": np.ones((4, 1), np.float32),
            "terminals": np.zeros((4, 1), np.float32),
            "valid": np.ones((4, 1), np.float32),
            "context": rng.normal(size=(4, 3, token_dim)).astype(np.float32),
            "context_mask": np.ones((4, 3), np.float32),
            "next_context": rng.normal(size=(4, 3, token_dim)).astype(
                np.float32
            ),
            "next_context_mask": np.ones((4, 3), np.float32),
        }

    def test_native_actor_and_zero_context_backbone_match_at_initialization(self):
        noised_action = jax.random.normal(jax.random.PRNGKey(2), (4, 2))
        time = jnp.linspace(0.0, 1.0, 4)
        np.testing.assert_array_equal(
            np.asarray(self.base.policy(self.observations, noised_action, time)),
            np.asarray(
                self.context_agent.policy(
                    self.observations, noised_action, time
                )
            ),
        )

        latent = jax.random.normal(jax.random.PRNGKey(3), (4, 4))
        base_q = self.base.target_critic(self.observations, noised_action)
        context_q = self.context_agent.target_critic(
            self.observations,
            noised_action,
            latent,
            jnp.zeros((4,)),
        )
        np.testing.assert_allclose(context_q, base_q, rtol=0, atol=1e-7)

    def test_one_update_trains_backbone_and_encoder_not_just_adapter(self):
        updated, info = self.context_agent.update(self.batch())
        for value in info.values():
            self.assertTrue(np.all(np.isfinite(np.asarray(value))))
        self.assertGreater(
            tree_l1(
                self.context_agent.critic.params,
                updated.critic.params,
                exclude=("ContextInput",),
            ),
            0.0,
        )
        self.assertGreater(
            tree_l1(
                self.context_agent.context_encoder.params,
                updated.context_encoder.params,
            ),
            0.0,
        )

    def test_actor_update_is_identical_to_native_qgf(self):
        batch = self.batch()
        updated_base, _ = self.base.update(batch)
        updated_context, _ = self.context_agent.update(batch)
        self.assertEqual(
            tree_l1(updated_base.policy.params, updated_context.policy.params),
            0.0,
        )

    def test_alpha_zero_actions_ignore_context(self):
        observation = jnp.arange(5, dtype=jnp.float32) / 10
        seed = jax.random.PRNGKey(11)
        token_dim = int(self.context_agent.config["context_token_dim"])
        context = jnp.ones((3, token_dim))
        mask = jnp.ones((3,))
        without_context = self.context_agent.sample_actions(
            observation, seed=seed, guidance_weight=0.0
        )
        with_context = self.context_agent.sample_actions(
            observation,
            seed=seed,
            guidance_weight=0.0,
            context=context,
            context_mask=mask,
        )
        np.testing.assert_allclose(
            with_context, without_context, rtol=0, atol=1e-7
        )


if __name__ == "__main__":
    unittest.main()
