import json
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

from agents.context_qgf import ContextQGFAgent
from agents.context_qgf import get_config as get_context_config
from agents.context_qgf_adapter import ContextQGFAdapterAgent
from agents.qgf import QGFAgent
from agents.qgf import get_config as get_qgf_config
from agents.qgf_qv_finetune import QGFQVFinetuneAgent
from utils.context_adapter import target_update_context_adapters, zero_context_inputs
from utils.flax_utils import save_agent, target_update
from utils.native_transfer import (
    adapter_initialization_audit,
    assert_transfer_compatible,
    source_backbone_finetune_config,
)
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

    def test_target_critic_uses_pre_update_critic(self):
        batch = self.batch()
        updated, _ = self.context_agent.update(batch)
        expected_target = target_update(
            self.context_agent.critic,
            self.context_agent.target_critic,
            self.context_agent.config["tau"],
        )
        self.assertEqual(
            tree_l1(updated.target_critic.params, expected_target.params),
            0.0,
        )

    def test_value_loss_uses_pre_update_encoder(self):
        batch = self.batch()
        value_rng = jax.random.fold_in(self.context_agent.rng, 2)
        expected_value, _ = self.context_agent.value.apply_loss_fn(
            loss_fn=lambda params: self.context_agent.value_loss(
                batch,
                value_params=params,
                context_params=self.context_agent.context_encoder.params,
                rng=value_rng,
            )
        )
        updated, _ = self.context_agent.update(batch)
        self.assertLess(
            tree_l1(updated.value.params, expected_value.params),
            1e-5,
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


class ContextQGFAdapterAgentTest(unittest.TestCase):
    def setUp(self):
        self.config = small_config()
        self.observations = jnp.zeros((4, 5), dtype=jnp.float32)
        self.actions = jnp.zeros((4, 2), dtype=jnp.float32)
        self.native = QGFAgent.create(
            17, self.observations, self.actions, self.config
        )
        self.adapter = ContextQGFAdapterAgent.create(
            29, self.observations, self.actions, self.config
        ).initialize_from_native(self.native)

    def batch(self):
        rng = np.random.default_rng(11)
        token_dim = int(self.adapter.config["context_token_dim"])
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
            "next_context": rng.normal(size=(4, 3, token_dim)).astype(np.float32),
            "next_context_mask": np.ones((4, 3), np.float32),
        }

    def test_transfer_is_equivalent_even_for_ready_nonzero_context(self):
        rng = np.random.default_rng(31)
        observations = jnp.asarray(rng.normal(size=(4, 5)), dtype=jnp.float32)
        actions = jnp.asarray(rng.normal(size=(4, 2)), dtype=jnp.float32)
        latent = jnp.asarray(rng.normal(size=(4, 4)), dtype=jnp.float32)
        ready = jnp.ones((4,), dtype=jnp.float32)

        native_q = self.native.target_critic(observations, actions)
        adapter_q = self.adapter.target_critic(observations, actions, latent, ready)
        np.testing.assert_allclose(adapter_q, native_q, rtol=0, atol=1e-7)
        native_v = self.native.value(observations)
        adapter_v = self.adapter.value(observations, None, latent, ready)
        np.testing.assert_allclose(adapter_v, native_v, rtol=0, atol=1e-7)

        def native_q_fn(candidate_action):
            return self.native._aggregate_q(
                self.native.target_critic(observations[:1], candidate_action[None])
            )[0]

        def adapter_q_fn(candidate_action):
            return self.adapter._aggregate_q(
                self.adapter.target_critic(
                    observations[:1], candidate_action[None], latent[:1], ready[:1]
                )
            )[0]

        native_grad = jax.grad(native_q_fn)(actions[0])
        adapter_grad = jax.grad(adapter_q_fn)(actions[0])
        np.testing.assert_allclose(adapter_grad, native_grad, rtol=0, atol=1e-7)

        token_dim = int(self.adapter.config["context_token_dim"])
        context = jnp.asarray(rng.normal(size=(3, token_dim)), dtype=jnp.float32)
        mask = jnp.ones((3,), dtype=jnp.float32)
        key = jax.random.PRNGKey(19)
        native_action = self.native.sample_actions(
            observations[0], seed=key, guidance_weight=0.04
        )
        adapter_action = self.adapter.sample_actions(
            observations[0],
            seed=key,
            guidance_weight=0.04,
            context=context,
            context_mask=mask,
            deterministic_latent=True,
        )
        np.testing.assert_allclose(adapter_action, native_action, rtol=0, atol=1e-7)
        self.assertEqual(
            tree_l1(
                self.adapter.critic.params,
                zero_context_inputs(self.adapter.critic.params),
            ),
            0.0,
        )

    def test_transfer_audit_records_zero_adapter_checkpoint_provenance(self):
        example_batch = {
            "observations": np.zeros((1, 5), dtype=np.float32),
            "actions": np.zeros((1, 2), dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            save_agent(self.native, checkpoint, 500_000)
            checkpoint.joinpath("flags.json").write_text("{}\n")
            audit = adapter_initialization_audit(
                self.native,
                self.adapter,
                example_batch,
                source_checkpoint=checkpoint,
                source_epoch=500_000,
                compatibility={"semantic_mismatches": {}},
            )
        self.assertEqual(audit["context_input_abs_max"], 0.0)
        self.assertTrue(audit["copied_hashes_match"])
        self.assertLessEqual(audit["equivalence_max_abs"]["target_q_max_abs"], 2e-6)

    def test_native_transfer_rejects_semantic_mismatch_but_allows_new_optimizer(self):
        source = get_qgf_config()
        source.actor_hidden_dims = self.config.actor_hidden_dims
        source.value_network_kwargs.hidden_dims = self.config.value_network_kwargs.hidden_dims
        source.denoise_steps = self.config.denoise_steps
        source.horizon_length = self.config.horizon_length
        source.action_chunking = self.config.action_chunking
        destination = self.config.copy_and_resolve_references()
        compatibility = assert_transfer_compatible(
            source,
            destination,
            {"env_name": "test", "reward_scale": 1.0, "reward_bias": 0.0, "sparse": False},
            {"env_name": "test", "reward_scale": 1.0, "reward_bias": 0.0, "sparse": False},
        )
        self.assertEqual(compatibility["semantic_mismatches"], {})
        destination.activation = "relu"
        with self.assertRaisesRegex(ValueError, "incompatible semantics"):
            assert_transfer_compatible(
                source,
                destination,
                {"env_name": "test", "reward_scale": 1.0, "reward_bias": 0.0, "sparse": False},
                {"env_name": "test", "reward_scale": 1.0, "reward_bias": 0.0, "sparse": False},
            )
        destination.activation = source.activation
        destination.critic_lr = 1e-5
        qv = source_backbone_finetune_config(
            source, destination, agent_name="qgf_qv_finetune"
        )
        self.assertEqual(qv["agent_name"], "qgf_qv_finetune")
        self.assertEqual(qv["critic_lr"], 1e-5)

    def test_only_adapter_and_encoder_change_after_two_updates(self):
        first, first_info = self.adapter.update(self.batch())
        second, second_info = first.update(self.batch())
        for value in {**first_info, **second_info}.values():
            self.assertTrue(np.all(np.isfinite(np.asarray(value))))
        self.assertEqual(
            tree_l1(self.adapter.policy.params, second.policy.params), 0.0
        )
        self.assertEqual(
            tree_l1(
                self.adapter.critic.params,
                second.critic.params,
                exclude=("ContextInput",),
            ),
            0.0,
        )
        self.assertEqual(
            tree_l1(
                self.adapter.value.params,
                second.value.params,
                exclude=("ContextInput",),
            ),
            0.0,
        )
        self.assertGreater(
            tree_l1(self.adapter.critic.params, second.critic.params), 0.0
        )
        self.assertGreater(
            tree_l1(
                self.adapter.context_encoder.params,
                second.context_encoder.params,
            ),
            0.0,
        )
        self.assertEqual(
            tree_l1(
                self.adapter.target_critic.params,
                first.target_critic.params,
                exclude=("ContextInput",),
            ),
            0.0,
        )
        expected_target = target_update_context_adapters(
            first.critic,
            self.adapter.target_critic,
            self.adapter.config["tau"],
        )
        self.assertEqual(
            tree_l1(first.target_critic.params, expected_target.params), 0.0
        )

    def test_native_qv_control_freezes_policy_and_tracks_new_critic(self):
        control = QGFQVFinetuneAgent.create(
            17, self.observations, self.actions, self.config
        )
        control = control.replace(
            policy=control.policy.replace(params=self.native.policy.params),
            critic=control.critic.replace(params=self.native.critic.params),
            target_critic=control.target_critic.replace(
                params=self.native.target_critic.params
            ),
            value=control.value.replace(params=self.native.value.params),
        )
        updated, _ = control.update(self.batch())
        self.assertEqual(tree_l1(control.policy.params, updated.policy.params), 0.0)
        self.assertGreater(tree_l1(control.critic.params, updated.critic.params), 0.0)
        expected_target = target_update(
            updated.critic, control.target_critic, control.config["tau"]
        )
        self.assertLess(
            tree_l1(updated.target_critic.params, expected_target.params), 1e-6
        )

    def test_adapter_checkpoint_uses_adapter_class_in_shared_evaluator_loader(self):
        from experiments.evaluate_task3_mc import load_checkpoint

        config = small_config()
        config.agent_name = "context_qgf_adapter"
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            save_agent(self.adapter, checkpoint, 9)
            checkpoint.joinpath("flags.json").write_text(
                json.dumps({"seed": 29, "agent": config.to_dict()})
            )
            loaded, _ = load_checkpoint(
                checkpoint,
                9,
                self.observations[0],
                self.actions[0],
                contextual=True,
            )
        self.assertIsInstance(loaded, ContextQGFAdapterAgent)
        self.assertEqual(loaded.config["agent_name"], "context_qgf_adapter")


if __name__ == "__main__":
    unittest.main()
