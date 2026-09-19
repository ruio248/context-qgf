import unittest

import numpy as np

from utils.context import context_is_ready_numpy
from utils.datasets import Dataset, deterministic_sequence_indices


class ContextDatasetTest(unittest.TestCase):
    def test_numpy_ready_gate_matches_minimum_completed_transitions(self):
        mask = np.array([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=np.float32)
        np.testing.assert_array_equal(
            context_is_ready_numpy(mask, min_count=3),
            np.array([0.0, 1.0], dtype=np.float32),
        )

    def test_deterministic_sequence_indices_depend_only_on_seed_and_step(self):
        first = deterministic_sequence_indices(100, 5, 16, seed=7, global_step=500_001)
        second = deterministic_sequence_indices(100, 5, 16, seed=7, global_step=500_001)
        later = deterministic_sequence_indices(100, 5, 16, seed=7, global_step=500_002)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first >= 0))
        self.assertTrue(np.all(first <= 95))
        self.assertFalse(np.array_equal(first, later))

    def test_history_is_causal_and_stops_at_episode_boundary(self):
        observations = np.arange(8, dtype=np.float32)[:, None]
        dataset = Dataset.create(
            observations=observations,
            actions=(100 + np.arange(8, dtype=np.float32))[:, None],
            rewards=np.arange(8, dtype=np.float32),
            masks=np.ones(8, dtype=np.float32),
            terminals=np.array([0, 0, 0, 1, 0, 0, 0, 1], np.float32),
            next_observations=observations + 0.5,
        )
        dataset.context_include_reward = True
        batch = dataset.sample_context_sequence(
            batch_size=4,
            sequence_length=1,
            context_length=3,
            discount=0.99,
            idxs=np.array([0, 2, 4, 6]),
        )
        np.testing.assert_array_equal(
            batch["context_mask"].sum(axis=1), np.array([0, 2, 0, 2])
        )
        np.testing.assert_array_equal(
            batch["next_context_mask"].sum(axis=1), np.array([1, 3, 1, 3])
        )
        # The final query belongs to episode two and cannot see episode one.
        np.testing.assert_array_equal(
            batch["context"][3, -2:, 0], np.array([4.0, 5.0])
        )

    def test_reward_free_contract_zeroes_only_reward_coordinate(self):
        observations = np.arange(4, dtype=np.float32)[:, None]
        dataset = Dataset.create(
            observations=observations,
            actions=np.ones((4, 1), np.float32),
            rewards=np.arange(4, dtype=np.float32),
            masks=np.ones(4, np.float32),
            terminals=np.array([0, 0, 0, 1], np.float32),
            next_observations=observations + 1,
        )
        dataset.context_include_reward = False
        batch = dataset.sample_context_sequence(
            1, 1, 2, 0.99, idxs=np.array([2])
        )
        context = batch["context"][0]
        valid = batch["context_mask"][0] > 0.5
        np.testing.assert_array_equal(context[valid, 2], np.zeros(valid.sum()))
        np.testing.assert_array_equal(context[valid, 3], np.ones(valid.sum()))


if __name__ == "__main__":
    unittest.main()
