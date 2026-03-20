import unittest

import torch

from pfedfsl_lite_utils import (
    aggregate_shared_states_by_weights,
    compute_route_softmax_weights,
    split_route_update_batch,
)


class PFedFslLiteRoutingUtilsTest(unittest.TestCase):
    def test_split_route_update_batch_preserves_partition(self):
        torch.manual_seed(0)
        train_input = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        train_target = torch.arange(10, dtype=torch.float32).reshape(10, 1)

        split = split_route_update_batch(train_input, train_target, route_ratio=0.3)

        self.assertEqual(split["route_input"].shape[0], 3)
        self.assertEqual(split["update_input"].shape[0], 7)
        self.assertEqual(split["route_target"].shape[0] + split["update_target"].shape[0], 10)

    def test_split_route_update_batch_falls_back_to_time_axis_for_single_sequence(self):
        torch.manual_seed(0)
        train_input = torch.arange(60, dtype=torch.float32).reshape(1, 12, 5)
        train_target = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)

        split = split_route_update_batch(train_input, train_target, route_ratio=0.25)

        self.assertEqual(tuple(split["route_input"].shape), (1, 3, 5))
        self.assertEqual(tuple(split["update_input"].shape), (1, 9, 5))
        self.assertEqual(tuple(split["route_target"].shape), (1, 3, 1))
        self.assertEqual(tuple(split["update_target"].shape), (1, 9, 1))

    def test_compute_route_softmax_weights_sums_to_one(self):
        weights = compute_route_softmax_weights([0.5, 1.0, 2.0], temperature=0.7)
        self.assertAlmostEqual(float(weights.sum().item()), 1.0, places=6)
        self.assertGreater(float(weights[0].item()), float(weights[1].item()))
        self.assertGreater(float(weights[1].item()), float(weights[2].item()))

    def test_lower_temperature_makes_distribution_sharper(self):
        cold = compute_route_softmax_weights([0.1, 0.2, 0.3], temperature=0.1)
        warm = compute_route_softmax_weights([0.1, 0.2, 0.3], temperature=1.0)
        self.assertGreater(float(cold[0].item()), float(warm[0].item()))

    def test_aggregate_shared_states_by_weights_preserves_keys_and_shapes(self):
        state_a = {
            "tcn.weight": torch.ones(2, 2),
            "tcn.bias": torch.zeros(2),
        }
        state_b = {
            "tcn.weight": torch.full((2, 2), 3.0),
            "tcn.bias": torch.full((2,), 2.0),
        }

        aggregated = aggregate_shared_states_by_weights([state_a, state_b], [0.25, 0.75])

        self.assertEqual(set(aggregated.keys()), set(state_a.keys()))
        self.assertEqual(tuple(aggregated["tcn.weight"].shape), (2, 2))
        self.assertTrue(torch.allclose(aggregated["tcn.weight"], torch.full((2, 2), 2.5)))
        self.assertTrue(torch.allclose(aggregated["tcn.bias"], torch.full((2,), 1.5)))


if __name__ == "__main__":
    unittest.main()
