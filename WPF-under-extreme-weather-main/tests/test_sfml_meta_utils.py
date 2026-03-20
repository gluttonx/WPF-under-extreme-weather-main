import unittest
import random

import torch

from sfml_meta_utils import (
    accumulate_weighted_state,
    choose_local_meta_task_classes,
    compose_personalized_meta_init_state,
    finalize_weighted_state,
    select_meta_shared_state,
    should_stop_early,
    update_best_meta_checkpoint,
)


class SFMLMetaUtilsTest(unittest.TestCase):
    def test_cross_cluster_sampling_prefers_distinct_support_and_query_classes(self):
        rng = random.Random(7)

        support_class_idx, query_class_idx = choose_local_meta_task_classes(
            class_sample_counts=[18, 15, 22],
            n_support=10,
            n_query=10,
            task_mode="cross_cluster",
            rng=rng,
        )

        self.assertNotEqual(support_class_idx, query_class_idx)
        self.assertGreaterEqual([18, 15, 22][support_class_idx], 10)
        self.assertGreaterEqual([18, 15, 22][query_class_idx], 10)

    def test_cross_cluster_sampling_falls_back_to_same_cluster_when_needed(self):
        rng = random.Random(3)

        support_class_idx, query_class_idx = choose_local_meta_task_classes(
            class_sample_counts=[23, 7, 6],
            n_support=10,
            n_query=10,
            task_mode="cross_cluster",
            rng=rng,
        )

        self.assertEqual((support_class_idx, query_class_idx), (0, 0))

    def test_same_cluster_sampling_requires_combined_budget(self):
        rng = random.Random(11)

        support_class_idx, query_class_idx = choose_local_meta_task_classes(
            class_sample_counts=[25, 18, 21],
            n_support=10,
            n_query=10,
            task_mode="same_cluster",
            rng=rng,
        )

        self.assertEqual(support_class_idx, query_class_idx)
        self.assertGreaterEqual([25, 18, 21][support_class_idx], 20)

    def test_sampling_raises_when_no_class_has_enough_examples(self):
        with self.assertRaises(ValueError):
            choose_local_meta_task_classes(
                class_sample_counts=[8, 9, 7],
                n_support=10,
                n_query=10,
                task_mode="cross_cluster",
                rng=random.Random(5),
            )

    def test_update_best_meta_checkpoint_keeps_lowest_query_loss_and_clones_state(self):
        first_state = {"shared.weight": torch.tensor([1.0, 2.0])}
        best = update_best_meta_checkpoint(
            best_record=None,
            epoch_index=0,
            query_loss=0.4,
            shared_state=first_state,
        )
        self.assertEqual(best["epoch"], 0)
        self.assertAlmostEqual(best["query_loss"], 0.4)
        self.assertTrue(torch.equal(best["shared_state"]["shared.weight"], torch.tensor([1.0, 2.0])))

        first_state["shared.weight"][0] = 99.0
        self.assertTrue(torch.equal(best["shared_state"]["shared.weight"], torch.tensor([1.0, 2.0])))

        worse_state = {"shared.weight": torch.tensor([3.0, 4.0])}
        kept = update_best_meta_checkpoint(
            best_record=best,
            epoch_index=1,
            query_loss=0.5,
            shared_state=worse_state,
        )
        self.assertIs(kept, best)
        self.assertEqual(kept["epoch"], 0)

        better_state = {"shared.weight": torch.tensor([5.0, 6.0])}
        updated = update_best_meta_checkpoint(
            best_record=best,
            epoch_index=2,
            query_loss=0.3,
            shared_state=better_state,
        )
        self.assertEqual(updated["epoch"], 2)
        self.assertAlmostEqual(updated["query_loss"], 0.3)
        self.assertTrue(torch.equal(updated["shared_state"]["shared.weight"], torch.tensor([5.0, 6.0])))

    def test_compose_personalized_meta_init_state_uses_shared_backbone_and_pretrain_local(self):
        shared_state = {
            "tcn.layer.weight": torch.tensor([10.0]),
            "tcn.layer.bias": torch.tensor([20.0]),
        }
        pretrain_local_state = {
            "fore_baselearner.weight": torch.tensor([30.0]),
            "lwp.weight": torch.tensor([40.0]),
        }

        mixed = compose_personalized_meta_init_state(shared_state, pretrain_local_state)

        self.assertEqual(set(mixed.keys()), set(shared_state.keys()) | set(pretrain_local_state.keys()))
        self.assertTrue(torch.equal(mixed["tcn.layer.weight"], torch.tensor([10.0])))
        self.assertTrue(torch.equal(mixed["fore_baselearner.weight"], torch.tensor([30.0])))
        self.assertTrue(torch.equal(mixed["lwp.weight"], torch.tensor([40.0])))

        shared_state["tcn.layer.weight"][0] = -1.0
        pretrain_local_state["fore_baselearner.weight"][0] = -2.0
        self.assertTrue(torch.equal(mixed["tcn.layer.weight"], torch.tensor([10.0])))
        self.assertTrue(torch.equal(mixed["fore_baselearner.weight"], torch.tensor([30.0])))

    def test_select_meta_shared_state_switches_between_best_and_latest(self):
        best = {"epoch": 2, "query_loss": 0.3, "shared_state": {"shared.weight": torch.tensor([1.0])}}
        latest = {"shared.weight": torch.tensor([9.0])}

        selected_best = select_meta_shared_state(best, latest, save_best_only=True)
        selected_latest = select_meta_shared_state(best, latest, save_best_only=False)

        self.assertTrue(torch.equal(selected_best["shared.weight"], torch.tensor([1.0])))
        self.assertTrue(torch.equal(selected_latest["shared.weight"], torch.tensor([9.0])))

        best["shared_state"]["shared.weight"][0] = -1.0
        latest["shared.weight"][0] = -2.0
        self.assertTrue(torch.equal(selected_best["shared.weight"], torch.tensor([1.0])))
        self.assertTrue(torch.equal(selected_latest["shared.weight"], torch.tensor([9.0])))

    def test_accumulate_and_finalize_weighted_state_computes_weighted_average(self):
        first = {"shared.weight": torch.tensor([1.0, 3.0])}
        second = {"shared.weight": torch.tensor([5.0, 7.0])}

        state_sum = accumulate_weighted_state(None, first, weight=2.0)
        state_sum = accumulate_weighted_state(state_sum, second, weight=1.0)
        averaged = finalize_weighted_state(state_sum, total_weight=3.0)

        self.assertTrue(torch.equal(averaged["shared.weight"], torch.tensor([7.0 / 3.0, 13.0 / 3.0])))

        first["shared.weight"][0] = -10.0
        second["shared.weight"][0] = -20.0
        self.assertTrue(torch.equal(averaged["shared.weight"], torch.tensor([7.0 / 3.0, 13.0 / 3.0])))

    def test_should_stop_early_respects_patience(self):
        self.assertFalse(should_stop_early(no_improve_rounds=3, patience=0))
        self.assertFalse(should_stop_early(no_improve_rounds=1, patience=3))
        self.assertFalse(should_stop_early(no_improve_rounds=2, patience=3))
        self.assertTrue(should_stop_early(no_improve_rounds=3, patience=3))
        self.assertTrue(should_stop_early(no_improve_rounds=5, patience=3))


if __name__ == "__main__":
    unittest.main()
