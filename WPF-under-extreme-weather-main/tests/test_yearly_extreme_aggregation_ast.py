import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class YearlyExtremeAggregationAstTest(unittest.TestCase):
    def test_training_script_declares_extreme_stage_aggregation_helpers(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def run_extreme_client_few_shot_update(", text)
        self.assertIn("def aggregate_extreme_client_states(", text)
        self.assertIn('if aggregation_mode == "uniform":', text)
        self.assertIn('if aggregation_mode == "reliability_aware":', text)
        self.assertIn("query_loss_by_station", text)

    def test_yearly_few_shot_loop_uses_common_local_meta_and_aggregation(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("base_model_path=get_local_meta_model_path(", text)
        self.assertIn('aggregation_mode="uniform"', text)
        self.assertIn('aggregation_mode="reliability_aware"', text)
        self.assertIn("compute_reliability_aware_weights(", text)

    def test_extreme_client_update_returns_best_loss_from_convergence_record(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn('"best_loss": convergence_record["best_loss"]', text)
        self.assertNotIn('"best_loss": convergence_record["best_metric"]', text)


if __name__ == "__main__":
    unittest.main()
