import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class ExtremeWeightedAggregationAstTest(unittest.TestCase):
    def test_training_script_declares_weighting_hyperparameters(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("EXTREME_WEIGHT_TAU", text)
        self.assertIn("EXTREME_WEIGHT_LAMBDA", text)
        self.assertIn("EXTREME_WEIGHT_MU", text)
        self.assertIn("EXTREME_WEIGHT_NU", text)

    def test_training_script_contains_reliability_weight_builders(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def compute_sample_count_reliability(", text)
        self.assertIn("def compute_query_reliability(", text)
        self.assertIn("def compute_scenario_similarity(", text)
        self.assertIn("def compute_reliability_aware_weights(", text)
        self.assertIn("m_k_c", text)
        self.assertIn("q_k_c", text)
        self.assertIn("sim_k_to_s_c", text)
        self.assertIn("alpha_k_to_s_c", text)


if __name__ == "__main__":
    unittest.main()
