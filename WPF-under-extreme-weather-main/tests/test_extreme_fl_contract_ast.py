import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class ExtremeFLContractAstTest(unittest.TestCase):
    def test_training_declares_three_model_extreme_paths(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertIn("def get_extreme_fedavg_model_path(station_id, class_idx):", text)
        self.assertIn("def get_proposed_a_model_path(station_id, class_idx):", text)
        self.assertIn("model_label=\"Extreme-FedAvg:target_refine\"", text)
        self.assertIn("model_label=\"Proposed-A:target_refine\"", text)

    def test_training_declares_effective_window_and_weighting_helpers(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertIn("def split_extreme_adapt_val(", text)
        self.assertIn("def apply_source_quality_gate(", text)
        self.assertIn("def compute_target_conditioned_usefulness(", text)
        self.assertIn("def aggregate_extreme_updates_uniform(", text)
        self.assertIn("def aggregate_extreme_updates_weighted(", text)
        self.assertIn("EXTREME_WEIGHT_BETA_SELF", text)


if __name__ == "__main__":
    unittest.main()
