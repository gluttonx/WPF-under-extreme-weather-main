import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class ExtremePrototypeFeaturesAstTest(unittest.TestCase):
    def test_training_script_declares_phi_descriptor_builder(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("EXTREME_SCENARIO_DESCRIPTOR_DIM", text)
        self.assertIn("def build_extreme_scenario_descriptor(", text)
        self.assertIn("mu_y", text)
        self.assertIn("std_y", text)
        self.assertIn("ramp_y", text)
        self.assertIn("drop_y", text)

    def test_training_script_contains_in_client_normalization_and_class_prototype_helpers(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def normalize_prototype_features_in_client(", text)
        self.assertIn("def compute_extreme_class_prototype(", text)
        self.assertIn("np.mean(", text)
        self.assertIn("np.std(", text)
        self.assertIn("np.min(", text)
        self.assertIn("np.max(", text)
        self.assertIn("np.diff(", text)


if __name__ == "__main__":
    unittest.main()
