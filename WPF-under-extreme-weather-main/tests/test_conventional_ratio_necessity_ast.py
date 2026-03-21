import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class ConventionalRatioNecessityAstTest(unittest.TestCase):
    def test_training_script_exposes_conventional_ratio_knob(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("CONVENTIONAL_RATIO", text)
        self.assertIn("SUPPORTED_CONVENTIONAL_RATIOS", text)
        self.assertIn("CONVENTIONAL_SUBSAMPLE_SEED_OFFSET", text)

    def test_training_script_contains_pretrain_and_meta_subsampling_hooks(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def subsample_pretrain_conventional_data(", text)
        self.assertIn("def subsample_meta_conventional_data(", text)
        self.assertIn("subsample_seed_offset", text)
        self.assertIn("clients_train_data = subsample_pretrain_conventional_data(", text)
        self.assertIn("all_stations_full_data = subsample_meta_conventional_data(", text)

    def test_meta_subsampling_preserves_minimum_episode_budget(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("max(20,", text)


if __name__ == "__main__":
    unittest.main()
