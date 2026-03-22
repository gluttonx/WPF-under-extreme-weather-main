import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class BalancedMetaSamplerAstTest(unittest.TestCase):
    def test_training_script_exposes_proposed_sampler_mode_knob(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("PROPOSED_META_SAMPLER_MODE", text)
        self.assertIn('env_str("PROPOSED_META_SAMPLER_MODE"', text)

    def test_training_script_contains_balanced_meta_sampler_helpers(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def sample_station_meta_batch_uniform(", text)
        self.assertIn("def sample_station_meta_batch_balanced(", text)
        self.assertIn("coverage_bonus", text)
        self.assertIn("size_bonus", text)
        self.assertIn("weighted random", text.lower())

    def test_only_proposed_path_uses_balanced_sampler_mode(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn('sampler_mode=PROPOSED_META_SAMPLER_MODE', text)
        self.assertIn('sampler_mode="uniform"', text)


if __name__ == "__main__":
    unittest.main()
