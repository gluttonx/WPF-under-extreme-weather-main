import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class RegimeMissingStressAstTest(unittest.TestCase):
    def test_training_script_exposes_regime_missing_knob_and_pattern(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("REGIME_MISSING_MODE", text)
        self.assertIn("DEFAULT_REGIME_MISSING_CLASS_MAP", text)
        self.assertIn("class_dropout", text)

    def test_training_script_contains_regime_missing_hooks_for_pretrain_and_meta(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def apply_regime_missing_to_meta_data(", text)
        self.assertIn("def apply_regime_missing_to_pretrain_data(", text)
        self.assertIn("all_stations_full_data = apply_regime_missing_to_meta_data(", text)
        self.assertIn("clients_train_data = apply_regime_missing_to_pretrain_data(", text)

    def test_training_script_adjusts_meta_tasks_per_epoch_after_class_dropout(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def resolve_local_meta_tasks_per_epoch(", text)
        self.assertIn("len(station_tasks) // 2", text)


if __name__ == "__main__":
    unittest.main()
