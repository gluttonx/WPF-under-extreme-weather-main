import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"
RESULTS_FILE = ROOT / "generate_multi_station_results.py"


class LocalizedLMTContractTest(unittest.TestCase):
    def test_training_uses_station_local_paths_for_lmt(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertIn("def get_local_pretrain_model_path(station_id):", text)
        self.assertIn("def get_local_meta_model_path(station_id):", text)
        self.assertIn("def get_local_meta_only_model_path(station_id):", text)
        self.assertIn('base_model_path=get_local_meta_model_path(station_id)', text)
        self.assertNotIn('base_model_path=PROPOSED_META_MODEL_PATH', text)

    def test_results_export_uses_lmt_and_station_local_pretrain(self):
        text = RESULTS_FILE.read_text(encoding="utf-8")
        self.assertRegex(
            text,
            r"model_names\s*=\s*\[\s*'LMT'\s*,\s*'Meta_Learning'\s*,\s*'Pre_Training'\s*\]",
        )
        self.assertIn(
            "pre_model_candidates = [",
            text,
        )
        self.assertIn(
            "f'model_fore_pre_station{station_id}_local.pth'",
            text,
        )


if __name__ == "__main__":
    unittest.main()
