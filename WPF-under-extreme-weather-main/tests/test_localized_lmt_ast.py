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
        self.assertIn("get_local_meta_model_path(station_id) if USE_FEDERATION and not USE_PSEUDO_FED else PROPOSED_META_MODEL_PATH", text)
        self.assertIn('model_label="LMT"', text)

    def test_results_export_uses_three_model_main_table(self):
        text = RESULTS_FILE.read_text(encoding="utf-8")

        for token in [
            "def resolve_model_names():",
            "'LMT'",
            "'Extreme-FedAvg'",
            "'Proposed-A'",
        ]:
            self.assertIn(token, text)

        self.assertNotIn("'Meta_Learning'", text)
        self.assertNotIn("'Pre_Training'", text)


if __name__ == "__main__":
    unittest.main()
