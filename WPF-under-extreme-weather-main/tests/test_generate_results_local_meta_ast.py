import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "generate_multi_station_results.py"


class GenerateResultsLocalMetaAstTest(unittest.TestCase):
    def test_eval_model_matches_current_training_architecture(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        self.assertNotIn("self.shared_adapter = model.SharedAdapter(", text)
        self.assertNotIn("y = self.shared_adapter(y)", text)
        self.assertIn("y = self.fore_baselearner(y)", text)

    def test_eval_script_uses_current_fed_pretrain_and_station_local_meta_artifacts(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        self.assertIn("model_fore_pre_federated.pth", text)
        self.assertIn("model_fore_train_task_query_meta_only_station{station_id}.pth", text)
        self.assertNotIn("model_fore_pre_station{station_id}_personalized.pth", text)
        self.assertNotIn("model_fore_meta_station{station_id}_personalized.pth", text)
        self.assertNotIn("model_fore_meta_only_station{station_id}_personalized.pth", text)


if __name__ == "__main__":
    unittest.main()
