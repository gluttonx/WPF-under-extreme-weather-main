import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "generate_multi_station_results.py"


class StrictFederatedEvaluationConfigTest(unittest.TestCase):
    def test_eval_model_definition_includes_shared_adapter(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        self.assertIn("self.shared_adapter = model.SharedAdapter(", text)
        self.assertIn("y = self.shared_adapter(y)", text)

    def test_eval_script_knows_station_personalized_pretrain_and_meta_fallback(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        self.assertIn("def resolve_station_pretrain_model_path(", text)
        self.assertIn("def resolve_meta_learning_model_path(", text)
        self.assertIn("def detect_strict_fed_meta_enabled_from_training_script(", text)
        self.assertIn("def detect_strict_fed_meta_only_available_from_training_script(", text)
        self.assertIn("def detect_strict_fed_meta_only_artifacts_available(", text)
        self.assertIn("def resolve_meta_learning_row_mode(", text)
        self.assertIn("def resolve_output_model_names(", text)
        self.assertIn("model_fore_pre_station{station_id}_personalized.pth", text)
        self.assertIn("model_fore_meta_station{station_id}_personalized.pth", text)
        self.assertIn("model_fore_meta_only_station{station_id}_personalized.pth", text)
        self.assertIn("meta_learning_available", text)

    def test_eval_script_distinguishes_omit_vs_na_meta_learning_row(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        self.assertIn("model_names = resolve_output_model_names(", text)
        self.assertIn("strict baseline 最终表不输出 Meta_Learning 行", text)
        self.assertIn("Meta_Learning 行将以 N/A 导出", text)


if __name__ == "__main__":
    unittest.main()
