import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "generate_multi_station_results.py"


class HighTempOnlyEvalAstTest(unittest.TestCase):
    def test_eval_uses_metadata_driven_extreme_class_count(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        for token in [
            'extreme_class_names = protocol_metadata.get("extreme_class_names",',
            'num_extreme_classes = int(protocol_metadata.get("num_extreme_classes",',
            'extreme_eval_labels = protocol_metadata.get("extreme_eval_labels",',
            "for i_class in range(num_extreme_classes):",
            "for eval_class in range(num_extreme_classes):",
            "weather_order = list(extreme_eval_labels)",
        ]:
            self.assertIn(token, text)

    def test_eval_can_export_vanilla_and_selective_fed_meta_rows(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        for token in [
            "Vanilla Fed-Normal-Meta + Local FT",
            "Selective Fed-Normal-Meta + Local FT",
            "Local-Meta-NoFT",
        ]:
            self.assertIn(token, text)

    def test_eval_supports_explicit_baseline_triplet_mode(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        for token in [
            'EVAL_MODEL_SET = os.getenv("EVAL_MODEL_SET", "").strip()',
            'if EVAL_MODEL_SET == "baseline-triplet":',
            "'Pretrain',",
            "LOCAL_META_NOFT_MODEL_NAME,",
            "'LMT',",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
