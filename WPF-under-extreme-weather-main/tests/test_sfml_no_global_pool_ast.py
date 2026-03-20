import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class SFMLNoGlobalPoolAstTest(unittest.TestCase):
    def test_strict_meta_does_not_use_global_task_pool(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        start = text.index("def run_strict_federated_meta_training(")
        end = text.index("model_fore_pre = build_forecast_model(")
        strict_meta_block = text[start:end]

        self.assertNotIn("sample_meta_batch(", strict_meta_block)
        self.assertNotIn("sample_legacy_global_meta_batch(", strict_meta_block)
        self.assertNotIn("task_pool.append((station_id, i_class))", strict_meta_block)
        self.assertIn("client_local_meta_round(", strict_meta_block)


if __name__ == "__main__":
    unittest.main()
