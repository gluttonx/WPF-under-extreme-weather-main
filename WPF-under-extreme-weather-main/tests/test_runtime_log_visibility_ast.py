import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class RuntimeLogVisibilityAstTest(unittest.TestCase):
    def test_training_script_exposes_progress_logging_helpers(self):
        self.assertTrue(TRAIN_FILE.exists(), "training script must exist")
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def progress_log(", text)
        self.assertIn("flush=True", text)
        self.assertIn("def should_log_epoch(", text)
        self.assertIn("def log_stage_banner(", text)
        self.assertIn('PRETRAIN_LOG_INTERVAL = env_int("PRETRAIN_LOG_INTERVAL", 100)', text)
        self.assertIn('META_LOG_INTERVAL = env_int("META_LOG_INTERVAL", 100)', text)
        self.assertIn('FEW_SHOT_LOG_INTERVAL = env_int("FEW_SHOT_LOG_INTERVAL", 1)', text)

    def test_training_script_uses_progress_logging_in_long_stages(self):
        self.assertTrue(TRAIN_FILE.exists(), "training script must exist")
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("should_log_epoch(", text)
        self.assertIn('stage_type="federated_pretrain"', text)
        self.assertIn('stage_type="local_meta"', text)
        self.assertIn('stage_type="few_shot"', text)
        self.assertIn("progress_log(", text)
        self.assertIn("log_stage_banner(", text)
        self.assertIn("PRETRAIN_LOG_INTERVAL", text)
        self.assertIn("META_LOG_INTERVAL", text)
        self.assertIn("FEW_SHOT_LOG_INTERVAL", text)
        self.assertIn("首次达到收敛条件", text)


if __name__ == "__main__":
    unittest.main()
