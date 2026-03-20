import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class RuntimeEnvConfigAstTest(unittest.TestCase):
    def test_training_script_exposes_core_runtime_knobs_via_env(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def env_flag(", text)
        self.assertIn("def env_int(", text)
        self.assertIn('USE_FEDERATION = env_flag("USE_FEDERATION", True)', text)
        self.assertIn('TRAIN_META_ONLY_BASELINE = env_flag("TRAIN_META_ONLY_BASELINE", True)', text)
        self.assertIn('FEW_SHOT_EPOCHS = env_int("FEW_SHOT_EPOCHS", 50)', text)
        self.assertIn('META_TASKS_PER_EPOCH = env_int("META_TASKS_PER_EPOCH", 5)', text)
        self.assertIn('PRETRAIN_EPOCHS = env_int("PRETRAIN_EPOCHS", 35000)', text)
        self.assertIn('PROPOSED_META_EPOCHS = env_int("PROPOSED_META_EPOCHS", 30000)', text)
        self.assertIn('META_ONLY_META_EPOCHS = env_int("META_ONLY_META_EPOCHS", 30000)', text)
        self.assertIn('FEW_SHOT_USE_CDRM = env_flag("FEW_SHOT_USE_CDRM", False)', text)
        self.assertIn('META_ONLY_USE_CDRM = env_flag("META_ONLY_USE_CDRM", True)', text)
        self.assertIn('META_ONLY_TRAIN_ALL_PARAMS = env_flag("META_ONLY_TRAIN_ALL_PARAMS", False)', text)
        self.assertIn('META_ONLY_DISABLE_LWP = env_flag("META_ONLY_DISABLE_LWP", False)', text)


if __name__ == "__main__":
    unittest.main()
