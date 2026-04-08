import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class SkipStageReuseContractTest(unittest.TestCase):
    def test_training_declares_safe_skip_flags_for_reusing_checkpoints(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertIn('SKIP_LOCAL_PRETRAIN = os.getenv("SKIP_LOCAL_PRETRAIN", "0") != "0"', text)
        self.assertIn('SKIP_LOCAL_META = os.getenv("SKIP_LOCAL_META", "0") != "0"', text)

    def test_skip_paths_must_validate_checkpoint_existence_before_continue(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertIn('if SKIP_LOCAL_PRETRAIN:', text)
        self.assertIn('if SKIP_LOCAL_META:', text)
        self.assertIn('if not os.path.exists(save_path):', text)
        self.assertIn('if not os.path.exists(local_meta_path):', text)
        self.assertIn('raise FileNotFoundError(', text)


if __name__ == "__main__":
    unittest.main()
