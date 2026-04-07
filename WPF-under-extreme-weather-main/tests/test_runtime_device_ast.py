import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class RuntimeDeviceFallbackTest(unittest.TestCase):
    def test_training_uses_cuda_if_available_else_cpu(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertIn('device=torch.device("cuda" if torch.cuda.is_available() else "cpu")', text)

    def test_penalty_does_not_hardcode_cuda(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertNotIn('torch.tensor(1.).cuda()', text)
        self.assertIn('torch.tensor(1., device=device).requires_grad_()', text)

    def test_training_budgets_allow_env_override_for_smoke(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertIn('PRETRAIN_EPOCHS = int(os.getenv("PRETRAIN_EPOCHS", "35000"))', text)
        self.assertIn('PROPOSED_META_EPOCHS = int(os.getenv("PROPOSED_META_EPOCHS", "30000"))', text)
        self.assertIn('META_ONLY_META_EPOCHS = int(os.getenv("META_ONLY_META_EPOCHS", "30000"))', text)
        self.assertIn('FEW_SHOT_EPOCHS = int(os.getenv("FEW_SHOT_EPOCHS", "50"))', text)


if __name__ == "__main__":
    unittest.main()
