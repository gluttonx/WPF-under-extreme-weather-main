import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class DeviceSelectionAstTest(unittest.TestCase):
    def test_device_selection_uses_cuda_fallback(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertIn('device=torch.device("cuda" if torch.cuda.is_available() else "cpu")', text)


if __name__ == "__main__":
    unittest.main()
