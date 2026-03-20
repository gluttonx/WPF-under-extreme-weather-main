import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class FedRepLiteAstTest(unittest.TestCase):
    def test_fedrep_lite_flags_and_helpers_exist(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("ENABLE_FEDREP_LITE =", text)
        self.assertIn('env_flag("ENABLE_FEDREP_LITE", False)', text)
        self.assertIn("FEDREP_HEAD_STEPS =", text)
        self.assertIn("FEDREP_BACKBONE_STEPS =", text)
        self.assertIn("def client_local_fedrep_round(", text)
        self.assertIn("head_stage_loss_pre_fedrep_lite", text)
        self.assertIn("backbone_stage_loss_pre_fedrep_lite", text)
        self.assertIn("client_result = client_local_fedrep_round(", text)


if __name__ == "__main__":
    unittest.main()
