import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class FewShotLossConfigTest(unittest.TestCase):
    def test_few_shot_uses_pure_mse_only(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertIn('FEW_SHOT_USE_CDRM = env_flag("FEW_SHOT_USE_CDRM", False)', text)

        match = re.search(
            r"def run_few_shot_adaptation\(.*?\n(.*?)\n\s*all_personalized_models",
            text,
            re.S,
        )
        self.assertIsNotNone(match)
        fn_body = match.group(1)
        self.assertIn("loss_en = loss2", fn_body)
        self.assertNotIn("if FEW_SHOT_USE_CDRM:", fn_body)
        self.assertNotIn("penalty(", fn_body)


if __name__ == "__main__":
    unittest.main()
