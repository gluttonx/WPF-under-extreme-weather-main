import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class FewShotLossConfigTest(unittest.TestCase):
    def test_few_shot_uses_pure_mse_only(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertIn("FEW_SHOT_USE_CDRM = False", text)

        match = re.search(
            r"def adapt_state_dict\(.*?\n(.*?)\n\s*def run_few_shot_adaptation",
            text,
            re.S,
        )
        self.assertIsNotNone(match)
        fn_body = match.group(1)
        self.assertNotIn("loss_en =", fn_body)
        self.assertNotIn("penalty(", fn_body)
        self.assertIn("loss_mse = loss_fn_1(outputs, adapt_target_device)", fn_body)
        self.assertIn("loss_mse.backward()", fn_body)


if __name__ == "__main__":
    unittest.main()
