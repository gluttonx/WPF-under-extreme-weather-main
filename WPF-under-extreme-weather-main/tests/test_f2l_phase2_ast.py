import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class F2LPhase2AstTest(unittest.TestCase):
    def test_phase2_flags_and_helpers_exist(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("ENABLE_F2L_PHASE2 =", text)
        self.assertIn("F2L_LOCAL_TASKS_PER_ROUND", text)
        self.assertIn("F2L_FINE_TUNE_STEPS", text)
        self.assertIn("F2L_LAMBDA_MI", text)
        self.assertIn("F2L_LAMBDA_KD", text)
        self.assertIn("def forward_features(", text)
        self.assertIn("def forward_with_features(", text)
        self.assertIn("def sample_local_meta_task(", text)
        self.assertIn("def client_local_f2l_round(", text)
        self.assertIn("def compute_f2l_mi_proxy_loss(", text)
        self.assertIn("def compute_f2l_kd_loss(", text)
        self.assertIn("if ENABLE_F2L_PHASE2:", text)


if __name__ == "__main__":
    unittest.main()
