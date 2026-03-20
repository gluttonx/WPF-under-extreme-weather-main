import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class Phase2PerfWave2AstTest(unittest.TestCase):
    def test_wave2_perf_flags_and_batching_helpers_exist(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertIn("PERF_PRIORITIZE_SPEED =", text)
        self.assertIn("F2L_BATCH_LOCAL_TASKS =", text)
        self.assertIn("def sample_local_meta_task_batch(", text)
        self.assertIn("if PERF_PRIORITIZE_SPEED and torch.cuda.is_available():", text)


if __name__ == "__main__":
    unittest.main()
