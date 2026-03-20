import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class Phase2PerfAstTest(unittest.TestCase):
    def test_phase2_perf_helpers_exist(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertIn("local_meta_task_cache =", text)
        self.assertIn("def prepare_local_meta_task_cache(", text)
        self.assertIn("def create_phase2_station_context(", text)
        self.assertIn("def reset_optimizer_state(", text)
        self.assertIn("def clone_state_dict(state_dict, target_device=None)", text)


if __name__ == "__main__":
    unittest.main()
