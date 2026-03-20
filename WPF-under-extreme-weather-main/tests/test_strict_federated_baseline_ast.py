import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class StrictFederatedBaselineConfigTest(unittest.TestCase):
    def test_strict_federated_baseline_helpers_and_flags_exist(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("ENABLE_FED_META_TRAIN =", text)
        self.assertIn('env_flag("ENABLE_FED_META_TRAIN", False)', text)
        self.assertIn("def is_shared_param(", text)
        self.assertIn("def is_local_param(", text)
        self.assertIn("def extract_shared_state_dict(", text)
        self.assertIn("def extract_local_state_dict(", text)
        self.assertIn("def load_mixed_state_dict(", text)
        self.assertIn("def server_aggregate_shared_updates(", text)
        self.assertIn("def client_local_pretrain_round(", text)
        self.assertIn("station_shared_states = {", text)
        self.assertNotIn("shared_global_state = server_aggregate_shared_updates(", text)


if __name__ == "__main__":
    unittest.main()
