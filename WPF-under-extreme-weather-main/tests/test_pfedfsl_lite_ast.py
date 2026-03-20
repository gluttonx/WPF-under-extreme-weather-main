import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class PFedFslLiteAstTest(unittest.TestCase):
    def test_pfedfsl_lite_flags_and_helpers_exist(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("ENABLE_PFEDFSL_LITE =", text)
        self.assertIn('env_flag("ENABLE_PFEDFSL_LITE", False)', text)
        self.assertIn("PFED_ROUTE_RATIO =", text)
        self.assertIn("PFED_ROUTE_TEMPERATURE =", text)
        self.assertIn("def split_route_update_batch(", text)
        self.assertIn("def compute_route_softmax_weights(", text)
        self.assertIn("def aggregate_shared_states_by_weights(", text)
        self.assertIn("def client_local_pfedfsl_lite_round(", text)
        self.assertIn("client_route_weight_maps", text)
        self.assertIn("client_route_loss_maps", text)
        self.assertIn("route_weight_pre_pfedfsl_lite_client", text)
        self.assertIn("route_loss_pre_pfedfsl_lite_client", text)

    def test_f2l_no_longer_enabled_by_default(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertIn('env_flag("ENABLE_F2L_PHASE2", False)', text)

    def test_single_global_fedavg_assignment_removed_from_mainline(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertNotIn("shared_global_state = server_aggregate_shared_updates(", text)
        self.assertRegex(text, r"station_shared_states\s*=\s*\{")


if __name__ == "__main__":
    unittest.main()
