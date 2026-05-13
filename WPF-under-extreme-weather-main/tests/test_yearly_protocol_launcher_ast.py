import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_FILE = ROOT / "run_three_station_yearly_protocol.py"


class YearlyProtocolLauncherAstTest(unittest.TestCase):
    def test_launcher_exposes_required_surface(self):
        self.assertTrue(LAUNCHER_FILE.exists(), "launcher file must exist")
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        self.assertIn("YEARLY_PROTOCOL_ENABLED", text)
        self.assertIn("YEARLY_PROTOCOL_METADATA_PATH", text)
        self.assertIn("TRAIN_META_ONLY_BASELINE", text)
        self.assertIn("META_SUPPORT_SHOTS", text)
        self.assertIn("META_QUERY_SHOTS", text)
        self.assertIn("choices=['build', 'train', 'eval', 'all']", text)
        self.assertIn("--smoke", text)
        self.assertIn("--dry-run", text)
        self.assertIn("--preset", text)
        self.assertIn("pilot", text)
        self.assertIn("pilot-medium", text)
        self.assertIn("formal-v1", text)

    def test_launcher_builds_expected_stage_commands_and_runtime_knobs(self):
        self.assertTrue(LAUNCHER_FILE.exists(), "launcher file must exist")
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        self.assertIn("build_three_station_extreme_yearly_protocol.py", text)
        self.assertIn("DemoModelTraining.py", text)
        self.assertIn("generate_multi_station_results.py", text)
        self.assertIn("PRETRAIN_EPOCHS", text)
        self.assertIn("PROPOSED_META_EPOCHS", text)
        self.assertIn("META_ONLY_META_EPOCHS", text)
        self.assertIn("FEW_SHOT_EPOCHS", text)
        self.assertIn("PRETRAIN_LOG_INTERVAL", text)
        self.assertIn("META_LOG_INTERVAL", text)
        self.assertIn("FEW_SHOT_LOG_INTERVAL", text)
        self.assertIn('"PRETRAIN_EPOCHS": "500"', text)
        self.assertIn('"PROPOSED_META_EPOCHS": "500"', text)
        self.assertIn('"FEW_SHOT_EPOCHS": "10"', text)
        self.assertIn('"PRETRAIN_EPOCHS": "2000"', text)
        self.assertIn('"PROPOSED_META_EPOCHS": "2000"', text)
        self.assertIn('"FEW_SHOT_EPOCHS": "20"', text)
        self.assertIn("PYTHONUNBUFFERED", text)
        self.assertIn("RUN_FEDERATED_PRETRAIN", text)
        self.assertIn('RUN_FEDERATED_PRETRAIN = "0"', text)
        self.assertIn('"RUN_FEDERATED_PRETRAIN": env.get("RUN_FEDERATED_PRETRAIN", RUN_FEDERATED_PRETRAIN)', text)
        self.assertIn('"-u"', text)
        self.assertIn("render_env_preview", text)
        self.assertIn("for stage_name in stages", text)


if __name__ == "__main__":
    unittest.main()
