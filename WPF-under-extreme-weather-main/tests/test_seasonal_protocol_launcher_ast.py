import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_FILE = ROOT / "run_six_client_seasonal_protocol.py"


class SeasonalProtocolLauncherAstTest(unittest.TestCase):
    def test_launcher_exposes_required_surface(self):
        self.assertTrue(LAUNCHER_FILE.exists(), "launcher file must exist")
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        self.assertIn("SEASONAL_PROTOCOL_ENABLED", text)
        self.assertIn("SEASONAL_PROTOCOL_METADATA_PATH", text)
        self.assertIn("META_SUPPORT_SHOTS", text)
        self.assertIn("META_QUERY_SHOTS", text)
        self.assertIn("choices=['build', 'train', 'eval', 'all']", text)
        self.assertIn("--smoke", text)
        self.assertIn("--dry-run", text)
        self.assertIn("--preset", text)
        self.assertIn("formal-v1", text)

    def test_launcher_builds_expected_stage_commands(self):
        self.assertTrue(LAUNCHER_FILE.exists(), "launcher file must exist")
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        self.assertIn("build_six_client_seasonal_protocol.py", text)
        self.assertIn("DemoModelTraining.py", text)
        self.assertIn("generate_multi_station_results.py", text)
        self.assertIn("PRETRAIN_EPOCHS", text)
        self.assertIn("PROPOSED_META_EPOCHS", text)
        self.assertIn("META_ONLY_META_EPOCHS", text)
        self.assertIn("FEW_SHOT_EPOCHS", text)
        self.assertIn("STRICT_PAPER_ORDER", text)
        self.assertIn("PROPOSED_META_SAMPLER_MODE", text)
        self.assertIn("CONVENTIONAL_RATIO", text)
        self.assertIn("REGIME_MISSING_MODE", text)
        self.assertIn("for stage_name in stages", text)
        self.assertIn("PYTHONUNBUFFERED", text)
        self.assertIn('"-u"', text)
        self.assertIn("render_env_preview", text)


if __name__ == "__main__":
    unittest.main()
