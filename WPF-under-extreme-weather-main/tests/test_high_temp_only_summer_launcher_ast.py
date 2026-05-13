import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_FILE = ROOT / "run_three_station_yearly_protocol.py"


class HighTempOnlySummerLauncherAstTest(unittest.TestCase):
    def test_launcher_declares_high_temp_only_protocol_constants_and_flag(self):
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        for token in [
            "HIGH_TEMP_ONLY_SUMMER_PROTOCOL_NAME",
            "HIGH_TEMP_ONLY_SUMMER_PROTOCOL_DATA_DIR",
            "HIGH_TEMP_ONLY_SUMMER_ARTIFACT_DIR",
            "--high-temp-only-summer",
            "high_temp_only_summer",
            "HIGH_TEMP_ONLY_SUMMER_PROTOCOL",
            "--meta-shot-regime",
            '"2x4"',
            '"3x3"',
        ]:
            self.assertIn(token, text)

    def test_launcher_wires_single_class_summer_defaults_and_meta_shot_regimes(self):
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        for token in [
            '"HIGH_TEMP_ONLY_SUMMER_PROTOCOL": "1"',
            '"EXTREME_SUPPORT_WINDOW_CAP": "0"',
            '"META_TASKS_PER_EPOCH": "2"',
            '"META_SUPPORT_SHOTS": "4"',
            '"META_QUERY_SHOTS": "4"',
            '"META_TASKS_PER_EPOCH": "3"',
            '"META_SUPPORT_SHOTS": "3"',
            '"META_QUERY_SHOTS": "3"',
        ]:
            self.assertIn(token, text)

    def test_launcher_declares_selective_fed_meta_flags_and_preview_keys(self):
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        for token in [
            "--enable-selective-fed-normal-meta",
            "ENABLE_SELECTIVE_FED_NORMAL_META",
            "SELECTIVE_FED_META_PROXY_RATIO",
            "SELECTIVE_FED_META_SELF_FLOOR",
            "SELECTIVE_FED_META_GAIN_MARGIN",
            "SELECTIVE_FED_META_GAIN_GAMMA",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
