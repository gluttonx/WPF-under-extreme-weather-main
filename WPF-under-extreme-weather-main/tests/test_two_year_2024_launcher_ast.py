import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_FILE = ROOT / "run_three_station_yearly_protocol.py"


class TwoYear2024LauncherAstTest(unittest.TestCase):
    def test_launcher_declares_new_two_year_k6_protocol_constants(self):
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        for token in [
            "TWO_YEAR_2024_K6_PROTOCOL_NAME",
            "TWO_YEAR_2024_K6_PROTOCOL_DATA_DIR",
            "TWO_YEAR_2024_K6_ARTIFACT_DIR",
            "TWO_YEAR_2024_ALL_EXTREME_PROTOCOL_NAME",
            "TWO_YEAR_2024_ALL_EXTREME_PROTOCOL_DATA_DIR",
            "TWO_YEAR_2024_ALL_EXTREME_ARTIFACT_DIR",
            "two_year_24_k6",
            "two_year_24_all_extreme",
        ]:
            self.assertIn(token, text)

    def test_launcher_exposes_new_protocol_switch(self):
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        for token in [
            "--two-year-2024-k6",
            "--two-year-2024-all-extreme",
            "two_year_2024_k6",
            "two_year_2024_all_extreme",
            '"PHASE_AUGMENT_STATIONS": "1"',
            '"META_SUPPORT_SHOTS": "3"',
            '"META_QUERY_SHOTS": "3"',
            '"EXTREME_SUPPORT_WINDOW_CAP": "0"',
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
