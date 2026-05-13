import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_FILE = ROOT / "build_three_station_extreme_yearly_protocol.py"


class ThreeStationExtremeYearlyProtocolAstTest(unittest.TestCase):
    def test_builder_exists_and_declares_yearly_protocol_assets(self):
        self.assertTrue(
            BUILDER_FILE.exists(),
            "Task 2 requires build_three_station_extreme_yearly_protocol.py to exist.",
        )

        text = BUILDER_FILE.read_text(encoding="utf-8")

        self.assertIn("PROTOCOL_DATA_DIR", text)
        self.assertIn("PROTOCOL_METADATA_PATH", text)
        self.assertIn("protocol_data", text)
        self.assertIn("2h_6p", text)
        self.assertIn("2223jilin_058_processed_4classes.xlsx", text)
        self.assertIn("2223jilin_059_processed_4classes.xlsx", text)
        self.assertIn("2223jilin_060_processed_4classes.xlsx", text)
        self.assertIn("SUPPORT_YEAR = 2022", text)
        self.assertIn("TEST_YEAR = 2023", text)

    def test_builder_exports_explicit_2022_support_and_2023_test_extreme_keys(self):
        self.assertTrue(
            BUILDER_FILE.exists(),
            "Task 2 requires build_three_station_extreme_yearly_protocol.py to exist.",
        )

        text = BUILDER_FILE.read_text(encoding="utf-8")

        self.assertIn('f"p_extre_class{class_index + 1}"', text)
        self.assertIn('f"nwp_extre_class{class_index + 1}_"', text)
        self.assertIn('f"p_test_extre_class{class_index + 1}"', text)
        self.assertIn('f"nwp_test_extre_class{class_index + 1}_"', text)
        self.assertIn("extreme_support_window_counts", text)
        self.assertIn("extreme_test_window_counts", text)


if __name__ == "__main__":
    unittest.main()
