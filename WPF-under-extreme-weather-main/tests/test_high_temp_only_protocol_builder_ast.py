import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_FILE = ROOT / "build_three_station_extreme_yearly_protocol.py"


class HighTempOnlyProtocolBuilderAstTest(unittest.TestCase):
    def test_builder_declares_high_temp_only_summer_switch_and_stationwise_k(self):
        text = BUILDER_FILE.read_text(encoding="utf-8")

        for token in [
            'HIGH_TEMP_ONLY_SUMMER_PROTOCOL = os.getenv("HIGH_TEMP_ONLY_SUMMER_PROTOCOL", "0") != "0"',
            "SUMMER_MONTHS = [6, 7, 8]",
            "HIGH_TEMP_ONLY_STATION_CONVENTIONAL_CLASSES = {",
            '"58": 7',
            '"61": 7',
            '"59": 6',
            '"62": 6',
            '"60": 5',
            '"63": 5',
            "def resolve_num_conventional_classes(",
        ]:
            self.assertIn(token, text)

    def test_builder_filters_summer_normal_and_exports_single_extreme_class_metadata(self):
        text = BUILDER_FILE.read_text(encoding="utf-8")

        for token in [
            "def filter_records_by_months(",
            "if HIGH_TEMP_ONLY_SUMMER_PROTOCOL:",
            '"single_year_summer_all_available_normal"',
            '"single_year_summer_all_available_high_temp"',
            '("extreme_high_temp", 0)',
            '"num_extreme_classes": len(EXTREME_CLASS_NAMES)',
            '"extreme_eval_labels": EXTREME_EVAL_LABELS',
            '"conventional_classes_by_station":',
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
