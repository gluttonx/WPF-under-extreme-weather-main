import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_FILE = ROOT / "build_three_station_extreme_yearly_protocol.py"


class HighWindSpringNoftProtocolBuilderAstTest(unittest.TestCase):
    def test_builder_declares_high_wind_spring_noft_switch_and_stationwise_k(self):
        text = BUILDER_FILE.read_text(encoding="utf-8")

        for token in [
            'HIGH_WIND_SPRING_NOFT_PROTOCOL = os.getenv("HIGH_WIND_SPRING_NOFT_PROTOCOL", "0") != "0"',
            "APRIL_MONTHS = [4]",
            "MAY_MONTHS = [5]",
            'HIGH_WIND_NORMAL_TRAIN_START = os.getenv("HIGH_WIND_NORMAL_TRAIN_START", "2022-04-01T00:00:00+00:00")',
            'HIGH_WIND_NORMAL_TRAIN_END = os.getenv("HIGH_WIND_NORMAL_TRAIN_END", "2022-04-29T16:00:00+00:00")',
            'HIGH_WIND_SPRING_BASE_STATION_IDS = {"59", "60"}',
            "HIGH_WIND_SPRING_STATION_CONVENTIONAL_CLASSES = {",
            '"59": 4',
            '"62": 4',
            '"60": 4',
            '"63": 4',
            "if HIGH_WIND_SPRING_NOFT_PROTOCOL:",
        ]:
            self.assertIn(token, text)

    def test_builder_filters_april_normal_and_may_contiguous_high_wind(self):
        text = BUILDER_FILE.read_text(encoding="utf-8")

        for token in [
            '"single_year_april_cut_normal"',
            '"single_year_april_contiguous_high_wind"',
            '("extreme_high_wind", 0)',
            "def build_contiguous_complete_windows(",
            "timedelta(hours=SAMPLE_INTERVAL_HOURS)",
            "filter_high_wind_normal_train_records",
            "sample_april_all_normal_records",
            "resolve_station_test_workbook",
            "filter_records_by_months(split_records_by_years(test_main_records, TEST_YEARS), MAY_MONTHS)",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
