import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_FILE = ROOT / "build_three_station_extreme_yearly_protocol.py"


class TwoYear2024ProtocolContractAstTest(unittest.TestCase):
    def test_base_station_config_uses_separate_train_and_test_sources(self):
        text = BUILDER_FILE.read_text(encoding="utf-8")

        for token in [
            '"train_workbook": "2223jilin_058_processed_4classes.xlsx"',
            '"test_workbook": "24jilin_058_processed_4classes.xlsx"',
            '"train_capacity": 50.0',
            '"test_capacity": 50.0',
            '"train_workbook": "2223jilin_059_processed_4classes.xlsx"',
            '"test_workbook": "24jilin_059_processed_4classes.xlsx"',
            '"test_capacity": 100.0',
            '"train_workbook": "2223jilin_060_processed_4classes.xlsx"',
            '"test_workbook": "24jilin_060_processed_4classes.xlsx"',
            '"test_capacity": 300.0',
        ]:
            self.assertIn(token, text)

    def test_builder_reads_separate_train_and_test_fields(self):
        text = BUILDER_FILE.read_text(encoding="utf-8")

        for token in [
            'train_workbook = station_config["train_workbook"]',
            "test_workbook = resolve_station_test_workbook(station_config)",
            'train_capacity = station_config["train_capacity"]',
            "test_capacity = resolve_station_test_capacity(station_config)",
        ]:
            self.assertIn(token, text)

    def test_metadata_exports_two_year_protocol_fields(self):
        text = BUILDER_FILE.read_text(encoding="utf-8")

        for token in [
            '"train_years":',
            '"test_years":',
            '"normal_sampling_policy":',
        ]:
            self.assertIn(token, text)

    def test_builder_declares_two_year_normal_budget_and_k6_extreme_cap(self):
        text = BUILDER_FILE.read_text(encoding="utf-8")

        for token in [
            "NORMAL_SUPPORT_TOTAL_POINTS = 360",
            "NORMAL_SUPPORT_POINTS_BY_YEAR = {2022: 180, 2023: 180}",
            'NORMAL_SAMPLING_POLICY = "two_year_balanced_month_stratified_30d"',
            'EXTREME_SUPPORT_WINDOW_CAP = int(os.getenv("EXTREME_SUPPORT_WINDOW_CAP", "6"))',
        ]:
            self.assertIn(token, text)

    def test_builder_supports_uncapped_extreme_sampling_when_cap_is_non_positive(self):
        text = BUILDER_FILE.read_text(encoding="utf-8")

        for token in [
            'if EXTREME_SUPPORT_WINDOW_CAP <= 0:',
            'return flatten_windows(selected_windows)',
            '"all" if EXTREME_SUPPORT_WINDOW_CAP <= 0 else int(EXTREME_SUPPORT_WINDOW_CAP)',
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
