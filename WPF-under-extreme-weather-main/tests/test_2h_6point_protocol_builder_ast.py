import importlib.util
import os
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BUILDER_FILE = ROOT / "build_three_station_extreme_yearly_protocol.py"


def load_builder_module():
    spec = importlib.util.spec_from_file_location("builder_2h_6point", BUILDER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TwoHourSixPointProtocolBuilderAstTest(unittest.TestCase):
    def test_builder_exposes_protocol_configuration(self):
        text = BUILDER_FILE.read_text(encoding="utf-8")

        for name in [
            "PROTOCOL_NAME",
            "SAMPLE_INTERVAL_HOURS",
            "DOWNSAMPLE_OFFSET",
            "LEN_REALP",
            "POINTS_PER_DAY",
            "WINDOW_SPAN_HOURS",
            "PROTOCOL_DATA_DIR",
            "PROTOCOL_METADATA_PATH",
        ]:
            self.assertIn(name, text)

        self.assertIn("os.getenv", text)
        self.assertIn("protocol_data", text)
        self.assertIn("2h_6p", text)
        self.assertNotIn('OUTPUT_DIR = ROOT / "three_station_yearly_protocol_data"', text)

    def test_downsample_records_uses_fixed_row_phase(self):
        builder = load_builder_module()
        base = datetime(2022, 1, 1)
        records = [
            builder.SheetRecord(date=base + timedelta(hours=index), values={"Power2": float(index)})
            for index in range(6)
        ]

        thinned = builder.downsample_records(records, interval_hours=2, offset=1)
        self.assertEqual([record.values["Power2"] for record in thinned], [1.0, 3.0, 5.0])

        untouched = builder.downsample_records(records, interval_hours=1, offset=0)
        self.assertEqual([record.values["Power2"] for record in untouched], list(range(6)))

    def test_metadata_contains_protocol_fields(self):
        text = BUILDER_FILE.read_text(encoding="utf-8")

        for field in [
            '"protocol_name": PROTOCOL_NAME',
            '"sample_interval_hours": SAMPLE_INTERVAL_HOURS',
            '"downsample_offset": DOWNSAMPLE_OFFSET',
            '"len_realp": LEN_REALP',
            '"points_per_day": POINTS_PER_DAY',
            '"window_span_hours": WINDOW_SPAN_HOURS',
            '"protocol_data_dir": str(OUTPUT_DIR)',
        ]:
            self.assertIn(field, text)

    def test_builder_exposes_six_station_phase_augmentation(self):
        text = BUILDER_FILE.read_text(encoding="utf-8")

        for token in [
            "PHASE_AUGMENT_STATIONS",
            "PHASE_AUGMENT_STATION_MAP",
            "build_protocol_station_configs",
            '"61"',
            '"62"',
            '"63"',
            '"downsample_offset": station_downsample_offset',
        ]:
            self.assertIn(token, text)

    def test_phase_augmentation_builds_61_62_63_with_complementary_offset(self):
        with patch.dict(
            os.environ,
            {
                "PHASE_AUGMENT_STATIONS": "1",
                "SAMPLE_INTERVAL_HOURS": "2",
                "DOWNSAMPLE_OFFSET": "1",
            },
            clear=False,
        ):
            builder = load_builder_module()

        station_configs = builder.build_protocol_station_configs()
        station_ids = [station["station_id"] for station in station_configs]
        offsets = {station["station_id"]: station["downsample_offset"] for station in station_configs}

        self.assertEqual(station_ids, ["58", "59", "60", "61", "62", "63"])
        self.assertEqual(offsets["58"], 1)
        self.assertEqual(offsets["59"], 1)
        self.assertEqual(offsets["60"], 1)
        self.assertEqual(offsets["61"], 0)
        self.assertEqual(offsets["62"], 0)
        self.assertEqual(offsets["63"], 0)

    def test_four_hour_phase_augmentation_uses_half_phase_complement(self):
        with patch.dict(
            os.environ,
            {
                "PHASE_AUGMENT_STATIONS": "1",
                "SAMPLE_INTERVAL_HOURS": "4",
                "DOWNSAMPLE_OFFSET": "1",
                "LEN_REALP": "3",
                "POINTS_PER_DAY": "6",
            },
            clear=False,
        ):
            builder = load_builder_module()

        station_configs = builder.build_protocol_station_configs()
        station_ids = [station["station_id"] for station in station_configs]
        offsets = {station["station_id"]: station["downsample_offset"] for station in station_configs}

        self.assertEqual(station_ids, ["58", "59", "60", "61", "62", "63"])
        self.assertEqual(offsets["58"], 1)
        self.assertEqual(offsets["59"], 1)
        self.assertEqual(offsets["60"], 1)
        self.assertEqual(offsets["61"], 3)
        self.assertEqual(offsets["62"], 3)
        self.assertEqual(offsets["63"], 3)


if __name__ == "__main__":
    unittest.main()
