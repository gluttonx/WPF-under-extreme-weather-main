import json
import os
import unittest
from pathlib import Path

import scipy.io as scio


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_DIR = ROOT / "protocol_data" / "2h_6p"
SIX_STATION_PROTOCOL_DIR = ROOT / "protocol_data" / "2h_6p_six_station"
PROTOCOL_DIR = Path(
    os.getenv(
        "PROTOCOL_DATA_DIR",
        str(SIX_STATION_PROTOCOL_DIR if (SIX_STATION_PROTOCOL_DIR / "protocol_metadata.json").exists() else DEFAULT_PROTOCOL_DIR),
    )
)
METADATA_PATH = PROTOCOL_DIR / "protocol_metadata.json"


class TwoHourSixPointGeneratedShapesTest(unittest.TestCase):
    def setUp(self):
        if not METADATA_PATH.exists():
            self.skipTest("2h/6p protocol metadata has not been generated yet")

    def test_generated_metadata_declares_2h_6point_protocol(self):
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

        self.assertIn(
            metadata["protocol_name"],
            {
                "three_station_2h_6point_protocol",
                "six_station_2h_6point_phase_augmented_protocol",
            },
        )
        self.assertEqual(metadata["sample_interval_hours"], 2)
        self.assertEqual(metadata["downsample_offset"], 1)
        self.assertEqual(metadata["len_realp"], 6)
        self.assertEqual(metadata["points_per_day"], 12)
        self.assertEqual(metadata["window_span_hours"], 12)

    def test_generated_station_mats_have_730_full_test_windows(self):
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        len_realp = int(metadata["len_realp"])
        points_per_day = int(metadata["points_per_day"])
        expected_test_points = 365 * points_per_day

        station_ids = [str(station["station_id"]) for station in metadata["stations"]]
        for station_id in station_ids:
            mat_path = PROTOCOL_DIR / f"{station_id}wf_4_train.mat"
            self.assertTrue(mat_path.exists(), f"missing {mat_path}")
            payload = scio.loadmat(mat_path)

            self.assertIn("p_1h", payload)
            self.assertIn("nwp_1h", payload)
            self.assertIn("p_2h", payload)
            self.assertIn("nwp_2h", payload)
            self.assertIn("p_test", payload)
            self.assertIn("nwp_test", payload)
            self.assertEqual(payload["p_test"].shape[0], expected_test_points)
            self.assertEqual(payload["p_test"].shape[0] // len_realp, 730)

    def test_extreme_test_payloads_are_available(self):
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        station_ids = [str(station["station_id"]) for station in metadata["stations"]]
        for station_id in station_ids:
            payload = scio.loadmat(PROTOCOL_DIR / f"{station_id}wf_4_train.mat")
            for class_index in range(1, 5):
                self.assertIn(f"p_extre_class{class_index}", payload)
                self.assertIn(f"nwp_extre_class{class_index}_", payload)
                self.assertIn(f"p_test_extre_class{class_index}", payload)
                self.assertIn(f"nwp_test_extre_class{class_index}_", payload)

    def test_six_station_phase_augmented_metadata_when_enabled(self):
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if not metadata.get("phase_augmentation_enabled", False):
            self.skipTest("metadata is not six-station phase augmented")

        station_ids = [str(station["station_id"]) for station in metadata["stations"]]
        offsets = {str(station["station_id"]): int(station["downsample_offset"]) for station in metadata["stations"]}

        self.assertEqual(station_ids, ["58", "59", "60", "61", "62", "63"])
        self.assertEqual(offsets["58"], 1)
        self.assertEqual(offsets["59"], 1)
        self.assertEqual(offsets["60"], 1)
        self.assertEqual(offsets["61"], 0)
        self.assertEqual(offsets["62"], 0)
        self.assertEqual(offsets["63"], 0)


if __name__ == "__main__":
    unittest.main()
