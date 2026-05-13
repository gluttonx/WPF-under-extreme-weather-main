import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_FILE = ROOT / "run_three_station_yearly_protocol.py"


class TwoHourSixPointLauncherAstTest(unittest.TestCase):
    def test_launcher_exposes_2h_6p_protocol_env(self):
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        for token in [
            "three_station_2h_6point_protocol",
            "six_station_2h_6point_phase_augmented_protocol",
            "six_station_4h_3point_phase_augmented_protocol",
            "hybrid_4h3p_normal_2h6p_extreme_protocol",
            "protocol_data",
            "2h_6p",
            "2h_6p_six_station",
            "4h_3p_six_station",
            "PROTOCOL_NAME",
            "PROTOCOL_DATA_DIR",
            "PROTOCOL_METADATA_PATH",
            "SAMPLE_INTERVAL_HOURS",
            "DOWNSAMPLE_OFFSET",
            "LEN_REALP",
            "POINTS_PER_DAY",
            "PHASE_AUGMENT_STATIONS",
            "PHASE_AUGMENT_COMPLEMENTARY_OFFSET",
            "ARTIFACT_DIR",
            "MODEL_OUTPUT_DIR",
            "BASE_MODEL_OUTPUT_DIR",
            "LOGS_TRAIN_DIR",
            "EXTREME_TARGET_REFINEMENT_EPOCHS",
        ]:
            self.assertIn(token, text)

    def test_launcher_exposes_reduced_epoch_presets(self):
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        for token in [
            "pilot-1k",
            "pilot-5k",
            "final-candidate",
            '"PRETRAIN_EPOCHS": "1000"',
            '"PROPOSED_META_EPOCHS": "1000"',
            '"FEW_SHOT_EPOCHS": "50"',
            '"EXTREME_TARGET_REFINEMENT_EPOCHS": "50"',
            '"PRETRAIN_EPOCHS": "5000"',
            '"PROPOSED_META_EPOCHS": "5000"',
            '"FEW_SHOT_EPOCHS": "200"',
            '"EXTREME_TARGET_REFINEMENT_EPOCHS": "200"',
            '"PRETRAIN_EPOCHS": "10000"',
            '"PROPOSED_META_EPOCHS": "10000"',
            '"FEW_SHOT_EPOCHS": "500"',
            '"EXTREME_TARGET_REFINEMENT_EPOCHS": "500"',
        ]:
            self.assertIn(token, text)

    def test_launcher_supports_six_station_flag(self):
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        self.assertIn("--six-station", text)
        self.assertIn("--four-hour", text)
        self.assertIn("six_station", text)
        self.assertIn('"PHASE_AUGMENT_STATIONS": "1"', text)

    def test_launcher_supports_four_hour_six_station_protocol(self):
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        for token in [
            "four_hour",
            "SIX_STATION_4H3P_PROTOCOL_NAME",
            "SIX_STATION_4H3P_PROTOCOL_DATA_DIR",
            "SIX_STATION_4H3P_ARTIFACT_DIR",
            '"SAMPLE_INTERVAL_HOURS": "4"',
            '"LEN_REALP": "3"',
            '"POINTS_PER_DAY": "6"',
            '"PHASE_AUGMENT_COMPLEMENTARY_OFFSET": "3"',
        ]:
            self.assertIn(token, text)

    def test_launcher_supports_hybrid_4h3p_normal_2h6p_extreme_protocol(self):
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        for token in [
            "--hybrid-extreme-2h",
            "hybrid_extreme_2h",
            "HYBRID_4H3P_NORMAL_2H6P_EXTREME_PROTOCOL_NAME",
            "HYBRID_4H3P_NORMAL_2H6P_EXTREME_ARTIFACT_DIR",
            "DEFAULT_HYBRID_BASE_MODEL_OUTPUT_DIR",
            '"BASE_MODEL_OUTPUT_DIR"',
            '"SKIP_LOCAL_PRETRAIN": "1"',
            '"SKIP_LOCAL_META": "1"',
            '"SKIP_FED_NORMAL_META": "1"',
            '"SAMPLE_INTERVAL_HOURS": "2"',
            '"LEN_REALP": "6"',
            '"POINTS_PER_DAY": "12"',
        ]:
            self.assertIn(token, text)

    def test_launcher_isolates_six_station_artifacts(self):
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        self.assertIn("SIX_STATION_ARTIFACT_DIR", text)
        self.assertIn("MODEL_OUTPUT_DIR", text)
        self.assertIn("LOGS_TRAIN_DIR", text)
        self.assertIn("artifacts", text)

    def test_launcher_previews_fed_normal_meta_env(self):
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        for token in [
            '"ENABLE_FED_NORMAL_META_PROPOSED"',
            '"FED_NORMAL_META_SELF_FLOOR"',
            '"SKIP_FED_NORMAL_META"',
            '"FED_NORMAL_META_RESTORE_BEST"',
            '"FED_NORMAL_META_SAVE_BEST"',
            '"FED_NORMAL_META_USE_BEST"',
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
