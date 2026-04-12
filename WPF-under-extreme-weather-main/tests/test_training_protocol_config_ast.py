import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class TrainingProtocolConfigAstTest(unittest.TestCase):
    def test_training_script_reads_protocol_config(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            "PROTOCOL_DATA_DIR",
            "PROTOCOL_METADATA_PATH",
            "PROTOCOL_NAME",
            "SAMPLE_INTERVAL_HOURS",
            "DOWNSAMPLE_OFFSET",
            "LEN_REALP",
            "POINTS_PER_DAY",
            "WINDOW_SPAN_HOURS",
            "os.getenv",
        ]:
            self.assertIn(token, text)

    def test_training_script_resolves_station_mats_from_protocol_dir(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def resolve_station_mat_path(station_id):", text)
        self.assertIn("PROTOCOL_DATA_DIR", text)
        self.assertIn("scio.loadmat(resolve_station_mat_path(station_id))", text)
        self.assertNotIn("scio.loadmat(dataFile)", text)

    def test_training_uses_protocol_test_payload_when_available(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("'p_test_00': wf_1.get('p_test')", text)
        self.assertIn("'nwp_test_00': wf_1.get('nwp_test')", text)
        self.assertIn("station_data[station_id]['p_test_00'] is not None", text)
        self.assertIn("test_target_p_st = station_data[station_id]['p_test_00']", text)
        self.assertIn("test_nwp_st = station_data[station_id]['nwp_test_00']", text)

    def test_training_script_does_not_pin_12_point_protocol(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("len_realp = LEN_REALP", text)
        self.assertIn("d = POINTS_PER_DAY", text)
        self.assertNotIn("len_realp=12", text)
        self.assertNotIn("d=24", text)

    def test_training_report_exports_protocol_fields(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for field in [
            '"protocol_name": PROTOCOL_NAME',
            '"protocol_data_dir": PROTOCOL_DATA_DIR',
            '"protocol_metadata_path": PROTOCOL_METADATA_PATH',
            '"sample_interval_hours": SAMPLE_INTERVAL_HOURS',
            '"downsample_offset": DOWNSAMPLE_OFFSET',
            '"len_realp": LEN_REALP',
            '"points_per_day": POINTS_PER_DAY',
        ]:
            self.assertIn(field, text)

    def test_training_reads_station_ids_from_protocol_metadata(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def resolve_station_ids():", text)
        self.assertIn("yearly_protocol_metadata.get(\"stations\", [])", text)
        self.assertIn("return [str(station[\"station_id\"]) for station in metadata_stations]", text)

    def test_training_source_loop_only_skips_exact_self(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("for source_station_id in station_ids:", text)
        self.assertIn("if source_station_id == target_station_id:", text)
        self.assertNotIn("physical_station", text)
        self.assertNotIn("same_physical", text)

    def test_training_artifacts_are_path_isolated(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            "ARTIFACT_DIR",
            "MODEL_OUTPUT_DIR",
            "LOGS_TRAIN_DIR",
            "resolve_artifact_path",
            "resolve_model_path",
            "ALL_STATIONS_TEST_RESULTS_PATH",
            "os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)",
            "SummaryWriter(os.path.join(LOGS_TRAIN_DIR, \"loss1\"))",
            "SummaryWriter(os.path.join(LOGS_TRAIN_DIR, \"loss2\"))",
        ]:
            self.assertIn(token, text)

        self.assertNotIn("SummaryWriter(\"./logs_train/loss1\")", text)
        self.assertNotIn("scio.savemat('all_stations_test_results.mat'", text)

    def test_training_declares_fed_normal_meta_proposed_config(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            'ENABLE_FED_NORMAL_META_PROPOSED = os.getenv("ENABLE_FED_NORMAL_META_PROPOSED", "0") != "0"',
            'FED_NORMAL_META_SELF_FLOOR = float(os.getenv("FED_NORMAL_META_SELF_FLOOR", "0.3"))',
            'SKIP_FED_NORMAL_META = os.getenv("SKIP_FED_NORMAL_META", "0") != "0"',
            "def get_fed_normal_meta_support_model_path(station_id):",
            "def get_fed_normal_meta_model_path(station_id):",
            "model_fore_train_task_support_fed_normal_meta_station",
            "model_fore_train_task_query_fed_normal_meta_station",
        ]:
            self.assertIn(token, text)

    def test_fed_normal_meta_samples_all_stations_with_target_weight_floor(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            "def compute_fed_normal_meta_station_weights(target_station_id, candidate_station_ids):",
            "FED_NORMAL_META_SELF_FLOOR",
            "def run_fed_normal_meta_training():",
            "fed_normal_meta_client_states",
            "aggregate_fed_normal_meta_states",
            "client_station_order = [station_id] +",
            "sample_station_ids=[client_station_id]",
            "fed_normal_meta_tag = f\"fed_normal_meta_station{station_id}\"",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
