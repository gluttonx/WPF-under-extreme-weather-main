import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "generate_multi_station_results.py"


class EvalTwoHourSixPointProtocolAstTest(unittest.TestCase):
    def test_eval_reads_protocol_config(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

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

    def test_eval_resolves_station_mats_from_protocol_dir(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        self.assertIn("def resolve_station_mat_path(station_id):", text)
        self.assertIn("scio.loadmat(resolve_station_mat_path(station_id))", text)
        self.assertNotIn("scio.loadmat(dataFile)", text)

    def test_eval_adds_protocol_columns(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        for column in [
            "'Protocol': PROTOCOL_NAME",
            "'Sample_Interval_Hours': SAMPLE_INTERVAL_HOURS",
            "'Window_Points': LEN_REALP",
            "'Window_Span_Hours': WINDOW_SPAN_HOURS",
        ]:
            self.assertIn(column, text)

    def test_eval_does_not_pin_12_point_protocol(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        self.assertIn("len_realp = LEN_REALP", text)
        self.assertNotIn("len_realp = 12", text)

    def test_eval_reads_station_ids_from_metadata_or_env_subset(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        self.assertIn("def resolve_station_ids():", text)
        self.assertIn("EVAL_STATION_IDS", text)
        self.assertIn("protocol_metadata.get(\"stations\", [])", text)
        self.assertIn("return [station_id.strip() for station_id in eval_station_ids.split(\",\") if station_id.strip()]", text)

    def test_eval_uses_dynamic_station_order_and_output_paths(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        self.assertIn("TASK_RESULTS_OUTPUT_PATH", text)
        self.assertIn("RESULTS_OUTPUT_PATH", text)
        self.assertIn("station_order = station_ids + ['Overall_Average']", text)
        self.assertNotIn("station_order = ['58', '59', '60', 'Overall_Average']", text)

    def test_eval_reads_models_from_model_output_dir(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        for token in [
            "ARTIFACT_DIR",
            "MODEL_OUTPUT_DIR",
            "LOGS_TRAIN_DIR",
            "resolve_model_path",
            "os.path.join(MODEL_OUTPUT_DIR",
            "os.path.join(LOGS_TRAIN_DIR, 'loss2'",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
