import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "generate_multi_station_results.py"


class YearlyExtremeResultsAstTest(unittest.TestCase):
    def test_eval_script_exposes_yearly_protocol_knobs_and_task_iterator(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        self.assertIn("YEARLY_PROTOCOL_ENABLED", text)
        self.assertIn("YEARLY_PROTOCOL_METADATA_PATH", text)
        self.assertIn("def load_yearly_protocol_metadata(", text)
        self.assertIn("def iter_valid_yearly_protocol_tasks(", text)
        self.assertIn("def run_yearly_protocol_evaluation(", text)
        self.assertIn("def build_yearly_task_level_results_df(", text)
        self.assertIn("def build_yearly_table_iv_results_df(", text)
        self.assertIn("def infer_yearly_training_durations_from_tensorboard(", text)

    def test_eval_script_supports_yearly_models_and_task_level_metrics(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        self.assertIn("LMT-new", text)
        self.assertIn("Extreme-FedAvg", text)
        self.assertIn("Proposed-A", text)
        self.assertIn('"Station"', text)
        self.assertIn("'station_id'", text)
        self.assertIn("'extreme_class'", text)
        self.assertIn("'Model'", text)
        self.assertIn("'nMAE_%'", text)
        self.assertIn("'nRMSE_%'", text)
        self.assertIn("'WD_%'", text)
        self.assertIn("'R_p<0.05_%'", text)
        self.assertIn("multi_station_performance_task_level.csv", text)
        self.assertIn("HighWind_E_M_%", text)
        self.assertIn("HighTemperature_E_R_%", text)
        self.assertIn("ColdWave_WD", text)
        self.assertIn("Training_duration_s", text)

    def test_yearly_table_iv_main_output_tracks_station_dimension(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        self.assertIn('row = {"Station": station_id, "Model": model_name}', text)
        self.assertIn('wide_df = wide_df[YEARLY_TABLE_IV_COLUMNS]', text)


if __name__ == "__main__":
    unittest.main()
