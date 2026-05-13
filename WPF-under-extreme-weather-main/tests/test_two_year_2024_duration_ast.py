import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"
EVAL_FILE = ROOT / "generate_multi_station_results.py"


class TwoYear2024DurationAstTest(unittest.TestCase):
    def test_training_convergence_records_capture_elapsed_seconds(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            '"started_at_unix": float(time.time())',
            'started_at_unix = finalized_record.pop("started_at_unix", None)',
            'finalized_record["elapsed_seconds"] = max(0.0, float(time.time()) - float(started_at_unix))',
        ]:
            self.assertIn(token, text)

    def test_eval_prefers_convergence_report_durations_before_tensorboard(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        for token in [
            'CONVERGENCE_REPORT_PATH = os.getenv(',
            'def infer_training_durations_from_convergence_report():',
            'duration_map = infer_training_durations_from_convergence_report()',
            'if all(np.isnan(v) for v in duration_map.values()):',
            'duration_map = infer_training_durations_from_tensorboard()',
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
