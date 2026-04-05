import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class ConvergenceDetectionAstTest(unittest.TestCase):
    def test_training_script_exposes_convergence_runtime_knobs(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn('ENABLE_CONVERGENCE_MONITOR = env_flag("ENABLE_CONVERGENCE_MONITOR", True)', text)
        self.assertIn('CONVERGENCE_REPORT_PATH = env_str("CONVERGENCE_REPORT_PATH", "training_convergence_report.json")', text)
        self.assertIn('CONVERGENCE_MIN_DELTA = env_float("CONVERGENCE_MIN_DELTA", 1e-4)', text)
        self.assertIn('CONVERGENCE_MIN_EPOCHS = env_int("CONVERGENCE_MIN_EPOCHS", 5)', text)

    def test_training_script_contains_convergence_helpers_and_report_fields(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def initialize_convergence_record(", text)
        self.assertIn("def update_convergence_record(", text)
        self.assertIn("def finalize_convergence_record(", text)
        self.assertIn("def export_convergence_report(", text)
        self.assertIn("convergence_epoch", text)
        self.assertIn("best_epoch", text)
        self.assertIn("best_loss", text)
        self.assertIn("final_loss", text)
        self.assertIn("converged", text)

    def test_training_script_wires_convergence_into_all_training_phases(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn('stage_type="federated_pretrain"', text)
        self.assertIn('stage_type="local_pretrain"', text)
        self.assertIn('stage_type="local_meta"', text)
        self.assertIn('stage_type="few_shot"', text)
        self.assertIn('export_convergence_report(', text)


if __name__ == "__main__":
    unittest.main()
