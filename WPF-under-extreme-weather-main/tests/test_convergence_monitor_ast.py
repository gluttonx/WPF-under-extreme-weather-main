import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class ConvergenceMonitorAstTest(unittest.TestCase):
    def test_training_declares_convergence_monitor_helpers(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertRegex(
            text,
            r'CONVERGENCE_REPORT_PATH\s*=\s*os\.getenv\(\s*"CONVERGENCE_REPORT_PATH"',
        )
        self.assertIn("def initialize_convergence_record(", text)
        self.assertIn("def update_convergence_record(", text)
        self.assertIn("def register_convergence_record(", text)
        self.assertIn("def export_convergence_report(", text)

    def test_training_wires_convergence_monitor_into_three_stages(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertIn('stage_type="local_pretrain"', text)
        self.assertIn('stage_type="local_meta"', text)
        self.assertIn('stage_type="few_shot"', text)
        self.assertIn("register_convergence_record(pretrain_convergence_record)", text)
        self.assertIn("register_convergence_record(convergence_record)", text)
        self.assertIn("export_convergence_report(", text)


if __name__ == "__main__":
    unittest.main()
