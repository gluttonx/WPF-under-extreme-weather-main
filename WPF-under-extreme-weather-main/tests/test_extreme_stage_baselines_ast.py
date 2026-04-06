import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class ExtremeStageBaselinesAstTest(unittest.TestCase):
    def _extract_function_block(self, text, func_name):
        marker = f"def {func_name}("
        start = text.index(marker)
        next_def = text.find("\ndef ", start + len(marker))
        if next_def == -1:
            return text[start:]
        return text[start:next_def]

    def test_training_script_declares_yearly_extreme_baseline_names(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("LMT-new", text)
        self.assertIn("Extreme-FedAvg", text)
        self.assertIn("Proposed-A", text)
        self.assertIn("EXTREME_FEDAVG_PLACEHOLDER_MODEL_PATH", text)
        self.assertIn("PROPOSED_A_PLACEHOLDER_MODEL_PATH", text)

    def test_training_script_exposes_yearly_extreme_baseline_routing(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        routing_block = self._extract_function_block(text, "resolve_yearly_extreme_base_model_path")

        self.assertIn("def resolve_yearly_extreme_base_model_path(", text)
        self.assertIn("def get_yearly_extreme_baseline_specs(", text)
        self.assertIn("if YEARLY_PROTOCOL_ENABLED:", text)
        self.assertIn("baseline_specs = get_yearly_extreme_baseline_specs(station_id)", text)
        self.assertIn('if baseline_name == "LMT-new":', routing_block)
        self.assertIn('if baseline_name == "Extreme-FedAvg":', routing_block)
        self.assertIn('if baseline_name == "Proposed-A":', routing_block)
        self.assertNotIn("PRETRAIN_MODEL_PATH", routing_block)
        self.assertNotIn("get_proposed_meta_model_path", routing_block)
        self.assertEqual(routing_block.count("get_local_meta_model_path(station_id)"), 3)


if __name__ == "__main__":
    unittest.main()
