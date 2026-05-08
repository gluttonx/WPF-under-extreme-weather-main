import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class HighTempOnlyTrainingAstTest(unittest.TestCase):
    def test_training_reads_meta_tasks_from_env_and_metadata_driven_extreme_class_count(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            'META_TASKS_PER_EPOCH = int(os.getenv("META_TASKS_PER_EPOCH", "5"))',
            'extreme_class_names = protocol_metadata.get("extreme_class_names",',
            'num_extreme_classes = int(protocol_metadata.get("num_extreme_classes",',
        ]:
            self.assertIn(token, text)

    def test_training_replaces_four_class_assumptions_with_dynamic_loops(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            "few_shot_model_count = 0 if SKIP_EXTREME_ADAPTATION_STAGE else len(station_ids) * num_extreme_classes * (4 if TRAIN_META_ONLY_BASELINE else 3)",
            "for class_index in range(num_extreme_classes):",
            "for i_class in range(num_extreme_classes):",
            "print(f\"    极端天气: {num_extreme_classes}类\")",
        ]:
            self.assertIn(token, text)

    def test_training_declares_selective_fed_meta_controls_and_helpers(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            'ENABLE_SELECTIVE_FED_NORMAL_META = os.getenv("ENABLE_SELECTIVE_FED_NORMAL_META", "0") != "0"',
            'SELECTIVE_FED_META_PROXY_RATIO = float(os.getenv("SELECTIVE_FED_META_PROXY_RATIO", "0.2"))',
            'SELECTIVE_FED_META_SELF_FLOOR = float(os.getenv("SELECTIVE_FED_META_SELF_FLOOR", "0.5"))',
            'SELECTIVE_FED_META_GAIN_MARGIN = float(os.getenv("SELECTIVE_FED_META_GAIN_MARGIN", "0.0"))',
            'SELECTIVE_FED_META_GAIN_GAMMA = float(os.getenv("SELECTIVE_FED_META_GAIN_GAMMA", "1.0"))',
            "def split_normal_meta_train_proxy_windows(",
            "def evaluate_target_proxy_loss(",
            "def compute_selective_fed_meta_gain(",
            "def aggregate_selective_fed_normal_meta_states(",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
