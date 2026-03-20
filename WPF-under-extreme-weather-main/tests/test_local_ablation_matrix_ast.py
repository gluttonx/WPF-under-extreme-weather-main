import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"
EVAL_FILE = ROOT / "generate_multi_station_results.py"


class LocalAblationMatrixAstTest(unittest.TestCase):
    def test_training_script_contains_local_pretrain_and_transfer_branches(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def get_local_pretrain_model_path(", text)
        self.assertIn("def get_local_meta_model_path(", text)
        self.assertIn("def run_local_pretrain(", text)
        self.assertIn("local_pretrain_state_dicts = {}", text)
        self.assertIn("query_model_path=get_local_meta_model_path(station_id)", text)
        self.assertIn('local_meta_model_name = f"./model_fore_station{station_id}_extreme{i_class}_local_meta.pth"', text)
        self.assertIn('transfer_model_name = f"./model_fore_station{station_id}_extreme{i_class}_transfer_only.pth"', text)

    def test_eval_script_contains_five_row_main_table(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        self.assertIn("Local_Meta_Transfer", text)
        self.assertIn("Transfer_Learning", text)
        self.assertIn("Local_PreTraining", text)
        self.assertIn("model_fore_pre_station{station_id}_local.pth", text)
        self.assertIn("model_fore_station{station_id}_extreme{class_idx}_local_meta.pth", text)
        self.assertIn("model_fore_station{station_id}_extreme{class_idx}_transfer_only.pth", text)


if __name__ == "__main__":
    unittest.main()
