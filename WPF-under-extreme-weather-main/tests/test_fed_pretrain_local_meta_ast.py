import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class FedPretrainLocalMetaAstTest(unittest.TestCase):
    def test_demo_model_training_uses_fedavg_pretrain_and_station_local_meta(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def client_local_pretrain_update(", text)
        self.assertIn("def server_aggregate_client_states(", text)
        self.assertIn("def sample_station_meta_batch(", text)
        self.assertIn("def run_local_meta_training(", text)

        self.assertIn("for station_id in station_ids:", text)
        self.assertIn("run_local_meta_training(", text)
        self.assertIn("get_proposed_meta_model_path(station_id)", text)

        self.assertNotIn("def sample_meta_batch(", text)
        self.assertNotIn("def run_meta_training(", text)
        self.assertNotIn("loss_en_avg = loss_en / task_num", text)


if __name__ == "__main__":
    unittest.main()
