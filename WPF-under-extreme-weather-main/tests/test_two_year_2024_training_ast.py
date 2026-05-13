import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class TwoYear2024TrainingAstTest(unittest.TestCase):
    def test_training_reads_meta_support_and_query_shots_from_env(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            'META_SUPPORT_SHOTS = int(os.getenv("META_SUPPORT_SHOTS", "10"))',
            'META_QUERY_SHOTS = int(os.getenv("META_QUERY_SHOTS", "10"))',
        ]:
            self.assertIn(token, text)

    def test_sample_meta_batch_uses_configurable_shot_counts(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            "meta_total_shots = META_SUPPORT_SHOTS + META_QUERY_SHOTS",
            "index_shot = random.sample(range(0, np.size(train_input_dataset[i_task], axis=0)), meta_total_shots)",
            "train_input_support_ = train_input_dataset[i_task][index_shot[0:META_SUPPORT_SHOTS], :, :]",
            "train_input_query_ = train_input_dataset[i_task][index_shot[META_SUPPORT_SHOTS:meta_total_shots], :, :]",
        ]:
            self.assertIn(token, text)

        for legacy_token in [
            "random.sample(range(0, np.size(train_input_dataset[i_task], axis=0)), 20)",
            "index_shot[0:10]",
            "index_shot[10:20]",
        ]:
            self.assertNotIn(legacy_token, text)

    def test_training_no_longer_hardcodes_ten_clusters_in_runtime_message(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertNotIn("聚类类别: 10类", text)
        self.assertIn("total_station_classes", text)


if __name__ == "__main__":
    unittest.main()
