import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"
PREP_FILE = ROOT / "build_six_client_seasonal_protocol.py"
EVAL_FILE = ROOT / "generate_multi_station_results.py"


class SixClientSeasonalProtocolAstTest(unittest.TestCase):
    def test_preprocessing_script_exists_and_contains_protocol_manifest(self):
        text = PREP_FILE.read_text(encoding="utf-8")

        self.assertIn("SEASONAL_PROTOCOL_CLIENTS", text)
        self.assertIn("WT1", text)
        self.assertIn("WT6", text)
        self.assertIn("24jilin_059_processed_4classes.xlsx", text)
        self.assertIn("24jilin_060_processed_4classes.xlsx", text)
        self.assertIn("CAPACITY_BY_SOURCE_STATION", text)
        self.assertIn("'058': 50", text)
        self.assertIn("'059': 100", text)
        self.assertIn("'060': 300", text)

    def test_preprocessing_script_contains_dynamic_k_helpers(self):
        text = PREP_FILE.read_text(encoding="utf-8")

        self.assertIn("def compute_protocol_k_max(", text)
        self.assertIn("def choose_feasible_k_by_elbow(", text)
        self.assertIn("def compute_sampler_class_count(", text)
        self.assertIn("support_query_total", text)
        self.assertIn("max(2, math.ceil(", text)

    def test_training_script_exposes_seasonal_protocol_knobs(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("SEASONAL_PROTOCOL_ENABLED", text)
        self.assertIn("SEASONAL_PROTOCOL_METADATA_PATH", text)
        self.assertIn("META_SUPPORT_SHOTS", text)
        self.assertIn("META_QUERY_SHOTS", text)
        self.assertIn("META_SUPPORT_SHOTS = env_int(", text)
        self.assertIn("META_QUERY_SHOTS = env_int(", text)

    def test_training_script_uses_dynamic_meta_episode_sizes(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("random.sample(range(0, np.size(task['nwp'], axis=0)), META_SUPPORT_SHOTS + META_QUERY_SHOTS)", text)
        self.assertIn("index_shot[0:META_SUPPORT_SHOTS]", text)
        self.assertIn("index_shot[META_SUPPORT_SHOTS:META_SUPPORT_SHOTS + META_QUERY_SHOTS]", text)

    def test_training_script_consumes_protocol_specific_sampler_counts(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def resolve_station_sampler_tasks_per_epoch(", text)
        self.assertIn("seasonal_protocol_metadata", text)
        self.assertIn("sampler_task_count", text)

    def test_eval_script_supports_task_level_reporting(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        self.assertIn("SEASONAL_PROTOCOL_ENABLED", text)
        self.assertIn("def load_seasonal_protocol_metadata(", text)
        self.assertIn("def iter_valid_protocol_tasks(", text)
        self.assertIn("client_id", text)
        self.assertIn("extreme_class", text)


if __name__ == "__main__":
    unittest.main()
