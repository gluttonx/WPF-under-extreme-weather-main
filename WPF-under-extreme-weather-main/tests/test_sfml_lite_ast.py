import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"
MODEL_FILE = ROOT / "model.py"


class SFMLLiteAstTest(unittest.TestCase):
    def test_sfml_flags_and_helpers_exist(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("ENABLE_STRICT_FED_META_TRAIN =", text)
        self.assertIn("ENABLE_STRICT_FED_META_ONLY =", text)
        self.assertIn("STRICT_META_USE_SECOND_ORDER =", text)
        self.assertIn("STRICT_META_EPOCHS =", text)
        self.assertIn("STRICT_META_LOCAL_TASKS_PER_ROUND =", text)
        self.assertIn("STRICT_META_INNER_STEPS =", text)
        self.assertIn("STRICT_META_SUPPORT_SIZE =", text)
        self.assertIn("STRICT_META_QUERY_SIZE =", text)
        self.assertIn("STRICT_META_INNER_LR =", text)
        self.assertIn("STRICT_META_USE_CDRM =", text)
        self.assertIn("STRICT_META_SHARED_BLOCK_START =", text)
        self.assertIn("SHARED_ADAPTER_BOTTLENECK =", text)
        self.assertIn("STRICT_META_TASK_MODE =", text)
        self.assertIn("STRICT_META_SAVE_BEST_ONLY =", text)
        self.assertIn("STRICT_META_EARLY_STOP_PATIENCE =", text)
        self.assertIn('STRICT_META_TASK_MODE = os.getenv("STRICT_META_TASK_MODE", "cross_cluster").strip().lower()', text)
        self.assertIn('STRICT_META_SAVE_BEST_ONLY = env_flag("STRICT_META_SAVE_BEST_ONLY", False)', text)
        self.assertIn('STRICT_META_EARLY_STOP_PATIENCE = env_int("STRICT_META_EARLY_STOP_PATIENCE", 0)', text)
        self.assertIn("def meta_inner_adapt(", text)
        self.assertIn("def compute_meta_support_loss(", text)
        self.assertIn("def compute_meta_query_loss(", text)
        self.assertIn("def create_strict_meta_station_context(", text)
        self.assertIn("strict_meta_station_contexts", text)
        self.assertIn("def client_local_meta_round(", text)
        self.assertIn("def run_strict_federated_meta_training(", text)
        self.assertIn("update_best_meta_checkpoint(", text)
        self.assertIn("accumulate_weighted_state(", text)
        self.assertIn("finalize_weighted_state(", text)
        self.assertIn("compose_personalized_meta_init_state(", text)
        self.assertIn("select_meta_shared_state(", text)
        self.assertIn("should_stop_early(", text)
        self.assertIn("def is_strict_meta_shared_param(", text)
        self.assertIn("def extract_strict_meta_shared_state_dict(", text)
        self.assertIn("PERSONALIZED_META_ONLY_MODEL_TEMPLATE", text)
        self.assertIn("STRICT_META_ONLY_SHARED_BACKBONE_MODEL_PATH", text)

    def test_sfml_mainline_uses_local_tasks_and_shared_aggregation(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        start = text.index("def client_local_meta_round(")
        end = text.index("model_fore_pre = build_forecast_model(")
        strict_meta_block = text[start:end]

        self.assertIn("sample_local_meta_task(", strict_meta_block)
        self.assertIn("choose_local_meta_task_classes(", text)
        self.assertIn("server_aggregate_shared_updates(", strict_meta_block)
        self.assertIn("best_meta_record", strict_meta_block)
        self.assertIn('set_trainable_params_by_scope(local_model, "local")', strict_meta_block)
        self.assertIn('set_trainable_params_by_scope(local_model, "shared_meta")', strict_meta_block)
        self.assertIn("current_local_states[station_id]", strict_meta_block)
        self.assertIn("no_improve_rounds", strict_meta_block)
        self.assertIn("STRICT_META_EARLY_STOP_PATIENCE", strict_meta_block)
        self.assertIn("strict_meta_task_mode=", text)
        self.assertIn("strict_meta_station_contexts.get(station_id)", strict_meta_block)
        self.assertNotIn("gc.collect()", strict_meta_block)

    def test_model_defines_shared_adapter_between_tcn_and_head(self):
        model_text = MODEL_FILE.read_text(encoding="utf-8")
        train_text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("class SharedAdapter(", model_text)
        self.assertIn("self.shared_adapter = model.SharedAdapter(", train_text)
        self.assertIn("y = self.shared_adapter(y)", train_text)


if __name__ == "__main__":
    unittest.main()
