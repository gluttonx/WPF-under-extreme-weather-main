import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class StrictDecoupledMetaAstTest(unittest.TestCase):
    def test_decoupled_helper_exists(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn('def set_trainable_params_by_scope(', text)
        self.assertIn('def is_strict_meta_shared_param(', text)
        self.assertIn('def is_shared_adapter_param(', text)
        self.assertIn('def is_finetune_param(', text)

    def test_client_local_meta_round_decouples_support_and_query_scopes(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        start = text.index("def client_local_meta_round(")
        end = text.index("def run_strict_federated_meta_training(")
        block = text[start:end]

        self.assertIn('set_trainable_params_by_scope(local_model, "local")', block)
        self.assertIn('set_trainable_params_by_scope(local_model, "shared_meta")', block)

    def test_final_strict_meta_checkpoint_uses_current_local_state(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        start = text.index("def run_strict_federated_meta_training(")
        end = text.index("model_fore_pre = build_forecast_model(")
        block = text[start:end]

        self.assertIn("current_local_states[station_id]", block)
        self.assertNotIn("clone_state_dict(initial_local_states[station_id]", block)

    def test_strict_meta_shared_scope_targets_adapter_only(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn('return name.startswith("shared_adapter.")', text)
        self.assertIn('return is_shared_adapter_param(name)', text)

    def test_downstream_finetune_scope_includes_local_and_shared_adapter(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn('scope not in {"shared", "shared_meta", "local", "finetune", "all"}', text)
        self.assertIn('return is_local_param(name) or is_shared_adapter_param(name)', text)
        self.assertIn('set_trainable_params_by_scope(model_fore_test_task_support, "finetune")', text)


if __name__ == "__main__":
    unittest.main()
