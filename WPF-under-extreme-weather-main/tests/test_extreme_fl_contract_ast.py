import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class ExtremeFLContractAstTest(unittest.TestCase):
    def test_training_declares_three_model_extreme_paths(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertIn("def get_extreme_fedavg_model_path(station_id, class_idx):", text)
        self.assertIn("def get_proposed_a_model_path(station_id, class_idx):", text)
        self.assertIn("model_label=\"Extreme-FedAvg:target_refine\"", text)
        self.assertIn("model_label=\"Proposed-A:target_refine\"", text)

    def test_training_declares_effective_window_and_weighting_helpers(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")
        self.assertIn("def split_extreme_adapt_val(", text)
        self.assertIn("def apply_source_quality_gate(", text)
        self.assertIn("def compute_target_conditioned_usefulness(", text)
        self.assertIn("def passes_source_target_gate(", text)
        self.assertIn("def aggregate_extreme_updates_uniform(", text)
        self.assertIn("def aggregate_extreme_updates_weighted(", text)
        self.assertIn("EXTREME_WEIGHT_BETA_SELF", text)
        self.assertIn("EXTREME_SOURCE_HARD_GATE", text)
        self.assertIn("EXTREME_SOURCE_MIN_TARGET_GAIN", text)

    def test_proposed_a_can_use_fed_normal_meta_base_without_changing_other_models(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            "def get_local_extreme_base_model_path(station_id):",
            "def get_proposed_a_base_model_path(station_id):",
            "ENABLE_FED_NORMAL_META_PROPOSED or ENABLE_SELECTIVE_FED_NORMAL_META",
            "return get_fed_normal_meta_model_path(station_id)",
            "local_shared_init_state = torch.load(",
            "proposed_shared_init_state = torch.load(",
            "base_state_dict=proposed_shared_init_state",
            "shared_init_state_dict=proposed_shared_init_state",
            "base_state_dict=local_shared_init_state",
        ]:
            self.assertIn(token, text)

    def test_extreme_adaptation_supports_proximal_anchor_regularization(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            "EXTREME_ANCHOR_REG_LAMBDA",
            "def compute_anchor_regularization(",
            "anchor_state_dict=base_state_dict",
            "loss_total = loss_mse + anchor_loss",
            '"extreme_anchor_reg_lambda": EXTREME_ANCHOR_REG_LAMBDA',
        ]:
            self.assertIn(token, text)

    def test_extreme_adaptation_supports_class_adaptive_target_caps(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            "EXTREME_TARGET_ADAPT_MAX_WINDOWS_BY_CLASS",
            "def resolve_extreme_target_adapt_max_windows(",
            "apply_target_adapt_kshot_limit(target_split_payload, class_idx=i_class)",
            '"extreme_target_adapt_max_windows_by_class": EXTREME_TARGET_ADAPT_MAX_WINDOWS_BY_CLASS',
        ]:
            self.assertIn(token, text)

    def test_proposed_a_uses_source_target_gain_weighting(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            "EXTREME_SOURCE_GAIN_WEIGHT_ETA",
            "def compute_source_target_gain_weight(",
            '"target_gain": source_target_gain',
            "target_gain_weight = compute_source_target_gain_weight(payload.get(\"target_gain\", 0.0), class_idx=class_idx)",
            '"extreme_source_gain_weight_eta": EXTREME_SOURCE_GAIN_WEIGHT_ETA',
        ]:
            self.assertIn(token, text)

    def test_proposed_a_supports_target_validation_fallback(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            "EXTREME_PROPOSED_VAL_FALLBACK",
            "EXTREME_PROPOSED_VAL_FALLBACK_MARGIN",
            "def select_proposed_final_state_by_target_validation(",
            "selected_proposed_state, proposed_selection_info = select_proposed_final_state_by_target_validation(",
            "save_state_dict(selected_proposed_state, proposed_model_name)",
            '"extreme_proposed_val_fallback": EXTREME_PROPOSED_VAL_FALLBACK',
            '"extreme_proposed_val_fallback_margin": EXTREME_PROPOSED_VAL_FALLBACK_MARGIN',
        ]:
            self.assertIn(token, text)

    def test_extreme_adaptation_supports_classwise_transfer_overrides(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            'EXTREME_WEIGHT_BETA_SELF_BY_CLASS = os.getenv("EXTREME_WEIGHT_BETA_SELF_BY_CLASS", "")',
            'EXTREME_SOURCE_HARD_GATE_BY_CLASS = os.getenv("EXTREME_SOURCE_HARD_GATE_BY_CLASS", "")',
            'EXTREME_SOURCE_MIN_TARGET_GAIN_BY_CLASS = os.getenv("EXTREME_SOURCE_MIN_TARGET_GAIN_BY_CLASS", "")',
            'EXTREME_SOURCE_GAIN_WEIGHT_ETA_BY_CLASS = os.getenv("EXTREME_SOURCE_GAIN_WEIGHT_ETA_BY_CLASS", "")',
            'EXTREME_PROPOSED_VAL_FALLBACK_BY_CLASS = os.getenv("EXTREME_PROPOSED_VAL_FALLBACK_BY_CLASS", "")',
            'EXTREME_PROPOSED_VAL_FALLBACK_MARGIN_BY_CLASS = os.getenv("EXTREME_PROPOSED_VAL_FALLBACK_MARGIN_BY_CLASS", "")',
            'EXTREME_SOURCE_TOP_K_BY_CLASS = os.getenv("EXTREME_SOURCE_TOP_K_BY_CLASS", "")',
            'EXTREME_FORCE_LOCAL_FALLBACK_BY_CLASS = os.getenv("EXTREME_FORCE_LOCAL_FALLBACK_BY_CLASS", "")',
            "def resolve_extreme_weight_beta_self(",
            "def resolve_extreme_source_hard_gate(",
            "def resolve_extreme_source_min_target_gain(",
            "def resolve_extreme_source_gain_weight_eta(",
            "def resolve_extreme_proposed_val_fallback(",
            "def resolve_extreme_proposed_val_fallback_margin(",
            "def resolve_extreme_source_top_k(",
            "def resolve_extreme_force_local_fallback(",
            'payload.get("target_gain", 0.0)',
            "source_update_payloads = source_update_payloads[:top_k]",
            'if resolve_extreme_force_local_fallback(i_class):',
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
