import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "generate_multi_station_results.py"


class FedMetaNoftEvalAstTest(unittest.TestCase):
    def test_eval_declares_fed_meta_noft_model_set(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        for token in [
            'VANILLA_FED_NORMAL_META_NOFT_MODEL_NAME = \'Vanilla Fed-Normal-Meta-NoFT\'',
            'SELECTIVE_FED_NORMAL_META_NOFT_MODEL_NAME = \'Selective Fed-Normal-Meta-NoFT\'',
            'TARGET_AWARE_PRETRAIN_MODEL_NAME = \'Target-Aware Pretrain\'',
            'TARGET_AWARE_META_NOFT_MODEL_NAME = \'Target-Aware Meta-NoFT\'',
            'TARGET_AWARE_SELECTIVE_FED_META_NOFT_MODEL_NAME = \'Target-Aware Selective Fed-Meta-NoFT\'',
            'TARGET_AWARE_SELECTIVE_FED_META_BIAS_CAL_MODEL_NAME = \'Target-Aware Selective Fed-Meta-NoFT + BiasCal\'',
            'TARGET_AWARE_SELECTIVE_FED_META_LOCAL_FT_MODEL_NAME = \'Target-Aware Selective Fed-Meta + Local FT\'',
            'if EVAL_MODEL_SET == "fed-meta-noft":',
            "ENABLE_TARGET_AWARE_META_NOFT",
            "ENABLE_TARGET_AWARE_SELECTIVE_FED_META",
            "ENABLE_TARGET_AWARE_SELECTIVE_FED_LOCAL_FT",
            "ENABLE_TARGET_AWARE_SELECTIVE_FED_BIAS_CAL",
            "TARGET_AWARE_SELECTIVE_FED_BIAS_CAL_MAX",
            "TARGET_AWARE_BASE_MODEL_OUTPUT_DIR",
            "TARGET_AWARE_SELECTIVE_FED_BASE_MODEL_OUTPUT_DIR",
            "build_bias_calibration_proxy_windows",
            "compute_target_aware_selective_fed_bias_calibration",
            "bias_calibration_by_station",
            "get_target_aware_pretrain_model_path",
            "get_target_aware_meta_model_path",
            "get_target_aware_selective_fed_meta_model_path",
            "get_target_aware_selective_fed_local_ft_model_path",
            "get_fed_normal_meta_model_path",
            "model_fore_train_task_query_target_aware_selective_fed_meta_station",
            "model_fore_station{station_id}_extreme{class_idx}_target_aware_selective_fed_local_ft.pth",
            "model_fore_train_task_query_fed_normal_meta_station",
            "active_fed_meta_noft_model_name",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
