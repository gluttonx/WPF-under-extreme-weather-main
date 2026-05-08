import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_FILE = ROOT / "run_three_station_yearly_protocol.py"


class HighWindSpringNoftLauncherAstTest(unittest.TestCase):
    def test_launcher_declares_high_wind_protocol_constants_and_flag(self):
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        for token in [
            "HIGH_WIND_SPRING_NOFT_PROTOCOL_NAME",
            '"four_client_2h_6point_high_wind_spring_noft_protocol"',
            "HIGH_WIND_SPRING_NOFT_PROTOCOL_DATA_DIR",
            '"high_wind_spring_noft_four_client"',
            "HIGH_WIND_SPRING_NOFT_ARTIFACT_DIR",
            "--high-wind-spring-noft",
            "high_wind_spring_noft",
            "HIGH_WIND_SPRING_NOFT_PROTOCOL",
        ]:
            self.assertIn(token, text)

    def test_launcher_wires_noft_meta_defaults(self):
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        for token in [
            '"HIGH_WIND_SPRING_NOFT_PROTOCOL": "1"',
            '"EXTREME_SUPPORT_WINDOW_CAP": "0"',
            '"EVAL_MODEL_SET": env.get("EVAL_MODEL_SET") or "fed-meta-noft"',
            '"SKIP_EXTREME_ADAPTATION_STAGE": "1"',
            '"HIGH_WIND_NORMAL_TRAIN_END": env.get("HIGH_WIND_NORMAL_TRAIN_END", "2022-04-29T16:00:00+00:00")',
            '"META_TASKS_PER_EPOCH": "2"',
            '"META_SUPPORT_SHOTS": "4"',
            '"META_QUERY_SHOTS": "4"',
            '"FEW_SHOT_EPOCHS": "0"',
            '"EXTREME_TARGET_REFINEMENT_EPOCHS": "0"',
            'if env.get("ENABLE_TARGET_AWARE_SELECTIVE_FED_LOCAL_FT", "0") != "0":',
            '"FEW_SHOT_EPOCHS": os.environ.get("FEW_SHOT_EPOCHS", "5")',
            '"FEW_SHOT_LR": os.environ.get("FEW_SHOT_LR", "5e-5")',
            '"SKIP_EXTREME_ADAPTATION_STAGE": "0"',
        ]:
            self.assertIn(token, text)

    def test_launcher_exposes_high_wind_aware_loss_env(self):
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        for token in [
            "HWA_PRETRAIN_LOSS",
            "HWA_META_LOSS",
            "HWA_SELECTIVE_PROXY_LOSS",
            "HWA_WIND_FEATURE_INDEX",
            "HWA_WIND_THRESHOLD",
            "HWA_WIND_RAMP_END",
            "HWA_HIGH_WIND_WEIGHT",
            "HWA_PRETRAIN_WINDOWED",
            "ENABLE_TARGET_AWARE_META_NOFT",
            "SKIP_TARGET_AWARE_PRETRAIN",
            "SKIP_TARGET_AWARE_META",
            "ENABLE_TARGET_AWARE_SELECTIVE_FED_META",
            "SKIP_TARGET_AWARE_SELECTIVE_FED_META",
            "ENABLE_TARGET_AWARE_SELECTIVE_FED_LOCAL_FT",
            "ENABLE_TARGET_AWARE_SELECTIVE_FED_BIAS_CAL",
            "SKIP_LEGACY_EXTREME_ADAPTATION",
            "TARGET_AWARE_PRETRAIN_HWA_LOSS",
            "TARGET_AWARE_META_CDRM_WEIGHT",
            "TARGET_AWARE_META_WIND_MEAN_THRESHOLD",
            "TARGET_AWARE_META_WIND_MAX_THRESHOLD",
            "TARGET_AWARE_META_MIN_EXTREME_POINTS",
            "TARGET_AWARE_SELECTIVE_FED_TOP_K",
            "TARGET_AWARE_SELECTIVE_FED_SELF_FLOOR",
            "TARGET_AWARE_SELECTIVE_FED_SOURCE_ALPHA_CAP",
            "TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_MAX_WINDOWS",
            "TARGET_AWARE_SELECTIVE_FED_BIAS_CAL_MIN",
            "TARGET_AWARE_SELECTIVE_FED_BIAS_CAL_MAX",
            "TARGET_AWARE_BASE_MODEL_OUTPUT_DIR",
            "TARGET_AWARE_SELECTIVE_FED_BASE_MODEL_OUTPUT_DIR",
            "TARGET_AWARE_SELECTIVE_FED_META_INIT_MODEL_DIR",
            "TARGET_AWARE_SELECTIVE_FED_META_EPOCH_OFFSET",
            "TRAIN_PRETRAIN_ONLY",
            "RUNTIME_ENV_OVERRIDE_KEYS",
            'if key in os.environ:',
            "GLOBAL_SEED",
            '"HWA_PRETRAIN_LOSS": env.get("HWA_PRETRAIN_LOSS", "0")',
            '"HWA_META_LOSS": env.get("HWA_META_LOSS", "0")',
            '"HWA_SELECTIVE_PROXY_LOSS": env.get("HWA_SELECTIVE_PROXY_LOSS", "0")',
            '"ENABLE_TARGET_AWARE_META_NOFT": env.get("ENABLE_TARGET_AWARE_META_NOFT", "0")',
            '"ENABLE_TARGET_AWARE_SELECTIVE_FED_META": env.get("ENABLE_TARGET_AWARE_SELECTIVE_FED_META", "0")',
            '"ENABLE_TARGET_AWARE_SELECTIVE_FED_LOCAL_FT": env.get("ENABLE_TARGET_AWARE_SELECTIVE_FED_LOCAL_FT", "0")',
            '"ENABLE_TARGET_AWARE_SELECTIVE_FED_BIAS_CAL": env.get("ENABLE_TARGET_AWARE_SELECTIVE_FED_BIAS_CAL", "0")',
            '"SKIP_LEGACY_EXTREME_ADAPTATION": env.get("SKIP_LEGACY_EXTREME_ADAPTATION", "0")',
            '"TARGET_AWARE_SELECTIVE_FED_META_INIT_MODEL_DIR": env.get("TARGET_AWARE_SELECTIVE_FED_META_INIT_MODEL_DIR", "")',
            '"TARGET_AWARE_SELECTIVE_FED_META_EPOCH_OFFSET": env.get("TARGET_AWARE_SELECTIVE_FED_META_EPOCH_OFFSET", "0")',
            '"TRAIN_PRETRAIN_ONLY": env.get("TRAIN_PRETRAIN_ONLY", "0")',
            '"TARGET_AWARE_PRETRAIN_HWA_LOSS": env.get("TARGET_AWARE_PRETRAIN_HWA_LOSS", "1")',
            '"TARGET_AWARE_SELECTIVE_FED_TOP_K": env.get("TARGET_AWARE_SELECTIVE_FED_TOP_K", "1")',
            '"GLOBAL_SEED": env.get("GLOBAL_SEED", "1029")',
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
