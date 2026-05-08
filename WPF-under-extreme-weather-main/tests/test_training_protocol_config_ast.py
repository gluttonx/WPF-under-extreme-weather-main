import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class TrainingProtocolConfigAstTest(unittest.TestCase):
    def test_training_script_reads_protocol_config(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            "PROTOCOL_DATA_DIR",
            "PROTOCOL_METADATA_PATH",
            "PROTOCOL_NAME",
            "SAMPLE_INTERVAL_HOURS",
            "DOWNSAMPLE_OFFSET",
            "LEN_REALP",
            "POINTS_PER_DAY",
            "WINDOW_SPAN_HOURS",
            "os.getenv",
        ]:
            self.assertIn(token, text)

    def test_training_script_resolves_station_mats_from_protocol_dir(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def resolve_station_mat_path(station_id):", text)
        self.assertIn("PROTOCOL_DATA_DIR", text)
        self.assertIn("scio.loadmat(resolve_station_mat_path(station_id))", text)
        self.assertNotIn("scio.loadmat(dataFile)", text)

    def test_training_uses_protocol_test_payload_when_available(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("'p_test_00': wf_1.get('p_test')", text)
        self.assertIn("'nwp_test_00': wf_1.get('nwp_test')", text)
        self.assertIn("station_data[station_id]['p_test_00'] is not None", text)
        self.assertIn("test_target_p_st = station_data[station_id]['p_test_00']", text)
        self.assertIn("test_nwp_st = station_data[station_id]['nwp_test_00']", text)

    def test_training_script_does_not_pin_12_point_protocol(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("len_realp = LEN_REALP", text)
        self.assertIn("d = POINTS_PER_DAY", text)
        self.assertNotIn("len_realp=12", text)
        self.assertNotIn("d=24", text)

    def test_training_report_exports_protocol_fields(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for field in [
            '"protocol_name": PROTOCOL_NAME',
            '"protocol_data_dir": PROTOCOL_DATA_DIR',
            '"protocol_metadata_path": PROTOCOL_METADATA_PATH',
            '"sample_interval_hours": SAMPLE_INTERVAL_HOURS',
            '"downsample_offset": DOWNSAMPLE_OFFSET',
            '"len_realp": LEN_REALP',
            '"points_per_day": POINTS_PER_DAY',
        ]:
            self.assertIn(field, text)

    def test_training_reads_station_ids_from_protocol_metadata(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def resolve_station_ids():", text)
        self.assertIn("yearly_protocol_metadata.get(\"stations\", [])", text)
        self.assertIn("return [str(station[\"station_id\"]) for station in metadata_stations]", text)

    def test_training_source_loop_only_skips_exact_self(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("for source_station_id in station_ids:", text)
        self.assertIn("if source_station_id == target_station_id:", text)
        self.assertNotIn("physical_station", text)
        self.assertNotIn("same_physical", text)

    def test_training_artifacts_are_path_isolated(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            "ARTIFACT_DIR",
            "MODEL_OUTPUT_DIR",
            "BASE_MODEL_OUTPUT_DIR",
            "LOGS_TRAIN_DIR",
            "resolve_artifact_path",
            "resolve_model_path",
            "resolve_base_model_path",
            "ALL_STATIONS_TEST_RESULTS_PATH",
            "os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)",
            "SummaryWriter(os.path.join(LOGS_TRAIN_DIR, \"loss1\"))",
            "SummaryWriter(os.path.join(LOGS_TRAIN_DIR, \"loss2\"))",
        ]:
            self.assertIn(token, text)

        self.assertNotIn("SummaryWriter(\"./logs_train/loss1\")", text)
        self.assertNotIn("scio.savemat('all_stations_test_results.mat'", text)

    def test_training_declares_fed_normal_meta_proposed_config(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            'ENABLE_FED_NORMAL_META_PROPOSED = os.getenv("ENABLE_FED_NORMAL_META_PROPOSED", "0") != "0"',
            'FED_NORMAL_META_SELF_FLOOR = float(os.getenv("FED_NORMAL_META_SELF_FLOOR", "0.3"))',
            'SKIP_FED_NORMAL_META = os.getenv("SKIP_FED_NORMAL_META", "0") != "0"',
            'FED_NORMAL_META_RESTORE_BEST = os.getenv("FED_NORMAL_META_RESTORE_BEST", "0") != "0"',
            'FED_NORMAL_META_SAVE_BEST = os.getenv("FED_NORMAL_META_SAVE_BEST", "0") != "0"',
            'FED_NORMAL_META_USE_BEST = os.getenv("FED_NORMAL_META_USE_BEST", "0") != "0"',
            "def get_fed_normal_meta_support_model_path(station_id):",
            "def get_fed_normal_meta_model_path(station_id):",
            "def get_fed_normal_meta_best_support_model_path(station_id):",
            "def get_fed_normal_meta_best_model_path(station_id):",
            "model_fore_train_task_support_fed_normal_meta_station",
            "model_fore_train_task_query_fed_normal_meta_station",
            "model_fore_train_task_support_fed_normal_meta_best_station",
            "model_fore_train_task_query_fed_normal_meta_best_station",
        ]:
            self.assertIn(token, text)

    def test_training_can_read_base_checkpoints_from_separate_dir(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            'BASE_MODEL_OUTPUT_DIR = os.getenv("BASE_MODEL_OUTPUT_DIR", MODEL_OUTPUT_DIR)',
            "def resolve_base_model_path(filename):",
            "def resolve_fed_normal_meta_model_path(filename):",
            "BASE_MODEL_OUTPUT_DIR != MODEL_OUTPUT_DIR",
            "can_reuse_base_dir = SKIP_LOCAL_PRETRAIN and SKIP_LOCAL_META",
            "get_local_pretrain_model_path(station_id)",
            "return resolve_base_model_path(f\"model_fore_pre_station{station_id}_local.pth\")",
            "return resolve_base_model_path(f\"model_fore_train_task_query_local_meta_station{station_id}.pth\")",
            "return resolve_fed_normal_meta_model_path(f\"model_fore_train_task_query_fed_normal_meta_best_station{station_id}.pth\")",
            "if SKIP_FED_NORMAL_META:",
            "return resolve_base_model_path(filename)",
            "return resolve_model_path(filename)",
            "return resolve_model_path(f\"model_fore_station{station_id}_extreme{class_idx}_proposed_a.pth\")",
            '"base_model_output_dir": BASE_MODEL_OUTPUT_DIR',
        ]:
            self.assertIn(token, text)

    def test_training_avoids_one_step_lwp_nan_and_bad_checkpoints(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            "if nwp_array.shape[0] == 1:",
            '"adapt_nwp": nwp_array',
            '"adapt_power": power_array',
            '"val_nwp": empty_nwp',
            '"val_power": empty_power',
            "last_finite_state = copy.deepcopy(base_state_dict)",
            "if not torch.isfinite(loss_total):",
            "non_finite_loss",
            "has_non_finite_state_dict",
            "model_fore_test_task_support.load_state_dict(copy.deepcopy(last_finite_state))",
        ]:
            self.assertIn(token, text)

    def test_fed_normal_meta_samples_all_stations_with_target_weight_floor(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            "def compute_fed_normal_meta_station_weights(target_station_id, candidate_station_ids):",
            "FED_NORMAL_META_SELF_FLOOR",
            "def run_fed_normal_meta_training():",
            "fed_normal_meta_client_states",
            "aggregate_fed_normal_meta_states",
            "client_station_order = [station_id] +",
            "fed_normal_meta_best_state",
            "if FED_NORMAL_META_SAVE_BEST and fed_normal_meta_best_state is not None:",
            "保存Fed-Normal-Meta best checkpoint",
            "sample_station_ids=[client_station_id]",
            "fed_normal_meta_tag = f\"fed_normal_meta_station{station_id}\"",
        ]:
            self.assertIn(token, text)

    def test_training_declares_selective_fed_meta_runtime_controls(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            'ENABLE_SELECTIVE_FED_NORMAL_META = os.getenv("ENABLE_SELECTIVE_FED_NORMAL_META", "0") != "0"',
            'SELECTIVE_FED_META_PROXY_RATIO = float(os.getenv("SELECTIVE_FED_META_PROXY_RATIO", "0.2"))',
            'SELECTIVE_FED_META_SELF_FLOOR = float(os.getenv("SELECTIVE_FED_META_SELF_FLOOR", "0.5"))',
            'SELECTIVE_FED_META_GAIN_MARGIN = float(os.getenv("SELECTIVE_FED_META_GAIN_MARGIN", "0.0"))',
            'SELECTIVE_FED_META_GAIN_GAMMA = float(os.getenv("SELECTIVE_FED_META_GAIN_GAMMA", "1.0"))',
            "if ENABLE_SELECTIVE_FED_NORMAL_META:",
        ]:
            self.assertIn(token, text)

    def test_training_can_skip_extreme_adaptation_for_noft_protocol(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            'SKIP_EXTREME_ADAPTATION_STAGE = os.getenv("SKIP_EXTREME_ADAPTATION_STAGE", "0") != "0"',
            "NoFT协议：跳过test_task_support/Few-shot适应",
            "for station_id in ([] if SKIP_EXTREME_ADAPTATION_STAGE else station_ids):",
            'model_paths["fed_meta_noft"] = get_fed_normal_meta_model_path(station_id)',
            "if not SKIP_EXTREME_ADAPTATION_STAGE:",
            '"skip_extreme_adaptation_stage": SKIP_EXTREME_ADAPTATION_STAGE',
        ]:
            self.assertIn(token, text)

    def test_training_declares_high_wind_aware_loss_controls(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            'HWA_PRETRAIN_LOSS = os.getenv("HWA_PRETRAIN_LOSS", "0") != "0"',
            'HWA_META_LOSS = os.getenv("HWA_META_LOSS", "0") != "0"',
            'HWA_SELECTIVE_PROXY_LOSS = os.getenv("HWA_SELECTIVE_PROXY_LOSS", "0") != "0"',
            'HWA_WIND_THRESHOLD = float(os.getenv("HWA_WIND_THRESHOLD", "10.0"))',
            'HWA_WIND_RAMP_END = float(os.getenv("HWA_WIND_RAMP_END", "13.9"))',
            'HWA_HIGH_WIND_WEIGHT = float(os.getenv("HWA_HIGH_WIND_WEIGHT", "4.0"))',
            'HWA_PRETRAIN_WINDOWED = os.getenv("HWA_PRETRAIN_WINDOWED", "0") != "0"',
            'ENABLE_TARGET_AWARE_META_NOFT = os.getenv("ENABLE_TARGET_AWARE_META_NOFT", "0") != "0"',
            'ENABLE_TARGET_AWARE_SELECTIVE_FED_META = os.getenv("ENABLE_TARGET_AWARE_SELECTIVE_FED_META", "0") != "0"',
            'SKIP_TARGET_AWARE_SELECTIVE_FED_META = os.getenv("SKIP_TARGET_AWARE_SELECTIVE_FED_META", "0") != "0"',
            'TARGET_AWARE_PRETRAIN_HWA_LOSS = os.getenv("TARGET_AWARE_PRETRAIN_HWA_LOSS", "1") != "0"',
            'TARGET_AWARE_META_CDRM_WEIGHT = float(os.getenv("TARGET_AWARE_META_CDRM_WEIGHT", "0.0"))',
            'TARGET_AWARE_META_WIND_MEAN_THRESHOLD = float(os.getenv("TARGET_AWARE_META_WIND_MEAN_THRESHOLD", "13.9"))',
            'TARGET_AWARE_META_WIND_MAX_THRESHOLD = float(os.getenv("TARGET_AWARE_META_WIND_MAX_THRESHOLD", "17.2"))',
            'TARGET_AWARE_META_MIN_EXTREME_POINTS = int(os.getenv("TARGET_AWARE_META_MIN_EXTREME_POINTS", "3"))',
            'TARGET_AWARE_SELECTIVE_FED_TOP_K = int(os.getenv("TARGET_AWARE_SELECTIVE_FED_TOP_K", "1"))',
            'TARGET_AWARE_SELECTIVE_FED_SELF_FLOOR = float(os.getenv("TARGET_AWARE_SELECTIVE_FED_SELF_FLOOR", "0.7"))',
            'TARGET_AWARE_SELECTIVE_FED_SOURCE_ALPHA_CAP = float(os.getenv("TARGET_AWARE_SELECTIVE_FED_SOURCE_ALPHA_CAP", "0.3"))',
            'GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "1029"))',
            "seed_torch(seed=GLOBAL_SEED)",
            "station_seed_base = GLOBAL_SEED + int(station_id) * 1000",
            "TARGET_AWARE_BASE_MODEL_OUTPUT_DIR",
            "def compute_hwa_weights_numpy_from_normalized",
            "def compute_hwa_weights_tensor_from_normalized",
            "def weighted_mse_loss",
            "def compute_state_delta",
            "def apply_weighted_state_deltas",
            "def build_target_aware_selective_proxy_tensors",
            "def build_windowed_pretrain_arrays",
            "def get_target_aware_task_score",
            "def run_target_aware_pretraining",
            "def run_target_aware_meta_training",
            "def run_target_aware_selective_fed_meta_training",
            "target_aware_selective_fed_meta_station",
            "get_target_aware_selective_fed_meta_model_path",
            "use_hwa_loss=TARGET_AWARE_PRETRAIN_HWA_LOSS",
            "hwa_weight",
            "input_windowed",
            "Train_weight_support",
            "Train_weight_query",
            "station_id=station_id",
            "target_aware=True",
            '"hwa_pretrain_loss": HWA_PRETRAIN_LOSS',
            '"hwa_meta_loss": HWA_META_LOSS',
            '"hwa_selective_proxy_loss": HWA_SELECTIVE_PROXY_LOSS',
            '"enable_target_aware_meta_noft": ENABLE_TARGET_AWARE_META_NOFT',
            '"enable_target_aware_selective_fed_meta": ENABLE_TARGET_AWARE_SELECTIVE_FED_META',
            '"global_seed": GLOBAL_SEED',
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
