import os
import time
import math
import json
import csv
import numpy as np
import torch.nn as nn
import torch
import copy
from torch.utils.tensorboard import SummaryWriter
import scipy.io as scio
import random
import model
from torch.nn.utils import weight_norm

# ========== [多场站运行开关] ==========
USE_FEDERATION = True  # True=多场站, False=单场站原方法
# 说明：设为False时完全退化为原始单场站元学习方法

# ========== [基线重置开关] ==========
USE_PSEUDO_FED = False  # False=去掉 shared pretrain/shared meta，恢复 local LMT 基线

# ========== 论文口径关键开关 ==========
TRAIN_META_ONLY_BASELINE = os.getenv("TRAIN_META_ONLY_BASELINE", "0") != "0"
TRAIN_PRETRAIN_ONLY = os.getenv("TRAIN_PRETRAIN_ONLY", "0") != "0"
RUN_FEDERATED_PRETRAIN = os.getenv("RUN_FEDERATED_PRETRAIN", "0") != "0"
FED_PRETRAIN_ALGO = os.getenv("FED_PRETRAIN_ALGO", "fedavg" if RUN_FEDERATED_PRETRAIN else "fedsgd").strip().lower()
FEDAVG_LOCAL_EPOCHS = max(1, int(os.getenv("FEDAVG_LOCAL_EPOCHS", "1")))
FEDAVG_CLIENT_WEIGHTING = os.getenv("FEDAVG_CLIENT_WEIGHTING", "uniform").strip().lower()
FEDAVG_LR = float(os.getenv("FEDAVG_LR", "0.0002"))
SKIP_LOCAL_PRETRAIN = os.getenv("SKIP_LOCAL_PRETRAIN", "0") != "0"
SKIP_LOCAL_META = os.getenv("SKIP_LOCAL_META", "0") != "0"
ENABLE_FEDTL_FT = os.getenv("ENABLE_FEDTL_FT", "0") != "0"
ENABLE_FED_NORMAL_META_PROPOSED = os.getenv("ENABLE_FED_NORMAL_META_PROPOSED", "0") != "0"
FED_NORMAL_META_SELF_FLOOR = float(os.getenv("FED_NORMAL_META_SELF_FLOOR", "0.3"))
SKIP_FED_NORMAL_META = os.getenv("SKIP_FED_NORMAL_META", "0") != "0"
FED_NORMAL_META_RESTORE_BEST = os.getenv("FED_NORMAL_META_RESTORE_BEST", "0") != "0"
FED_NORMAL_META_SAVE_BEST = os.getenv("FED_NORMAL_META_SAVE_BEST", "0") != "0"
FED_NORMAL_META_USE_BEST = os.getenv("FED_NORMAL_META_USE_BEST", "0") != "0"
ENABLE_SELECTIVE_FED_NORMAL_META = os.getenv("ENABLE_SELECTIVE_FED_NORMAL_META", "0") != "0"
SELECTIVE_FED_META_PROXY_RATIO = float(os.getenv("SELECTIVE_FED_META_PROXY_RATIO", "0.2"))
SELECTIVE_FED_META_SELF_FLOOR = float(os.getenv("SELECTIVE_FED_META_SELF_FLOOR", "0.5"))
SELECTIVE_FED_META_GAIN_MARGIN = float(os.getenv("SELECTIVE_FED_META_GAIN_MARGIN", "0.0"))
SELECTIVE_FED_META_GAIN_GAMMA = float(os.getenv("SELECTIVE_FED_META_GAIN_GAMMA", "1.0"))
HWA_PRETRAIN_LOSS = os.getenv("HWA_PRETRAIN_LOSS", "0") != "0"
HWA_META_LOSS = os.getenv("HWA_META_LOSS", "0") != "0"
HWA_SELECTIVE_PROXY_LOSS = os.getenv("HWA_SELECTIVE_PROXY_LOSS", "0") != "0"
HWA_WIND_FEATURE_INDEX = int(os.getenv("HWA_WIND_FEATURE_INDEX", "0"))
HWA_WIND_THRESHOLD = float(os.getenv("HWA_WIND_THRESHOLD", "10.0"))
HWA_WIND_RAMP_END = float(os.getenv("HWA_WIND_RAMP_END", "13.9"))
HWA_HIGH_WIND_WEIGHT = float(os.getenv("HWA_HIGH_WIND_WEIGHT", "4.0"))
HWA_PRETRAIN_WINDOWED = os.getenv("HWA_PRETRAIN_WINDOWED", "0") != "0"
ENABLE_TARGET_AWARE_META_NOFT = os.getenv("ENABLE_TARGET_AWARE_META_NOFT", "0") != "0"
SKIP_TARGET_AWARE_PRETRAIN = os.getenv("SKIP_TARGET_AWARE_PRETRAIN", "0") != "0"
SKIP_TARGET_AWARE_META = os.getenv("SKIP_TARGET_AWARE_META", "0") != "0"
ENABLE_TARGET_AWARE_SELECTIVE_FED_META = os.getenv("ENABLE_TARGET_AWARE_SELECTIVE_FED_META", "0") != "0"
SKIP_TARGET_AWARE_SELECTIVE_FED_META = os.getenv("SKIP_TARGET_AWARE_SELECTIVE_FED_META", "0") != "0"
ENABLE_TARGET_AWARE_SELECTIVE_FED_LOCAL_FT = os.getenv("ENABLE_TARGET_AWARE_SELECTIVE_FED_LOCAL_FT", "0") != "0"
SKIP_LEGACY_EXTREME_ADAPTATION = os.getenv("SKIP_LEGACY_EXTREME_ADAPTATION", "0") != "0"
TARGET_AWARE_PRETRAIN_HWA_LOSS = os.getenv("TARGET_AWARE_PRETRAIN_HWA_LOSS", "1") != "0"
TARGET_AWARE_META_CDRM_WEIGHT = float(os.getenv("TARGET_AWARE_META_CDRM_WEIGHT", "0.0"))
TARGET_AWARE_META_SIM_FLOOR = float(os.getenv("TARGET_AWARE_META_SIM_FLOOR", "0.05"))
TARGET_AWARE_META_TASK_WEIGHT_ETA = float(os.getenv("TARGET_AWARE_META_TASK_WEIGHT_ETA", "1.0"))
TARGET_AWARE_META_WIND_MEAN_THRESHOLD = float(os.getenv("TARGET_AWARE_META_WIND_MEAN_THRESHOLD", "13.9"))
TARGET_AWARE_META_WIND_MAX_THRESHOLD = float(os.getenv("TARGET_AWARE_META_WIND_MAX_THRESHOLD", "17.2"))
TARGET_AWARE_META_MIN_EXTREME_POINTS = int(os.getenv("TARGET_AWARE_META_MIN_EXTREME_POINTS", "3"))
TARGET_AWARE_SELECTIVE_FED_TOP_K = int(os.getenv("TARGET_AWARE_SELECTIVE_FED_TOP_K", "1"))
TARGET_AWARE_SELECTIVE_FED_SELF_FLOOR = float(os.getenv("TARGET_AWARE_SELECTIVE_FED_SELF_FLOOR", "0.7"))
TARGET_AWARE_SELECTIVE_FED_SOURCE_ALPHA_CAP = float(os.getenv("TARGET_AWARE_SELECTIVE_FED_SOURCE_ALPHA_CAP", "0.3"))
TARGET_AWARE_SELECTIVE_FED_ALPHA_GRID = os.getenv("TARGET_AWARE_SELECTIVE_FED_ALPHA_GRID", "")
TARGET_AWARE_SELECTIVE_FED_GAIN_MARGIN = float(os.getenv("TARGET_AWARE_SELECTIVE_FED_GAIN_MARGIN", "0.0"))
TARGET_AWARE_SELECTIVE_FED_GAIN_GAMMA = float(os.getenv("TARGET_AWARE_SELECTIVE_FED_GAIN_GAMMA", "1.0"))
TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_MAX_WINDOWS = int(os.getenv("TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_MAX_WINDOWS", "8"))
TARGET_AWARE_SELECTIVE_FED_PROXY_EXTREME_WEIGHT = float(os.getenv("TARGET_AWARE_SELECTIVE_FED_PROXY_EXTREME_WEIGHT", "2.0"))
TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_WEIGHT = float(os.getenv("TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_WEIGHT", "1.0"))
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "1029"))
if FED_NORMAL_META_RESTORE_BEST:
    FED_NORMAL_META_SAVE_BEST = True
    FED_NORMAL_META_USE_BEST = True
if FED_NORMAL_META_USE_BEST:
    FED_NORMAL_META_SAVE_BEST = True
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", ".")


def resolve_artifact_path(filename):
    if os.path.isabs(filename):
        return filename
    if ARTIFACT_DIR in ("", "."):
        return filename
    return os.path.join(ARTIFACT_DIR, filename)


MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", resolve_artifact_path("models") if ARTIFACT_DIR not in ("", ".") else ".")
BASE_MODEL_OUTPUT_DIR = os.getenv("BASE_MODEL_OUTPUT_DIR", MODEL_OUTPUT_DIR)
TARGET_AWARE_BASE_MODEL_OUTPUT_DIR = os.getenv("TARGET_AWARE_BASE_MODEL_OUTPUT_DIR", MODEL_OUTPUT_DIR)
TARGET_AWARE_SELECTIVE_FED_BASE_MODEL_OUTPUT_DIR = os.getenv(
    "TARGET_AWARE_SELECTIVE_FED_BASE_MODEL_OUTPUT_DIR",
    TARGET_AWARE_BASE_MODEL_OUTPUT_DIR,
)
LOCAL_PRETRAIN_INIT_MODEL_DIR = os.getenv("LOCAL_PRETRAIN_INIT_MODEL_DIR", "")
LOCAL_META_INIT_MODEL_DIR = os.getenv("LOCAL_META_INIT_MODEL_DIR", "")
TARGET_AWARE_PRETRAIN_INIT_MODEL_DIR = os.getenv("TARGET_AWARE_PRETRAIN_INIT_MODEL_DIR", "")
TARGET_AWARE_META_INIT_MODEL_DIR = os.getenv("TARGET_AWARE_META_INIT_MODEL_DIR", "")
TARGET_AWARE_SELECTIVE_FED_META_INIT_MODEL_DIR = os.getenv("TARGET_AWARE_SELECTIVE_FED_META_INIT_MODEL_DIR", "")
LOCAL_PRETRAIN_EPOCH_OFFSET = int(os.getenv("LOCAL_PRETRAIN_EPOCH_OFFSET", "0"))
LOCAL_META_EPOCH_OFFSET = int(os.getenv("LOCAL_META_EPOCH_OFFSET", "0"))
TARGET_AWARE_PRETRAIN_EPOCH_OFFSET = int(os.getenv("TARGET_AWARE_PRETRAIN_EPOCH_OFFSET", "0"))
TARGET_AWARE_META_EPOCH_OFFSET = int(os.getenv("TARGET_AWARE_META_EPOCH_OFFSET", "0"))
TARGET_AWARE_SELECTIVE_FED_META_EPOCH_OFFSET = int(os.getenv("TARGET_AWARE_SELECTIVE_FED_META_EPOCH_OFFSET", "0"))
if BASE_MODEL_OUTPUT_DIR != MODEL_OUTPUT_DIR:
    can_reuse_base_dir = SKIP_LOCAL_PRETRAIN and SKIP_LOCAL_META
    if not can_reuse_base_dir:
        raise ValueError(
            "BASE_MODEL_OUTPUT_DIR can differ from MODEL_OUTPUT_DIR only when "
            "local pretrain and local meta checkpoints are skipped"
        )
LOGS_TRAIN_DIR = os.getenv("LOGS_TRAIN_DIR", resolve_artifact_path("logs_train"))
ALL_STATIONS_TEST_RESULTS_PATH = os.getenv(
    "ALL_STATIONS_TEST_RESULTS_PATH",
    resolve_artifact_path("all_stations_test_results.mat"),
)
FEW_SHOT_EPOCHS = int(os.getenv("FEW_SHOT_EPOCHS", "50"))             # 论文口径：每个极端天气 fine-tune 50 epochs
FEW_SHOT_LR = float(os.getenv("FEW_SHOT_LR", "0.0002"))
ENABLE_FT_SWEEP = os.getenv("ENABLE_FT_SWEEP", "0") != "0"
FT_SWEEP_EPOCHS = os.getenv("FT_SWEEP_EPOCHS", "")
FT_SWEEP_OUTPUT_PATH = os.getenv("FT_SWEEP_OUTPUT_PATH", resolve_artifact_path("ft_sweep_results.csv"))
FT_SWEEP_SAVE_CHECKPOINTS = os.getenv("FT_SWEEP_SAVE_CHECKPOINTS", "1") != "0"
FT_SWEEP_EVAL_TEST = os.getenv("FT_SWEEP_EVAL_TEST", "1") != "0"
SKIP_EXTREME_ADAPTATION_STAGE = os.getenv("SKIP_EXTREME_ADAPTATION_STAGE", "0") != "0"
FEW_SHOT_USE_CDRM = False
FEW_SHOT_CDRM_WEIGHT = 5.0
META_TASKS_PER_EPOCH = int(os.getenv("META_TASKS_PER_EPOCH", "5"))
META_SUPPORT_SHOTS = int(os.getenv("META_SUPPORT_SHOTS", "10"))
META_QUERY_SHOTS = int(os.getenv("META_QUERY_SHOTS", "10"))
PRETRAIN_EPOCHS = int(os.getenv("PRETRAIN_EPOCHS", "35000"))
PROPOSED_META_EPOCHS = int(os.getenv("PROPOSED_META_EPOCHS", "30000"))
META_ONLY_META_EPOCHS = int(os.getenv("META_ONLY_META_EPOCHS", "30000"))
PRETRAIN_LOG_INTERVAL = int(os.getenv("PRETRAIN_LOG_INTERVAL", "100"))
META_LOG_INTERVAL = int(os.getenv("META_LOG_INTERVAL", "100"))
FEW_SHOT_LOG_INTERVAL = int(os.getenv("FEW_SHOT_LOG_INTERVAL", "1"))
ENABLE_CONVERGENCE_MONITOR = os.getenv("ENABLE_CONVERGENCE_MONITOR", "1") != "0"
CONVERGENCE_REPORT_PATH = os.getenv(
    "CONVERGENCE_REPORT_PATH",
    resolve_artifact_path("training_convergence_report.json"),
)
CONVERGENCE_MIN_DELTA = float(os.getenv("CONVERGENCE_MIN_DELTA", "1e-4"))
CONVERGENCE_MIN_EPOCHS = int(os.getenv("CONVERGENCE_MIN_EPOCHS", "5"))
CONVERGENCE_PATIENCE_PRETRAIN = int(os.getenv("CONVERGENCE_PATIENCE_PRETRAIN", "200"))
CONVERGENCE_PATIENCE_META = int(os.getenv("CONVERGENCE_PATIENCE_META", "100"))
CONVERGENCE_PATIENCE_FEW_SHOT = int(os.getenv("CONVERGENCE_PATIENCE_FEW_SHOT", "5"))
# 论文消融口径：Meta-only = 去掉 pre-training，其余训练机制保持一致
META_ONLY_USE_CDRM = True
META_ONLY_TRAIN_ALL_PARAMS = False
META_ONLY_DISABLE_LWP = False
EXTREME_ADAPT_RATIO = float(os.getenv("EXTREME_ADAPT_RATIO", "0.75"))
EXTREME_MIN_VAL_HORIZON = int(os.getenv("EXTREME_MIN_VAL_HORIZON", "4"))
EXTREME_SOURCE_QUALITY_KEEP_RATIO = float(os.getenv("EXTREME_SOURCE_QUALITY_KEEP_RATIO", "0.8"))
EXTREME_MIN_EFFECTIVE_WINDOWS = int(os.getenv("EXTREME_MIN_EFFECTIVE_WINDOWS", "1"))
EXTREME_SOURCE_BORROW_BUDGET_GAMMA = float(os.getenv("EXTREME_SOURCE_BORROW_BUDGET_GAMMA", "1.0"))
EXTREME_USEFULNESS_ADAPT_STEPS = int(os.getenv("EXTREME_USEFULNESS_ADAPT_STEPS", "1"))
EXTREME_WEIGHT_LAMBDA = float(os.getenv("EXTREME_WEIGHT_LAMBDA", "1.0"))
EXTREME_WEIGHT_MU = float(os.getenv("EXTREME_WEIGHT_MU", "1.0"))
EXTREME_WEIGHT_NU = float(os.getenv("EXTREME_WEIGHT_NU", "1.0"))
EXTREME_WEIGHT_BETA_SELF = float(os.getenv("EXTREME_WEIGHT_BETA_SELF", "0.5"))
EXTREME_WEIGHT_BETA_SELF_BY_CLASS = os.getenv("EXTREME_WEIGHT_BETA_SELF_BY_CLASS", "")
EXTREME_WEIGHT_TAU_Q = float(os.getenv("EXTREME_WEIGHT_TAU_Q", "1.0"))
EXTREME_WEIGHT_TAU_T = float(os.getenv("EXTREME_WEIGHT_TAU_T", "1.0"))
EXTREME_TARGET_REFINEMENT_EPOCHS = int(
    os.getenv("EXTREME_TARGET_REFINEMENT_EPOCHS", str(max(1, FEW_SHOT_EPOCHS // 2)))
)
EXTREME_TARGET_ADAPT_MAX_WINDOWS = int(os.getenv("EXTREME_TARGET_ADAPT_MAX_WINDOWS", "0"))
EXTREME_TARGET_ADAPT_MAX_WINDOWS_BY_CLASS = os.getenv("EXTREME_TARGET_ADAPT_MAX_WINDOWS_BY_CLASS", "")
EXTREME_ANCHOR_REG_LAMBDA = float(os.getenv("EXTREME_ANCHOR_REG_LAMBDA", "0.0"))
EXTREME_SOURCE_HARD_GATE = os.getenv("EXTREME_SOURCE_HARD_GATE", "0") != "0"
EXTREME_SOURCE_HARD_GATE_BY_CLASS = os.getenv("EXTREME_SOURCE_HARD_GATE_BY_CLASS", "")
EXTREME_SOURCE_MIN_TARGET_GAIN = float(os.getenv("EXTREME_SOURCE_MIN_TARGET_GAIN", "0.0"))
EXTREME_SOURCE_MIN_TARGET_GAIN_BY_CLASS = os.getenv("EXTREME_SOURCE_MIN_TARGET_GAIN_BY_CLASS", "")
EXTREME_SOURCE_GAIN_WEIGHT_ETA = float(os.getenv("EXTREME_SOURCE_GAIN_WEIGHT_ETA", "0.0"))
EXTREME_SOURCE_GAIN_WEIGHT_ETA_BY_CLASS = os.getenv("EXTREME_SOURCE_GAIN_WEIGHT_ETA_BY_CLASS", "")
EXTREME_SOURCE_TOP_K_BY_CLASS = os.getenv("EXTREME_SOURCE_TOP_K_BY_CLASS", "")
EXTREME_FORCE_LOCAL_FALLBACK_BY_CLASS = os.getenv("EXTREME_FORCE_LOCAL_FALLBACK_BY_CLASS", "")
EXTREME_PROPOSED_VAL_FALLBACK = os.getenv("EXTREME_PROPOSED_VAL_FALLBACK", "0") != "0"
EXTREME_PROPOSED_VAL_FALLBACK_BY_CLASS = os.getenv("EXTREME_PROPOSED_VAL_FALLBACK_BY_CLASS", "")
EXTREME_PROPOSED_VAL_FALLBACK_MARGIN = float(os.getenv("EXTREME_PROPOSED_VAL_FALLBACK_MARGIN", "0.0"))
EXTREME_PROPOSED_VAL_FALLBACK_MARGIN_BY_CLASS = os.getenv("EXTREME_PROPOSED_VAL_FALLBACK_MARGIN_BY_CLASS", "")

# ========== [数据协议] 支持 1h/12点 legacy 与 2h/6点 yearly protocol ==========
YEARLY_PROTOCOL_ENABLED = os.getenv("YEARLY_PROTOCOL_ENABLED", "0") != "0"
SEASONAL_PROTOCOL_ENABLED = os.getenv("SEASONAL_PROTOCOL_ENABLED", "0") != "0"
DEFAULT_YEARLY_PROTOCOL_METADATA_PATH = "three_station_yearly_protocol_data/three_station_yearly_protocol_metadata.json"
YEARLY_PROTOCOL_METADATA_PATH = os.getenv("YEARLY_PROTOCOL_METADATA_PATH", DEFAULT_YEARLY_PROTOCOL_METADATA_PATH)


def load_yearly_protocol_metadata(metadata_path):
    if not metadata_path or not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


PROTOCOL_METADATA_PATH = os.getenv(
    "PROTOCOL_METADATA_PATH",
    YEARLY_PROTOCOL_METADATA_PATH if YEARLY_PROTOCOL_ENABLED else "",
)
yearly_protocol_metadata = load_yearly_protocol_metadata(PROTOCOL_METADATA_PATH)
protocol_metadata = yearly_protocol_metadata
PROTOCOL_NAME = os.getenv(
    "PROTOCOL_NAME",
    yearly_protocol_metadata.get("protocol_name", "legacy_1h_12point"),
)
PROTOCOL_DATA_DIR = os.getenv(
    "PROTOCOL_DATA_DIR",
    yearly_protocol_metadata.get("protocol_data_dir", ""),
)
LEN_REALP = int(os.getenv("LEN_REALP", str(yearly_protocol_metadata.get("len_realp", 12))))
POINTS_PER_DAY = int(os.getenv("POINTS_PER_DAY", str(yearly_protocol_metadata.get("points_per_day", 24))))
SAMPLE_INTERVAL_HOURS = int(
    os.getenv(
        "SAMPLE_INTERVAL_HOURS",
        str(yearly_protocol_metadata.get("sample_interval_hours", max(1, 24 // max(1, POINTS_PER_DAY)))),
    )
)
DOWNSAMPLE_OFFSET = int(os.getenv("DOWNSAMPLE_OFFSET", str(yearly_protocol_metadata.get("downsample_offset", 0))))
WINDOW_SPAN_HOURS = int(
    os.getenv(
        "WINDOW_SPAN_HOURS",
        str(yearly_protocol_metadata.get("window_span_hours", SAMPLE_INTERVAL_HOURS * LEN_REALP)),
    )
)
TRAIN_YEARS = yearly_protocol_metadata.get("train_years", [2022])
TEST_YEARS = yearly_protocol_metadata.get("test_years", [2023])
extreme_class_names = protocol_metadata.get("extreme_class_names", ["high_wind", "high_temp", "cold_wave", "frost"])
if not extreme_class_names:
    extreme_class_names = ["high_wind", "high_temp", "cold_wave", "frost"]
num_extreme_classes = int(protocol_metadata.get("num_extreme_classes", len(extreme_class_names)))
extreme_eval_labels = protocol_metadata.get(
    "extreme_eval_labels",
    [str(class_name).replace("_", " ").title().replace(" ", "") for class_name in extreme_class_names],
)
if len(extreme_eval_labels) < num_extreme_classes:
    for class_index in range(len(extreme_eval_labels), num_extreme_classes):
        extreme_eval_labels.append(f"ExtremeClass{class_index + 1}")
convergence_records = []


def resolve_station_mat_path(station_id):
    filename = f"{station_id}wf_4_train.mat"
    if PROTOCOL_DATA_DIR:
        candidate = os.path.join(PROTOCOL_DATA_DIR, filename)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Protocol mat missing: {candidate}")
        return candidate
    return filename


def resolve_station_ids():
    metadata_stations = yearly_protocol_metadata.get("stations", [])
    if USE_FEDERATION and metadata_stations:
        return [str(station["station_id"]) for station in metadata_stations]
    if USE_FEDERATION:
        return ['58', '59', '60']
    return ['58']


def resolve_model_path(filename):
    if os.path.isabs(filename):
        return filename
    return os.path.join(MODEL_OUTPUT_DIR, filename)


def resolve_base_model_path(filename):
    if os.path.isabs(filename):
        return filename
    return os.path.join(BASE_MODEL_OUTPUT_DIR, filename)


def resolve_target_aware_base_model_path(filename):
    if os.path.isabs(filename):
        return filename
    return os.path.join(TARGET_AWARE_BASE_MODEL_OUTPUT_DIR, filename)


def resolve_target_aware_selective_fed_base_model_path(filename):
    if os.path.isabs(filename):
        return filename
    return os.path.join(TARGET_AWARE_SELECTIVE_FED_BASE_MODEL_OUTPUT_DIR, filename)


def resolve_init_model_path(init_model_dir, filename):
    if not init_model_dir:
        return None
    if os.path.isabs(filename):
        return filename
    return os.path.join(init_model_dir, filename)


def resolve_fed_normal_meta_model_path(filename):
    if SKIP_FED_NORMAL_META:
        return resolve_base_model_path(filename)
    return resolve_model_path(filename)


def print_protocol_banner():
    progress_log("=" * 70)
    progress_log("数据协议配置")
    progress_log("=" * 70)
    progress_log(f"  protocol_name: {PROTOCOL_NAME}")
    progress_log(f"  protocol_data_dir: {PROTOCOL_DATA_DIR or '(legacy root mats)'}")
    progress_log(f"  protocol_metadata_path: {PROTOCOL_METADATA_PATH or '(none)'}")
    progress_log(f"  yearly_protocol_enabled: {YEARLY_PROTOCOL_ENABLED}")
    progress_log(f"  seasonal_protocol_enabled: {SEASONAL_PROTOCOL_ENABLED}")
    progress_log(f"  sample_interval_hours: {SAMPLE_INTERVAL_HOURS}")
    progress_log(f"  downsample_offset: {DOWNSAMPLE_OFFSET}")
    progress_log(f"  len_realp: {LEN_REALP}")
    progress_log(f"  points_per_day: {POINTS_PER_DAY}")
    progress_log(f"  window_span_hours: {WINDOW_SPAN_HOURS}")
    progress_log(f"  artifact_dir: {ARTIFACT_DIR}")
    progress_log(f"  model_output_dir: {MODEL_OUTPUT_DIR}")
    progress_log(f"  base_model_output_dir: {BASE_MODEL_OUTPUT_DIR}")
    progress_log(f"  target_aware_base_model_output_dir: {TARGET_AWARE_BASE_MODEL_OUTPUT_DIR}")
    progress_log(f"  target_aware_selective_fed_base_model_output_dir: {TARGET_AWARE_SELECTIVE_FED_BASE_MODEL_OUTPUT_DIR}")
    progress_log(f"  local_pretrain_init_model_dir: {LOCAL_PRETRAIN_INIT_MODEL_DIR or '(none)'}")
    progress_log(f"  local_meta_init_model_dir: {LOCAL_META_INIT_MODEL_DIR or '(none)'}")
    progress_log(f"  target_aware_pretrain_init_model_dir: {TARGET_AWARE_PRETRAIN_INIT_MODEL_DIR or '(none)'}")
    progress_log(f"  target_aware_meta_init_model_dir: {TARGET_AWARE_META_INIT_MODEL_DIR or '(none)'}")
    progress_log(f"  target_aware_selective_fed_meta_init_model_dir: {TARGET_AWARE_SELECTIVE_FED_META_INIT_MODEL_DIR or '(none)'}")
    progress_log(f"  local_pretrain_epoch_offset: {LOCAL_PRETRAIN_EPOCH_OFFSET}")
    progress_log(f"  local_meta_epoch_offset: {LOCAL_META_EPOCH_OFFSET}")
    progress_log(f"  target_aware_pretrain_epoch_offset: {TARGET_AWARE_PRETRAIN_EPOCH_OFFSET}")
    progress_log(f"  target_aware_meta_epoch_offset: {TARGET_AWARE_META_EPOCH_OFFSET}")
    progress_log(f"  target_aware_selective_fed_meta_epoch_offset: {TARGET_AWARE_SELECTIVE_FED_META_EPOCH_OFFSET}")
    progress_log(f"  logs_train_dir: {LOGS_TRAIN_DIR}")
    progress_log(f"  train_pretrain_only: {TRAIN_PRETRAIN_ONLY}")
    progress_log(f"  run_federated_pretrain: {RUN_FEDERATED_PRETRAIN}")
    progress_log(f"  fed_pretrain_algo: {FED_PRETRAIN_ALGO}")
    progress_log(f"  fedavg_local_epochs: {FEDAVG_LOCAL_EPOCHS}")
    progress_log(f"  fedavg_client_weighting: {FEDAVG_CLIENT_WEIGHTING}")
    progress_log(f"  enable_fedtl_ft: {ENABLE_FEDTL_FT}")
    progress_log(f"  enable_ft_sweep: {ENABLE_FT_SWEEP}")
    progress_log(f"  ft_sweep_epochs: {FT_SWEEP_EPOCHS or '(none)'}")
    progress_log(f"  enable_fed_normal_meta_proposed: {ENABLE_FED_NORMAL_META_PROPOSED}")
    progress_log(f"  fed_normal_meta_self_floor: {FED_NORMAL_META_SELF_FLOOR}")
    progress_log(f"  skip_fed_normal_meta: {SKIP_FED_NORMAL_META}")
    progress_log(f"  fed_normal_meta_restore_best: {FED_NORMAL_META_RESTORE_BEST}")
    progress_log(f"  fed_normal_meta_save_best: {FED_NORMAL_META_SAVE_BEST}")
    progress_log(f"  fed_normal_meta_use_best: {FED_NORMAL_META_USE_BEST}")
    progress_log(f"  enable_selective_fed_normal_meta: {ENABLE_SELECTIVE_FED_NORMAL_META}")
    progress_log(f"  selective_fed_meta_proxy_ratio: {SELECTIVE_FED_META_PROXY_RATIO}")
    progress_log(f"  selective_fed_meta_self_floor: {SELECTIVE_FED_META_SELF_FLOOR}")
    progress_log(f"  selective_fed_meta_gain_margin: {SELECTIVE_FED_META_GAIN_MARGIN}")
    progress_log(f"  selective_fed_meta_gain_gamma: {SELECTIVE_FED_META_GAIN_GAMMA}")
    progress_log(f"  hwa_pretrain_loss: {HWA_PRETRAIN_LOSS}")
    progress_log(f"  hwa_meta_loss: {HWA_META_LOSS}")
    progress_log(f"  hwa_selective_proxy_loss: {HWA_SELECTIVE_PROXY_LOSS}")
    progress_log(f"  hwa_wind_feature_index: {HWA_WIND_FEATURE_INDEX}")
    progress_log(f"  hwa_wind_threshold: {HWA_WIND_THRESHOLD}")
    progress_log(f"  hwa_wind_ramp_end: {HWA_WIND_RAMP_END}")
    progress_log(f"  hwa_high_wind_weight: {HWA_HIGH_WIND_WEIGHT}")
    progress_log(f"  hwa_pretrain_windowed: {HWA_PRETRAIN_WINDOWED}")
    progress_log(f"  enable_target_aware_meta_noft: {ENABLE_TARGET_AWARE_META_NOFT}")
    progress_log(f"  skip_target_aware_pretrain: {SKIP_TARGET_AWARE_PRETRAIN}")
    progress_log(f"  skip_target_aware_meta: {SKIP_TARGET_AWARE_META}")
    progress_log(f"  enable_target_aware_selective_fed_meta: {ENABLE_TARGET_AWARE_SELECTIVE_FED_META}")
    progress_log(f"  skip_target_aware_selective_fed_meta: {SKIP_TARGET_AWARE_SELECTIVE_FED_META}")
    progress_log(f"  enable_target_aware_selective_fed_local_ft: {ENABLE_TARGET_AWARE_SELECTIVE_FED_LOCAL_FT}")
    progress_log(f"  skip_legacy_extreme_adaptation: {SKIP_LEGACY_EXTREME_ADAPTATION}")
    progress_log(f"  target_aware_pretrain_hwa_loss: {TARGET_AWARE_PRETRAIN_HWA_LOSS}")
    progress_log(f"  target_aware_meta_cdrm_weight: {TARGET_AWARE_META_CDRM_WEIGHT}")
    progress_log(f"  target_aware_meta_sim_floor: {TARGET_AWARE_META_SIM_FLOOR}")
    progress_log(f"  target_aware_meta_task_weight_eta: {TARGET_AWARE_META_TASK_WEIGHT_ETA}")
    progress_log(f"  target_aware_meta_wind_mean_threshold: {TARGET_AWARE_META_WIND_MEAN_THRESHOLD}")
    progress_log(f"  target_aware_meta_wind_max_threshold: {TARGET_AWARE_META_WIND_MAX_THRESHOLD}")
    progress_log(f"  target_aware_meta_min_extreme_points: {TARGET_AWARE_META_MIN_EXTREME_POINTS}")
    progress_log(f"  target_aware_selective_fed_top_k: {TARGET_AWARE_SELECTIVE_FED_TOP_K}")
    progress_log(f"  target_aware_selective_fed_self_floor: {TARGET_AWARE_SELECTIVE_FED_SELF_FLOOR}")
    progress_log(f"  target_aware_selective_fed_source_alpha_cap: {TARGET_AWARE_SELECTIVE_FED_SOURCE_ALPHA_CAP}")
    progress_log(f"  target_aware_selective_fed_alpha_grid: {TARGET_AWARE_SELECTIVE_FED_ALPHA_GRID or '(auto)'}")
    progress_log(f"  target_aware_selective_fed_gain_margin: {TARGET_AWARE_SELECTIVE_FED_GAIN_MARGIN}")
    progress_log(f"  target_aware_selective_fed_proxy_normal_max_windows: {TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_MAX_WINDOWS}")
    progress_log(f"  target_aware_selective_fed_proxy_extreme_weight: {TARGET_AWARE_SELECTIVE_FED_PROXY_EXTREME_WEIGHT}")
    progress_log(f"  target_aware_selective_fed_proxy_normal_weight: {TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_WEIGHT}")
    progress_log(f"  global_seed: {GLOBAL_SEED}")
    progress_log(f"  skip_extreme_adaptation_stage: {SKIP_EXTREME_ADAPTATION_STAGE}")
    progress_log(f"  few_shot_lr: {FEW_SHOT_LR}")
    if ENABLE_SELECTIVE_FED_NORMAL_META:
        progress_log("  selective_fed_meta_mode: target_proxy_validated")


def progress_log(message=""):
    print(message, flush=True)


def should_log_epoch(epoch_index, total_epochs, interval, warmup_epochs=10):
    current_epoch = int(epoch_index) + 1
    if current_epoch <= max(1, int(warmup_epochs)):
        return True
    if current_epoch >= int(total_epochs):
        return True
    return current_epoch % max(1, int(interval)) == 0


def initialize_convergence_record(stage_type, stage_id, total_epochs, patience):
    return {
        "stage_type": stage_type,
        "stage_id": stage_id,
        "total_epochs": int(total_epochs),
        "patience": int(patience),
        "min_delta": float(CONVERGENCE_MIN_DELTA),
        "min_epochs": int(CONVERGENCE_MIN_EPOCHS),
        "converged": False,
        "convergence_epoch": None,
        "best_epoch": None,
        "best_loss": None,
        "final_loss": None,
        "started_at_unix": float(time.time()),
        "_last_improved_epoch": None,
        "_last_announced_convergence_epoch": None,
    }


def format_convergence_loss(loss_value):
    return "nan" if loss_value is None else f"{float(loss_value):.6f}"


def update_convergence_record(record, epoch, loss_value):
    if not ENABLE_CONVERGENCE_MONITOR or record is None:
        return

    current_epoch = int(epoch) + 1
    loss_value = float(loss_value)
    record["final_loss"] = loss_value

    if record["best_loss"] is None or loss_value < (record["best_loss"] - record["min_delta"]):
        record["best_loss"] = loss_value
        record["best_epoch"] = current_epoch
        record["_last_improved_epoch"] = current_epoch
        record["converged"] = False
        record["convergence_epoch"] = None
        return

    last_improved_epoch = record["_last_improved_epoch"]
    if (
        current_epoch >= record["min_epochs"]
        and record["convergence_epoch"] is None
        and last_improved_epoch is not None
        and (current_epoch - last_improved_epoch) >= record["patience"]
    ):
        record["converged"] = True
        record["convergence_epoch"] = current_epoch
        if record.get("_last_announced_convergence_epoch") != current_epoch:
            progress_log(
                f"  收敛检测[{record['stage_type']}:{record['stage_id']}]: "
                f"首次达到收敛条件 @ epoch {current_epoch} "
                f"(best_epoch={record['best_epoch']}, best_loss={format_convergence_loss(record['best_loss'])}, "
                f"final_loss={format_convergence_loss(record['final_loss'])})"
            )
            record["_last_announced_convergence_epoch"] = current_epoch


def finalize_convergence_record(record):
    if record is None:
        return None
    finalized_record = copy.deepcopy(record)
    started_at_unix = finalized_record.pop("started_at_unix", None)
    if started_at_unix is None:
        finalized_record["elapsed_seconds"] = None
    else:
        finalized_record["elapsed_seconds"] = max(0.0, float(time.time()) - float(started_at_unix))
    finalized_record.pop("_last_improved_epoch", None)
    finalized_record.pop("_last_announced_convergence_epoch", None)
    return finalized_record


def register_convergence_record(record):
    if not ENABLE_CONVERGENCE_MONITOR or record is None:
        return None
    finalized_record = finalize_convergence_record(record)
    convergence_records.append(finalized_record)
    if finalized_record["converged"]:
        progress_log(
            f"  收敛检测[{finalized_record['stage_type']}:{finalized_record['stage_id']}]: "
            f"已收敛 @ epoch {finalized_record['convergence_epoch']} "
            f"(best_epoch={finalized_record['best_epoch']}, best_loss={format_convergence_loss(finalized_record['best_loss'])}, "
            f"final_loss={format_convergence_loss(finalized_record['final_loss'])})"
        )
    else:
        progress_log(
            f"  收敛检测[{finalized_record['stage_type']}:{finalized_record['stage_id']}]: "
            f"未收敛 (best_epoch={finalized_record['best_epoch']}, "
            f"best_loss={format_convergence_loss(finalized_record['best_loss'])}, "
            f"final_loss={format_convergence_loss(finalized_record['final_loss'])})"
        )
    return finalized_record


def export_convergence_report(report_path, records, run_config):
    if not ENABLE_CONVERGENCE_MONITOR:
        return
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    report_payload = {
        "generated_at_unix": float(time.time()),
        "run_config": run_config,
        "records": records,
    }
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report_payload, report_file, indent=2, ensure_ascii=False)
    progress_log(f"✓ 收敛检测报告已保存: {report_path}")

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, mode='pre', kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [model.TemporalBlock_v2(in_channels, out_channels, kernel_size,  stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, mode=mode)]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def reshape_complete_window_pairs(power_array, nwp_array, len_realp, dem_realp, dem_realc):
    window_count = min(
        int(np.size(power_array, axis=0) // len_realp),
        int(np.size(nwp_array, axis=0) // len_realp),
    )
    if window_count <= 0:
        return (
            np.empty((0, len_realp, dem_realp), dtype=np.float32),
            np.empty((0, len_realp, dem_realc), dtype=np.float32),
        )

    power_trim = np.asarray(power_array[:window_count * len_realp], dtype=np.float32)
    nwp_trim = np.asarray(nwp_array[:window_count * len_realp], dtype=np.float32)
    return (
        power_trim.reshape(window_count, len_realp, dem_realp),
        nwp_trim.reshape(window_count, len_realp, dem_realc),
    )


def split_normal_meta_train_proxy_windows(window_payloads, proxy_ratio=None, seed=GLOBAL_SEED):
    proxy_ratio = SELECTIVE_FED_META_PROXY_RATIO if proxy_ratio is None else float(proxy_ratio)
    proxy_ratio = min(max(proxy_ratio, 0.0), 0.9)
    rng = np.random.RandomState(seed)
    split_payloads = []
    for payload in window_payloads:
        nwp_windows = np.asarray(payload["nwp"], dtype=np.float32)
        power_windows = np.asarray(payload["p"], dtype=np.float32)
        window_count = min(nwp_windows.shape[0], power_windows.shape[0])
        if window_count <= 1:
            split_payloads.append({
                "meta_train_nwp": nwp_windows[:window_count],
                "meta_train_power": power_windows[:window_count],
                "proxy_nwp": np.empty((0,) + nwp_windows.shape[1:], dtype=np.float32),
                "proxy_power": np.empty((0,) + power_windows.shape[1:], dtype=np.float32),
            })
            continue
        indices = np.arange(window_count)
        rng.shuffle(indices)
        meta_total_shots = META_SUPPORT_SHOTS + META_QUERY_SHOTS
        max_proxy_count = max(0, window_count - meta_total_shots)
        requested_proxy_count = max(1, int(round(window_count * proxy_ratio)))
        proxy_count = min(max_proxy_count, requested_proxy_count)
        if proxy_count <= 0:
            train_indices = np.sort(indices)
            split_payloads.append({
                "meta_train_nwp": nwp_windows[train_indices],
                "meta_train_power": power_windows[train_indices],
                "proxy_nwp": np.empty((0,) + nwp_windows.shape[1:], dtype=np.float32),
                "proxy_power": np.empty((0,) + power_windows.shape[1:], dtype=np.float32),
            })
            continue
        proxy_indices = np.sort(indices[:proxy_count])
        train_indices = np.sort(indices[proxy_count:])
        split_payloads.append({
            "meta_train_nwp": nwp_windows[train_indices],
            "meta_train_power": power_windows[train_indices],
            "proxy_nwp": nwp_windows[proxy_indices],
            "proxy_power": power_windows[proxy_indices],
        })
    return split_payloads


def build_station_normal_meta_train_proxy_payloads(p_conven_class_st, nwp_conven_class_st, station_id):
    total_station_classes = np.size(p_conven_class_st, axis=1)
    nwp_feature_count = np.size(nwp_conven_class_st, axis=1)
    meta_train_power_classes = np.empty((1, total_station_classes), dtype=object)
    meta_train_nwp_classes = np.empty((1, nwp_feature_count), dtype=object)
    for i_nwp in range(nwp_feature_count):
        meta_train_nwp_classes[0, i_nwp] = np.empty((1, total_station_classes), dtype=object)

    proxy_nwp_windows_all = []
    proxy_power_windows_all = []
    station_seed_base = GLOBAL_SEED + int(station_id) * 1000

    for i_class in range(total_station_classes):
        class_nwp_windows = None
        for i_nwp in range(nwp_feature_count):
            nwp_data = nwp_conven_class_st[0, i_nwp][0, i_class]
            num_samples = nwp_data.shape[0] // len_realp
            nwp_reshaped = np.asarray(
                nwp_data[:num_samples * len_realp].reshape(num_samples, len_realp, 1),
                dtype=np.float32,
            )
            if class_nwp_windows is None:
                class_nwp_windows = nwp_reshaped
            else:
                class_nwp_windows = np.concatenate((class_nwp_windows, nwp_reshaped), axis=2)

        p_data = p_conven_class_st[0, i_class]
        num_samples = p_data.shape[0] // len_realp
        class_power_windows = np.asarray(
            p_data[:num_samples * len_realp].reshape(num_samples, len_realp, 1),
            dtype=np.float32,
        )

        split_payload = split_normal_meta_train_proxy_windows(
            [{"nwp": class_nwp_windows, "p": class_power_windows}],
            proxy_ratio=SELECTIVE_FED_META_PROXY_RATIO,
            seed=station_seed_base + i_class,
        )[0]

        meta_train_power_classes[0, i_class] = split_payload["meta_train_power"].reshape(-1, 1)
        for i_nwp in range(nwp_feature_count):
            meta_train_nwp_classes[0, i_nwp][0, i_class] = split_payload["meta_train_nwp"][:, :, i_nwp].reshape(-1, 1)

        if split_payload["proxy_nwp"].shape[0] > 0:
            proxy_nwp_windows_all.append(split_payload["proxy_nwp"])
            proxy_power_windows_all.append(split_payload["proxy_power"])

    if proxy_nwp_windows_all:
        proxy_input = np.concatenate(proxy_nwp_windows_all, axis=0).astype(np.float32)
        proxy_target = np.concatenate(proxy_power_windows_all, axis=0).astype(np.float32)
    else:
        proxy_input = np.empty((0, len_realp, dem_realc), dtype=np.float32)
        proxy_target = np.empty((0, len_realp, dem_realp), dtype=np.float32)

    return meta_train_power_classes, meta_train_nwp_classes, proxy_input, proxy_target


## data processing
seed_torch(seed=GLOBAL_SEED)
os.makedirs(ARTIFACT_DIR, exist_ok=True) if ARTIFACT_DIR not in ("", ".") else None
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_TRAIN_DIR, exist_ok=True)
print_protocol_banner()

# ========== [联邦修改] 多场站数据加载 ==========
station_ids = resolve_station_ids()
if USE_FEDERATION:
    print("="*70)
    print(f"联邦模式：加载{len(station_ids)}个场站数据（{', '.join(station_ids)}）")
    print("="*70)
else:
    print("="*70)
    print(f"单场站模式：加载场站{station_ids[0]}数据（原方法）")
    print("="*70)

# [联邦修改] 循环加载所有场站数据
station_data = {}
for station_id in station_ids:
    print(f"  加载 {resolve_station_mat_path(station_id)}...")
    wf_1 = scio.loadmat(resolve_station_mat_path(station_id))
    station_data[station_id] = {
        'p': wf_1['p_1h'],
        'p_conven_00': wf_1['p_conven'],
        'p_conven_class_00': wf_1['p_conven_class'],
        'p_test_00': wf_1.get('p_test'),
        'nwp': wf_1['nwp_1h'],
        'nwp_test_00': wf_1.get('nwp_test'),
        'nwp_conven_00': wf_1['nwp_conven_'],
        'nwp_conven_class_00': wf_1['nwp_conven_class_'],
        'extreme_support_power_by_class': [],
        'extreme_test_power_by_class': [],
        'extreme_support_nwp_by_class': [],
        'extreme_test_nwp_by_class': [],
    }
    for class_index in range(num_extreme_classes):
        support_power_key = f'p_extre_class{class_index + 1}'
        test_power_key = f'p_test_extre_class{class_index + 1}'
        support_nwp_key = f'nwp_extre_class{class_index + 1}_'
        test_nwp_key = f'nwp_test_extre_class{class_index + 1}_'
        station_data[station_id]['extreme_support_power_by_class'].append(wf_1[support_power_key])
        station_data[station_id]['extreme_test_power_by_class'].append(wf_1.get(test_power_key, wf_1[support_power_key]))
        station_data[station_id]['extreme_support_nwp_by_class'].append(wf_1[support_nwp_key])
        station_data[station_id]['extreme_test_nwp_by_class'].append(wf_1.get(test_nwp_key, wf_1[support_nwp_key]))

# ========== [联邦修改] 删除"主场站"概念，所有场站一视同仁 ==========
# 为第一个场站准备变量（用于参数初始化，后续会处理所有场站）
primary_station = station_ids[0]
p = station_data[primary_station]['p']
nwp = station_data[primary_station]['nwp']
p_conven_00 = station_data[primary_station]['p_conven_00']
nwp_conven_00 = station_data[primary_station]['nwp_conven_00']
p_conven_class_00 = station_data[primary_station]['p_conven_class_00']
nwp_conven_class_00 = station_data[primary_station]['nwp_conven_class_00']
P_load1=p[:,0]
P_load=P_load1.reshape(np.size(P_load1,axis=0),-1)
P_nwp1=nwp
nwp_index=[0,1,2,3,4]
for i in range(np.size(nwp_conven_class_00)):
    if i==0:
        P_nwp=P_nwp1[:,nwp_index[i]].reshape(np.size(P_nwp1,axis=0),-1)
    else:
        P_nwp=np.concatenate((P_nwp,P_nwp1[:,nwp_index[i]].reshape(np.size(P_nwp1,axis=0),-1)),axis=1)


# Define training equipment
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device0=torch.device("cpu")

# 模型文件路径（兼容保留）
PRETRAIN_MODEL_PATH = resolve_model_path("model_fore_pre_federated.pth" if USE_FEDERATION else "model_fore_pre.pth")
PROPOSED_SUPPORT_MODEL_PATH = resolve_model_path("model_fore_train_task_support_proposed.pth")
PROPOSED_META_MODEL_PATH = resolve_model_path("model_fore_train_task_query_proposed.pth")
META_ONLY_SUPPORT_MODEL_PATH = resolve_model_path("model_fore_train_task_support_meta_only.pth")
META_ONLY_MODEL_PATH = resolve_model_path("model_fore_train_task_query_meta_only.pth")


def get_local_pretrain_model_path(station_id):
    return resolve_base_model_path(f"model_fore_pre_station{station_id}_local.pth")


def get_local_meta_support_model_path(station_id):
    return resolve_base_model_path(f"model_fore_train_task_support_local_meta_station{station_id}.pth")


def get_local_meta_model_path(station_id):
    return resolve_base_model_path(f"model_fore_train_task_query_local_meta_station{station_id}.pth")


def get_target_aware_pretrain_model_path(station_id):
    if SKIP_TARGET_AWARE_PRETRAIN:
        return resolve_target_aware_base_model_path(f"model_fore_pre_station{station_id}_target_aware.pth")
    return resolve_model_path(f"model_fore_pre_station{station_id}_target_aware.pth")


def get_target_aware_meta_support_model_path(station_id):
    if SKIP_TARGET_AWARE_META:
        return resolve_target_aware_base_model_path(f"model_fore_train_task_support_target_aware_meta_station{station_id}.pth")
    return resolve_model_path(f"model_fore_train_task_support_target_aware_meta_station{station_id}.pth")


def get_target_aware_meta_model_path(station_id):
    if SKIP_TARGET_AWARE_META:
        return resolve_target_aware_base_model_path(f"model_fore_train_task_query_target_aware_meta_station{station_id}.pth")
    return resolve_model_path(f"model_fore_train_task_query_target_aware_meta_station{station_id}.pth")


def get_target_aware_selective_fed_meta_support_model_path(station_id):
    if SKIP_TARGET_AWARE_SELECTIVE_FED_META:
        return resolve_target_aware_selective_fed_base_model_path(
            f"model_fore_train_task_support_target_aware_selective_fed_meta_station{station_id}.pth"
        )
    return resolve_model_path(f"model_fore_train_task_support_target_aware_selective_fed_meta_station{station_id}.pth")


def get_target_aware_selective_fed_meta_model_path(station_id):
    if SKIP_TARGET_AWARE_SELECTIVE_FED_META:
        return resolve_target_aware_selective_fed_base_model_path(
            f"model_fore_train_task_query_target_aware_selective_fed_meta_station{station_id}.pth"
        )
    return resolve_model_path(f"model_fore_train_task_query_target_aware_selective_fed_meta_station{station_id}.pth")


def get_fed_normal_meta_support_model_path(station_id):
    return resolve_fed_normal_meta_model_path(f"model_fore_train_task_support_fed_normal_meta_station{station_id}.pth")


def get_fed_normal_meta_model_path(station_id):
    return resolve_fed_normal_meta_model_path(f"model_fore_train_task_query_fed_normal_meta_station{station_id}.pth")


def get_fed_normal_meta_best_support_model_path(station_id):
    return resolve_fed_normal_meta_model_path(f"model_fore_train_task_support_fed_normal_meta_best_station{station_id}.pth")


def get_fed_normal_meta_best_model_path(station_id):
    return resolve_fed_normal_meta_model_path(f"model_fore_train_task_query_fed_normal_meta_best_station{station_id}.pth")


def get_local_meta_only_support_model_path(station_id):
    return resolve_base_model_path(f"model_fore_train_task_support_meta_only_station{station_id}.pth")


def get_local_meta_only_model_path(station_id):
    return resolve_base_model_path(f"model_fore_train_task_query_meta_only_station{station_id}.pth")


def get_lmt_model_path(station_id, class_idx):
    return resolve_model_path(f"model_fore_station{station_id}_extreme{class_idx}.pth")


def get_meta_only_extreme_model_path(station_id, class_idx):
    return resolve_model_path(f"model_fore_station{station_id}_extreme{class_idx}_meta_only.pth")


def get_extreme_fedavg_model_path(station_id, class_idx):
    return resolve_model_path(f"model_fore_station{station_id}_extreme{class_idx}_extreme_fedavg.pth")


def get_proposed_a_model_path(station_id, class_idx):
    return resolve_model_path(f"model_fore_station{station_id}_extreme{class_idx}_proposed_a.pth")


def get_fed_meta_local_ft_model_path(station_id, class_idx):
    return resolve_model_path(f"model_fore_station{station_id}_extreme{class_idx}_fed_meta_local_ft.pth")


def get_fedtl_ft_model_path(station_id, class_idx):
    return resolve_model_path(f"model_fore_station{station_id}_extreme{class_idx}_fedtl_ft.pth")


def get_target_aware_selective_fed_local_ft_model_path(station_id, class_idx):
    return resolve_model_path(f"model_fore_station{station_id}_extreme{class_idx}_target_aware_selective_fed_local_ft.pth")


def load_stage_init_state(init_model_dir, filename, fallback_state_dict, stage_label):
    init_path = resolve_init_model_path(init_model_dir, filename)
    if init_path is None:
        return copy.deepcopy(fallback_state_dict)
    if not os.path.exists(init_path):
        raise FileNotFoundError(f"{stage_label} resume checkpoint missing: {init_path}")
    progress_log(f"✓ {stage_label} 从 checkpoint 初始化: {init_path}")
    return torch.load(init_path, map_location=device)


def get_local_extreme_base_model_path(station_id):
    return get_local_meta_model_path(station_id) if USE_FEDERATION and not USE_PSEUDO_FED else PROPOSED_META_MODEL_PATH


def get_proposed_a_base_model_path(station_id):
    if ENABLE_TARGET_AWARE_SELECTIVE_FED_META:
        return get_target_aware_selective_fed_meta_model_path(station_id)
    if ENABLE_FED_NORMAL_META_PROPOSED or ENABLE_SELECTIVE_FED_NORMAL_META:
        if FED_NORMAL_META_USE_BEST:
            return get_fed_normal_meta_best_model_path(station_id)
        return get_fed_normal_meta_model_path(station_id)
    return get_local_extreme_base_model_path(station_id)


# Define Parameters
dem_realp=1
len_realp = LEN_REALP
Cap=50  # 总装机容量 (MW)
m=365
d = POINTS_PER_DAY
ooo=365
# 数据已经是标幺值，不需要再归一化
Series_day = P_load.reshape(-1,dem_realp)
nwp_day = (P_nwp/np.max(abs(P_nwp),axis=0)).reshape(-1,dem_realp*np.size(P_nwp,axis=1))
dem_realc=np.size(P_nwp,axis=1)
p_conven_=p_conven_00  # 数据已经是标幺值
nwp_conven_=np.empty([1,5],dtype=object)
for i in range(np.size(nwp_conven_00,axis=1)):
    nwp_conven_[0,i]=nwp_conven_00[:,i].reshape(-1,1)/np.max(abs(P_nwp[:,i]),axis=0)
for i_nwp in range(np.size(nwp_conven_, axis=1)):
    if i_nwp == 0:
        nwp_conven_1 = nwp_conven_[0, i_nwp].transpose(1, 0)
        nwp_conven_1 = nwp_conven_1[:, :, np.newaxis]
    else:
        nwp_conven_0 = nwp_conven_[0, i_nwp].transpose(1, 0)
        nwp_conven_1 = np.concatenate((nwp_conven_1, nwp_conven_0[:, :, np.newaxis]), axis=2)
p_conven_1 = p_conven_.transpose(1, 0)
p_conven_1 = p_conven_1[:, :, np.newaxis]
train_input_c = nwp_conven_1  # 用于参数初始化
train_target_p = p_conven_1


def compute_hwa_weights_numpy_from_normalized(input_array, wind_scale):
    input_array = np.asarray(input_array, dtype=np.float32)
    weights = np.ones(input_array.shape[:-1] + (1,), dtype=np.float32)
    if HWA_WIND_FEATURE_INDEX < 0 or HWA_WIND_FEATURE_INDEX >= input_array.shape[-1]:
        return weights
    max_weight = max(1.0, float(HWA_HIGH_WIND_WEIGHT))
    if max_weight <= 1.0:
        return weights
    ramp_width = max(float(HWA_WIND_RAMP_END) - float(HWA_WIND_THRESHOLD), 1e-6)
    raw_wind = input_array[..., HWA_WIND_FEATURE_INDEX:HWA_WIND_FEATURE_INDEX + 1] * float(wind_scale)
    ramp = np.clip((raw_wind - float(HWA_WIND_THRESHOLD)) / ramp_width, 0.0, 1.0)
    return (1.0 + (max_weight - 1.0) * ramp).astype(np.float32)


def compute_hwa_weights_tensor_from_normalized(input_tensor, wind_scale):
    if HWA_WIND_FEATURE_INDEX < 0 or HWA_WIND_FEATURE_INDEX >= input_tensor.shape[-1]:
        return torch.ones_like(input_tensor[..., :1])
    max_weight = max(1.0, float(HWA_HIGH_WIND_WEIGHT))
    if max_weight <= 1.0:
        return torch.ones_like(input_tensor[..., :1])
    ramp_width = max(float(HWA_WIND_RAMP_END) - float(HWA_WIND_THRESHOLD), 1e-6)
    raw_wind = input_tensor[..., HWA_WIND_FEATURE_INDEX:HWA_WIND_FEATURE_INDEX + 1] * float(wind_scale)
    ramp = torch.clamp((raw_wind - float(HWA_WIND_THRESHOLD)) / ramp_width, min=0.0, max=1.0)
    return 1.0 + (max_weight - 1.0) * ramp


def weighted_mse_loss(outputs, targets, weights=None):
    if weights is None:
        return loss_fn_1(outputs, targets)
    weights = weights.to(device=outputs.device, dtype=outputs.dtype)
    return torch.sum(weights * (outputs - targets) ** 2) / torch.clamp(torch.sum(weights), min=1e-6)


def reshape_sequence_to_windows(array, feature_count):
    array = np.asarray(array, dtype=np.float32).reshape(-1, feature_count)
    window_count = int(array.shape[0] // len_realp)
    if window_count <= 0:
        return np.empty((0, len_realp, feature_count), dtype=np.float32)
    return array[:window_count * len_realp].reshape(window_count, len_realp, feature_count)


def build_windowed_pretrain_arrays(station_id):
    station_payload = clients_train_data[str(station_id)]
    return {
        'input': reshape_sequence_to_windows(station_payload['input'], dem_realc),
        'target': reshape_sequence_to_windows(station_payload['target'], dem_realp),
        'hwa_weight': reshape_sequence_to_windows(station_payload['hwa_weight'], dem_realp),
    }


def high_wind_window_mask_from_raw_wind(raw_wind_windows):
    raw_wind_windows = np.asarray(raw_wind_windows, dtype=np.float32)
    if raw_wind_windows.size == 0:
        return np.zeros((0,), dtype=bool)
    mean_wind = np.mean(raw_wind_windows, axis=1)
    max_wind = np.max(raw_wind_windows, axis=1)
    high_point_count = np.sum(raw_wind_windows > TARGET_AWARE_META_WIND_MEAN_THRESHOLD, axis=1)
    cut_point_count = np.sum(raw_wind_windows > TARGET_AWARE_META_WIND_MAX_THRESHOLD, axis=1)
    return (
        (mean_wind > TARGET_AWARE_META_WIND_MEAN_THRESHOLD)
        | (max_wind > TARGET_AWARE_META_WIND_MAX_THRESHOLD)
        | (high_point_count >= TARGET_AWARE_META_MIN_EXTREME_POINTS)
        | (cut_point_count >= 1)
    )


def compute_window_feature_summary(input_windows, power_windows, wind_scale):
    input_windows = np.asarray(input_windows, dtype=np.float32)
    power_windows = np.asarray(power_windows, dtype=np.float32)
    if input_windows.shape[0] == 0:
        return {
            'mean_wind': TARGET_AWARE_META_WIND_MEAN_THRESHOLD,
            'max_wind': TARGET_AWARE_META_WIND_MAX_THRESHOLD,
            'high_ratio': 0.0,
            'power_mean': 0.0,
            'power_ramp': 0.0,
        }
    raw_wind = input_windows[..., HWA_WIND_FEATURE_INDEX] * float(wind_scale)
    power_values = power_windows[..., 0]
    high_mask = high_wind_window_mask_from_raw_wind(raw_wind)
    if raw_wind.shape[0] > 0:
        ramp_values = np.abs(np.diff(power_values, axis=1)) if power_values.shape[1] > 1 else np.zeros_like(power_values)
    else:
        ramp_values = np.zeros((0,), dtype=np.float32)
    return {
        'mean_wind': float(np.mean(raw_wind)),
        'max_wind': float(np.max(raw_wind)),
        'high_ratio': float(np.mean(high_mask)) if high_mask.size else 0.0,
        'power_mean': float(np.mean(power_values)) if power_values.size else 0.0,
        'power_ramp': float(np.mean(ramp_values)) if np.size(ramp_values) else 0.0,
    }


def closeness_score(value, target_value, scale):
    scale = max(float(scale), 1e-6)
    return float(math.exp(-abs(float(value) - float(target_value)) / scale))


target_aware_task_score_cache = {}


def build_extreme_support_windows(station_id, class_idx=0):
    station_id = str(station_id)
    station_payload = all_stations_full_data[station_id]
    nwp_extre_st = station_payload['nwp_extre']
    p_extre_st = station_payload['p_extre']
    nwp_windows = None
    for i_nwp in range(np.size(nwp_extre_st, axis=1)):
        nwp_data = nwp_extre_st[0, i_nwp][0, class_idx]
        nwp_feature_windows = reshape_sequence_to_windows(nwp_data, 1)
        if nwp_windows is None:
            nwp_windows = nwp_feature_windows
        else:
            nwp_windows = np.concatenate((nwp_windows, nwp_feature_windows), axis=2)
    power_windows = reshape_sequence_to_windows(p_extre_st[0, class_idx], dem_realp)
    window_count = min(nwp_windows.shape[0], power_windows.shape[0])
    return nwp_windows[:window_count], power_windows[:window_count]


def get_target_aware_task_score(station_id, class_idx, target_station_id=None):
    station_id = str(station_id)
    target_station_id = str(target_station_id or station_id)
    cache_key = (station_id, int(class_idx), target_station_id)
    if cache_key in target_aware_task_score_cache:
        return target_aware_task_score_cache[cache_key]

    nwp_conven_class_st = all_stations_full_data[station_id]['nwp_conven_class']
    p_conven_class_st = all_stations_full_data[station_id]['p_conven_class']
    task_nwp_windows = None
    for i_nwp in range(np.size(nwp_conven_class_st, axis=1)):
        nwp_data = nwp_conven_class_st[0, i_nwp][0, class_idx]
        nwp_feature_windows = reshape_sequence_to_windows(nwp_data, 1)
        if task_nwp_windows is None:
            task_nwp_windows = nwp_feature_windows
        else:
            task_nwp_windows = np.concatenate((task_nwp_windows, nwp_feature_windows), axis=2)
    task_power_windows = reshape_sequence_to_windows(p_conven_class_st[0, class_idx], dem_realp)
    target_nwp_windows, target_power_windows = build_extreme_support_windows(target_station_id, class_idx=0)
    task_summary = compute_window_feature_summary(
        task_nwp_windows,
        task_power_windows,
        wind_scale=get_station_hwa_wind_scale(station_id),
    )
    target_summary = compute_window_feature_summary(
        target_nwp_windows,
        target_power_windows,
        wind_scale=get_station_hwa_wind_scale(station_id),
    )
    score = (
        0.35 * closeness_score(task_summary['mean_wind'], target_summary['mean_wind'], 4.0)
        + 0.25 * closeness_score(task_summary['max_wind'], target_summary['max_wind'], 5.0)
        + 0.20 * task_summary['high_ratio']
        + 0.10 * closeness_score(task_summary['power_mean'], target_summary['power_mean'], 0.25)
        + 0.10 * closeness_score(task_summary['power_ramp'], target_summary['power_ramp'], 0.15)
    )
    score = float(min(max(score, 0.0), 1.0))
    target_aware_task_score_cache[cache_key] = score
    return score


def weighted_sample_tasks(task_pool, task_weights, sample_count):
    if sample_count >= len(task_pool):
        return list(task_pool)
    weights = np.asarray(task_weights, dtype=np.float64)
    if not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
        return random.sample(task_pool, sample_count)
    probabilities = weights / np.sum(weights)
    selected_indices = np.random.choice(
        np.arange(len(task_pool)),
        size=sample_count,
        replace=False,
        p=probabilities,
    )
    return [task_pool[int(index)] for index in selected_indices]


def get_station_hwa_wind_scale(station_id):
    return float(all_stations_full_data[str(station_id)].get("hwa_wind_scale", 1.0))


# ========== [联邦新增] 准备所有场站的数据（元训练、测试都用） ==========
print("\n" + "="*70)
print("准备所有场站的完整数据（元训练+极端天气+测试集）")
print("="*70)

# 准备所有客户端的常规天气数据（已在预训练中准备）
# ========== [联邦新增] 准备所有客户端的常规天气数据 ==========
if USE_FEDERATION:
    print("\n准备联邦客户端数据（常规天气）...")
    clients_train_data = {}
    
    for station_id in station_ids:
        print(f"  处理场站 {station_id}...")
        # [原代码逻辑] 每个场站使用相同的处理流程
        p_st = station_data[station_id]['p']
        nwp_st = station_data[station_id]['nwp']
        P_nwp1_st = nwp_st
        
        # 构建P_nwp_st（用于归一化）
        for i in range(5):
            if i==0:
                P_nwp_st = P_nwp1_st[:,nwp_index[i]].reshape(np.size(P_nwp1_st,axis=0),-1)
            else:
                P_nwp_st = np.concatenate((P_nwp_st,P_nwp1_st[:,nwp_index[i]].reshape(np.size(P_nwp1_st,axis=0),-1)),axis=1)
        
        # 处理常规天气数据
        p_conven_st = station_data[station_id]['p_conven_00']
        nwp_conven_st = station_data[station_id]['nwp_conven_00']
        
        p_conven_st_ = p_conven_st
        nwp_conven_st_ = np.empty([1,5],dtype=object)
        for i in range(np.size(nwp_conven_st,axis=1)):
            nwp_conven_st_[0,i] = nwp_conven_st[:,i].reshape(-1,1)/np.max(abs(P_nwp_st[:,i]),axis=0)
        
        for i_nwp in range(np.size(nwp_conven_st_, axis=1)):
            if i_nwp == 0:
                nwp_conven_1_st = nwp_conven_st_[0, i_nwp].transpose(1, 0)
                nwp_conven_1_st = nwp_conven_1_st[:, :, np.newaxis]
            else:
                nwp_conven_0_st = nwp_conven_st_[0, i_nwp].transpose(1, 0)
                nwp_conven_1_st = np.concatenate((nwp_conven_1_st, nwp_conven_0_st[:, :, np.newaxis]), axis=2)
        
        p_conven_1_st = p_conven_st_.transpose(1, 0)
        p_conven_1_st = p_conven_1_st[:, :, np.newaxis]
        hwa_wind_scale_st = float(np.max(abs(P_nwp_st[:, HWA_WIND_FEATURE_INDEX]), axis=0)) if 0 <= HWA_WIND_FEATURE_INDEX < P_nwp_st.shape[1] else 1.0
        hwa_pretrain_weight_st = compute_hwa_weights_numpy_from_normalized(
            nwp_conven_1_st,
            wind_scale=hwa_wind_scale_st,
        )
        
        # [联邦新增] 存储客户端数据
        clients_train_data[station_id] = {
            'input': nwp_conven_1_st,
            'target': p_conven_1_st,
            'hwa_weight': hwa_pretrain_weight_st,
            'hwa_wind_scale': hwa_wind_scale_st,
        }
        print(f"    shape: {nwp_conven_1_st.shape} → {p_conven_1_st.shape}")
    
    print(f"  总数据量: {sum([clients_train_data[s]['input'].shape[1] for s in station_ids])} 样本")
# ========== [联邦新增] 结束 ==========

# ========== [联邦修改] 准备所有场站的元训练、极端天气和测试数据 ==========
print(f"\n准备所有场站的元训练和测试数据（{len(station_ids)}场站一视同仁）...")
all_stations_full_data = {}

for station_id in station_ids:
    print(f"\n  场站 {station_id}:")
    
    # 获取该场站数据
    p_st = station_data[station_id]['p']
    nwp_st = station_data[station_id]['nwp']
    
    P_load1_st = p_st[:,0]
    P_load_st = P_load1_st.reshape(np.size(P_load1_st,axis=0),-1)
    P_nwp1_st = nwp_st
    
    for i in range(5):
        if i==0:
            P_nwp_st = P_nwp1_st[:,nwp_index[i]].reshape(np.size(P_nwp1_st,axis=0),-1)
        else:
            P_nwp_st = np.concatenate((P_nwp_st,P_nwp1_st[:,nwp_index[i]].reshape(np.size(P_nwp1_st,axis=0),-1)),axis=1)
    
    Series_day_st = P_load_st.reshape(-1,dem_realp)
    nwp_day_st = (P_nwp_st/np.max(abs(P_nwp_st),axis=0)).reshape(-1,dem_realp*np.size(P_nwp_st,axis=1))
    
    # 测试集（2023年）：yearly protocol 直接使用 builder 输出的 p_test/nwp_test。
    if (
        (SEASONAL_PROTOCOL_ENABLED or YEARLY_PROTOCOL_ENABLED)
        and station_data[station_id]['p_test_00'] is not None
        and station_data[station_id]['nwp_test_00'] is not None
    ):
        test_target_p_st = station_data[station_id]['p_test_00']
        test_nwp_st = station_data[station_id]['nwp_test_00']
        test_target_p_st, test_input_c_st = reshape_complete_window_pairs(
            power_array=test_target_p_st,
            nwp_array=(test_nwp_st/np.max(abs(P_nwp_st),axis=0)),
            len_realp=len_realp,
            dem_realp=dem_realp,
            dem_realc=dem_realc,
        )
    else:
        test_target_p_st, test_input_c_st = reshape_complete_window_pairs(
            power_array=Series_day_st[m*d//dem_realp:(m*d+ooo*d)//dem_realp,:],
            nwp_array=nwp_day_st[m*d//dem_realp:(m*d+ooo*d)//dem_realp,:],
            len_realp=len_realp,
            dem_realp=dem_realp,
            dem_realc=dem_realc,
        )
    
    # 聚类类别（用于元训练）
    p_conven_class_st = station_data[station_id]['p_conven_class_00']
    nwp_conven_class_st = station_data[station_id]['nwp_conven_class_00'].copy()
    for i in range(np.size(nwp_conven_class_st,axis=1)):
        nwp_conven_class_st[0,i] = nwp_conven_class_st[0,i]/np.max(abs(P_nwp_st[:,i]),axis=0)
    if ENABLE_SELECTIVE_FED_NORMAL_META:
        (
            p_conven_class_st,
            nwp_conven_class_st,
            normal_proxy_input_st,
            normal_proxy_target_st,
        ) = build_station_normal_meta_train_proxy_payloads(
            p_conven_class_st=p_conven_class_st,
            nwp_conven_class_st=nwp_conven_class_st,
            station_id=station_id,
        )
    else:
        normal_proxy_input_st = np.empty((0, len_realp, dem_realc), dtype=np.float32)
        normal_proxy_target_st = np.empty((0, len_realp, dem_realp), dtype=np.float32)
    
    # 极端天气类别（用于Few-shot）
    p_extre_st = np.empty([1, num_extreme_classes],dtype=object)
    nwp_extre_st = np.empty([1,5],dtype=object)
    p_test_extre_st = np.empty([1, num_extreme_classes],dtype=object)
    nwp_test_extre_st = np.empty([1,5],dtype=object)
    
    for i_class in range(num_extreme_classes):
        p_extre_st[0,i_class] = station_data[station_id]['extreme_support_power_by_class'][i_class]
        p_test_extre_st[0,i_class] = station_data[station_id]['extreme_test_power_by_class'][i_class]
    
    for i_nwp in range(5):
        nwp_extre_st[0,i_nwp] = np.empty([1, num_extreme_classes],dtype=object)
        nwp_test_extre_st[0,i_nwp] = np.empty([1, num_extreme_classes],dtype=object)
        for i_class in range(num_extreme_classes):
            nwp_extre_st[0, i_nwp][0, i_class] = station_data[station_id]['extreme_support_nwp_by_class'][i_class][0, i_nwp]
            nwp_test_extre_st[0, i_nwp][0, i_class] = station_data[station_id]['extreme_test_nwp_by_class'][i_class][0, i_nwp]
    
    for i in range(np.size(nwp_extre_st,axis=1)):
        nwp_extre_st[0,i] = nwp_extre_st[0,i]/np.max(abs(P_nwp_st[:,i]),axis=0)
        nwp_test_extre_st[0,i] = nwp_test_extre_st[0,i]/np.max(abs(P_nwp_st[:,i]),axis=0)

    if (SEASONAL_PROTOCOL_ENABLED or YEARLY_PROTOCOL_ENABLED):
        p_test_extre_source = p_test_extre_st
        nwp_test_extre_source = nwp_test_extre_st
    else:
        p_test_extre_source = p_extre_st
        nwp_test_extre_source = nwp_extre_st
    
    # 存储该场站的完整数据
    all_stations_full_data[station_id] = {
        'P_nwp': P_nwp_st,
        'hwa_wind_scale': float(np.max(abs(P_nwp_st[:, HWA_WIND_FEATURE_INDEX]), axis=0)) if 0 <= HWA_WIND_FEATURE_INDEX < P_nwp_st.shape[1] else 1.0,
        'test_input': test_input_c_st,
        'test_target': test_target_p_st,
        'p_conven_class': p_conven_class_st,
        'nwp_conven_class': nwp_conven_class_st,
        'normal_proxy_input': normal_proxy_input_st,
        'normal_proxy_target': normal_proxy_target_st,
        'p_extre': p_extre_st,
        'nwp_extre': nwp_extre_st,
        'p_test_extre': p_test_extre_st,
        'nwp_test_extre': nwp_test_extre_st
    }
    total_station_classes = np.size(p_conven_class_st, axis=1)
    test_year_label = "-".join(str(year) for year in TEST_YEARS)
    print(f"    测试集{test_year_label}: {test_target_p_st.shape}")
    print(f"    聚类类别: {total_station_classes}类")
    print(f"    极端天气: {num_extreme_classes}类")

print(f"\n✓ 所有场站数据准备完成！")

# [保留] 为了兼容部分原代码，保留变量
test_target_p = all_stations_full_data[station_ids[0]]['test_target']
test_input_c = all_stations_full_data[station_ids[0]]['test_input']
p_conven_class_00 = all_stations_full_data[station_ids[0]]['p_conven_class']
nwp_conven_class_00 = all_stations_full_data[station_ids[0]]['nwp_conven_class']
P_nwp = all_stations_full_data[station_ids[0]]['P_nwp']
p_conven_class_ = p_conven_class_00
nwp_conven_class_ = nwp_conven_class_00


## Define forecasting model
class model_fore(nn.Module):
    def __init__(self,input_channel_fore, output_channel_fore, mode, output_size_baselearner=1,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super().__init__()
        self.mode = mode
        self.tcn = TemporalConvNet(input_channel_fore, output_channel_fore, mode, kernel_size, dropout)
        self.emb_dropout = emb_dropout
        self.drop = nn.Dropout(emb_dropout)
        self.fore_baselearner = nn.Linear(output_channel_fore[-1], output_size_baselearner)
        self.init_weights()
    def init_weights(self):
        self.fore_baselearner.bias.data.fill_(0)
        self.fore_baselearner.weight.data.normal_(0, 0.01)
    def get_trainable_params(self):
        """根据mode返回需要训练的参数"""
        if self.mode == 'pre':
            # 预训练：训练所有参数
            return self.parameters()
        else:
            # 元学习：只训练TCN中的LWP层和最后的预测层
            trainable_params = []
            for module in self.tcn.modules():
                if hasattr(module, 'get_trainable_params'):
                    trainable_params.extend(module.get_trainable_params())
            trainable_params.extend(self.fore_baselearner.parameters())
            return trainable_params
    def forward(self, x):
        y = self.drop(x)
        y = self.tcn(y.transpose(1, 2)).transpose(1, 2)
        y = self.fore_baselearner(y)
        return y.contiguous()
model_fore_pre = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8],mode='pre')
model_fore_train_task_support = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8],mode='train_task_support')
model_fore_train_task_query = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8],mode='train_task_query')
model_fore_test_task_support = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8],mode='test_task_support')
model_fore_test_task_query = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8],mode='test_task_support')
# 保存一份随机初始化权重，供 meta-learning only 使用（不经过 pre-train）
pretrain_random_init_state = copy.deepcopy(model_fore_pre.state_dict())
meta_only_random_init_state = copy.deepcopy(model_fore_train_task_query.state_dict())


## Define loss
loss_fn_1=nn.MSELoss()
def penalty(logits, y):
    scale = torch.tensor(1., device=device).requires_grad_()
    loss1 = loss_fn_1(logits[0::2] * scale, y[0::2])
    loss2 = loss_fn_1(logits[1::2] * scale, y[1::2])
    # grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    # return torch.sum(grad ** 2)
    grad_batch1 = torch.autograd.grad(loss1, [scale], create_graph=True)[0]
    grad_batch2 = torch.autograd.grad(loss2, [scale], create_graph=True)[0]
    return torch.sum(grad_batch1 * grad_batch2)


## Define optimizer
optimizer_fore_pre = torch.optim.Adam(model_fore_pre.get_trainable_params(), lr=0.0002, betas=(0.5, 0.999))
optimizer_fore_train_task_support = torch.optim.Adam(model_fore_train_task_support.get_trainable_params(), lr=0.0002, betas=(0.5, 0.999))
optimizer_fore_train_task_query = torch.optim.Adam(model_fore_train_task_query.get_trainable_params(), lr=0.0002, betas=(0.5, 0.999))
optimizer_fore_test_task_support = torch.optim.Adam(model_fore_test_task_support.get_trainable_params(), lr=0.0002, betas=(0.5, 0.999))

# [原代码保留] 主场站数据转为Tensor
Train_target_p=torch.tensor(train_target_p,dtype=torch.float32)
Train_input_c=torch.tensor(train_input_c,dtype=torch.float32)
Test_target_p=torch.tensor(test_target_p,dtype=torch.float32)
Test_input_c=torch.tensor(test_input_c,dtype=torch.float32)

# ========== [联邦新增] 所有客户端数据转为Tensor ==========
if USE_FEDERATION:
    clients_train_tensor = {}
    for station_id in station_ids:
        windowed_pretrain_arrays = build_windowed_pretrain_arrays(station_id)
        clients_train_tensor[station_id] = {
            'input': torch.tensor(clients_train_data[station_id]['input'], dtype=torch.float32),
            'target': torch.tensor(clients_train_data[station_id]['target'], dtype=torch.float32),
            'hwa_weight': torch.tensor(clients_train_data[station_id]['hwa_weight'], dtype=torch.float32),
            'input_windowed': torch.tensor(windowed_pretrain_arrays['input'], dtype=torch.float32),
            'target_windowed': torch.tensor(windowed_pretrain_arrays['target'], dtype=torch.float32),
            'hwa_weight_windowed': torch.tensor(windowed_pretrain_arrays['hwa_weight'], dtype=torch.float32),
        }
    task_num = len(station_ids)  # [联邦新增] 客户端数量=3
else:
    task_num = 1  # [原代码] 单场站
# ========== [联邦新增] 结束 ==========

model_fore_pre = model_fore_pre.to(device)
model_fore_train_task_support = model_fore_train_task_support.to(device)
model_fore_train_task_query = model_fore_train_task_query.to(device)
model_fore_test_task_support = model_fore_test_task_support.to(device)
model_fore_test_task_query = model_fore_test_task_query.to(device)


## pre-train
if USE_FEDERATION and (USE_PSEUDO_FED or RUN_FEDERATED_PRETRAIN):
    print("#########################################################################——————————联邦预训练（Federation Pre-train）——————————############################################################")
    print(f"客户端数量: {task_num}, 场站: {', '.join(station_ids)}")
elif USE_FEDERATION:
    print("#########################################################################——————————本地预训练（Local Pre-train）——————————############################################################")
    print(f"场站数量: {task_num}, 场站: {', '.join(station_ids)}")
else:
    print("#########################################################################——————————预训练（Pre-train）——————————############################################################")

total_train_step = 0
total_test_step = 0
epoch1_pre = PRETRAIN_EPOCHS
writer1 = SummaryWriter(os.path.join(LOGS_TRAIN_DIR, "loss1"))
writer2 = SummaryWriter(os.path.join(LOGS_TRAIN_DIR, "loss2"))


def get_pretrain_penalty_weight(epoch_idx):
    if epoch_idx < 10000:
        return 0
    if epoch_idx < 20000:
        return 1
    if epoch_idx < 30000:
        return 5
    return 10


def run_single_pretrain_epoch(train_input, train_target, optimizer, penalty_weight, hwa_weight=None, use_hwa_loss=None):
    if use_hwa_loss is None:
        use_hwa_loss = HWA_PRETRAIN_LOSS
    model_fore_pre.train()
    train_input = train_input.to(device)
    train_target = train_target.to(device)
    hwa_weight = hwa_weight.to(device) if hwa_weight is not None else None
    train_outputs_pre = model_fore_pre(train_input)
    loss1 = penalty(train_outputs_pre, train_target)
    loss2 = weighted_mse_loss(
        train_outputs_pre,
        train_target,
        hwa_weight if use_hwa_loss else None,
    )
    loss_en = penalty_weight * loss1 + loss2
    optimizer.zero_grad()
    loss_en.backward()
    optimizer.step()
    return loss1.item(), loss2.item()


def compute_pretrain_eval_losses(penalty_weight):
    model_fore_pre.eval()
    total_loss1 = 0.0
    total_loss2 = 0.0
    with torch.no_grad():
        for station_id in station_ids:
            train_target = clients_train_tensor[station_id]['target'].to(device)
            train_input = clients_train_tensor[station_id]['input'].to(device)
            train_hwa_weight = clients_train_tensor[station_id]['hwa_weight'].to(device)
            train_outputs_pre = model_fore_pre(train_input)
            loss2 = weighted_mse_loss(
                train_outputs_pre,
                train_target,
                train_hwa_weight if HWA_PRETRAIN_LOSS else None,
            )
            total_loss1 += loss2.item()
            total_loss2 += loss2.item()
    return total_loss1 / task_num, total_loss2 / task_num


def aggregate_client_states(client_states, client_weights):
    weight_sum = float(sum(client_weights))
    if weight_sum <= 0:
        raise ValueError("FedAvg client weights must sum to a positive value")
    aggregate_state = copy.deepcopy(client_states[0])
    for name in aggregate_state.keys():
        reference_tensor = aggregate_state[name]
        if not torch.is_tensor(reference_tensor) or not torch.is_floating_point(reference_tensor):
            aggregate_state[name] = copy.deepcopy(reference_tensor)
            continue
        weighted_sum = None
        for state_dict, weight in zip(client_states, client_weights):
            contribution = state_dict[name].detach().float().cpu() * (float(weight) / weight_sum)
            weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
        aggregate_state[name] = weighted_sum.type_as(reference_tensor)
    return aggregate_state


def run_fedavg_pretraining_round(global_state, penalty_weight):
    client_states = []
    client_weights = []
    local_loss1_total = 0.0
    local_loss2_total = 0.0
    for station_id in station_ids:
        model_fore_pre.load_state_dict(copy.deepcopy(global_state))
        optimizer_local = torch.optim.Adam(
            model_fore_pre.get_trainable_params(), lr=FEDAVG_LR, betas=(0.5, 0.999)
        )
        station_loss1 = 0.0
        station_loss2 = 0.0
        train_input = clients_train_tensor[station_id]['input']
        train_target = clients_train_tensor[station_id]['target']
        train_hwa_weight = clients_train_tensor[station_id]['hwa_weight']
        for _ in range(FEDAVG_LOCAL_EPOCHS):
            loss1_value, loss2_value = run_single_pretrain_epoch(
                train_input,
                train_target,
                optimizer_local,
                penalty_weight,
                hwa_weight=train_hwa_weight,
                use_hwa_loss=HWA_PRETRAIN_LOSS,
            )
            station_loss1 += loss1_value
            station_loss2 += loss2_value
        client_states.append(copy.deepcopy(model_fore_pre.state_dict()))
        if FEDAVG_CLIENT_WEIGHTING == "sample":
            client_weights.append(float(max(1, train_input.shape[0])))
        else:
            client_weights.append(1.0)
        local_loss1_total += station_loss1 / FEDAVG_LOCAL_EPOCHS
        local_loss2_total += station_loss2 / FEDAVG_LOCAL_EPOCHS
    return aggregate_client_states(client_states, client_weights), local_loss1_total / task_num, local_loss2_total / task_num


def run_federated_pretraining():
    start_time = time.time()
    pretrain_convergence_record = initialize_convergence_record(
        stage_type="federated_pretrain",
        stage_id="all_stations",
        total_epochs=epoch1_pre,
        patience=CONVERGENCE_PATIENCE_PRETRAIN,
    )
    progress_log(
        "联邦预训练算法: "
        f"{FED_PRETRAIN_ALGO} "
        f"(local_epochs={FEDAVG_LOCAL_EPOCHS}, weighting={FEDAVG_CLIENT_WEIGHTING})"
    )
    for i in range(epoch1_pre):
        penalty_weight = get_pretrain_penalty_weight(i)
        if FED_PRETRAIN_ALGO == "fedavg":
            global_state = copy.deepcopy(model_fore_pre.state_dict())
            aggregate_state, local_loss1, local_loss2 = run_fedavg_pretraining_round(global_state, penalty_weight)
            model_fore_pre.load_state_dict(aggregate_state)
            loss1_display, loss2_display = compute_pretrain_eval_losses(penalty_weight)
        else:
            model_fore_pre.train()
            optimizer_fore_pre.zero_grad()
            total_loss1 = 0.0
            total_loss2 = 0.0

            for station_id in station_ids:
                train_target = clients_train_tensor[station_id]['target'].to(device)
                train_input = clients_train_tensor[station_id]['input'].to(device)
                train_hwa_weight = clients_train_tensor[station_id]['hwa_weight'].to(device)
                train_outputs_pre = model_fore_pre(train_input)
                loss1 = penalty(train_outputs_pre, train_target)
                loss2 = weighted_mse_loss(
                    train_outputs_pre,
                    train_target,
                    train_hwa_weight if HWA_PRETRAIN_LOSS else None,
                )
                loss_en = penalty_weight * loss1 + loss2
                (loss_en / task_num).backward()
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()

            optimizer_fore_pre.step()
            loss1_display = total_loss1 / task_num
            loss2_display = total_loss2 / task_num

        update_convergence_record(pretrain_convergence_record, i, loss2_display)

        if should_log_epoch(i, epoch1_pre, interval=PRETRAIN_LOG_INTERVAL):
            end_time = time.time()
            print(end_time - start_time)
            print("[Epoch %d/%d] [loss_mse: %f] " % (i, epoch1_pre, loss2_display))
            writer1.add_scalar("loss_mse_pre", loss1_display, i)
            writer2.add_scalar("loss_mse_pre", loss2_display, i)

    register_convergence_record(pretrain_convergence_record)
    model_fore_pre.eval()
    torch.save(model_fore_pre.state_dict(), PRETRAIN_MODEL_PATH)
    print(f"\n✓ 联邦预训练完成: {PRETRAIN_MODEL_PATH}")


def run_local_pretraining():
    for station_id in station_ids:
        print("\n" + "=" * 70)
        print(f"开始本地预训练: station {station_id}")
        print("=" * 70)
        save_path = get_local_pretrain_model_path(station_id)
        if SKIP_LOCAL_PRETRAIN:
            if not os.path.exists(save_path):
                raise FileNotFoundError(
                    f"SKIP_LOCAL_PRETRAIN=1 但未找到已有 checkpoint: {save_path}"
                )
            progress_log(f"✓ 跳过本地预训练，复用已有 checkpoint: {save_path}")
            continue

        init_state = load_stage_init_state(
            LOCAL_PRETRAIN_INIT_MODEL_DIR,
            f"model_fore_pre_station{station_id}_local.pth",
            pretrain_random_init_state,
            f"local pretrain station {station_id}",
        )
        model_fore_pre.load_state_dict(init_state)
        optimizer_local = torch.optim.Adam(
            model_fore_pre.get_trainable_params(), lr=0.0002, betas=(0.5, 0.999)
        )
        start_time = time.time()
        epoch_offset = max(0, int(LOCAL_PRETRAIN_EPOCH_OFFSET))
        total_epochs = epoch_offset + epoch1_pre
        pretrain_convergence_record = initialize_convergence_record(
            stage_type="local_pretrain",
            stage_id=station_id,
            total_epochs=total_epochs,
            patience=CONVERGENCE_PATIENCE_PRETRAIN,
        )
        train_input = clients_train_tensor[station_id]['input']
        train_target = clients_train_tensor[station_id]['target']
        train_hwa_weight = clients_train_tensor[station_id]['hwa_weight']

        for i in range(epoch1_pre):
            global_epoch = epoch_offset + i
            penalty_weight = get_pretrain_penalty_weight(global_epoch)
            loss1_display, loss2_display = run_single_pretrain_epoch(
                train_input,
                train_target,
                optimizer_local,
                penalty_weight,
                hwa_weight=train_hwa_weight,
                use_hwa_loss=TARGET_AWARE_PRETRAIN_HWA_LOSS,
            )
            update_convergence_record(pretrain_convergence_record, global_epoch, loss2_display)

            if should_log_epoch(i, epoch1_pre, interval=PRETRAIN_LOG_INTERVAL):
                end_time = time.time()
                print(end_time - start_time)
                print(
                    f"[station {station_id}] [Epoch {global_epoch + 1}/{total_epochs}] [loss_mse: {loss2_display:.6f}] "
                )
                writer1.add_scalar(f"loss_mse_pre_station{station_id}", loss1_display, global_epoch)
                writer2.add_scalar(f"loss_mse_pre_station{station_id}", loss2_display, global_epoch)

        register_convergence_record(pretrain_convergence_record)
        model_fore_pre.eval()
        torch.save(model_fore_pre.state_dict(), save_path)
        print(f"✓ 本地预训练完成: {save_path}")


def run_target_aware_pretraining():
    if not ENABLE_TARGET_AWARE_META_NOFT:
        return
    if not (USE_FEDERATION and not USE_PSEUDO_FED):
        raise RuntimeError("Target-Aware Meta-NoFT currently requires local multi-station mode")
    for station_id in station_ids:
        print("\n" + "=" * 70)
        print(f"开始Target-Aware窗口化预训练: station {station_id}")
        print("=" * 70)
        save_path = get_target_aware_pretrain_model_path(station_id)
        if SKIP_TARGET_AWARE_PRETRAIN:
            if not os.path.exists(save_path):
                raise FileNotFoundError(
                    f"SKIP_TARGET_AWARE_PRETRAIN=1 但未找到已有 checkpoint: {save_path}"
                )
            progress_log(f"✓ 跳过Target-Aware预训练，复用已有 checkpoint: {save_path}")
            continue

        init_state = load_stage_init_state(
            TARGET_AWARE_PRETRAIN_INIT_MODEL_DIR,
            f"model_fore_pre_station{station_id}_target_aware.pth",
            pretrain_random_init_state,
            f"target-aware pretrain station {station_id}",
        )
        model_fore_pre.load_state_dict(init_state)
        optimizer_local = torch.optim.Adam(
            model_fore_pre.get_trainable_params(), lr=0.0002, betas=(0.5, 0.999)
        )
        start_time = time.time()
        epoch_offset = max(0, int(TARGET_AWARE_PRETRAIN_EPOCH_OFFSET))
        total_epochs = epoch_offset + epoch1_pre
        pretrain_convergence_record = initialize_convergence_record(
            stage_type="target_aware_pretrain",
            stage_id=station_id,
            total_epochs=total_epochs,
            patience=CONVERGENCE_PATIENCE_PRETRAIN,
        )
        if HWA_PRETRAIN_WINDOWED:
            train_input = clients_train_tensor[station_id]['input_windowed']
            train_target = clients_train_tensor[station_id]['target_windowed']
            train_hwa_weight = clients_train_tensor[station_id]['hwa_weight_windowed']
        else:
            train_input = clients_train_tensor[station_id]['input']
            train_target = clients_train_tensor[station_id]['target']
            train_hwa_weight = clients_train_tensor[station_id]['hwa_weight']

        for i in range(epoch1_pre):
            global_epoch = epoch_offset + i
            penalty_weight = get_pretrain_penalty_weight(global_epoch)
            loss1_display, loss2_display = run_single_pretrain_epoch(
                train_input,
                train_target,
                optimizer_local,
                penalty_weight,
                hwa_weight=train_hwa_weight,
            )
            update_convergence_record(pretrain_convergence_record, global_epoch, loss2_display)

            if should_log_epoch(i, epoch1_pre, interval=PRETRAIN_LOG_INTERVAL):
                end_time = time.time()
                print(end_time - start_time)
                print(
                    f"[target-aware station {station_id}] [Epoch {global_epoch + 1}/{total_epochs}] "
                    f"[loss_mse: {loss2_display:.6f}] "
                )
                writer1.add_scalar(f"loss_mse_target_aware_pre_station{station_id}", loss1_display, global_epoch)
                writer2.add_scalar(f"loss_mse_target_aware_pre_station{station_id}", loss2_display, global_epoch)

        register_convergence_record(pretrain_convergence_record)
        model_fore_pre.eval()
        torch.save(model_fore_pre.state_dict(), save_path)
        print(f"✓ Target-Aware窗口化预训练完成: {save_path}")


if USE_FEDERATION and (USE_PSEUDO_FED or RUN_FEDERATED_PRETRAIN):
    run_federated_pretraining()
elif USE_FEDERATION:
    run_local_pretraining()
else:
    start_time = time.time()
    pretrain_convergence_record = initialize_convergence_record(
        stage_type="local_pretrain",
        stage_id="single_station",
        total_epochs=epoch1_pre,
        patience=CONVERGENCE_PATIENCE_PRETRAIN,
    )
    for i in range(epoch1_pre):
        penalty_weight = get_pretrain_penalty_weight(i)
        loss1_display, loss2_display = run_single_pretrain_epoch(
            Train_input_c,
            Train_target_p,
            optimizer_fore_pre,
            penalty_weight,
        )
        update_convergence_record(pretrain_convergence_record, i, loss2_display)

        if should_log_epoch(i, epoch1_pre, interval=PRETRAIN_LOG_INTERVAL):
            end_time = time.time()
            print(end_time - start_time)
            print("[Epoch %d/%d] [loss_mse: %f] " % (i, epoch1_pre, loss2_display))
            writer1.add_scalar("loss_mse_pre", loss1_display, i)
            writer2.add_scalar("loss_mse_pre", loss2_display, i)

    register_convergence_record(pretrain_convergence_record)
    model_fore_pre.eval()
    torch.save(model_fore_pre.state_dict(), PRETRAIN_MODEL_PATH)
    print(f"\n✓ 预训练完成: {PRETRAIN_MODEL_PATH}")


run_target_aware_pretraining()

if TRAIN_PRETRAIN_ONLY:
    export_convergence_report(
        CONVERGENCE_REPORT_PATH,
        convergence_records,
        {
            "protocol_name": PROTOCOL_NAME,
            "protocol_data_dir": PROTOCOL_DATA_DIR,
            "protocol_metadata_path": PROTOCOL_METADATA_PATH,
            "artifact_dir": ARTIFACT_DIR,
            "model_output_dir": MODEL_OUTPUT_DIR,
            "base_model_output_dir": BASE_MODEL_OUTPUT_DIR,
            "target_aware_base_model_output_dir": TARGET_AWARE_BASE_MODEL_OUTPUT_DIR,
            "local_pretrain_init_model_dir": LOCAL_PRETRAIN_INIT_MODEL_DIR,
            "target_aware_pretrain_init_model_dir": TARGET_AWARE_PRETRAIN_INIT_MODEL_DIR,
            "local_pretrain_epoch_offset": LOCAL_PRETRAIN_EPOCH_OFFSET,
            "target_aware_pretrain_epoch_offset": TARGET_AWARE_PRETRAIN_EPOCH_OFFSET,
            "logs_train_dir": LOGS_TRAIN_DIR,
            "yearly_protocol_enabled": YEARLY_PROTOCOL_ENABLED,
            "seasonal_protocol_enabled": SEASONAL_PROTOCOL_ENABLED,
            "use_federation": USE_FEDERATION,
            "use_pseudo_fed": USE_PSEUDO_FED,
            "train_meta_only_baseline": TRAIN_META_ONLY_BASELINE,
            "train_pretrain_only": TRAIN_PRETRAIN_ONLY,
            "run_federated_pretrain": RUN_FEDERATED_PRETRAIN,
            "fed_pretrain_algo": FED_PRETRAIN_ALGO,
            "fedavg_local_epochs": FEDAVG_LOCAL_EPOCHS,
            "fedavg_client_weighting": FEDAVG_CLIENT_WEIGHTING,
            "enable_fedtl_ft": ENABLE_FEDTL_FT,
            "skip_local_pretrain": SKIP_LOCAL_PRETRAIN,
            "enable_target_aware_meta_noft": ENABLE_TARGET_AWARE_META_NOFT,
            "skip_target_aware_pretrain": SKIP_TARGET_AWARE_PRETRAIN,
            "hwa_pretrain_windowed": HWA_PRETRAIN_WINDOWED,
            "target_aware_pretrain_hwa_loss": TARGET_AWARE_PRETRAIN_HWA_LOSS,
            "global_seed": GLOBAL_SEED,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "pretrain_log_interval": PRETRAIN_LOG_INTERVAL,
        },
    )
    progress_log("✓ TRAIN_PRETRAIN_ONLY=1：预训练完成，跳过 meta/fine-tune/eval 阶段")
    raise SystemExit(0)


def sample_meta_batch(sample_station_ids=None, target_aware=False, target_station_id=None):
    """按给定场站范围采样 meta 任务。"""
    if sample_station_ids is None:
        sample_station_ids = station_ids
    if target_station_id is None:
        target_station_id = sample_station_ids[0]

    task_pool = []
    task_weights = []
    meta_total_shots = META_SUPPORT_SHOTS + META_QUERY_SHOTS
    for station_id in sample_station_ids:
        p_conven_class_st = all_stations_full_data[station_id]['p_conven_class']
        total_station_classes = np.size(p_conven_class_st, axis=1)
        for i_class in range(total_station_classes):
            p_data = p_conven_class_st[0, i_class]
            num_samples = p_data.shape[0] // len_realp
            if num_samples >= meta_total_shots:
                task_pool.append((station_id, i_class))
                if target_aware:
                    task_score = get_target_aware_task_score(station_id, i_class, target_station_id=target_station_id)
                    task_weights.append(max(float(TARGET_AWARE_META_SIM_FLOOR), task_score))

    if not task_pool:
        raise ValueError(
            f"No valid meta tasks found for support/query shots "
            f"{META_SUPPORT_SHOTS}/{META_QUERY_SHOTS}"
        )

    tasks_per_epoch = min(META_TASKS_PER_EPOCH, len(task_pool))
    if target_aware:
        sampled_tasks = weighted_sample_tasks(task_pool, task_weights, tasks_per_epoch)
    else:
        sampled_tasks = random.sample(task_pool, tasks_per_epoch)

    selected_tasks = []
    for station_id, i_class in sampled_tasks:
        nwp_conven_class_st = all_stations_full_data[station_id]['nwp_conven_class']
        p_conven_class_st = all_stations_full_data[station_id]['p_conven_class']

        for i_nwp in range(np.size(nwp_conven_class_st, axis=1)):
            nwp_data = nwp_conven_class_st[0, i_nwp][0, i_class]
            num_samples = nwp_data.shape[0] // len_realp
            nwp_reshaped = nwp_data[:num_samples * len_realp].reshape(num_samples, len_realp, 1)
            if i_nwp == 0:
                nwp_conven_class_1 = nwp_reshaped
            else:
                nwp_conven_class_1 = np.concatenate((nwp_conven_class_1, nwp_reshaped), axis=2)

        p_data = p_conven_class_st[0, i_class]
        num_samples = p_data.shape[0] // len_realp
        p_conven_class_1 = p_data[:num_samples * len_realp].reshape(num_samples, len_realp, 1)
        hwa_weight_class_1 = compute_hwa_weights_numpy_from_normalized(
            nwp_conven_class_1,
            wind_scale=get_station_hwa_wind_scale(station_id),
        )
        task_score = get_target_aware_task_score(station_id, i_class, target_station_id=target_station_id) if target_aware else 0.0
        task_weight = 1.0 + float(TARGET_AWARE_META_TASK_WEIGHT_ETA) * float(task_score)
        if target_aware:
            hwa_weight_class_1 = hwa_weight_class_1 * task_weight

        selected_tasks.append({
            'nwp': nwp_conven_class_1,
            'p': p_conven_class_1,
            'hwa_weight': hwa_weight_class_1,
        })

    train_input_dataset = [task['nwp'] for task in selected_tasks]
    train_target_dataset = [task['p'] for task in selected_tasks]
    train_weight_dataset = [task['hwa_weight'] for task in selected_tasks]
    num_tasks = len(selected_tasks)

    for i_task in range(num_tasks):
        index_shot = random.sample(range(0, np.size(train_input_dataset[i_task], axis=0)), meta_total_shots)
        train_input_support_ = train_input_dataset[i_task][index_shot[0:META_SUPPORT_SHOTS], :, :]
        train_input_query_ = train_input_dataset[i_task][index_shot[META_SUPPORT_SHOTS:meta_total_shots], :, :]
        train_target_support_ = train_target_dataset[i_task][index_shot[0:META_SUPPORT_SHOTS], :, :]
        train_target_query_ = train_target_dataset[i_task][index_shot[META_SUPPORT_SHOTS:meta_total_shots], :, :]
        train_weight_support_ = train_weight_dataset[i_task][index_shot[0:META_SUPPORT_SHOTS], :, :]
        train_weight_query_ = train_weight_dataset[i_task][index_shot[META_SUPPORT_SHOTS:meta_total_shots], :, :]
        if i_task == 0:
            train_input_support = train_input_support_
            train_input_query = train_input_query_
            train_target_support = train_target_support_
            train_target_query = train_target_query_
            train_weight_support = train_weight_support_
            train_weight_query = train_weight_query_
        else:
            train_input_support = np.concatenate((train_input_support, train_input_support_), axis=0)
            train_input_query = np.concatenate((train_input_query, train_input_query_), axis=0)
            train_target_support = np.concatenate((train_target_support, train_target_support_), axis=0)
            train_target_query = np.concatenate((train_target_query, train_target_query_), axis=0)
            train_weight_support = np.concatenate((train_weight_support, train_weight_support_), axis=0)
            train_weight_query = np.concatenate((train_weight_query, train_weight_query_), axis=0)

    return (
        torch.tensor(train_target_support, dtype=torch.float32),
        torch.tensor(train_input_support, dtype=torch.float32),
        torch.tensor(train_target_query, dtype=torch.float32),
        torch.tensor(train_input_query, dtype=torch.float32),
        torch.tensor(train_weight_support, dtype=torch.float32),
        torch.tensor(train_weight_query, dtype=torch.float32),
    )


def freeze_lwp_as_identity(model_instance):
    """
    将 LWP 固定为恒等变换(scale=1, shift=0)，用于传统 meta-learning 基线。
    """
    for module in model_instance.modules():
        if isinstance(module, model.LWP):
            with torch.no_grad():
                module.scale.fill_(1.0)
                module.shift.zero_()
            module.scale.requires_grad = False
            module.shift.requires_grad = False


def get_meta_trainable_params(model_instance, train_all_params=False, disable_lwp=False):
    """
    元训练参数选择：
    - train_all_params=True: 训练全部参数（传统 meta-learning）
    - disable_lwp=True: 从可训练参数中移除 LWP 参数
    """
    if train_all_params:
        if disable_lwp:
            return [p for name, p in model_instance.named_parameters() if "lwp" not in name]
        return list(model_instance.parameters())

    if disable_lwp:
        return list(model_instance.fore_baselearner.parameters())

    return list(model_instance.get_trainable_params())


def aggregate_fed_normal_meta_states(weighted_states):
    aggregate_state = copy.deepcopy(weighted_states[0][0])
    for name in aggregate_state.keys():
        reference_tensor = aggregate_state[name]
        if not torch.is_floating_point(reference_tensor):
            aggregate_state[name] = reference_tensor.clone()
            continue
        weighted_sum = None
        for state_dict, alpha in weighted_states:
            contrib = state_dict[name].detach().to(dtype=torch.float32, device="cpu") * float(alpha)
            weighted_sum = contrib if weighted_sum is None else weighted_sum + contrib
        aggregate_state[name] = weighted_sum.to(dtype=reference_tensor.dtype, device=reference_tensor.device)
    return aggregate_state


def split_normal_meta_train_proxy_windows(window_payloads, proxy_ratio=None, seed=GLOBAL_SEED):
    proxy_ratio = SELECTIVE_FED_META_PROXY_RATIO if proxy_ratio is None else float(proxy_ratio)
    proxy_ratio = min(max(proxy_ratio, 0.0), 0.9)
    rng = np.random.RandomState(seed)
    split_payloads = []
    for payload in window_payloads:
        nwp_windows = np.asarray(payload["nwp"], dtype=np.float32)
        power_windows = np.asarray(payload["p"], dtype=np.float32)
        window_count = min(nwp_windows.shape[0], power_windows.shape[0])
        if window_count <= 1:
            split_payloads.append({
                "meta_train_nwp": nwp_windows[:window_count],
                "meta_train_power": power_windows[:window_count],
                "proxy_nwp": np.empty((0,) + nwp_windows.shape[1:], dtype=np.float32),
                "proxy_power": np.empty((0,) + power_windows.shape[1:], dtype=np.float32),
            })
            continue
        indices = np.arange(window_count)
        rng.shuffle(indices)
        meta_total_shots = META_SUPPORT_SHOTS + META_QUERY_SHOTS
        max_proxy_count = max(0, window_count - meta_total_shots)
        requested_proxy_count = max(1, int(round(window_count * proxy_ratio)))
        proxy_count = min(max_proxy_count, requested_proxy_count)
        if proxy_count <= 0:
            proxy_indices = np.empty((0,), dtype=int)
            train_indices = np.sort(indices)
            split_payloads.append({
                "meta_train_nwp": nwp_windows[train_indices],
                "meta_train_power": power_windows[train_indices],
                "proxy_nwp": np.empty((0,) + nwp_windows.shape[1:], dtype=np.float32),
                "proxy_power": np.empty((0,) + power_windows.shape[1:], dtype=np.float32),
            })
            continue
        proxy_indices = np.sort(indices[:proxy_count])
        train_indices = np.sort(indices[proxy_count:])
        split_payloads.append({
            "meta_train_nwp": nwp_windows[train_indices],
            "meta_train_power": power_windows[train_indices],
            "proxy_nwp": nwp_windows[proxy_indices],
            "proxy_power": power_windows[proxy_indices],
        })
    return split_payloads


def evaluate_target_proxy_loss(state_dict, proxy_input_tensor, proxy_target_tensor, station_id=None):
    if proxy_input_tensor.shape[0] == 0 or proxy_target_tensor.shape[0] == 0:
        return None
    model_fore_test_task_query.load_state_dict(copy.deepcopy(state_dict))
    model_fore_test_task_query.eval()
    with torch.no_grad():
        outputs = model_fore_test_task_query(proxy_input_tensor.to(device))
        proxy_target_device = proxy_target_tensor.to(device)
        proxy_weights = None
        if HWA_SELECTIVE_PROXY_LOSS and station_id is not None:
            proxy_weights = compute_hwa_weights_tensor_from_normalized(
                proxy_input_tensor.to(device),
                wind_scale=get_station_hwa_wind_scale(station_id),
            )
        loss_value = weighted_mse_loss(outputs, proxy_target_device, proxy_weights)
    if not torch.isfinite(loss_value):
        return None
    return float(loss_value.item())


def evaluate_weighted_proxy_loss(state_dict, proxy_input_tensor, proxy_target_tensor, proxy_weight_tensor=None):
    if proxy_input_tensor.shape[0] == 0 or proxy_target_tensor.shape[0] == 0:
        return None
    model_fore_test_task_query.load_state_dict(copy.deepcopy(state_dict))
    model_fore_test_task_query.eval()
    with torch.no_grad():
        outputs = model_fore_test_task_query(proxy_input_tensor.to(device))
        proxy_target_device = proxy_target_tensor.to(device)
        weights = proxy_weight_tensor.to(device) if proxy_weight_tensor is not None else None
        loss_value = weighted_mse_loss(outputs, proxy_target_device, weights)
    if not torch.isfinite(loss_value):
        return None
    return float(loss_value.item())


def compute_selective_fed_meta_gain(self_proxy_loss, candidate_proxy_loss):
    if self_proxy_loss is None or candidate_proxy_loss is None:
        return None
    return float(self_proxy_loss) - float(candidate_proxy_loss)


def aggregate_selective_fed_normal_meta_states(self_state, accepted_source_states, self_floor=None):
    self_floor = SELECTIVE_FED_META_SELF_FLOOR if self_floor is None else float(self_floor)
    self_floor = min(max(self_floor, 0.0), 1.0)
    if not accepted_source_states:
        return aggregate_fed_normal_meta_states([(self_state, 1.0)])
    source_weight_sum = sum(float(weight) for _, weight in accepted_source_states)
    if source_weight_sum <= 0.0:
        return aggregate_fed_normal_meta_states([(self_state, 1.0)])
    weighted_states = [(self_state, self_floor)]
    remaining_weight = 1.0 - self_floor
    for state_dict, source_weight in accepted_source_states:
        weighted_states.append((state_dict, remaining_weight * float(source_weight) / source_weight_sum))
    return aggregate_fed_normal_meta_states(weighted_states)


def compute_state_delta(updated_state, base_state):
    delta_state = {}
    for name, tensor in updated_state.items():
        if torch.is_floating_point(tensor):
            delta_state[name] = tensor.detach().to(dtype=torch.float32, device="cpu") - base_state[name].detach().to(dtype=torch.float32, device="cpu")
        else:
            delta_state[name] = torch.zeros((), dtype=torch.float32)
    return delta_state


def apply_weighted_state_deltas(base_state, weighted_deltas):
    updated_state = copy.deepcopy(base_state)
    for name, tensor in updated_state.items():
        if not torch.is_floating_point(tensor):
            updated_state[name] = tensor.clone()
            continue
        acc = tensor.detach().to(dtype=torch.float32, device="cpu")
        for delta_state, alpha in weighted_deltas:
            acc = acc + delta_state[name].detach().to(dtype=torch.float32, device="cpu") * float(alpha)
        updated_state[name] = acc.to(dtype=tensor.dtype, device=tensor.device)
    return updated_state


def parse_target_aware_selective_alpha_grid():
    max_alpha = min(max(float(TARGET_AWARE_SELECTIVE_FED_SOURCE_ALPHA_CAP), 0.0), 1.0)
    self_floor_cap = 1.0 - min(max(float(TARGET_AWARE_SELECTIVE_FED_SELF_FLOOR), 0.0), 1.0)
    max_alpha = min(max_alpha, max(0.0, self_floor_cap))
    if max_alpha <= 0.0:
        return []
    if TARGET_AWARE_SELECTIVE_FED_ALPHA_GRID.strip():
        alpha_values = []
        for raw_value in TARGET_AWARE_SELECTIVE_FED_ALPHA_GRID.split(","):
            raw_value = raw_value.strip()
            if not raw_value:
                continue
            alpha = float(raw_value)
            if 0.0 < alpha <= max_alpha:
                alpha_values.append(alpha)
        if alpha_values:
            return sorted(set(alpha_values))
    default_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    return [alpha for alpha in default_grid if alpha <= max_alpha + 1e-12] or [max_alpha]


def build_normal_high_wind_like_proxy_windows(station_id):
    station_id = str(station_id)
    max_windows = max(0, int(TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_MAX_WINDOWS))
    if max_windows <= 0:
        return (
            np.empty((0, len_realp, dem_realc), dtype=np.float32),
            np.empty((0, len_realp, dem_realp), dtype=np.float32),
        )

    nwp_conven_class_st = all_stations_full_data[station_id]['nwp_conven_class']
    p_conven_class_st = all_stations_full_data[station_id]['p_conven_class']
    wind_scale = get_station_hwa_wind_scale(station_id)
    candidates = []
    for i_class in range(np.size(p_conven_class_st, axis=1)):
        class_nwp_windows = None
        for i_nwp in range(np.size(nwp_conven_class_st, axis=1)):
            nwp_data = nwp_conven_class_st[0, i_nwp][0, i_class]
            nwp_feature_windows = reshape_sequence_to_windows(nwp_data, 1)
            class_nwp_windows = (
                nwp_feature_windows
                if class_nwp_windows is None
                else np.concatenate((class_nwp_windows, nwp_feature_windows), axis=2)
            )
        class_power_windows = reshape_sequence_to_windows(p_conven_class_st[0, i_class], dem_realp)
        window_count = min(class_nwp_windows.shape[0], class_power_windows.shape[0])
        if window_count <= 0:
            continue
        class_nwp_windows = class_nwp_windows[:window_count]
        class_power_windows = class_power_windows[:window_count]
        wind_values = class_nwp_windows[:, :, HWA_WIND_FEATURE_INDEX] * float(wind_scale)
        mean_wind = np.mean(wind_values, axis=1)
        max_wind = np.max(wind_values, axis=1)
        highwind_mask = (mean_wind >= HWA_WIND_THRESHOLD) | (max_wind >= TARGET_AWARE_META_WIND_MEAN_THRESHOLD)
        for window_idx in np.where(highwind_mask)[0]:
            score = float(mean_wind[window_idx] + 0.25 * max_wind[window_idx])
            candidates.append((score, class_nwp_windows[window_idx], class_power_windows[window_idx]))

    if not candidates:
        return (
            np.empty((0, len_realp, dem_realc), dtype=np.float32),
            np.empty((0, len_realp, dem_realp), dtype=np.float32),
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:max_windows]
    nwp_windows = np.stack([item[1] for item in selected], axis=0).astype(np.float32)
    power_windows = np.stack([item[2] for item in selected], axis=0).astype(np.float32)
    return nwp_windows, power_windows


def build_target_aware_selective_proxy_tensors(station_id):
    extreme_nwp, extreme_power = build_extreme_support_windows(station_id, class_idx=0)
    normal_nwp, normal_power = build_normal_high_wind_like_proxy_windows(station_id)
    input_parts = []
    target_parts = []
    weight_parts = []
    if extreme_nwp.shape[0] > 0:
        input_parts.append(extreme_nwp.astype(np.float32))
        target_parts.append(extreme_power.astype(np.float32))
        extreme_weights = compute_hwa_weights_numpy_from_normalized(
            extreme_nwp,
            wind_scale=get_station_hwa_wind_scale(station_id),
        ) * float(TARGET_AWARE_SELECTIVE_FED_PROXY_EXTREME_WEIGHT)
        weight_parts.append(extreme_weights.astype(np.float32))
    if normal_nwp.shape[0] > 0:
        input_parts.append(normal_nwp.astype(np.float32))
        target_parts.append(normal_power.astype(np.float32))
        normal_weights = compute_hwa_weights_numpy_from_normalized(
            normal_nwp,
            wind_scale=get_station_hwa_wind_scale(station_id),
        ) * float(TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_WEIGHT)
        weight_parts.append(normal_weights.astype(np.float32))
    if not input_parts:
        return (
            torch.empty((0, len_realp, dem_realc), dtype=torch.float32),
            torch.empty((0, len_realp, dem_realp), dtype=torch.float32),
            torch.empty((0, len_realp, dem_realp), dtype=torch.float32),
        )
    proxy_input = np.concatenate(input_parts, axis=0).astype(np.float32)
    proxy_target = np.concatenate(target_parts, axis=0).astype(np.float32)
    proxy_weight = np.concatenate(weight_parts, axis=0).astype(np.float32)
    return (
        torch.tensor(proxy_input, dtype=torch.float32),
        torch.tensor(proxy_target, dtype=torch.float32),
        torch.tensor(proxy_weight, dtype=torch.float32),
    )


def build_station_normal_meta_train_proxy_payloads(p_conven_class_st, nwp_conven_class_st, station_id):
    total_station_classes = np.size(p_conven_class_st, axis=1)
    nwp_feature_count = np.size(nwp_conven_class_st, axis=1)
    meta_train_power_classes = np.empty((1, total_station_classes), dtype=object)
    meta_train_nwp_classes = np.empty((1, nwp_feature_count), dtype=object)
    for i_nwp in range(nwp_feature_count):
        meta_train_nwp_classes[0, i_nwp] = np.empty((1, total_station_classes), dtype=object)

    proxy_nwp_windows_all = []
    proxy_power_windows_all = []
    station_seed_base = GLOBAL_SEED + int(station_id) * 1000

    for i_class in range(total_station_classes):
        class_nwp_windows = None
        for i_nwp in range(nwp_feature_count):
            nwp_data = nwp_conven_class_st[0, i_nwp][0, i_class]
            num_samples = nwp_data.shape[0] // len_realp
            nwp_reshaped = np.asarray(
                nwp_data[:num_samples * len_realp].reshape(num_samples, len_realp, 1),
                dtype=np.float32,
            )
            if class_nwp_windows is None:
                class_nwp_windows = nwp_reshaped
            else:
                class_nwp_windows = np.concatenate((class_nwp_windows, nwp_reshaped), axis=2)

        p_data = p_conven_class_st[0, i_class]
        num_samples = p_data.shape[0] // len_realp
        class_power_windows = np.asarray(
            p_data[:num_samples * len_realp].reshape(num_samples, len_realp, 1),
            dtype=np.float32,
        )

        split_payload = split_normal_meta_train_proxy_windows(
            [{"nwp": class_nwp_windows, "p": class_power_windows}],
            proxy_ratio=SELECTIVE_FED_META_PROXY_RATIO,
            seed=station_seed_base + i_class,
        )[0]

        meta_train_power_classes[0, i_class] = split_payload["meta_train_power"].reshape(-1, 1)
        for i_nwp in range(nwp_feature_count):
            meta_train_nwp_classes[0, i_nwp][0, i_class] = split_payload["meta_train_nwp"][:, :, i_nwp].reshape(-1, 1)

        if split_payload["proxy_nwp"].shape[0] > 0:
            proxy_nwp_windows_all.append(split_payload["proxy_nwp"])
            proxy_power_windows_all.append(split_payload["proxy_power"])

    if proxy_nwp_windows_all:
        proxy_input = np.concatenate(proxy_nwp_windows_all, axis=0).astype(np.float32)
        proxy_target = np.concatenate(proxy_power_windows_all, axis=0).astype(np.float32)
    else:
        proxy_input = np.empty((0, len_realp, dem_realc), dtype=np.float32)
        proxy_target = np.empty((0, len_realp, dem_realp), dtype=np.float32)

    return meta_train_power_classes, meta_train_nwp_classes, proxy_input, proxy_target


def count_station_normal_meta_windows(station_id):
    p_conven_class_st = all_stations_full_data[station_id]['p_conven_class']
    total_station_classes = np.size(p_conven_class_st, axis=1)
    total_windows = 0
    for i_class in range(total_station_classes):
        p_data = p_conven_class_st[0, i_class]
        total_windows += max(0, int(p_data.shape[0] // len_realp))
    return total_windows


def compute_fed_normal_meta_station_weights(target_station_id, candidate_station_ids):
    candidate_station_ids = [str(station_id) for station_id in candidate_station_ids]
    if target_station_id not in candidate_station_ids:
        raise ValueError(f"target station {target_station_id} is not in candidate stations")
    if len(candidate_station_ids) == 1:
        return {target_station_id: 1.0}

    self_floor = min(max(float(FED_NORMAL_META_SELF_FLOOR), 0.0), 1.0)
    source_station_ids = [station_id for station_id in candidate_station_ids if station_id != target_station_id]
    source_counts = {
        station_id: count_station_normal_meta_windows(station_id)
        for station_id in source_station_ids
    }
    total_source_count = float(sum(source_counts.values()))
    if total_source_count <= 0.0:
        return {
            station_id: 1.0 if station_id == target_station_id else 0.0
            for station_id in candidate_station_ids
        }

    station_weights = {target_station_id: self_floor}
    remaining_weight = 1.0 - self_floor
    for station_id in source_station_ids:
        station_weights[station_id] = remaining_weight * float(source_counts[station_id]) / total_source_count
    return station_weights


def run_meta_support_query_update(
    base_state_dict,
    sample_station_ids,
    meta_tag,
    epoch_idx,
    use_cdrm=True,
    cdrm_weight=10.0,
    train_all_params=False,
    disable_lwp=False,
    target_aware=False,
    target_station_id=None,
):
    support_params = get_meta_trainable_params(
        model_fore_train_task_support,
        train_all_params=train_all_params,
        disable_lwp=disable_lwp
    )
    query_params = get_meta_trainable_params(
        model_fore_train_task_query,
        train_all_params=train_all_params,
        disable_lwp=disable_lwp
    )
    optimizer_support = torch.optim.Adam(support_params, lr=0.0002, betas=(0.5, 0.999))
    optimizer_query = torch.optim.Adam(query_params, lr=0.0002, betas=(0.5, 0.999))

    (
        Train_target_support,
        Train_input_support,
        Train_target_query,
        Train_input_query,
        Train_weight_support,
        Train_weight_query,
    ) = sample_meta_batch(
        sample_station_ids=sample_station_ids,
        target_aware=target_aware,
        target_station_id=target_station_id,
    )

    model_fore_train_task_support.load_state_dict(copy.deepcopy(base_state_dict))
    if disable_lwp:
        freeze_lwp_as_identity(model_fore_train_task_support)
    model_fore_train_task_support.train()
    Train_target_support = Train_target_support.to(device)
    Train_input_support = Train_input_support.to(device)
    Train_weight_support = Train_weight_support.to(device)
    Train_outputs_support = model_fore_train_task_support(Train_input_support)
    if use_cdrm:
        loss1 = penalty(Train_outputs_support, Train_target_support)
    else:
        loss1 = torch.zeros((), dtype=torch.float32, device=device)
    loss2 = weighted_mse_loss(
        Train_outputs_support,
        Train_target_support,
        Train_weight_support if (HWA_META_LOSS or target_aware) else None,
    )
    loss_en = float(cdrm_weight) * loss1 + loss2 if use_cdrm else loss2
    optimizer_support.zero_grad()
    loss_en.backward()
    optimizer_support.step()
    model_fore_train_task_support.eval()
    support_state = copy.deepcopy(model_fore_train_task_support.state_dict())

    model_fore_train_task_query.load_state_dict(copy.deepcopy(support_state))
    if disable_lwp:
        freeze_lwp_as_identity(model_fore_train_task_query)
    model_fore_train_task_query.train()
    Train_target_query = Train_target_query.to(device)
    Train_input_query = Train_input_query.to(device)
    Train_weight_query = Train_weight_query.to(device)
    Train_outputs_query_ = model_fore_train_task_query(Train_input_query)
    if use_cdrm:
        loss1_q = penalty(Train_outputs_query_, Train_target_query)
    else:
        loss1_q = torch.zeros((), dtype=torch.float32, device=device)
    loss2_q = weighted_mse_loss(
        Train_outputs_query_,
        Train_target_query,
        Train_weight_query if (HWA_META_LOSS or target_aware) else None,
    )
    loss_en_q = float(cdrm_weight) * loss1_q + loss2_q if use_cdrm else loss2_q
    optimizer_query.zero_grad()
    loss_en_q.backward()
    optimizer_query.step()
    model_fore_train_task_query.eval()
    query_state = copy.deepcopy(model_fore_train_task_query.state_dict())

    writer1.add_scalar(f"loss_penalty_train_task_support_{meta_tag}", loss1.item(), epoch_idx)
    writer2.add_scalar(f"loss_mse_train_task_support_{meta_tag}", loss2.item(), epoch_idx)
    writer1.add_scalar(f"loss_penalty_train_task_query_{meta_tag}", loss1_q.item(), epoch_idx)
    writer2.add_scalar(f"loss_mse_train_task_query_{meta_tag}", loss2_q.item(), epoch_idx)
    return query_state, loss2_q.item()


def run_meta_training(
    meta_tag,
    init_state_dict,
    support_model_path,
    query_model_path,
    epoch_train_task=70000,
    use_cdrm=True,
    cdrm_weight=10.0,
    train_all_params=False,
    disable_lwp=False,
    sample_station_ids=None,
    target_aware=False,
    target_station_id=None,
    epoch_offset=0,
):
    """
    单次元训练过程：
    - proposed: init_state_dict 为 pre-train 权重（CDRM + LWP 轻量更新）
    - meta_only: init_state_dict 为随机初始化（传统基线：可关闭CDRM、全参数更新）
    """
    print("\n" + "=" * 70)
    print(f"开始元训练: {meta_tag}")
    print(
        f"  use_cdrm={use_cdrm}, cdrm_weight={cdrm_weight}, "
        f"target_aware={target_aware}, train_all_params={train_all_params}, disable_lwp={disable_lwp}"
    )
    if sample_station_ids is None:
        sample_station_ids = station_ids
    if target_station_id is None:
        target_station_id = sample_station_ids[0]
    total_task_pool = sum(np.size(all_stations_full_data[s]['p_conven_class'], axis=1) for s in sample_station_ids)
    epoch_offset = max(0, int(epoch_offset))
    total_epochs = epoch_offset + int(epoch_train_task)
    print(
        f"  tasks_per_epoch={META_TASKS_PER_EPOCH}, task_pool={total_task_pool} "
        f"({len(sample_station_ids)} stations: {', '.join(sample_station_ids)})"
    )
    if epoch_offset:
        print(f"  resume_epoch_offset={epoch_offset}, total_epoch_after_run={total_epochs}")
    print("=" * 70)

    support_params = get_meta_trainable_params(
        model_fore_train_task_support,
        train_all_params=train_all_params,
        disable_lwp=disable_lwp
    )
    query_params = get_meta_trainable_params(
        model_fore_train_task_query,
        train_all_params=train_all_params,
        disable_lwp=disable_lwp
    )

    optimizer_support = torch.optim.Adam(support_params, lr=0.0002, betas=(0.5, 0.999))
    optimizer_query = torch.optim.Adam(query_params, lr=0.0002, betas=(0.5, 0.999))
    if meta_tag.startswith("local_meta_station"):
        stage_type = "local_meta"
        convergence_record = initialize_convergence_record(
            stage_type="local_meta",
            stage_id=meta_tag,
            total_epochs=total_epochs,
            patience=CONVERGENCE_PATIENCE_META,
        )
    elif meta_tag.startswith("target_aware_meta_station"):
        stage_type = "target_aware_meta"
        convergence_record = initialize_convergence_record(
            stage_type=stage_type,
            stage_id=meta_tag,
            total_epochs=total_epochs,
            patience=CONVERGENCE_PATIENCE_META,
        )
    else:
        stage_type = meta_tag
        convergence_record = initialize_convergence_record(
            stage_type=stage_type,
            stage_id=meta_tag,
            total_epochs=total_epochs,
            patience=CONVERGENCE_PATIENCE_META,
        )

    for i_t in range(epoch_train_task):
        global_epoch = epoch_offset + i_t
        (
            Train_target_support,
            Train_input_support,
            Train_target_query,
            Train_input_query,
            Train_weight_support,
            Train_weight_query,
        ) = sample_meta_batch(
            sample_station_ids=sample_station_ids,
            target_aware=target_aware,
            target_station_id=target_station_id,
        )

        print(
            "[##################################################################"
            f"——{meta_tag}:train_task_support_Epoch {global_epoch + 1}/{total_epochs}——"
            "############################################################]"
        )

        if i_t == 0:
            base_state = copy.deepcopy(init_state_dict)
        else:
            base_state = torch.load(query_model_path)
        model_fore_train_task_support.load_state_dict(copy.deepcopy(base_state))

        if disable_lwp:
            freeze_lwp_as_identity(model_fore_train_task_support)

        model_fore_train_task_support.train()
        Train_target_support = Train_target_support.to(device)
        Train_input_support = Train_input_support.to(device)
        Train_weight_support = Train_weight_support.to(device)
        Train_outputs_support = model_fore_train_task_support(Train_input_support)
        if use_cdrm:
            loss1 = penalty(Train_outputs_support, Train_target_support)
        else:
            loss1 = torch.zeros((), dtype=torch.float32, device=device)
        loss2 = weighted_mse_loss(
            Train_outputs_support,
            Train_target_support,
            Train_weight_support if (HWA_META_LOSS or target_aware) else None,
        )
        loss_en = float(cdrm_weight) * loss1 + loss2 if use_cdrm else loss2
        optimizer_support.zero_grad()
        loss_en.backward()
        optimizer_support.step()
        model_fore_train_task_support.eval()
        support_state = copy.deepcopy(model_fore_train_task_support.state_dict())
        torch.save(support_state, support_model_path)

        writer1.add_scalar(f"loss_penalty_train_task_support_{meta_tag}", loss1.item(), global_epoch)
        writer2.add_scalar(f"loss_mse_train_task_support_{meta_tag}", loss2.item(), global_epoch)

        print(
            "[##################################################################"
            f"——{meta_tag}:train_task_query_Epoch {global_epoch + 1}/{total_epochs}——"
            "############################################################]"
        )

        # 严格 support->query 链路：query 以本轮 support 更新后的参数为起点
        model_fore_train_task_query.load_state_dict(copy.deepcopy(support_state))

        if disable_lwp:
            freeze_lwp_as_identity(model_fore_train_task_query)

        model_fore_train_task_query.train()
        Train_target_query = Train_target_query.to(device)
        Train_input_query = Train_input_query.to(device)
        Train_weight_query = Train_weight_query.to(device)
        Train_outputs_query_ = model_fore_train_task_query(Train_input_query)
        if use_cdrm:
            loss1_q = penalty(Train_outputs_query_, Train_target_query)
        else:
            loss1_q = torch.zeros((), dtype=torch.float32, device=device)
        loss2_q = weighted_mse_loss(
            Train_outputs_query_,
            Train_target_query,
            Train_weight_query if (HWA_META_LOSS or target_aware) else None,
        )
        loss_en_q = float(cdrm_weight) * loss1_q + loss2_q if use_cdrm else loss2_q
        optimizer_query.zero_grad()
        loss_en_q.backward()
        optimizer_query.step()
        model_fore_train_task_query.eval()
        torch.save(model_fore_train_task_query.state_dict(), query_model_path)

        writer1.add_scalar(f"loss_penalty_train_task_query_{meta_tag}", loss1_q.item(), global_epoch)
        writer2.add_scalar(f"loss_mse_train_task_query_{meta_tag}", loss2_q.item(), global_epoch)
        update_convergence_record(convergence_record, global_epoch, loss2_q.item())

        if should_log_epoch(i_t, epoch_train_task, interval=META_LOG_INTERVAL):
            progress_log(
                f"  收敛追踪[{stage_type}:{meta_tag}] "
                f"epoch={global_epoch + 1}/{total_epochs} "
                f"query_mse={loss2_q.item():.6f}"
            )

    register_convergence_record(convergence_record)
    print(f"✓ 元训练完成: {query_model_path}")


def run_shared_meta_training():
    proposed_init_state = torch.load(PRETRAIN_MODEL_PATH)
    run_meta_training(
        meta_tag="proposed",
        init_state_dict=proposed_init_state,
        support_model_path=PROPOSED_SUPPORT_MODEL_PATH,
        query_model_path=PROPOSED_META_MODEL_PATH,
        epoch_train_task=PROPOSED_META_EPOCHS,
        use_cdrm=True,
        train_all_params=False,
        disable_lwp=False
    )

    if TRAIN_META_ONLY_BASELINE:
        run_meta_training(
            meta_tag="meta_only",
            init_state_dict=meta_only_random_init_state,
            support_model_path=META_ONLY_SUPPORT_MODEL_PATH,
            query_model_path=META_ONLY_MODEL_PATH,
            epoch_train_task=META_ONLY_META_EPOCHS,
            use_cdrm=META_ONLY_USE_CDRM,
            train_all_params=META_ONLY_TRAIN_ALL_PARAMS,
            disable_lwp=META_ONLY_DISABLE_LWP
        )


def run_local_meta_training():
    for station_id in station_ids:
        print("\n" + "=" * 70)
        print(f"开始本地元训练: station {station_id}")
        print("=" * 70)

        local_pretrain_path = get_local_pretrain_model_path(station_id)
        local_meta_path = get_local_meta_model_path(station_id)
        if SKIP_LOCAL_META:
            if ENABLE_FEDTL_FT and RUN_FEDERATED_PRETRAIN:
                progress_log("✓ FedTL-FT模式跳过本地元训练：该基线只使用联邦预训练 + 本地FT")
                continue
            if not os.path.exists(local_meta_path):
                raise FileNotFoundError(
                    f"SKIP_LOCAL_META=1 但未找到已有 checkpoint: {local_meta_path}"
                )
            progress_log(f"✓ 跳过本地元训练，复用已有 checkpoint: {local_meta_path}")
            continue
        if LOCAL_META_INIT_MODEL_DIR:
            local_meta_init_path = resolve_init_model_path(
                LOCAL_META_INIT_MODEL_DIR,
                f"model_fore_train_task_query_local_meta_station{station_id}.pth",
            )
            if not os.path.exists(local_meta_init_path):
                raise FileNotFoundError(f"local meta station {station_id} resume checkpoint missing: {local_meta_init_path}")
            progress_log(f"✓ local meta station {station_id} 从 checkpoint 初始化: {local_meta_init_path}")
            local_meta_init_state = torch.load(local_meta_init_path, map_location=device)
        else:
            local_meta_init_state = torch.load(local_pretrain_path, map_location=device)
        run_meta_training(
            meta_tag=f"local_meta_station{station_id}",
            init_state_dict=local_meta_init_state,
            support_model_path=get_local_meta_support_model_path(station_id),
            query_model_path=local_meta_path,
            epoch_train_task=PROPOSED_META_EPOCHS,
            use_cdrm=True,
            train_all_params=False,
            disable_lwp=False,
            sample_station_ids=[station_id],
            epoch_offset=LOCAL_META_EPOCH_OFFSET,
        )

        if TRAIN_META_ONLY_BASELINE:
            run_meta_training(
                meta_tag=f"meta_only_station{station_id}",
                init_state_dict=meta_only_random_init_state,
                support_model_path=get_local_meta_only_support_model_path(station_id),
                query_model_path=get_local_meta_only_model_path(station_id),
                epoch_train_task=META_ONLY_META_EPOCHS,
                use_cdrm=META_ONLY_USE_CDRM,
                train_all_params=META_ONLY_TRAIN_ALL_PARAMS,
                disable_lwp=META_ONLY_DISABLE_LWP,
                sample_station_ids=[station_id]
            )


def run_target_aware_meta_training():
    if not ENABLE_TARGET_AWARE_META_NOFT:
        return
    if not (USE_FEDERATION and not USE_PSEUDO_FED):
        raise RuntimeError("Target-Aware Meta-NoFT currently requires local multi-station mode")
    for station_id in station_ids:
        print("\n" + "=" * 70)
        print(f"开始Target-Aware元训练: station {station_id}")
        print("=" * 70)

        target_aware_pretrain_path = get_target_aware_pretrain_model_path(station_id)
        target_aware_meta_path = get_target_aware_meta_model_path(station_id)
        if SKIP_TARGET_AWARE_META:
            if not os.path.exists(target_aware_meta_path):
                raise FileNotFoundError(
                    f"SKIP_TARGET_AWARE_META=1 但未找到已有 checkpoint: {target_aware_meta_path}"
                )
            progress_log(f"✓ 跳过Target-Aware元训练，复用已有 checkpoint: {target_aware_meta_path}")
            continue
        if not os.path.exists(target_aware_pretrain_path):
            raise FileNotFoundError(
                f"Target-Aware元训练缺少预训练 checkpoint: {target_aware_pretrain_path}"
            )
        if TARGET_AWARE_META_INIT_MODEL_DIR:
            target_aware_meta_init_path = resolve_init_model_path(
                TARGET_AWARE_META_INIT_MODEL_DIR,
                f"model_fore_train_task_query_target_aware_meta_station{station_id}.pth",
            )
            if not os.path.exists(target_aware_meta_init_path):
                raise FileNotFoundError(
                    f"target-aware meta station {station_id} resume checkpoint missing: {target_aware_meta_init_path}"
                )
            progress_log(f"✓ target-aware meta station {station_id} 从 checkpoint 初始化: {target_aware_meta_init_path}")
            target_aware_meta_init_state = torch.load(target_aware_meta_init_path, map_location=device)
        else:
            target_aware_meta_init_state = torch.load(target_aware_pretrain_path, map_location=device)
        run_meta_training(
            meta_tag=f"target_aware_meta_station{station_id}",
            init_state_dict=target_aware_meta_init_state,
            support_model_path=get_target_aware_meta_support_model_path(station_id),
            query_model_path=target_aware_meta_path,
            epoch_train_task=PROPOSED_META_EPOCHS,
            use_cdrm=abs(float(TARGET_AWARE_META_CDRM_WEIGHT)) > 0.0,
            cdrm_weight=TARGET_AWARE_META_CDRM_WEIGHT,
            train_all_params=False,
            disable_lwp=False,
            sample_station_ids=[station_id],
            target_aware=True,
            target_station_id=station_id,
            epoch_offset=TARGET_AWARE_META_EPOCH_OFFSET,
        )


def run_target_aware_selective_fed_meta_training():
    if not ENABLE_TARGET_AWARE_SELECTIVE_FED_META:
        return
    if not ENABLE_TARGET_AWARE_META_NOFT:
        raise RuntimeError("Target-Aware Selective Fed Meta requires ENABLE_TARGET_AWARE_META_NOFT=1")
    if not (USE_FEDERATION and not USE_PSEUDO_FED):
        raise RuntimeError("Target-Aware Selective Fed Meta currently requires local multi-station mode")

    alpha_grid = parse_target_aware_selective_alpha_grid()
    if not alpha_grid:
        raise ValueError("Target-Aware Selective Fed alpha grid is empty; check self_floor/source_alpha_cap")
    top_k = max(1, int(TARGET_AWARE_SELECTIVE_FED_TOP_K))
    use_cdrm = abs(float(TARGET_AWARE_META_CDRM_WEIGHT)) > 0.0

    for station_id in station_ids:
        print("\n" + "=" * 70)
        print(f"开始Target-Aware Selective Fed Meta: target station {station_id}")
        print(
            f"  alpha_grid={alpha_grid}, top_k={top_k}, "
            f"gain_margin={TARGET_AWARE_SELECTIVE_FED_GAIN_MARGIN}"
        )
        print("=" * 70)

        target_aware_pretrain_path = get_target_aware_pretrain_model_path(station_id)
        selective_support_path = get_target_aware_selective_fed_meta_support_model_path(station_id)
        selective_meta_path = get_target_aware_selective_fed_meta_model_path(station_id)
        if SKIP_TARGET_AWARE_SELECTIVE_FED_META:
            if not os.path.exists(selective_meta_path):
                raise FileNotFoundError(
                    f"SKIP_TARGET_AWARE_SELECTIVE_FED_META=1 但未找到已有 checkpoint: {selective_meta_path}"
                )
            progress_log(f"✓ 跳过Target-Aware Selective Fed Meta，复用已有 checkpoint: {selective_meta_path}")
            continue
        if not os.path.exists(target_aware_pretrain_path):
            raise FileNotFoundError(
                f"Target-Aware Selective Fed Meta缺少预训练 checkpoint: {target_aware_pretrain_path}"
            )

        proxy_input_tensor, proxy_target_tensor, proxy_weight_tensor = build_target_aware_selective_proxy_tensors(station_id)
        progress_log(
            f"  target_proxy_windows={proxy_input_tensor.shape[0]} "
            f"(normal_highwind_cap={TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_MAX_WINDOWS})"
        )
        if proxy_input_tensor.shape[0] == 0:
            raise ValueError(f"Target station {station_id} has no proxy windows for selective fed validation")

        if TARGET_AWARE_SELECTIVE_FED_META_INIT_MODEL_DIR:
            selective_init_path = resolve_init_model_path(
                TARGET_AWARE_SELECTIVE_FED_META_INIT_MODEL_DIR,
                f"model_fore_train_task_query_target_aware_selective_fed_meta_station{station_id}.pth",
            )
            if not os.path.exists(selective_init_path):
                raise FileNotFoundError(
                    f"Target-Aware Selective Fed Meta station {station_id} resume checkpoint missing: {selective_init_path}"
                )
            progress_log(f"✓ Target-Aware Selective Fed Meta station {station_id} 从 checkpoint 初始化: {selective_init_path}")
            current_state = torch.load(selective_init_path, map_location=device)
        else:
            current_state = torch.load(target_aware_pretrain_path, map_location=device)
        selective_tag = f"target_aware_selective_fed_meta_station{station_id}"
        epoch_offset = max(0, int(TARGET_AWARE_SELECTIVE_FED_META_EPOCH_OFFSET))
        total_epochs = epoch_offset + PROPOSED_META_EPOCHS
        if epoch_offset:
            progress_log(
                f"  Target-Aware Selective Fed Meta resume_epoch_offset={epoch_offset}, "
                f"total_epoch_after_run={total_epochs}"
            )
        convergence_record = initialize_convergence_record(
            stage_type="target_aware_selective_fed_meta",
            stage_id=selective_tag,
            total_epochs=total_epochs,
            patience=CONVERGENCE_PATIENCE_META,
        )

        for i_t in range(PROPOSED_META_EPOCHS):
            global_epoch = epoch_offset + i_t
            self_state, self_query_loss = run_meta_support_query_update(
                base_state_dict=current_state,
                sample_station_ids=[station_id],
                meta_tag=f"{selective_tag}_self",
                epoch_idx=global_epoch,
                use_cdrm=use_cdrm,
                cdrm_weight=TARGET_AWARE_META_CDRM_WEIGHT,
                train_all_params=False,
                disable_lwp=False,
                target_aware=True,
                target_station_id=station_id,
            )
            self_delta = compute_state_delta(self_state, current_state)
            self_proxy_loss = evaluate_weighted_proxy_loss(
                self_state,
                proxy_input_tensor,
                proxy_target_tensor,
                proxy_weight_tensor,
            )

            source_candidates = []
            source_diagnostics = []
            for source_station_id in station_ids:
                if source_station_id == station_id:
                    continue
                source_state, source_query_loss = run_meta_support_query_update(
                    base_state_dict=current_state,
                    sample_station_ids=[source_station_id],
                    meta_tag=f"{selective_tag}_source{source_station_id}",
                    epoch_idx=global_epoch,
                    use_cdrm=use_cdrm,
                    cdrm_weight=TARGET_AWARE_META_CDRM_WEIGHT,
                    train_all_params=False,
                    disable_lwp=False,
                    target_aware=True,
                    target_station_id=station_id,
                )
                source_delta = compute_state_delta(source_state, current_state)
                best_source_record = None
                for alpha in alpha_grid:
                    candidate_state = apply_weighted_state_deltas(
                        self_state,
                        [(source_delta, alpha)],
                    )
                    candidate_proxy_loss = evaluate_weighted_proxy_loss(
                        candidate_state,
                        proxy_input_tensor,
                        proxy_target_tensor,
                        proxy_weight_tensor,
                    )
                    gain = compute_selective_fed_meta_gain(self_proxy_loss, candidate_proxy_loss)
                    record = {
                        "source_station_id": source_station_id,
                        "alpha": float(alpha),
                        "source_query_loss": float(source_query_loss),
                        "proxy_loss": candidate_proxy_loss,
                        "gain": gain,
                    }
                    if (
                        best_source_record is None
                        or (
                            gain is not None
                            and best_source_record["gain"] is not None
                            and gain > best_source_record["gain"]
                        )
                        or (gain is not None and best_source_record["gain"] is None)
                    ):
                        best_source_record = {**record, "source_delta": source_delta}
                if best_source_record is not None:
                    accepted = (
                        best_source_record["gain"] is not None
                        and best_source_record["gain"] > TARGET_AWARE_SELECTIVE_FED_GAIN_MARGIN
                    )
                    best_source_record["accepted"] = accepted
                    source_diagnostics.append({
                        key: value
                        for key, value in best_source_record.items()
                        if key != "source_delta"
                    })
                    if accepted:
                        source_candidates.append(best_source_record)

            source_candidates.sort(key=lambda item: float(item["gain"]), reverse=True)
            selected_sources = source_candidates[:top_k]
            if selected_sources:
                gain_values = [
                    max(float(item["gain"]), 0.0) ** max(float(TARGET_AWARE_SELECTIVE_FED_GAIN_GAMMA), 0.0)
                    for item in selected_sources
                ]
                gain_sum = sum(gain_values)
                if gain_sum <= 0.0:
                    weighted_source_deltas = [
                        (item["source_delta"], float(item["alpha"]) / len(selected_sources))
                        for item in selected_sources
                    ]
                else:
                    total_alpha_cap = max(alpha_grid)
                    weighted_source_deltas = [
                        (item["source_delta"], total_alpha_cap * gain_weight / gain_sum)
                        for item, gain_weight in zip(selected_sources, gain_values)
                    ]
                current_state = apply_weighted_state_deltas(self_state, weighted_source_deltas)
            else:
                current_state = copy.deepcopy(self_state)

            aggregate_proxy_loss = evaluate_weighted_proxy_loss(
                current_state,
                proxy_input_tensor,
                proxy_target_tensor,
                proxy_weight_tensor,
            )
            if aggregate_proxy_loss is None:
                aggregate_proxy_loss = float(self_query_loss)

            torch.save(current_state, selective_support_path)
            torch.save(current_state, selective_meta_path)
            writer2.add_scalar(
                f"loss_mse_target_aware_selective_fed_meta_proxy_station{station_id}",
                float(aggregate_proxy_loss),
                global_epoch,
            )
            if self_proxy_loss is not None:
                writer2.add_scalar(
                    f"loss_mse_target_aware_selective_fed_meta_self_proxy_station{station_id}",
                    float(self_proxy_loss),
                    global_epoch,
                )
            update_convergence_record(convergence_record, global_epoch, aggregate_proxy_loss)

            if should_log_epoch(i_t, PROPOSED_META_EPOCHS, interval=META_LOG_INTERVAL):
                progress_log(
                    f"  target_aware_selective_fed[target={station_id}] "
                    f"epoch={global_epoch + 1}/{total_epochs} "
                    f"self_proxy_loss={self_proxy_loss} "
                    f"aggregate_proxy_loss={aggregate_proxy_loss} "
                    f"accepted_sources={[item['source_station_id'] for item in selected_sources]} "
                    f"source_diagnostics={source_diagnostics}"
                )

        register_convergence_record(convergence_record)
        progress_log(f"✓ Target-Aware Selective Fed Meta完成: {selective_meta_path}")


def run_fed_normal_meta_training():
    if not (ENABLE_FED_NORMAL_META_PROPOSED or ENABLE_SELECTIVE_FED_NORMAL_META):
        return
    if not (USE_FEDERATION and not USE_PSEUDO_FED):
        return

    for station_id in station_ids:
        print("\n" + "=" * 70)
        print(f"开始Fed-Normal-Meta元训练: target station {station_id}")
        print("=" * 70)

        local_pretrain_path = get_local_pretrain_model_path(station_id)
        fed_normal_meta_support_path = get_fed_normal_meta_support_model_path(station_id)
        fed_normal_meta_path = get_fed_normal_meta_model_path(station_id)
        fed_normal_meta_best_support_path = get_fed_normal_meta_best_support_model_path(station_id)
        fed_normal_meta_best_path = get_fed_normal_meta_best_model_path(station_id)
        if SKIP_FED_NORMAL_META:
            checkpoint_path = fed_normal_meta_best_path if FED_NORMAL_META_USE_BEST else fed_normal_meta_path
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"SKIP_FED_NORMAL_META=1 但未找到已有 checkpoint: {checkpoint_path}"
                )
            progress_log(f"✓ 跳过Fed-Normal-Meta，复用已有 checkpoint: {checkpoint_path}")
            continue

        fed_normal_meta_weights = compute_fed_normal_meta_station_weights(station_id, station_ids)
        progress_log(f"  aggregation_weights={fed_normal_meta_weights}")
        current_state = torch.load(local_pretrain_path, map_location=device)
        fed_normal_meta_tag = f"fed_normal_meta_station{station_id}"
        convergence_record = initialize_convergence_record(
            stage_type="fed_normal_meta",
            stage_id=fed_normal_meta_tag,
            total_epochs=PROPOSED_META_EPOCHS,
            patience=CONVERGENCE_PATIENCE_META,
        )
        fed_normal_meta_best_state = None
        fed_normal_meta_best_loss = None
        fed_normal_meta_best_epoch = None

        for i_t in range(PROPOSED_META_EPOCHS):
            fed_normal_meta_client_states = []
            weighted_query_loss = 0.0

            client_station_order = [station_id] + [
                source_station_id for source_station_id in station_ids
                if source_station_id != station_id
            ]
            for client_station_id in client_station_order:
                client_weight = float(fed_normal_meta_weights.get(client_station_id, 0.0))
                if client_weight <= 0.0:
                    continue
                client_state, client_query_loss = run_meta_support_query_update(
                    base_state_dict=current_state,
                    sample_station_ids=[client_station_id],
                    meta_tag=f"{fed_normal_meta_tag}_client{client_station_id}",
                    epoch_idx=i_t,
                    use_cdrm=True,
                    train_all_params=False,
                    disable_lwp=False,
                )
                fed_normal_meta_client_states.append((client_state, client_weight))
                weighted_query_loss += client_weight * float(client_query_loss)

            if ENABLE_SELECTIVE_FED_NORMAL_META:
                client_state_map = {
                    client_station_order[idx]: state_dict
                    for idx, (state_dict, _) in enumerate(fed_normal_meta_client_states)
                }
                proxy_input_tensor = torch.tensor(
                    all_stations_full_data[station_id]["normal_proxy_input"],
                    dtype=torch.float32,
                )
                proxy_target_tensor = torch.tensor(
                    all_stations_full_data[station_id]["normal_proxy_target"],
                    dtype=torch.float32,
                )
                self_state = client_state_map[station_id]
                self_proxy_loss = evaluate_target_proxy_loss(
                    self_state,
                    proxy_input_tensor,
                    proxy_target_tensor,
                    station_id=station_id,
                )
                source_gain_diagnostics = []
                accepted_source_states = []
                for source_station_id in client_station_order:
                    if source_station_id == station_id:
                        continue
                    source_state = client_state_map[source_station_id]
                    source_proxy_loss = evaluate_target_proxy_loss(
                        source_state,
                        proxy_input_tensor,
                        proxy_target_tensor,
                        station_id=station_id,
                    )
                    source_gain = compute_selective_fed_meta_gain(self_proxy_loss, source_proxy_loss)
                    accepted = (
                        source_gain is not None
                        and source_gain > SELECTIVE_FED_META_GAIN_MARGIN
                    )
                    source_weight = 0.0
                    if accepted:
                        source_weight = max(float(source_gain), 0.0) ** max(SELECTIVE_FED_META_GAIN_GAMMA, 0.0)
                        if source_weight > 0.0:
                            accepted_source_states.append((source_state, source_weight))
                    source_gain_diagnostics.append(
                        {
                            "source_station_id": source_station_id,
                            "proxy_loss": source_proxy_loss,
                            "gain": source_gain,
                            "accepted": accepted,
                            "weight_base": source_weight,
                        }
                    )
                current_state = aggregate_selective_fed_normal_meta_states(
                    self_state=self_state,
                    accepted_source_states=accepted_source_states,
                    self_floor=SELECTIVE_FED_META_SELF_FLOOR,
                )
                aggregate_proxy_loss = evaluate_target_proxy_loss(
                    current_state,
                    proxy_input_tensor,
                    proxy_target_tensor,
                    station_id=station_id,
                )
                weighted_query_loss = (
                    float(aggregate_proxy_loss)
                    if aggregate_proxy_loss is not None
                    else (float(self_proxy_loss) if self_proxy_loss is not None else weighted_query_loss)
                )
                accepted_sources = [
                    diagnostic["source_station_id"]
                    for diagnostic in source_gain_diagnostics
                    if diagnostic["accepted"]
                ]
                progress_log(
                    f"  selective_fed_meta[target={station_id}] "
                    f"epoch={i_t + 1}/{PROPOSED_META_EPOCHS} "
                    f"self_proxy_loss={self_proxy_loss} "
                    f"accepted_sources={accepted_sources} "
                    f"gains={[diagnostic['gain'] for diagnostic in source_gain_diagnostics]}"
                )
            else:
                current_state = aggregate_fed_normal_meta_states(fed_normal_meta_client_states)
            if (
                fed_normal_meta_best_loss is None
                or weighted_query_loss < (fed_normal_meta_best_loss - CONVERGENCE_MIN_DELTA)
            ):
                fed_normal_meta_best_loss = float(weighted_query_loss)
                fed_normal_meta_best_epoch = int(i_t) + 1
                fed_normal_meta_best_state = copy.deepcopy(current_state)
            torch.save(current_state, fed_normal_meta_support_path)
            torch.save(current_state, fed_normal_meta_path)
            update_convergence_record(convergence_record, i_t, weighted_query_loss)

            if should_log_epoch(i_t, PROPOSED_META_EPOCHS, interval=META_LOG_INTERVAL):
                progress_log(
                    f"  收敛追踪[fed_normal_meta:fed_normal_meta_station{station_id}] "
                    f"epoch={i_t + 1}/{PROPOSED_META_EPOCHS} "
                    f"weighted_query_mse={weighted_query_loss:.6f}"
                )

        if FED_NORMAL_META_SAVE_BEST and fed_normal_meta_best_state is not None:
            torch.save(fed_normal_meta_best_state, fed_normal_meta_best_support_path)
            torch.save(fed_normal_meta_best_state, fed_normal_meta_best_path)
            convergence_record["saved_best_checkpoint"] = True
            convergence_record["best_checkpoint_path"] = fed_normal_meta_best_path
            convergence_record["restored_best_checkpoint"] = bool(FED_NORMAL_META_USE_BEST)
            convergence_record["restored_best_epoch"] = fed_normal_meta_best_epoch
            convergence_record["restored_best_loss"] = fed_normal_meta_best_loss
            progress_log(
                f"  ✓ 保存Fed-Normal-Meta best checkpoint: {fed_normal_meta_best_path}, "
                f"epoch={fed_normal_meta_best_epoch}, "
                f"best_weighted_query_mse={fed_normal_meta_best_loss:.6f}"
            )
        else:
            convergence_record["saved_best_checkpoint"] = False
            convergence_record["restored_best_checkpoint"] = False

        register_convergence_record(convergence_record)
        progress_log(f"✓ Fed-Normal-Meta元训练完成: {fed_normal_meta_path}")


if USE_FEDERATION and not USE_PSEUDO_FED:
    run_local_meta_training()
    run_target_aware_meta_training()
    run_target_aware_selective_fed_meta_training()
    run_fed_normal_meta_training()
else:
    run_shared_meta_training()


## test_task_support
legacy_extreme_model_multiplier = 0 if SKIP_LEGACY_EXTREME_ADAPTATION else (4 if TRAIN_META_ONLY_BASELINE else 3)
target_aware_selective_ft_multiplier = 1 if ENABLE_TARGET_AWARE_SELECTIVE_FED_LOCAL_FT else 0
fedtl_ft_multiplier = 1 if ENABLE_FEDTL_FT else 0
few_shot_model_count = (
    0 if SKIP_EXTREME_ADAPTATION_STAGE
    else len(station_ids) * num_extreme_classes * (
        legacy_extreme_model_multiplier + target_aware_selective_ft_multiplier + fedtl_ft_multiplier
    )
)
if SKIP_EXTREME_ADAPTATION_STAGE:
    print("##################################################################——————————NoFT协议：跳过test_task_support/Few-shot适应——————————############################################################")
else:
    print(f"##################################################################——————————test_task_support（Few-shot适应：共{few_shot_model_count}个模型）——————————############################################################")

# ========== [联邦修改] 为所有场站的所有极端天气类别训练个性化模型 ==========
all_personalized_models = {}  # 存储所有个性化模型
ft_sweep_records = []

def extract_extreme_windows_for_station_class(station_id, class_idx, split="support"):
    if split == "test":
        nwp_extre_st = all_stations_full_data[station_id]['nwp_test_extre']
        p_extre_st = all_stations_full_data[station_id]['p_test_extre']
    else:
        nwp_extre_st = all_stations_full_data[station_id]['nwp_extre']
        p_extre_st = all_stations_full_data[station_id]['p_extre']

    nwp_windows = None
    num_samples = 0
    for i_nwp in range(np.size(nwp_extre_st, axis=1)):
        nwp_data = nwp_extre_st[0, i_nwp][0, class_idx]
        num_samples = nwp_data.shape[0] // len_realp
        nwp_reshaped = nwp_data[:num_samples * len_realp].reshape(num_samples, len_realp, 1)
        if i_nwp == 0:
            nwp_windows = nwp_reshaped
        else:
            nwp_windows = np.concatenate((nwp_windows, nwp_reshaped), axis=2)

    p_data = p_extre_st[0, class_idx]
    num_samples = p_data.shape[0] // len_realp
    power_windows = p_data[:num_samples * len_realp].reshape(num_samples, len_realp, 1)

    if nwp_windows is None:
        nwp_windows = np.empty((0, len_realp, dem_realc), dtype=np.float32)
    return nwp_windows.astype(np.float32), power_windows.astype(np.float32)


def parse_ft_sweep_epochs(max_epoch):
    raw_value = FT_SWEEP_EPOCHS.strip()
    if not raw_value:
        return []
    parsed_epochs = []
    for item in raw_value.split(","):
        value = item.strip()
        if not value:
            continue
        epoch_value = int(value)
        if epoch_value < 0:
            continue
        if epoch_value <= max_epoch:
            parsed_epochs.append(epoch_value)
    return sorted(set(parsed_epochs))


def calc_basic_forecast_metrics(true_events, pred_events):
    true_events = np.asarray(true_events, dtype=np.float32)
    pred_events = np.asarray(pred_events, dtype=np.float32)
    if true_events.size == 0:
        return {"nMAE_%": None, "nRMSE_%": None, "WD_%": None}
    err = true_events - pred_events
    nmae = float(np.mean(np.mean(np.abs(err), axis=1)) * 100.0)
    nrmse = float(np.mean(np.sqrt(np.mean(err ** 2, axis=1))) * 100.0)
    wd = float(np.mean(np.mean(np.abs(np.sort(true_events, axis=1) - np.sort(pred_events, axis=1)), axis=1)) * 100.0)
    return {"nMAE_%": nmae, "nRMSE_%": nrmse, "WD_%": wd}


def predict_state_dict_windows(state_dict, input_tensor):
    if input_tensor.shape[0] == 0:
        return np.empty((0, len_realp), dtype=np.float32)
    model_fore_test_task_query.load_state_dict(copy.deepcopy(state_dict))
    model_fore_test_task_query.eval()
    with torch.no_grad():
        outputs = model_fore_test_task_query(input_tensor.to(device))
    return outputs.to(device0).numpy().reshape(input_tensor.shape[0], len_realp)


def build_ft_sweep_eval_fn(model_label, station_id, class_idx, target_payload):
    test_nwp_windows, test_power_windows = extract_extreme_windows_for_station_class(station_id, class_idx, split="test")
    test_input_tensor = torch.tensor(test_nwp_windows, dtype=torch.float32)
    test_target_tensor = torch.tensor(test_power_windows, dtype=torch.float32)
    true_test_events = test_power_windows.reshape(test_power_windows.shape[0], len_realp) if test_power_windows.shape[0] else np.empty((0, len_realp), dtype=np.float32)
    class_label = extreme_eval_labels[class_idx] if class_idx < len(extreme_eval_labels) else f"Class{class_idx + 1}"

    def _eval(state_dict, epoch_value, checkpoint_path):
        pred_test_events = predict_state_dict_windows(state_dict, test_input_tensor) if FT_SWEEP_EVAL_TEST else np.empty_like(true_test_events)
        test_metrics = calc_basic_forecast_metrics(true_test_events, pred_test_events) if FT_SWEEP_EVAL_TEST else {"nMAE_%": None, "nRMSE_%": None, "WD_%": None}
        record = {
            "Model": model_label,
            "Station": station_id,
            "Extreme_Class_Index": int(class_idx),
            "Extreme_Class": class_label,
            "FT_Epoch": int(epoch_value),
            "Adapt_Windows": int(target_payload["adapt_input"].shape[0]),
            "Val_Windows": int(target_payload["val_input"].shape[0]),
            "All_Support_Windows": int(target_payload["all_input"].shape[0]),
            "Test_Windows": int(test_input_tensor.shape[0]),
            "Adapt_MSE": evaluate_state_dict_loss(state_dict, target_payload["adapt_input"], target_payload["adapt_target"]),
            "Val_MSE": evaluate_state_dict_loss(state_dict, target_payload["val_input"], target_payload["val_target"]),
            "SupportAll_MSE": evaluate_state_dict_loss(state_dict, target_payload["all_input"], target_payload["all_target"]),
            "Test_MSE": evaluate_state_dict_loss(state_dict, test_input_tensor, test_target_tensor) if FT_SWEEP_EVAL_TEST else None,
            "Test_nMAE_%": test_metrics["nMAE_%"],
            "Test_nRMSE_%": test_metrics["nRMSE_%"],
            "Test_WD_%": test_metrics["WD_%"],
            "Checkpoint_Path": checkpoint_path or "",
        }
        ft_sweep_records.append(record)

    return _eval


def make_ft_sweep_checkpoint_path(model_tag, station_id, class_idx, epoch_value):
    return resolve_model_path(
        f"model_fore_station{station_id}_extreme{class_idx}_{model_tag}_ft_epoch{epoch_value}.pth"
    )


def export_ft_sweep_records():
    if not ft_sweep_records:
        return
    output_path = FT_SWEEP_OUTPUT_PATH
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fieldnames = list(ft_sweep_records[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ft_sweep_records)
    print(f"✓ FT sweep记录已保存: {output_path}")


def split_extreme_adapt_val(nwp_windows, power_windows, adapt_ratio=EXTREME_ADAPT_RATIO):
    nwp_array = np.asarray(nwp_windows, dtype=np.float32)
    power_array = np.asarray(power_windows, dtype=np.float32)

    if nwp_array.shape[0] == 0:
        empty_nwp = np.empty((0, len_realp, dem_realc), dtype=np.float32)
        empty_power = np.empty((0, len_realp, 1), dtype=np.float32)
        return {
            "adapt_nwp": empty_nwp,
            "adapt_power": empty_power,
            "val_nwp": empty_nwp,
            "val_power": empty_power,
            "full_window_count": 0,
        }

    if nwp_array.shape[0] == 1:
        empty_nwp = np.empty((0, nwp_array.shape[1], nwp_array.shape[2]), dtype=np.float32)
        empty_power = np.empty((0, power_array.shape[1], power_array.shape[2]), dtype=np.float32)
        return {
            "adapt_nwp": nwp_array,
            "adapt_power": power_array,
            "val_nwp": empty_nwp,
            "val_power": empty_power,
            "full_window_count": 1,
        }

    val_window_count = min(max(1, int(round((1.0 - float(adapt_ratio)) * nwp_array.shape[0]))), nwp_array.shape[0] - 1)
    adapt_window_count = nwp_array.shape[0] - val_window_count
    return {
        "adapt_nwp": nwp_array[:adapt_window_count],
        "adapt_power": power_array[:adapt_window_count],
        "val_nwp": nwp_array[adapt_window_count:],
        "val_power": power_array[adapt_window_count:],
        "full_window_count": int(nwp_array.shape[0]),
    }


def resolve_extreme_target_adapt_max_windows(class_idx=None):
    if EXTREME_TARGET_ADAPT_MAX_WINDOWS_BY_CLASS.strip():
        class_caps = [cap.strip() for cap in EXTREME_TARGET_ADAPT_MAX_WINDOWS_BY_CLASS.split(",")]
        if class_idx is not None and 0 <= int(class_idx) < len(class_caps) and class_caps[int(class_idx)] != "":
            return max(0, int(class_caps[int(class_idx)]))
    return max(0, int(EXTREME_TARGET_ADAPT_MAX_WINDOWS))


def resolve_classwise_override(raw_values, class_idx, caster, default_value):
    if not str(raw_values).strip():
        return default_value
    class_values = [value.strip() for value in str(raw_values).split(",")]
    if class_idx is None or not (0 <= int(class_idx) < len(class_values)):
        return default_value
    raw_value = class_values[int(class_idx)]
    if raw_value == "":
        return default_value
    return caster(raw_value)


def resolve_extreme_weight_beta_self(class_idx=None):
    return float(
        resolve_classwise_override(
            EXTREME_WEIGHT_BETA_SELF_BY_CLASS,
            class_idx,
            float,
            EXTREME_WEIGHT_BETA_SELF,
        )
    )


def resolve_extreme_source_hard_gate(class_idx=None):
    return bool(
        resolve_classwise_override(
            EXTREME_SOURCE_HARD_GATE_BY_CLASS,
            class_idx,
            lambda value: str(value).strip() not in ("0", "false", "False"),
            EXTREME_SOURCE_HARD_GATE,
        )
    )


def resolve_extreme_source_min_target_gain(class_idx=None):
    return float(
        resolve_classwise_override(
            EXTREME_SOURCE_MIN_TARGET_GAIN_BY_CLASS,
            class_idx,
            float,
            EXTREME_SOURCE_MIN_TARGET_GAIN,
        )
    )


def resolve_extreme_source_gain_weight_eta(class_idx=None):
    return float(
        resolve_classwise_override(
            EXTREME_SOURCE_GAIN_WEIGHT_ETA_BY_CLASS,
            class_idx,
            float,
            EXTREME_SOURCE_GAIN_WEIGHT_ETA,
        )
    )


def resolve_extreme_proposed_val_fallback(class_idx=None):
    return bool(
        resolve_classwise_override(
            EXTREME_PROPOSED_VAL_FALLBACK_BY_CLASS,
            class_idx,
            lambda value: str(value).strip() not in ("0", "false", "False"),
            EXTREME_PROPOSED_VAL_FALLBACK,
        )
    )


def resolve_extreme_proposed_val_fallback_margin(class_idx=None):
    return float(
        resolve_classwise_override(
            EXTREME_PROPOSED_VAL_FALLBACK_MARGIN_BY_CLASS,
            class_idx,
            float,
            EXTREME_PROPOSED_VAL_FALLBACK_MARGIN,
        )
    )


def resolve_extreme_source_top_k(class_idx=None):
    return int(
        resolve_classwise_override(
            EXTREME_SOURCE_TOP_K_BY_CLASS,
            class_idx,
            int,
            0,
        )
    )


def resolve_extreme_force_local_fallback(class_idx=None):
    return bool(
        resolve_classwise_override(
            EXTREME_FORCE_LOCAL_FALLBACK_BY_CLASS,
            class_idx,
            lambda value: str(value).strip() not in ("0", "false", "False"),
            False,
        )
    )


def apply_target_adapt_kshot_limit(split_payload, class_idx=None):
    """Limit only target-station adapt windows; source windows keep the normal split."""
    max_windows = int(resolve_extreme_target_adapt_max_windows(class_idx))
    if max_windows <= 0:
        return split_payload

    adapt_nwp = split_payload["adapt_nwp"]
    adapt_power = split_payload["adapt_power"]
    if adapt_nwp.shape[0] <= max_windows:
        return split_payload

    limited_payload = copy.deepcopy(split_payload)
    heldout_nwp = adapt_nwp[max_windows:]
    heldout_power = adapt_power[max_windows:]
    limited_payload["adapt_nwp"] = adapt_nwp[:max_windows]
    limited_payload["adapt_power"] = adapt_power[:max_windows]
    limited_payload["val_nwp"] = np.concatenate((heldout_nwp, split_payload["val_nwp"]), axis=0)
    limited_payload["val_power"] = np.concatenate((heldout_power, split_payload["val_power"]), axis=0)
    limited_payload["target_adapt_max_windows"] = max_windows
    return limited_payload


def to_tensor_payload(split_payload):
    return {
        "adapt_input": torch.tensor(split_payload["adapt_nwp"], dtype=torch.float32),
        "adapt_target": torch.tensor(split_payload["adapt_power"], dtype=torch.float32),
        "val_input": torch.tensor(split_payload["val_nwp"], dtype=torch.float32),
        "val_target": torch.tensor(split_payload["val_power"], dtype=torch.float32),
        "all_input": torch.tensor(
            np.concatenate((split_payload["adapt_nwp"], split_payload["val_nwp"]), axis=0),
            dtype=torch.float32,
        ),
        "all_target": torch.tensor(
            np.concatenate((split_payload["adapt_power"], split_payload["val_power"]), axis=0),
            dtype=torch.float32,
        ),
        "full_window_count": int(split_payload["full_window_count"]),
    }


def evaluate_state_dict_loss(state_dict, input_tensor, target_tensor):
    if input_tensor.shape[0] == 0 or target_tensor.shape[0] == 0:
        return None
    model_fore_test_task_query.load_state_dict(copy.deepcopy(state_dict))
    model_fore_test_task_query.eval()
    with torch.no_grad():
        outputs = model_fore_test_task_query(input_tensor.to(device))
        loss_value = loss_fn_1(outputs, target_tensor.to(device))
    if not torch.isfinite(loss_value):
        return None
    return float(loss_value.item())


def has_non_finite_state_dict(state_dict):
    for value in state_dict.values():
        if torch.is_tensor(value) and not torch.isfinite(value).all().item():
            return True
    return False


def compute_anchor_regularization(model_instance, anchor_state_dict):
    if EXTREME_ANCHOR_REG_LAMBDA <= 0 or anchor_state_dict is None:
        return torch.zeros((), dtype=torch.float32, device=device)

    anchor_loss = torch.zeros((), dtype=torch.float32, device=device)
    matched_param_count = 0
    for name, param in model_instance.named_parameters():
        if not param.requires_grad:
            continue
        anchor_tensor = anchor_state_dict.get(name)
        if anchor_tensor is None or not torch.is_tensor(anchor_tensor) or not torch.is_floating_point(anchor_tensor):
            continue
        anchor_tensor = anchor_tensor.to(device=device, dtype=param.dtype)
        anchor_loss = anchor_loss + torch.mean((param - anchor_tensor) ** 2)
        matched_param_count += 1

    if matched_param_count == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    return EXTREME_ANCHOR_REG_LAMBDA * anchor_loss / matched_param_count


def adapt_state_dict(
    base_state_dict,
    adapt_input_tensor,
    adapt_target_tensor,
    epochs,
    log_tag=None,
    model_label=None,
    anchor_state_dict=None,
    checkpoint_epochs=None,
    checkpoint_path_builder=None,
    sweep_eval_fn=None,
):
    if adapt_input_tensor.shape[0] == 0 or adapt_target_tensor.shape[0] == 0:
        return copy.deepcopy(base_state_dict), None

    checkpoint_epochs = set(checkpoint_epochs or [])

    def record_sweep_checkpoint(state_dict, epoch_value):
        if epoch_value not in checkpoint_epochs:
            return
        checkpoint_path = None
        if checkpoint_path_builder is not None:
            checkpoint_path = checkpoint_path_builder(epoch_value)
            if FT_SWEEP_SAVE_CHECKPOINTS:
                save_state_dict(state_dict, checkpoint_path)
        if sweep_eval_fn is not None:
            sweep_eval_fn(state_dict, epoch_value, checkpoint_path)

    record_sweep_checkpoint(copy.deepcopy(base_state_dict), 0)

    if epochs <= 0:
        return copy.deepcopy(base_state_dict), evaluate_state_dict_loss(base_state_dict, adapt_input_tensor, adapt_target_tensor)

    if anchor_state_dict is None:
        anchor_state_dict = base_state_dict

    model_fore_test_task_support.load_state_dict(copy.deepcopy(base_state_dict))
    last_finite_state = copy.deepcopy(base_state_dict)
    optimizer = torch.optim.Adam(
        model_fore_test_task_support.get_trainable_params(), lr=FEW_SHOT_LR, betas=(0.5, 0.999)
    )
    adapt_input_device = adapt_input_tensor.to(device)
    adapt_target_device = adapt_target_tensor.to(device)
    final_loss = None
    convergence_record = initialize_convergence_record(
        stage_type="few_shot",
        stage_id=log_tag or (model_label or "few_shot"),
        total_epochs=epochs,
        patience=CONVERGENCE_PATIENCE_FEW_SHOT,
    )
    convergence_record["non_finite_loss"] = False
    convergence_record["non_finite_state"] = False

    for i in range(epochs):
        model_fore_test_task_support.train()
        outputs = model_fore_test_task_support(adapt_input_device)
        loss_mse = loss_fn_1(outputs, adapt_target_device)
        anchor_loss = compute_anchor_regularization(model_fore_test_task_support, anchor_state_dict)
        loss_total = loss_mse + anchor_loss
        if not torch.isfinite(loss_total):
            convergence_record["non_finite_loss"] = True
            progress_log(
                f"      [{model_label or 'few_shot'}] non-finite loss at epoch {i + 1}; "
                "restoring last finite state"
            )
            model_fore_test_task_support.load_state_dict(copy.deepcopy(last_finite_state))
            break
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        current_state = copy.deepcopy(model_fore_test_task_support.state_dict())
        if has_non_finite_state_dict(current_state):
            convergence_record["non_finite_state"] = True
            progress_log(
                f"      [{model_label or 'few_shot'}] non-finite state at epoch {i + 1}; "
                "restoring last finite state"
            )
            model_fore_test_task_support.load_state_dict(copy.deepcopy(last_finite_state))
            break
        last_finite_state = current_state
        final_loss = float(loss_mse.item())
        update_convergence_record(convergence_record, i, final_loss)
        record_sweep_checkpoint(current_state, i + 1)

        log_interval = FEW_SHOT_LOG_INTERVAL if log_tag is not None else max(1, min(20, epochs))
        if log_tag is not None and should_log_epoch(i, epochs, interval=log_interval):
            if model_label is not None:
                print(
                    f"      [{model_label}] [Epoch {i+1}/{epochs}] "
                    f"[loss_mse: {loss_mse.item():.6f}] [anchor_loss: {anchor_loss.item():.6f}]"
                )
            writer1.add_scalar(f"loss_penalty_{log_tag}", anchor_loss.item(), i)
            writer2.add_scalar(f"loss_mse_{log_tag}", loss_mse.item(), i)

    model_fore_test_task_support.eval()
    register_convergence_record(convergence_record)
    return copy.deepcopy(model_fore_test_task_support.state_dict()), final_loss


def run_few_shot_adaptation(base_model_path, save_path, log_tag, model_label, test_input_tensor, test_target_tensor):
    adapted_state, _ = adapt_state_dict(
        base_state_dict=torch.load(base_model_path, map_location=device),
        adapt_input_tensor=test_input_tensor,
        adapt_target_tensor=test_target_tensor,
        epochs=FEW_SHOT_EPOCHS,
        log_tag=log_tag,
        model_label=model_label,
    )
    torch.save(adapted_state, save_path)
    print(f"    ✓ 保存({model_label}): {save_path}")


def compute_window_self_losses(base_model_path, nwp_windows, power_windows):
    if nwp_windows.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    base_state_dict = torch.load(base_model_path, map_location=device)
    losses = []
    for window_index in range(nwp_windows.shape[0]):
        input_tensor = torch.tensor(nwp_windows[window_index:window_index + 1], dtype=torch.float32)
        target_tensor = torch.tensor(power_windows[window_index:window_index + 1], dtype=torch.float32)
        loss_value = evaluate_state_dict_loss(base_state_dict, input_tensor, target_tensor)
        losses.append(1e6 if loss_value is None else float(loss_value))
    return np.asarray(losses, dtype=np.float32)


def apply_source_quality_gate(source_model_path, nwp_windows, power_windows):
    quality_losses = compute_window_self_losses(source_model_path, nwp_windows, power_windows)
    if quality_losses.size == 0:
        return {
            "nwp_windows": np.empty((0, len_realp, dem_realc), dtype=np.float32),
            "power_windows": np.empty((0, len_realp, 1), dtype=np.float32),
            "selected_indices": np.empty((0,), dtype=np.int64),
            "quality_losses": quality_losses,
        }

    keep_count = min(
        quality_losses.size,
        max(EXTREME_MIN_EFFECTIVE_WINDOWS, int(math.ceil(EXTREME_SOURCE_QUALITY_KEEP_RATIO * quality_losses.size))),
    )
    selected_indices = np.argsort(quality_losses)[:keep_count]
    selected_indices = np.sort(selected_indices.astype(np.int64))
    return {
        "nwp_windows": np.asarray(nwp_windows, dtype=np.float32)[selected_indices],
        "power_windows": np.asarray(power_windows, dtype=np.float32)[selected_indices],
        "selected_indices": selected_indices,
        "quality_losses": quality_losses[selected_indices],
    }


def compute_target_conditioned_usefulness(shared_init_state_dict, gated_nwp_windows, gated_power_windows, target_val_input, target_val_target):
    if gated_nwp_windows.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    base_target_val_loss = evaluate_state_dict_loss(shared_init_state_dict, target_val_input, target_val_target)
    if base_target_val_loss is None:
        return np.empty((0,), dtype=np.float32)

    usefulness_scores = []
    for window_index in range(gated_nwp_windows.shape[0]):
        source_input = torch.tensor(gated_nwp_windows[window_index:window_index + 1], dtype=torch.float32)
        source_target = torch.tensor(gated_power_windows[window_index:window_index + 1], dtype=torch.float32)
        adapted_state, _ = adapt_state_dict(
            base_state_dict=shared_init_state_dict,
            adapt_input_tensor=source_input,
            adapt_target_tensor=source_target,
            epochs=EXTREME_USEFULNESS_ADAPT_STEPS,
        )
        adapted_target_val_loss = evaluate_state_dict_loss(adapted_state, target_val_input, target_val_target)
        if adapted_target_val_loss is None:
            usefulness_scores.append(-1e6)
        else:
            usefulness_scores.append(float(base_target_val_loss - adapted_target_val_loss))
    return np.asarray(usefulness_scores, dtype=np.float32)


def select_effective_source_windows(shared_init_state_dict, source_station_id, class_idx, target_payload):
    source_nwp_windows, source_power_windows = extract_extreme_windows_for_station_class(source_station_id, class_idx)
    gated_payload = apply_source_quality_gate(
        source_model_path=get_local_meta_model_path(source_station_id),
        nwp_windows=source_nwp_windows,
        power_windows=source_power_windows,
    )
    gated_nwp_windows = gated_payload["nwp_windows"]
    gated_power_windows = gated_payload["power_windows"]

    usefulness_scores = compute_target_conditioned_usefulness(
        shared_init_state_dict=shared_init_state_dict,
        gated_nwp_windows=gated_nwp_windows,
        gated_power_windows=gated_power_windows,
        target_val_input=target_payload["val_input"],
        target_val_target=target_payload["val_target"],
    )

    positive_indices = np.where(usefulness_scores > 0.0)[0]
    if positive_indices.size == 0:
        return {
            "candidate_window_count": int(source_nwp_windows.shape[0]),
            "gated_window_count": int(gated_nwp_windows.shape[0]),
            "effective_window_count": 0,
            "nwp_windows": np.empty((0, len_realp, dem_realc), dtype=np.float32),
            "power_windows": np.empty((0, len_realp, 1), dtype=np.float32),
            "usefulness_scores": np.empty((0,), dtype=np.float32),
        }

    borrow_budget = min(
        positive_indices.size,
        max(
            EXTREME_MIN_EFFECTIVE_WINDOWS,
            int(math.ceil(EXTREME_SOURCE_BORROW_BUDGET_GAMMA * max(1, target_payload["full_window_count"]))),
        ),
    )
    ranked_positive = positive_indices[np.argsort(usefulness_scores[positive_indices])[-borrow_budget:]]
    ranked_positive = np.sort(ranked_positive.astype(np.int64))
    return {
        "candidate_window_count": int(source_nwp_windows.shape[0]),
        "gated_window_count": int(gated_nwp_windows.shape[0]),
        "effective_window_count": int(ranked_positive.size),
        "nwp_windows": gated_nwp_windows[ranked_positive],
        "power_windows": gated_power_windows[ranked_positive],
        "usefulness_scores": usefulness_scores[ranked_positive],
    }


def compute_sample_count_reliability(sample_count):
    return float(math.log1p(max(0, int(sample_count))))


def compute_query_reliability(query_loss):
    if query_loss is None:
        return 1.0
    return float(math.exp(-EXTREME_WEIGHT_TAU_Q * float(query_loss)))


def compute_target_transferability(target_val_loss):
    if target_val_loss is None:
        return 1.0
    return float(math.exp(-EXTREME_WEIGHT_TAU_T * float(target_val_loss)))


def compute_source_target_gain_weight(target_gain, class_idx=None):
    gain_weight_eta = resolve_extreme_source_gain_weight_eta(class_idx)
    if gain_weight_eta <= 0:
        return 1.0
    target_gain = max(0.0, float(target_gain))
    if target_gain <= 0:
        return 0.0
    return float(target_gain ** gain_weight_eta)


def aggregate_state_dicts(weighted_states):
    aggregate_state = copy.deepcopy(weighted_states[0][0])
    for name in aggregate_state.keys():
        reference_tensor = aggregate_state[name]
        if not torch.is_floating_point(reference_tensor):
            aggregate_state[name] = reference_tensor.clone()
            continue
        weighted_sum = None
        for state_dict, alpha in weighted_states:
            contrib = state_dict[name].detach().float().cpu() * float(alpha)
            weighted_sum = contrib if weighted_sum is None else weighted_sum + contrib
        aggregate_state[name] = weighted_sum.type_as(reference_tensor)
    return aggregate_state


def aggregate_extreme_updates_uniform(self_update_payload, source_update_payloads):
    all_payloads = [self_update_payload] + list(source_update_payloads)
    uniform_alpha = 1.0 / len(all_payloads)
    aggregate_state = aggregate_state_dicts(
        [(payload["state_dict"], uniform_alpha) for payload in all_payloads]
    )
    return aggregate_state, {
        payload["station_id"]: uniform_alpha for payload in all_payloads
    }


def aggregate_extreme_updates_weighted(target_station_id, self_update_payload, source_update_payloads, class_idx=None):
    if not source_update_payloads:
        return copy.deepcopy(self_update_payload["state_dict"]), {target_station_id: 1.0}

    source_scores = {}
    for payload in source_update_payloads:
        m_k_to_s_c = compute_sample_count_reliability(payload["effective_window_count"])
        q_k_to_s_c = compute_query_reliability(payload["source_val_loss"])
        t_k_to_s_c = compute_target_transferability(payload["target_val_loss"])
        target_gain_weight = compute_source_target_gain_weight(payload.get("target_gain", 0.0), class_idx=class_idx)
        source_scores[payload["station_id"]] = float(
            (m_k_to_s_c ** EXTREME_WEIGHT_LAMBDA)
            * (q_k_to_s_c ** EXTREME_WEIGHT_MU)
            * (t_k_to_s_c ** EXTREME_WEIGHT_NU)
            * target_gain_weight
        )

    total_source_score = float(sum(source_scores.values()))
    beta_self = min(max(float(resolve_extreme_weight_beta_self(class_idx)), 0.0), 1.0)
    remaining_weight = 1.0 - beta_self

    if total_source_score <= 0:
        uniform_source_alpha = remaining_weight / len(source_update_payloads)
        weighted_states = [(self_update_payload["state_dict"], beta_self)]
        weight_map = {target_station_id: beta_self}
        for payload in source_update_payloads:
            weighted_states.append((payload["state_dict"], uniform_source_alpha))
            weight_map[payload["station_id"]] = uniform_source_alpha
        return aggregate_state_dicts(weighted_states), weight_map

    weighted_states = [(self_update_payload["state_dict"], beta_self)]
    weight_map = {target_station_id: beta_self}
    for payload in source_update_payloads:
        alpha_value = remaining_weight * source_scores[payload["station_id"]] / total_source_score
        weighted_states.append((payload["state_dict"], alpha_value))
        weight_map[payload["station_id"]] = alpha_value
    return aggregate_state_dicts(weighted_states), weight_map


def save_state_dict(state_dict, save_path):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(state_dict, save_path)
    return save_path


def run_target_refinement(base_state_dict, target_payload, log_tag, model_label):
    return adapt_state_dict(
        base_state_dict=base_state_dict,
        adapt_input_tensor=target_payload["adapt_input"],
        adapt_target_tensor=target_payload["adapt_target"],
        epochs=EXTREME_TARGET_REFINEMENT_EPOCHS,
        log_tag=log_tag,
        model_label=model_label,
        anchor_state_dict=base_state_dict,
    )


def select_proposed_final_state_by_target_validation(
    proposed_state_dict,
    lmt_state_dict,
    target_payload,
    station_id,
    class_idx,
):
    proposed_val_loss = evaluate_state_dict_loss(
        proposed_state_dict,
        target_payload["val_input"],
        target_payload["val_target"],
    )
    lmt_val_loss = evaluate_state_dict_loss(
        lmt_state_dict,
        target_payload["val_input"],
        target_payload["val_target"],
    )
    selection_info = {
        "selected": "proposed",
        "proposed_val_loss": proposed_val_loss,
        "lmt_val_loss": lmt_val_loss,
        "margin": resolve_extreme_proposed_val_fallback_margin(class_idx),
    }
    if not resolve_extreme_proposed_val_fallback(class_idx):
        return copy.deepcopy(proposed_state_dict), selection_info

    should_fallback = False
    if proposed_val_loss is None and lmt_val_loss is not None:
        should_fallback = True
    elif proposed_val_loss is not None and lmt_val_loss is not None:
        should_fallback = (
            float(proposed_val_loss)
            > float(lmt_val_loss) - float(resolve_extreme_proposed_val_fallback_margin(class_idx))
        )

    if should_fallback:
        selection_info["selected"] = "fallback_lmt"
        selected_state_dict = lmt_state_dict
    else:
        selected_state_dict = proposed_state_dict

    print(
        f"    [proposed_val_select] station={station_id}, class={class_idx}, "
        f"selected={selection_info['selected']}, proposed_val_loss={proposed_val_loss}, "
        f"lmt_val_loss={lmt_val_loss}, margin={resolve_extreme_proposed_val_fallback_margin(class_idx)}"
    )
    return copy.deepcopy(selected_state_dict), selection_info


def passes_source_target_gate(source_target_val_loss, reference_target_val_loss, class_idx=None):
    if not resolve_extreme_source_hard_gate(class_idx):
        return True
    if source_target_val_loss is None or reference_target_val_loss is None:
        return False
    target_gain = float(reference_target_val_loss) - float(source_target_val_loss)
    return target_gain >= float(resolve_extreme_source_min_target_gain(class_idx))


def build_source_update_payloads(
    shared_init_state_dict,
    target_station_id,
    class_idx,
    target_payload,
    log_prefix,
    reference_target_val_loss=None,
):
    source_update_payloads = []
    for source_station_id in station_ids:
        if source_station_id == target_station_id:
            continue
        source_screening_payload = select_effective_source_windows(
            shared_init_state_dict=shared_init_state_dict,
            source_station_id=source_station_id,
            class_idx=class_idx,
            target_payload=target_payload,
        )
        print(
            f"    [screen:{log_prefix}] source_station={source_station_id} -> target_station={target_station_id}, "
            f"candidate={source_screening_payload['candidate_window_count']}, "
            f"gated={source_screening_payload['gated_window_count']}, "
            f"effective={source_screening_payload['effective_window_count']}"
        )
        if source_screening_payload["effective_window_count"] == 0:
            continue

        source_split_payload = split_extreme_adapt_val(
            source_screening_payload["nwp_windows"],
            source_screening_payload["power_windows"],
        )
        source_payload = to_tensor_payload(source_split_payload)
        source_state_dict, source_final_loss = adapt_state_dict(
            base_state_dict=shared_init_state_dict,
            adapt_input_tensor=source_payload["adapt_input"],
            adapt_target_tensor=source_payload["adapt_target"],
            epochs=FEW_SHOT_EPOCHS,
            log_tag=f"{log_prefix}_station{source_station_id}_to_{target_station_id}_class{class_idx}",
            model_label=f"{log_prefix}:{source_station_id}->target{target_station_id}",
        )
        source_val_loss = (
            evaluate_state_dict_loss(
                source_state_dict,
                source_payload["val_input"],
                source_payload["val_target"],
            ) if source_payload["val_input"].shape[0] > 0 else source_final_loss
        )
        source_target_val_loss = evaluate_state_dict_loss(
            source_state_dict,
            target_payload["val_input"],
            target_payload["val_target"],
        )
        source_target_gain = (
            max(0.0, float(reference_target_val_loss) - float(source_target_val_loss))
            if source_target_val_loss is not None and reference_target_val_loss is not None
            else 0.0
        )
        if not passes_source_target_gate(
            source_target_val_loss,
            reference_target_val_loss,
            class_idx=class_idx,
        ):
            print(
                f"    [hard_gate:{log_prefix}] skip source_station={source_station_id} -> "
                f"target_station={target_station_id}, source_target_val_loss={source_target_val_loss}, "
                f"reference_target_val_loss={reference_target_val_loss}, "
                f"min_gain={resolve_extreme_source_min_target_gain(class_idx)}"
            )
            continue
        source_update_payloads.append({
            "station_id": source_station_id,
            "state_dict": source_state_dict,
            "effective_window_count": source_screening_payload["effective_window_count"],
            "source_val_loss": source_val_loss,
            "target_val_loss": source_target_val_loss,
            "target_gain": source_target_gain,
        })
    top_k = resolve_extreme_source_top_k(class_idx)
    if top_k > 0 and len(source_update_payloads) > top_k:
        source_update_payloads.sort(
            key=lambda payload: (
                float(payload.get("target_gain", 0.0)),
                -float(payload.get("target_val_loss", 1e9) if payload.get("target_val_loss", None) is not None else 1e9),
                float(payload.get("effective_window_count", 0)),
            ),
            reverse=True,
        )
        source_update_payloads = source_update_payloads[:top_k]
    return source_update_payloads


for station_id in ([] if SKIP_EXTREME_ADAPTATION_STAGE else station_ids):
    print(f"\n{'='*70}")
    print(f"场站 {station_id} 的Few-shot适应")
    print(f"{'='*70}")
    
    for i_class in range(num_extreme_classes):
        class_label = extreme_eval_labels[i_class] if i_class < len(extreme_eval_labels) else f"Class{i_class + 1}"
        print(f"\n  极端天气类别 {i_class+1} ({class_label}):")

        nwp_extre_class, p_extre_class = extract_extreme_windows_for_station_class(station_id, i_class)
        num_samples = nwp_extre_class.shape[0]
        target_split_payload = split_extreme_adapt_val(nwp_extre_class, p_extre_class)
        target_split_payload = apply_target_adapt_kshot_limit(target_split_payload, class_idx=i_class)
        target_payload = to_tensor_payload(target_split_payload)
        target_adapt_max_windows = resolve_extreme_target_adapt_max_windows(i_class)

        print(f"    样本数: {num_samples}")
        if target_adapt_max_windows > 0:
            print(f"    target K-shot adapt窗口上限: {target_adapt_max_windows}")
        print(
            f"    训练轮数: {FEW_SHOT_EPOCHS}, "
            f"few-shot loss={'CDRM+MSE' if FEW_SHOT_USE_CDRM else 'MSE'}"
        )
        print(
            f"    adapt/val: {target_payload['adapt_input'].shape[0]} / "
            f"{target_payload['val_input'].shape[0]}"
        )

        if ENABLE_TARGET_AWARE_SELECTIVE_FED_LOCAL_FT:
            if not ENABLE_TARGET_AWARE_SELECTIVE_FED_META:
                raise RuntimeError(
                    "Target-Aware Selective Fed-Meta + Local FT requires ENABLE_TARGET_AWARE_SELECTIVE_FED_META=1"
                )
            target_aware_selective_base_path = get_target_aware_selective_fed_meta_model_path(station_id)
            if not os.path.exists(target_aware_selective_base_path):
                raise FileNotFoundError(
                    "Target-Aware Selective Fed-Meta + Local FT缺少NoFT checkpoint: "
                    f"{target_aware_selective_base_path}"
                )
            target_aware_selective_ft_state_dict, _ = adapt_state_dict(
                base_state_dict=torch.load(target_aware_selective_base_path, map_location=device),
                adapt_input_tensor=target_payload["all_input"],
                adapt_target_tensor=target_payload["all_target"],
                epochs=FEW_SHOT_EPOCHS,
                log_tag=f"target_aware_selective_fed_local_ft_station{station_id}_class{i_class}",
                model_label="Target-Aware Selective Fed-Meta + Local FT",
            )
            target_aware_selective_ft_model_name = get_target_aware_selective_fed_local_ft_model_path(station_id, i_class)
            save_state_dict(target_aware_selective_ft_state_dict, target_aware_selective_ft_model_name)
            print(f"    ✓ 保存(Target-Aware Selective Fed-Meta + Local FT): {target_aware_selective_ft_model_name}")
            all_personalized_models[
                f'target_aware_selective_fed_local_ft_{station_id}_class{i_class}'
            ] = target_aware_selective_ft_model_name

        if ENABLE_FEDTL_FT:
            if not os.path.exists(PRETRAIN_MODEL_PATH):
                raise FileNotFoundError(
                    "FedTL-FT缺少联邦预训练checkpoint: "
                    f"{PRETRAIN_MODEL_PATH}"
                )
            fedtl_sweep_epochs = parse_ft_sweep_epochs(FEW_SHOT_EPOCHS) if ENABLE_FT_SWEEP else []
            fedtl_ft_state_dict, _ = adapt_state_dict(
                base_state_dict=torch.load(PRETRAIN_MODEL_PATH, map_location=device),
                adapt_input_tensor=target_payload["all_input"],
                adapt_target_tensor=target_payload["all_target"],
                epochs=FEW_SHOT_EPOCHS,
                log_tag=f"fedtl_ft_station{station_id}_class{i_class}",
                model_label="FedTL-FT",
                checkpoint_epochs=fedtl_sweep_epochs,
                checkpoint_path_builder=(
                    lambda epoch_value, station_id=station_id, i_class=i_class: make_ft_sweep_checkpoint_path(
                        "fedtl", station_id, i_class, epoch_value
                    )
                ),
                sweep_eval_fn=build_ft_sweep_eval_fn("FedTL-FT", station_id, i_class, target_payload)
                if ENABLE_FT_SWEEP
                else None,
            )
            fedtl_ft_model_name = get_fedtl_ft_model_path(station_id, i_class)
            save_state_dict(fedtl_ft_state_dict, fedtl_ft_model_name)
            print(f"    ✓ 保存(FedTL-FT): {fedtl_ft_model_name}")
            all_personalized_models[f'fedtl_ft_{station_id}_class{i_class}'] = fedtl_ft_model_name

        if SKIP_LEGACY_EXTREME_ADAPTATION:
            continue

        local_base_model_path = get_local_extreme_base_model_path(station_id)
        proposed_base_model_path = get_proposed_a_base_model_path(station_id)
        local_shared_init_state = torch.load(local_base_model_path, map_location=device)
        if proposed_base_model_path == local_base_model_path:
            proposed_shared_init_state = local_shared_init_state
        else:
            proposed_shared_init_state = torch.load(proposed_base_model_path, map_location=device)

        lmt_sweep_epochs = parse_ft_sweep_epochs(FEW_SHOT_EPOCHS) if ENABLE_FT_SWEEP else []
        lmt_state_dict, lmt_final_loss = adapt_state_dict(
            base_state_dict=local_shared_init_state,
            adapt_input_tensor=target_payload["all_input"],
            adapt_target_tensor=target_payload["all_target"],
            epochs=FEW_SHOT_EPOCHS,
            log_tag=f"lmt_station{station_id}_class{i_class}",
            model_label="LMT",
            checkpoint_epochs=lmt_sweep_epochs,
            checkpoint_path_builder=(
                lambda epoch_value, station_id=station_id, i_class=i_class: make_ft_sweep_checkpoint_path(
                    "original_local", station_id, i_class, epoch_value
                )
            ),
            sweep_eval_fn=build_ft_sweep_eval_fn("Original-Local-FT", station_id, i_class, target_payload)
            if ENABLE_FT_SWEEP
            else None,
        )
        lmt_model_name = get_lmt_model_path(station_id, i_class)
        save_state_dict(lmt_state_dict, lmt_model_name)
        print(f"    ✓ 保存(LMT): {lmt_model_name}")
        all_personalized_models[f'lmt_{station_id}_class{i_class}'] = lmt_model_name

        if ENABLE_FED_NORMAL_META_PROPOSED or ENABLE_SELECTIVE_FED_NORMAL_META:
            fed_meta_local_ft_state_dict, _ = adapt_state_dict(
                base_state_dict=proposed_shared_init_state,
                adapt_input_tensor=target_payload["all_input"],
                adapt_target_tensor=target_payload["all_target"],
                epochs=FEW_SHOT_EPOCHS,
                log_tag=f"fed_meta_local_ft_station{station_id}_class{i_class}",
                model_label="Fed-Meta + Local FT",
            )
            fed_meta_local_ft_model_name = get_fed_meta_local_ft_model_path(station_id, i_class)
            save_state_dict(fed_meta_local_ft_state_dict, fed_meta_local_ft_model_name)
            print(f"    ✓ 保存(Fed-Meta + Local FT): {fed_meta_local_ft_model_name}")
            all_personalized_models[f'fed_meta_local_ft_{station_id}_class{i_class}'] = fed_meta_local_ft_model_name

        if resolve_extreme_force_local_fallback(i_class):
            print(f"    [classwise_fallback] station={station_id}, class={i_class}: force local fallback -> reuse LMT")
            fedavg_model_name = get_extreme_fedavg_model_path(station_id, i_class)
            save_state_dict(lmt_state_dict, fedavg_model_name)
            all_personalized_models[f'extreme_fedavg_{station_id}_class{i_class}'] = fedavg_model_name
            proposed_model_name = get_proposed_a_model_path(station_id, i_class)
            save_state_dict(lmt_state_dict, proposed_model_name)
            all_personalized_models[f'proposed_a_{station_id}_class{i_class}'] = proposed_model_name
            print(f"    ✓ 保存(Extreme-FedAvg/Proposed-A fallback=LMT): {fedavg_model_name}, {proposed_model_name}")
            continue

        target_val_loss = evaluate_state_dict_loss(
            lmt_state_dict,
            target_payload["val_input"],
            target_payload["val_target"],
        )
        self_update_payload = {
            "station_id": station_id,
            "state_dict": lmt_state_dict,
            "effective_window_count": target_payload["full_window_count"],
            "source_val_loss": lmt_final_loss if lmt_final_loss is not None else target_val_loss,
            "target_val_loss": target_val_loss,
        }

        source_update_payloads = build_source_update_payloads(
            shared_init_state_dict=local_shared_init_state,
            target_station_id=station_id,
            class_idx=i_class,
            target_payload=target_payload,
            log_prefix="extreme_source",
            reference_target_val_loss=self_update_payload["target_val_loss"],
        )

        fedavg_aggregate_state, fedavg_weight_map = aggregate_extreme_updates_uniform(
            self_update_payload,
            source_update_payloads,
        )
        fedavg_final_state, _ = run_target_refinement(
            base_state_dict=fedavg_aggregate_state,
            target_payload=target_payload,
            log_tag=f"extreme_fedavg_station{station_id}_class{i_class}",
            model_label="Extreme-FedAvg:target_refine",
        )
        fedavg_model_name = get_extreme_fedavg_model_path(station_id, i_class)
        save_state_dict(fedavg_final_state, fedavg_model_name)
        print(f"    ✓ 保存(Extreme-FedAvg): {fedavg_model_name} weights={fedavg_weight_map}")
        all_personalized_models[f'extreme_fedavg_{station_id}_class{i_class}'] = fedavg_model_name

        if proposed_base_model_path == local_base_model_path:
            proposed_self_update_payload = self_update_payload
            proposed_source_update_payloads = source_update_payloads
        else:
            proposed_self_state_dict, proposed_self_final_loss = adapt_state_dict(
                base_state_dict=proposed_shared_init_state,
                adapt_input_tensor=target_payload["adapt_input"],
                adapt_target_tensor=target_payload["adapt_target"],
                epochs=FEW_SHOT_EPOCHS,
                log_tag=f"proposed_self_station{station_id}_class{i_class}",
                model_label="Proposed-A:self_update",
            )
            proposed_target_val_loss = evaluate_state_dict_loss(
                proposed_self_state_dict,
                target_payload["val_input"],
                target_payload["val_target"],
            )
            proposed_self_update_payload = {
                "station_id": station_id,
                "state_dict": proposed_self_state_dict,
                "effective_window_count": target_payload["full_window_count"],
                "source_val_loss": (
                    proposed_self_final_loss
                    if proposed_self_final_loss is not None
                    else proposed_target_val_loss
                ),
                "target_val_loss": proposed_target_val_loss,
            }
            proposed_source_update_payloads = build_source_update_payloads(
                shared_init_state_dict=proposed_shared_init_state,
                target_station_id=station_id,
                class_idx=i_class,
                target_payload=target_payload,
                log_prefix="proposed_source",
                reference_target_val_loss=proposed_self_update_payload["target_val_loss"],
            )

        proposed_aggregate_state, proposed_weight_map = aggregate_extreme_updates_weighted(
            target_station_id=station_id,
            self_update_payload=proposed_self_update_payload,
            source_update_payloads=proposed_source_update_payloads,
            class_idx=i_class,
        )
        proposed_final_state, _ = run_target_refinement(
            base_state_dict=proposed_aggregate_state,
            target_payload=target_payload,
            log_tag=f"proposed_a_station{station_id}_class{i_class}",
            model_label="Proposed-A:target_refine",
        )
        selected_proposed_state, proposed_selection_info = select_proposed_final_state_by_target_validation(
            proposed_state_dict=proposed_final_state,
            lmt_state_dict=lmt_state_dict,
            target_payload=target_payload,
            station_id=station_id,
            class_idx=i_class,
        )
        proposed_model_name = get_proposed_a_model_path(station_id, i_class)
        save_state_dict(selected_proposed_state, proposed_model_name)
        print(
            f"    ✓ 保存(Proposed-A): {proposed_model_name} "
            f"weights={proposed_weight_map} selection={proposed_selection_info}"
        )
        all_personalized_models[f'proposed_a_{station_id}_class{i_class}'] = proposed_model_name

        # Meta-only：同口径执行 step-11 few-shot，确保与论文消融对齐
        if TRAIN_META_ONLY_BASELINE:
            meta_only_model_name = get_meta_only_extreme_model_path(station_id, i_class)
            run_few_shot_adaptation(
                base_model_path=get_local_meta_only_model_path(station_id) if USE_FEDERATION and not USE_PSEUDO_FED else META_ONLY_MODEL_PATH,
                save_path=meta_only_model_name,
                log_tag=f"meta_only_station{station_id}_class{i_class}",
                model_label="Meta-only",
                test_input_tensor=target_payload["all_input"],
                test_target_tensor=target_payload["all_target"]
            )
            all_personalized_models[f'meta_only_{station_id}_class{i_class}'] = meta_only_model_name

writer1.close()
writer2.close()
export_ft_sweep_records()
if SKIP_EXTREME_ADAPTATION_STAGE:
    print("\n✓ NoFT协议已跳过Few-shot/Local-FT/Extreme-FedAvg/Proposed-A兼容模型生成")
else:
    print(f"\n✓ Few-shot训练完成，生成了 {len(all_personalized_models)} 个个性化模型")
# ========== [联邦修改] 保存所有场站的测试结果 ==========
print("\n" + "="*70)
print("生成所有场站的测试预测结果")
print("="*70)

all_test_results = {}  # 存储所有场站的测试结果
fed_meta_local_ft_label = (
    "Selective Fed-Normal-Meta + Local FT"
    if ENABLE_SELECTIVE_FED_NORMAL_META
    else "Vanilla Fed-Normal-Meta + Local FT"
)

for station_id in station_ids:
    print(f"\n场站 {station_id} 预测:")
    
    # 获取该场站的测试数据
    Test_input_c_st = torch.tensor(all_stations_full_data[station_id]['test_input'], dtype=torch.float32)
    Test_target_p_st = all_stations_full_data[station_id]['test_target']
    
    all_test_results[station_id] = {
        'test_input': Test_input_c_st,
        'test_target': Test_target_p_st,
        'predictions': {}
    }
    
    # 预测：该场站的4个极端天气模型
    for i_class in range(num_extreme_classes):
        model_paths = {
            "pretrain": get_local_pretrain_model_path(station_id),
            "local_meta_noft": get_local_meta_model_path(station_id),
        }
        if ENABLE_TARGET_AWARE_META_NOFT:
            model_paths["target_aware_pretrain"] = get_target_aware_pretrain_model_path(station_id)
            model_paths["target_aware_meta_noft"] = get_target_aware_meta_model_path(station_id)
        if ENABLE_TARGET_AWARE_SELECTIVE_FED_META:
            model_paths["target_aware_selective_fed_meta_noft"] = get_target_aware_selective_fed_meta_model_path(station_id)
        if ENABLE_FED_NORMAL_META_PROPOSED or ENABLE_SELECTIVE_FED_NORMAL_META:
            model_paths["fed_meta_noft"] = get_fed_normal_meta_model_path(station_id)
        if not SKIP_EXTREME_ADAPTATION_STAGE:
            if ENABLE_TARGET_AWARE_SELECTIVE_FED_LOCAL_FT:
                model_paths[
                    "target_aware_selective_fed_local_ft"
                ] = get_target_aware_selective_fed_local_ft_model_path(station_id, i_class)
            if ENABLE_FEDTL_FT:
                model_paths["fedtl_ft"] = get_fedtl_ft_model_path(station_id, i_class)
            if not SKIP_LEGACY_EXTREME_ADAPTATION:
                model_paths["lmt"] = get_lmt_model_path(station_id, i_class)
                if ENABLE_FED_NORMAL_META_PROPOSED or ENABLE_SELECTIVE_FED_NORMAL_META:
                    model_paths["fed_meta_local_ft"] = get_fed_meta_local_ft_model_path(station_id, i_class)
                model_paths["extreme_fedavg"] = get_extreme_fedavg_model_path(station_id, i_class)
                model_paths["proposed_a"] = get_proposed_a_model_path(station_id, i_class)

        for model_key, model_name in model_paths.items():
            if not os.path.exists(model_name):
                continue
            model_fore_test_task_query.load_state_dict(torch.load(model_name, map_location=device))
            with torch.no_grad():
                Test_input_device = Test_input_c_st.to(device)
                Test_output = model_fore_test_task_query(Test_input_device)
                test_output = Test_output.to(device0)
                test_output_np = np.array(test_output.reshape(-1,dem_realp))
                all_test_results[station_id]['predictions'][f'{model_key}_extreme_{i_class}'] = test_output_np
        class_label = extreme_eval_labels[i_class] if i_class < len(extreme_eval_labels) else f"Class{i_class + 1}"
        logged_labels = ["Pretrain", "Local-Meta-NoFT"]
        if ENABLE_TARGET_AWARE_META_NOFT:
            logged_labels.extend(["Target-Aware Pretrain", "Target-Aware Meta-NoFT"])
        if ENABLE_TARGET_AWARE_SELECTIVE_FED_META:
            logged_labels.append("Target-Aware Selective Fed-Meta-NoFT")
        if ENABLE_FED_NORMAL_META_PROPOSED or ENABLE_SELECTIVE_FED_NORMAL_META:
            logged_labels.append(
                "Selective Fed-Normal-Meta-NoFT"
                if ENABLE_SELECTIVE_FED_NORMAL_META
                else "Vanilla Fed-Normal-Meta-NoFT"
            )
        if not SKIP_EXTREME_ADAPTATION_STAGE:
            if ENABLE_TARGET_AWARE_SELECTIVE_FED_LOCAL_FT:
                logged_labels.append("Target-Aware Selective Fed-Meta + Local FT")
            if ENABLE_FEDTL_FT:
                logged_labels.append("FedTL-FT")
            if not SKIP_LEGACY_EXTREME_ADAPTATION:
                logged_labels.append("LMT")
                if ENABLE_FED_NORMAL_META_PROPOSED or ENABLE_SELECTIVE_FED_NORMAL_META:
                    logged_labels.append(fed_meta_local_ft_label)
                logged_labels.extend(["Extreme-FedAvg", "Proposed-A"])
        print(f"  ✓ 极端类别{i_class+1}({class_label})（{'/'.join(logged_labels)}）")

# 保存所有结果
print("\n保存所有场站测试结果...")
all_results_dir = os.path.dirname(ALL_STATIONS_TEST_RESULTS_PATH)
if all_results_dir:
    os.makedirs(all_results_dir, exist_ok=True)
scio.savemat(ALL_STATIONS_TEST_RESULTS_PATH, {'all_test_results': all_test_results, 'Cap': Cap})
print(f"✓ 已保存: {ALL_STATIONS_TEST_RESULTS_PATH}")
export_convergence_report(
    CONVERGENCE_REPORT_PATH,
    convergence_records,
    {
        "protocol_name": PROTOCOL_NAME,
        "protocol_data_dir": PROTOCOL_DATA_DIR,
        "protocol_metadata_path": PROTOCOL_METADATA_PATH,
        "artifact_dir": ARTIFACT_DIR,
        "model_output_dir": MODEL_OUTPUT_DIR,
        "base_model_output_dir": BASE_MODEL_OUTPUT_DIR,
        "target_aware_base_model_output_dir": TARGET_AWARE_BASE_MODEL_OUTPUT_DIR,
        "target_aware_selective_fed_base_model_output_dir": TARGET_AWARE_SELECTIVE_FED_BASE_MODEL_OUTPUT_DIR,
        "local_pretrain_init_model_dir": LOCAL_PRETRAIN_INIT_MODEL_DIR,
        "local_meta_init_model_dir": LOCAL_META_INIT_MODEL_DIR,
        "target_aware_pretrain_init_model_dir": TARGET_AWARE_PRETRAIN_INIT_MODEL_DIR,
        "target_aware_meta_init_model_dir": TARGET_AWARE_META_INIT_MODEL_DIR,
        "target_aware_selective_fed_meta_init_model_dir": TARGET_AWARE_SELECTIVE_FED_META_INIT_MODEL_DIR,
        "local_pretrain_epoch_offset": LOCAL_PRETRAIN_EPOCH_OFFSET,
        "local_meta_epoch_offset": LOCAL_META_EPOCH_OFFSET,
        "target_aware_pretrain_epoch_offset": TARGET_AWARE_PRETRAIN_EPOCH_OFFSET,
        "target_aware_meta_epoch_offset": TARGET_AWARE_META_EPOCH_OFFSET,
        "target_aware_selective_fed_meta_epoch_offset": TARGET_AWARE_SELECTIVE_FED_META_EPOCH_OFFSET,
        "logs_train_dir": LOGS_TRAIN_DIR,
        "all_stations_test_results_path": ALL_STATIONS_TEST_RESULTS_PATH,
        "sample_interval_hours": SAMPLE_INTERVAL_HOURS,
        "downsample_offset": DOWNSAMPLE_OFFSET,
        "len_realp": LEN_REALP,
        "points_per_day": POINTS_PER_DAY,
        "window_span_hours": WINDOW_SPAN_HOURS,
        "yearly_protocol_enabled": YEARLY_PROTOCOL_ENABLED,
        "seasonal_protocol_enabled": SEASONAL_PROTOCOL_ENABLED,
        "use_federation": USE_FEDERATION,
            "use_pseudo_fed": USE_PSEUDO_FED,
            "train_meta_only_baseline": TRAIN_META_ONLY_BASELINE,
            "train_pretrain_only": TRAIN_PRETRAIN_ONLY,
            "run_federated_pretrain": RUN_FEDERATED_PRETRAIN,
            "fed_pretrain_algo": FED_PRETRAIN_ALGO,
            "fedavg_local_epochs": FEDAVG_LOCAL_EPOCHS,
            "fedavg_client_weighting": FEDAVG_CLIENT_WEIGHTING,
            "skip_local_pretrain": SKIP_LOCAL_PRETRAIN,
            "skip_local_meta": SKIP_LOCAL_META,
            "enable_fedtl_ft": ENABLE_FEDTL_FT,
            "enable_ft_sweep": ENABLE_FT_SWEEP,
            "ft_sweep_epochs": FT_SWEEP_EPOCHS,
            "ft_sweep_output_path": FT_SWEEP_OUTPUT_PATH,
            "enable_fed_normal_meta_proposed": ENABLE_FED_NORMAL_META_PROPOSED,
        "fed_normal_meta_self_floor": FED_NORMAL_META_SELF_FLOOR,
        "skip_fed_normal_meta": SKIP_FED_NORMAL_META,
        "fed_normal_meta_restore_best": FED_NORMAL_META_RESTORE_BEST,
        "fed_normal_meta_save_best": FED_NORMAL_META_SAVE_BEST,
        "fed_normal_meta_use_best": FED_NORMAL_META_USE_BEST,
        "enable_selective_fed_normal_meta": ENABLE_SELECTIVE_FED_NORMAL_META,
        "selective_fed_meta_proxy_ratio": SELECTIVE_FED_META_PROXY_RATIO,
        "selective_fed_meta_self_floor": SELECTIVE_FED_META_SELF_FLOOR,
        "selective_fed_meta_gain_margin": SELECTIVE_FED_META_GAIN_MARGIN,
        "selective_fed_meta_gain_gamma": SELECTIVE_FED_META_GAIN_GAMMA,
        "hwa_pretrain_loss": HWA_PRETRAIN_LOSS,
        "hwa_meta_loss": HWA_META_LOSS,
        "hwa_selective_proxy_loss": HWA_SELECTIVE_PROXY_LOSS,
        "hwa_wind_feature_index": HWA_WIND_FEATURE_INDEX,
        "hwa_wind_threshold": HWA_WIND_THRESHOLD,
        "hwa_wind_ramp_end": HWA_WIND_RAMP_END,
        "hwa_high_wind_weight": HWA_HIGH_WIND_WEIGHT,
        "hwa_pretrain_windowed": HWA_PRETRAIN_WINDOWED,
        "enable_target_aware_meta_noft": ENABLE_TARGET_AWARE_META_NOFT,
        "skip_target_aware_pretrain": SKIP_TARGET_AWARE_PRETRAIN,
        "skip_target_aware_meta": SKIP_TARGET_AWARE_META,
        "enable_target_aware_selective_fed_meta": ENABLE_TARGET_AWARE_SELECTIVE_FED_META,
        "skip_target_aware_selective_fed_meta": SKIP_TARGET_AWARE_SELECTIVE_FED_META,
        "enable_target_aware_selective_fed_local_ft": ENABLE_TARGET_AWARE_SELECTIVE_FED_LOCAL_FT,
        "skip_legacy_extreme_adaptation": SKIP_LEGACY_EXTREME_ADAPTATION,
        "target_aware_pretrain_hwa_loss": TARGET_AWARE_PRETRAIN_HWA_LOSS,
        "target_aware_meta_cdrm_weight": TARGET_AWARE_META_CDRM_WEIGHT,
        "target_aware_meta_sim_floor": TARGET_AWARE_META_SIM_FLOOR,
        "target_aware_meta_task_weight_eta": TARGET_AWARE_META_TASK_WEIGHT_ETA,
        "target_aware_meta_wind_mean_threshold": TARGET_AWARE_META_WIND_MEAN_THRESHOLD,
        "target_aware_meta_wind_max_threshold": TARGET_AWARE_META_WIND_MAX_THRESHOLD,
        "target_aware_meta_min_extreme_points": TARGET_AWARE_META_MIN_EXTREME_POINTS,
        "target_aware_selective_fed_top_k": TARGET_AWARE_SELECTIVE_FED_TOP_K,
        "target_aware_selective_fed_self_floor": TARGET_AWARE_SELECTIVE_FED_SELF_FLOOR,
        "target_aware_selective_fed_source_alpha_cap": TARGET_AWARE_SELECTIVE_FED_SOURCE_ALPHA_CAP,
        "target_aware_selective_fed_alpha_grid": TARGET_AWARE_SELECTIVE_FED_ALPHA_GRID,
        "target_aware_selective_fed_gain_margin": TARGET_AWARE_SELECTIVE_FED_GAIN_MARGIN,
        "target_aware_selective_fed_gain_gamma": TARGET_AWARE_SELECTIVE_FED_GAIN_GAMMA,
        "target_aware_selective_fed_proxy_normal_max_windows": TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_MAX_WINDOWS,
        "target_aware_selective_fed_proxy_extreme_weight": TARGET_AWARE_SELECTIVE_FED_PROXY_EXTREME_WEIGHT,
        "target_aware_selective_fed_proxy_normal_weight": TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_WEIGHT,
        "global_seed": GLOBAL_SEED,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "proposed_meta_epochs": PROPOSED_META_EPOCHS,
        "meta_only_meta_epochs": META_ONLY_META_EPOCHS,
        "few_shot_epochs": FEW_SHOT_EPOCHS,
        "few_shot_lr": FEW_SHOT_LR,
        "skip_extreme_adaptation_stage": SKIP_EXTREME_ADAPTATION_STAGE,
        "pretrain_log_interval": PRETRAIN_LOG_INTERVAL,
        "meta_log_interval": META_LOG_INTERVAL,
        "few_shot_log_interval": FEW_SHOT_LOG_INTERVAL,
        "extreme_weight_beta_self": EXTREME_WEIGHT_BETA_SELF,
        "extreme_weight_beta_self_by_class": EXTREME_WEIGHT_BETA_SELF_BY_CLASS,
        "extreme_source_borrow_budget_gamma": EXTREME_SOURCE_BORROW_BUDGET_GAMMA,
        "extreme_target_refinement_epochs": EXTREME_TARGET_REFINEMENT_EPOCHS,
        "extreme_target_adapt_max_windows": EXTREME_TARGET_ADAPT_MAX_WINDOWS,
        "extreme_target_adapt_max_windows_by_class": EXTREME_TARGET_ADAPT_MAX_WINDOWS_BY_CLASS,
        "extreme_anchor_reg_lambda": EXTREME_ANCHOR_REG_LAMBDA,
        "extreme_source_hard_gate": EXTREME_SOURCE_HARD_GATE,
        "extreme_source_hard_gate_by_class": EXTREME_SOURCE_HARD_GATE_BY_CLASS,
        "extreme_source_min_target_gain": EXTREME_SOURCE_MIN_TARGET_GAIN,
        "extreme_source_min_target_gain_by_class": EXTREME_SOURCE_MIN_TARGET_GAIN_BY_CLASS,
        "extreme_source_gain_weight_eta": EXTREME_SOURCE_GAIN_WEIGHT_ETA,
        "extreme_source_gain_weight_eta_by_class": EXTREME_SOURCE_GAIN_WEIGHT_ETA_BY_CLASS,
        "extreme_source_top_k_by_class": EXTREME_SOURCE_TOP_K_BY_CLASS,
        "extreme_force_local_fallback_by_class": EXTREME_FORCE_LOCAL_FALLBACK_BY_CLASS,
        "extreme_proposed_val_fallback": EXTREME_PROPOSED_VAL_FALLBACK,
        "extreme_proposed_val_fallback_by_class": EXTREME_PROPOSED_VAL_FALLBACK_BY_CLASS,
        "extreme_proposed_val_fallback_margin": EXTREME_PROPOSED_VAL_FALLBACK_MARGIN,
        "extreme_proposed_val_fallback_margin_by_class": EXTREME_PROPOSED_VAL_FALLBACK_MARGIN_BY_CLASS,
        "enable_convergence_monitor": ENABLE_CONVERGENCE_MONITOR,
    },
)

print("\n" + "="*70)
print("✓✓✓ 训练和测试全部完成！")
if SKIP_EXTREME_ADAPTATION_STAGE:
    print("NoFT协议未生成个性化微调模型")
else:
    summary_labels = []
    if ENABLE_TARGET_AWARE_SELECTIVE_FED_LOCAL_FT:
        summary_labels.append("Target-Aware Selective Fed-Meta + Local FT")
    if not SKIP_LEGACY_EXTREME_ADAPTATION:
        summary_labels.append("LMT")
        if ENABLE_FED_NORMAL_META_PROPOSED or ENABLE_SELECTIVE_FED_NORMAL_META:
            summary_labels.append(fed_meta_local_ft_label)
        summary_labels.extend(["Extreme-FedAvg", "Proposed-A"])
    print(f"生成的个性化微调模型: {len(all_personalized_models)}个（含 {' / '.join(summary_labels)}）")
print("="*70)

# [删除] 原来的单场站保存代码
# 以下代码不再需要
if False:  # 禁用原代码
    test_outputs_query_00= np.empty([1, 6], dtype=object)
    test_outputs_support_00= np.empty([1, 4], dtype=object)
    
    ## save
    for i_model in range(6):
        with torch.no_grad():
            if i_model<4:
                model_fore_test_task_query.load_state_dict(torch.load("model_fore_test_task_support_%d.pth"%(i_model)))
            elif i_model==4:
                model_fore_test_task_query.load_state_dict(torch.load("model_fore_train_task_query.pth" ))
            elif i_model==5:
                model_fore_test_task_query.load_state_dict(torch.load("model_fore_pre.pth" ))
            Test_input_c = Test_input_c.to(device)
            Test_output_query=model_fore_test_task_query(Test_input_c)
            test_outputs_query=Test_output_query.to(device0)
            test_outputs_query_=np.array(test_outputs_query.reshape(-1,dem_realp))
            test_outputs_query_00[0,i_model]=test_outputs_query_
            if i_model < 4:
                test_outputs_support_list=Test_outputs_support_list[i_model].to(device0)
                test_outputs_support_ = np.array(test_outputs_support_list.reshape(-1, dem_realp))
                test_outputs_support_00[0,i_model]=test_outputs_support_
            train_outputs_pre=Train_outputs_pre.to(device0)
            train_outputs_support=Train_outputs_support.to(device0)
            train_outputs_query=Train_outputs_query.to(device0)
            pass  # [联邦修改] 原单场站保存逻辑已被新的多场站逻辑替代
