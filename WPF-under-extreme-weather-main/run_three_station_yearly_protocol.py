#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Launcher for the three-station yearly extreme protocol."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
YEARLY_PROTOCOL_ENABLED = "1"
SEASONAL_PROTOCOL_ENABLED = "0"
THREE_STATION_PROTOCOL_NAME = "three_station_2h_6point_protocol"
SIX_STATION_PROTOCOL_NAME = "six_station_2h_6point_phase_augmented_protocol"
TWO_YEAR_2024_K6_PROTOCOL_NAME = "six_station_2h_6point_two_year_2024_k6_protocol"
TWO_YEAR_2024_ALL_EXTREME_PROTOCOL_NAME = "six_station_2h_6point_two_year_2024_all_extreme_protocol"
HIGH_TEMP_ONLY_SUMMER_PROTOCOL_NAME = "six_station_2h_6point_high_temp_only_summer_protocol"
HIGH_WIND_SPRING_NOFT_PROTOCOL_NAME = "four_client_2h_6point_high_wind_spring_noft_protocol"
SIX_STATION_4H3P_PROTOCOL_NAME = "six_station_4h_3point_phase_augmented_protocol"
HYBRID_4H3P_NORMAL_2H6P_EXTREME_PROTOCOL_NAME = "hybrid_4h3p_normal_2h6p_extreme_protocol"
THREE_STATION_PROTOCOL_DATA_DIR = str(ROOT / "protocol_data" / "2h_6p")
SIX_STATION_PROTOCOL_DATA_DIR = str(ROOT / "protocol_data" / "2h_6p_six_station")
TWO_YEAR_2024_K6_PROTOCOL_DATA_DIR = str(ROOT / "protocol_data" / "two_year_24_k6_six_station")
TWO_YEAR_2024_ALL_EXTREME_PROTOCOL_DATA_DIR = str(ROOT / "protocol_data" / "two_year_24_all_extreme_six_station")
HIGH_TEMP_ONLY_SUMMER_PROTOCOL_DATA_DIR = str(ROOT / "protocol_data" / "high_temp_only_summer_six_station")
HIGH_WIND_SPRING_NOFT_PROTOCOL_DATA_DIR = str(ROOT / "protocol_data" / "high_wind_spring_noft_four_client")
SIX_STATION_4H3P_PROTOCOL_DATA_DIR = str(ROOT / "protocol_data" / "4h_3p_six_station")
SIX_STATION_ARTIFACT_DIR = str(ROOT / "artifacts" / "2h6p_six_station")
TWO_YEAR_2024_K6_ARTIFACT_DIR = str(ROOT / "artifacts" / "two_year_24_k6_six_station")
TWO_YEAR_2024_ALL_EXTREME_ARTIFACT_DIR = str(ROOT / "artifacts" / "two_year_24_all_extreme_six_station")
HIGH_TEMP_ONLY_SUMMER_ARTIFACT_DIR = str(ROOT / "artifacts" / "high_temp_only_summer_six_station")
HIGH_WIND_SPRING_NOFT_ARTIFACT_DIR = str(ROOT / "artifacts" / "high_wind_spring_noft_four_client")
SIX_STATION_4H3P_ARTIFACT_DIR = str(ROOT / "artifacts" / "4h3p_six_station")
HYBRID_4H3P_NORMAL_2H6P_EXTREME_ARTIFACT_DIR = str(ROOT / "artifacts" / "4h3p_normal_2h6p_extreme")
DEFAULT_HYBRID_BASE_MODEL_OUTPUT_DIR = str(
    ROOT / "artifacts" / "4h3p_six_station" / "fed-normal-meta-self08-best-5k" / "models"
)
PROTOCOL_NAME = THREE_STATION_PROTOCOL_NAME
PROTOCOL_DATA_DIR = THREE_STATION_PROTOCOL_DATA_DIR
PROTOCOL_METADATA_PATH = str(Path(PROTOCOL_DATA_DIR) / "protocol_metadata.json")
YEARLY_PROTOCOL_METADATA_PATH = PROTOCOL_METADATA_PATH
SAMPLE_INTERVAL_HOURS = "2"
DOWNSAMPLE_OFFSET = "1"
LEN_REALP = "6"
POINTS_PER_DAY = "12"
PHASE_AUGMENT_STATIONS = "0"
USE_FEDERATION = "1"
RUN_FEDERATED_PRETRAIN = "0"
TRAIN_META_ONLY_BASELINE = "0"
META_SUPPORT_SHOTS = "10"
META_QUERY_SHOTS = "10"
SMOKE_PRESET_NAME = "smoke"
PILOT_PRESET_NAME = "pilot"
PILOT_MEDIUM_PRESET_NAME = "pilot-medium"
FORMAL_PRESET_NAME = "formal-v1"
PILOT_1K_PRESET_NAME = "pilot-1k"
PILOT_5K_PRESET_NAME = "pilot-5k"
FINAL_CANDIDATE_PRESET_NAME = "final-candidate"
REFINE_K1_PRESET_NAME = "refine-k1"
REFINE_K2_PRESET_NAME = "refine-k2"
META_SHOT_REGIME_2X4 = "2x4"
META_SHOT_REGIME_3X3 = "3x3"
PREVIEW_ENV_KEYS = [
    "PROTOCOL_NAME",
    "PROTOCOL_DATA_DIR",
    "PROTOCOL_METADATA_PATH",
    "SAMPLE_INTERVAL_HOURS",
    "DOWNSAMPLE_OFFSET",
    "LEN_REALP",
    "POINTS_PER_DAY",
    "PHASE_AUGMENT_STATIONS",
    "PHASE_AUGMENT_COMPLEMENTARY_OFFSET",
    "ARTIFACT_DIR",
    "MODEL_OUTPUT_DIR",
    "BASE_MODEL_OUTPUT_DIR",
    "TARGET_AWARE_BASE_MODEL_OUTPUT_DIR",
    "LOGS_TRAIN_DIR",
    "CONVERGENCE_REPORT_PATH",
    "ALL_STATIONS_TEST_RESULTS_PATH",
    "RESULTS_OUTPUT_PATH",
    "TASK_RESULTS_OUTPUT_PATH",
    "YEARLY_PROTOCOL_ENABLED",
    "SEASONAL_PROTOCOL_ENABLED",
    "YEARLY_PROTOCOL_METADATA_PATH",
    "HIGH_TEMP_ONLY_SUMMER_PROTOCOL",
    "HIGH_WIND_SPRING_NOFT_PROTOCOL",
    "HIGH_WIND_NORMAL_TRAIN_START",
    "HIGH_WIND_NORMAL_TRAIN_END",
    "USE_FEDERATION",
    "RUN_FEDERATED_PRETRAIN",
    "TRAIN_META_ONLY_BASELINE",
    "SKIP_LOCAL_PRETRAIN",
    "SKIP_LOCAL_META",
    "ENABLE_FED_NORMAL_META_PROPOSED",
    "FED_NORMAL_META_SELF_FLOOR",
    "SKIP_FED_NORMAL_META",
    "FED_NORMAL_META_RESTORE_BEST",
    "FED_NORMAL_META_SAVE_BEST",
    "FED_NORMAL_META_USE_BEST",
    "ENABLE_SELECTIVE_FED_NORMAL_META",
    "SELECTIVE_FED_META_PROXY_RATIO",
    "SELECTIVE_FED_META_SELF_FLOOR",
    "SELECTIVE_FED_META_GAIN_MARGIN",
    "SELECTIVE_FED_META_GAIN_GAMMA",
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
    "TARGET_AWARE_PRETRAIN_HWA_LOSS",
    "TARGET_AWARE_META_CDRM_WEIGHT",
    "TARGET_AWARE_META_SIM_FLOOR",
    "TARGET_AWARE_META_TASK_WEIGHT_ETA",
    "TARGET_AWARE_META_WIND_MEAN_THRESHOLD",
    "TARGET_AWARE_META_WIND_MAX_THRESHOLD",
    "TARGET_AWARE_META_MIN_EXTREME_POINTS",
    "TARGET_AWARE_SELECTIVE_FED_TOP_K",
    "TARGET_AWARE_SELECTIVE_FED_SELF_FLOOR",
    "TARGET_AWARE_SELECTIVE_FED_SOURCE_ALPHA_CAP",
    "TARGET_AWARE_SELECTIVE_FED_ALPHA_GRID",
    "TARGET_AWARE_SELECTIVE_FED_GAIN_MARGIN",
    "TARGET_AWARE_SELECTIVE_FED_GAIN_GAMMA",
    "TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_MAX_WINDOWS",
    "TARGET_AWARE_SELECTIVE_FED_PROXY_EXTREME_WEIGHT",
    "TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_WEIGHT",
    "GLOBAL_SEED",
    "META_TASKS_PER_EPOCH",
    "META_SUPPORT_SHOTS",
    "META_QUERY_SHOTS",
    "EVAL_MODEL_SET",
    "SKIP_EXTREME_ADAPTATION_STAGE",
    "PROPOSED_META_SAMPLER_MODE",
    "PRETRAIN_EPOCHS",
    "PROPOSED_META_EPOCHS",
    "META_ONLY_META_EPOCHS",
    "FEW_SHOT_EPOCHS",
    "EXTREME_TARGET_REFINEMENT_EPOCHS",
    "EXTREME_SUPPORT_WINDOW_CAP",
    "EXTREME_TARGET_ADAPT_MAX_WINDOWS",
    "EXTREME_TARGET_ADAPT_MAX_WINDOWS_BY_CLASS",
    "EXTREME_WEIGHT_BETA_SELF",
    "EXTREME_WEIGHT_BETA_SELF_BY_CLASS",
    "EXTREME_ANCHOR_REG_LAMBDA",
    "EXTREME_SOURCE_HARD_GATE",
    "EXTREME_SOURCE_HARD_GATE_BY_CLASS",
    "EXTREME_SOURCE_MIN_TARGET_GAIN",
    "EXTREME_SOURCE_MIN_TARGET_GAIN_BY_CLASS",
    "EXTREME_SOURCE_GAIN_WEIGHT_ETA",
    "EXTREME_SOURCE_GAIN_WEIGHT_ETA_BY_CLASS",
    "EXTREME_SOURCE_TOP_K_BY_CLASS",
    "EXTREME_FORCE_LOCAL_FALLBACK_BY_CLASS",
    "EXTREME_PROPOSED_VAL_FALLBACK",
    "EXTREME_PROPOSED_VAL_FALLBACK_BY_CLASS",
    "EXTREME_PROPOSED_VAL_FALLBACK_MARGIN",
    "EXTREME_PROPOSED_VAL_FALLBACK_MARGIN_BY_CLASS",
    "PRETRAIN_LOG_INTERVAL",
    "META_LOG_INTERVAL",
    "FEW_SHOT_LOG_INTERVAL",
]


def build_env(
    preset=None,
    smoke=False,
    six_station=False,
    four_hour=False,
    hybrid_extreme_2h=False,
    two_year_2024_k6=False,
    two_year_2024_all_extreme=False,
    high_temp_only_summer=False,
    high_wind_spring_noft=False,
    meta_shot_regime=META_SHOT_REGIME_2X4,
    enable_selective_fed_normal_meta=False,
):
    env = os.environ.copy()
    selected_preset = SMOKE_PRESET_NAME if smoke else preset
    two_year_2024_mode = two_year_2024_k6 or two_year_2024_all_extreme
    six_station_mode = (
        six_station or four_hour or hybrid_extreme_2h or two_year_2024_mode
        or high_temp_only_summer or high_wind_spring_noft
    )
    if high_wind_spring_noft:
        protocol_name = HIGH_WIND_SPRING_NOFT_PROTOCOL_NAME
        protocol_data_dir = HIGH_WIND_SPRING_NOFT_PROTOCOL_DATA_DIR
        artifact_base_dir = HIGH_WIND_SPRING_NOFT_ARTIFACT_DIR
    elif high_temp_only_summer:
        protocol_name = HIGH_TEMP_ONLY_SUMMER_PROTOCOL_NAME
        protocol_data_dir = HIGH_TEMP_ONLY_SUMMER_PROTOCOL_DATA_DIR
        artifact_base_dir = HIGH_TEMP_ONLY_SUMMER_ARTIFACT_DIR
    elif two_year_2024_all_extreme:
        protocol_name = TWO_YEAR_2024_ALL_EXTREME_PROTOCOL_NAME
        protocol_data_dir = TWO_YEAR_2024_ALL_EXTREME_PROTOCOL_DATA_DIR
        artifact_base_dir = TWO_YEAR_2024_ALL_EXTREME_ARTIFACT_DIR
    elif two_year_2024_k6:
        protocol_name = TWO_YEAR_2024_K6_PROTOCOL_NAME
        protocol_data_dir = TWO_YEAR_2024_K6_PROTOCOL_DATA_DIR
        artifact_base_dir = TWO_YEAR_2024_K6_ARTIFACT_DIR
    elif hybrid_extreme_2h:
        protocol_name = HYBRID_4H3P_NORMAL_2H6P_EXTREME_PROTOCOL_NAME
        protocol_data_dir = SIX_STATION_PROTOCOL_DATA_DIR
        artifact_base_dir = HYBRID_4H3P_NORMAL_2H6P_EXTREME_ARTIFACT_DIR
    elif four_hour:
        protocol_name = SIX_STATION_4H3P_PROTOCOL_NAME
        protocol_data_dir = SIX_STATION_4H3P_PROTOCOL_DATA_DIR
        artifact_base_dir = SIX_STATION_4H3P_ARTIFACT_DIR
    elif six_station_mode:
        protocol_name = SIX_STATION_PROTOCOL_NAME
        protocol_data_dir = SIX_STATION_PROTOCOL_DATA_DIR
        artifact_base_dir = SIX_STATION_ARTIFACT_DIR
    else:
        protocol_name = PROTOCOL_NAME
        protocol_data_dir = PROTOCOL_DATA_DIR
        artifact_base_dir = "."
    protocol_metadata_path = str(Path(protocol_data_dir) / "protocol_metadata.json")
    if six_station_mode:
        artifact_dir = env.get("ARTIFACT_DIR", str(Path(artifact_base_dir) / str(selected_preset or "default")))
    else:
        artifact_dir = env.get("ARTIFACT_DIR", ".")
    model_output_dir = env.get("MODEL_OUTPUT_DIR", str(Path(artifact_dir) / "models") if artifact_dir not in ("", ".") else ".")
    base_model_output_dir = env.get(
        "BASE_MODEL_OUTPUT_DIR",
        DEFAULT_HYBRID_BASE_MODEL_OUTPUT_DIR if hybrid_extreme_2h else model_output_dir,
    )
    target_aware_base_model_output_dir = env.get("TARGET_AWARE_BASE_MODEL_OUTPUT_DIR", model_output_dir)
    logs_train_dir = env.get("LOGS_TRAIN_DIR", str(Path(artifact_dir) / "logs_train") if artifact_dir not in ("", ".") else "logs_train")
    default_results_dir = Path(artifact_dir) / "results" if artifact_dir not in ("", ".") else Path(".")
    env.update({
        "PYTHONUNBUFFERED": "1",
        "PROTOCOL_NAME": protocol_name,
        "PROTOCOL_DATA_DIR": protocol_data_dir,
        "PROTOCOL_METADATA_PATH": protocol_metadata_path,
        "SAMPLE_INTERVAL_HOURS": SAMPLE_INTERVAL_HOURS,
        "DOWNSAMPLE_OFFSET": DOWNSAMPLE_OFFSET,
        "LEN_REALP": LEN_REALP,
        "POINTS_PER_DAY": POINTS_PER_DAY,
        "PHASE_AUGMENT_STATIONS": "1" if six_station_mode else PHASE_AUGMENT_STATIONS,
        "PHASE_AUGMENT_COMPLEMENTARY_OFFSET": env.get("PHASE_AUGMENT_COMPLEMENTARY_OFFSET", "0"),
        "ARTIFACT_DIR": artifact_dir,
        "MODEL_OUTPUT_DIR": model_output_dir,
        "BASE_MODEL_OUTPUT_DIR": base_model_output_dir,
        "TARGET_AWARE_BASE_MODEL_OUTPUT_DIR": target_aware_base_model_output_dir,
        "LOGS_TRAIN_DIR": logs_train_dir,
        "CONVERGENCE_REPORT_PATH": env.get(
            "CONVERGENCE_REPORT_PATH",
            str(Path(artifact_dir) / "training_convergence_report.json") if artifact_dir not in ("", ".") else "training_convergence_report.json",
        ),
        "ALL_STATIONS_TEST_RESULTS_PATH": env.get(
            "ALL_STATIONS_TEST_RESULTS_PATH",
            str(Path(artifact_dir) / "all_stations_test_results.mat") if artifact_dir not in ("", ".") else "all_stations_test_results.mat",
        ),
        "RESULTS_OUTPUT_PATH": env.get(
            "RESULTS_OUTPUT_PATH",
            str(default_results_dir / "multi_station_performance.csv"),
        ),
        "TASK_RESULTS_OUTPUT_PATH": env.get(
            "TASK_RESULTS_OUTPUT_PATH",
            str(default_results_dir / "multi_station_performance_task_level.csv"),
        ),
        "YEARLY_PROTOCOL_ENABLED": YEARLY_PROTOCOL_ENABLED,
        "SEASONAL_PROTOCOL_ENABLED": SEASONAL_PROTOCOL_ENABLED,
        "YEARLY_PROTOCOL_METADATA_PATH": protocol_metadata_path,
        "HIGH_TEMP_ONLY_SUMMER_PROTOCOL": "1" if high_temp_only_summer else "0",
        "HIGH_WIND_SPRING_NOFT_PROTOCOL": "1" if high_wind_spring_noft else "0",
        "USE_FEDERATION": USE_FEDERATION,
        "RUN_FEDERATED_PRETRAIN": RUN_FEDERATED_PRETRAIN,
        "TRAIN_META_ONLY_BASELINE": TRAIN_META_ONLY_BASELINE,
        "SKIP_LOCAL_PRETRAIN": env.get("SKIP_LOCAL_PRETRAIN", "0"),
        "SKIP_LOCAL_META": env.get("SKIP_LOCAL_META", "0"),
        "ENABLE_FED_NORMAL_META_PROPOSED": env.get("ENABLE_FED_NORMAL_META_PROPOSED", "0"),
        "FED_NORMAL_META_SELF_FLOOR": env.get("FED_NORMAL_META_SELF_FLOOR", "0.3"),
        "SKIP_FED_NORMAL_META": env.get("SKIP_FED_NORMAL_META", "0"),
        "FED_NORMAL_META_RESTORE_BEST": env.get("FED_NORMAL_META_RESTORE_BEST", "0"),
        "FED_NORMAL_META_SAVE_BEST": env.get("FED_NORMAL_META_SAVE_BEST", "0"),
        "FED_NORMAL_META_USE_BEST": env.get("FED_NORMAL_META_USE_BEST", "0"),
        "ENABLE_SELECTIVE_FED_NORMAL_META": "1" if enable_selective_fed_normal_meta else env.get("ENABLE_SELECTIVE_FED_NORMAL_META", "0"),
        "SELECTIVE_FED_META_PROXY_RATIO": env.get("SELECTIVE_FED_META_PROXY_RATIO", "0.2"),
        "SELECTIVE_FED_META_SELF_FLOOR": env.get("SELECTIVE_FED_META_SELF_FLOOR", "0.5"),
        "SELECTIVE_FED_META_GAIN_MARGIN": env.get("SELECTIVE_FED_META_GAIN_MARGIN", "0.0"),
        "SELECTIVE_FED_META_GAIN_GAMMA": env.get("SELECTIVE_FED_META_GAIN_GAMMA", "1.0"),
        "HWA_PRETRAIN_LOSS": env.get("HWA_PRETRAIN_LOSS", "0"),
        "HWA_META_LOSS": env.get("HWA_META_LOSS", "0"),
        "HWA_SELECTIVE_PROXY_LOSS": env.get("HWA_SELECTIVE_PROXY_LOSS", "0"),
        "HWA_WIND_FEATURE_INDEX": env.get("HWA_WIND_FEATURE_INDEX", "0"),
        "HWA_WIND_THRESHOLD": env.get("HWA_WIND_THRESHOLD", "10.0"),
        "HWA_WIND_RAMP_END": env.get("HWA_WIND_RAMP_END", "13.9"),
        "HWA_HIGH_WIND_WEIGHT": env.get("HWA_HIGH_WIND_WEIGHT", "4.0"),
        "HWA_PRETRAIN_WINDOWED": env.get("HWA_PRETRAIN_WINDOWED", "0"),
        "ENABLE_TARGET_AWARE_META_NOFT": env.get("ENABLE_TARGET_AWARE_META_NOFT", "0"),
        "SKIP_TARGET_AWARE_PRETRAIN": env.get("SKIP_TARGET_AWARE_PRETRAIN", "0"),
        "SKIP_TARGET_AWARE_META": env.get("SKIP_TARGET_AWARE_META", "0"),
        "ENABLE_TARGET_AWARE_SELECTIVE_FED_META": env.get("ENABLE_TARGET_AWARE_SELECTIVE_FED_META", "0"),
        "SKIP_TARGET_AWARE_SELECTIVE_FED_META": env.get("SKIP_TARGET_AWARE_SELECTIVE_FED_META", "0"),
        "TARGET_AWARE_PRETRAIN_HWA_LOSS": env.get("TARGET_AWARE_PRETRAIN_HWA_LOSS", "1"),
        "TARGET_AWARE_META_CDRM_WEIGHT": env.get("TARGET_AWARE_META_CDRM_WEIGHT", "0.0"),
        "TARGET_AWARE_META_SIM_FLOOR": env.get("TARGET_AWARE_META_SIM_FLOOR", "0.05"),
        "TARGET_AWARE_META_TASK_WEIGHT_ETA": env.get("TARGET_AWARE_META_TASK_WEIGHT_ETA", "1.0"),
        "TARGET_AWARE_META_WIND_MEAN_THRESHOLD": env.get("TARGET_AWARE_META_WIND_MEAN_THRESHOLD", "13.9"),
        "TARGET_AWARE_META_WIND_MAX_THRESHOLD": env.get("TARGET_AWARE_META_WIND_MAX_THRESHOLD", "17.2"),
        "TARGET_AWARE_META_MIN_EXTREME_POINTS": env.get("TARGET_AWARE_META_MIN_EXTREME_POINTS", "3"),
        "TARGET_AWARE_SELECTIVE_FED_TOP_K": env.get("TARGET_AWARE_SELECTIVE_FED_TOP_K", "1"),
        "TARGET_AWARE_SELECTIVE_FED_SELF_FLOOR": env.get("TARGET_AWARE_SELECTIVE_FED_SELF_FLOOR", "0.7"),
        "TARGET_AWARE_SELECTIVE_FED_SOURCE_ALPHA_CAP": env.get("TARGET_AWARE_SELECTIVE_FED_SOURCE_ALPHA_CAP", "0.3"),
        "TARGET_AWARE_SELECTIVE_FED_ALPHA_GRID": env.get("TARGET_AWARE_SELECTIVE_FED_ALPHA_GRID", ""),
        "TARGET_AWARE_SELECTIVE_FED_GAIN_MARGIN": env.get("TARGET_AWARE_SELECTIVE_FED_GAIN_MARGIN", "0.0"),
        "TARGET_AWARE_SELECTIVE_FED_GAIN_GAMMA": env.get("TARGET_AWARE_SELECTIVE_FED_GAIN_GAMMA", "1.0"),
        "TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_MAX_WINDOWS": env.get("TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_MAX_WINDOWS", "8"),
        "TARGET_AWARE_SELECTIVE_FED_PROXY_EXTREME_WEIGHT": env.get("TARGET_AWARE_SELECTIVE_FED_PROXY_EXTREME_WEIGHT", "2.0"),
        "TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_WEIGHT": env.get("TARGET_AWARE_SELECTIVE_FED_PROXY_NORMAL_WEIGHT", "1.0"),
        "GLOBAL_SEED": env.get("GLOBAL_SEED", "1029"),
        "META_SUPPORT_SHOTS": META_SUPPORT_SHOTS,
        "META_QUERY_SHOTS": META_QUERY_SHOTS,
        "EVAL_MODEL_SET": env.get("EVAL_MODEL_SET", ""),
        "EXTREME_SUPPORT_WINDOW_CAP": env.get("EXTREME_SUPPORT_WINDOW_CAP", "6"),
        "PROPOSED_META_SAMPLER_MODE": "balanced",
    })
    if two_year_2024_mode:
        env.update({
            "META_SUPPORT_SHOTS": "3",
            "META_QUERY_SHOTS": "3",
        })
    if two_year_2024_all_extreme:
        env.update({
            "EXTREME_SUPPORT_WINDOW_CAP": "0",
        })
    if high_temp_only_summer:
        env.update({
            "EXTREME_SUPPORT_WINDOW_CAP": "0",
        })
    if high_wind_spring_noft:
        env.update({
            "EXTREME_SUPPORT_WINDOW_CAP": "0",
            "EVAL_MODEL_SET": env.get("EVAL_MODEL_SET") or "fed-meta-noft",
            "SKIP_EXTREME_ADAPTATION_STAGE": "1",
            "HIGH_WIND_NORMAL_TRAIN_START": env.get("HIGH_WIND_NORMAL_TRAIN_START", "2022-04-01T00:00:00+00:00"),
            "HIGH_WIND_NORMAL_TRAIN_END": env.get("HIGH_WIND_NORMAL_TRAIN_END", "2022-04-29T16:00:00+00:00"),
        })
    if four_hour:
        env.update({
            "SAMPLE_INTERVAL_HOURS": "4",
            "LEN_REALP": "3",
            "POINTS_PER_DAY": "6",
            "PHASE_AUGMENT_COMPLEMENTARY_OFFSET": "3",
        })
    if hybrid_extreme_2h:
        env.update({
            "SAMPLE_INTERVAL_HOURS": "2",
            "LEN_REALP": "6",
            "POINTS_PER_DAY": "12",
            "PHASE_AUGMENT_COMPLEMENTARY_OFFSET": "0",
            "SKIP_LOCAL_PRETRAIN": "1",
            "SKIP_LOCAL_META": "1",
            "SKIP_FED_NORMAL_META": "1",
        })
    if high_temp_only_summer and not smoke:
        if meta_shot_regime == META_SHOT_REGIME_3X3:
            env.update({
                "META_TASKS_PER_EPOCH": "3",
                "META_SUPPORT_SHOTS": "3",
                "META_QUERY_SHOTS": "3",
            })
        else:
            env.update({
                "META_TASKS_PER_EPOCH": "2",
                "META_SUPPORT_SHOTS": "4",
                "META_QUERY_SHOTS": "4",
            })
    if selected_preset == SMOKE_PRESET_NAME:
        env.update({
            "PRETRAIN_EPOCHS": "1",
            "PROPOSED_META_EPOCHS": "1",
            "META_ONLY_META_EPOCHS": "1",
            "FEW_SHOT_EPOCHS": "1",
            "EXTREME_TARGET_REFINEMENT_EPOCHS": "1",
            "META_TASKS_PER_EPOCH": "1",
            "META_SUPPORT_SHOTS": "1",
            "META_QUERY_SHOTS": "1",
            "PRETRAIN_LOG_INTERVAL": "1",
            "META_LOG_INTERVAL": "1",
            "FEW_SHOT_LOG_INTERVAL": "1",
        })
    elif selected_preset == PILOT_1K_PRESET_NAME:
        env.update({
            "PRETRAIN_EPOCHS": "1000",
            "PROPOSED_META_EPOCHS": "1000",
            "META_ONLY_META_EPOCHS": "1000",
            "FEW_SHOT_EPOCHS": "50",
            "EXTREME_TARGET_REFINEMENT_EPOCHS": "50",
            "PRETRAIN_LOG_INTERVAL": "50",
            "META_LOG_INTERVAL": "50",
            "FEW_SHOT_LOG_INTERVAL": "10",
        })
    elif selected_preset == PILOT_5K_PRESET_NAME:
        env.update({
            "PRETRAIN_EPOCHS": "5000",
            "PROPOSED_META_EPOCHS": "5000",
            "META_ONLY_META_EPOCHS": "5000",
            "FEW_SHOT_EPOCHS": "200",
            "EXTREME_TARGET_REFINEMENT_EPOCHS": "200",
            "PRETRAIN_LOG_INTERVAL": "100",
            "META_LOG_INTERVAL": "100",
            "FEW_SHOT_LOG_INTERVAL": "20",
        })
    elif selected_preset == FINAL_CANDIDATE_PRESET_NAME:
        env.update({
            "PRETRAIN_EPOCHS": "10000",
            "PROPOSED_META_EPOCHS": "10000",
            "META_ONLY_META_EPOCHS": "10000",
            "FEW_SHOT_EPOCHS": "500",
            "EXTREME_TARGET_REFINEMENT_EPOCHS": "500",
            "PRETRAIN_LOG_INTERVAL": "200",
            "META_LOG_INTERVAL": "200",
            "FEW_SHOT_LOG_INTERVAL": "50",
        })
    elif selected_preset == REFINE_K1_PRESET_NAME:
        env.update({
            "SKIP_LOCAL_PRETRAIN": "1",
            "SKIP_LOCAL_META": "1",
            "PRETRAIN_EPOCHS": "5000",
            "PROPOSED_META_EPOCHS": "5000",
            "META_ONLY_META_EPOCHS": "5000",
            "FEW_SHOT_EPOCHS": "200",
            "EXTREME_TARGET_REFINEMENT_EPOCHS": "200",
            "EXTREME_TARGET_ADAPT_MAX_WINDOWS": "1",
            "PRETRAIN_LOG_INTERVAL": "100",
            "META_LOG_INTERVAL": "100",
            "FEW_SHOT_LOG_INTERVAL": "20",
        })
    elif selected_preset == REFINE_K2_PRESET_NAME:
        env.update({
            "SKIP_LOCAL_PRETRAIN": "1",
            "SKIP_LOCAL_META": "1",
            "PRETRAIN_EPOCHS": "5000",
            "PROPOSED_META_EPOCHS": "5000",
            "META_ONLY_META_EPOCHS": "5000",
            "FEW_SHOT_EPOCHS": "200",
            "EXTREME_TARGET_REFINEMENT_EPOCHS": "200",
            "EXTREME_TARGET_ADAPT_MAX_WINDOWS": "2",
            "PRETRAIN_LOG_INTERVAL": "100",
            "META_LOG_INTERVAL": "100",
            "FEW_SHOT_LOG_INTERVAL": "20",
        })
    elif selected_preset == PILOT_PRESET_NAME:
        env.update({
            "PRETRAIN_EPOCHS": "500",
            "PROPOSED_META_EPOCHS": "500",
            "META_ONLY_META_EPOCHS": "500",
            "FEW_SHOT_EPOCHS": "10",
            "EXTREME_TARGET_REFINEMENT_EPOCHS": "10",
            "PRETRAIN_LOG_INTERVAL": "50",
            "META_LOG_INTERVAL": "50",
            "FEW_SHOT_LOG_INTERVAL": "1",
        })
    elif selected_preset == PILOT_MEDIUM_PRESET_NAME:
        env.update({
            "PRETRAIN_EPOCHS": "2000",
            "PROPOSED_META_EPOCHS": "2000",
            "META_ONLY_META_EPOCHS": "2000",
            "FEW_SHOT_EPOCHS": "20",
            "EXTREME_TARGET_REFINEMENT_EPOCHS": "20",
            "PRETRAIN_LOG_INTERVAL": "100",
            "META_LOG_INTERVAL": "100",
            "FEW_SHOT_LOG_INTERVAL": "1",
        })
    elif selected_preset == FORMAL_PRESET_NAME:
        env.update({
            "PRETRAIN_EPOCHS": "35000",
            "PROPOSED_META_EPOCHS": "30000",
            "META_ONLY_META_EPOCHS": "30000",
            "FEW_SHOT_EPOCHS": "50",
            "EXTREME_TARGET_REFINEMENT_EPOCHS": "50",
            "PRETRAIN_LOG_INTERVAL": "100",
            "META_LOG_INTERVAL": "100",
            "FEW_SHOT_LOG_INTERVAL": "1",
        })
    if high_wind_spring_noft:
        if selected_preset != SMOKE_PRESET_NAME:
            env.update({
                "META_TASKS_PER_EPOCH": "2",
                "META_SUPPORT_SHOTS": "4",
                "META_QUERY_SHOTS": "4",
            })
        env.update({
            "FEW_SHOT_EPOCHS": "0",
            "EXTREME_TARGET_REFINEMENT_EPOCHS": "0",
        })
    if high_temp_only_summer and not smoke and meta_shot_regime == META_SHOT_REGIME_3X3:
        for epoch_key in ["PROPOSED_META_EPOCHS", "META_ONLY_META_EPOCHS"]:
            env[epoch_key] = str(max(1, (int(env[epoch_key]) * 2 + 2) // 3))
    return env


def build_stage_commands(stages):
    commands = {
        "build": [sys.executable, "-u", "build_three_station_extreme_yearly_protocol.py"],
        "train": [sys.executable, "-u", "DemoModelTraining.py"],
        "eval": [sys.executable, "-u", "generate_multi_station_results.py"],
    }
    stage_commands = []
    for stage_name in stages:
        stage_commands.append((stage_name, commands[stage_name]))
    return stage_commands


def render_env_preview(env):
    return " ".join(f"{key}={env[key]}" for key in PREVIEW_ENV_KEYS if key in env)


def render_command(command):
    return " ".join(command)


def print_stage_banner(stage_name, env, command):
    print("\n" + "=" * 70, flush=True)
    print(f"stage={stage_name}", flush=True)
    print(f"env  {render_env_preview(env)}", flush=True)
    print(f"exec {render_command(command)}", flush=True)
    print("=" * 70, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run three-station yearly extreme protocol stages")
    parser.add_argument("stage", choices=['build', 'train', 'eval', 'all'])
    parser.add_argument(
        "--preset",
        choices=[
            SMOKE_PRESET_NAME,
            PILOT_PRESET_NAME,
            PILOT_MEDIUM_PRESET_NAME,
            FORMAL_PRESET_NAME,
            PILOT_1K_PRESET_NAME,
            PILOT_5K_PRESET_NAME,
            FINAL_CANDIDATE_PRESET_NAME,
            REFINE_K1_PRESET_NAME,
            REFINE_K2_PRESET_NAME,
        ],
        default=PILOT_1K_PRESET_NAME,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--six-station", action="store_true")
    parser.add_argument("--two-year-2024-k6", action="store_true")
    parser.add_argument("--two-year-2024-all-extreme", action="store_true")
    parser.add_argument("--high-temp-only-summer", action="store_true")
    parser.add_argument("--high-wind-spring-noft", action="store_true")
    parser.add_argument("--enable-selective-fed-normal-meta", action="store_true")
    parser.add_argument(
        "--meta-shot-regime",
        choices=[META_SHOT_REGIME_2X4, META_SHOT_REGIME_3X3],
        default=META_SHOT_REGIME_2X4,
    )
    parser.add_argument("--four-hour", action="store_true")
    parser.add_argument("--hybrid-extreme-2h", action="store_true")
    args = parser.parse_args()

    stages = [args.stage] if args.stage != "all" else ["build", "train", "eval"]
    env = build_env(
        preset=args.preset,
        smoke=args.smoke,
        six_station=(
            args.six_station
            or args.four_hour
            or args.hybrid_extreme_2h
            or args.two_year_2024_k6
            or args.two_year_2024_all_extreme
            or args.high_temp_only_summer
            or args.high_wind_spring_noft
        ),
        four_hour=args.four_hour,
        hybrid_extreme_2h=args.hybrid_extreme_2h,
        two_year_2024_k6=args.two_year_2024_k6,
        two_year_2024_all_extreme=args.two_year_2024_all_extreme,
        high_temp_only_summer=args.high_temp_only_summer,
        high_wind_spring_noft=args.high_wind_spring_noft,
        meta_shot_regime=args.meta_shot_regime,
        enable_selective_fed_normal_meta=args.enable_selective_fed_normal_meta,
    )

    for stage_name, command in build_stage_commands(stages):
        print_stage_banner(stage_name, env, command)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=str(ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
