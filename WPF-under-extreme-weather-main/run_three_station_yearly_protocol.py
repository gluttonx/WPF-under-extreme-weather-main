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
THREE_STATION_PROTOCOL_DATA_DIR = str(ROOT / "protocol_data" / "2h_6p")
SIX_STATION_PROTOCOL_DATA_DIR = str(ROOT / "protocol_data" / "2h_6p_six_station")
SIX_STATION_ARTIFACT_DIR = str(ROOT / "artifacts" / "2h6p_six_station")
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
PREVIEW_ENV_KEYS = [
    "PROTOCOL_NAME",
    "PROTOCOL_DATA_DIR",
    "PROTOCOL_METADATA_PATH",
    "SAMPLE_INTERVAL_HOURS",
    "DOWNSAMPLE_OFFSET",
    "LEN_REALP",
    "POINTS_PER_DAY",
    "PHASE_AUGMENT_STATIONS",
    "ARTIFACT_DIR",
    "MODEL_OUTPUT_DIR",
    "LOGS_TRAIN_DIR",
    "CONVERGENCE_REPORT_PATH",
    "ALL_STATIONS_TEST_RESULTS_PATH",
    "RESULTS_OUTPUT_PATH",
    "TASK_RESULTS_OUTPUT_PATH",
    "YEARLY_PROTOCOL_ENABLED",
    "SEASONAL_PROTOCOL_ENABLED",
    "YEARLY_PROTOCOL_METADATA_PATH",
    "USE_FEDERATION",
    "RUN_FEDERATED_PRETRAIN",
    "TRAIN_META_ONLY_BASELINE",
    "META_SUPPORT_SHOTS",
    "META_QUERY_SHOTS",
    "PROPOSED_META_SAMPLER_MODE",
    "PRETRAIN_EPOCHS",
    "PROPOSED_META_EPOCHS",
    "META_ONLY_META_EPOCHS",
    "FEW_SHOT_EPOCHS",
    "EXTREME_TARGET_REFINEMENT_EPOCHS",
    "EXTREME_TARGET_ADAPT_MAX_WINDOWS",
    "PRETRAIN_LOG_INTERVAL",
    "META_LOG_INTERVAL",
    "FEW_SHOT_LOG_INTERVAL",
]


def build_env(preset=None, smoke=False, six_station=False):
    env = os.environ.copy()
    selected_preset = SMOKE_PRESET_NAME if smoke else preset
    protocol_name = SIX_STATION_PROTOCOL_NAME if six_station else PROTOCOL_NAME
    protocol_data_dir = SIX_STATION_PROTOCOL_DATA_DIR if six_station else PROTOCOL_DATA_DIR
    protocol_metadata_path = str(Path(protocol_data_dir) / "protocol_metadata.json")
    if six_station:
        artifact_dir = env.get("ARTIFACT_DIR", str(Path(SIX_STATION_ARTIFACT_DIR) / str(selected_preset or "default")))
    else:
        artifact_dir = env.get("ARTIFACT_DIR", ".")
    model_output_dir = env.get("MODEL_OUTPUT_DIR", str(Path(artifact_dir) / "models") if artifact_dir not in ("", ".") else ".")
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
        "PHASE_AUGMENT_STATIONS": "1" if six_station else PHASE_AUGMENT_STATIONS,
        "ARTIFACT_DIR": artifact_dir,
        "MODEL_OUTPUT_DIR": model_output_dir,
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
        "USE_FEDERATION": USE_FEDERATION,
        "RUN_FEDERATED_PRETRAIN": RUN_FEDERATED_PRETRAIN,
        "TRAIN_META_ONLY_BASELINE": TRAIN_META_ONLY_BASELINE,
        "META_SUPPORT_SHOTS": META_SUPPORT_SHOTS,
        "META_QUERY_SHOTS": META_QUERY_SHOTS,
        "PROPOSED_META_SAMPLER_MODE": "balanced",
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
    args = parser.parse_args()

    stages = [args.stage] if args.stage != "all" else ["build", "train", "eval"]
    env = build_env(preset=args.preset, smoke=args.smoke, six_station=args.six_station)

    for stage_name, command in build_stage_commands(stages):
        print_stage_banner(stage_name, env, command)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=str(ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
