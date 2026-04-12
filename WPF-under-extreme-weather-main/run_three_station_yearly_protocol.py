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
YEARLY_PROTOCOL_METADATA_PATH = str(
    ROOT / "three_station_yearly_protocol_data" / "three_station_yearly_protocol_metadata.json"
)
USE_FEDERATION = "1"
RUN_FEDERATED_PRETRAIN = "0"
TRAIN_META_ONLY_BASELINE = "0"
META_SUPPORT_SHOTS = "10"
META_QUERY_SHOTS = "10"
SMOKE_PRESET_NAME = "smoke"
PILOT_PRESET_NAME = "pilot"
PILOT_MEDIUM_PRESET_NAME = "pilot-medium"
FORMAL_PRESET_NAME = "formal-v1"
PREVIEW_ENV_KEYS = [
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
    "PRETRAIN_LOG_INTERVAL",
    "META_LOG_INTERVAL",
    "FEW_SHOT_LOG_INTERVAL",
]


def build_env(preset=None, smoke=False):
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "YEARLY_PROTOCOL_ENABLED": YEARLY_PROTOCOL_ENABLED,
        "SEASONAL_PROTOCOL_ENABLED": SEASONAL_PROTOCOL_ENABLED,
        "YEARLY_PROTOCOL_METADATA_PATH": YEARLY_PROTOCOL_METADATA_PATH,
        "USE_FEDERATION": USE_FEDERATION,
        "RUN_FEDERATED_PRETRAIN": RUN_FEDERATED_PRETRAIN,
        "TRAIN_META_ONLY_BASELINE": TRAIN_META_ONLY_BASELINE,
        "META_SUPPORT_SHOTS": META_SUPPORT_SHOTS,
        "META_QUERY_SHOTS": META_QUERY_SHOTS,
        "PROPOSED_META_SAMPLER_MODE": "balanced",
    })
    selected_preset = SMOKE_PRESET_NAME if smoke else preset
    if selected_preset == SMOKE_PRESET_NAME:
        env.update({
            "PRETRAIN_EPOCHS": "1",
            "PROPOSED_META_EPOCHS": "1",
            "META_ONLY_META_EPOCHS": "1",
            "FEW_SHOT_EPOCHS": "1",
            "META_TASKS_PER_EPOCH": "1",
            "META_SUPPORT_SHOTS": "1",
            "META_QUERY_SHOTS": "1",
            "PRETRAIN_LOG_INTERVAL": "1",
            "META_LOG_INTERVAL": "1",
            "FEW_SHOT_LOG_INTERVAL": "1",
        })
    elif selected_preset == PILOT_PRESET_NAME:
        env.update({
            "PRETRAIN_EPOCHS": "500",
            "PROPOSED_META_EPOCHS": "500",
            "META_ONLY_META_EPOCHS": "500",
            "FEW_SHOT_EPOCHS": "10",
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
        choices=[SMOKE_PRESET_NAME, PILOT_PRESET_NAME, PILOT_MEDIUM_PRESET_NAME, FORMAL_PRESET_NAME],
        default=PILOT_PRESET_NAME,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stages = [args.stage] if args.stage != "all" else ["build", "train", "eval"]
    env = build_env(preset=args.preset, smoke=args.smoke)

    for stage_name, command in build_stage_commands(stages):
        print_stage_banner(stage_name, env, command)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=str(ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
