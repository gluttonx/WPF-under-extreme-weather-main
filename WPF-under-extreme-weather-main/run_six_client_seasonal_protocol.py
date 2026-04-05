#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Launcher for the six-client seasonal scarcity protocol."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEASONAL_PROTOCOL_ENABLED = "1"
SEASONAL_PROTOCOL_METADATA_PATH = str(ROOT / "seasonal_protocol_data" / "seasonal_protocol_metadata.json")
META_SUPPORT_SHOTS = "5"
META_QUERY_SHOTS = "5"
SMOKE_PRESET_NAME = "smoke"
FORMAL_PRESET_NAME = "formal-v1"
PREVIEW_ENV_KEYS = [
    "SEASONAL_PROTOCOL_ENABLED",
    "SEASONAL_PROTOCOL_METADATA_PATH",
    "META_SUPPORT_SHOTS",
    "META_QUERY_SHOTS",
    "PROPOSED_META_SAMPLER_MODE",
    "PRETRAIN_EPOCHS",
    "PROPOSED_META_EPOCHS",
    "META_ONLY_META_EPOCHS",
    "FEW_SHOT_EPOCHS",
    "CONVENTIONAL_RATIO",
    "REGIME_MISSING_MODE",
    "STRICT_PAPER_ORDER",
]


def build_env(preset=None, smoke=False):
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "SEASONAL_PROTOCOL_ENABLED": SEASONAL_PROTOCOL_ENABLED,
        "SEASONAL_PROTOCOL_METADATA_PATH": SEASONAL_PROTOCOL_METADATA_PATH,
        "META_SUPPORT_SHOTS": META_SUPPORT_SHOTS,
        "META_QUERY_SHOTS": META_QUERY_SHOTS,
    })
    selected_preset = SMOKE_PRESET_NAME if smoke else preset
    if selected_preset == SMOKE_PRESET_NAME:
        env.update({
            "PRETRAIN_EPOCHS": "1",
            "PROPOSED_META_EPOCHS": "1",
            "META_ONLY_META_EPOCHS": "1",
            "FEW_SHOT_EPOCHS": "1",
            "STRICT_PAPER_ORDER": "0",
        })
    elif selected_preset == FORMAL_PRESET_NAME:
        env.update({
            "PROPOSED_META_SAMPLER_MODE": "balanced",
            "PRETRAIN_EPOCHS": "4000",
            "PROPOSED_META_EPOCHS": "3000",
            "META_ONLY_META_EPOCHS": "3000",
            "FEW_SHOT_EPOCHS": "20",
            "CONVENTIONAL_RATIO": "1.0",
            "REGIME_MISSING_MODE": "none",
        })
    return env


def build_stage_commands(stages):
    commands = {
        "build": [sys.executable, "-u", "build_six_client_seasonal_protocol.py"],
        "train": [sys.executable, "-u", "DemoModelTraining.py"],
        "eval": [sys.executable, "-u", "generate_multi_station_results.py"],
    }
    stage_commands = []
    for stage_name in stages:
        stage_commands.append((stage_name, commands[stage_name]))
    return stage_commands


def render_env_preview(env):
    return " ".join(f"{key}={env[key]}" for key in PREVIEW_ENV_KEYS if key in env)


def main():
    parser = argparse.ArgumentParser(description="Run six-client seasonal protocol stages")
    parser.add_argument("stage", choices=['build', 'train', 'eval', 'all'])
    parser.add_argument("--preset", choices=[SMOKE_PRESET_NAME, FORMAL_PRESET_NAME], default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stages = [args.stage] if args.stage != "all" else ["build", "train", "eval"]
    env = build_env(preset=args.preset, smoke=args.smoke)

    for stage_name, command in build_stage_commands(stages):
        rendered = " ".join(command)
        if args.dry_run:
            print(f"[{stage_name}] env {render_env_preview(env)}", flush=True)
            print(f"[{stage_name}] {rendered}", flush=True)
            continue
        print(f"[{stage_name}] env {render_env_preview(env)}", flush=True)
        print(f"[{stage_name}] {rendered}", flush=True)
        subprocess.run(command, cwd=str(ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
