#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    prefixes: tuple[str, ...]
    total_epochs: int


RUN_SPECS = {
    "selective": {
        "label": "Selective Fed-Meta",
        "artifact_dir": "pilot-1k-selective-2x4",
    },
    "vanilla": {
        "label": "Vanilla Fed-Meta",
        "artifact_dir": "pilot-1k-vanilla-fed-meta-2x4",
    },
}

STAGE_LAYOUT = (
    (
        StageSpec("pretrain", "Pretrain", ("loss_mse_pre_station",), 1000),
    ),
    (
        StageSpec(
            "local_meta",
            "Local Meta",
            (
                "loss_mse_train_task_support_local_meta_station",
                "loss_mse_train_task_query_local_meta_station",
            ),
            1000,
        ),
    ),
    (
        StageSpec(
            "fed_meta",
            "Fed-Normal-Meta",
            (
                "loss_mse_train_task_support_fed_normal_meta_station",
                "loss_mse_train_task_query_fed_normal_meta_station",
            ),
            1000,
        ),
    ),
    (
        StageSpec(
            "few_shot",
            "Few-Shot Fine-Tune",
            (
                "loss_mse_lmt_station",
                "loss_mse_fed_meta_local_ft_station",
            ),
            50,
        ),
    ),
)

CURVE_STYLE = {
    "loss_mse_pre_station": ("Avg Pretrain MSE", "#1f77b4"),
    "loss_mse_train_task_support_local_meta_station": ("Support MSE", "#1f77b4"),
    "loss_mse_train_task_query_local_meta_station": ("Query MSE", "#d62728"),
    "loss_mse_train_task_support_fed_normal_meta_station": ("Support MSE", "#1f77b4"),
    "loss_mse_train_task_query_fed_normal_meta_station": ("Query MSE", "#d62728"),
    "loss_mse_lmt_station": ("LMT FT MSE", "#2ca02c"),
    "loss_mse_fed_meta_local_ft_station": ("Fed-Meta+Local-FT MSE", "#9467bd"),
}


def load_scalars(log_dir: Path) -> dict[str, list]:
    accumulator = event_accumulator.EventAccumulator(
        str(log_dir),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    return {tag: accumulator.Scalars(tag) for tag in accumulator.Tags().get("scalars", [])}


def build_long_dataframe(run_name: str, run_label: str, scalars: dict[str, list], stage_spec: StageSpec) -> pd.DataFrame:
    rows: list[dict] = []
    for prefix in stage_spec.prefixes:
        matching_tags = sorted(tag for tag in scalars if tag.startswith(prefix))
        curve_label, _ = CURVE_STYLE[prefix]
        for tag in matching_tags:
            for event in scalars[tag]:
                rows.append(
                    {
                        "run": run_name,
                        "run_label": run_label,
                        "stage": stage_spec.key,
                        "stage_label": stage_spec.label,
                        "curve_prefix": prefix,
                        "curve_label": curve_label,
                        "series_tag": tag,
                        "epoch": int(event.step) + 1,
                        "loss_mse": float(event.value),
                        "total_epochs": stage_spec.total_epochs,
                    }
                )
    return pd.DataFrame(rows)


def summarize_curve(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = (
        df.groupby(["run", "run_label", "stage", "stage_label", "curve_prefix", "curve_label", "epoch", "total_epochs"], as_index=False)
        .agg(loss_mean=("loss_mse", "mean"), loss_std=("loss_mse", "std"), series_count=("series_tag", "nunique"))
    )
    grouped["loss_std"] = grouped["loss_std"].fillna(0.0)
    coverage = (
        grouped.groupby(["run", "stage", "curve_prefix"], as_index=False)
        .agg(observed_epochs=("epoch", "nunique"), total_epochs=("total_epochs", "max"))
    )
    coverage["dense_logging"] = coverage["observed_epochs"] >= coverage["total_epochs"]
    return grouped.merge(coverage, on=["run", "stage", "curve_prefix", "total_epochs"], how="left")


def stage_note(stage_frame: pd.DataFrame) -> str:
    observed = int(stage_frame["observed_epochs"].min())
    total = int(stage_frame["total_epochs"].max())
    if observed >= total:
        return "full epoch logging"
    return f"sparse logging: {observed}/{total} epochs"


def plot_stage_panel(ax, stage_frame: pd.DataFrame, run_label: str, stage_label: str) -> None:
    if stage_frame.empty:
        ax.set_title(f"{run_label} | {stage_label}\nno data")
        ax.axis("off")
        return

    prefixes = list(dict.fromkeys(stage_frame["curve_prefix"].tolist()))
    note = stage_note(stage_frame)
    for prefix in prefixes:
        curve_frame = stage_frame[stage_frame["curve_prefix"] == prefix].sort_values("epoch")
        curve_label, color = CURVE_STYLE[prefix]
        sparse = not bool(curve_frame["dense_logging"].iloc[0])
        marker = "o" if sparse else None
        ax.plot(
            curve_frame["epoch"],
            curve_frame["loss_mean"],
            label=curve_label,
            color=color,
            linewidth=1.8,
            marker=marker,
            markersize=3 if marker else None,
        )
        if int(curve_frame["series_count"].max()) > 1:
            ax.fill_between(
                curve_frame["epoch"],
                curve_frame["loss_mean"] - curve_frame["loss_std"],
                curve_frame["loss_mean"] + curve_frame["loss_std"],
                color=color,
                alpha=0.12,
                linewidth=0,
            )

    ax.set_title(f"{run_label} | {stage_label}\n{note}", fontsize=10)
    ax.set_ylabel("MSE loss")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)


def make_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=len(STAGE_LAYOUT),
        ncols=len(RUN_SPECS),
        figsize=(14, 14),
        constrained_layout=True,
    )

    run_order = ["selective", "vanilla"]
    for row_index, stage_group in enumerate(STAGE_LAYOUT):
        stage_spec = stage_group[0]
        for col_index, run_name in enumerate(run_order):
            ax = axes[row_index, col_index]
            run_label = RUN_SPECS[run_name]["label"]
            panel_df = summary_df[
                (summary_df["run"] == run_name)
                & (summary_df["stage"] == stage_spec.key)
            ]
            plot_stage_panel(ax, panel_df, run_label, stage_spec.label)
            if row_index == len(STAGE_LAYOUT) - 1:
                ax.set_xlabel("Epoch")

    fig.suptitle(
        "High-Temperature Summer Protocol Loss Curves by Epoch\nSelective vs Vanilla Fed-Meta pilot-1k-2x4",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare loss curves between selective and vanilla high-temp pilots.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir or (repo_root / "artifacts" / "high_temp_only_summer_six_station" / "loss_curve_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    loss2_frames: list[pd.DataFrame] = []
    for run_name, run_info in RUN_SPECS.items():
        log_dir = repo_root / "artifacts" / "high_temp_only_summer_six_station" / run_info["artifact_dir"] / "logs_train" / "loss2"
        scalars = load_scalars(log_dir)
        for stage_group in STAGE_LAYOUT:
            stage_spec = stage_group[0]
            loss2_frames.append(build_long_dataframe(run_name, run_info["label"], scalars, stage_spec))

    long_df = pd.concat(loss2_frames, ignore_index=True)
    summary_df = summarize_curve(long_df)

    long_csv = output_dir / "loss_curve_epoch_long.csv"
    summary_csv = output_dir / "loss_curve_epoch_summary.csv"
    figure_path = output_dir / "loss_curve_epoch_comparison_selective_vs_vanilla.png"

    long_df.to_csv(long_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    make_figure(summary_df, figure_path)

    coverage_lines = []
    coverage = (
        summary_df.groupby(["run_label", "stage_label", "curve_label", "observed_epochs", "total_epochs"], as_index=False)
        .size()
        .drop(columns=["size"])
        .drop_duplicates()
        .sort_values(["run_label", "stage_label", "curve_label"])
    )
    for _, row in coverage.iterrows():
        coverage_lines.append(
            f"{row['run_label']} | {row['stage_label']} | {row['curve_label']}: "
            f"{int(row['observed_epochs'])}/{int(row['total_epochs'])} logged epochs"
        )

    print(f"Saved figure: {figure_path}")
    print(f"Saved long CSV: {long_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print("Coverage:")
    for line in coverage_lines:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
