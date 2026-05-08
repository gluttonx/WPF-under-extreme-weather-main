#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


@dataclass(frozen=True)
class CurveSpec:
    stage: str
    label: str
    tag_prefixes: tuple[str, ...]
    color: str


CURVES = (
    CurveSpec(
        "Baseline Pretrain",
        "Local pretrain MSE",
        ("loss_mse_pre_station",),
        "#305f72",
    ),
    CurveSpec(
        "Baseline Meta",
        "Local meta query MSE",
        ("loss_mse_train_task_query_local_meta_station",),
        "#b23a48",
    ),
    CurveSpec(
        "Target-Aware Pretrain",
        "Target-aware pretrain MSE",
        ("loss_mse_target_aware_pre_station",),
        "#2f7f4f",
    ),
    CurveSpec(
        "Target-Aware Meta",
        "Target-aware meta query MSE",
        ("loss_mse_train_task_query_target_aware_meta_station",),
        "#6b4e9b",
    ),
    CurveSpec(
        "Selective Fed Meta",
        "Aggregate proxy MSE",
        ("loss_mse_target_aware_selective_fed_meta_proxy_station",),
        "#d97706",
    ),
    CurveSpec(
        "Selective Fed Meta",
        "Self proxy MSE",
        ("loss_mse_target_aware_selective_fed_meta_self_proxy_station",),
        "#64748b",
    ),
    CurveSpec(
        "Selective Fed Meta",
        "Legacy candidate query MSE",
        ("loss_mse_train_task_query_target_aware_selective_fed_meta_station",),
        "#9a3412",
    ),
)

PRETRAIN_STAGES = frozenset({"Baseline Pretrain", "Target-Aware Pretrain"})


def load_scalars(log_dir: Path) -> dict[str, list]:
    accumulator = event_accumulator.EventAccumulator(
        str(log_dir),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    return {tag: accumulator.Scalars(tag) for tag in accumulator.Tags().get("scalars", [])}


def collect_run_frame(artifact_dir: Path, run_label: str, pretrain_only: bool = False) -> pd.DataFrame:
    log_dir = artifact_dir / "logs_train" / "loss2"
    scalars = load_scalars(log_dir)
    rows: list[dict] = []
    for curve in CURVES:
        if pretrain_only and curve.stage not in PRETRAIN_STAGES:
            continue
        for prefix in curve.tag_prefixes:
            matching_tags = sorted(tag for tag in scalars if tag.startswith(prefix))
            for tag in matching_tags:
                for event in scalars[tag]:
                    rows.append(
                        {
                            "run_label": run_label,
                            "artifact_dir": str(artifact_dir),
                            "stage": curve.stage,
                            "curve": curve.label,
                            "tag_prefix": prefix,
                            "series_tag": tag,
                            "epoch": int(event.step) + 1,
                            "loss_mse": float(event.value),
                        }
                    )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.sort_values(["run_label", "stage", "curve", "epoch", "source_priority"])
    frame = frame.drop_duplicates(
        subset=["run_label", "stage", "curve", "series_tag", "epoch"],
        keep="last",
    )
    return (
        frame.groupby(["run_label", "artifact_dir", "stage", "curve", "epoch"], as_index=False)
        .agg(
            loss_mean=("loss_mse", "mean"),
            loss_std=("loss_mse", "std"),
            series_count=("series_tag", "nunique"),
        )
        .fillna({"loss_std": 0.0})
    )


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    stage_order = list(dict.fromkeys(curve.stage for curve in CURVES))
    run_order = list(dict.fromkeys(summary["run_label"].tolist()))
    fig, axes = plt.subplots(
        nrows=len(stage_order),
        ncols=len(run_order),
        figsize=(6.2 * max(1, len(run_order)), 3.2 * len(stage_order)),
        squeeze=False,
        constrained_layout=True,
    )
    color_by_curve = {curve.label: curve.color for curve in CURVES}

    for row_idx, stage in enumerate(stage_order):
        for col_idx, run_label in enumerate(run_order):
            ax = axes[row_idx][col_idx]
            panel = summary[
                (summary["stage"] == stage)
                & (summary["run_label"] == run_label)
            ]
            if panel.empty:
                ax.set_axis_off()
                ax.set_title(f"{run_label} | {stage}\nno scalar data")
                continue
            for curve_name, curve_frame in panel.groupby("curve"):
                curve_frame = curve_frame.sort_values("epoch")
                color = color_by_curve.get(curve_name, "#111827")
                ax.plot(
                    curve_frame["epoch"],
                    curve_frame["loss_mean"],
                    label=curve_name,
                    color=color,
                    linewidth=1.6,
                )
                if int(curve_frame["series_count"].max()) > 1:
                    ax.fill_between(
                        curve_frame["epoch"],
                        curve_frame["loss_mean"] - curve_frame["loss_std"],
                        curve_frame["loss_mean"] + curve_frame["loss_std"],
                        color=color,
                        alpha=0.14,
                        linewidth=0,
                    )
            ax.set_title(f"{run_label} | {stage}", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE loss")
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.legend(frameon=False, fontsize=8)

    fig.suptitle("High-Wind Spring NoFT Training Loss by Epoch", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot high-wind spring NoFT pretrain/meta/selective-fed MSE loss by true epoch."
    )
    parser.add_argument("--artifact-dir", action="append", type=Path, required=True)
    parser.add_argument(
        "--pretrain-history-artifact-dir",
        action="append",
        type=Path,
        default=[],
        help="Previous artifacts used only to prepend pretrain curves, not meta/selective-fed curves.",
    )
    parser.add_argument("--label", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    artifact_dirs = [path.resolve() for path in args.artifact_dir]
    labels = args.label or [path.name for path in artifact_dirs]
    if len(labels) != len(artifact_dirs):
        raise ValueError("--label count must match --artifact-dir count")

    output_dir = args.output_dir or (artifact_dirs[-1] / "loss_curve_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for history_dir in args.pretrain_history_artifact_dir:
        history_dir = history_dir.resolve()
        for label in labels:
            history_frame = collect_run_frame(history_dir, label, pretrain_only=True)
            if not history_frame.empty:
                history_frame["source_priority"] = 0
                frames.append(history_frame)
    for path, label in zip(artifact_dirs, labels):
        run_frame = collect_run_frame(path, label)
        if not run_frame.empty:
            run_frame["source_priority"] = 1
            frames.append(run_frame)
    long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary_df = summarize(long_df)

    long_path = output_dir / "high_wind_loss_epoch_long.csv"
    summary_path = output_dir / "high_wind_loss_epoch_summary.csv"
    figure_path = output_dir / "high_wind_loss_epoch_curves.png"
    long_df.to_csv(long_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    if not summary_df.empty:
        plot_summary(summary_df, figure_path)

    print(f"Saved long CSV: {long_path}")
    print(f"Saved summary CSV: {summary_path}")
    if not summary_df.empty:
        print(f"Saved figure: {figure_path}")
    else:
        print("No scalar data found; figure was not generated.")


if __name__ == "__main__":
    main()
