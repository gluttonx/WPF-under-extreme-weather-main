#!/usr/bin/env python
"""Plot fine-tuning sweep curves from DemoModelTraining FT sweep CSV files."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="append", required=True, help="FT sweep CSV path. Repeat for comparison.")
    parser.add_argument("--label", action="append", default=None, help="Label for the matching --csv path.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    return parser.parse_args()


def load_frames(csv_paths, labels):
    frames = []
    if labels is None:
        labels = []
    for index, csv_path in enumerate(csv_paths):
        path = Path(csv_path)
        label = labels[index] if index < len(labels) else path.stem
        frame = pd.read_csv(path)
        frame["Run_Label"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def weighted_average(group, value_col):
    values = pd.to_numeric(group[value_col], errors="coerce")
    weights = pd.to_numeric(group["Test_Windows"], errors="coerce").fillna(1.0).clip(lower=1.0)
    valid = values.notna()
    if not valid.any():
        return float("nan")
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def aggregate_sweep(frame):
    rows = []
    for (run_label, model, ft_epoch), group in frame.groupby(["Run_Label", "Model", "FT_Epoch"], as_index=False):
        rows.append(
            {
                "Run_Label": run_label,
                "Model": model,
                "FT_Epoch": int(ft_epoch),
                "SupportAll_MSE": weighted_average(group, "SupportAll_MSE"),
                "Val_MSE": weighted_average(group, "Val_MSE"),
                "Test_MSE": weighted_average(group, "Test_MSE"),
                "Test_nMAE_%": weighted_average(group, "Test_nMAE_%"),
            }
        )
    return pd.DataFrame(rows).sort_values(["Run_Label", "Model", "FT_Epoch"])


def plot_curves(agg, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=160)
    style_columns = [
        ("SupportAll_MSE", "support MSE"),
        ("Val_MSE", "held-out support MSE"),
        ("Test_MSE", "test MSE"),
    ]
    for (run_label, model), group in agg.groupby(["Run_Label", "Model"]):
        curve_label = f"{run_label}: {model}"
        for column, metric_label in style_columns:
            if group[column].notna().any():
                axes[0].plot(group["FT_Epoch"], group[column], marker="o", linewidth=1.6, label=f"{curve_label} {metric_label}")
        if group["Test_nMAE_%"].notna().any():
            axes[1].plot(group["FT_Epoch"], group["Test_nMAE_%"], marker="o", linewidth=1.8, label=curve_label)

    axes[0].set_title("Fine-tune Loss")
    axes[0].set_xlabel("FT epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)

    axes[1].set_title("Test nMAE")
    axes[1].set_xlabel("FT epoch")
    axes[1].set_ylabel("nMAE (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"saved: {output_path}")


def main():
    args = parse_args()
    frame = load_frames(args.csv, args.label)
    agg = aggregate_sweep(frame)
    plot_curves(agg, args.output)


if __name__ == "__main__":
    main()
