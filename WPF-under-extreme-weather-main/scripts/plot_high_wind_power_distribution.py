#!/usr/bin/env python
"""Plot normalized power density for 2022-04 normal and high-wind samples."""
import argparse
import json
from pathlib import Path
from datetime import datetime
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from scipy.stats import gaussian_kde

TARGET_COL = "Power2"
RAW_BASE_STATIONS = [
    {
        "station_id": "59",
        "workbook": "2223jilin_059_processed_4classes.xlsx",
        "capacity": 50.0,
    },
    {
        "station_id": "60",
        "workbook": "2223jilin_060_processed_4classes.xlsx",
        "capacity": 100.0,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["raw-base", "protocol"],
        default="raw-base",
        help=(
            "raw-base uses the original 059/060 Excel rows for the data-distribution figure; "
            "protocol uses the windowed .mat samples that enter the training protocol."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing raw station Excel files when --source raw-base is used.",
    )
    parser.add_argument(
        "--protocol-dir",
        default="protocol_data/high_wind_spring_noft_four_client",
        help="Directory containing protocol metadata and station .mat files.",
    )
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--month", type=int, default=4)
    parser.add_argument(
        "--normal-start",
        default=None,
        help="Inclusive datetime bound for normal-weather rows in raw-base mode.",
    )
    parser.add_argument(
        "--normal-end",
        default=None,
        help="Inclusive datetime bound for normal-weather rows in raw-base mode.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/high_wind_spring_noft_four_client/power_distribution",
        help="Directory for figure outputs.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_station_assets(protocol_dir):
    metadata_path = Path(protocol_dir) / "protocol_metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    assets = []
    for station in metadata["stations"]:
        station_id = str(station["station_id"])
        asset_path = Path(protocol_dir) / station["asset_path"]
        assets.append((station_id, asset_path))
    return metadata, assets


def flatten_power(array):
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return np.clip(values, 0.0, 1.0)


def parse_iso8601(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def cell_text(cell, shared_strings):
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    value_node = cell.find(f"{ns}v")
    inline_string = cell.find(f"{ns}is")
    if value_node is not None and value_node.text is not None:
        value = value_node.text
        if cell.attrib.get("t") == "s":
            return shared_strings[int(value)]
        return value
    if inline_string is not None:
        return "".join(text.text or "" for text in inline_string.iter(f"{ns}t"))
    return ""


def column_letters_to_index(cell_ref):
    column_letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in column_letters:
        index = index * 26 + (ord(ch.upper()) - ord("A") + 1)
    return max(0, index - 1)


def row_values(row, shared_strings, header_len=None):
    cells = list(row)
    if header_len is None:
        header_len = 0
        for cell in cells:
            header_len = max(header_len, column_letters_to_index(cell.attrib.get("r", "")) + 1)
    values = [""] * header_len
    for cell in cells:
        column_index = column_letters_to_index(cell.attrib.get("r", ""))
        if column_index >= len(values):
            values.extend([""] * (column_index + 1 - len(values)))
        values[column_index] = cell_text(cell, shared_strings)
    return values


def load_shared_strings(workbook_zip):
    shared_strings = []
    shared_path = "xl/sharedStrings.xml"
    if shared_path not in workbook_zip.namelist():
        return shared_strings
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    root = ET.fromstring(workbook_zip.read(shared_path))
    for item in root:
        shared_strings.append("".join(node.text or "" for node in item.iter(f"{ns}t")))
    return shared_strings


def load_sheet_power_rows(path, sheet_name, capacity):
    ns_main = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    ns_rel = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    rows_out = []
    with ZipFile(path) as workbook_zip:
        workbook_root = ET.fromstring(workbook_zip.read("xl/workbook.xml"))
        rel_root = ET.fromstring(workbook_zip.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rel_root}
        shared_strings = load_shared_strings(workbook_zip)

        target_path = None
        for sheet in workbook_root.find(f"{ns_main}sheets"):
            if sheet.attrib["name"] == sheet_name:
                target_path = rel_map[sheet.attrib[f"{ns_rel}id"]]
                break
        if target_path is None:
            raise ValueError(f"sheet not found: {sheet_name} in {path}")

        sheet_root = ET.fromstring(workbook_zip.read(f"xl/{target_path}"))
        sheet_rows = sheet_root.find(f"{ns_main}sheetData")
        iterator = iter(sheet_rows)
        header_row = next(iterator)
        headers = row_values(header_row, shared_strings)
        date_index = headers.index("date")
        power_index = headers.index(TARGET_COL)

        for row in iterator:
            raw_values = row_values(row, shared_strings, header_len=len(headers))
            if len(raw_values) <= max(date_index, power_index):
                continue
            if not raw_values[date_index] or not raw_values[power_index]:
                continue
            try:
                dt_value = parse_iso8601(raw_values[date_index])
                power_value = float(raw_values[power_index]) / float(capacity)
            except Exception:
                continue
            rows_out.append((dt_value, power_value))
    return rows_out


def collect_raw_base_power_values(data_dir, year, month, normal_start, normal_end):
    normal_values = []
    high_wind_values = []
    per_station_rows = []
    normal_start_dt = parse_iso8601(normal_start) if normal_start else None
    normal_end_dt = parse_iso8601(normal_end) if normal_end else None

    for station in RAW_BASE_STATIONS:
        station_id = station["station_id"]
        workbook_path = Path(data_dir) / station["workbook"]
        capacity = float(station["capacity"])

        normal_rows = load_sheet_power_rows(workbook_path, "normal_weather", capacity)
        high_wind_rows = load_sheet_power_rows(workbook_path, "extreme_high_wind", capacity)

        if normal_start_dt is not None and normal_end_dt is not None:
            normal = np.asarray(
                [power for dt_value, power in normal_rows if normal_start_dt <= dt_value <= normal_end_dt],
                dtype=np.float64,
            )
        else:
            normal = np.asarray(
                [power for dt_value, power in normal_rows if dt_value.year == year and dt_value.month == month],
                dtype=np.float64,
            )
        high_wind = np.asarray(
            [power for dt_value, power in high_wind_rows if dt_value.year == year and dt_value.month == month],
            dtype=np.float64,
        )
        normal = flatten_power(normal)
        high_wind = flatten_power(high_wind)
        normal_values.append(normal)
        high_wind_values.append(high_wind)
        per_station_rows.append(
            {
                "station_id": station_id,
                "normal_n": int(normal.size),
                "normal_mean": float(np.mean(normal)),
                "normal_median": float(np.median(normal)),
                "high_wind_n": int(high_wind.size),
                "high_wind_mean": float(np.mean(high_wind)),
                "high_wind_median": float(np.median(high_wind)),
            }
        )
    return np.concatenate(normal_values), np.concatenate(high_wind_values), per_station_rows


def bounded_kde(values, grid):
    kde = gaussian_kde(values)
    density = kde(grid)
    density[grid < 0.0] = 0.0
    density[grid > 1.0] = 0.0
    area = np.trapz(density, grid)
    if area > 0:
        density = density / area
    return density


def collect_power_values(assets):
    normal_values = []
    high_wind_values = []
    per_station_rows = []
    for station_id, asset_path in assets:
        mat = scio.loadmat(asset_path)
        normal = flatten_power(mat["p_conven"])
        high_wind = flatten_power(mat["p_extre_class1"])
        normal_values.append(normal)
        high_wind_values.append(high_wind)
        per_station_rows.append(
            {
                "station_id": station_id,
                "normal_n": int(normal.size),
                "normal_mean": float(np.mean(normal)),
                "normal_median": float(np.median(normal)),
                "high_wind_n": int(high_wind.size),
                "high_wind_mean": float(np.mean(high_wind)),
                "high_wind_median": float(np.median(high_wind)),
            }
        )
    return np.concatenate(normal_values), np.concatenate(high_wind_values), per_station_rows


def apply_ieee_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_distribution(normal_values, high_wind_values, output_dir, dpi):
    apply_ieee_style()
    grid = np.linspace(0.0, 1.0, 401)
    normal_density = bounded_kde(normal_values, grid)
    high_wind_density = bounded_kde(high_wind_values, grid)

    normal_color = "#0072B2"
    high_wind_color = "#D55E00"

    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    ax.fill_between(grid, normal_density, color=normal_color, alpha=0.18, linewidth=0)
    ax.plot(
        grid,
        normal_density,
        color=normal_color,
        linewidth=1.55,
        label=f"Normal weather (n={normal_values.size})",
    )
    ax.fill_between(grid, high_wind_density, color=high_wind_color, alpha=0.20, linewidth=0)
    ax.plot(
        grid,
        high_wind_density,
        color=high_wind_color,
        linewidth=1.55,
        label=f"High wind (n={high_wind_values.size})",
    )

    normal_median = float(np.median(normal_values))
    high_wind_median = float(np.median(high_wind_values))
    ax.axvline(normal_median, color=normal_color, linestyle="--", linewidth=0.85, alpha=0.75)
    ax.axvline(high_wind_median, color=high_wind_color, linestyle="--", linewidth=0.85, alpha=0.75)

    y_text = ax.get_ylim()[1] * 0.86
    ax.text(normal_median - 0.015, y_text, "median", color=normal_color, ha="right", va="center", fontsize=6.8)
    ax.text(high_wind_median + 0.015, y_text, "median", color=high_wind_color, ha="left", va="center", fontsize=6.8)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Normalized power output (p.u.)")
    ax.set_ylabel("Density")
    ax.set_title("Power distribution in high-wind and normal weather")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="0.88", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.4)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "high_wind_power_distribution_2022_04"
    for suffix in ["pdf", "svg"]:
        fig.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")
    return stem


def write_summary(output_dir, metadata, normal_values, high_wind_values, per_station_rows):
    output_path = Path(output_dir) / "high_wind_power_distribution_summary.csv"
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("station_id,normal_n,normal_mean,normal_median,high_wind_n,high_wind_mean,high_wind_median\n")
        for row in per_station_rows:
            fh.write(
                f"{row['station_id']},{row['normal_n']},{row['normal_mean']:.6f},"
                f"{row['normal_median']:.6f},{row['high_wind_n']},"
                f"{row['high_wind_mean']:.6f},{row['high_wind_median']:.6f}\n"
            )
        fh.write(
            f"ALL,{normal_values.size},{np.mean(normal_values):.6f},{np.median(normal_values):.6f},"
            f"{high_wind_values.size},{np.mean(high_wind_values):.6f},{np.median(high_wind_values):.6f}\n"
        )
    metadata_path = Path(output_dir) / "high_wind_power_distribution_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "normal_train_start": metadata.get("normal_train_start"),
                "normal_train_end": metadata.get("normal_train_end"),
                "data_source": metadata.get("data_source"),
                "high_wind_filter": metadata.get("high_wind_filter"),
                "extreme_support_sampling_policy": metadata.get("extreme_support_sampling_policy"),
                "stations": [str(item["station_id"]) for item in metadata["stations"]],
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )


def main():
    args = parse_args()
    if args.source == "raw-base":
        metadata = {
            "data_source": "raw_base_excel_rows",
            "normal_train_start": args.normal_start,
            "normal_train_end": args.normal_end,
            "high_wind_filter": f"{args.year:04d}-{args.month:02d} extreme_high_wind raw rows",
            "normal_filter": (
                f"{args.year:04d}-{args.month:02d} normal_weather raw rows"
                if args.normal_start is None or args.normal_end is None
                else f"{args.normal_start} to {args.normal_end} normal_weather raw rows"
            ),
            "extreme_support_sampling_policy": "raw_rows_no_downsampling_no_window_filtering",
            "stations": [{"station_id": item["station_id"]} for item in RAW_BASE_STATIONS],
        }
        normal_values, high_wind_values, per_station_rows = collect_raw_base_power_values(
            args.data_dir,
            args.year,
            args.month,
            args.normal_start,
            args.normal_end,
        )
    else:
        metadata, assets = load_station_assets(args.protocol_dir)
        metadata["data_source"] = "protocol_windowed_mat_samples"
        normal_values, high_wind_values, per_station_rows = collect_power_values(assets)
    stem = plot_distribution(normal_values, high_wind_values, args.output_dir, args.dpi)
    write_summary(args.output_dir, metadata, normal_values, high_wind_values, per_station_rows)
    print(f"saved: {stem.with_suffix('.png')}")
    print(f"saved: {stem.with_suffix('.pdf')}")
    print(f"saved: {stem.with_suffix('.svg')}")
    print(f"saved: {stem.with_suffix('.tiff')}")
    print(f"normal: n={normal_values.size}, mean={np.mean(normal_values):.4f}, median={np.median(normal_values):.4f}")
    print(f"high_wind: n={high_wind_values.size}, mean={np.mean(high_wind_values):.4f}, median={np.median(high_wind_values):.4f}")


if __name__ == "__main__":
    main()
