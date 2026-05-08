#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model


DEFAULT_PROTOCOL_DATA_DIR = Path("protocol_data/high_wind_spring_noft_four_client")
DEFAULT_ARTIFACT_DIR = Path("artifacts/high_wind_spring_noft_four_client/pretrain-diagnostic")
DEFAULT_CHANNELS = (128, 96, 64, 48, 32, 16, 8)
BASELINE_VARIANTS = ("constant_0p9", "train_mean", "train_median")
TRAINED_VARIANTS = ("long_sequence", "window6", "window6_highwind_weighted")
ALL_VARIANTS = BASELINE_VARIANTS + TRAINED_VARIANTS


@dataclass(frozen=True)
class VariantConfig:
    name: str
    train_layout: str
    weighted: bool = False


VARIANT_CONFIGS = {
    "long_sequence": VariantConfig("long_sequence", "long", weighted=False),
    "window6": VariantConfig("window6", "window", weighted=False),
    "window6_highwind_weighted": VariantConfig(
        "window6_highwind_weighted",
        "window",
        weighted=True,
    ),
}


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: tuple[int, ...],
        mode: str = "pre",
        kernel_size: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(
                model.TemporalBlock_v2(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                    mode=mode,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ModelFore(nn.Module):
    def __init__(
        self,
        input_channel_fore: int = 5,
        output_channel_fore: tuple[int, ...] = DEFAULT_CHANNELS,
        mode: str = "pre",
        output_size_baselearner: int = 1,
        kernel_size: int = 2,
        dropout: float = 0.3,
        emb_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.tcn = TemporalConvNet(input_channel_fore, output_channel_fore, mode, kernel_size, dropout)
        self.drop = nn.Dropout(emb_dropout)
        self.fore_baselearner = nn.Linear(output_channel_fore[-1], output_size_baselearner)
        self.init_weights()

    def init_weights(self) -> None:
        self.fore_baselearner.bias.data.fill_(0)
        self.fore_baselearner.weight.data.normal_(0, 0.01)

    def get_trainable_params(self):
        if self.mode == "pre":
            return self.parameters()
        trainable_params = []
        for module in self.tcn.modules():
            if hasattr(module, "get_trainable_params"):
                trainable_params.extend(module.get_trainable_params())
        trainable_params.extend(self.fore_baselearner.parameters())
        return trainable_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(x)
        y = self.tcn(y.transpose(1, 2)).transpose(1, 2)
        return self.fore_baselearner(y).contiguous()


@dataclass
class StationData:
    station_id: str
    normal_nwp: np.ndarray
    normal_power: np.ndarray
    normal_nwp_norm: np.ndarray
    test_nwp_norm: np.ndarray
    test_power: np.ndarray
    train_scales: np.ndarray
    len_realp: int

    @property
    def window_count(self) -> int:
        return self.test_power.shape[0] // self.len_realp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose high-wind Pretrain behavior by comparing long-sequence, "
            "strict-window, and high-wind-weighted normal-weather pretraining."
        )
    )
    parser.add_argument("--protocol-data-dir", type=Path, default=DEFAULT_PROTOCOL_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--stations", default="", help="Comma-separated station ids. Defaults to metadata stations.")
    parser.add_argument("--variants", default=",".join(ALL_VARIANTS), help="Comma-separated variants to run.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true", help="Use one epoch and write into artifact-dir/smoke.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--high-wind-threshold", type=float, default=10.0)
    parser.add_argument("--high-wind-weight", type=float, default=4.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-models", action="store_true")
    return parser.parse_args()


def load_metadata(protocol_data_dir: Path) -> dict:
    metadata_path = protocol_data_dir / "protocol_metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_station_ids(args: argparse.Namespace, metadata: dict) -> list[str]:
    if args.stations.strip():
        return [part.strip() for part in args.stations.split(",") if part.strip()]
    stations = metadata.get("stations", [])
    if stations:
        return [str(station["station_id"]) for station in stations]
    return [path.name.split("wf_")[0] for path in sorted(args.protocol_data_dir.glob("*wf_4_train.mat"))]


def resolve_len_realp(metadata: dict) -> int:
    return int(metadata.get("len_realp", 6))


def get_cell_vector(cell_array: np.ndarray, feature_idx: int) -> np.ndarray:
    value = cell_array[0, feature_idx]
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = value.item()
    return np.asarray(value, dtype=np.float32).reshape(-1, 1)


def load_station_data(protocol_data_dir: Path, station_id: str, len_realp: int) -> StationData:
    mat_path = protocol_data_dir / f"{station_id}wf_4_train.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing station mat: {mat_path}")

    payload = scio.loadmat(mat_path)
    normal_nwp = np.asarray(payload["nwp_1h"][:, [0, 1, 2, 3, 4]], dtype=np.float32)
    normal_power = np.asarray(payload["p_1h"], dtype=np.float32).reshape(-1, 1)
    train_scales = np.max(np.abs(normal_nwp), axis=0)
    train_scales = np.where(train_scales == 0.0, 1.0, train_scales).astype(np.float32)
    normal_nwp_norm = normal_nwp / train_scales

    test_nwp_raw = np.concatenate(
        [get_cell_vector(payload["nwp_test_extre_class1_"], i) for i in range(5)],
        axis=1,
    )
    test_power = np.asarray(payload["p_test_extre_class1"], dtype=np.float32).reshape(-1, 1)
    usable_test_points = (test_power.shape[0] // len_realp) * len_realp
    if usable_test_points <= 0:
        raise ValueError(f"Station {station_id} has no full high-wind test window")

    return StationData(
        station_id=station_id,
        normal_nwp=normal_nwp,
        normal_power=normal_power,
        normal_nwp_norm=normal_nwp_norm,
        test_nwp_norm=(test_nwp_raw / train_scales)[:usable_test_points],
        test_power=test_power[:usable_test_points],
        train_scales=train_scales,
        len_realp=len_realp,
    )


def make_long_tensor(station: StationData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor(station.normal_nwp_norm.reshape(1, -1, 5), dtype=torch.float32)
    y = torch.tensor(station.normal_power.reshape(1, -1, 1), dtype=torch.float32)
    weights = torch.ones_like(y)
    return x, y, weights


def make_window_tensor(station: StationData, high_wind_threshold: float, high_wind_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    usable_points = (station.normal_power.shape[0] // station.len_realp) * station.len_realp
    x = station.normal_nwp_norm[:usable_points].reshape(-1, station.len_realp, 5)
    y = station.normal_power[:usable_points].reshape(-1, station.len_realp, 1)
    raw_wind = station.normal_nwp[:usable_points, 0].reshape(-1, station.len_realp, 1)
    weights = np.where(raw_wind >= high_wind_threshold, high_wind_weight, 1.0).astype(np.float32)
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


def make_test_tensor(station: StationData) -> tuple[torch.Tensor, np.ndarray]:
    x = station.test_nwp_norm.reshape(-1, station.len_realp, 5)
    y = station.test_power.reshape(-1, station.len_realp)
    return torch.tensor(x, dtype=torch.float32), y


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(weights * (pred - target) ** 2) / torch.clamp(torch.sum(weights), min=1.0)


def train_pretrain_variant(
    station: StationData,
    variant: VariantConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ModelFore, dict]:
    if variant.train_layout == "long":
        train_x, train_y, train_weights = make_long_tensor(station)
    else:
        train_x, train_y, train_weights = make_window_tensor(
            station,
            high_wind_threshold=args.high_wind_threshold,
            high_wind_weight=args.high_wind_weight,
        )
    if not variant.weighted:
        train_weights = torch.ones_like(train_weights)

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    train_weights = train_weights.to(device)

    model_fore = ModelFore(mode="pre").to(device)
    optimizer = torch.optim.Adam(model_fore.get_trainable_params(), lr=args.lr, betas=(0.5, 0.999))
    start_time = time.time()
    final_loss = float("nan")

    for epoch in range(args.epochs):
        model_fore.train()
        optimizer.zero_grad()
        pred = model_fore(train_x)
        loss = weighted_mse(pred, train_y, train_weights)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
        if args.log_interval > 0 and ((epoch + 1) % args.log_interval == 0 or epoch == 0 or epoch + 1 == args.epochs):
            print(
                f"station={station.station_id} variant={variant.name} "
                f"epoch={epoch + 1}/{args.epochs} train_weighted_mse={final_loss:.6f}",
                flush=True,
            )

    elapsed_seconds = time.time() - start_time
    return model_fore, {
        "train_weighted_mse": final_loss,
        "elapsed_seconds": elapsed_seconds,
        "train_layout": variant.train_layout,
        "weighted": variant.weighted,
    }


def calc_metrics(true_events: np.ndarray, pred_events: np.ndarray) -> dict:
    err = true_events - pred_events
    return {
        "nMAE_%": float(np.mean(np.abs(err)) * 100.0),
        "nRMSE_%": float(np.sqrt(np.mean(err ** 2)) * 100.0),
        "bias_%": float(np.mean(pred_events - true_events) * 100.0),
        "true_mean": float(np.mean(true_events)),
        "pred_mean": float(np.mean(pred_events)),
        "true_min": float(np.min(true_events)),
        "true_max": float(np.max(true_events)),
        "pred_min": float(np.min(pred_events)),
        "pred_max": float(np.max(pred_events)),
    }


def evaluate_model(model_fore: ModelFore, station: StationData, device: torch.device) -> tuple[dict, np.ndarray]:
    test_x, true_events = make_test_tensor(station)
    model_fore.eval()
    with torch.no_grad():
        pred_events = model_fore(test_x.to(device)).detach().cpu().numpy().reshape(true_events.shape)
    return calc_metrics(true_events, pred_events), pred_events


def evaluate_constant(station: StationData, value: float) -> tuple[dict, np.ndarray]:
    _, true_events = make_test_tensor(station)
    pred_events = np.full_like(true_events, value, dtype=np.float32)
    return calc_metrics(true_events, pred_events), pred_events


def evaluate_train_statistic(station: StationData, statistic: str) -> tuple[dict, np.ndarray]:
    if statistic == "mean":
        value = float(np.mean(station.normal_power))
    elif statistic == "median":
        value = float(np.median(station.normal_power))
    else:
        raise ValueError(statistic)
    return evaluate_constant(station, value)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_overall_rows(result_rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in result_rows:
        groups.setdefault(row["variant"], []).append(row)

    overall_rows = []
    for variant, rows in sorted(groups.items()):
        weights = np.asarray([float(row["samples"]) for row in rows], dtype=np.float64)
        metric_row = {
            "station": "Overall_SampleWeighted",
            "variant": variant,
            "samples": int(np.sum(weights)),
        }
        for column in ["nMAE_%", "nRMSE_%", "bias_%", "true_mean", "pred_mean", "true_min", "true_max", "pred_min", "pred_max"]:
            values = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
            metric_row[column] = float(np.average(values, weights=weights))
        metric_row["train_weighted_mse"] = np.nan
        metric_row["elapsed_seconds"] = float(np.nansum([float(row["elapsed_seconds"]) for row in rows]))
        metric_row["train_layout"] = rows[0]["train_layout"]
        metric_row["weighted"] = rows[0]["weighted"]
        overall_rows.append(metric_row)
    return overall_rows


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.artifact_dir = args.artifact_dir / "smoke"
        args.log_interval = 1

    selected_variants = [part.strip() for part in args.variants.split(",") if part.strip()]
    unknown = sorted(set(selected_variants) - set(ALL_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; available={ALL_VARIANTS}")

    set_reproducible_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.protocol_data_dir)
    len_realp = resolve_len_realp(metadata)
    station_ids = resolve_station_ids(args, metadata)

    print("=" * 70)
    print("High-wind Pretrain Diagnostic")
    print("=" * 70)
    print(f"protocol_data_dir={args.protocol_data_dir}")
    print(f"artifact_dir={args.artifact_dir}")
    print(f"stations={','.join(station_ids)}")
    print(f"variants={','.join(selected_variants)}")
    print(f"epochs={args.epochs} device={device} len_realp={len_realp}")
    print(f"high_wind_threshold={args.high_wind_threshold} high_wind_weight={args.high_wind_weight}")

    result_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for station_id in station_ids:
        station = load_station_data(args.protocol_data_dir, station_id, len_realp)
        print(
            f"\nstation={station_id} train_points={station.normal_power.shape[0]} "
            f"test_windows={station.window_count} test_true_mean={float(np.mean(station.test_power)):.4f}",
            flush=True,
        )

        for variant_name in selected_variants:
            train_info = {
                "train_weighted_mse": np.nan,
                "elapsed_seconds": 0.0,
                "train_layout": "baseline",
                "weighted": False,
            }
            if variant_name == "constant_0p9":
                metrics, pred_events = evaluate_constant(station, 0.9)
            elif variant_name == "train_mean":
                metrics, pred_events = evaluate_train_statistic(station, "mean")
            elif variant_name == "train_median":
                metrics, pred_events = evaluate_train_statistic(station, "median")
            else:
                trained_model, train_info = train_pretrain_variant(
                    station,
                    VARIANT_CONFIGS[variant_name],
                    args,
                    device,
                )
                metrics, pred_events = evaluate_model(trained_model, station, device)
                if args.save_models:
                    model_path = args.artifact_dir / "models" / f"model_pretrain_diag_station{station_id}_{variant_name}.pth"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(trained_model.state_dict(), model_path)

            _, true_events = make_test_tensor(station)
            row = {
                "station": station_id,
                "variant": variant_name,
                "samples": int(true_events.shape[0]),
                **metrics,
                **train_info,
            }
            result_rows.append(row)
            print(
                f"  {variant_name}: nMAE={metrics['nMAE_%']:.4f}% "
                f"pred_mean={metrics['pred_mean']:.4f} bias={metrics['bias_%']:.4f}%",
                flush=True,
            )

            for event_idx in range(true_events.shape[0]):
                for step_idx in range(true_events.shape[1]):
                    prediction_rows.append(
                        {
                            "station": station_id,
                            "variant": variant_name,
                            "event": event_idx,
                            "step": step_idx,
                            "true": float(true_events[event_idx, step_idx]),
                            "pred": float(pred_events[event_idx, step_idx]),
                        }
                    )

    overall_rows = build_overall_rows(result_rows)
    all_result_rows = result_rows + overall_rows
    result_fields = [
        "station",
        "variant",
        "samples",
        "nMAE_%",
        "nRMSE_%",
        "bias_%",
        "true_mean",
        "pred_mean",
        "true_min",
        "true_max",
        "pred_min",
        "pred_max",
        "train_weighted_mse",
        "elapsed_seconds",
        "train_layout",
        "weighted",
    ]
    prediction_fields = ["station", "variant", "event", "step", "true", "pred"]

    write_csv(args.artifact_dir / "diagnostic_results.csv", all_result_rows, result_fields)
    write_csv(args.artifact_dir / "diagnostic_overall.csv", overall_rows, result_fields)
    write_csv(args.artifact_dir / "diagnostic_predictions.csv", prediction_rows, prediction_fields)

    print("\nOverall sample-weighted:")
    for row in overall_rows:
        print(
            f"  {row['variant']}: nMAE={row['nMAE_%']:.4f}% "
            f"pred_mean={row['pred_mean']:.4f} bias={row['bias_%']:.4f}%"
        )
    print(f"\nSaved: {args.artifact_dir / 'diagnostic_results.csv'}")
    print(f"Saved: {args.artifact_dir / 'diagnostic_overall.csv'}")
    print(f"Saved: {args.artifact_dir / 'diagnostic_predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
