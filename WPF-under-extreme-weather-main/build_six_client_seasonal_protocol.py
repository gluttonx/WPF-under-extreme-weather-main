#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build six-client seasonal protocol assets from raw station xlsx files.

This script avoids any dependency on openpyxl by parsing xlsx XML directly.
"""
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
from scipy.io import savemat

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "seasonal_protocol_data"
METADATA_PATH = OUTPUT_DIR / "seasonal_protocol_metadata.json"

SELECTED_FEATURES = [
    "wind_speed_100m",
    "wind_direction_100m",
    "temperature_2m",
    "pressure_msl",
    "relative_humidity_2m",
]
TARGET_COL = "Power2"
EXTREME_SHEETS = OrderedDict(
    [
        ("extreme_high_wind", 0),
        ("extreme_high_temp", 1),
        ("extreme_cold_wave", 2),
        ("extreme_frost", 3),
    ]
)
EXTREME_CLASS_NAMES = ["high_wind", "high_temp", "cold_wave", "frost"]
CAPACITY_BY_WORKBOOK = {
    "2223jilin_058_processed_4classes.xlsx": 50.0,
    "2223jilin_059_processed_4classes.xlsx": 50.0,
    "2223jilin_060_processed_4classes.xlsx": 100.0,
    "24jilin_058_processed_4classes.xlsx": 50.0,
    "24jilin_059_processed_4classes.xlsx": 100.0,
    "24jilin_060_processed_4classes.xlsx": 300.0,
}
META_SUPPORT_SHOTS = 5
META_QUERY_SHOTS = 5
LEN_REALP = 12

SEASONAL_PROTOCOL_CLIENTS = [
    {
        "client_id": "58",
        "client_name": "WT1",
        "source_station_id": "058",
        "train_start": "2022-03-01T00:00:00+08:00",
        "train_end": "2022-06-01T00:00:00+08:00",
        "test_start": "2023-03-01T00:00:00+08:00",
        "test_end": "2023-06-01T00:00:00+08:00",
        "train_workbooks": ["2223jilin_058_processed_4classes.xlsx"],
        "test_workbooks": ["2223jilin_058_processed_4classes.xlsx"],
    },
    {
        "client_id": "59",
        "client_name": "WT2",
        "source_station_id": "059",
        "train_start": "2022-03-01T00:00:00+08:00",
        "train_end": "2022-06-01T00:00:00+08:00",
        "test_start": "2023-03-01T00:00:00+08:00",
        "test_end": "2023-06-01T00:00:00+08:00",
        "train_workbooks": ["2223jilin_059_processed_4classes.xlsx"],
        "test_workbooks": ["2223jilin_059_processed_4classes.xlsx"],
    },
    {
        "client_id": "60",
        "client_name": "WT3",
        "source_station_id": "060",
        "train_start": "2022-06-01T00:00:00+08:00",
        "train_end": "2022-09-01T00:00:00+08:00",
        "test_start": "2023-06-01T00:00:00+08:00",
        "test_end": "2023-09-01T00:00:00+08:00",
        "train_workbooks": ["2223jilin_060_processed_4classes.xlsx"],
        "test_workbooks": ["2223jilin_060_processed_4classes.xlsx"],
    },
    {
        "client_id": "61",
        "client_name": "WT4",
        "source_station_id": "058",
        "train_start": "2022-06-01T00:00:00+08:00",
        "train_end": "2022-09-01T00:00:00+08:00",
        "test_start": "2023-06-01T00:00:00+08:00",
        "test_end": "2023-09-01T00:00:00+08:00",
        "train_workbooks": ["2223jilin_058_processed_4classes.xlsx"],
        "test_workbooks": ["2223jilin_058_processed_4classes.xlsx"],
    },
    {
        "client_id": "62",
        "client_name": "WT5",
        "source_station_id": "059",
        "train_start": "2022-11-01T00:00:00+08:00",
        "train_end": "2023-02-01T00:00:00+08:00",
        "test_start": "2023-11-01T00:00:00+08:00",
        "test_end": "2024-02-01T00:00:00+08:00",
        "train_workbooks": ["2223jilin_059_processed_4classes.xlsx"],
        "test_workbooks": ["2223jilin_059_processed_4classes.xlsx", "24jilin_059_processed_4classes.xlsx"],
    },
    {
        "client_id": "63",
        "client_name": "WT6",
        "source_station_id": "060",
        "train_start": "2022-11-01T00:00:00+08:00",
        "train_end": "2023-02-01T00:00:00+08:00",
        "test_start": "2023-11-01T00:00:00+08:00",
        "test_end": "2024-02-01T00:00:00+08:00",
        "train_workbooks": ["2223jilin_060_processed_4classes.xlsx"],
        "test_workbooks": ["2223jilin_060_processed_4classes.xlsx", "24jilin_060_processed_4classes.xlsx"],
    },
]

MAIN_SHEET_BY_SOURCE = {
    "058": "jilin_058",
    "059": "jilin_059",
    "060": "jilin_060",
}


@dataclass
class SheetRecord:
    date: datetime
    values: Dict[str, float]


def clone_sheet_record(record: SheetRecord) -> SheetRecord:
    return SheetRecord(
        date=record.date,
        values=dict(record.values),
    )


def clone_workbook_sheets(workbook: Dict[str, List[SheetRecord]]) -> Dict[str, List[SheetRecord]]:
    return {
        sheet_name: [clone_sheet_record(record) for record in records]
        for sheet_name, records in workbook.items()
    }


def parse_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _cell_text(cell, shared_strings):
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


def _column_letters_to_index(cell_ref: str) -> int:
    column_letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in column_letters:
        index = index * 26 + (ord(ch.upper()) - ord("A") + 1)
    return max(0, index - 1)


def _row_values(row, shared_strings, header_len: int = None):
    cells = list(row)
    if header_len is None:
        header_len = 0
        for cell in cells:
            header_len = max(header_len, _column_letters_to_index(cell.attrib.get("r", "")) + 1)
    values = [""] * header_len
    for cell in cells:
        column_index = _column_letters_to_index(cell.attrib.get("r", ""))
        if column_index >= len(values):
            values.extend([""] * (column_index + 1 - len(values)))
        values[column_index] = _cell_text(cell, shared_strings)
    return values


def _load_shared_strings(workbook_zip: ZipFile) -> List[str]:
    shared_strings = []
    shared_path = "xl/sharedStrings.xml"
    if shared_path not in workbook_zip.namelist():
        return shared_strings
    root = ET.fromstring(workbook_zip.read(shared_path))
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    for item in root:
        shared_strings.append("".join(node.text or "" for node in item.iter(f"{ns}t")))
    return shared_strings


def load_xlsx_workbook(path: Path) -> Dict[str, List[SheetRecord]]:
    ns_main = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    ns_rel = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    workbook_data: Dict[str, List[SheetRecord]] = {}
    with ZipFile(path) as workbook_zip:
        workbook_root = ET.fromstring(workbook_zip.read("xl/workbook.xml"))
        rel_root = ET.fromstring(workbook_zip.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rel_root}
        shared_strings = _load_shared_strings(workbook_zip)

        for sheet in workbook_root.find(f"{ns_main}sheets"):
            sheet_name = sheet.attrib["name"]
            target = rel_map[sheet.attrib[f"{ns_rel}id"]]
            sheet_root = ET.fromstring(workbook_zip.read(f"xl/{target}"))
            rows = sheet_root.find(f"{ns_main}sheetData")
            iterator = iter(rows)
            header_row = next(iterator)
            headers = _row_values(header_row, shared_strings)
            try:
                date_index = headers.index("date")
            except ValueError:
                workbook_data[sheet_name] = []
                continue

            records: List[SheetRecord] = []
            for row in iterator:
                raw_values = _row_values(row, shared_strings, header_len=len(headers))
                if len(raw_values) <= date_index:
                    continue
                if not raw_values[date_index]:
                    continue
                payload = {}
                for header, value in zip(headers, raw_values):
                    payload[header] = value
                try:
                    date_value = parse_iso8601(payload["date"])
                except Exception:
                    continue
                numeric_payload = {}
                for key in SELECTED_FEATURES + [TARGET_COL]:
                    numeric_payload[key] = float(payload[key])
                records.append(SheetRecord(date=date_value, values=numeric_payload))
            workbook_data[sheet_name] = records
    return workbook_data


def slice_records(records: List[SheetRecord], start: datetime, end: datetime) -> List[SheetRecord]:
    return [record for record in records if start <= record.date < end]


def records_to_arrays(records: List[SheetRecord]) -> Tuple[np.ndarray, np.ndarray]:
    if not records:
        return np.empty((0, 1)), np.empty((0, len(SELECTED_FEATURES)))
    power = np.array([[record.values[TARGET_COL]] for record in records], dtype=np.float32)
    nwp = np.array([[record.values[name] for name in SELECTED_FEATURES] for record in records], dtype=np.float32)
    return power, nwp


def standardize_features(values: np.ndarray) -> np.ndarray:
    means = np.mean(values, axis=0)
    stds = np.std(values, axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return (values - means) / stds


def compute_protocol_k_max(normal_hours: int, len_realp: int = LEN_REALP, support_query_total: int = META_SUPPORT_SHOTS + META_QUERY_SHOTS) -> int:
    n_windows = normal_hours // len_realp
    return max(2, n_windows // support_query_total)


def _line_distance_elbow(inertias: List[float]) -> int:
    if len(inertias) == 1:
        return 0
    x = np.arange(len(inertias), dtype=np.float64)
    y = np.array(inertias, dtype=np.float64)
    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])
    line = end - start
    norm = np.linalg.norm(line)
    if norm < 1e-8:
        return 0
    distances = []
    for idx in range(len(inertias)):
        point = np.array([x[idx], y[idx]])
        distance = abs(np.cross(line, point - start)) / norm
        distances.append(distance)
    return int(np.argmax(distances))


def fit_kmeans_with_labels(features: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    if KMeans is not None:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(features)
        return labels, float(model.inertia_)

    rng = np.random.default_rng(42)
    if features.shape[0] < k:
        raise ValueError(f"样本数 {features.shape[0]} 小于 K={k}")

    center_indices = rng.choice(features.shape[0], size=k, replace=False)
    centers = features[center_indices].copy()
    labels = np.zeros(features.shape[0], dtype=np.int64)

    for _ in range(100):
        distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_index in range(k):
            cluster_points = features[labels == cluster_index]
            if cluster_points.size == 0:
                centers[cluster_index] = features[rng.integers(0, features.shape[0])]
            else:
                centers[cluster_index] = np.mean(cluster_points, axis=0)

    inertia = float(np.sum((features - centers[labels]) ** 2))
    return labels, inertia


def cluster_window_counts(labels: np.ndarray, len_realp: int) -> List[int]:
    counts = []
    for class_index in range(int(np.max(labels)) + 1):
        counts.append(int(np.sum(labels == class_index) // len_realp))
    return counts


def choose_feasible_k_by_elbow(
    normal_features: np.ndarray,
    len_realp: int = LEN_REALP,
    support_query_total: int = META_SUPPORT_SHOTS + META_QUERY_SHOTS,
) -> Tuple[int, Dict[str, object]]:
    normal_hours = int(normal_features.shape[0])
    k_max = min(8, compute_protocol_k_max(normal_hours, len_realp=len_realp, support_query_total=support_query_total))
    candidates = list(range(2, max(2, k_max) + 1))
    candidate_results = []
    inertias = []
    for k in candidates:
        labels, inertia = fit_kmeans_with_labels(normal_features, k)
        window_counts = cluster_window_counts(labels, len_realp=len_realp)
        candidate_results.append(
            {
                "k": k,
                "labels": labels,
                "inertia": inertia,
                "window_counts": window_counts,
                "is_feasible": min(window_counts) >= support_query_total,
            }
        )
        inertias.append(inertia)

    elbow_offset = _line_distance_elbow(inertias)
    chosen = candidate_results[elbow_offset]
    if not chosen["is_feasible"]:
        feasible_results = [result for result in reversed(candidate_results[: elbow_offset + 1]) if result["is_feasible"]]
        if not feasible_results:
            feasible_results = [result for result in reversed(candidate_results) if result["is_feasible"]]
        if feasible_results:
            chosen = feasible_results[0]
        else:
            chosen = min(candidate_results, key=lambda item: min(item["window_counts"]))
    info = {
        "candidate_k": candidates,
        "candidate_inertia": inertias,
        "chosen_window_counts": chosen["window_counts"],
        "k_max": k_max,
    }
    return int(chosen["k"]), {
        "labels": chosen["labels"],
        "info": info,
    }


def compute_sampler_class_count(k: int) -> int:
    return max(2, math.ceil(k / 2))


def normalize_power(records: List[SheetRecord], capacity: float):
    cap = float(capacity)
    for record in records:
        record.values[TARGET_COL] = float(record.values[TARGET_COL]) / cap


def merge_workbooks_by_sheet(workbooks: List[Dict[str, List[SheetRecord]]]) -> Dict[str, List[SheetRecord]]:
    merged: Dict[str, List[SheetRecord]] = {}
    for workbook in workbooks:
        for sheet_name, records in workbook.items():
            merged.setdefault(sheet_name, [])
            merged[sheet_name].extend(clone_sheet_record(record) for record in records)
    for sheet_name in merged:
        merged[sheet_name].sort(key=lambda record: record.date)
    return merged


def build_cluster_objects(records: List[SheetRecord], labels: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    power_by_class = np.empty((1, k), dtype=object)
    nwp_by_feature = np.empty((1, len(SELECTED_FEATURES)), dtype=object)
    per_feature_class_objects = []
    for _ in SELECTED_FEATURES:
        per_feature_class_objects.append(np.empty((1, k), dtype=object))

    for class_index in range(k):
        class_records = [record for record, label in zip(records, labels) if int(label) == class_index]
        class_power, class_nwp = records_to_arrays(class_records)
        power_by_class[0, class_index] = class_power.reshape(-1, 1)
        for feature_index, _feature_name in enumerate(SELECTED_FEATURES):
            per_feature_class_objects[feature_index][0, class_index] = class_nwp[:, feature_index].reshape(-1, 1)

    for feature_index in range(len(SELECTED_FEATURES)):
        nwp_by_feature[0, feature_index] = per_feature_class_objects[feature_index]

    return power_by_class, nwp_by_feature


def build_extreme_objects(records: List[SheetRecord]) -> Tuple[np.ndarray, np.ndarray]:
    power, nwp = records_to_arrays(records)
    nwp_objects = np.empty((1, len(SELECTED_FEATURES)), dtype=object)
    for feature_index in range(len(SELECTED_FEATURES)):
        nwp_objects[0, feature_index] = nwp[:, feature_index].reshape(-1, 1)
    return power.reshape(-1, 1), nwp_objects


def serialize_client_assets(client_config: Dict[str, object], workbook_cache: Dict[str, Dict[str, List[SheetRecord]]]) -> Dict[str, object]:
    source_station_id = client_config["source_station_id"]
    train_start = parse_iso8601(client_config["train_start"])
    train_end = parse_iso8601(client_config["train_end"])
    test_start = parse_iso8601(client_config["test_start"])
    test_end = parse_iso8601(client_config["test_end"])
    first_month_end = datetime.fromisoformat(client_config["train_start"])  # placeholder overwritten below
    if train_start.month == 11:
        first_month_end = datetime.fromisoformat(f"{train_start.year}-12-01T00:00:00+08:00")
    else:
        next_month = train_start.month + 1
        next_year = train_start.year + (1 if next_month == 13 else 0)
        next_month = 1 if next_month == 13 else next_month
        first_month_end = datetime.fromisoformat(f"{next_year:04d}-{next_month:02d}-01T00:00:00+08:00")

    train_workbooks = []
    for workbook_name in client_config["train_workbooks"]:
        workbook = clone_workbook_sheets(workbook_cache[workbook_name])
        capacity = CAPACITY_BY_WORKBOOK[workbook_name]
        for records in workbook.values():
            normalize_power(records, capacity)
        train_workbooks.append(workbook)

    test_workbooks = []
    for workbook_name in client_config["test_workbooks"]:
        workbook = clone_workbook_sheets(workbook_cache[workbook_name])
        capacity = CAPACITY_BY_WORKBOOK[workbook_name]
        for records in workbook.values():
            normalize_power(records, capacity)
        test_workbooks.append(workbook)

    merged_train = merge_workbooks_by_sheet(train_workbooks)
    merged_test = merge_workbooks_by_sheet(test_workbooks)

    main_sheet = MAIN_SHEET_BY_SOURCE[source_station_id]
    train_full = slice_records(merged_train[main_sheet], train_start, train_end)
    test_full = slice_records(merged_test[main_sheet], test_start, test_end)

    first_month_normal = slice_records(merged_train["normal_weather"], train_start, first_month_end)
    normal_power, normal_nwp = records_to_arrays(first_month_normal)
    standardized_normal = standardize_features(normal_nwp)
    chosen_k, clustering_payload = choose_feasible_k_by_elbow(
        standardized_normal,
        len_realp=LEN_REALP,
        support_query_total=META_SUPPORT_SHOTS + META_QUERY_SHOTS,
    )
    labels = clustering_payload["labels"]
    p_conven_class, nwp_conven_class = build_cluster_objects(first_month_normal, labels, chosen_k)

    train_power, train_nwp = records_to_arrays(train_full)
    test_power, test_nwp = records_to_arrays(test_full)

    mat_dict = {
        "p_1h": train_power.reshape(-1, 1),
        "nwp_1h": train_nwp,
        "p_conven": normal_power.reshape(-1, 1),
        "nwp_conven_": normal_nwp,
        "p_conven_class": p_conven_class,
        "nwp_conven_class_": nwp_conven_class,
        "p_test": test_power.reshape(-1, 1),
        "nwp_test": test_nwp,
    }

    valid_class_indices = []
    support_counts = {}
    test_counts = {}
    for sheet_name, class_index in EXTREME_SHEETS.items():
        train_records = slice_records(merged_train[sheet_name], train_start, train_end)
        test_records = slice_records(merged_test[sheet_name], test_start, test_end)
        train_power_class, train_nwp_class = build_extreme_objects(train_records)
        test_power_class, test_nwp_class = build_extreme_objects(test_records)
        mat_dict[f"p_extre_class{class_index + 1}"] = train_power_class
        mat_dict[f"nwp_extre_class{class_index + 1}_"] = train_nwp_class
        mat_dict[f"p_test_extre_class{class_index + 1}"] = test_power_class
        mat_dict[f"nwp_test_extre_class{class_index + 1}_"] = test_nwp_class
        support_counts[EXTREME_CLASS_NAMES[class_index]] = int(train_power_class.shape[0] // LEN_REALP)
        test_counts[EXTREME_CLASS_NAMES[class_index]] = int(test_power_class.shape[0] // LEN_REALP)
        if train_power_class.shape[0] > 0 and test_power_class.shape[0] > 0:
            valid_class_indices.append(class_index)

    client_mat_path = OUTPUT_DIR / f"{client_config['client_id']}wf_seasonal_protocol.mat"
    savemat(client_mat_path, mat_dict)
    return {
        "client_id": client_config["client_id"],
        "client_name": client_config["client_name"],
        "source_station_id": source_station_id,
        "train_start": client_config["train_start"],
        "train_end": client_config["train_end"],
        "test_start": client_config["test_start"],
        "test_end": client_config["test_end"],
        "main_sheet": main_sheet,
        "asset_path": client_mat_path.name,
        "valid_class_indices": valid_class_indices,
        "valid_class_names": [EXTREME_CLASS_NAMES[index] for index in valid_class_indices],
        "support_window_counts": support_counts,
        "test_window_counts": test_counts,
        "normal_hours": int(normal_power.shape[0]),
        "normal_window_count": int(normal_power.shape[0] // LEN_REALP),
        "chosen_k": chosen_k,
        "sampler_task_count": compute_sampler_class_count(chosen_k),
        "meta_support_shots": META_SUPPORT_SHOTS,
        "meta_query_shots": META_QUERY_SHOTS,
        "len_realp": LEN_REALP,
        "clustering": clustering_payload["info"],
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    workbook_cache = {}
    workbook_names = sorted({name for client in SEASONAL_PROTOCOL_CLIENTS for name in client["train_workbooks"] + client["test_workbooks"]})
    for workbook_name in workbook_names:
        workbook_cache[workbook_name] = load_xlsx_workbook(ROOT / workbook_name)

    metadata = {
        "protocol_name": "six_client_seasonal_protocol",
        "len_realp": LEN_REALP,
        "meta_support_shots": META_SUPPORT_SHOTS,
        "meta_query_shots": META_QUERY_SHOTS,
        "extreme_class_names": EXTREME_CLASS_NAMES,
        "clients": [],
    }
    for client in SEASONAL_PROTOCOL_CLIENTS:
        client_metadata = serialize_client_assets(client, workbook_cache)
        metadata["clients"].append(client_metadata)
        print(
            f"{client_metadata['client_name']} ({client_metadata['client_id']}): "
            f"K={client_metadata['chosen_k']}, "
            f"sampler_task_count={client_metadata['sampler_task_count']}, "
            f"valid_classes={client_metadata['valid_class_names']}"
        )

    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ saved seasonal protocol metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
