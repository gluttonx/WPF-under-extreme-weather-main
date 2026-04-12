#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build yearly 3-station protocol assets from raw 2022/2023 station xlsx files.

The new protocol keeps the original RAPP-style backbone inputs, but makes the
extreme few-shot path explicit:
  - 2022 extreme support
  - 2023 extreme test

It does not reuse old mixed-year p_extre_class* mats as source-of-truth.
"""
import json
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
from scipy.io import savemat

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "three_station_yearly_protocol_data"
METADATA_PATH = OUTPUT_DIR / "three_station_yearly_protocol_metadata.json"

SUPPORT_YEAR = 2022
TEST_YEAR = 2023
LEN_REALP = 12
NUM_CONVENTIONAL_CLASSES = 10

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
EXTREME_SUPPORT_POWER_KEYS = [
    "p_extre_class1",
    "p_extre_class2",
    "p_extre_class3",
    "p_extre_class4",
]
EXTREME_SUPPORT_NWP_KEYS = [
    "nwp_extre_class1_",
    "nwp_extre_class2_",
    "nwp_extre_class3_",
    "nwp_extre_class4_",
]
EXTREME_TEST_POWER_KEYS = [
    "p_test_extre_class1",
    "p_test_extre_class2",
    "p_test_extre_class3",
    "p_test_extre_class4",
]
EXTREME_TEST_NWP_KEYS = [
    "nwp_test_extre_class1_",
    "nwp_test_extre_class2_",
    "nwp_test_extre_class3_",
    "nwp_test_extre_class4_",
]
YEARLY_PROTOCOL_STATIONS = [
    {
        "station_id": "58",
        "source_station_id": "058",
        "main_sheet": "jilin_058",
        "workbook": "2223jilin_058_processed_4classes.xlsx",
        "capacity": 50.0,
    },
    {
        "station_id": "59",
        "source_station_id": "059",
        "main_sheet": "jilin_059",
        "workbook": "2223jilin_059_processed_4classes.xlsx",
        "capacity": 50.0,
    },
    {
        "station_id": "60",
        "source_station_id": "060",
        "main_sheet": "jilin_060",
        "workbook": "2223jilin_060_processed_4classes.xlsx",
        "capacity": 100.0,
    },
]


@dataclass
class SheetRecord:
    date: datetime
    values: Dict[str, float]


def clone_sheet_record(record: SheetRecord) -> SheetRecord:
    return SheetRecord(date=record.date, values=dict(record.values))


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
                if len(raw_values) <= date_index or not raw_values[date_index]:
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


def normalize_power(records: List[SheetRecord], capacity: float):
    for record in records:
        record.values[TARGET_COL] = float(record.values[TARGET_COL]) / float(capacity)


def split_records_by_year(records: List[SheetRecord], year: int) -> List[SheetRecord]:
    return [clone_sheet_record(record) for record in records if record.date.year == year]


def records_to_arrays(records: List[SheetRecord]) -> Tuple[np.ndarray, np.ndarray]:
    if not records:
        return (
            np.empty((0, 1), dtype=np.float32),
            np.empty((0, len(SELECTED_FEATURES)), dtype=np.float32),
        )
    power = np.array([[record.values[TARGET_COL]] for record in records], dtype=np.float32)
    nwp = np.array(
        [[record.values[feature_name] for feature_name in SELECTED_FEATURES] for record in records],
        dtype=np.float32,
    )
    return power, nwp


def build_extreme_objects(records: List[SheetRecord]) -> Tuple[np.ndarray, np.ndarray]:
    power, nwp = records_to_arrays(records)
    nwp_objects = np.empty((1, len(SELECTED_FEATURES)), dtype=object)
    for feature_index in range(len(SELECTED_FEATURES)):
        nwp_objects[0, feature_index] = nwp[:, feature_index].reshape(-1, 1)
    return power.reshape(-1, 1), nwp_objects


def standardize_features(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    means = np.mean(values, axis=0)
    stds = np.std(values, axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return (values - means) / stds


def fit_kmeans_with_labels(features: np.ndarray, k: int) -> np.ndarray:
    if features.shape[0] < k:
        raise ValueError(f"样本数 {features.shape[0]} 小于常规聚类数 {k}")
    if KMeans is not None:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        return model.fit_predict(features)

    rng = np.random.default_rng(42)
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
    return labels


def build_cluster_objects(records: List[SheetRecord], labels: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    power_by_class = np.empty((1, k), dtype=object)
    nwp_by_feature = np.empty((1, len(SELECTED_FEATURES)), dtype=object)
    per_feature_class_objects = [np.empty((1, k), dtype=object) for _ in SELECTED_FEATURES]

    for class_index in range(k):
        class_records = [record for record, label in zip(records, labels) if int(label) == class_index]
        class_power, class_nwp = records_to_arrays(class_records)
        power_by_class[0, class_index] = class_power.reshape(-1, 1)
        for feature_index in range(len(SELECTED_FEATURES)):
            per_feature_class_objects[feature_index][0, class_index] = class_nwp[:, feature_index].reshape(-1, 1)

    for feature_index in range(len(SELECTED_FEATURES)):
        nwp_by_feature[0, feature_index] = per_feature_class_objects[feature_index]
    return power_by_class, nwp_by_feature


def build_yearly_station_asset(station_config: Dict[str, object], workbook_cache: Dict[str, Dict[str, List[SheetRecord]]]) -> Dict[str, object]:
    workbook_name = station_config["workbook"]
    capacity = station_config["capacity"]
    workbook = workbook_cache[workbook_name]

    workbook_copy = {
        sheet_name: [clone_sheet_record(record) for record in records]
        for sheet_name, records in workbook.items()
    }
    for records in workbook_copy.values():
        normalize_power(records, capacity)

    main_sheet_records = workbook_copy[station_config["main_sheet"]]
    normal_records = workbook_copy["normal_weather"]

    yearly_main_support = split_records_by_year(main_sheet_records, SUPPORT_YEAR)
    yearly_main_test = split_records_by_year(main_sheet_records, TEST_YEAR)
    yearly_normal_support = split_records_by_year(normal_records, SUPPORT_YEAR)

    normal_power, normal_nwp = records_to_arrays(yearly_normal_support)
    standardized_normal = standardize_features(normal_nwp)
    labels = fit_kmeans_with_labels(standardized_normal, NUM_CONVENTIONAL_CLASSES)
    p_conven_class, nwp_conven_class = build_cluster_objects(
        yearly_normal_support,
        labels,
        NUM_CONVENTIONAL_CLASSES,
    )

    train_power, train_nwp = records_to_arrays(yearly_main_support)
    test_power, test_nwp = records_to_arrays(yearly_main_test)

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

    extreme_support_window_counts = {}
    extreme_test_window_counts = {}
    for sheet_name, class_index in EXTREME_SHEETS.items():
        support_records = split_records_by_year(workbook_copy[sheet_name], SUPPORT_YEAR)
        test_records = split_records_by_year(workbook_copy[sheet_name], TEST_YEAR)
        support_power, support_nwp = build_extreme_objects(support_records)
        test_power_class, test_nwp_class = build_extreme_objects(test_records)
        mat_dict[EXTREME_SUPPORT_POWER_KEYS[class_index]] = support_power
        mat_dict[EXTREME_SUPPORT_NWP_KEYS[class_index]] = support_nwp
        mat_dict[EXTREME_TEST_POWER_KEYS[class_index]] = test_power_class
        mat_dict[EXTREME_TEST_NWP_KEYS[class_index]] = test_nwp_class
        extreme_support_window_counts[EXTREME_CLASS_NAMES[class_index]] = int(support_power.shape[0] // LEN_REALP)
        extreme_test_window_counts[EXTREME_CLASS_NAMES[class_index]] = int(test_power_class.shape[0] // LEN_REALP)

    asset_path = OUTPUT_DIR / f"{station_config['station_id']}wf_yearly_protocol.mat"
    savemat(asset_path, mat_dict)

    return {
        "station_id": station_config["station_id"],
        "source_station_id": station_config["source_station_id"],
        "main_sheet": station_config["main_sheet"],
        "workbook": workbook_name,
        "asset_path": asset_path.name,
        "support_year": SUPPORT_YEAR,
        "test_year": TEST_YEAR,
        "main_support_hours": int(train_power.shape[0]),
        "main_test_hours": int(test_power.shape[0]),
        "conventional_support_hours": int(normal_power.shape[0]),
        "extreme_support_window_counts": extreme_support_window_counts,
        "extreme_test_window_counts": extreme_test_window_counts,
        "num_conventional_classes": NUM_CONVENTIONAL_CLASSES,
        "len_realp": LEN_REALP,
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    workbook_cache = {}
    for station_config in YEARLY_PROTOCOL_STATIONS:
        workbook_name = station_config["workbook"]
        workbook_cache[workbook_name] = load_xlsx_workbook(ROOT / workbook_name)

    metadata = {
        "protocol_name": "three_station_yearly_extreme_protocol",
        "support_year": SUPPORT_YEAR,
        "test_year": TEST_YEAR,
        "len_realp": LEN_REALP,
        "num_conventional_classes": NUM_CONVENTIONAL_CLASSES,
        "extreme_class_names": EXTREME_CLASS_NAMES,
        "stations": [],
    }

    for station_config in YEARLY_PROTOCOL_STATIONS:
        station_metadata = build_yearly_station_asset(station_config, workbook_cache)
        metadata["stations"].append(station_metadata)
        print(
            f"station {station_metadata['station_id']}: "
            f"support_windows={station_metadata['extreme_support_window_counts']}, "
            f"test_windows={station_metadata['extreme_test_window_counts']}"
        )

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved yearly protocol metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
