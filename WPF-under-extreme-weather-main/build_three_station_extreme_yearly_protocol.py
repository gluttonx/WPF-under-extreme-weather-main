#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build yearly protocol assets from raw station xlsx files.

Current main protocol:
  - train on 2022 + 2023
  - test on 2024
  - keep 2h / 6-point windows
  - cap normal support to 30-day-equivalent
  - cap extreme support to K=6 windows per class (or all-available when cap<=0)
"""
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
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
PROTOCOL_NAME = os.getenv("PROTOCOL_NAME", "three_station_2h_6point_protocol")
SAMPLE_INTERVAL_HOURS = int(os.getenv("SAMPLE_INTERVAL_HOURS", "2"))
DOWNSAMPLE_OFFSET = int(os.getenv("DOWNSAMPLE_OFFSET", "1"))
LEN_REALP = int(os.getenv("LEN_REALP", "6"))
POINTS_PER_DAY = int(os.getenv("POINTS_PER_DAY", str(24 // SAMPLE_INTERVAL_HOURS)))
WINDOW_SPAN_HOURS = SAMPLE_INTERVAL_HOURS * LEN_REALP
OUTPUT_DIR = Path(os.getenv("PROTOCOL_DATA_DIR", str(ROOT / "protocol_data" / "2h_6p")))
METADATA_PATH = Path(os.getenv("PROTOCOL_METADATA_PATH", str(OUTPUT_DIR / "protocol_metadata.json")))
PHASE_AUGMENT_STATIONS = os.getenv("PHASE_AUGMENT_STATIONS", "0") != "0"
PHASE_AUGMENT_STATION_MAP = {
    "58": "61",
    "59": "62",
    "60": "63",
}
HIGH_TEMP_ONLY_SUMMER_PROTOCOL = os.getenv("HIGH_TEMP_ONLY_SUMMER_PROTOCOL", "0") != "0"
HIGH_WIND_SPRING_NOFT_PROTOCOL = os.getenv("HIGH_WIND_SPRING_NOFT_PROTOCOL", "0") != "0"
SUMMER_MONTHS = [6, 7, 8]
APRIL_MONTHS = [4]
MAY_MONTHS = [5]
HIGH_WIND_NORMAL_TRAIN_START = os.getenv("HIGH_WIND_NORMAL_TRAIN_START", "2022-04-01T00:00:00+00:00")
HIGH_WIND_NORMAL_TRAIN_END = os.getenv("HIGH_WIND_NORMAL_TRAIN_END", "2022-04-29T16:00:00+00:00")
HIGH_TEMP_ONLY_STATION_CONVENTIONAL_CLASSES = {
    "58": 7,
    "61": 7,
    "59": 6,
    "62": 6,
    "60": 5,
    "63": 5,
}
HIGH_WIND_SPRING_STATION_CONVENTIONAL_CLASSES = {
    "59": 4,
    "62": 4,
    "60": 4,
    "63": 4,
}
HIGH_WIND_SPRING_BASE_STATION_IDS = {"59", "60"}

SUPPORT_YEAR = 2022  # legacy compatibility token for older tests/docs
TEST_YEAR = 2023  # legacy compatibility token for older tests/docs
EXTREME_SUPPORT_WINDOW_CAP = int(os.getenv("EXTREME_SUPPORT_WINDOW_CAP", "6"))

SELECTED_FEATURES = [
    "wind_speed_100m",
    "wind_direction_100m",
    "temperature_2m",
    "pressure_msl",
    "relative_humidity_2m",
]
TARGET_COL = "Power2"
if HIGH_TEMP_ONLY_SUMMER_PROTOCOL:
    TRAIN_YEARS = [2023]
    TEST_YEARS = [2024]
    NUM_CONVENTIONAL_CLASSES = 6
    NORMAL_SUPPORT_TOTAL_POINTS = 0
    NORMAL_SUPPORT_POINTS_BY_YEAR = {2023: 0}
    NORMAL_SAMPLING_POLICY = "single_year_summer_all_available_normal"
    EXTREME_SUPPORT_SAMPLING_POLICY = "single_year_summer_all_available_high_temp"
    EXTREME_SHEETS = OrderedDict([("extreme_high_temp", 0)])
    EXTREME_CLASS_NAMES = ["high_temp"]
    EXTREME_EVAL_LABELS = ["HighTemperature"]
elif HIGH_WIND_SPRING_NOFT_PROTOCOL:
    TRAIN_YEARS = [2022]
    TEST_YEARS = [2023]
    NUM_CONVENTIONAL_CLASSES = 4
    NORMAL_SUPPORT_TOTAL_POINTS = 0
    NORMAL_SUPPORT_POINTS_BY_YEAR = {2022: 0}
    NORMAL_SAMPLING_POLICY = "single_year_april_cut_normal"
    EXTREME_SUPPORT_SAMPLING_POLICY = "single_year_april_contiguous_high_wind"
    EXTREME_SHEETS = OrderedDict([("extreme_high_wind", 0)])
    EXTREME_CLASS_NAMES = ["high_wind"]
    EXTREME_EVAL_LABELS = ["HighWind"]
else:
    TRAIN_YEARS = [2022, 2023]
    TEST_YEARS = [2024]
    NUM_CONVENTIONAL_CLASSES = 5
    NORMAL_SUPPORT_TOTAL_POINTS = 360
    NORMAL_SUPPORT_POINTS_BY_YEAR = {2022: 180, 2023: 180}
    NORMAL_SAMPLING_POLICY = "two_year_balanced_month_stratified_30d"
    EXTREME_SUPPORT_SAMPLING_POLICY = (
        "two_year_all_available_extreme"
        if EXTREME_SUPPORT_WINDOW_CAP <= 0
        else "two_year_balanced_time_stratified_k6"
    )
    EXTREME_SHEETS = OrderedDict(
        [
            ("extreme_high_wind", 0),
            ("extreme_high_temp", 1),
            ("extreme_cold_wave", 2),
            ("extreme_frost", 3),
        ]
    )
    EXTREME_CLASS_NAMES = ["high_wind", "high_temp", "cold_wave", "frost"]
    EXTREME_EVAL_LABELS = ["HighWind", "HighTemperature", "ColdWave", "Frost"]

EXTREME_SUPPORT_POWER_KEYS = [f"p_extre_class{class_index + 1}" for class_index in range(len(EXTREME_CLASS_NAMES))]
EXTREME_SUPPORT_NWP_KEYS = [f"nwp_extre_class{class_index + 1}_" for class_index in range(len(EXTREME_CLASS_NAMES))]
EXTREME_TEST_POWER_KEYS = [f"p_test_extre_class{class_index + 1}" for class_index in range(len(EXTREME_CLASS_NAMES))]
EXTREME_TEST_NWP_KEYS = [f"nwp_test_extre_class{class_index + 1}_" for class_index in range(len(EXTREME_CLASS_NAMES))]
BASE_YEARLY_PROTOCOL_STATIONS = [
    {
        "station_id": "58",
        "source_station_id": "058",
        "main_sheet": "jilin_058",
        "train_workbook": "2223jilin_058_processed_4classes.xlsx",
        "test_workbook": "24jilin_058_processed_4classes.xlsx",
        "train_capacity": 50.0,
        "test_capacity": 50.0,
    },
    {
        "station_id": "59",
        "source_station_id": "059",
        "main_sheet": "jilin_059",
        "train_workbook": "2223jilin_059_processed_4classes.xlsx",
        "test_workbook": "24jilin_059_processed_4classes.xlsx",
        "train_capacity": 50.0,
        "test_capacity": 100.0,
    },
    {
        "station_id": "60",
        "source_station_id": "060",
        "main_sheet": "jilin_060",
        "train_workbook": "2223jilin_060_processed_4classes.xlsx",
        "test_workbook": "24jilin_060_processed_4classes.xlsx",
        "train_capacity": 100.0,
        "test_capacity": 300.0,
    },
]


def build_protocol_station_configs():
    station_configs = []
    for station_config in BASE_YEARLY_PROTOCOL_STATIONS:
        if (
            HIGH_WIND_SPRING_NOFT_PROTOCOL
            and station_config["station_id"] not in HIGH_WIND_SPRING_BASE_STATION_IDS
        ):
            continue
        base_config = dict(station_config)
        base_config["downsample_offset"] = DOWNSAMPLE_OFFSET
        base_config["phase_role"] = "base"
        base_config["phase_source_station_id"] = station_config["station_id"]
        station_configs.append(base_config)

    if not PHASE_AUGMENT_STATIONS:
        return station_configs

    if SAMPLE_INTERVAL_HOURS < 2 or SAMPLE_INTERVAL_HOURS % 2 != 0:
        raise ValueError("PHASE_AUGMENT_STATIONS requires an even SAMPLE_INTERVAL_HOURS >= 2")

    complementary_offset = int(
        os.getenv(
            "PHASE_AUGMENT_COMPLEMENTARY_OFFSET",
            str((DOWNSAMPLE_OFFSET + SAMPLE_INTERVAL_HOURS // 2) % SAMPLE_INTERVAL_HOURS),
        )
    )
    if complementary_offset == DOWNSAMPLE_OFFSET:
        raise ValueError("phase augmentation requires a complementary offset different from DOWNSAMPLE_OFFSET")

    for station_config in BASE_YEARLY_PROTOCOL_STATIONS:
        if (
            HIGH_WIND_SPRING_NOFT_PROTOCOL
            and station_config["station_id"] not in HIGH_WIND_SPRING_BASE_STATION_IDS
        ):
            continue
        augmented_config = dict(station_config)
        augmented_config["station_id"] = PHASE_AUGMENT_STATION_MAP[station_config["station_id"]]
        augmented_config["downsample_offset"] = complementary_offset
        augmented_config["phase_role"] = "complementary"
        augmented_config["phase_source_station_id"] = station_config["station_id"]
        station_configs.append(augmented_config)

    return station_configs


YEARLY_PROTOCOL_STATIONS = build_protocol_station_configs()


@dataclass
class SheetRecord:
    date: datetime
    values: Dict[str, float]


def clone_sheet_record(record: SheetRecord) -> SheetRecord:
    return SheetRecord(date=record.date, values=dict(record.values))


def parse_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def filter_records_by_datetime_range(records: List[SheetRecord], start_value: str, end_value: str) -> List[SheetRecord]:
    start_dt = parse_iso8601(start_value)
    end_dt = parse_iso8601(end_value)
    return [clone_sheet_record(record) for record in records if start_dt <= record.date <= end_dt]


def filter_high_wind_normal_train_records(records: List[SheetRecord]) -> List[SheetRecord]:
    return filter_records_by_datetime_range(
        filter_records_by_months(
            split_records_by_years(records, TRAIN_YEARS),
            APRIL_MONTHS,
        ),
        HIGH_WIND_NORMAL_TRAIN_START,
        HIGH_WIND_NORMAL_TRAIN_END,
    )


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


def split_records_by_years(records: List[SheetRecord], years: List[int]) -> List[SheetRecord]:
    year_set = {int(year) for year in years}
    return [clone_sheet_record(record) for record in records if record.date.year in year_set]


def filter_records_by_months(records: List[SheetRecord], months: List[int]) -> List[SheetRecord]:
    month_set = {int(month) for month in months}
    return [clone_sheet_record(record) for record in records if record.date.month in month_set]


def resolve_num_conventional_classes(station_id: str) -> int:
    if HIGH_TEMP_ONLY_SUMMER_PROTOCOL:
        return int(HIGH_TEMP_ONLY_STATION_CONVENTIONAL_CLASSES[str(station_id)])
    if HIGH_WIND_SPRING_NOFT_PROTOCOL:
        return int(HIGH_WIND_SPRING_STATION_CONVENTIONAL_CLASSES[str(station_id)])
    return int(NUM_CONVENTIONAL_CLASSES)


def resolve_normal_support_total_points_metadata(conventional_support_hours: int):
    if HIGH_TEMP_ONLY_SUMMER_PROTOCOL or HIGH_WIND_SPRING_NOFT_PROTOCOL:
        return "all"
    return int(NORMAL_SUPPORT_TOTAL_POINTS)


def resolve_station_test_workbook(station_config: Dict[str, object]) -> str:
    if HIGH_WIND_SPRING_NOFT_PROTOCOL:
        return str(station_config["train_workbook"])
    return str(station_config["test_workbook"])


def resolve_station_test_capacity(station_config: Dict[str, object]) -> float:
    if HIGH_WIND_SPRING_NOFT_PROTOCOL:
        return float(station_config["train_capacity"])
    return float(station_config["test_capacity"])


def validate_protocol_config():
    if SAMPLE_INTERVAL_HOURS < 1:
        raise ValueError("SAMPLE_INTERVAL_HOURS must be >= 1")
    if DOWNSAMPLE_OFFSET < 0 or DOWNSAMPLE_OFFSET >= SAMPLE_INTERVAL_HOURS:
        raise ValueError("DOWNSAMPLE_OFFSET must be in [0, SAMPLE_INTERVAL_HOURS)")
    if WINDOW_SPAN_HOURS != 12:
        raise ValueError("protocol must preserve 12h windows")
    if POINTS_PER_DAY * SAMPLE_INTERVAL_HOURS != 24:
        raise ValueError("POINTS_PER_DAY must match SAMPLE_INTERVAL_HOURS")


def downsample_records(records, interval_hours=SAMPLE_INTERVAL_HOURS, offset=DOWNSAMPLE_OFFSET):
    if interval_hours == 1:
        return list(records)
    return [record for index, record in enumerate(records) if index % interval_hours == offset]


def allocate_counts_by_group(group_sizes: Dict[int, int], target_total: int) -> Dict[int, int]:
    if target_total <= 0:
        return {group_key: 0 for group_key in group_sizes}
    total_available = int(sum(group_sizes.values()))
    if total_available <= target_total:
        return {group_key: int(group_sizes[group_key]) for group_key in group_sizes}

    exact = {
        group_key: (float(group_sizes[group_key]) / float(total_available)) * float(target_total)
        for group_key in group_sizes
    }
    allocated = {
        group_key: min(int(group_sizes[group_key]), int(np.floor(exact[group_key])))
        for group_key in group_sizes
    }
    remaining = int(target_total - sum(allocated.values()))
    if remaining <= 0:
        return allocated

    ranked_keys = sorted(
        group_sizes.keys(),
        key=lambda group_key: (exact[group_key] - allocated[group_key], group_sizes[group_key], -group_key),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for group_key in ranked_keys:
            if allocated[group_key] >= int(group_sizes[group_key]):
                continue
            allocated[group_key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return allocated


def select_evenly_spaced_items(items, count: int):
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    boundaries = np.linspace(0, len(items), num=count + 1, dtype=int)
    selected = []
    for index in range(count):
        start = int(boundaries[index])
        end = int(boundaries[index + 1])
        last = max(start, end - 1)
        pick_index = (start + last) // 2
        selected.append(items[pick_index])
    return selected


def sample_month_stratified_records(records: List[SheetRecord], target_count: int) -> List[SheetRecord]:
    if target_count <= 0 or not records:
        return []
    ordered_records = sorted(records, key=lambda record: record.date)
    month_groups = OrderedDict()
    for record in ordered_records:
        month_groups.setdefault(record.date.month, []).append(record)
    quotas = allocate_counts_by_group(
        {int(month): len(month_records) for month, month_records in month_groups.items()},
        target_count,
    )
    sampled = []
    for month, month_records in month_groups.items():
        sampled.extend(select_evenly_spaced_items(month_records, quotas.get(int(month), 0)))
    return sorted(sampled, key=lambda record: record.date)


def sample_two_year_balanced_normal_records(records: List[SheetRecord], station_downsample_offset: int) -> Tuple[List[SheetRecord], Dict[str, int]]:
    sampled_by_year = {}
    for year, target_count in NORMAL_SUPPORT_POINTS_BY_YEAR.items():
        year_records = downsample_records(
            split_records_by_year(records, int(year)),
            offset=station_downsample_offset,
        )
        sampled_by_year[int(year)] = sample_month_stratified_records(year_records, int(target_count))
    combined = []
    for year in TRAIN_YEARS:
        combined.extend(sampled_by_year.get(int(year), []))
    combined = sorted(combined, key=lambda record: record.date)
    sampled_counts = {str(year): len(sampled_by_year.get(int(year), [])) for year in TRAIN_YEARS}
    return combined, sampled_counts


def sample_summer_all_normal_records(records: List[SheetRecord], station_downsample_offset: int) -> Tuple[List[SheetRecord], Dict[str, int]]:
    filtered_records = filter_records_by_months(
        split_records_by_years(records, TRAIN_YEARS),
        SUMMER_MONTHS,
    )
    sampled_records = downsample_records(filtered_records, offset=station_downsample_offset)
    sampled_counts = {
        str(year): len([record for record in sampled_records if record.date.year == int(year)])
        for year in TRAIN_YEARS
    }
    return sampled_records, sampled_counts


def sample_april_all_normal_records(records: List[SheetRecord], station_downsample_offset: int) -> Tuple[List[SheetRecord], Dict[str, int]]:
    filtered_records = (
        filter_high_wind_normal_train_records(records)
        if HIGH_WIND_SPRING_NOFT_PROTOCOL
        else filter_records_by_months(
            split_records_by_years(records, TRAIN_YEARS),
            APRIL_MONTHS,
        )
    )
    sampled_records = downsample_records(filtered_records, offset=station_downsample_offset)
    sampled_counts = {
        str(year): len([record for record in sampled_records if record.date.year == int(year)])
        for year in TRAIN_YEARS
    }
    return sampled_records, sampled_counts


def sample_normal_support_records(records: List[SheetRecord], station_downsample_offset: int) -> Tuple[List[SheetRecord], Dict[str, int]]:
    if HIGH_TEMP_ONLY_SUMMER_PROTOCOL:
        return sample_summer_all_normal_records(records, station_downsample_offset)
    if HIGH_WIND_SPRING_NOFT_PROTOCOL:
        return sample_april_all_normal_records(records, station_downsample_offset)
    return sample_two_year_balanced_normal_records(records, station_downsample_offset)


def build_complete_windows(records: List[SheetRecord]) -> List[List[SheetRecord]]:
    window_count = len(records) // LEN_REALP
    return [
        records[window_index * LEN_REALP:(window_index + 1) * LEN_REALP]
        for window_index in range(window_count)
    ]


def build_contiguous_complete_windows(records: List[SheetRecord]) -> List[List[SheetRecord]]:
    if not records:
        return []
    ordered_records = sorted(records, key=lambda record: record.date)
    expected_step = timedelta(hours=SAMPLE_INTERVAL_HOURS)
    windows = []
    segment = [ordered_records[0]]
    for previous_record, current_record in zip(ordered_records, ordered_records[1:]):
        if current_record.date - previous_record.date == expected_step:
            segment.append(current_record)
        else:
            windows.extend(build_complete_windows(segment))
            segment = [current_record]
    windows.extend(build_complete_windows(segment))
    return windows


def flatten_windows(windows: List[List[SheetRecord]]) -> List[SheetRecord]:
    flattened = []
    for window_records in windows:
        flattened.extend(window_records)
    return flattened


def sample_extreme_support_records(records: List[SheetRecord], station_downsample_offset: int) -> List[SheetRecord]:
    if HIGH_TEMP_ONLY_SUMMER_PROTOCOL:
        filtered_records = filter_records_by_months(
            split_records_by_years(records, TRAIN_YEARS),
            SUMMER_MONTHS,
        )
        filtered_records = downsample_records(filtered_records, offset=station_downsample_offset)
        return flatten_windows(build_complete_windows(filtered_records))

    if HIGH_WIND_SPRING_NOFT_PROTOCOL:
        filtered_records = filter_records_by_months(
            split_records_by_years(records, TRAIN_YEARS),
            APRIL_MONTHS,
        )
        filtered_records = downsample_records(filtered_records, offset=station_downsample_offset)
        return flatten_windows(build_contiguous_complete_windows(filtered_records))

    windows_by_year = {}
    for year in TRAIN_YEARS:
        year_records = downsample_records(
            split_records_by_year(records, int(year)),
            offset=station_downsample_offset,
        )
        windows_by_year[int(year)] = build_complete_windows(year_records)

    if EXTREME_SUPPORT_WINDOW_CAP <= 0:
        selected_windows = []
        for year in TRAIN_YEARS:
            year = int(year)
            selected_windows.extend(windows_by_year[year])
        selected_windows = sorted(selected_windows, key=lambda window_records: window_records[0].date)
        return flatten_windows(selected_windows)

    preferred_per_year = EXTREME_SUPPORT_WINDOW_CAP // max(1, len(TRAIN_YEARS))
    selected_counts = {
        int(year): min(preferred_per_year, len(windows_by_year[int(year)]))
        for year in TRAIN_YEARS
    }
    selected_total = sum(selected_counts.values())
    remaining = max(0, EXTREME_SUPPORT_WINDOW_CAP - selected_total)
    ranked_years = sorted(
        TRAIN_YEARS,
        key=lambda year: len(windows_by_year[int(year)]) - selected_counts[int(year)],
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for year in ranked_years:
            year = int(year)
            available_extra = len(windows_by_year[year]) - selected_counts[year]
            if available_extra <= 0:
                continue
            selected_counts[year] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break

    selected_windows = []
    for year in TRAIN_YEARS:
        year = int(year)
        year_windows = windows_by_year[year]
        selected_windows.extend(select_evenly_spaced_items(year_windows, selected_counts[year]))
    selected_windows = sorted(selected_windows, key=lambda window_records: window_records[0].date)
    return flatten_windows(selected_windows)


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
    train_workbook = station_config["train_workbook"]
    test_workbook = resolve_station_test_workbook(station_config)
    train_capacity = station_config["train_capacity"]
    test_capacity = resolve_station_test_capacity(station_config)
    station_downsample_offset = int(station_config.get("downsample_offset", DOWNSAMPLE_OFFSET))
    train_workbook_copy = {
        sheet_name: [clone_sheet_record(record) for record in records]
        for sheet_name, records in workbook_cache[train_workbook].items()
    }
    test_workbook_copy = {
        sheet_name: [clone_sheet_record(record) for record in records]
        for sheet_name, records in workbook_cache[test_workbook].items()
    }
    for records in train_workbook_copy.values():
        normalize_power(records, train_capacity)
    for records in test_workbook_copy.values():
        normalize_power(records, test_capacity)

    train_normal_records = train_workbook_copy["normal_weather"]
    if HIGH_TEMP_ONLY_SUMMER_PROTOCOL:
        train_main_records = train_normal_records
        test_main_records = test_workbook_copy["extreme_high_temp"]
    elif HIGH_WIND_SPRING_NOFT_PROTOCOL:
        train_main_records = train_normal_records
        test_main_records = test_workbook_copy["extreme_high_wind"]
    else:
        train_main_records = train_workbook_copy[station_config["main_sheet"]]
        test_main_records = test_workbook_copy[station_config["main_sheet"]]

    if HIGH_TEMP_ONLY_SUMMER_PROTOCOL:
        yearly_main_support = downsample_records(
            filter_records_by_months(split_records_by_years(train_main_records, TRAIN_YEARS), SUMMER_MONTHS),
            offset=station_downsample_offset,
        )
        yearly_main_test = downsample_records(
            filter_records_by_months(split_records_by_years(test_main_records, TEST_YEARS), SUMMER_MONTHS),
            offset=station_downsample_offset,
        )
    elif HIGH_WIND_SPRING_NOFT_PROTOCOL:
        yearly_main_support = downsample_records(
            filter_high_wind_normal_train_records(train_main_records),
            offset=station_downsample_offset,
        )
        yearly_main_test = flatten_windows(
            build_contiguous_complete_windows(
                downsample_records(
                    filter_records_by_months(split_records_by_years(test_main_records, TEST_YEARS), MAY_MONTHS),
                    offset=station_downsample_offset,
                )
            )
        )
    else:
        yearly_main_support = downsample_records(
            split_records_by_years(train_main_records, TRAIN_YEARS),
            offset=station_downsample_offset,
        )
        yearly_main_test = downsample_records(
            split_records_by_years(test_main_records, TEST_YEARS),
            offset=station_downsample_offset,
        )
    yearly_normal_support, normal_sampled_counts_by_year = sample_normal_support_records(
        train_normal_records,
        station_downsample_offset=station_downsample_offset,
    )

    normal_power, normal_nwp = records_to_arrays(yearly_normal_support)
    standardized_normal = standardize_features(normal_nwp)
    station_k = resolve_num_conventional_classes(station_config["station_id"])
    labels = fit_kmeans_with_labels(standardized_normal, station_k)
    p_conven_class, nwp_conven_class = build_cluster_objects(
        yearly_normal_support,
        labels,
        station_k,
    )

    train_power, train_nwp = records_to_arrays(yearly_main_support)
    test_power, test_nwp = records_to_arrays(yearly_main_test)

    mat_dict = {
        "p_1h": train_power.reshape(-1, 1),
        "nwp_1h": train_nwp,
        f"p_{SAMPLE_INTERVAL_HOURS}h": train_power.reshape(-1, 1),
        f"nwp_{SAMPLE_INTERVAL_HOURS}h": train_nwp,
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
        support_records = sample_extreme_support_records(
            train_workbook_copy[sheet_name],
            station_downsample_offset=station_downsample_offset,
        )
        if HIGH_WIND_SPRING_NOFT_PROTOCOL:
            test_records = flatten_windows(
                build_contiguous_complete_windows(
                    downsample_records(
                        filter_records_by_months(split_records_by_years(test_workbook_copy[sheet_name], TEST_YEARS), MAY_MONTHS),
                        offset=station_downsample_offset,
                    )
                )
            )
        else:
            test_records = downsample_records(
                split_records_by_years(test_workbook_copy[sheet_name], TEST_YEARS),
                offset=station_downsample_offset,
            )
        support_power, support_nwp = build_extreme_objects(support_records)
        test_power_class, test_nwp_class = build_extreme_objects(test_records)
        mat_dict[EXTREME_SUPPORT_POWER_KEYS[class_index]] = support_power
        mat_dict[EXTREME_SUPPORT_NWP_KEYS[class_index]] = support_nwp
        mat_dict[EXTREME_TEST_POWER_KEYS[class_index]] = test_power_class
        mat_dict[EXTREME_TEST_NWP_KEYS[class_index]] = test_nwp_class
        extreme_support_window_counts[EXTREME_CLASS_NAMES[class_index]] = int(support_power.shape[0] // LEN_REALP)
        extreme_test_window_counts[EXTREME_CLASS_NAMES[class_index]] = int(test_power_class.shape[0] // LEN_REALP)

    asset_path = OUTPUT_DIR / f"{station_config['station_id']}wf_4_train.mat"
    savemat(asset_path, mat_dict)

    return {
        "station_id": station_config["station_id"],
        "source_station_id": station_config["source_station_id"],
        "main_sheet": station_config["main_sheet"],
        "train_workbook": train_workbook,
        "test_workbook": test_workbook,
        "train_capacity": float(train_capacity),
        "test_capacity": float(test_capacity),
        "asset_path": asset_path.name,
        "train_years": list(TRAIN_YEARS),
        "test_years": list(TEST_YEARS),
        "main_support_hours": int(train_power.shape[0]),
        "main_test_hours": int(test_power.shape[0]),
        "conventional_support_hours": int(normal_power.shape[0]),
        "normal_support_total_points": resolve_normal_support_total_points_metadata(int(normal_power.shape[0])),
        "normal_support_points_by_year": normal_sampled_counts_by_year,
        "normal_sampling_policy": NORMAL_SAMPLING_POLICY,
        "normal_train_start": HIGH_WIND_NORMAL_TRAIN_START if HIGH_WIND_SPRING_NOFT_PROTOCOL else None,
        "normal_train_end": HIGH_WIND_NORMAL_TRAIN_END if HIGH_WIND_SPRING_NOFT_PROTOCOL else None,
        "extreme_support_window_counts": extreme_support_window_counts,
        "extreme_test_window_counts": extreme_test_window_counts,
        "extreme_support_window_cap": "all" if EXTREME_SUPPORT_WINDOW_CAP <= 0 else int(EXTREME_SUPPORT_WINDOW_CAP),
        "extreme_support_sampling_policy": EXTREME_SUPPORT_SAMPLING_POLICY,
        "num_conventional_classes": station_k,
        "protocol_name": PROTOCOL_NAME,
        "sample_interval_hours": SAMPLE_INTERVAL_HOURS,
        "downsample_offset": station_downsample_offset,
        "len_realp": LEN_REALP,
        "points_per_day": POINTS_PER_DAY,
        "window_span_hours": WINDOW_SPAN_HOURS,
        "phase_augmentation_enabled": PHASE_AUGMENT_STATIONS,
        "phase_role": station_config.get("phase_role", "base"),
        "phase_source_station_id": station_config.get("phase_source_station_id", station_config["station_id"]),
    }


def main():
    validate_protocol_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    workbook_cache = {}
    for station_config in YEARLY_PROTOCOL_STATIONS:
        for workbook_name in [station_config["train_workbook"], resolve_station_test_workbook(station_config)]:
            if workbook_name not in workbook_cache:
                workbook_cache[workbook_name] = load_xlsx_workbook(ROOT / workbook_name)

    metadata = {
        "protocol_name": PROTOCOL_NAME,
        "train_years": list(TRAIN_YEARS),
        "test_years": list(TEST_YEARS),
        "sample_interval_hours": SAMPLE_INTERVAL_HOURS,
        "downsample_offset": DOWNSAMPLE_OFFSET,
        "len_realp": LEN_REALP,
        "points_per_day": POINTS_PER_DAY,
        "window_span_hours": WINDOW_SPAN_HOURS,
        "protocol_data_dir": str(OUTPUT_DIR),
        "protocol_metadata_path": str(METADATA_PATH),
        "phase_augmentation_enabled": PHASE_AUGMENT_STATIONS,
        "phase_augment_station_map": PHASE_AUGMENT_STATION_MAP if PHASE_AUGMENT_STATIONS else {},
        "high_wind_active_base_station_ids": (
            sorted(HIGH_WIND_SPRING_BASE_STATION_IDS) if HIGH_WIND_SPRING_NOFT_PROTOCOL else []
        ),
        "normal_support_total_points": (
            "all" if (HIGH_TEMP_ONLY_SUMMER_PROTOCOL or HIGH_WIND_SPRING_NOFT_PROTOCOL)
            else NORMAL_SUPPORT_TOTAL_POINTS
        ),
        "normal_support_points_by_year": {str(year): count for year, count in NORMAL_SUPPORT_POINTS_BY_YEAR.items()},
        "normal_sampling_policy": NORMAL_SAMPLING_POLICY,
        "normal_train_start": HIGH_WIND_NORMAL_TRAIN_START if HIGH_WIND_SPRING_NOFT_PROTOCOL else None,
        "normal_train_end": HIGH_WIND_NORMAL_TRAIN_END if HIGH_WIND_SPRING_NOFT_PROTOCOL else None,
        "extreme_support_window_cap": "all" if EXTREME_SUPPORT_WINDOW_CAP <= 0 else int(EXTREME_SUPPORT_WINDOW_CAP),
        "extreme_support_sampling_policy": EXTREME_SUPPORT_SAMPLING_POLICY,
        "num_conventional_classes": NUM_CONVENTIONAL_CLASSES,
        "conventional_classes_by_station": (
            HIGH_TEMP_ONLY_STATION_CONVENTIONAL_CLASSES if HIGH_TEMP_ONLY_SUMMER_PROTOCOL
            else HIGH_WIND_SPRING_STATION_CONVENTIONAL_CLASSES if HIGH_WIND_SPRING_NOFT_PROTOCOL
            else {}
        ),
        "extreme_class_names": EXTREME_CLASS_NAMES,
        "extreme_eval_labels": EXTREME_EVAL_LABELS,
        "num_extreme_classes": len(EXTREME_CLASS_NAMES),
        "stations": [],
    }

    for station_config in YEARLY_PROTOCOL_STATIONS:
        station_metadata = build_yearly_station_asset(station_config, workbook_cache)
        metadata["stations"].append(station_metadata)
        print(
            f"station {station_metadata['station_id']}: "
            f"offset={station_metadata['downsample_offset']}, "
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
