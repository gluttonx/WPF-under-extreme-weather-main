import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as scio


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "build_six_client_seasonal_protocol.py"
WORKBOOK_CAPACITY_MAP = {
    "2223jilin_058_processed_4classes.xlsx": 50.0,
    "2223jilin_059_processed_4classes.xlsx": 50.0,
    "2223jilin_060_processed_4classes.xlsx": 100.0,
    "24jilin_058_processed_4classes.xlsx": 50.0,
    "24jilin_059_processed_4classes.xlsx": 100.0,
    "24jilin_060_processed_4classes.xlsx": 300.0,
}


def load_builder_module():
    spec = importlib.util.spec_from_file_location("seasonal_builder_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flatten_object_matrix(obj_mat):
    values = []
    for idx in range(obj_mat.shape[1]):
        arr = np.array(obj_mat[0, idx]).reshape(-1)
        if arr.size:
            values.append(arr.astype(float))
    return np.concatenate(values) if values else np.array([], dtype=float)


class SeasonalDataIntegrityTest(unittest.TestCase):
    def test_merge_workbooks_by_sheet_returns_detached_records(self):
        builder = load_builder_module()
        original = builder.SheetRecord(
            date=builder.parse_iso8601("2022-03-01T00:00:00+08:00"),
            values={feature: float(index + 1) for index, feature in enumerate(builder.SELECTED_FEATURES + [builder.TARGET_COL])},
        )
        workbook = {"jilin_058": [original]}

        merged = builder.merge_workbooks_by_sheet([workbook])
        merged["jilin_058"][0].values[builder.TARGET_COL] = 999.0

        self.assertNotEqual(original.values[builder.TARGET_COL], merged["jilin_058"][0].values[builder.TARGET_COL])

    def test_client63_conventional_scale_matches_per_workbook_capacity(self):
        builder = load_builder_module()
        client_config = [item for item in builder.SEASONAL_PROTOCOL_CLIENTS if item["client_id"] == "63"][0]

        fresh_workbook = builder.load_xlsx_workbook(ROOT / client_config["train_workbooks"][0])
        normal_records = builder.merge_workbooks_by_sheet([fresh_workbook])["normal_weather"]
        train_start = builder.parse_iso8601(client_config["train_start"])
        train_end = builder.datetime.fromisoformat(f"{train_start.year}-12-01T00:00:00+08:00")
        first_month_normal = builder.slice_records(normal_records, train_start, train_end)
        workbook_name = client_config["train_workbooks"][0]
        cap = WORKBOOK_CAPACITY_MAP[workbook_name]
        for record in first_month_normal:
            record.values[builder.TARGET_COL] = float(record.values[builder.TARGET_COL]) / cap
        expected_mean = float(np.mean([record.values[builder.TARGET_COL] for record in first_month_normal]))

        workbook_names = sorted(
            {
                name
                for item in builder.SEASONAL_PROTOCOL_CLIENTS
                for name in item["train_workbooks"] + item["test_workbooks"]
            }
        )
        workbook_cache = {name: builder.load_xlsx_workbook(ROOT / name) for name in workbook_names}

        with tempfile.TemporaryDirectory() as temp_dir:
            builder.OUTPUT_DIR = Path(temp_dir)
            builder.OUTPUT_DIR.mkdir(exist_ok=True)
            for item in builder.SEASONAL_PROTOCOL_CLIENTS:
                builder.serialize_client_assets(item, workbook_cache)

            client63_mat = scio.loadmat(builder.OUTPUT_DIR / "63wf_seasonal_protocol.mat")
            actual_mean = float(flatten_object_matrix(client63_mat["p_conven_class"]).mean())

        self.assertAlmostEqual(actual_mean, expected_mean, places=6)

    def test_client63_test_scale_matches_mixed_workbook_capacities(self):
        builder = load_builder_module()
        client_config = [item for item in builder.SEASONAL_PROTOCOL_CLIENTS if item["client_id"] == "63"][0]

        expected_values = []
        test_start = builder.parse_iso8601(client_config["test_start"])
        test_end = builder.parse_iso8601(client_config["test_end"])
        main_sheet = builder.MAIN_SHEET_BY_SOURCE[client_config["source_station_id"]]
        for workbook_name in client_config["test_workbooks"]:
            workbook = builder.load_xlsx_workbook(ROOT / workbook_name)
            records = builder.slice_records(workbook[main_sheet], test_start, test_end)
            cap = WORKBOOK_CAPACITY_MAP[workbook_name]
            expected_values.extend(float(record.values[builder.TARGET_COL]) / cap for record in records)
        expected_mean = float(np.mean(expected_values))

        workbook_names = sorted(
            {
                name
                for item in builder.SEASONAL_PROTOCOL_CLIENTS
                for name in item["train_workbooks"] + item["test_workbooks"]
            }
        )
        workbook_cache = {name: builder.load_xlsx_workbook(ROOT / name) for name in workbook_names}

        with tempfile.TemporaryDirectory() as temp_dir:
            builder.OUTPUT_DIR = Path(temp_dir)
            builder.OUTPUT_DIR.mkdir(exist_ok=True)
            for item in builder.SEASONAL_PROTOCOL_CLIENTS:
                builder.serialize_client_assets(item, workbook_cache)

            client63_mat = scio.loadmat(builder.OUTPUT_DIR / "63wf_seasonal_protocol.mat")
            actual_mean = float(np.array(client63_mat["p_test"]).reshape(-1).astype(float).mean())

        self.assertAlmostEqual(actual_mean, expected_mean, places=6)


if __name__ == "__main__":
    unittest.main()
