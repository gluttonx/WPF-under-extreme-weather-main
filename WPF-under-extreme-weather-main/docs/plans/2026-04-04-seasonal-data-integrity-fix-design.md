# Seasonal Data Integrity Fix Design

**Goal:** Rebuild all six seasonal `.mat` assets so their values match the intended single-pass normalization and time slicing from the source `.xlsx` files.

## Problem

The current seasonal asset builder mutates shared `SheetRecord` objects stored in `workbook_cache`. Because the same workbook objects are reused across clients and across train/test merges, `normalize_power(...)` is applied multiple times to the same records. This corrupts the exported `.mat` values, especially for later clients such as `60~63`, and makes meta-training losses print as `0.000000` from the first epoch.

## Root Cause

- `load_xlsx_workbook(...)` builds mutable `SheetRecord` objects.
- `merge_workbooks_by_sheet(...)` reuses those same objects instead of detaching them.
- `serialize_client_assets(...)` calls `normalize_power(...)` in place on the merged train/test sheet lists.
- Repeated client serialization therefore repeatedly divides already-normalized `Power2` values.

## Selected Fix

Use a single root-cause fix: make merged workbook sheets contain detached copies of every `SheetRecord`, including a copied `values` dict. Keep `normalize_power(...)` as-is, but ensure it only ever mutates per-client copies.

## Validation

- Unit test that merged sheet records are detached from the workbook cache.
- Integration test that sequentially serializing all six clients produces a correct-scale `63wf_seasonal_protocol.mat` conventional target distribution.
- Rebuild all seasonal assets.
- Recheck all six clients against fresh single-pass expectations from the `.xlsx` sources.
