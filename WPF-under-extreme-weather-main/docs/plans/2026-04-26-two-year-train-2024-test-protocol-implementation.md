# Two-Year Train / 2024 Test Protocol Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a new 2h/6p six-station protocol that trains on `2022 + 2023`, tests on `2024`, keeps normal-weather training capped at `30-day-equivalent`, and correctly normalizes train and test data with different station capacities.

**Architecture:** Extend the yearly protocol builder to support separate train and test workbooks plus separate train and test capacities per base station. Keep all normalization, downsampling, phase augmentation, and metadata generation inside the builder, then wire launcher presets so the new protocol can be built, trained, and evaluated without touching legacy root-level assets.

**Tech Stack:** Python, NumPy, SciPy `.mat`, xlsx XML parsing helpers already in `build_three_station_extreme_yearly_protocol.py`, existing training/evaluation scripts, unittest/AST contract tests.

---

## Guardrails

Do not create a standalone `24jilin_58-60.py` preprocessing pipeline.

Do not overwrite old `2022 support / 2023 test` protocol assets.

Do not run full pilot or final training locally. Only run smoke-level builder and contract checks. The user will run long jobs on the 4090 server.

## Task 1: Add Protocol Design Contract Tests

**Files:**
- Create: `WPF-under-extreme-weather-main/tests/test_two_year_2024_protocol_contract_ast.py`
- Modify later: `WPF-under-extreme-weather-main/build_three_station_extreme_yearly_protocol.py`

**Step 1: Write the failing test**

Add AST checks asserting the builder defines configuration fields equivalent to:

```python
train_workbook
test_workbook
train_capacity
test_capacity
```

and that metadata writing includes:

```python
train_years
test_years
normal_sampling_policy
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_two_year_2024_protocol_contract_ast -v
```

Expected: FAIL because the current builder only has one `workbook` and one `capacity`.

**Step 3: Commit**

```bash
git add tests/test_two_year_2024_protocol_contract_ast.py
git commit -m "test: define two-year train and 2024 test protocol contract"
```

Only commit if the user explicitly asks for commits.

## Task 2: Refactor Base Station Configuration

**Files:**
- Modify: `WPF-under-extreme-weather-main/build_three_station_extreme_yearly_protocol.py`
- Test: `WPF-under-extreme-weather-main/tests/test_two_year_2024_protocol_contract_ast.py`

**Step 1: Write the failing test**

Add a direct import test or AST assertion that base station configs contain separate train/test fields, for example:

```python
{
    "station_id": "59",
    "train_workbook": "2223jilin_059_processed_4classes.xlsx",
    "test_workbook": "24jilin_059_processed_4classes.xlsx",
    "train_capacity": 50.0,
    "test_capacity": 100.0,
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_two_year_2024_protocol_contract_ast -v
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Replace the current base configuration entries so they include:

- `train_workbook`
- `test_workbook`
- `train_capacity`
- `test_capacity`

for:

- `58`: `50 -> 50`
- `59`: `50 -> 100`
- `60`: `100 -> 300`

Preserve phase augmentation so `61/62/63` inherit the matching base station's train/test sources and capacities.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_two_year_2024_protocol_contract_ast -v
```

Expected: PASS.

## Task 3: Separate Train/Test Workbook Loading and Normalization

**Files:**
- Modify: `WPF-under-extreme-weather-main/build_three_station_extreme_yearly_protocol.py`
- Test: `WPF-under-extreme-weather-main/tests/test_two_year_2024_protocol_contract_ast.py`

**Step 1: Write the failing test**

Add a contract test asserting the builder no longer uses one shared `workbook_name` plus one shared `capacity` inside `build_yearly_station_asset`.

Instead, it should load:

```python
train_workbook
test_workbook
train_capacity
test_capacity
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_two_year_2024_protocol_contract_ast -v
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Refactor the builder so:

- train-side main/normal/extreme records come from the `2223` workbook
- test-side main/extreme records come from the `24` workbook
- train-side rows are normalized by `train_capacity`
- test-side rows are normalized by `test_capacity`

Do not mutate shared cached workbook objects in-place. Continue cloning records before normalization.

**Step 4: Run smoke verification**

Run a small verification command that prints per-station protocol config and exits without training:

```bash
python build_three_station_extreme_yearly_protocol.py
```

Use a smoke-only output directory if needed. Expected: metadata is generated without exceptions.

## Task 4: Replace Single-Year Normal Sampling With Two-Year Capped Sampling

**Files:**
- Modify: `WPF-under-extreme-weather-main/build_three_station_extreme_yearly_protocol.py`
- Test: `WPF-under-extreme-weather-main/tests/test_two_year_2024_protocol_contract_ast.py`

**Step 1: Write the failing test**

Add a contract test asserting:

- total normal support budget remains `360` points
- the builder exposes a policy equivalent to `two_year_balanced_month_stratified_30d`

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_two_year_2024_protocol_contract_ast -v
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Implement a normal-support sampling helper that:

- takes `2022 normal_weather` plus `2023 normal_weather`
- samples `180` points from `2022`
- samples `180` points from `2023`
- uses month-stratified sampling within each year
- preserves station-specific downsampling offset before clustering

Record the final sampled counts in metadata.

**Step 4: Run verification**

Run a smoke build and inspect metadata:

```bash
python - <<'PY'
import json
data=json.load(open('protocol_data/.../protocol_metadata.json'))
print(data['normal_sampling_policy'])
PY
```

Expected: policy string present and counts sum to `360` per station.

## Task 5: Rebuild Extreme Train/Test Counts For The New Protocol

**Files:**
- Modify: `WPF-under-extreme-weather-main/build_three_station_extreme_yearly_protocol.py`
- Test: `WPF-under-extreme-weather-main/tests/test_two_year_2024_protocol_contract_ast.py`

**Step 1: Write the failing test**

Add a direct smoke check script or contract assertion that metadata still emits:

- `extreme_support_window_counts`
- `extreme_test_window_counts`

but now those counts correspond to:

- train=`2223`
- test=`24`

**Step 2: Run test to verify it fails**

Run the metadata smoke test. Expected: FAIL or mismatch with current single-workbook logic.

**Step 3: Write minimal implementation**

Ensure:

- support windows are computed from the full `2223` extreme sheets after 2h downsampling
- test windows are computed from the full `24` extreme sheets after 2h downsampling

No year split should remain inside the new main protocol for extreme data.

**Step 4: Run verification**

Run the builder and confirm counts roughly match the known expected ranges:

- `high_wind`: `13-18`
- `high_temp`: `6-12`
- `cold_wave`: `6-7`
- `frost`: `32-50`

for train windows across the six protocol stations.

## Task 6: Expand Metadata and Logging

**Files:**
- Modify: `WPF-under-extreme-weather-main/build_three_station_extreme_yearly_protocol.py`
- Modify: `WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py`

**Step 1: Write the failing test**

Add AST checks asserting metadata includes:

- `train_years`
- `test_years`
- `train_workbook`
- `test_workbook`
- `train_capacity`
- `test_capacity`
- `normal_sampling_policy`

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_two_year_2024_protocol_contract_ast -v
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Add those fields to protocol metadata and ensure launcher preview logging prints the active protocol name and data dir.

**Step 4: Run test to verify it passes**

Run the same unittest command. Expected: PASS.

## Task 7: Add New Protocol Presets To The Launcher

**Files:**
- Modify: `WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py`
- Test: `WPF-under-extreme-weather-main/tests/test_2h_6point_launcher_ast.py`

**Step 1: Write the failing test**

Add or extend launcher AST tests so the preset table includes a new explicit protocol name for the two-year train / 2024 test flow, for example:

- `2h6p-2223train-24test-smoke`
- `2h6p-2223train-24test-pilot`

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_2h_6point_launcher_ast -v
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Add presets that set:

- protocol data dir
- protocol name
- six-station mode
- 2h/6p parameters
- new builder mode flags if required

Do not start with final training budgets. Provide:

- `smoke`
- `pilot`

only.

**Step 4: Run test to verify it passes**

Run the same unittest command. Expected: PASS.

## Task 8: Smoke-Build The New Protocol

**Files:**
- Use: `WPF-under-extreme-weather-main/build_three_station_extreme_yearly_protocol.py`
- Use: `WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py`

**Step 1: Run builder smoke**

Run:

```bash
python run_three_station_yearly_protocol.py build --preset 2h6p-2223train-24test-smoke --dry-run
```

Expected: command preview is correct.

**Step 2: Run real builder smoke**

Run the real build for the smoke preset.

Expected:

- `.mat` files generated in an isolated protocol directory
- metadata generated
- no root-level legacy files overwritten

**Step 3: Inspect metadata**

Verify:

- train/test workbook names are correct
- train/test capacities are correct
- normal budget remains capped
- train/test extreme counts are nonzero and reasonable

## Task 9: Rebaseline On The New Protocol

**Files:**
- Use existing training/evaluation stack
- Possibly modify preset wiring only if needed later

**Step 1: Prepare 4090 commands**

After smoke passes, prepare screen-ready commands for:

1. `Local-Meta-FT` on the new protocol
2. best historical federated baseline on the new protocol

Do not run them locally.

**Step 2: Record expected result interpretation**

Document that the first question is:

> Does the best historical federated family improve materially once extreme training uses `22-23` instead of only `2022`?

If yes, keep exploring within the new protocol.

If no, revisit the algorithm family after the new baseline is established.

## Task 10: Commit Documentation And Planning Artifacts

**Files:**
- Create: `WPF-under-extreme-weather-main/docs/plans/2026-04-26-two-year-train-2024-test-protocol-design.md`
- Create: `WPF-under-extreme-weather-main/docs/plans/2026-04-26-two-year-train-2024-test-protocol-implementation.md`

**Step 1: Add the new docs**

```bash
git add docs/plans/2026-04-26-two-year-train-2024-test-protocol-design.md \
        docs/plans/2026-04-26-two-year-train-2024-test-protocol-implementation.md
```

**Step 2: Commit**

```bash
git commit -m "docs: define two-year train and 2024 test protocol"
```

Only commit if the user explicitly asks for commits.
