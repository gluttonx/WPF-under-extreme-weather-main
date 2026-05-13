# 2h/6-Point Full Protocol Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a full-flow 2-hour / 6-point data protocol that rebuilds all station data from source xlsx files, trains all stages under the same lower-resolution setting, and evaluates only within the matching protocol.

**Architecture:** The protocol builder becomes the source of truth for sampling interval, downsample phase, window length, and output data directory. Training and evaluation scripts read the same protocol metadata and refuse to silently fall back to stale root-level 1h mats when a protocol data directory is configured. The model architecture stays unchanged; only the data window length and day-point assumptions become protocol-driven.

**Tech Stack:** Python, NumPy, SciPy `.mat`, pandas-compatible xlsx parsing helpers already in `build_three_station_extreme_yearly_protocol.py`, PyTorch, pytest AST/behavior tests.

---

## Guardrails

Do not run full training from this plan. Codex may run only smoke-level commands and lightweight tests. Full pilot/final runs should be handed to the user as commands for the 4090 machine.

Do not overwrite root-level `58wf_4_train.mat`, `59wf_4_train.mat`, or `60wf_4_train.mat` during implementation. All 2h/6-point outputs must go under an isolated protocol directory such as `protocol_data/2h_6p/`.

Do not reuse 1h/12 checkpoints for final 2h/6 results. State dicts may load technically, but the protocol semantics differ.

## Task 1: Add Builder Protocol Tests

**Files:**
- Create: `WPF-under-extreme-weather-main/tests/test_2h_6point_protocol_builder_ast.py`
- Modify later: `WPF-under-extreme-weather-main/build_three_station_extreme_yearly_protocol.py`

**Step 1: Write AST tests for protocol constants and isolated output**

Create a test that parses `build_three_station_extreme_yearly_protocol.py` and asserts it defines or reads:

```python
SAMPLE_INTERVAL_HOURS
DOWNSAMPLE_OFFSET
LEN_REALP
POINTS_PER_DAY
PROTOCOL_NAME
OUTPUT_DIR
```

Also assert the script no longer hardcodes only `OUTPUT_DIR = ROOT / "three_station_yearly_protocol_data"` without allowing a protocol-specific output directory.

**Step 2: Write behavioral helper tests**

Add a small direct import test for a pure helper named `downsample_records`. The test should construct six fake records with increasing `date` values and assert:

```python
downsample_records(records, interval_hours=2, offset=1)
```

returns records at indexes `[1, 3, 5]`.

Also assert:

```python
downsample_records(records, interval_hours=1, offset=0)
```

returns all records.

**Step 3: Run tests to verify failure**

Run:

```bash
python -m pytest WPF-under-extreme-weather-main/tests/test_2h_6point_protocol_builder_ast.py -q
```

Expected before implementation: FAIL because `downsample_records` and protocol constants are missing.

If pytest is unavailable in the local environment, record that and continue with AST inspection only. Do not install dependencies unless the user asks.

**Step 4: Commit point after implementation**

After implementation and passing tests:

```bash
git add WPF-under-extreme-weather-main/tests/test_2h_6point_protocol_builder_ast.py WPF-under-extreme-weather-main/build_three_station_extreme_yearly_protocol.py
git commit -m "test: define 2h 6-point protocol builder contract"
```

Only commit if the user explicitly asks for commits.

## Task 2: Parameterize The Yearly Protocol Builder

**Files:**
- Modify: `WPF-under-extreme-weather-main/build_three_station_extreme_yearly_protocol.py`

**Step 1: Add protocol configuration**

Add env-driven defaults near existing constants:

```python
PROTOCOL_NAME = os.getenv("PROTOCOL_NAME", "three_station_2h_6point_protocol")
SAMPLE_INTERVAL_HOURS = int(os.getenv("SAMPLE_INTERVAL_HOURS", "2"))
DOWNSAMPLE_OFFSET = int(os.getenv("DOWNSAMPLE_OFFSET", "1"))
LEN_REALP = int(os.getenv("LEN_REALP", "6"))
POINTS_PER_DAY = int(os.getenv("POINTS_PER_DAY", str(24 // SAMPLE_INTERVAL_HOURS)))
WINDOW_SPAN_HOURS = SAMPLE_INTERVAL_HOURS * LEN_REALP
OUTPUT_DIR = Path(os.getenv("PROTOCOL_DATA_DIR", ROOT / "protocol_data" / "2h_6p"))
METADATA_PATH = OUTPUT_DIR / "protocol_metadata.json"
```

Keep `LEN_REALP=12`, `POINTS_PER_DAY=24`, and the old output directory available through env for legacy reproduction, but default this new path to 2h/6.

**Step 2: Add validation**

Add a function:

```python
def validate_protocol_config():
    if SAMPLE_INTERVAL_HOURS < 1:
        raise ValueError("SAMPLE_INTERVAL_HOURS must be >= 1")
    if DOWNSAMPLE_OFFSET < 0 or DOWNSAMPLE_OFFSET >= SAMPLE_INTERVAL_HOURS:
        raise ValueError("DOWNSAMPLE_OFFSET must be in [0, SAMPLE_INTERVAL_HOURS)")
    if WINDOW_SPAN_HOURS != 12:
        raise ValueError("2h/6 protocol must preserve 12h windows")
    if POINTS_PER_DAY * SAMPLE_INTERVAL_HOURS != 24:
        raise ValueError("POINTS_PER_DAY must match SAMPLE_INTERVAL_HOURS")
```

Call it at the beginning of `main()`.

**Step 3: Add downsampling helper**

Implement:

```python
def downsample_records(records, interval_hours=SAMPLE_INTERVAL_HOURS, offset=DOWNSAMPLE_OFFSET):
    if interval_hours == 1:
        return list(records)
    return [record for index, record in enumerate(records) if index % interval_hours == offset]
```

This deliberately uses row index after year split, not clock-hour parsing. It preserves the fixed phase in each sheet/year split and avoids the 2022 length determining the 2023 phase.

**Step 4: Apply downsampling after year splits**

In `build_yearly_station_asset`, change:

```python
yearly_main_support = split_records_by_year(main_sheet_records, SUPPORT_YEAR)
yearly_main_test = split_records_by_year(main_sheet_records, TEST_YEAR)
yearly_normal_support = split_records_by_year(normal_records, SUPPORT_YEAR)
```

to:

```python
yearly_main_support = downsample_records(split_records_by_year(main_sheet_records, SUPPORT_YEAR))
yearly_main_test = downsample_records(split_records_by_year(main_sheet_records, TEST_YEAR))
yearly_normal_support = downsample_records(split_records_by_year(normal_records, SUPPORT_YEAR))
```

Apply the same pattern to every extreme support/test sheet before `build_extreme_objects`.

**Step 5: Write compatibility and explicit keys**

Continue writing old compatibility keys:

```python
"p_1h": train_power.reshape(-1, 1)
"nwp_1h": train_nwp
```

Also add explicit aliases:

```python
"p_2h": train_power.reshape(-1, 1)
"nwp_2h": train_nwp
```

Only write `p_2h` and `nwp_2h` when `SAMPLE_INTERVAL_HOURS == 2`, or write generic aliases:

```python
f"p_{SAMPLE_INTERVAL_HOURS}h"
f"nwp_{SAMPLE_INTERVAL_HOURS}h"
```

**Step 6: Expand metadata**

Add these fields at top level:

```python
"protocol_name": PROTOCOL_NAME,
"sample_interval_hours": SAMPLE_INTERVAL_HOURS,
"downsample_offset": DOWNSAMPLE_OFFSET,
"len_realp": LEN_REALP,
"points_per_day": POINTS_PER_DAY,
"window_span_hours": WINDOW_SPAN_HOURS,
"protocol_data_dir": str(OUTPUT_DIR),
```

Add the same fields per station where useful, and keep `extreme_support_window_counts` and `extreme_test_window_counts`.

**Step 7: Run builder contract tests**

Run:

```bash
python -m pytest WPF-under-extreme-weather-main/tests/test_2h_6point_protocol_builder_ast.py -q
```

Expected after implementation: PASS.

## Task 3: Add Training Protocol Loader Tests

**Files:**
- Create: `WPF-under-extreme-weather-main/tests/test_training_protocol_config_ast.py`
- Modify later: `WPF-under-extreme-weather-main/DemoModelTraining.py`

**Step 1: Write AST test for hardcoded assumptions**

Parse `DemoModelTraining.py` and assert:

- `len_realp=12` is not assigned as a fixed literal without env/metadata override.
- `d=24` is not assigned as a fixed literal without env/metadata override.
- `PROTOCOL_DATA_DIR` is read from `os.getenv`.
- `LEN_REALP` and `POINTS_PER_DAY` are read from env or metadata.

**Step 2: Write AST test for `.mat` loading**

Assert the script uses a helper such as:

```python
resolve_station_mat_path(station_id)
```

and does not rely only on:

```python
dataFile = f'{station_id}wf_4_train'
scio.loadmat(dataFile)
```

**Step 3: Run tests to verify failure**

Run:

```bash
python -m pytest WPF-under-extreme-weather-main/tests/test_training_protocol_config_ast.py -q
```

Expected before implementation: FAIL.

## Task 4: Parameterize `DemoModelTraining.py`

**Files:**
- Modify: `WPF-under-extreme-weather-main/DemoModelTraining.py`

**Step 1: Add protocol env vars near runtime config**

Add:

```python
PROTOCOL_DATA_DIR = os.getenv("PROTOCOL_DATA_DIR", "")
PROTOCOL_METADATA_PATH = os.getenv("PROTOCOL_METADATA_PATH", "")
LEN_REALP = int(os.getenv("LEN_REALP", "12"))
POINTS_PER_DAY = int(os.getenv("POINTS_PER_DAY", "24"))
SAMPLE_INTERVAL_HOURS = int(os.getenv("SAMPLE_INTERVAL_HOURS", str(24 // POINTS_PER_DAY)))
DOWNSAMPLE_OFFSET = int(os.getenv("DOWNSAMPLE_OFFSET", "0"))
```

If metadata loading is straightforward, load `protocol_metadata.json` and use it to set defaults when env vars are absent. Keep env vars as the final override.

**Step 2: Add path resolver**

Add:

```python
def resolve_station_mat_path(station_id):
    filename = f"{station_id}wf_4_train.mat"
    if PROTOCOL_DATA_DIR:
        candidate = os.path.join(PROTOCOL_DATA_DIR, filename)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Protocol mat missing: {candidate}")
        return candidate
    return filename
```

Use this wherever station mats are loaded.

**Step 3: Replace hardcoded data dimensions**

Change:

```python
len_realp=12
d=24
```

to:

```python
len_realp = LEN_REALP
d = POINTS_PER_DAY
```

Keep:

```python
dem_realp = 1
Cap = 50
m = 365
ooo = 365
```

unless the later 2024 protocol requires different split lengths.

**Step 4: Add protocol startup logging**

Print one block before data loading:

```text
Protocol:
  data_dir: ...
  metadata: ...
  sample_interval_hours: 2
  downsample_offset: 1
  len_realp: 6
  points_per_day: 12
  window_span_hours: 12
```

**Step 5: Save protocol info into convergence report**

Add to `run_config`:

```python
"protocol_data_dir": PROTOCOL_DATA_DIR,
"protocol_metadata_path": PROTOCOL_METADATA_PATH,
"sample_interval_hours": SAMPLE_INTERVAL_HOURS,
"downsample_offset": DOWNSAMPLE_OFFSET,
"len_realp": LEN_REALP,
"points_per_day": POINTS_PER_DAY,
```

**Step 6: Run tests**

Run:

```bash
python -m pytest WPF-under-extreme-weather-main/tests/test_training_protocol_config_ast.py -q
```

Expected after implementation: PASS.

## Task 5: Add Evaluation Protocol Tests

**Files:**
- Create: `WPF-under-extreme-weather-main/tests/test_eval_2h_6point_protocol_ast.py`
- Modify later: `WPF-under-extreme-weather-main/generate_multi_station_results.py`

**Step 1: Write AST tests**

Assert `generate_multi_station_results.py`:

- Reads `PROTOCOL_DATA_DIR`.
- Reads `LEN_REALP` or metadata-derived `len_realp`.
- Does not hardcode `len_realp = 12` as the only path.
- Adds protocol columns to CSV rows: `Protocol`, `Sample_Interval_Hours`, `Window_Points`, `Window_Span_Hours`.
- Uses a resolver for station mat paths.

**Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest WPF-under-extreme-weather-main/tests/test_eval_2h_6point_protocol_ast.py -q
```

Expected before implementation: FAIL.

## Task 6: Parameterize `generate_multi_station_results.py`

**Files:**
- Modify: `WPF-under-extreme-weather-main/generate_multi_station_results.py`

**Step 1: Add protocol config**

Add the same env/metadata config pattern as training:

```python
PROTOCOL_DATA_DIR = os.getenv("PROTOCOL_DATA_DIR", "")
PROTOCOL_METADATA_PATH = os.getenv("PROTOCOL_METADATA_PATH", "")
PROTOCOL_NAME = os.getenv("PROTOCOL_NAME", "legacy_1h_12point")
LEN_REALP = int(os.getenv("LEN_REALP", "12"))
POINTS_PER_DAY = int(os.getenv("POINTS_PER_DAY", "24"))
SAMPLE_INTERVAL_HOURS = int(os.getenv("SAMPLE_INTERVAL_HOURS", str(24 // POINTS_PER_DAY)))
WINDOW_SPAN_HOURS = SAMPLE_INTERVAL_HOURS * LEN_REALP
```

Use metadata defaults when available.

**Step 2: Add resolver**

Add:

```python
def resolve_station_mat_path(station_id):
    filename = f"{station_id}wf_4_train.mat"
    if PROTOCOL_DATA_DIR:
        candidate = os.path.join(PROTOCOL_DATA_DIR, filename)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Protocol mat missing: {candidate}")
        return candidate
    return filename
```

Use it in the station loop.

**Step 3: Replace fixed `len_realp`**

Change:

```python
len_realp = 12
```

to:

```python
len_realp = LEN_REALP
```

**Step 4: Add protocol columns to every row**

In `all_results.append`, add:

```python
"Protocol": PROTOCOL_NAME,
"Sample_Interval_Hours": SAMPLE_INTERVAL_HOURS,
"Window_Points": LEN_REALP,
"Window_Span_Hours": WINDOW_SPAN_HOURS,
```

**Step 5: Run tests**

Run:

```bash
python -m pytest WPF-under-extreme-weather-main/tests/test_eval_2h_6point_protocol_ast.py -q
```

Expected after implementation: PASS.

## Task 7: Add Launcher Preset Tests

**Files:**
- Create: `WPF-under-extreme-weather-main/tests/test_2h_6point_launcher_ast.py`
- Modify later: `WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py`

**Step 1: Write tests**

Assert the launcher contains:

- A `2h-6p` protocol preset or equivalent flag.
- Env preview keys for `PROTOCOL_DATA_DIR`, `PROTOCOL_METADATA_PATH`, `SAMPLE_INTERVAL_HOURS`, `DOWNSAMPLE_OFFSET`, `LEN_REALP`, `POINTS_PER_DAY`, `EXTREME_TARGET_REFINEMENT_EPOCHS`.
- Pilot presets that include `PRETRAIN_EPOCHS=1000`, `PROPOSED_META_EPOCHS=1000`, `FEW_SHOT_EPOCHS=50`, and `EXTREME_TARGET_REFINEMENT_EPOCHS=50`.
- A dry-run path that prints commands without executing training.

**Step 2: Run test to verify failure**

Run:

```bash
python -m pytest WPF-under-extreme-weather-main/tests/test_2h_6point_launcher_ast.py -q
```

Expected before implementation: FAIL.

## Task 8: Add 2h/6 Launcher Presets

**Files:**
- Modify: `WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py`

**Step 1: Add env preview keys**

Append:

```python
"PROTOCOL_NAME",
"PROTOCOL_DATA_DIR",
"PROTOCOL_METADATA_PATH",
"SAMPLE_INTERVAL_HOURS",
"DOWNSAMPLE_OFFSET",
"LEN_REALP",
"POINTS_PER_DAY",
"EXTREME_TARGET_REFINEMENT_EPOCHS",
```

**Step 2: Add 2h/6 protocol defaults**

Set:

```python
PROTOCOL_NAME = "three_station_2h_6point_protocol"
PROTOCOL_DATA_DIR = str(ROOT / "protocol_data" / "2h_6p")
PROTOCOL_METADATA_PATH = str(ROOT / "protocol_data" / "2h_6p" / "protocol_metadata.json")
SAMPLE_INTERVAL_HOURS = "2"
DOWNSAMPLE_OFFSET = "1"
LEN_REALP = "6"
POINTS_PER_DAY = "12"
```

**Step 3: Add epoch presets**

Use these presets:

```python
SMOKE:
  PRETRAIN_EPOCHS=1
  PROPOSED_META_EPOCHS=1
  FEW_SHOT_EPOCHS=1
  EXTREME_TARGET_REFINEMENT_EPOCHS=1

PILOT_1K:
  PRETRAIN_EPOCHS=1000
  PROPOSED_META_EPOCHS=1000
  META_ONLY_META_EPOCHS=1000
  FEW_SHOT_EPOCHS=50
  EXTREME_TARGET_REFINEMENT_EPOCHS=50

PILOT_5K:
  PRETRAIN_EPOCHS=5000
  PROPOSED_META_EPOCHS=5000
  META_ONLY_META_EPOCHS=5000
  FEW_SHOT_EPOCHS=200
  EXTREME_TARGET_REFINEMENT_EPOCHS=200

FINAL_CANDIDATE:
  PRETRAIN_EPOCHS=10000
  PROPOSED_META_EPOCHS=10000
  META_ONLY_META_EPOCHS=10000
  FEW_SHOT_EPOCHS=500
  EXTREME_TARGET_REFINEMENT_EPOCHS=500
```

Do not set `80000/70000` as the formal default. Keep those only as a manual override command if needed.

**Step 4: Run launcher dry-run**

Run:

```bash
python WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py all --preset pilot-1k --dry-run
```

Expected: It prints build/train/eval stages with 2h/6 env and does not execute training.

## Task 9: Add Builder Shape Smoke

**Files:**
- Create: `WPF-under-extreme-weather-main/tests/test_2h_6point_generated_shapes.py`

**Step 1: Write a generated-file test**

This test should be skipped if `protocol_data/2h_6p/protocol_metadata.json` does not exist.

If files exist, assert:

- Metadata has `sample_interval_hours == 2`.
- Metadata has `downsample_offset == 1`.
- Metadata has `len_realp == 6`.
- Metadata has `points_per_day == 12`.
- Metadata has `window_span_hours == 12`.
- Each `58/59/60wf_4_train.mat` exists in `protocol_data/2h_6p/`.
- Each station’s 2023 full test length reshapes to 730 windows when using `LEN_REALP=6` and `POINTS_PER_DAY=12`.
- Every extreme class has nonnegative window counts and logs exact counts.

**Step 2: Run builder only**

Run:

```bash
cd WPF-under-extreme-weather-main
PROTOCOL_NAME=three_station_2h_6point_protocol \
PROTOCOL_DATA_DIR=protocol_data/2h_6p \
PROTOCOL_METADATA_PATH=protocol_data/2h_6p/protocol_metadata.json \
SAMPLE_INTERVAL_HOURS=2 DOWNSAMPLE_OFFSET=1 LEN_REALP=6 POINTS_PER_DAY=12 \
python -u build_three_station_extreme_yearly_protocol.py
```

Expected: No training. It writes 2h/6 mats and metadata under `protocol_data/2h_6p/`.

**Step 3: Run generated shape test**

Run:

```bash
python -m pytest WPF-under-extreme-weather-main/tests/test_2h_6point_generated_shapes.py -q
```

Expected after builder run: PASS.

## Task 10: Smoke Training Command For Codex Only

**Files:**
- No code changes if previous tasks pass.

**Step 1: Run training dry-run if supported**

If launcher supports dry-run:

```bash
python WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py train --preset smoke --dry-run
```

Expected: Prints training command with `LEN_REALP=6`, `POINTS_PER_DAY=12`, and `EXTREME_TARGET_REFINEMENT_EPOCHS=1`.

**Step 2: Optional true smoke**

Only if the user approves Codex running a tiny smoke:

```bash
cd WPF-under-extreme-weather-main
PROTOCOL_NAME=three_station_2h_6point_protocol \
PROTOCOL_DATA_DIR=protocol_data/2h_6p \
PROTOCOL_METADATA_PATH=protocol_data/2h_6p/protocol_metadata.json \
SAMPLE_INTERVAL_HOURS=2 DOWNSAMPLE_OFFSET=1 LEN_REALP=6 POINTS_PER_DAY=12 \
PRETRAIN_EPOCHS=1 PROPOSED_META_EPOCHS=1 META_ONLY_META_EPOCHS=1 \
FEW_SHOT_EPOCHS=1 EXTREME_TARGET_REFINEMENT_EPOCHS=1 \
PRETRAIN_LOG_INTERVAL=1 META_LOG_INTERVAL=1 FEW_SHOT_LOG_INTERVAL=1 \
python -u DemoModelTraining.py
```

Expected: The script enters all stages without shape errors. It is acceptable if performance is meaningless.

Do not run pilot or final from Codex.

## Task 11: User-Facing Pilot Commands

**Files:**
- Create or update: `WPF-under-extreme-weather-main/docs/plans/2026-04-11-2h-6point-runbook.md`

**Step 1: Add smoke command**

Document the builder smoke and training smoke commands.

**Step 2: Add Pilot-1k command**

Document:

```bash
cd /tmp/wpf-worktrees/restore-raw-data-83688a3/WPF-under-extreme-weather-main

PROTOCOL_NAME=three_station_2h_6point_protocol \
PROTOCOL_DATA_DIR=protocol_data/2h_6p \
PROTOCOL_METADATA_PATH=protocol_data/2h_6p/protocol_metadata.json \
SAMPLE_INTERVAL_HOURS=2 DOWNSAMPLE_OFFSET=1 LEN_REALP=6 POINTS_PER_DAY=12 \
PRETRAIN_EPOCHS=1000 PROPOSED_META_EPOCHS=1000 META_ONLY_META_EPOCHS=1000 \
FEW_SHOT_EPOCHS=50 EXTREME_TARGET_REFINEMENT_EPOCHS=50 \
PRETRAIN_LOG_INTERVAL=50 META_LOG_INTERVAL=50 FEW_SHOT_LOG_INTERVAL=10 \
python -u DemoModelTraining.py 2>&1 | tee logs/2h_6p_pilot1k_train_$(date +%Y%m%d_%H%M%S).log
```

Then eval:

```bash
PROTOCOL_NAME=three_station_2h_6point_protocol \
PROTOCOL_DATA_DIR=protocol_data/2h_6p \
PROTOCOL_METADATA_PATH=protocol_data/2h_6p/protocol_metadata.json \
SAMPLE_INTERVAL_HOURS=2 DOWNSAMPLE_OFFSET=1 LEN_REALP=6 POINTS_PER_DAY=12 \
python -u generate_multi_station_results.py 2>&1 | tee logs/2h_6p_pilot1k_eval_$(date +%Y%m%d_%H%M%S).log
```

**Step 3: Add Pilot-5k and final-candidate commands**

Use the same command shape with:

```text
5000 / 5000 / 200 / 200
10000 or 20000 / 10000 or 20000 / 500 / 500
```

State that final epoch counts must be chosen after inspecting `training_convergence_report.json`.

## Task 12: Verification Checklist Before Pilot

**Files:**
- No code changes.

Before giving the user Pilot-1k commands, verify:

```bash
git status --short
python -m pytest WPF-under-extreme-weather-main/tests/test_2h_6point_protocol_builder_ast.py -q
python -m pytest WPF-under-extreme-weather-main/tests/test_training_protocol_config_ast.py -q
python -m pytest WPF-under-extreme-weather-main/tests/test_eval_2h_6point_protocol_ast.py -q
python -m pytest WPF-under-extreme-weather-main/tests/test_2h_6point_launcher_ast.py -q
python -m pytest WPF-under-extreme-weather-main/tests/test_2h_6point_generated_shapes.py -q
```

If pytest is unavailable, run equivalent Python one-off checks or clearly report that automated pytest verification could not run.

## Expected End State

After implementation:

- `protocol_data/2h_6p/` contains isolated 2h/6-point `.mat` files and metadata.
- Training logs print protocol fields before loading station data.
- Evaluation CSV rows include protocol fields.
- No root-level 1h mats are overwritten.
- Smoke run has passed shape checks.
- The user receives Pilot-1k and optional Pilot-5k commands to run on the 4090.

