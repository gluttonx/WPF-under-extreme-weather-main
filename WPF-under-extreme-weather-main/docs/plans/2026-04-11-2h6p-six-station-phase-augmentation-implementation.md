# 2h/6p Six-Station Phase Augmentation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a six-station 2h/6p protocol where complementary hour-phase data become stations `61 / 62 / 63`.

**Architecture:** Keep the existing 2h/6p protocol structure, but make station lists metadata-driven. The builder creates six `.mat` files when `PHASE_AUGMENT_STATIONS=1`; training and evaluation read all stations from metadata.

**Tech Stack:** Python, NumPy, SciPy `.mat`, PyTorch, pytest/unittest AST tests.

---

### Task 1: Builder Contract Tests

**Files:**
- Modify: `tests/test_2h_6point_protocol_builder_ast.py`

**Steps:**
1. Assert the builder exposes `PHASE_AUGMENT_STATIONS`, `PHASE_AUGMENT_STATION_MAP`, and `build_protocol_station_configs`.
2. Assert station IDs `61 / 62 / 63` are generated when phase augmentation is enabled.
3. Assert per-station metadata contains `downsample_offset`.

### Task 2: Builder Implementation

**Files:**
- Modify: `build_three_station_extreme_yearly_protocol.py`

**Steps:**
1. Rename the base station list to `BASE_YEARLY_PROTOCOL_STATIONS`.
2. Add `PHASE_AUGMENT_STATIONS` env flag.
3. Add `PHASE_AUGMENT_STATION_MAP = {"58": "61", "59": "62", "60": "63"}`.
4. Add `build_protocol_station_configs()` that returns either 3 or 6 station configs.
5. Use each station config's own `downsample_offset` in all support/test/normal/extreme downsampling.
6. Write six station entries and six `.mat` files when enabled.

### Task 3: Training Station List Tests

**Files:**
- Modify: `tests/test_training_protocol_config_ast.py`

**Steps:**
1. Assert training has `resolve_station_ids`.
2. Assert training reads station IDs from yearly protocol metadata.
3. Assert the source loop still only skips exact self, not same physical station.

### Task 4: Training Implementation

**Files:**
- Modify: `DemoModelTraining.py`

**Steps:**
1. Add `resolve_station_ids()` near protocol config helpers.
2. If `USE_FEDERATION` and metadata contains stations, return those metadata station IDs.
3. Else preserve legacy fallback `['58', '59', '60']`.
4. Update logging text to use `len(station_ids)` and dynamic IDs.
5. Export `station_ids` in convergence report already exists; keep it dynamic.

### Task 5: Evaluation Tests

**Files:**
- Modify: `tests/test_eval_2h_6point_protocol_ast.py`

**Steps:**
1. Assert eval has `resolve_station_ids`.
2. Assert eval supports `EVAL_STATION_IDS`.
3. Assert station ordering is dynamic, not pinned to `['58', '59', '60']`.
4. Assert output paths can be overridden.

### Task 6: Evaluation Implementation

**Files:**
- Modify: `generate_multi_station_results.py`

**Steps:**
1. Add `resolve_station_ids()` from metadata plus optional `EVAL_STATION_IDS`.
2. Add `RESULTS_OUTPUT_PATH` and `TASK_RESULTS_OUTPUT_PATH` env variables.
3. Replace fixed station list and fixed sorting with dynamic station IDs plus `Overall_Average`.
4. Update printed row counts using dynamic station count.

### Task 7: Launcher Tests

**Files:**
- Modify: `tests/test_2h_6point_launcher_ast.py`

**Steps:**
1. Assert launcher exposes `PHASE_AUGMENT_STATIONS`.
2. Assert launcher exposes a six-station protocol preset or mode.
3. Assert preview env includes `PHASE_AUGMENT_STATIONS`.

### Task 8: Launcher Implementation

**Files:**
- Modify: `run_three_station_yearly_protocol.py`

**Steps:**
1. Add `PHASE_AUGMENT_STATIONS`.
2. Add `SIX_STATION_PROTOCOL_NAME` and `SIX_STATION_PROTOCOL_DATA_DIR`.
3. Add `--six-station` flag.
4. When enabled, set protocol name/data dir/metadata path and `PHASE_AUGMENT_STATIONS=1`.
5. Keep pilot-5k budget unchanged and no K-shot max windows.

### Task 9: Generated Shape Tests

**Files:**
- Modify: `tests/test_2h_6point_generated_shapes.py`

**Steps:**
1. If six-station metadata exists, assert six station IDs.
2. Assert `61 / 62 / 63` mats exist.
3. Assert `58` and `61` have complementary per-station offsets.

### Task 10: Smoke Verification

**Commands:**

```bash
python -m pytest tests/test_2h_6point_protocol_builder_ast.py tests/test_training_protocol_config_ast.py tests/test_eval_2h_6point_protocol_ast.py tests/test_2h_6point_launcher_ast.py -q
```

```bash
python run_three_station_yearly_protocol.py build --preset smoke --six-station
```

```bash
python -m pytest tests/test_2h_6point_generated_shapes.py -q
```

```bash
python run_three_station_yearly_protocol.py all --preset smoke --six-station
```

The last command is smoke only. Do not run pilot-5k in Codex.

