# High-Temperature-Only Summer Protocol Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a new six-station `2h/6p` protocol that trains only on `2023-06~08` summer normal weather plus `2023` `High temperature`, tests on `2024` `High temperature`, and compares `LMT`, `Extreme-FedAvg`, and `Proposed-A` under the approved `2×4` and `3×3` meta settings.

**Architecture:** Extend the existing yearly protocol builder and launcher rather than creating a separate preprocessing pipeline. Keep the current model family and optimization logic, but make protocol filtering, per-station `k`, and single-class extreme exports explicit in the builder and training/evaluation code paths.

**Tech Stack:** Python, NumPy, SciPy `.mat`, existing xlsx XML parsing helpers, `DemoModelTraining.py`, `run_three_station_yearly_protocol.py`, repository AST/contract tests, shell launch commands.

---

## Guardrails

Do not overwrite or rename the current `two_year_2024_k6` assets.

Do not create a new standalone preprocessing script outside the protocol builder.

Do not mix this new protocol with `Fed-Normal-Meta`, class-wise gating, or class-wise fallback in the first implementation.

Do not run long training locally. Builder smoke tests and AST checks are enough for local verification.

## Task 1: Add Builder Contract Tests For The New Seasonal Single-Class Protocol

**Files:**
- Create: `WPF-under-extreme-weather-main/tests/test_high_temp_only_protocol_builder_ast.py`
- Modify later: `WPF-under-extreme-weather-main/build_three_station_extreme_yearly_protocol.py`

**Step 1: Write the failing test**

Add AST checks asserting the builder can define configuration equivalent to:

```python
TRAIN_YEARS = [2023]
TEST_YEARS = [2024]
NORMAL_FILTER_MONTHS = [6, 7, 8]
EXTREME_SUPPORT_FILTER_MONTHS = [6, 7, 8]
EXTREME_CLASS_NAMES = ["high_temp"]
NUM_CONVENTIONAL_CLASSES_BY_STATION = {
    "58": 7, "61": 7,
    "59": 6, "62": 6,
    "60": 5, "63": 5,
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_high_temp_only_protocol_builder_ast -v
```

Expected: FAIL because the current builder still encodes the two-year four-class protocol.

**Step 3: Commit**

```bash
git add tests/test_high_temp_only_protocol_builder_ast.py
git commit -m "test: define high-temp-only summer protocol contract"
```

Only commit if the user explicitly asks for commits.

## Task 2: Add A New Named Protocol Mode To The Launcher

**Files:**
- Modify: `WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py`
- Test: `WPF-under-extreme-weather-main/tests/test_high_temp_only_protocol_builder_ast.py`

**Step 1: Write the failing test**

Add AST checks asserting the launcher defines a distinct protocol name and output roots for the new protocol, for example:

```python
HIGH_TEMP_ONLY_PROTOCOL_NAME = "six_station_2h_6point_high_temp_2023summer_protocol"
HIGH_TEMP_ONLY_PROTOCOL_DATA_DIR = ...
HIGH_TEMP_ONLY_ARTIFACT_DIR = ...
```

and that `build_env(...)` can select this mode separately from the existing `two_year_2024_k6` mode.

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_high_temp_only_protocol_builder_ast -v
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Extend `run_three_station_yearly_protocol.py` so the new protocol can be selected through a dedicated flag or mode branch. The branch should set:

- new protocol name
- new protocol data dir
- new artifact dir
- `PHASE_AUGMENT_STATIONS=1`
- new meta-setting presets for `2×4` and `3×3`

**Step 4: Run smoke verification**

Run:

```bash
python run_three_station_yearly_protocol.py --help
```

Expected: the new protocol mode or preset appears in the CLI help or branch structure.

## Task 3: Extend The Builder For Summer-Only Normal And Single-Class High Temperature

**Files:**
- Modify: `WPF-under-extreme-weather-main/build_three_station_extreme_yearly_protocol.py`
- Test: `WPF-under-extreme-weather-main/tests/test_high_temp_only_protocol_builder_ast.py`

**Step 1: Write the failing test**

Add contract checks asserting the builder:

- filters `normal_weather` to `2023-06~08`
- filters `extreme_high_temp` support to `2023-06~08`
- filters `extreme_high_temp` test to `2024`
- no longer requires all four extreme classes for this protocol mode

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_high_temp_only_protocol_builder_ast -v
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Refactor the builder so the protocol mode can:

- use full `2023-06~08` normal rows without the old `360-point` cap
- export only the `high_temp` extreme support/test arrays for this protocol mode
- store per-station conventional-cluster counts instead of one global `NUM_CONVENTIONAL_CLASSES`
- record the active month filters and single-extreme-class setting in metadata

**Step 4: Run smoke verification**

Run:

```bash
python build_three_station_extreme_yearly_protocol.py
```

with the new protocol mode env enabled and a scratch output directory.

Expected:

- metadata writes successfully
- six stations are emitted
- only `high_temp` support/test counts are recorded for the new protocol mode

## Task 4: Make Training And Evaluation Support Single-Class Extreme Exports

**Files:**
- Modify: `WPF-under-extreme-weather-main/DemoModelTraining.py`
- Test: `WPF-under-extreme-weather-main/tests/test_high_temp_only_training_ast.py`

**Step 1: Write the failing test**

Create `WPF-under-extreme-weather-main/tests/test_high_temp_only_training_ast.py` with AST checks asserting:

- the extreme-class loop can be driven by protocol metadata instead of hard-coded `range(4)`
- result exports can emit a single `HighTemperature_*` block plus `Overall_Average`
- model-count calculations no longer assume exactly four extreme classes

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_high_temp_only_training_ast -v
```

Expected: FAIL because the current script assumes four extreme classes in several places.

**Step 3: Write minimal implementation**

Refactor `DemoModelTraining.py` so the new protocol mode:

- reads the active extreme-class list from protocol metadata
- keeps per-station `p_conven_class` counts as-is
- loops over one extreme class for few-shot training and evaluation
- exports rows labeled consistently as `HighTemperature`
- still produces the same three methods:
  - `LMT`
  - `Extreme-FedAvg`
  - `Proposed-A`

Do not enable `TRAIN_META_ONLY_BASELINE` in this protocol's first pass.

**Step 4: Run smoke verification**

Run a one-epoch smoke command with the new protocol and verify the script reaches model export without referencing missing class-2/3/4 arrays.

## Task 5: Wire The Approved `2×4` And `3×3` Meta Presets

**Files:**
- Modify: `WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py`
- Optionally modify: `WPF-under-extreme-weather-main/DemoModelTraining.py` if exposure bookkeeping needs explicit logging
- Test: `WPF-under-extreme-weather-main/tests/test_high_temp_only_training_ast.py`

**Step 1: Write the failing test**

Add AST checks asserting the launcher exposes two explicit preset branches for the new protocol:

```python
high-temp-2x4
high-temp-3x3
```

or equivalent preset names that set:

```python
META_TASKS_PER_EPOCH
META_SUPPORT_SHOTS
META_QUERY_SHOTS
PROPOSED_META_EPOCHS
META_ONLY_META_EPOCHS
```

according to the approved exposure-matched plan.

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_high_temp_only_training_ast -v
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Add two launcher presets:

### `high-temp-2x4`

```bash
META_TASKS_PER_EPOCH=2
META_SUPPORT_SHOTS=4
META_QUERY_SHOTS=4
PROPOSED_META_EPOCHS=12000
META_ONLY_META_EPOCHS=12000
```

### `high-temp-3x3`

```bash
META_TASKS_PER_EPOCH=3
META_SUPPORT_SHOTS=3
META_QUERY_SHOTS=3
PROPOSED_META_EPOCHS=8000
META_ONLY_META_EPOCHS=8000
```

Both presets must keep:

```bash
PRETRAIN_EPOCHS
FEW_SHOT_EPOCHS
EXTREME_TARGET_REFINEMENT_EPOCHS
learning rate = 0.0002
betas = (0.5, 0.999)
```

consistent unless explicitly changed later.

**Step 4: Run smoke verification**

Run:

```bash
python run_three_station_yearly_protocol.py --preset high-temp-2x4 --build-only
python run_three_station_yearly_protocol.py --preset high-temp-3x3 --build-only
```

If `--build-only` does not yet exist, add it in the smallest possible way and verify it only builds protocol assets plus prints env preview.

## Task 6: Add The First Runbook Commands

**Files:**
- Create: `WPF-under-extreme-weather-main/docs/plans/2026-04-27-high-temp-only-summer-runbook.md`
- Modify if needed: `WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py`

**Step 1: Write the runbook content**

Document the exact intended first-run commands for the 4090 server.

Include a build step:

```bash
python run_three_station_yearly_protocol.py --high-temp-only --build-only
```

Include the three-method `2×4` run:

```bash
python run_three_station_yearly_protocol.py --high-temp-only --preset high-temp-2x4
```

Include the three-method `3×3` run:

```bash
python run_three_station_yearly_protocol.py --high-temp-only --preset high-temp-3x3
```

Include the expected artifact roots, for example:

```bash
artifacts/high_temp_only_summer/high-temp-2x4
artifacts/high_temp_only_summer/high-temp-3x3
```

**Step 2: Verify the documented commands**

Run the build-only command locally and confirm:

- protocol assets are created in the documented directory
- env preview prints the expected meta settings

## Task 7: Run Contract Tests And Smoke Checks

**Files:**
- Test: `WPF-under-extreme-weather-main/tests/test_high_temp_only_protocol_builder_ast.py`
- Test: `WPF-under-extreme-weather-main/tests/test_high_temp_only_training_ast.py`

**Step 1: Run tests**

```bash
python -m unittest \
  tests.test_high_temp_only_protocol_builder_ast \
  tests.test_high_temp_only_training_ast -v
```

Expected: PASS.

**Step 2: Run smoke build**

```bash
python run_three_station_yearly_protocol.py --high-temp-only --preset smoke --build-only
```

Expected:

- metadata exists
- six station assets exist
- no four-class assumptions are triggered

**Step 3: Run smoke train**

```bash
python run_three_station_yearly_protocol.py --high-temp-only --preset smoke
```

Expected:

- the run finishes with one extreme class
- exported CSV contains per-station `HighTemperature` rows and `Overall_Average`

## Intended Method Mapping For The First Real Experiment

### Baseline 1: `LMT`

- local summer-normal pretrain
- local summer-normal meta-train
- target-only `High temperature` few-shot adaptation
- target refinement

### Baseline 2: `Extreme-FedAvg`

- same normal-weather base as `LMT`
- extreme-stage source updates borrowed from other stations
- uniform aggregation
- target refinement

### Main method: `Proposed-A`

- same normal-weather base as `LMT`
- extreme-stage source updates borrowed from other stations
- reliability-aware weighted aggregation
- target refinement

### Explicitly excluded from the first pass

- `Fed-Normal-Meta`
- class-wise gate
- class-wise fallback
- `Meta-only`

## First Long-Run Matrix

After code implementation and smoke verification, the first long-run matrix should be:

1. `high-temp-2x4` with `LMT / Extreme-FedAvg / Proposed-A`
2. `high-temp-3x3` with `LMT / Extreme-FedAvg / Proposed-A`

Pick the better of the two by `Overall_Average` `HighTemperature_nMAE_%`, then use that setting for any later follow-up experiments.
