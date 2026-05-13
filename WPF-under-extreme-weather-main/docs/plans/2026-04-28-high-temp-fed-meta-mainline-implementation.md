# High-Temperature Fed-Meta Mainline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-center the `High temperature only` summer protocol around federated normal meta-learning, while correcting the `LMT` target-shot drift and promoting `Local-Meta-NoFT` to a first-class baseline.

**Architecture:** Keep the current summer-only high-temperature protocol assets and six-station phase augmentation. Correct the baseline fine-tune semantics first, then route the proposed mainline through `Fed-Normal-Meta` at the normal-meta stage while keeping the final high-temperature fine-tune local. Keep extreme-stage federation methods as optional ablations, not as the initial mainline.

**Tech Stack:** Python, PyTorch, scipy `.mat` assets, launcher env flags, unittest AST/contract tests, local smoke plus 4090 screen runs.

---

### Task 1: Add Baseline Contract Tests

**Files:**
- Modify: `WPF-under-extreme-weather-main/tests/test_high_temp_only_training_ast.py`
- Modify: `WPF-under-extreme-weather-main/tests/test_high_temp_only_eval_ast.py`
- Modify: `WPF-under-extreme-weather-main/tests/test_training_protocol_config_ast.py`

**Step 1: Write the failing tests**

Add assertions for:

- paper-consistent `LMT` target-shot usage without `adapt/val` holdout in the corrected path
- export support for `Local-Meta-NoFT`
- proposed base-model resolver preferring fed-meta checkpoints when the new mode is enabled
- extreme-stage federation remaining optional rather than implicit in the mainline

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest \
  tests.test_high_temp_only_training_ast \
  tests.test_high_temp_only_eval_ast \
  tests.test_training_protocol_config_ast -v
```

Expected: FAIL because the corrected routing and new baseline tokens are not all present.

### Task 2: Add Launcher Flags For The Fed-Meta Mainline

**Files:**
- Modify: `WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py`
- Test: `WPF-under-extreme-weather-main/tests/test_high_temp_only_summer_launcher_ast.py`

**Step 1: Write the failing test**

Add assertions for launcher preview or env export keys covering:

- `ENABLE_FED_NORMAL_META_PROPOSED`
- `FED_NORMAL_META_SELF_FLOOR`
- a dedicated high-temp fed-meta mode selector
- optional switch for evaluating `Local-Meta-NoFT`

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_high_temp_only_summer_launcher_ast -v
```

Expected: FAIL because the new launcher controls are missing.

**Step 3: Write minimal implementation**

Extend the launcher so the high-temp-only path can preview and pass through:

- fed-meta enablement
- self-floor
- whether to skip fed-meta reuse
- whether to emit `Local-Meta-NoFT` in evaluation

Do not add new presets yet; use explicit flags or env overrides.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_high_temp_only_summer_launcher_ast -v
```

Expected: PASS.

### Task 3: Correct The LMT T-Stage To Use All Target Shots

**Files:**
- Modify: `WPF-under-extreme-weather-main/DemoModelTraining.py`
- Test: `WPF-under-extreme-weather-main/tests/test_high_temp_only_training_ast.py`

**Step 1: Write the failing test**

Add assertions that the corrected `LMT` path no longer trains only on `target_payload["adapt_*"]` for the new mainline.

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_high_temp_only_training_ast -v
```

Expected: FAIL because current `LMT` still uses the split payload.

**Step 3: Write minimal implementation**

In `DemoModelTraining.py`:

- keep the existing data split available if other diagnostics still need it
- add a corrected `LMT` training path for the high-temp mainline that uses all available target support windows for fine-tuning
- keep the optimizer scope identical to the paper-consistent setting:
  - update only LWP plus prediction head
  - same learning rate and betas
  - same epoch budget

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_high_temp_only_training_ast -v
```

Expected: PASS.

### Task 4: Expose Local-Meta-NoFT As A First-Class Baseline

**Files:**
- Modify: `WPF-under-extreme-weather-main/DemoModelTraining.py`
- Modify: `WPF-under-extreme-weather-main/generate_multi_station_results.py`
- Test: `WPF-under-extreme-weather-main/tests/test_high_temp_only_eval_ast.py`

**Step 1: Write the failing test**

Add assertions that evaluation/export logic can emit a `Local-Meta-NoFT` row for the high-temp-only protocol.

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_high_temp_only_eval_ast -v
```

Expected: FAIL because the baseline is not exported yet.

**Step 3: Write minimal implementation**

In `DemoModelTraining.py`:

- add a direct evaluation branch from each station's local meta checkpoint to the high-temperature test set

In `generate_multi_station_results.py`:

- include `Local-Meta-NoFT` in the table schema and aggregation logic

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_high_temp_only_eval_ast -v
```

Expected: PASS.

### Task 5: Route The Proposed Mainline Through Fed-Normal-Meta

**Files:**
- Modify: `WPF-under-extreme-weather-main/DemoModelTraining.py`
- Test: `WPF-under-extreme-weather-main/tests/test_high_temp_only_training_ast.py`

**Step 1: Write the failing test**

Add assertions that, when the fed-meta mainline is enabled:

- proposed initialization resolves to fed-normal-meta checkpoints
- corrected `LMT` still resolves to local meta
- the final high-temperature fine-tune remains local

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_high_temp_only_training_ast -v
```

Expected: FAIL because routing is not yet separated.

**Step 3: Write minimal implementation**

In `DemoModelTraining.py`:

- reuse the existing `run_fed_normal_meta_training()` skeleton for the high-temp-only protocol
- add a base-model resolver for the new mainline
- make the proposed mainline start from `fed_normal_meta_station{s}` for each target station
- keep the final high-temperature fine-tune local in this mainline

Do not yet enable extreme-stage source aggregation in the mainline path.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_high_temp_only_training_ast -v
```

Expected: PASS.

### Task 6: Keep Extreme-Stage Federation As Optional Ablation

**Files:**
- Modify: `WPF-under-extreme-weather-main/DemoModelTraining.py`
- Modify: `WPF-under-extreme-weather-main/generate_multi_station_results.py`
- Test: `WPF-under-extreme-weather-main/tests/test_high_temp_only_eval_ast.py`

**Step 1: Write the failing test**

Add assertions that the mainline comparison can run without mandatory `Extreme-FedAvg` or `Proposed-A` extreme-stage aggregation rows, while still allowing them as optional ablations.

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_high_temp_only_eval_ast -v
```

Expected: FAIL because current result generation assumes the existing trio.

**Step 3: Write minimal implementation**

Adjust model lists and export logic so the corrected main table can be:

- `Pretrain`
- `Local-Meta-NoFT`
- `LMT`
- `Fed-Normal-Meta + Local FT`

and the old extreme-stage federation methods can be toggled back on as ablations.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_high_temp_only_eval_ast -v
```

Expected: PASS.

### Task 7: Run Local Verification And Smoke

**Files:**
- Verify: `WPF-under-extreme-weather-main/DemoModelTraining.py`
- Verify: `WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py`
- Verify: `WPF-under-extreme-weather-main/generate_multi_station_results.py`
- Verify: `WPF-under-extreme-weather-main/build_three_station_extreme_yearly_protocol.py`

**Step 1: Run targeted test suite**

Run:

```bash
python -m unittest \
  tests.test_high_temp_only_summer_launcher_ast \
  tests.test_high_temp_only_protocol_builder_ast \
  tests.test_high_temp_only_training_ast \
  tests.test_high_temp_only_eval_ast \
  tests.test_training_protocol_config_ast -v
```

Expected: PASS.

**Step 2: Run compilation check**

Run:

```bash
python -m py_compile \
  run_three_station_yearly_protocol.py \
  build_three_station_extreme_yearly_protocol.py \
  DemoModelTraining.py \
  generate_multi_station_results.py
```

Expected: exit `0`.

**Step 3: Run local smoke**

Run:

```bash
python run_three_station_yearly_protocol.py all \
  --preset smoke \
  --high-temp-only-summer \
  --meta-shot-regime 2x4 \
  --enable-fed-normal-meta-proposed
```

Expected:

- protocol build succeeds
- train/eval complete locally
- result table includes `Local-Meta-NoFT`
- proposed mainline uses fed-meta initialization and local final FT

### Task 8: 4090 Pilot Handoff

**Files:**
- Output: `WPF-under-extreme-weather-main/logs/*.log`
- Output: `WPF-under-extreme-weather-main/artifacts/high_temp_only_summer_six_station/*`

**Step 1: Pilot command**

Run in an attached `screen` session:

```bash
REPO=/tmp/wpf-worktrees/restore-raw-data-83688a3/WPF-under-extreme-weather-main
screen -L -Logfile "$REPO/logs/high_temp_fed_meta_pilot_2x4.log" -S ht_fed_meta_pilot_2x4 bash -lc "cd '$REPO' && ARTIFACT_DIR='$REPO/artifacts/high_temp_only_summer_six_station/fed-meta-pilot-2x4' python run_three_station_yearly_protocol.py all --preset pilot-1k --high-temp-only-summer --meta-shot-regime 2x4 --enable-fed-normal-meta-proposed; exec bash"
```

Expected:

- local smoke assumptions hold on 4090
- result table contains corrected baseline set

**Step 2: Final-candidate command**

Run only after pilot review:

```bash
REPO=/tmp/wpf-worktrees/restore-raw-data-83688a3/WPF-under-extreme-weather-main
screen -L -Logfile "$REPO/logs/high_temp_fed_meta_final_2x4.log" -S ht_fed_meta_final_2x4 bash -lc "cd '$REPO' && ARTIFACT_DIR='$REPO/artifacts/high_temp_only_summer_six_station/fed-meta-final-2x4' python run_three_station_yearly_protocol.py all --preset final-candidate --high-temp-only-summer --meta-shot-regime 2x4 --enable-fed-normal-meta-proposed; exec bash"
```

Expected:

- `Overall_Average` proposed mainline beats corrected `LMT`
- `Local-Meta-NoFT` remains below or near the proposed mainline
