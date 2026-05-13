# High-Temperature Selective Fed-Meta Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement target-proxy-validated selective federated normal meta-learning for the high-temperature-only summer protocol, while keeping vanilla fed-meta as a baseline and delaying all extreme-stage federation to later ablations.

**Architecture:** Reuse the current personalized `Fed-Normal-Meta` skeleton, but split each station's summer normal windows into meta-train and proxy subsets, evaluate every client-returned meta update on target-side proxy tasks, reject harmful sources, and aggregate only accepted sources with self-floor protection. The final high-temperature adaptation remains local and uses the corrected all-shot `LMT` semantics.

**Tech Stack:** Python, PyTorch, scipy `.mat` assets, launcher flags, unittest AST/contract tests, local smoke runs, 4090 screen-based pilots.

---

### Task 1: Add Method Contract Tests

**Files:**
- Modify: `WPF-under-extreme-weather-main/tests/test_high_temp_only_training_ast.py`
- Modify: `WPF-under-extreme-weather-main/tests/test_training_protocol_config_ast.py`
- Modify: `WPF-under-extreme-weather-main/tests/test_high_temp_only_eval_ast.py`

**Step 1: Write the failing tests**

Add assertions for:

- selective fed-meta enable flag
- proxy split ratio config
- self-floor config
- gain margin config
- gain exponent config
- dedicated helper names for target-proxy validation and selective aggregation
- main result-table support for both vanilla and selective fed-meta rows

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest \
  tests.test_high_temp_only_training_ast \
  tests.test_training_protocol_config_ast \
  tests.test_high_temp_only_eval_ast -v
```

Expected: FAIL because the selective fed-meta tokens and routing are absent.

### Task 2: Add Launcher Controls For Selective Fed-Meta

**Files:**
- Modify: `WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py`
- Test: `WPF-under-extreme-weather-main/tests/test_high_temp_only_summer_launcher_ast.py`

**Step 1: Write the failing test**

Add assertions for launcher support of:

- `ENABLE_SELECTIVE_FED_NORMAL_META`
- `SELECTIVE_FED_META_PROXY_RATIO`
- `SELECTIVE_FED_META_SELF_FLOOR`
- `SELECTIVE_FED_META_GAIN_MARGIN`
- `SELECTIVE_FED_META_GAIN_GAMMA`

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_high_temp_only_summer_launcher_ast -v
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Extend the launcher to preview and pass through the new selective fed-meta env keys.

Keep `ENABLE_FED_NORMAL_META_PROPOSED` available because vanilla fed-meta remains a baseline.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_high_temp_only_summer_launcher_ast -v
```

Expected: PASS.

### Task 3: Split Summer Normal Windows Into Meta-Train And Proxy

**Files:**
- Modify: `WPF-under-extreme-weather-main/DemoModelTraining.py`
- Test: `WPF-under-extreme-weather-main/tests/test_high_temp_only_training_ast.py`

**Step 1: Write the failing test**

Add assertions that the high-temp-only path defines separate normal-meta train and proxy subsets for each station.

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_high_temp_only_training_ast -v
```

Expected: FAIL.

**Step 3: Write minimal implementation**

In `DemoModelTraining.py`:

- derive station-wise normal windows from the existing `p_conven_class / nwp_conven_class` assets
- split the windows once using a fixed seed
- create:
  - `meta_train` windows for local client updates
  - `proxy` windows for target-side evaluation

The split must be window-level and deterministic.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_high_temp_only_training_ast -v
```

Expected: PASS.

### Task 4: Add Target-Proxy Evaluation Helpers

**Files:**
- Modify: `WPF-under-extreme-weather-main/DemoModelTraining.py`
- Test: `WPF-under-extreme-weather-main/tests/test_high_temp_only_training_ast.py`

**Step 1: Write the failing test**

Add assertions for helper functions that:

- evaluate a returned client model on target proxy tasks
- compute proxy gain relative to the target self-update

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_high_temp_only_training_ast -v
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Add helper functions that:

- load candidate client states into a shared eval model
- compute proxy loss on `Q_s^proxy`
- compute `gain_{s,c}^r`

Keep the loss aligned with the normal-meta stage metric used for model selection.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_high_temp_only_training_ast -v
```

Expected: PASS.

### Task 5: Implement Selective Aggregation

**Files:**
- Modify: `WPF-under-extreme-weather-main/DemoModelTraining.py`
- Test: `WPF-under-extreme-weather-main/tests/test_high_temp_only_training_ast.py`

**Step 1: Write the failing test**

Add assertions for selective aggregation helpers that:

- reject sources whose gain does not exceed the margin
- retain self-update always
- soft-weight only accepted sources
- fall back to self-only when no source is accepted

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_high_temp_only_training_ast -v
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Implement:

- accepted-source selection from target-proxy gains
- soft weighting from positive gains raised to `gamma`
- self-floor insertion
- final weighted parameter aggregation

Keep the first implementation simple:

- margin `m = 0`
- exponent `gamma = 1`
- no top-k

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_high_temp_only_training_ast -v
```

Expected: PASS.

### Task 6: Extend Fed-Normal-Meta Training Loop

**Files:**
- Modify: `WPF-under-extreme-weather-main/DemoModelTraining.py`
- Test: `WPF-under-extreme-weather-main/tests/test_high_temp_only_training_ast.py`

**Step 1: Write the failing test**

Add assertions that the fed-meta training loop can run in:

- vanilla mode
- selective mode

and that selective mode performs:

- client local meta update
- target-proxy validation
- selective aggregation

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_high_temp_only_training_ast -v
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Reuse `run_fed_normal_meta_training()` as the orchestration skeleton, but add a selective branch:

- vanilla branch: old self-floor weighted FedAvg
- selective branch: target-proxy-validated hard reject plus soft weighting

Keep both branches available because vanilla remains a baseline.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_high_temp_only_training_ast -v
```

Expected: PASS.

### Task 7: Expose Vanilla And Selective Fed-Meta In Result Tables

**Files:**
- Modify: `WPF-under-extreme-weather-main/generate_multi_station_results.py`
- Modify: `WPF-under-extreme-weather-main/DemoModelTraining.py`
- Test: `WPF-under-extreme-weather-main/tests/test_high_temp_only_eval_ast.py`

**Step 1: Write the failing test**

Add assertions that the high-temp-only output table can include:

- `Vanilla Fed-Normal-Meta + Local FT`
- `Selective Fed-Normal-Meta + Local FT`

and optionally:

- `Fed-Normal-Meta-NoFT`

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_high_temp_only_eval_ast -v
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Extend result generation and model evaluation routing so the corrected main comparison set can be exported directly.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_high_temp_only_eval_ast -v
```

Expected: PASS.

### Task 8: Add Diagnostics For Source Admission

**Files:**
- Modify: `WPF-under-extreme-weather-main/DemoModelTraining.py`
- Verify: `WPF-under-extreme-weather-main/artifacts/.../training_convergence_report.json`

**Step 1: Write the failing test**

Add assertions that selective fed-meta logs and convergence reports include source-admission diagnostics such as:

- accepted source count
- accepted source ids
- per-source proxy gains

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_high_temp_only_training_ast -v
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Log round-level diagnostics for each target station:

- self proxy loss
- each source proxy loss
- each source gain
- final accepted set
- final aggregation weights

These are required to prove the selective mechanism is active.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_high_temp_only_training_ast -v
```

Expected: PASS.

### Task 9: Local Verification And Smoke

**Files:**
- Verify: `WPF-under-extreme-weather-main/DemoModelTraining.py`
- Verify: `WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py`
- Verify: `WPF-under-extreme-weather-main/generate_multi_station_results.py`
- Verify: `WPF-under-extreme-weather-main/build_three_station_extreme_yearly_protocol.py`

**Step 1: Run targeted tests**

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

**Step 2: Run compilation**

Run:

```bash
python -m py_compile \
  run_three_station_yearly_protocol.py \
  build_three_station_extreme_yearly_protocol.py \
  DemoModelTraining.py \
  generate_multi_station_results.py
```

Expected: exit `0`.

**Step 3: Run smoke for vanilla fed-meta**

Run:

```bash
python run_three_station_yearly_protocol.py all \
  --preset smoke \
  --high-temp-only-summer \
  --meta-shot-regime 2x4 \
  --enable-fed-normal-meta-proposed
```

Expected: PASS.

**Step 4: Run smoke for selective fed-meta**

Run:

```bash
python run_three_station_yearly_protocol.py all \
  --preset smoke \
  --high-temp-only-summer \
  --meta-shot-regime 2x4 \
  --enable-selective-fed-normal-meta
```

Expected:

- full pipeline completes
- result table contains vanilla and selective rows
- diagnostics show accepted-source behavior

### Task 10: 4090 Pilot Handoff

**Files:**
- Output: `WPF-under-extreme-weather-main/logs/*.log`
- Output: `WPF-under-extreme-weather-main/artifacts/high_temp_only_summer_six_station/*`

**Step 1: Vanilla baseline pilot**

Run in attached `screen`:

```bash
REPO=/tmp/wpf-worktrees/restore-raw-data-83688a3/WPF-under-extreme-weather-main
screen -L -Logfile "$REPO/logs/high_temp_vanilla_fed_meta_pilot_2x4.log" -S ht_vanilla_fed_meta_pilot_2x4 bash -lc "cd '$REPO' && ARTIFACT_DIR='$REPO/artifacts/high_temp_only_summer_six_station/vanilla-fed-meta-pilot-2x4' python run_three_station_yearly_protocol.py all --preset pilot-1k --high-temp-only-summer --meta-shot-regime 2x4 --enable-fed-normal-meta-proposed; exec bash"
```

**Step 2: Selective proposed pilot**

Run in attached `screen`:

```bash
REPO=/tmp/wpf-worktrees/restore-raw-data-83688a3/WPF-under-extreme-weather-main
screen -L -Logfile "$REPO/logs/high_temp_selective_fed_meta_pilot_2x4.log" -S ht_selective_fed_meta_pilot_2x4 bash -lc "cd '$REPO' && ARTIFACT_DIR='$REPO/artifacts/high_temp_only_summer_six_station/selective-fed-meta-pilot-2x4' python run_three_station_yearly_protocol.py all --preset pilot-1k --high-temp-only-summer --meta-shot-regime 2x4 --enable-selective-fed-normal-meta; exec bash"
```

Expected pilot review:

- selective fed-meta beats vanilla fed-meta on `Overall_Average`
- selective fed-meta beats corrected `LMT`
- diagnostics show non-trivial source rejection or reweighting behavior

**Step 3: Final candidate**

Run only after pilot review:

```bash
REPO=/tmp/wpf-worktrees/restore-raw-data-83688a3/WPF-under-extreme-weather-main
screen -L -Logfile "$REPO/logs/high_temp_selective_fed_meta_final_2x4.log" -S ht_selective_fed_meta_final_2x4 bash -lc "cd '$REPO' && ARTIFACT_DIR='$REPO/artifacts/high_temp_only_summer_six_station/selective-fed-meta-final-2x4' python run_three_station_yearly_protocol.py all --preset final-candidate --high-temp-only-summer --meta-shot-regime 2x4 --enable-selective-fed-normal-meta; exec bash"
```

Expected:

- proposed selective fed-meta is the best row in the corrected main table
- `Overall_Average` improvement over corrected `LMT` reaches the target band
