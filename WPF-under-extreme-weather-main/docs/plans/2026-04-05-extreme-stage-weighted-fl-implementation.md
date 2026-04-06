# Extreme-Stage Weighted FL Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild the 3-station yearly protocol from raw `xlsx`, then implement same-class extreme-stage weighted FL on top of the original `RAPP-original data` training skeleton.

**Architecture:** Keep the original three-stage training backbone, remove dependence on the seasonal six-client path for the new main method, and add a new yearly extreme protocol plus a target-conditioned reliability-aware aggregation module for extreme adaptation. Validation first compares `LMT-new`, `Extreme-FedAvg`, and `Proposed-A` under the same `2022 support -> 2023 test` split.

**Tech Stack:** Python, PyTorch, SciPy `.mat`, raw `.xlsx` parsing, existing `DemoModelTraining.py`, existing result scripts.

---

### Task 1: Freeze the yearly 3-station protocol in code-level docs

**Files:**
- Create: `docs/plans/2026-04-05-extreme-stage-weighted-fl-design.md`
- Create: `docs/plans/2026-04-05-extreme-stage-weighted-fl-implementation.md`
- Modify: `AGENTS.md`

**Step 1: Write the design doc**
- Copy the validated method definition into a permanent design document.

**Step 2: Append the decision log**
- Add a new dated section to `AGENTS.md` recording:
  - the new main proposition,
  - why seasonal normal-only FL is no longer the main route,
  - the two new innovation points.

**Step 3: Verify paths and naming**

Run: `ls docs/plans | grep 2026-04-05`
Expected: both new files appear.

### Task 2: Add a raw-yearly extreme protocol builder

**Files:**
- Create: `build_three_station_extreme_yearly_protocol.py`
- Test: `tests/test_three_station_extreme_yearly_protocol_ast.py`

**Step 1: Write the failing test**
- Assert the builder exports explicit `2022 support` and `2023 test` extreme assets per station/class.

**Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_three_station_extreme_yearly_protocol_ast -v`
Expected: FAIL because builder does not exist yet.

**Step 3: Write minimal implementation**
- Read raw `2223jilin_058/059/060_processed_4classes.xlsx`
- Build yearly split assets:
  - `2022` extreme support
  - `2023` extreme test
  - keep conventional train/test information needed by the original skeleton
- Do not reuse old mixed-year `p_extre_class*` as the new protocol source of truth.

**Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_three_station_extreme_yearly_protocol_ast -v`
Expected: PASS.

### Task 3: Introduce a new protocol switch in training

**Files:**
- Modify: `DemoModelTraining.py`
- Test: `tests/test_three_station_yearly_protocol_ast.py`

**Step 1: Write the failing test**
- Assert a dedicated protocol flag/path exists for the new yearly extreme protocol.

**Step 2: Run it to verify it fails**

Run: `python -m unittest tests.test_three_station_yearly_protocol_ast -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
- Add a protocol switch that loads the new yearly assets
- Keep the old seasonal path intact but separate
- Ensure training-side extreme support and test-side extreme eval come from explicit yearly splits.

**Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_three_station_yearly_protocol_ast -v`
Expected: PASS.

### Task 4: Add new baselines for yearly extreme-stage comparison

**Files:**
- Modify: `DemoModelTraining.py`
- Test: `tests/test_extreme_stage_baselines_ast.py`

**Step 1: Write the failing test**
- Assert training/eval can distinguish:
  - `LMT-new`
  - `Extreme-FedAvg`
  - `Proposed-A`

**Step 2: Run it to verify it fails**

Run: `python -m unittest tests.test_extreme_stage_baselines_ast -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
- Keep local-only path for `LMT-new`
- Make all three yearly methods share `local_pretrain -> local_meta`
- Add actual same-class extreme local updates before server aggregation
- Add plain same-class `FedAvg`
- Add target-conditioned weighted aggregation for `Proposed-A`

**Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_extreme_stage_baselines_ast -v`
Expected: PASS.

### Task 5: Implement scenario prototype extraction

**Files:**
- Modify: `DemoModelTraining.py`
- Test: `tests/test_extreme_prototype_features_ast.py`

**Step 1: Write the failing test**
- Assert a function exists that builds `phi(x,y)` with the agreed components.

**Step 2: Run it to verify it fails**

Run: `python -m unittest tests.test_extreme_prototype_features_ast -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
- Add `phi(x,y)` using the validated statistics
- Normalize features in-client before distance use
- Add class prototype computation.

**Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_extreme_prototype_features_ast -v`
Expected: PASS.

### Task 6: Implement reliability-aware weighting

**Files:**
- Modify: `DemoModelTraining.py`
- Test: `tests/test_extreme_weighted_aggregation_ast.py`

**Step 1: Write the failing test**
- Assert weight construction uses `m`, `q`, `sim` and normalizes to `alpha`.

**Step 2: Run it to verify it fails**

Run: `python -m unittest tests.test_extreme_weighted_aggregation_ast -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
- Add:
  - sample-count reliability
  - query-loss reliability
  - scenario similarity
- Use defaults `tau=1`, `lambda=1`, `mu=1`, `nu=2`
- Aggregate per `(target_station, extreme_class)`.

**Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_extreme_weighted_aggregation_ast -v`
Expected: PASS.

### Task 7: Extend result export to new yearly protocol

**Files:**
- Modify: `generate_multi_station_results.py`
- Test: `tests/test_yearly_extreme_results_ast.py`

**Step 1: Write the failing test**
- Assert results can export task-level metrics for yearly extreme protocol and new models.

**Step 2: Run it to verify it fails**

Run: `python -m unittest tests.test_yearly_extreme_results_ast -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
- Export task-level rows by `(station, extreme_class, model)`
- Keep the same metric columns:
  - `nMAE_%`
  - `nRMSE_%`
  - `WD_%`
  - `R_p<0.05_%`

**Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_yearly_extreme_results_ast -v`
Expected: PASS.

### Task 8: Run CPU smoke validation only

**Files:**
- Modify: none or minimal launcher/config if needed

**Step 1: Build yearly assets**

Run: `python build_three_station_extreme_yearly_protocol.py`
Expected: yearly protocol assets generated.

**Step 2: Run a short CPU smoke train**

Run: use `1/1/1` or similarly tiny budget under the new yearly protocol.
Expected: end-to-end chain completes.

**Step 3: Run result export**

Run: `python generate_multi_station_results.py`
Expected: task-level result CSV generated for the new protocol.

**Step 4: Verify non-formal status**
- Record that this is `CPU smoke / debug validation`, not `4090 formal run`.

### Task 9: Prepare 4090 formal run handoff

**Files:**
- Modify: `AGENTS.md` if needed for the new protocol decision log

**Step 1: Freeze formal preset recommendations**
- Keep CPU smoke and 4090 formal runs explicitly separated.

**Step 2: Write exact 4090 commands**
- Provide build/train/eval commands for the yearly protocol.

**Step 3: Define success checks**
- Confirm result CSV exists
- Confirm each model path is populated
- Confirm task-level metrics are generated for the intended station/class tasks.
