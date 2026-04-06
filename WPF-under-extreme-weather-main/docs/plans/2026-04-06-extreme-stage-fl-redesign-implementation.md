# Extreme-Stage FL Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current prototype-similarity-heavy extreme-stage aggregation with a shared-init, effective-window-screened, target-conditioned FL design that is appropriate for the `58 / 59 / 60` yearly clean ablation.

**Architecture:** Keep the yearly backbone fixed as `local_pretrain -> local_meta -> extreme adaptation`. Remove the old `sim_{k->s}^c`-centered weighting path from `Proposed-A`, introduce `adapt / val` splits for extreme-stage local adaptation, add source-quality gate + target-conditioned usefulness screening, and make `Extreme-FedAvg` / `Proposed-A` share the same screened source pools and shared initialization. Preserve current evaluation outputs: `multi_station_performance.csv` as `Station x Model` TABLE IV-style wide table and `multi_station_performance_task_level.csv` as the raw 36-row task-level table.

**Tech Stack:** Python, PyTorch, NumPy, SciPy `.mat`, pandas, unittest, existing yearly launcher and TensorBoard duration inference.

---

### Task 1: Lock the redesign contract with failing AST tests

**Files:**
- Modify: `tests/test_extreme_stage_baselines_ast.py`
- Modify: `tests/test_extreme_weighted_aggregation_ast.py`
- Create: `tests/test_extreme_window_screening_ast.py`
- Create: `tests/test_extreme_adapt_val_split_ast.py`

**Step 1: Write the failing test**
- Assert `DemoModelTraining.py` no longer describes extreme-stage splits as `support/query` for the yearly redesign path.
- Assert the training script declares helpers or symbols for:
  - `adapt / val` split construction
  - source-quality gate
  - target-conditioned usefulness screening
  - shared-init source updates
  - `beta_self`
  - transferability term `t_{k->s,c}`
- Assert `Extreme-FedAvg` and `Proposed-A` both consume screened effective-window sets, while differing only in aggregation rule.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_extreme_stage_baselines_ast tests.test_extreme_weighted_aggregation_ast tests.test_extreme_window_screening_ast tests.test_extreme_adapt_val_split_ast -v`
Expected: FAIL because the current implementation still uses the old reliability-aware path and does not expose the new helpers.

**Step 3: Write minimal implementation**
- Do not touch production code until the RED failure is observed.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_extreme_stage_baselines_ast tests.test_extreme_weighted_aggregation_ast tests.test_extreme_window_screening_ast tests.test_extreme_adapt_val_split_ast -v`
Expected: PASS.

### Task 2: Implement extreme-stage `adapt / val` split builders

**Files:**
- Modify: `DemoModelTraining.py`
- Test: `tests/test_extreme_adapt_val_split_ast.py`

**Step 1: Write the failing test**
- Assert a dedicated helper exists for building target-side `D_{s,c}^{adapt}` and `D_{s,c}^{val}` from yearly extreme windows.
- Assert the helper includes the single-window `12-step horizon` fallback when `|D_{s,c}| = 1`.
- Assert source-side effective-window sets can also be split into `adapt / val` when enough windows remain.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_extreme_adapt_val_split_ast -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
- Add explicit split builders for:
  - target extreme windows
  - screened source effective windows
- Keep the implementation deterministic and lightweight.
- Do not reuse meta-learning terminology in names or logs.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_extreme_adapt_val_split_ast -v`
Expected: PASS.

### Task 3: Implement source-quality gate and usefulness screening

**Files:**
- Modify: `DemoModelTraining.py`
- Test: `tests/test_extreme_window_screening_ast.py`

**Step 1: Write the failing test**
- Assert the training script exposes helpers for:
  - source-quality score `e_{k,c}^{self}(w)`
  - `Q_{k,c}` gate construction
  - target-conditioned usefulness score `u_{k->s,c}(w)`
  - effective-window selection `E_{k->s,c}`
  - budget control via `B_min` and `gamma`

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_extreme_window_screening_ast -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
- Implement source-quality gate using local-meta prediction error on each source window.
- Implement usefulness screening using target validation loss improvement under shared initialization.
- Ensure both non-target source stations can contribute windows.
- Ensure windows with non-positive usefulness do not enter `E_{k->s,c}`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_extreme_window_screening_ast -v`
Expected: PASS.

### Task 4: Replace the old weighting path with shared-init extreme FL

**Files:**
- Modify: `DemoModelTraining.py`
- Modify: `tests/test_extreme_weighted_aggregation_ast.py`
- Modify: `tests/test_yearly_extreme_aggregation_ast.py`

**Step 1: Write the failing test**
- Assert `theta_{s,c}^{(0)} = theta_s^{meta}` is the shared initialization anchor for both target and source updates.
- Assert the source update path no longer initializes from each source station’s own `local_meta`.
- Assert `Extreme-FedAvg` uses uniform aggregation over target self update plus all available screened source updates.
- Assert `Proposed-A` uses `m`, `q`, `t`, `beta_self`, and normalized `tilde_alpha`.
- Assert `sim_{k->s}^c` / prototype-similarity no longer drives the main yearly weighting path.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_extreme_weighted_aggregation_ast tests.test_yearly_extreme_aggregation_ast -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
- Remove the old prototype-similarity-centered yearly weighting path from `Proposed-A`.
- Implement:
  - target self update from shared init
  - source updates from the same shared init
  - `Extreme-FedAvg` aggregation
  - `Proposed-A` weighted aggregation with:
    - `m_{k->s,c} = log(1 + |E_{k->s,c}^{adapt}|)`
    - `q_{k->s,c} = exp(-tau_q * L(theta_{k->s,c}^{(1)} ; E_{k->s,c}^{val}))` when source val exists
    - `t_{k->s,c} = exp(-tau_t * L(theta_{k->s,c}^{(1)} ; D_{s,c}^{val}))`
    - `a_{k->s,c} = (m^lambda)(q^mu)(t^nu)`
    - `theta_{s,c}^{agg-prop} = beta_self * theta_{s->s,c}^{(1)} + (1 - beta_self) * sum_k tilde_alpha_{k->s,c} theta_{k->s,c}^{(1)}`
- Add a short target refinement step after aggregation for both `Extreme-FedAvg` and `Proposed-A`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_extreme_weighted_aggregation_ast tests.test_yearly_extreme_aggregation_ast -v`
Expected: PASS.

### Task 5: Add diagnostics and logging for the redesign

**Files:**
- Modify: `DemoModelTraining.py`
- Create: `tests/test_extreme_logging_contract_ast.py`

**Step 1: Write the failing test**
- Assert the yearly extreme path logs:
  - target `adapt / val` sizes
  - each source `|P|`, `|Q|`, `|E|`
  - `m`, `q`, `t`
  - `beta_self`
  - source weights for `Extreme-FedAvg` and `Proposed-A`
  - target refinement summary

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_extreme_logging_contract_ast -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
- Add structured `progress_log(...)` lines at the yearly few-shot stage only.
- Keep logging concise enough for `screen`, but sufficient for diagnosing why `Proposed-A` wins or loses.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_extreme_logging_contract_ast -v`
Expected: PASS.

### Task 6: Keep evaluation outputs stable while updating method semantics

**Files:**
- Modify: `generate_multi_station_results.py`
- Modify: `tests/test_yearly_extreme_results_ast.py`

**Step 1: Write the failing test**
- Assert the yearly evaluation script still exports:
  - `multi_station_performance.csv`
  - `multi_station_performance_task_level.csv`
- Assert the main wide table still includes `Station`, `Model`, four weather blocks, `Training_duration_s`, and `R_p<0.05_%`.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_yearly_extreme_results_ast -v`
Expected: FAIL only if output schema drifted during redesign.

**Step 3: Write minimal implementation**
- Update evaluation only if the redesign changes model naming or intermediate output routing.
- Do not change the paper-facing CSV schema unless strictly required.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_yearly_extreme_results_ast -v`
Expected: PASS.

### Task 7: Update docs and session handoff after code changes

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/plans/2026-04-06-extreme-stage-fl-redesign-design.md`
- Create: `docs/plans/2026-04-06-extreme-stage-fl-redesign-implementation.md` (this file, update status notes if needed)

**Step 1: Write the failing test**
- No AST test required for docs unless a contract is added to code.

**Step 2: Update docs**
- After implementation, sync:
  - actual helper names
  - actual default hyperparameters
  - any fallback behavior for tiny-window classes
  - any deviations from the design doc

**Step 3: Update long-term handoff**
- Save the final redesign summary and current execution state to long-term memory after the code lands.

### Task 8: Verification ladder after implementation

**Files:**
- No new files

**Step 1: Run targeted tests**

Run: `python -m unittest tests.test_extreme_stage_baselines_ast tests.test_extreme_weighted_aggregation_ast tests.test_extreme_window_screening_ast tests.test_extreme_adapt_val_split_ast tests.test_yearly_extreme_aggregation_ast tests.test_yearly_extreme_results_ast tests.test_extreme_logging_contract_ast -v`
Expected: PASS.

**Step 2: Run syntax verification**

Run: `python -m py_compile DemoModelTraining.py generate_multi_station_results.py run_three_station_yearly_protocol.py`
Expected: PASS.

**Step 3: Run yearly smoke**

Run: `python run_three_station_yearly_protocol.py all --preset smoke 2>&1 | tee logs/yearly_smoke_redesign_$(date +%Y%m%d_%H%M%S).log`
Expected: PASS with `build -> train -> eval` completed and redesigned yearly logs visible.

**Step 4: Run clean pilot-medium**

Run: `python run_three_station_yearly_protocol.py all --preset pilot-medium 2>&1 | tee logs/yearly_pilot_medium_redesign_$(date +%Y%m%d_%H%M%S).log`
Expected: PASS and produce redesigned task-level + wide-table results.

**Step 5: Decision gate**
- Only advance to `formal-v1` if:
  - `Proposed-A` now beats `Extreme-FedAvg` on the main error metrics in a stable way
  - the diagnostic logs show non-degenerate `m / q / t / alpha`
  - the redesign no longer collapses to count-only weighting.
