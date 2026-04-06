# Yearly Clean Extreme Ablation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rework the yearly extreme ablation so all three methods share `local_pretrain -> local_meta`, differ only in extreme-stage collaboration, and export a paper-style `TABLE IV` CSV plus a raw task-level CSV.

**Architecture:** Update yearly baseline routing in `DemoModelTraining.py` so all methods initialize from per-station `local_meta` weights. Add explicit same-class extreme local adaptation + server aggregation helpers for plain FedAvg and reliability-aware weighted aggregation. Update `generate_multi_station_results.py` to preserve raw task-level rows and build a paper-facing wide table by recomputing metrics from concatenated event predictions.

**Tech Stack:** Python, PyTorch, NumPy, pandas, unittest, existing TensorBoard duration inference.

---

### Task 1: Lock the new yearly contract with failing tests

**Files:**
- Modify: `tests/test_extreme_stage_baselines_ast.py`
- Modify: `tests/test_yearly_extreme_results_ast.py`
- Create: `tests/test_yearly_extreme_aggregation_ast.py`

**Step 1: Write the failing test**
- Assert yearly baseline routing no longer falls back to `PRETRAIN_MODEL_PATH` or proposed federated meta for `Extreme-FedAvg` / `Proposed-A`.
- Assert `DemoModelTraining.py` declares helpers for same-class extreme aggregation.
- Assert `generate_multi_station_results.py` writes both `multi_station_performance.csv` and `multi_station_performance_task_level.csv`.
- Assert yearly wide output contains `HighWind_E_M_%`, `Training_duration_s`, and `R_p<0.05_%`.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_extreme_stage_baselines_ast tests.test_yearly_extreme_results_ast tests.test_yearly_extreme_aggregation_ast -v`
Expected: FAIL because the current implementation still routes yearly methods through federated init and lacks wide-table helpers.

**Step 3: Write minimal implementation**
- Only after seeing RED, update the production code to satisfy the new contract.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_extreme_stage_baselines_ast tests.test_yearly_extreme_results_ast tests.test_yearly_extreme_aggregation_ast -v`
Expected: PASS.

### Task 2: Implement clean yearly baseline routing and extreme-stage aggregation

**Files:**
- Modify: `DemoModelTraining.py`

**Step 1: Write the failing test**
- Add assertions that yearly baselines use per-station `local_meta` base models.
- Add assertions for plain and weighted aggregation helper definitions and use inside yearly few-shot loop.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_extreme_stage_baselines_ast tests.test_yearly_extreme_aggregation_ast -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
- Route all yearly baseline specs through `get_local_meta_model_path(station_id)`.
- Add helper(s) to:
  - fine-tune one client on one extreme class and return tuned state + loss proxy;
  - aggregate tuned states uniformly for `Extreme-FedAvg`;
  - aggregate tuned states by reliability-aware weights for `Proposed-A`.
- Save yearly personalized models to the existing filenames.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_extreme_stage_baselines_ast tests.test_yearly_extreme_aggregation_ast -v`
Expected: PASS.

### Task 3: Implement yearly TABLE IV export and raw task-level export

**Files:**
- Modify: `generate_multi_station_results.py`

**Step 1: Write the failing test**
- Assert yearly evaluation exposes helpers for task-level export, paper-table aggregation, and yearly duration mapping.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_yearly_extreme_results_ast -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
- Keep raw yearly rows in `multi_station_performance_task_level.csv`.
- Build `multi_station_performance.csv` as a `TABLE IV`-style wide table.
- Recompute metrics from concatenated events rather than averaging percentages.
- Add yearly duration mapping for `LMT-new`, `Extreme-FedAvg`, and `Proposed-A`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_yearly_extreme_results_ast -v`
Expected: PASS.

### Task 4: Verify syntax and launcher compatibility

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/plans/2026-04-05-extreme-stage-weighted-fl-design.md`
- Modify: `docs/plans/2026-04-05-extreme-stage-weighted-fl-implementation.md`

**Step 1: Write the failing test**
- Add AST assertions only if the docs need a contract check; otherwise skip test creation and rely on direct verification.

**Step 2: Run verification commands**

Run: `python -m py_compile DemoModelTraining.py generate_multi_station_results.py run_three_station_yearly_protocol.py`
Expected: PASS.

Run: `python run_three_station_yearly_protocol.py all --smoke --dry-run`
Expected: PASS and show yearly build/train/eval commands with updated contract.

**Step 3: Update docs**
- Record the clean ablation definition in `AGENTS.md`.
- Update the 2026-04-05 design/implementation notes so future sessions do not inherit the older federated-init interpretation.

### Task 5: Final verification

**Files:**
- No new files

**Step 1: Run targeted tests**

Run: `python -m unittest tests.test_extreme_stage_baselines_ast tests.test_yearly_extreme_results_ast tests.test_yearly_extreme_aggregation_ast -v`
Expected: PASS.

**Step 2: Run syntax verification**

Run: `python -m py_compile DemoModelTraining.py generate_multi_station_results.py`
Expected: PASS.

**Step 3: Run launcher dry-run**

Run: `python run_three_station_yearly_protocol.py all --preset smoke --dry-run`
Expected: PASS.

**Step 4: Summarize actual status**
- Report exactly which checks passed.
- Explicitly state that old pilot/pilot-medium results are no longer valid under the new definition.
