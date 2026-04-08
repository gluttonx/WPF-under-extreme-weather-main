# LMT-Based Extreme FL Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `Extreme-FedAvg` and `Proposed-A` on top of the restored `LMT` baseline, while keeping the main CSV in the prior three-model `fb2c67a` format.

**Architecture:** Preserve the restored `local_pretrain -> local_meta -> LMT few-shot` path. Add target-conditioned extreme FL only inside the class-specific few-shot stage, with two-source effective-window screening, shared-init local updates, uniform aggregation for `Extreme-FedAvg`, and reliability-aware aggregation for `Proposed-A`.

**Tech Stack:** Python, PyTorch, NumPy, SciPy `.mat`, pandas, unittest.

---

### Task 1: Lock the three-model contract

**Files:**
- Modify: `tests/test_localized_lmt_ast.py`
- Create: `tests/test_extreme_fl_contract_ast.py`

**Step 1: Write the failing test**
- Assert `generate_multi_station_results.py` exports only:
  - `LMT`
  - `Extreme-FedAvg`
  - `Proposed-A`
- Assert `DemoModelTraining.py` declares explicit save-path helpers for the two new extreme FL branches.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_localized_lmt_ast tests.test_extreme_fl_contract_ast -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
- Add the new helper names and reduce result-model order to the three required methods.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_localized_lmt_ast tests.test_extreme_fl_contract_ast -v`
Expected: PASS.

### Task 2: Implement effective-window screening and shared-init updates

**Files:**
- Modify: `DemoModelTraining.py`
- Test: `tests/test_extreme_fl_contract_ast.py`

**Step 1: Write the failing test**
- Assert helpers exist for:
  - extreme window extraction
  - `adapt / val` split
  - source-quality gate
  - target-conditioned usefulness
  - shared-init source update
  - uniform vs weighted aggregation

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_extreme_fl_contract_ast -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
- Keep `LMT` untouched.
- Add `Extreme-FedAvg` and `Proposed-A` paths.
- Start both from the same target-station `local_meta` initialization.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_extreme_fl_contract_ast -v`
Expected: PASS.

### Task 3: Update export and validation

**Files:**
- Modify: `generate_multi_station_results.py`
- Modify: `tests/test_localized_lmt_ast.py`

**Step 1: Write the failing test**
- Assert the main CSV keeps the old wide-table format but only for the three active methods.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_localized_lmt_ast -v`
Expected: FAIL.

**Step 3: Write minimal implementation**
- Add model resolution for:
  - `LMT`
  - `Extreme-FedAvg`
  - `Proposed-A`
- Remove `Meta_Learning` and `Pre_Training` rows from the exported main table.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_localized_lmt_ast -v`
Expected: PASS.

### Task 4: Verify with smoke

**Files:**
- No new files

**Step 1: Run tests**

Run: `python -m unittest tests.test_localized_lmt_ast tests.test_extreme_fl_contract_ast tests.test_runtime_device_ast tests.test_few_shot_loss_ast -v`
Expected: PASS.

**Step 2: Run syntax verification**

Run: `python -m py_compile DemoModelTraining.py generate_multi_station_results.py`
Expected: PASS.

**Step 3: Run smoke**

Run: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PRETRAIN_EPOCHS=1 PROPOSED_META_EPOCHS=1 META_ONLY_META_EPOCHS=1 FEW_SHOT_EPOCHS=1 TRAIN_META_ONLY_BASELINE=0 python DemoModelTraining.py`

Run: `python generate_multi_station_results.py`

Expected: PASS and produce a three-model `multi_station_performance.csv`.
