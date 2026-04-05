# Seasonal Launcher Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reproducible launcher for the six-client seasonal scarcity protocol that can build assets, launch training, and run evaluation with a consistent env preset.

**Architecture:** Keep orchestration out of `DemoModelTraining.py`. Introduce one small Python launcher that composes subprocess commands and seasonal env presets. Keep legacy direct script execution untouched.

**Tech Stack:** Python stdlib (`argparse`, `os`, `subprocess`), unittest AST checks

---

### Task 1: Lock launcher surface with a failing AST test

**Files:**
- Create: `tests/test_seasonal_protocol_launcher_ast.py`
- Create: `run_six_client_seasonal_protocol.py`

**Step 1: Write the failing test**
- Assert the launcher file exists.
- Assert it contains stage names `build`, `train`, `eval`, `all`.
- Assert it exports `SEASONAL_PROTOCOL_ENABLED`, `SEASONAL_PROTOCOL_METADATA_PATH`, `META_SUPPORT_SHOTS`, `META_QUERY_SHOTS`.
- Assert it supports `--smoke` and `--dry-run`.

**Step 2: Run test to verify it fails**
Run:
```bash
python -m unittest tests.test_seasonal_protocol_launcher_ast -v
```
Expected: FAIL because the launcher does not exist yet.

**Step 3: Write minimal implementation**
- Create the launcher with argparse surface only.

**Step 4: Run test to verify it passes**
Run the same command and expect PASS.

### Task 2: Implement seasonal env preset and subprocess orchestration

**Files:**
- Modify: `run_six_client_seasonal_protocol.py`
- Test: `tests/test_seasonal_protocol_launcher_ast.py`

**Step 1: Write the failing test**
- Assert the launcher builds commands for `build_six_client_seasonal_protocol.py`, `DemoModelTraining.py`, and `generate_multi_station_results.py`.
- Assert `all` runs stages in build → train → eval order.
- Assert smoke mode injects `PRETRAIN_EPOCHS=1`, `PROPOSED_META_EPOCHS=1`, `META_ONLY_META_EPOCHS=1`, `FEW_SHOT_EPOCHS=1`, `STRICT_PAPER_ORDER=0`.

**Step 2: Run test to verify it fails**
Run:
```bash
python -m unittest tests.test_seasonal_protocol_launcher_ast -v
```
Expected: FAIL because orchestration details are still missing.

**Step 3: Write minimal implementation**
- Add `build_env`, `build_stage_commands`, and subprocess execution.
- Add `--dry-run` to print commands instead of running them.

**Step 4: Run test to verify it passes**
Run the same command and expect PASS.

### Task 3: Validate launcher smoke path

**Files:**
- Modify: `seasonal_protocol_data/README.md`

**Step 1: Run syntax and launcher dry-run verification**
Run:
```bash
python -m py_compile run_six_client_seasonal_protocol.py
python run_six_client_seasonal_protocol.py all --smoke --dry-run
```
Expected: PASS and print build/train/eval commands with seasonal env preset.

**Step 2: Document usage**
- Add a short usage block to `seasonal_protocol_data/README.md` covering full run and smoke dry-run.
