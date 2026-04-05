# Seasonal Formal Preset and Convergence Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a formal seasonal launcher preset and non-invasive convergence reporting across all training phases.

**Architecture:** Keep orchestration in the launcher and keep convergence monitoring inside `DemoModelTraining.py` as a reusable helper. Do not couple convergence detection to optimizer control.

**Tech Stack:** Python stdlib (`argparse`, `json`, `subprocess`), unittest AST checks, existing PyTorch training loops

---

### Task 1: Lock launcher preset and convergence symbols with failing AST tests

**Files:**
- Modify: `tests/test_seasonal_protocol_launcher_ast.py`
- Create: `tests/test_convergence_detection_ast.py`

**Step 1: Write the failing test**
- Assert launcher contains `formal-v1` and the formal budget env names.
- Assert training script contains convergence helper names and convergence report path/config symbols.

**Step 2: Run test to verify it fails**
Run:
```bash
python -m unittest tests.test_seasonal_protocol_launcher_ast tests.test_convergence_detection_ast -v
```
Expected: FAIL because preset and convergence helpers do not exist yet.

**Step 3: Write minimal implementation**
- No implementation in this task.

**Step 4: Run test to verify it still fails for expected reasons**
Run the same command and confirm only the missing preset/convergence symbols fail.

### Task 2: Add `formal-v1` preset to the launcher

**Files:**
- Modify: `run_six_client_seasonal_protocol.py`
- Modify: `seasonal_protocol_data/README.md`

**Step 1: Implement preset selection**
- Add `--preset` with at least `smoke` and `formal-v1`.
- Keep `--smoke` as compatibility shorthand.
- Ensure `formal-v1` injects the agreed seasonal budget.

**Step 2: Run tests**
Run:
```bash
python -m unittest tests.test_seasonal_protocol_launcher_ast -v
python run_six_client_seasonal_protocol.py all --preset formal-v1 --dry-run
```
Expected: PASS and print formal build/train/eval commands.

### Task 3: Add reusable convergence monitoring to all training loops

**Files:**
- Modify: `DemoModelTraining.py`
- Test: `tests/test_convergence_detection_ast.py`

**Step 1: Implement convergence helper**
- Add a reusable monitor/helper for `patience + min_delta` plateau detection.
- Add a report accumulator and JSON export.

**Step 2: Wire helper into loops**
- federated pretrain
- per-station local pretrain
- per-station meta training (`proposed`, `local_meta`, `meta_only`)
- per-station/class few-shot adaptation

**Step 3: Run tests**
Run:
```bash
python -m unittest tests.test_convergence_detection_ast -v
python -m py_compile DemoModelTraining.py
```
Expected: PASS.

### Task 4: Smoke-validate convergence report generation

**Files:**
- Modify: `AGENTS.md`

**Step 1: Run smoke flow with formal dry-run and regular smoke train**
Run:
```bash
python run_six_client_seasonal_protocol.py all --preset formal-v1 --dry-run
python run_six_client_seasonal_protocol.py all --smoke
```
Expected: dry-run prints formal preset commands; smoke run writes convergence JSON.

**Step 2: Record findings**
- Append the convergence design/result note under a `##` heading in `AGENTS.md`.
