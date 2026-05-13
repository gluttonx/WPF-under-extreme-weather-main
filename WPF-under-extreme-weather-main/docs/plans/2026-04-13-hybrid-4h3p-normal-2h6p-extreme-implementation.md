# Hybrid 4h3p Normal 2h6p Extreme Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an explicit hybrid experiment mode that reuses 4h/3p normal-stage checkpoints and reruns only 2h/6p extreme few-shot/eval.

**Architecture:** The launcher selects a 2h/6p six-station runtime protocol and a separate hybrid artifact output directory, while `BASE_MODEL_OUTPUT_DIR` points to the existing 4h/3p base checkpoint directory. The training script reads base checkpoints from `BASE_MODEL_OUTPUT_DIR` and writes personalized extreme checkpoints to `MODEL_OUTPUT_DIR`.

**Tech Stack:** Python, PyTorch, scipy `.mat`, pandas CSV, unittest AST contract tests.

---

### Task 1: Contract Tests

**Files:**
- Modify: `tests/test_2h_6point_launcher_ast.py`
- Modify: `tests/test_training_protocol_config_ast.py`

**Steps:**
1. Add launcher assertions for `--hybrid-extreme-2h`, hybrid protocol/artifact constants, `BASE_MODEL_OUTPUT_DIR`, and skip flags.
2. Add training assertions for `BASE_MODEL_OUTPUT_DIR`, `resolve_base_model_path`, base checkpoint helper usage, one-window full-window split behavior, and non-finite few-shot guards.
3. Run focused tests and verify they fail before implementation.

### Task 2: Runtime Implementation

**Files:**
- Modify: `DemoModelTraining.py`
- Modify: `run_three_station_yearly_protocol.py`

**Steps:**
1. Add `BASE_MODEL_OUTPUT_DIR` and use it for local/Fed-Normal-Meta base checkpoint helpers.
2. Keep extreme personalized model paths on `MODEL_OUTPUT_DIR`.
3. Add a guard that rejects `BASE_MODEL_OUTPUT_DIR` unless normal-stage checkpoints are being skipped.
4. Add `--hybrid-extreme-2h` launcher mode with 2h/6p protocol, hybrid artifact root, base model dir, and skip flags.
5. Change one-window extreme split to use the full window for adaptation with empty validation.
6. Add non-finite loss/state guards in `adapt_state_dict`.

### Task 3: Verification

**Commands:**
- `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_2h_6point_launcher_ast tests.test_training_protocol_config_ast -v`
- `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_*.py' -v`
- `PYTHONDONTWRITEBYTECODE=1 python -m py_compile DemoModelTraining.py run_three_station_yearly_protocol.py generate_multi_station_results.py build_three_station_extreme_yearly_protocol.py`
- `PYTHONDONTWRITEBYTECODE=1 ENABLE_FED_NORMAL_META_PROPOSED=1 FED_NORMAL_META_SELF_FLOOR=0.8 FED_NORMAL_META_SAVE_BEST=1 FED_NORMAL_META_USE_BEST=1 python -u run_three_station_yearly_protocol.py train --preset pilot-5k --hybrid-extreme-2h --dry-run`
- `git diff --check`
