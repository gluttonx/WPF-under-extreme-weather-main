# Runtime Log Visibility Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `screen + tee` seasonal runs display real-time stage and epoch progress by forcing unbuffered launcher execution and adding flush-safe progress logging in `DemoModelTraining.py`.

**Architecture:** Update the launcher so every child Python stage runs unbuffered, then add a tiny progress logging layer to `DemoModelTraining.py` for stage banners and long-loop epoch logs. Keep training semantics unchanged.

**Tech Stack:** Python 3.8, `argparse`, `subprocess`, PyTorch training script, `unittest` AST/text tests.

---

### Task 1: Add failing launcher logging tests

**Files:**
- Modify: `WPF-under-extreme-weather-main/tests/test_seasonal_protocol_launcher_ast.py`

**Step 1: Write the failing test**
- Assert launcher text contains `PYTHONUNBUFFERED`.
- Assert stage commands include `"-u"` before script names.

**Step 2: Run test to verify it fails**
- Run: `python -m unittest WPF-under-extreme-weather-main/tests/test_seasonal_protocol_launcher_ast.py -v`

**Step 3: Write minimal implementation**
- Update launcher env and command construction.

**Step 4: Run test to verify it passes**
- Run the same unittest command.

### Task 2: Add failing training log visibility tests

**Files:**
- Create: `WPF-under-extreme-weather-main/tests/test_runtime_log_visibility_ast.py`
- Modify: `WPF-under-extreme-weather-main/DemoModelTraining.py`

**Step 1: Write the failing test**
- Assert script contains `def progress_log(`.
- Assert script contains `flush=True`.
- Assert script contains `def should_log_epoch(`.
- Assert script uses `should_log_epoch(` in long training loops.

**Step 2: Run test to verify it fails**
- Run: `python -m unittest WPF-under-extreme-weather-main/tests/test_runtime_log_visibility_ast.py -v`

**Step 3: Write minimal implementation**
- Add helpers and route key progress prints through them.

**Step 4: Run test to verify it passes**
- Run the same unittest command.

### Task 3: Verify syntax and smoke behavior

**Files:**
- Modify: `WPF-under-extreme-weather-main/run_six_client_seasonal_protocol.py`
- Modify: `WPF-under-extreme-weather-main/DemoModelTraining.py`

**Step 1: Run compile verification**
- Run: `python -m py_compile WPF-under-extreme-weather-main/run_six_client_seasonal_protocol.py WPF-under-extreme-weather-main/DemoModelTraining.py`

**Step 2: Run CPU smoke validation**
- Run: `cd WPF-under-extreme-weather-main && python run_six_client_seasonal_protocol.py all --smoke`
- Confirm stage headers and epoch lines appear during execution rather than being delayed until stage end.

### Task 4: Update records

**Files:**
- Modify: `WPF-under-extreme-weather-main/AGENTS.md`

**Step 1: Document runtime log visibility rule**
- Add a new `##` section under the runtime/convergence area describing unbuffered launcher execution and flush-safe progress logging.

**Step 2: Save long-term memory**
- Save a high-priority decision note stating that seasonal formal runs should use unbuffered logging so `screen + tee` shows phase/epoch progress in real time.
