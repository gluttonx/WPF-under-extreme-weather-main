# 4090 Screen Runtime Contract Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reproducible yearly launcher and formalize the 4090 `screen` runtime/logging contract for the current three-station yearly workflow.

**Architecture:** Reuse the seasonal launcher pattern for the yearly workflow, add minimal runtime-log helpers and env knobs in `DemoModelTraining.py`, then document the resulting contract in `AGENTS.md` and long-term memory.

**Tech Stack:** Python 3.8, `argparse`, `subprocess`, PyTorch, `unittest` AST tests, markdown docs, memory-keeper.

---

### Task 1: Lock the expected launcher/runtime contract with failing tests

**Files:**
- Create: `WPF-under-extreme-weather-main/tests/test_yearly_protocol_launcher_ast.py`
- Create: `WPF-under-extreme-weather-main/tests/test_4090_runtime_contract_ast.py`
- Modify: `WPF-under-extreme-weather-main/tests/test_runtime_log_visibility_ast.py`

**Step 1: Write the failing tests**

- Assert the yearly launcher file exists and exposes:
  - `build/train/eval/all`
  - `--smoke`, `--preset`, `--dry-run`
  - `pilot`, `pilot-medium`, `formal-v1`
  - yearly env names and runtime log interval envs
- Assert `DemoModelTraining.py` exposes:
  - `log_stage_banner(...)`
  - `PRETRAIN_LOG_INTERVAL`
  - `META_LOG_INTERVAL`
  - `FEW_SHOT_LOG_INTERVAL`
  - an immediate convergence-announcement string
- Assert `AGENTS.md` contains the 4090 screen runtime contract language.

**Step 2: Run tests to verify they fail**

Run:

```bash
python -m unittest \
  WPF-under-extreme-weather-main.tests.test_yearly_protocol_launcher_ast \
  WPF-under-extreme-weather-main.tests.test_runtime_log_visibility_ast \
  WPF-under-extreme-weather-main.tests.test_4090_runtime_contract_ast -v
```

Expected: failures because the launcher and contract text do not exist yet.

### Task 2: Implement the yearly launcher

**Files:**
- Create: `WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py`
- Test: `WPF-under-extreme-weather-main/tests/test_yearly_protocol_launcher_ast.py`

**Step 1: Write minimal implementation**

- Mirror the seasonal launcher structure.
- Force `PYTHONUNBUFFERED=1`.
- Use `python -u` for `build`, `train`, `eval`.
- Add `smoke / pilot / pilot-medium / formal-v1` presets.
- Add stage banners and env preview output.

**Step 2: Run targeted tests**

Run:

```bash
python -m unittest WPF-under-extreme-weather-main.tests.test_yearly_protocol_launcher_ast -v
```

Expected: pass.

### Task 3: Tighten runtime logging in the training script

**Files:**
- Modify: `WPF-under-extreme-weather-main/DemoModelTraining.py`
- Test: `WPF-under-extreme-weather-main/tests/test_runtime_log_visibility_ast.py`

**Step 1: Write minimal implementation**

- Add `log_stage_banner(...)`.
- Add `PRETRAIN_LOG_INTERVAL`, `META_LOG_INTERVAL`, `FEW_SHOT_LOG_INTERVAL`.
- Route key phase headers through `progress_log`.
- Replace hard-coded long-stage intervals with the new env knobs.
- Emit a one-time immediate convergence message when the criterion is first met.

**Step 2: Run targeted tests**

Run:

```bash
python -m unittest WPF-under-extreme-weather-main.tests.test_runtime_log_visibility_ast -v
```

Expected: pass.

### Task 4: Document the 4090 runtime contract

**Files:**
- Modify: `WPF-under-extreme-weather-main/AGENTS.md`
- Test: `WPF-under-extreme-weather-main/tests/test_4090_runtime_contract_ast.py`

**Step 1: Update the runtime section**

- Add a dated section describing:
  - `run_three_station_yearly_protocol.py`
  - `screen + tee`
  - `PYTHONUNBUFFERED=1`
  - stage/epoch/convergence logging rhythm
  - `smoke / pilot / formal` budget semantics

**Step 2: Run targeted test**

Run:

```bash
python -m unittest WPF-under-extreme-weather-main.tests.test_4090_runtime_contract_ast -v
```

Expected: pass.

### Task 5: Verify the full change set

**Files:**
- Verify: `WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py`
- Verify: `WPF-under-extreme-weather-main/DemoModelTraining.py`
- Verify: `WPF-under-extreme-weather-main/AGENTS.md`

**Step 1: Run focused test suite**

```bash
python -m unittest \
  WPF-under-extreme-weather-main.tests.test_yearly_protocol_launcher_ast \
  WPF-under-extreme-weather-main.tests.test_runtime_log_visibility_ast \
  WPF-under-extreme-weather-main.tests.test_4090_runtime_contract_ast -v
```

**Step 2: Run syntax verification**

```bash
python -m py_compile \
  WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py \
  WPF-under-extreme-weather-main/DemoModelTraining.py
```

**Step 3: Run launcher dry-run smoke**

```bash
python WPF-under-extreme-weather-main/run_three_station_yearly_protocol.py all --smoke --dry-run
```

Expected: clear stage banners and the correct env preview without starting long training.

### Task 6: Persist the decision

**Files:**
- None in repo

**Step 1: Save long-term memory**

- Store the 4090 runtime contract under a dated key for session bootstrap, including launcher path, logging policy, and the `smoke / pilot / formal` budget rule.
