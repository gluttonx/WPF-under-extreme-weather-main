# 4090 Screen Runtime Contract Design

**Goal:** Standardize the three-station yearly formal run path so `screen` sessions show real-time progress, the launcher exposes smoke/pilot/formal budgets, and the training script emits consistent phase/epoch/convergence logs.

**Context:** The current codebase already has `progress_log(...)`, `should_log_epoch(...)`, convergence summaries, and a six-client seasonal launcher. What is still missing is a single runtime contract for the current mainline workflow: the three-station yearly protocol for `LMT-new`, `Extreme-FedAvg`, and `Proposed-A` on remote `RTX 4090`.

## Problem

The operational path for the current yearly workflow is still split across ad-hoc commands and old seasonal documentation:

- there is no dedicated yearly launcher for `build -> train -> eval`
- the user wants to run inside `screen` and see live stage/epoch progress
- current logs are partially standardized, but stage banners and log interval knobs are not formalized for the yearly formal path
- the formal budget is large (`35000 / 30000 / 50`), so the workflow also needs an explicit `pilot` tier rather than assuming every run should immediately use paper-scale epochs

## Approaches considered

### Approach A: Document shell commands only

Document the recommended `screen` command and leave the yearly workflow as raw environment variables plus direct Python calls.

- Pros: smallest change.
- Cons: still error-prone, no single launcher surface, and budget tiers remain implicit.

### Approach B: Add a dedicated yearly launcher and formalize log/runtime policy

Add `run_three_station_yearly_protocol.py`, make it force unbuffered Python execution, expose `smoke / pilot / pilot-medium / formal-v1` presets, and tighten the training log contract around stage banners, interval knobs, and immediate convergence announcements.

- Pros: operationally reproducible, aligned with current three-station mainline, minimal algorithm risk.
- Cons: one more entrypoint to maintain.

### Approach C: Fold launcher logic into `DemoModelTraining.py`

- Pros: fewer files.
- Cons: mixes orchestration and training logic, hurts maintainability, and is inconsistent with the existing seasonal pattern.

## Selected design

Use **Approach B**.

### Launcher

Add `run_three_station_yearly_protocol.py` with stages:

- `build`
- `train`
- `eval`
- `all`

The launcher will:

- force `PYTHONUNBUFFERED=1`
- invoke child stages with `python -u`
- pin yearly protocol env (`YEARLY_PROTOCOL_ENABLED=1`, `SEASONAL_PROTOCOL_ENABLED=0`)
- default to the current yearly baseline path (`TRAIN_META_ONLY_BASELINE=0`)
- expose four presets:
  - `smoke`
  - `pilot`
  - `pilot-medium`
  - `formal-v1`
- support `--dry-run`
- print a clear per-stage banner before launching each subprocess

### Training/runtime logging

Keep the existing helpers and extend them minimally:

- `progress_log(...)` remains the single flush-safe logger
- add `log_stage_banner(...)` to standardize readable phase headers
- add env-configurable interval knobs:
  - `PRETRAIN_LOG_INTERVAL`
  - `META_LOG_INTERVAL`
  - `FEW_SHOT_LOG_INTERVAL`
- preserve the logging rhythm:
  - first 10 epochs: every epoch
  - then every 50 or 100 epochs, depending on preset/env
  - last epoch: always log
- when convergence is first detected, print an immediate line with stage id, convergence epoch, best epoch, and best loss
- retain the end-of-stage convergence summary/report export

### Budget tiers

The runtime contract should explicitly encode:

- `smoke`: pipeline correctness only
- `pilot`: conservative directional validation on 4090
- `pilot-medium`: medium-budget directional validation on 4090
- `formal`: paper-scale budget only after protocol and metrics are stable

This directly addresses the concern that development should not default to tens of thousands of epochs.

### Documentation and memory

Update `AGENTS.md` in the runtime/4090 area with a dated `4090 screen runtime contract` section that states:

- use the yearly launcher for formal runs
- `screen + tee` is supported through unbuffered output
- the log rhythm and convergence announcement policy
- `smoke / pilot / formal` budget semantics

Persist the same contract into long-term memory for future session bootstrap.

## Non-goals

- No training algorithm change
- No new evaluation metric change
- No automatic screen session creation from Python
- No forced GPU selection in code; `CUDA_VISIBLE_DEVICES` stays user-controlled

## Validation

- AST tests for the yearly launcher surface
- AST tests for new runtime logging helpers/knobs
- AST test for `AGENTS.md` runtime contract text
- `py_compile` for touched Python files
- `python run_three_station_yearly_protocol.py all --smoke --dry-run`
- focused `unittest` run for new/updated AST tests
