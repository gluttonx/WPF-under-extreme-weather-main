# Runtime Log Visibility Design

**Goal:** Ensure `screen + tee` formal runs show timely, human-readable stage and epoch progress in the terminal without changing any training algorithm or budget semantics.

**Context:** The current six-client seasonal launcher works functionally, but under `screen` plus `tee` the user does not reliably see detailed stage/epoch progress while a run is active. This is caused by output buffering on piped Python stdout and by sparse progress prints in some long stages.

## Problem

The current `run_six_client_seasonal_protocol.py` launches child Python processes with normal buffered stdout. When the user runs the launcher through `screen` and pipes output into `tee`, Python output may be block-buffered rather than line-buffered. As a result, stage headers and epoch progress can appear late, making it hard to tell whether the run is healthy or where it currently is.

This is especially problematic for long `4090 formal run` jobs, where the runtime contract already requires the agent not to execute the full run locally. The remote user therefore depends on clear real-time terminal output to monitor progress.

## Approaches Considered

### Approach A: Launcher-only unbuffered stdout

Force unbuffered Python output in the launcher by setting `PYTHONUNBUFFERED=1` and changing stage commands to `python -u ...`.

- Pros: Minimal code change; directly addresses pipe buffering.
- Cons: Does not improve stage/epoch readability if training logs are too sparse.

### Approach B: Launcher unbuffered stdout + explicit progress logging

Combine unbuffered launcher execution with a small logging utility inside `DemoModelTraining.py` that always flushes and standardizes progress output for long stages.

- Pros: Fixes pipe buffering and improves visibility of stage/epoch progression; minimal behavioral risk.
- Cons: Slightly larger change set than launcher-only.

### Approach C: Progress bars (`tqdm`)

Introduce interactive progress bars for pretrain/meta/few-shot loops.

- Pros: Rich interactive feedback.
- Cons: Poor fit for `screen + tee`; tends to produce noisy or unreadable logs.

## Selected Design

Use **Approach B**.

### Launcher changes

- Add `PYTHONUNBUFFERED=1` to launcher environment.
- Invoke all stage Python scripts with `-u`.
- Keep existing stage command preview behavior.

### Training log changes

Add two minimal helpers to `DemoModelTraining.py`:

- `progress_log(message)`: wraps `print(..., flush=True)` so key progress lines always flush.
- `should_log_epoch(epoch_index, total_epochs, interval, warmup_epochs=10)`: prints every epoch during early warmup, then prints at a coarse interval, and also at the final epoch.

Apply the helper only at key visibility points:

- long stage banners
- federated/standalone pretrain epoch progress
- local pretrain stage start/end
- local meta stage start/end and epoch progress
- few-shot stage start/end and epoch progress
- convergence summary export line

### Logging policy

- Pretrain: first 10 epochs every epoch, then every 100 epochs, and last epoch.
- Local meta: preserve existing per-epoch visibility in smoke/small runs; for long runs use first 10 then every 100 epochs and last epoch.
- Few-shot: keep existing per-epoch logs because budget is small (`20` by default in formal preset).
- All key lines must flush immediately.

## Non-goals

- No algorithm change.
- No early stopping.
- No new dependencies.
- No interactive progress bar system.

## Validation

- AST/text tests verify launcher contains unbuffered execution hooks.
- AST/text tests verify `DemoModelTraining.py` contains the new progress logging helpers and key calls.
- `py_compile` for changed Python files.
- `python run_six_client_seasonal_protocol.py all --smoke` on CPU to confirm logs stream in real time.

## Runtime Contract Alignment

This change improves observability for both:

- `CPU smoke / debug validation`
- `4090 formal run`

It does not alter the rule that full-budget formal runs must be executed by the user on remote `RTX 4090`.
