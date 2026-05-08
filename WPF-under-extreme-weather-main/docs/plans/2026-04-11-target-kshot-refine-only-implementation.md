# Target K-Shot Refine-Only Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a refine-only stress protocol that reuses completed 2h/6-point pretrain/meta checkpoints while limiting only target-station extreme adapt windows.

**Architecture:** Add an environment-controlled target adapt cap in `DemoModelTraining.py`. The cap is applied after the target extreme adapt/validation split and before tensor conversion; source-station screening and source splits remain unchanged. Add launcher presets for K=1 and K=2 that set `SKIP_LOCAL_PRETRAIN=1` and `SKIP_LOCAL_META=1`.

**Tech Stack:** Python, PyTorch training script, unittest AST tests.

---

### Task 1: Lock Behavior With AST Test

**Files:**
- Create: `tests/test_target_kshot_refine_ast.py`

**Steps:**
1. Add tests asserting `EXTREME_TARGET_ADAPT_MAX_WINDOWS` exists.
2. Add tests asserting `apply_target_adapt_kshot_limit()` is called for `target_split_payload` only.
3. Add tests asserting launcher exposes `refine-k1` and `refine-k2`.
4. Run `python -m unittest tests.test_target_kshot_refine_ast -v` and verify failure before implementation.

### Task 2: Implement Target-Only Adapt Cap

**Files:**
- Modify: `DemoModelTraining.py`

**Steps:**
1. Add `EXTREME_TARGET_ADAPT_MAX_WINDOWS = int(os.getenv(..., "0"))`.
2. Add `apply_target_adapt_kshot_limit(split_payload)`.
3. If max windows is disabled or adapt window count is already within cap, return the split unchanged.
4. If adapt has more than K windows, keep the first K adapt windows and move the remaining adapt windows into validation so target information is not discarded from weighting diagnostics.
5. Call this function only for `target_split_payload`.
6. Export the setting in the convergence report.

### Task 3: Add Refine-Only Launcher Presets

**Files:**
- Modify: `run_three_station_yearly_protocol.py`

**Steps:**
1. Add preset names `refine-k1` and `refine-k2`.
2. Add `EXTREME_TARGET_ADAPT_MAX_WINDOWS` to preview env keys.
3. For both presets, set `SKIP_LOCAL_PRETRAIN=1`, `SKIP_LOCAL_META=1`, `FEW_SHOT_EPOCHS=200`, and `EXTREME_TARGET_REFINEMENT_EPOCHS=200`.
4. Set K to 1 or 2 depending on preset.
5. Include presets in argparse choices.

### Task 4: Verify

**Commands:**
- `python -m unittest tests.test_target_kshot_refine_ast -v`
- `python -m unittest tests.test_training_protocol_config_ast tests.test_2h_6point_launcher_ast tests.test_target_kshot_refine_ast -v`
- `python run_three_station_yearly_protocol.py train --preset refine-k1 --dry-run`

**Manual full run command for user GPU:**
- `python -u run_three_station_yearly_protocol.py train --preset refine-k1 2>&1 | tee logs/2h6p_refine_k1_train_$(date +%Y%m%d_%H%M%S).log`
- `python -u run_three_station_yearly_protocol.py eval --preset refine-k1 2>&1 | tee logs/2h6p_refine_k1_eval_$(date +%Y%m%d_%H%M%S).log`
