# Fed-Normal-Meta Proposed-A Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a minimal Proposed-A-only Fed-Normal-Meta experiment for the six-station 2h/6p protocol.

**Architecture:** Keep LMT and Extreme-FedAvg on target local-meta initialization. Add a new target-conditioned fed-normal-meta checkpoint per station, trained from the target local-pretrain checkpoint while sampling normal-weather meta tasks from all six stations. Use that checkpoint only for Proposed-A extreme-stage screening, source updates, weighted aggregation, and target refinement.

**Tech Stack:** Python, PyTorch, scipy `.mat` assets, unittest AST/contract tests, existing launcher env presets.

---

### Task 1: Add Training Contract Tests

**Files:**
- Modify: `tests/test_extreme_fl_contract_ast.py`
- Modify: `tests/test_training_protocol_config_ast.py`

**Step 1: Write the failing tests**

Add assertions for:

- `ENABLE_FED_NORMAL_META_PROPOSED`
- `FED_NORMAL_META_SELF_FLOOR`
- `get_fed_normal_meta_model_path(station_id)`
- `run_fed_normal_meta_training()`
- Proposed-A loads `get_proposed_a_base_model_path(station_id)`
- LMT and Extreme-FedAvg continue to use `get_local_meta_model_path(station_id)`

**Step 2: Run tests to verify RED**

Run:

```bash
python -m unittest tests.test_extreme_fl_contract_ast tests.test_training_protocol_config_ast -v
```

Expected: FAIL because the new tokens are absent.

### Task 2: Add Fed-Normal-Meta Config And Paths

**Files:**
- Modify: `DemoModelTraining.py`

**Step 1: Implement minimal config**

Add:

```python
ENABLE_FED_NORMAL_META_PROPOSED = os.getenv("ENABLE_FED_NORMAL_META_PROPOSED", "0") != "0"
FED_NORMAL_META_SELF_FLOOR = float(os.getenv("FED_NORMAL_META_SELF_FLOOR", "0.3"))
SKIP_FED_NORMAL_META = os.getenv("SKIP_FED_NORMAL_META", "0") != "0"
```

Add path helpers:

```python
def get_fed_normal_meta_support_model_path(station_id):
    return resolve_model_path(f"model_fore_train_task_support_fed_normal_meta_station{station_id}.pth")


def get_fed_normal_meta_model_path(station_id):
    return resolve_model_path(f"model_fore_train_task_query_fed_normal_meta_station{station_id}.pth")
```

**Step 2: Run focused tests**

Run:

```bash
python -m unittest tests.test_extreme_fl_contract_ast tests.test_training_protocol_config_ast -v
```

Expected: partial progress; tests for functions/config should pass, behavior-routing tests may still fail.

### Task 3: Add Self-Floor FedAvg Normal-Meta Training

**Files:**
- Modify: `DemoModelTraining.py`

**Step 1: Add task count helper**

Compute per-station normal-weather meta task-window counts from `all_stations_full_data`.

**Step 2: Add normal-meta weights**

Implement:

```python
def compute_fed_normal_meta_station_weights(target_station_id, candidate_station_ids):
    ...
```

Rules:

- target weight is at least `FED_NORMAL_META_SELF_FLOOR`
- remaining weight is allocated to non-target stations by task-window counts
- if source count is zero, target gets weight `1.0`

**Step 3: Add training runner**

Implement `run_fed_normal_meta_training()` after local meta training. For each target station:

- initialize from `get_local_pretrain_model_path(station_id)`
- call `run_meta_training(...)`
- pass `sample_station_ids=station_ids`
- pass the computed station weights
- write the fed-normal-meta support/query checkpoints

### Task 4: Extend Meta Sampler To Use Station Weights

**Files:**
- Modify: `DemoModelTraining.py`

**Step 1: Update function signatures**

Add optional `station_sampling_weights=None` to:

- `sample_meta_batch(...)`
- `run_meta_training(...)`

**Step 2: Implement weighted station task selection**

When weights are provided:

- sample a station using the normalized weight map
- sample one class from that station
- repeat until `META_TASKS_PER_EPOCH` tasks are selected

When weights are absent:

- preserve existing global task-pool random sampling.

**Step 3: Preserve default behavior**

Existing local meta calls must still pass `sample_station_ids=[station_id]` and no weights.

### Task 5: Split Extreme Base Models

**Files:**
- Modify: `DemoModelTraining.py`

**Step 1: Add base resolver**

```python
def get_proposed_a_base_model_path(station_id):
    if ENABLE_FED_NORMAL_META_PROPOSED:
        return get_fed_normal_meta_model_path(station_id)
    return get_local_meta_model_path(station_id) if USE_FEDERATION and not USE_PSEUDO_FED else PROPOSED_META_MODEL_PATH
```

**Step 2: Change extreme loop**

Use:

- `local_shared_init_state` for LMT and Extreme-FedAvg
- `proposed_shared_init_state` for Proposed-A screening/source updates/aggregation/refinement

This intentionally makes Proposed-A's effective-window screening target-conditioned by the new prior while leaving Extreme-FedAvg unchanged.

### Task 6: Add Launcher Preview Support

**Files:**
- Modify: `run_three_station_yearly_protocol.py`

**Step 1: Add preview keys**

Add:

- `ENABLE_FED_NORMAL_META_PROPOSED`
- `FED_NORMAL_META_SELF_FLOOR`
- `SKIP_FED_NORMAL_META`

No new preset is required for the minimal experiment; use env overrides.

### Task 7: Verify

Run:

```bash
python -m unittest tests.test_extreme_fl_contract_ast tests.test_training_protocol_config_ast tests.test_skip_stage_reuse_ast tests.test_2h_6point_launcher_ast -v
python -m py_compile DemoModelTraining.py generate_multi_station_results.py run_three_station_yearly_protocol.py
ENABLE_FED_NORMAL_META_PROPOSED=1 python -u run_three_station_yearly_protocol.py train --smoke --six-station --dry-run
```

Expected:

- all selected tests pass
- compilation exits 0
- dry-run preview includes the fed-normal-meta env keys

### Task 8: 4090 Pilot Handoff

Run on the 4090 environment:

```bash
ARTIFACT_DIR=artifacts/2h6p_six_station/fed-normal-meta-pilot-5k \
ENABLE_FED_NORMAL_META_PROPOSED=1 \
FED_NORMAL_META_SELF_FLOOR=0.3 \
python -u run_three_station_yearly_protocol.py train --preset pilot-5k --six-station \
  2>&1 | tee logs/2h6p_six_station_fed_normal_meta_pilot5k_train_$(date +%Y%m%d_%H%M%S).log

ARTIFACT_DIR=artifacts/2h6p_six_station/fed-normal-meta-pilot-5k \
ENABLE_FED_NORMAL_META_PROPOSED=1 \
FED_NORMAL_META_SELF_FLOOR=0.3 \
python -u run_three_station_yearly_protocol.py eval --preset pilot-5k --six-station \
  2>&1 | tee logs/2h6p_six_station_fed_normal_meta_pilot5k_eval_all6_$(date +%Y%m%d_%H%M%S).log
```

Expected result file:

`artifacts/2h6p_six_station/fed-normal-meta-pilot-5k/results/multi_station_performance.csv`

Primary criterion:

`Overall_Average Proposed-A Mean nMAE <= 27.4073`.

