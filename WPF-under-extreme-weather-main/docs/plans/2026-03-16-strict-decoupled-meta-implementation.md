# Strict Decoupled Meta Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement F2L-inspired strict decoupled meta-learning where support updates local-only, query updates shared-only, the server aggregates shared-only, and final `Proposed` uses final shared plus final local.

**Architecture:** Rework the strict meta client round into a two-phase parameter-scope split. Support will adapt only local parameters and persist the resulting local meta state on each client. Query will then update only the shared backbone using the support-adapted local state as context. The server continues to aggregate shared backbone updates only. Final checkpoint semantics move to fixed-length final checkpoints rather than best-only / early-stop as the mainline.

**Tech Stack:** Python, PyTorch, existing `DemoModelTraining.py` training script, unittest AST tests, TensorBoard logging.

---

### Task 1: Add tests that lock the new parameter-scope split

**Files:**
- Modify: `tests/test_sfml_lite_ast.py`
- Create: `tests/test_strict_decoupled_meta_ast.py`

**Step 1: Write the failing test**

Add assertions that the strict meta path contains:

- `set_trainable_params_by_scope(` or equivalent scope helper
- local-only support update
- shared-only query update
- final checkpoint composition from `current_local_states` rather than `initial_local_states`

**Step 2: Run test to verify it fails**

Run:
```bash
python -m unittest tests.test_strict_decoupled_meta_ast -v
```

Expected: FAIL because the new helper and final-local composition are not implemented yet.

**Step 3: Write minimal implementation**

No implementation in this task.

**Step 4: Run test to verify it still fails for the expected reason**

Run the same command and confirm the missing symbols/strings are the only failures.

**Step 5: Commit**

Do not commit yet unless explicitly requested.

### Task 2: Add a scope helper for strict meta parameter selection

**Files:**
- Modify: `DemoModelTraining.py`
- Test: `tests/test_strict_decoupled_meta_ast.py`

**Step 1: Write the failing test**

Add a test that expects a helper like:

```python
def set_trainable_params_by_scope(model_instance, scope):
    ...
```

with supported scopes `shared` and `local`.

**Step 2: Run test to verify it fails**

Run:
```bash
python -m unittest tests.test_strict_decoupled_meta_ast -v
```

Expected: FAIL because helper does not exist.

**Step 3: Write minimal implementation**

Add a helper in `DemoModelTraining.py` that:

- sets `requires_grad=True` only on shared params when `scope == "shared"`
- sets `requires_grad=True` only on local params when `scope == "local"`
- returns the list of enabled parameters

**Step 4: Run test to verify it passes**

Run:
```bash
python -m unittest tests.test_strict_decoupled_meta_ast -v
```

Expected: PASS for helper-existence checks.

**Step 5: Commit**

Do not commit yet unless explicitly requested.

### Task 3: Refactor support phase to update local-only

**Files:**
- Modify: `DemoModelTraining.py`
- Test: `tests/test_strict_decoupled_meta_ast.py`

**Step 1: Write the failing test**

Add a test that expects the support path in `client_local_meta_round(...)` to:

- call the scope helper with `"local"`
- avoid shared updates during support

**Step 2: Run test to verify it fails**

Run:
```bash
python -m unittest tests.test_strict_decoupled_meta_ast -v
```

Expected: FAIL because support/query are not yet decoupled by scope.

**Step 3: Write minimal implementation**

In `client_local_meta_round(...)`:

- before support adaptation, enable only local params
- run `meta_inner_adapt(...)`
- capture `psi_i_sup` from the updated model

**Step 4: Run test to verify it passes**

Run:
```bash
python -m unittest tests.test_strict_decoupled_meta_ast -v
```

Expected: PASS for support-scope checks.

**Step 5: Commit**

Do not commit yet unless explicitly requested.

### Task 4: Refactor query phase to update shared-only

**Files:**
- Modify: `DemoModelTraining.py`
- Test: `tests/test_strict_decoupled_meta_ast.py`

**Step 1: Write the failing test**

Add a test that expects the query path in `client_local_meta_round(...)` to:

- freeze local params after support
- enable shared params only for query update

**Step 2: Run test to verify it fails**

Run:
```bash
python -m unittest tests.test_strict_decoupled_meta_ast -v
```

Expected: FAIL because query still updates mixed scopes.

**Step 3: Write minimal implementation**

In `client_local_meta_round(...)`:

- after support, freeze local params at `psi_i_sup`
- enable shared params only
- compute query loss and apply update to shared params only

**Step 4: Run test to verify it passes**

Run:
```bash
python -m unittest tests.test_strict_decoupled_meta_ast -v
```

Expected: PASS for query-scope checks.

**Step 5: Commit**

Do not commit yet unless explicitly requested.

### Task 5: Change strict meta state propagation to keep final local states

**Files:**
- Modify: `DemoModelTraining.py`
- Test: `tests/test_strict_decoupled_meta_ast.py`

**Step 1: Write the failing test**

Add a test that expects final strict meta checkpoint composition to use:

- final shared state
- final `current_local_states[station_id]`

and not:

- `initial_local_states[station_id]`

**Step 2: Run test to verify it fails**

Run:
```bash
python -m unittest tests.test_strict_decoupled_meta_ast -v
```

Expected: FAIL because final checkpoint currently rebuilds from initial local state.

**Step 3: Write minimal implementation**

In `run_strict_federated_meta_training(...)`:

- compose final per-station strict meta checkpoint from:
  - final selected shared state
  - final client-local strict meta state for that station

**Step 4: Run test to verify it passes**

Run:
```bash
python -m unittest tests.test_strict_decoupled_meta_ast -v
```

Expected: PASS.

**Step 5: Commit**

Do not commit yet unless explicitly requested.

### Task 6: Remove best-only / early-stop dependence from mainline configuration

**Files:**
- Modify: `DemoModelTraining.py`
- Modify: `tests/test_sfml_lite_ast.py`

**Step 1: Write the failing test**

Add or adjust AST expectations so mainline strict decoupled meta is compatible with:

- fixed final checkpoint semantics
- diagnostic-only best/early-stop flags

**Step 2: Run test to verify it fails**

Run:
```bash
python -m unittest tests.test_sfml_lite_ast tests.test_strict_decoupled_meta_ast -v
```

Expected: FAIL because current mainline still depends on best-only semantics.

**Step 3: Write minimal implementation**

Update defaults/comments and final-save logic so:

- mainline semantics use final checkpoints
- best-only / early-stop remain optional diagnostics

**Step 4: Run test to verify it passes**

Run:
```bash
python -m unittest tests.test_sfml_lite_ast tests.test_strict_decoupled_meta_ast -v
```

Expected: PASS.

**Step 5: Commit**

Do not commit yet unless explicitly requested.

### Task 7: Preserve evaluation semantics

**Files:**
- Modify: `generate_multi_station_results.py`
- Test: `tests/test_generate_multi_station_results_ast.py`

**Step 1: Write the failing test**

Add a test only if needed to lock that:

- `Proposed` loads the strict-decoupled per-station meta initialization
- `Pre_Training` still loads pretrain-only checkpoints
- `Meta_Learning` semantics remain unchanged unless strict `meta-only` is explicitly enabled

**Step 2: Run test to verify it fails**

Run:
```bash
python -m unittest tests.test_generate_multi_station_results_ast -v
```

Expected: FAIL only if evaluation semantics are broken by the strict-decoupled change.

**Step 3: Write minimal implementation**

Adjust evaluation lookup only if the strict-decoupled checkpoint naming or composition changes require it.

**Step 4: Run test to verify it passes**

Run:
```bash
python -m unittest tests.test_generate_multi_station_results_ast -v
```

Expected: PASS.

**Step 5: Commit**

Do not commit yet unless explicitly requested.

### Task 8: Run full local verification

**Files:**
- Verify: `DemoModelTraining.py`
- Verify: `generate_multi_station_results.py`
- Verify: `sfml_meta_utils.py`
- Verify: tests under `tests/`

**Step 1: Run unit tests**

Run:
```bash
python -m unittest tests.test_sfml_meta_utils tests.test_sfml_lite_ast tests.test_strict_decoupled_meta_ast tests.test_sfml_no_global_pool_ast tests.test_generate_multi_station_results_ast tests.test_strict_federated_baseline_ast
```

Expected: PASS.

**Step 2: Run static compilation**

Run:
```bash
python -m py_compile DemoModelTraining.py generate_multi_station_results.py sfml_meta_utils.py pfedfsl_lite_utils.py
```

Expected: no output, exit code 0.

**Step 3: Run CPU smoke**

Run:
```bash
ENABLE_FED_META_TRAIN=0 \
ENABLE_STRICT_FED_META_TRAIN=1 \
ENABLE_STRICT_FED_META_ONLY=0 \
STRICT_META_USE_SECOND_ORDER=0 \
PRETRAIN_EPOCHS=1 \
STRICT_META_EPOCHS=1 \
FEW_SHOT_EPOCHS=1 \
python DemoModelTraining.py
```

Expected:
- strict meta completes
- per-station meta checkpoints are saved
- few-shot models are saved
- `all_stations_test_results.mat` is generated

**Step 4: Commit**

Do not commit yet unless explicitly requested.

### Task 9: Prepare the 4090 experiment commands

**Files:**
- No code change required

**Step 1: Write the 4090 command for decoupled strict**

Document the run command with:

- `ENABLE_STRICT_FED_META_TRAIN=1`
- `ENABLE_STRICT_FED_META_ONLY=0`
- fixed `STRICT_META_EPOCHS`
- no best-only / early-stop dependence in the mainline

**Step 2: Write the matched comparison command**

Document the legacy comparison command so the user can run apples-to-apples on the 4090.

**Step 3: Verify commands reference the current copied files**

Expected: commands use isolated `/tmp/...` run directories and `python -u ... | tee ...`.

**Step 4: Commit**

Do not commit yet unless explicitly requested.
