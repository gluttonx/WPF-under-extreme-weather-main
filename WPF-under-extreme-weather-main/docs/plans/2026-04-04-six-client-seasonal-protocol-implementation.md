# Six-Client Seasonal Protocol Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the approved six-client seasonal scarcity protocol, including protocol-specific data extraction, dynamic conventional clustering, lighter meta episodes, and per-`(client,class)` reporting.

**Architecture:** Keep the existing model family intact and introduce a protocol-specific data/config layer around it. Build new seasonal client assets from raw `xlsx` files, then update `DemoModelTraining.py` and `generate_multi_station_results.py` to consume those assets and the new dynamic meta-task rules.

**Tech Stack:** Python, NumPy, pandas/`xlsx` parsing, SciPy `.mat`, PyTorch, unittest AST tests

---

### Task 1: Lock the protocol constants and dynamic meta-task rules with AST tests

**Files:**
- Create: `tests/test_six_client_seasonal_protocol_ast.py`
- Modify: `tests/test_balanced_meta_sampler_ast.py`

**Step 1: Write the failing test**

Add AST/string assertions for:

- a six-client protocol config or manifest loader
- `len_realp = 12` or equivalent protocol-fed sequence length
- meta episode split `5 support + 5 query`
- dynamic `K_max` logic based on `normal_hours`, `len_realp`, and `support + query`
- sampler count rule `max(2, ceil(K / 2))`

**Step 2: Run test to verify it fails**

Run:
```bash
python -m unittest tests.test_six_client_seasonal_protocol_ast tests.test_balanced_meta_sampler_ast -v
```

Expected: FAIL because the new protocol helpers and dynamic rules do not exist yet.

**Step 3: Write minimal implementation**

No implementation in this task.

**Step 4: Run test to verify it still fails for the expected reasons**

Run the same command and confirm the failures are only for the missing protocol symbols and rules.

**Step 5: Commit**

Do not commit unless explicitly requested.

### Task 2: Add a dedicated preprocessing script for six-client seasonal assets

**Files:**
- Create: `build_six_client_seasonal_protocol.py`
- Create: `seasonal_protocol_data/README.md`
- Test: `tests/test_six_client_seasonal_protocol_ast.py`

**Step 1: Write the failing test**

Add assertions that the new preprocessing script contains:

- the six approved client window definitions
- 2024 workbook handling for `WT5/WT6`
- client-specific capacity normalization for `2024`
- protocol metadata export

**Step 2: Run test to verify it fails**

Run:
```bash
python -m unittest tests.test_six_client_seasonal_protocol_ast -v
```

Expected: FAIL because the preprocessing script does not yet exist.

**Step 3: Write minimal implementation**

Create `build_six_client_seasonal_protocol.py` that:

- reads the six source workbooks
- slices train/test windows for `WT1` through `WT6`
- builds:
  - first-month `normal_weather` pool
  - full-train-block `extreme weather` support pool
  - test-block `extreme weather` pool
- applies `2024` per-unit scaling with capacities `50/100/300`
- writes:
  - six protocol-specific `.mat` files under `seasonal_protocol_data/`
  - one metadata JSON file describing windows, valid classes, `K`, and sampler count

**Step 4: Run test to verify it passes**

Run:
```bash
python -m unittest tests.test_six_client_seasonal_protocol_ast -v
```

Expected: PASS for existence/config checks.

**Step 5: Commit**

Do not commit unless explicitly requested.

### Task 3: Implement dynamic conventional clustering in the preprocessing script

**Files:**
- Modify: `build_six_client_seasonal_protocol.py`
- Test: `tests/test_six_client_seasonal_protocol_ast.py`

**Step 1: Write the failing test**

Add assertions that the preprocessing script:

- computes non-overlapping conventional windows from the first training month
- derives `K_max = floor(N_windows / (support + query))`
- searches elbow candidates over `2..K_max`
- backs off to the nearest feasible `K` if the elbow choice yields any cluster with fewer than `support + query` windows

**Step 2: Run test to verify it fails**

Run:
```bash
python -m unittest tests.test_six_client_seasonal_protocol_ast -v
```

Expected: FAIL because dynamic `K` selection is not implemented yet.

**Step 3: Write minimal implementation**

In `build_six_client_seasonal_protocol.py`:

- add helper functions for:
  - conventional window counting
  - elbow candidate scoring
  - feasibility filtering by minimum cluster window count
- store per-client:
  - chosen `K`
  - per-cluster window counts
  - `sample_count = max(2, ceil(K / 2))`

**Step 4: Run test to verify it passes**

Run:
```bash
python -m unittest tests.test_six_client_seasonal_protocol_ast -v
```

Expected: PASS.

**Step 5: Commit**

Do not commit unless explicitly requested.

### Task 4: Teach `DemoModelTraining.py` to consume protocol-specific assets and lighter meta episodes

**Files:**
- Modify: `DemoModelTraining.py`
- Modify: `tests/test_fed_pretrain_local_meta_ast.py`
- Modify: `tests/test_balanced_meta_sampler_ast.py`
- Test: `tests/test_six_client_seasonal_protocol_ast.py`

**Step 1: Write the failing test**

Add assertions that the training script:

- can load the new six-client protocol asset/metadata path
- uses protocol-specific per-client class counts instead of a fixed `10`
- uses `5 support + 5 query` in meta batch construction
- uses per-client sampler count derived from metadata or dynamic `K`

**Step 2: Run test to verify it fails**

Run:
```bash
python -m unittest tests.test_six_client_seasonal_protocol_ast tests.test_balanced_meta_sampler_ast tests.test_fed_pretrain_local_meta_ast -v
```

Expected: FAIL because the training script still assumes legacy full-year assets and `10/10` episodes.

**Step 3: Write minimal implementation**

In `DemoModelTraining.py`:

- add protocol configuration toggle/path handling
- replace the fixed `10`-class assumption with per-client metadata-driven class counts
- update `build_meta_batch_from_tasks(...)` to sample `5` support and `5` query windows
- update balanced sampling to use per-client `sample_count`
- keep legacy behavior available when the seasonal protocol is disabled

**Step 4: Run test to verify it passes**

Run:
```bash
python -m unittest tests.test_six_client_seasonal_protocol_ast tests.test_balanced_meta_sampler_ast tests.test_fed_pretrain_local_meta_ast -v
```

Expected: PASS.

**Step 5: Commit**

Do not commit unless explicitly requested.

### Task 5: Update result generation to report per-(client,class) tasks directly

**Files:**
- Modify: `generate_multi_station_results.py`
- Modify: `tests/test_generate_multi_station_results_ast.py`

**Step 1: Write the failing test**

Add assertions that result generation under the seasonal protocol:

- emits one row per valid `(client, class)` task
- excludes classes missing from either support or test side
- does not force a global average row as the primary output

**Step 2: Run test to verify it fails**

Run:
```bash
python -m unittest tests.test_generate_multi_station_results_ast -v
```

Expected: FAIL because the current reporting logic still assumes the legacy station/category table structure.

**Step 3: Write minimal implementation**

In `generate_multi_station_results.py`:

- detect the seasonal protocol mode
- iterate through valid `(client, class)` tasks from metadata
- report `nMAE_%`, `nRMSE_%`, and `WD_%` for each task
- optionally append a lightweight wins summary, but do not make it the main output

**Step 4: Run test to verify it passes**

Run:
```bash
python -m unittest tests.test_generate_multi_station_results_ast -v
```

Expected: PASS.

**Step 5: Commit**

Do not commit unless explicitly requested.

### Task 6: Run preprocessing and CPU smoke validation

**Files:**
- Run: `build_six_client_seasonal_protocol.py`
- Run: `DemoModelTraining.py`
- Run: `generate_multi_station_results.py`

**Step 1: Run protocol preprocessing**

Run:
```bash
python build_six_client_seasonal_protocol.py
```

Expected:

- six protocol `.mat` files created under `seasonal_protocol_data/`
- one protocol metadata JSON created
- per-client `K` and valid classes printed

**Step 2: Run short CPU smoke training**

Run:
```bash
PRETRAIN_EPOCHS=2 META_EPOCHS=2 FEW_SHOT_EPOCHS=1 python DemoModelTraining.py
```

Expected:

- script completes without shape or sampler errors
- seasonal protocol metadata is loaded
- per-client class counts and sampler counts are logged

**Step 3: Run result generation smoke**

Run:
```bash
python generate_multi_station_results.py
```

Expected:

- per-`(client,class)` rows are emitted
- no invalid class rows appear for unmatched support/test classes

**Step 4: Record findings**

Add a short note to the active design/session docs summarizing:

- chosen `K` per client
- whether any client was forced to back off from the elbow candidate
- whether CPU smoke completed

**Step 5: Commit**

Do not commit unless explicitly requested.
