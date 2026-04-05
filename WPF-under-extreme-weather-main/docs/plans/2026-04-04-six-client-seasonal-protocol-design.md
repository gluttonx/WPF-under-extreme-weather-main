# Six-Client Seasonal Protocol Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the old full-year three-station protocol with a six-client seasonal protocol that creates target-data scarcity while keeping `Proposed` and `LOCAL_META_TRANSFER` comparable under the same training budget.

**Architecture:** Add a protocol-specific data layer built from the raw `xlsx` files rather than the old full-year `.mat` files. Each experimental client is defined by a fixed seasonal training window and a matched later-year test window. Phase 1, Phase 2, few-shot support, and evaluation then consume protocol-specific conventional/extreme splits instead of the historical full-year splits.

**Tech Stack:** Python, NumPy, pandas/`xlsx` parsing, SciPy `.mat`, PyTorch, existing AST/unittest test suite

---

## Design Summary

### Why the old protocol is no longer appropriate

The old `2022` full-year train / `2023` full-year test setup leaves each local station with enough conventional data that `LOCAL_META_TRANSFER` can already learn a strong local prior. That weakens the necessity claim for cross-client federated prior sharing. The new protocol therefore creates explicit scarcity by restricting each experimental client to one seasonal training block and one matched later-year seasonal test block.

### Experimental client definition

The experiment will use six protocol clients rather than the historical three full-year stations.

- `WT1 (58)`: train `2022-03-01` to `2022-05-31`, test `2023-03-01` to `2023-05-31`
- `WT2 (59)`: train `2022-03-01` to `2022-05-31`, test `2023-03-01` to `2023-05-31`
- `WT3 (60)`: train `2022-06-01` to `2022-08-31`, test `2023-06-01` to `2023-08-31`
- `WT4 (61)`: train `2022-06-01` to `2022-08-31`, test `2023-06-01` to `2023-08-31` using the `058` source workbook
- `WT5 (62)`: train `2022-11-01` to `2023-01-31`, test `2023-11-01` to `2024-01-31` using the `059` source workbook plus `2024` workbook splice
- `WT6 (63)`: train `2022-11-01` to `2023-01-31`, test `2023-11-01` to `2024-01-31` using the `060` source workbook plus `2024` workbook splice

This is a protocol-level six-client construction, not a claim about six independently observed physical farms.

### Seasonal regime mapping

The six clients are intentionally regime-specific:

- `WT1`, `WT2`: wind-season clients, primary evaluation class `high_wind`
- `WT3`, `WT4`: hot-season clients, primary evaluation class `high_temp`
- `WT5`, `WT6`: winter clients, primary evaluation classes `cold_wave` and `frost`

Any class that appears only on one side of the train/test split for a client is excluded from final reporting for that client.

### Phase definitions

- **Phase 1 / Proposed federated pre-train**
  - Use each client’s training block **first month** `normal_weather`
  - Federated aggregation remains on the `Proposed` path only
- **Phase 1 / LOCAL_META_TRANSFER local pre-train**
  - Use the target client’s training block **first month** `normal_weather`
- **Phase 2 / local meta-training**
  - Use the same target client **first month** `normal_weather`
  - `Proposed` starts from federated prior
  - `LOCAL_META_TRANSFER` starts from local prior
- **Few-shot fine-tuning**
  - Use the target client’s **full training block** `extreme weather` as the support pool
- **Testing**
  - Use the matched later-year seasonal test block `extreme weather`

### Reporting rule

The main table will report metrics directly for each valid `(client, extreme_class)` task rather than collapsing them into one macro average. The expected main tasks are:

- `(58, high_wind)`
- `(59, high_wind)`
- `(60, high_temp)`
- `(61, high_temp)`
- `(62, cold_wave)`
- `(62, frost)`
- `(63, cold_wave)`
- `(63, frost)`

For each task, report `nMAE_%`, `nRMSE_%`, and `WD_%` for both `Proposed` and `LOCAL_META_TRANSFER`, plus the signed difference. A lightweight textual summary such as `wins x/8` is optional, but no forced global averaging is required.

## Meta-Training Redesign

### Motivation

The old setup assumed full-year conventional data, fixed `10` KMeans classes, and per-class episode sampling of `10 support + 10 query`. Under the new protocol, each client only contributes the first month of conventional data to meta-training. The meta-task granularity must therefore be re-derived from the new sample budget.

### Keep the sequence length at 12

Use:

- `len_realp = 12`

Reason:

- It preserves the current half-day temporal context already assumed by the model and existing pipeline
- It avoids the severe window collapse that would happen at `24`
- It does not shorten context so aggressively that the regime structure becomes too local

### Reduce meta episode size to 5/5

Change the per-class episode from:

- historical `10 support + 10 query`

to:

- new `5 support + 5 query`

Reason:

- Under the new one-month conventional budget, total normal windows per client are only on the order of a few dozen
- `5/5` keeps support and query both meaningful while lowering the minimum feasible per-cluster window count from `20` to `10`

### Dynamic K selection

Do **not** hard-code `K=10`.

For each client:

1. Use only the client’s training-block **first month** `normal_weather`
2. Segment it into non-overlapping windows of length `len_realp`
3. Let:
   - `L = len_realp`
   - `M = support + query`
   - `N_windows = floor(normal_hours / L)`
   - `K_max = floor(N_windows / M)`
4. Run elbow search only over `K in [2, K_max]`
5. Accept the elbow-selected `K` only if every cluster has at least `M` windows
6. If not, step downward until the first feasible `K` is found

This makes the upper bound data-driven rather than fixed by heuristic prejudice.

### Meta sampler count

For the `Proposed` balanced sampler, set:

- `sample_count = max(2, ceil(K / 2))`

Examples:

- `K=2 -> 2`
- `K=3 -> 2`
- `K=4 -> 2`
- `K=5 -> 3`
- `K=6 -> 3`

This preserves regime diversity per episode without forcing all classes every round.

## Data Processing Rules

### Source workbooks

- Historical source workbooks:
  - `2223jilin_058_processed_4classes.xlsx`
  - `2223jilin_059_processed_4classes.xlsx`
  - `2223jilin_060_processed_4classes.xlsx`
- Added 2024 workbooks:
  - `24jilin_058_processed_4classes.xlsx`
  - `24jilin_059_processed_4classes.xlsx`
  - `24jilin_060_processed_4classes.xlsx`

### 2024 capacity normalization

When slicing `2024` rows, convert `Power2` to per-unit with the correct capacity:

- `058 -> 50`
- `059 -> 100`
- `060 -> 300`

This rule applies before any protocol-specific `.mat` export or training-data assembly.

### Scope of old assets

The historical full-year `.mat` files and their fixed `10` conventional classes should remain as legacy baseline assets. The new six-client seasonal protocol should use newly generated protocol-specific assets rather than overloading the historical files in place.

## File Impact

Expected implementation impact:

- New protocol preprocessing script for seasonal client extraction and dynamic clustering
- `DemoModelTraining.py` updates for:
  - six-client protocol loading
  - dynamic class counts per client
  - configurable `support/query`
  - balanced sampler count derived from `K`
- `generate_multi_station_results.py` updates for per-task reporting under the six-client protocol
- New AST tests locking the protocol wiring and dynamic meta-task rules

## Risks and Controls

### Risk 1: Elbow method prefers an infeasible `K`

Control:

- Apply the explicit feasibility check `min_cluster_windows >= support + query`

### Risk 2: Test-side classes appear without train-side support

Control:

- Report only the intersection of train-support and test classes for each client

### Risk 3: 2024 data scale mismatch

Control:

- Normalize `2024` `Power2` with client-specific capacities before any merged winter test export

### Risk 4: Old result scripts silently average incompatible tasks

Control:

- Make per-`(client,class)` reporting the default output format under the seasonal protocol

## Validation Plan

- Structural validation:
  - protocol metadata contains all six clients with correct train/test windows
  - dynamic `K` is recorded per client
  - sampler count matches `max(2, ceil(K/2))`
- Data validation:
  - each exported client has non-empty first-month `normal_weather`
  - each reported task has class presence on both support and test sides
  - `2024` winter slices use the correct capacity normalization
- Training smoke validation:
  - CPU short-run smoke test with tiny epoch counts only
- Formal conclusion policy:
  - all substantive metrics must still come from the user’s `RTX 4090` environment

## Assumptions and Unverified Items

- The elbow-selected `K` values have not yet been run; only the selection rule is fixed
- The final protocol asset format has not yet been implemented, so exact output filenames remain a design choice
- No claim is made here that the six clients represent six physically independent wind farms
