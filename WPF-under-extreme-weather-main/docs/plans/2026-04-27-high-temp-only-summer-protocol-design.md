# Single-Class High-Temperature Summer Protocol Design

## Goal

Replace the current four-extreme-class two-year mainline with a cleaner single-class protocol focused only on `High temperature`, while preserving:

- the `2h / 6-point / 12h window` setup
- six protocol stations through phase augmentation
- the current three-way comparison: `LMT`, `Extreme-FedAvg`, `Proposed-A`

The purpose is to align normal-weather training, extreme-weather adaptation, and evaluation inside the same summer regime, so the experiment tests transfer under a narrower and more defensible seasonal shift.

## Decision

Use a new summer-only high-temperature protocol as a separate branch of the existing yearly-protocol framework.

Do **not** overwrite the current `two_year_2024_k6` assets or semantics.

Do **not** introduce a new offline preprocessing pipeline.

Instead, extend the existing protocol builder and launcher so the repository can build a new named protocol with:

- summer-only `2023` normal-weather training
- summer-only `2023` `extreme_high_temp` support
- `2024` `extreme_high_temp` test
- per-station normal-task cluster counts
- single-extreme-class exports and reports

## Data Scope

### Training workbooks

- `2223jilin_058_processed_4classes.xlsx`
- `2223jilin_059_processed_4classes.xlsx`
- `2223jilin_060_processed_4classes.xlsx`

### Test workbooks

- `24jilin_058_processed_4classes.xlsx`
- `24jilin_059_processed_4classes.xlsx`
- `24jilin_060_processed_4classes.xlsx`

### Time filters

#### Normal-weather meta-training pool

- source: `normal_weather`
- year: `2023`
- months: `6, 7, 8`

#### Extreme-weather support pool

- source: `extreme_high_temp`
- year: `2023`
- months: `6, 7, 8`

#### Extreme-weather test pool

- source: `extreme_high_temp`
- year: `2024`
- no additional month filter beyond the sheet contents

Current workbook inspection shows the `2024` `extreme_high_temp` rows all fall in July, so an explicit month restriction is not required for the first implementation.

## Six-Station Protocol

Keep the current six-station phase-augmentation definition:

- base stations: `58`, `59`, `60`
- complementary stations: `61`, `62`, `63`

Keep the same mapping:

- `58 -> 61`
- `59 -> 62`
- `60 -> 63`

Keep the current 2h meaning:

- base stations use `downsample_offset = 1`
- complementary stations use `downsample_offset = 0`

This remains an hour-phase split, not a date split.

## Capacity Normalization

Keep the current yearly-protocol normalization setup:

| Base station | Train workbook | Train capacity | Test workbook | Test capacity |
| --- | --- | ---: | --- | ---: |
| 58 | `2223jilin_058_processed_4classes.xlsx` | 50 | `24jilin_058_processed_4classes.xlsx` | 50 |
| 59 | `2223jilin_059_processed_4classes.xlsx` | 50 | `24jilin_059_processed_4classes.xlsx` | 100 |
| 60 | `2223jilin_060_processed_4classes.xlsx` | 100 | `24jilin_060_processed_4classes.xlsx` | 300 |

The complementary stations inherit the matching base station train/test sources and capacities.

## Normal-Meta Task Construction

### Cluster counts

Use per-station `k`, not a shared `k`.

The summer-2023 normal pool under the six-station `2h/6p` protocol supports:

- `58 = 7`
- `61 = 7`
- `59 = 6`
- `62 = 6`
- `60 = 5`
- `63 = 5`

This comes from:

- station-wise elbow behavior on `2023-06~08 normal`
- feasibility constraints on minimum windows per cluster
- the fact that the current training code already accepts per-station class counts when reading `p_conven_class`

### Normal budget

Do **not** retain the old `360-point` two-year cap.

Use the full `2023-06~08 normal_weather` pool for each phase station. Under `2h/6p`, this leaves about `175-178` complete windows per phase station, which is large enough to support clustering without artificial subsampling.

## Meta-Training Parameter Logic

The original paper's transferable logic is:

1. determine `k` by elbow plus sample-feasibility
2. determine `k* × n*` around the richest target extreme-sample scale
3. compare a small number of structured candidates instead of tuning arbitrary values

For this new protocol, the richest `2023` target support scale is:

- `58/61 = 5` windows
- `59/62 = 7` windows
- `60/63 = 8` windows

So the new anchor is `8`.

### Candidate comparison

Use the following first-round comparison:

- main candidate: `k* = 2`, `n* = 4`
- control candidate: `k* = 3`, `n* = 3`

Do **not** use `4 × 2` as the main candidate, because `n* = 2` is too far from the true target-side few-shot scale.

### Exposure matching

Do not compare settings by raw meta epochs.

Compare them by total task exposure:

- `total_task_exposure = META_TASKS_PER_EPOCH × META_EPOCHS`

For the first controlled comparison, use:

- `2 × 4`: `META_TASKS_PER_EPOCH = 2`, `META_SUPPORT_SHOTS = 4`, `META_QUERY_SHOTS = 4`, `META_EPOCHS = 12000`
- `3 × 3`: `META_TASKS_PER_EPOCH = 3`, `META_SUPPORT_SHOTS = 3`, `META_QUERY_SHOTS = 3`, `META_EPOCHS = 8000`

Both settings then receive the same total exposure of `24000` sampled tasks.

## Optimization Hyperparameters

Keep the existing optimization settings unless a later experiment disproves them:

- learning rate for support/query/target adaptation: `0.0002`
- Adam `betas = (0.5, 0.999)`
- support/query iteration = `1`
- `lambda = 10`
- `lambda warm-up = 0 -> 1 -> 5 -> 10`

These are still defensible because:

- the network architecture is unchanged
- the support/query update mechanism is unchanged
- only the protocol distribution has changed

## Method Definitions

### `LMT`

`LMT` remains the local transfer baseline.

It uses:

- local summer-normal pretrain
- local summer-normal meta-train
- target-only `High temperature` few-shot adaptation and target refinement

It does **not** borrow source-station extreme updates.

### `Extreme-FedAvg`

`Extreme-FedAvg` shares the same normal-weather base as `LMT`.

At the extreme stage, it:

- adapts source-station `High temperature` updates
- aggregates them with the target self update using uniform weighting
- runs target refinement

This isolates the value of extreme-stage federation without reliability weighting.

### `Proposed-A`

`Proposed-A` also shares the same normal-weather base as `LMT`.

At the extreme stage, it:

- adapts source-station `High temperature` updates
- aggregates them with the target self update using the current reliability-aware weighting logic
- runs target refinement

For the first high-temperature-only protocol, keep `Proposed-A` in its clean base form.

Do **not** add:

- Fed-Normal-Meta
- class-wise gating
- class-wise fallback

Those mechanisms can be revisited only after the clean single-class protocol is established.

## Evaluation Plan

Only evaluate `High temperature`.

The exported results should contain:

- station `58`
- station `59`
- station `60`
- station `61`
- station `62`
- station `63`
- `Overall_Average`

If inexpensive to keep, also export a pooled or sample-weighted overall row as a diagnostic, but the user-facing main table should still match the planned presentation of per-station rows plus `Overall_Average`.

## Selection Rule

Choose the winning meta setting by:

1. `Overall_Average` `HighTemperature_nMAE_%` relative gap: `Proposed-A vs LMT`
2. `Overall_Average` `HighTemperature_nRMSE_%` relative gap
3. `HighTemperature_WD_%`

Additional constraint:

- the selected setting must not make `Proposed-A` clearly weaker than `Extreme-FedAvg` on `Overall_Average`

If `2×4` and `3×3` are close, prefer `2×4` because it is more faithful to the target-side few-shot scale.

## Risks

### Risk 1: The protocol remains too narrow

This is acceptable for now because the purpose is to test whether a season-aligned single-class problem is easier to separate than the current mixed-class mainline.

### Risk 2: Old four-class assumptions remain hard-coded

This is the main implementation risk. The builder, training loop, few-shot model counting, export labels, and evaluation loops all currently assume four extreme classes.

### Risk 3: Result interpretation overstates six-station independence

The six protocol stations still come from three physical stations plus hour-phase augmentation. This should be described accurately in the experimental protocol and discussion.

## Recommended Next Step

Implement this protocol without changing the algorithm family:

1. build the new single-class seasonal protocol assets
2. run `LMT`, `Extreme-FedAvg`, and `Proposed-A` under `2×4`
3. rerun the same trio under `3×3`
4. pick the winning meta setting before introducing any additional mechanism
