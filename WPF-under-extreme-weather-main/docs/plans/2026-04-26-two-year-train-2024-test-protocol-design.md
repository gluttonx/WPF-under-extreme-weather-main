# Two-Year Train / 2024 Test Protocol Design

## Goal

Replace the current `2022 support / 2023 test` 2h/6p six-station protocol with a stronger main protocol:

- train on `2022 + 2023`
- test on `2024`
- keep `2h/6p`
- keep six protocol stations through phase augmentation
- keep normal-weather training budget capped at `30-day-equivalent`

The purpose is to remove the current extreme-data bottleneck for `high_temp` and `cold_wave` without giving up the low-data setting on normal weather.

## Decision

Use a single protocol builder as the source of truth. Do **not** introduce a separate `24jilin_58-60.py` preprocessing script.

Instead, extend [build_three_station_extreme_yearly_protocol.py](/tmp/wpf-worktrees/restore-raw-data-83688a3/WPF-under-extreme-weather-main/build_three_station_extreme_yearly_protocol.py) so each base station configuration carries:

- `train_workbook`
- `test_workbook`
- `train_capacity`
- `test_capacity`

This keeps normalization logic in one place and avoids drift between offline scripts and protocol metadata.

## Data Sources

### Training workbooks

- `2223jilin_058_processed_4classes.xlsx`
- `2223jilin_059_processed_4classes.xlsx`
- `2223jilin_060_processed_4classes.xlsx`

These files already contain `2022` and `2023` rows together.

### Test workbooks

- `24jilin_058_processed_4classes.xlsx`
- `24jilin_059_processed_4classes.xlsx`
- `24jilin_060_processed_4classes.xlsx`

### Capacities

For the new protocol, use:

| Base station | Train workbook | Train capacity | Test workbook | Test capacity |
| --- | --- | ---: | --- | ---: |
| 58 | `2223jilin_058_processed_4classes.xlsx` | 50 | `24jilin_058_processed_4classes.xlsx` | 50 |
| 59 | `2223jilin_059_processed_4classes.xlsx` | 50 | `24jilin_059_processed_4classes.xlsx` | 100 |
| 60 | `2223jilin_060_processed_4classes.xlsx` | 100 | `24jilin_060_processed_4classes.xlsx` | 300 |

The six protocol stations remain:

- base: `58`, `59`, `60`
- complementary: `61`, `62`, `63`

with the same phase-augmentation mapping:

- `58 -> 61`
- `59 -> 62`
- `60 -> 63`

## Protocol Definition

### Sampling

Keep the existing 2h/6-point setting:

- `SAMPLE_INTERVAL_HOURS = 2`
- `LEN_REALP = 6`
- `POINTS_PER_DAY = 12`
- `WINDOW_SPAN_HOURS = 12`

Keep phase augmentation exactly as before:

- base stations use `downsample_offset = 1`
- complementary stations use `downsample_offset = 0`

### Train / test split

#### Main and extreme weather

- training rows come from the full `2223` workbook
- testing rows come from the full `24` workbook

There is no longer a `support year = 2022` and `test year = 2023` split inside a single workbook for the main protocol.

#### Normal weather

Normal-weather training must remain budget-limited.

Use a `30-day-equivalent = 360 points` budget sampled from the combined `2022 + 2023` normal-weather pool.

Recommended default:

- `180` points sampled from `2022`
- `180` points sampled from `2023`
- within each year, use month-stratified sampling

This preserves the low-data normal setting while allowing extreme-weather training to benefit from two years of accumulated rare events.

## Why This Protocol Is Better

The current `2022 support / 2023 test` protocol creates a structural problem:

- `high_temp` support windows are often only `1-4`
- `cold_wave` support windows are often only `1-2`

Under the proposed `22-23 train / 24 test` protocol, the training-window counts rise substantially:

- `high_wind`: roughly `13-18`
- `high_temp`: roughly `6-12`
- `cold_wave`: roughly `6-7`
- `frost`: roughly `32-50`

This removes the worst data scarcity issue before changing the algorithm again.

## Model-Comparison Strategy

Do not introduce a new algorithm immediately.

Use the new protocol to rerun the strongest existing baselines first:

1. `Local-Meta-FT`
2. the historically best `Fed-Normal-Meta / Proposed-A` family variant

This answers the most important question first:

> Was the old `+2%~3%` ceiling mainly caused by the old protocol's rare-extreme bottleneck?

If the historical federated family improves materially under the new protocol, that is stronger evidence than introducing another brand-new method too early.

## Recommended Implementation Route

### Recommended

Extend the yearly protocol builder so it can:

- read separate train and test workbooks
- normalize train and test using different capacities
- sample normal weather from a two-year pool with a fixed total budget
- write protocol metadata that makes the train/test source explicit

### Not recommended

Create a separate `24jilin_58-60.py` script that manually rewrites `2024` workbooks or `.mat` files.

That would duplicate logic already present in the builder:

- xlsx parsing
- normalization
- downsampling
- phase augmentation
- metadata recording

## Required Metadata Additions

The new protocol metadata should explicitly record:

- `protocol_name`
- `train_years = [2022, 2023]`
- `test_years = [2024]`
- `train_workbook`
- `test_workbook`
- `train_capacity`
- `test_capacity`
- `normal_sampling_policy`
- per-station train/test extreme window counts

This is necessary because the new protocol is no longer inferable from file names alone.

## Risks

### Risk 1: Capacity mismatch causes silent normalization errors

This is the main new protocol risk, because station `59` and `60` have different train/test capacities.

Control:

- store both train and test capacities explicitly in metadata
- print them in builder logs
- add tests that verify the configured capacities per station

### Risk 2: New protocol breaks comparability with old result tables

This is acceptable. The new protocol is intended to become the main protocol.

Control:

- do not mix old and new results in the same headline table
- include protocol name in every result export

### Risk 3: Normal budget accidentally expands to two full years

That would weaken the low-data setting and muddy the story.

Control:

- enforce `360` total normal points in the builder
- record sampled counts by year in metadata

## Recommended Next Step

Implement the protocol first, then run:

1. builder count validation
2. `Local-Meta-FT` under the new protocol
3. the strongest historical federated baseline under the new protocol

Only after those results land should we decide whether a new class-adaptive federated method is still needed.
