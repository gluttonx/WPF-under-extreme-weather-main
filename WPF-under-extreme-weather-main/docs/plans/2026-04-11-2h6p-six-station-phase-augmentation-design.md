# 2h/6p Six-Station Phase Augmentation Design

## Goal

Run a new `2h / 6-point / 12h-window` pilot-5k experiment where the unused complementary 1h sampling phase is converted into three added stations `61 / 62 / 63`. The purpose is to increase the effective number of source stations so `Proposed-A` has a plausible mechanism to open a larger gap over `LMT`.

## Fixed User Constraints

- Treat `61 / 62 / 63` operationally as added stations.
- Do not exclude same-physical complementary phase sources.
- For example, when target station is `58`, source station `61` is allowed.
- Do not use the K-shot target-adapt variants for this pilot. Return to the original `2h6p_pilot_5k` budget.

## Station Mapping

| Station ID | Raw workbook | Raw sheet | Downsample phase |
|---|---|---|---|
| `58` | `2223jilin_058_processed_4classes.xlsx` | `jilin_058` | configured base offset, currently `1` |
| `59` | `2223jilin_059_processed_4classes.xlsx` | `jilin_059` | configured base offset, currently `1` |
| `60` | `2223jilin_060_processed_4classes.xlsx` | `jilin_060` | configured base offset, currently `1` |
| `61` | `2223jilin_058_processed_4classes.xlsx` | `jilin_058` | complementary offset, currently `0` |
| `62` | `2223jilin_059_processed_4classes.xlsx` | `jilin_059` | complementary offset, currently `0` |
| `63` | `2223jilin_060_processed_4classes.xlsx` | `jilin_060` | complementary offset, currently `0` |

The current protocol uses `SAMPLE_INTERVAL_HOURS=2`, so the complementary phase is `(DOWNSAMPLE_OFFSET + 1) % 2`.

## Training Semantics

All six station IDs are loaded from protocol metadata and participate in:

- local pretrain
- local meta train
- LMT extreme adaptation
- Extreme-FedAvg source updates
- Proposed-A source updates and weighted aggregation

The source loop keeps the existing rule `source_station_id != target_station_id`. No same-physical filtering is added.

## Evaluation Semantics

The default evaluation reads all station IDs from metadata and writes a six-station table plus `Overall_Average`.

An optional `EVAL_STATION_IDS=58,59,60` mode is useful for direct comparison with the previous three-station pilot. This is a reporting subset only; it does not change how the six-station models were trained.

## Success Criterion

The result is useful only if the six-station pilot materially improves `Proposed-A vs LMT`. The current hard target remains at least `5%` relative improvement on major error metrics. The analysis must report:

- `Proposed-A vs LMT`
- `Proposed-A vs Extreme-FedAvg`
- six-station overall average
- optional original-target subset average for `58 / 59 / 60`

## Runtime Boundary

Codex runs only smoke and structural validation. The user runs pilot-5k on RTX 4090.

