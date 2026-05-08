# 2h/6-Point Full-Flow Data Protocol Design

## Goal

Convert the current 1-hour / 12-point, 12-hour-window wind power forecasting protocol into a 2-hour / 6-point protocol across the full pipeline: raw data construction, local pretraining, proposed meta-learning, extreme few-shot refinement, and final evaluation.

The purpose is not to create a micro-tuned few-shot-only stress test. The purpose is to reduce effective temporal information before every training stage, so local pretraining, meta-learning, LMT few-shot adaptation, Extreme-FedAvg, and Proposed-A are all trained and evaluated under the same lower-resolution data condition.

## Core Protocol

The 12-hour physical forecasting window is preserved. Only the sampling resolution changes.

| Protocol | Sampling interval | Points per 12h window | Points per day | Expected windows per 365-day year |
| --- | ---: | ---: | ---: | ---: |
| Current | 1h | 12 | 24 | 730 |
| New | 2h | 6 | 12 | 730 |

Use a fixed downsampling phase. To stay consistent with the earlier sparse experiment discussion, the default phase should be `offset=1`: after sorting each sheet by timestamp, keep rows with `row_index % 2 == 1`. This must be applied to all relevant sheets, not only extreme-weather sheets:

- main full series: `jilin_058`, `jilin_059`, `jilin_060`
- normal weather: `normal_weather`
- extreme classes: `extreme_high_wind`, `extreme_high_temp`, `extreme_cold_wave`, `extreme_frost`
- future external-test sheets if 2024 is later enabled

The key invariant is:

```text
window_span_hours = sample_interval_hours * len_realp = 2 * 6 = 12h
```

So this is not the previous “thin half the support windows” experiment. It keeps the task semantics as 12-hour prediction, but each window now contains six observations instead of twelve.

## Scope

This protocol must rebuild all training and evaluation data from source xlsx files. It should not reuse any 1h/12-point checkpoints for final results, because those checkpoints were trained with a different temporal resolution and different information content.

The migrated `4.6.17` files provide the right source-data chain:

- `2223jilin_058_processed_4classes.xlsx`
- `2223jilin_059_processed_4classes.xlsx`
- `2223jilin_060_processed_4classes.xlsx`
- `24jilin_058_processed_4classes.xlsx`
- `24jilin_059_processed_4classes.xlsx`
- `24jilin_060_processed_4classes.xlsx`
- `2223jilin_58-60.py`
- `build_three_station_extreme_yearly_protocol.py`
- `run_three_station_yearly_protocol.py`

For the first minimum experiment, use 2022/2023 data only, matching the current paper-style setting. Keep 2024 as an optional external validation set, not as training data, unless we explicitly define a second protocol such as “train on 2022/2023, external-test on 2024”.

## Recommended Implementation Route

Use `build_three_station_extreme_yearly_protocol.py` as the source-of-truth builder, not the repeated legacy `2223jilin_58-60.py` script. The legacy script is useful as a compatibility reference, but it duplicates station logic three times and is easier to desynchronize.

The builder should support protocol parameters:

```text
SAMPLE_INTERVAL_HOURS=2
DOWNSAMPLE_OFFSET=1
LEN_REALP=6
POINTS_PER_DAY=12
WINDOW_SPAN_HOURS=12
```

The generated `.mat` files should keep old key names where possible, because `DemoModelTraining.py` and `generate_multi_station_results.py` currently expect keys such as:

- `p_1h`, `nwp_1h`
- `p_conven`, `nwp_conven_`
- `p_conven_class`, `nwp_conven_class_`
- `p_extre_class1` to `p_extre_class4`
- `nwp_extre_class1_` to `nwp_extre_class4_`

However, the metadata must explicitly state that these are 2h-sampled arrays even if the legacy key names still say `1h`. Add metadata fields:

```json
{
  "sample_interval_hours": 2,
  "downsample_offset": 1,
  "len_realp": 6,
  "points_per_day": 12,
  "window_span_hours": 12,
  "protocol_name": "three_station_2h_6point_protocol"
}
```

For the formal version, prefer adding explicit aliases such as `p_2h` and `nwp_2h` while still writing `p_1h` and `nwp_1h` for backward compatibility. That avoids ambiguity in future analysis.

## Files To Modify

### `build_three_station_extreme_yearly_protocol.py`

This is the main data protocol file.

Required changes:

- Add protocol constants or env/CLI arguments for `SAMPLE_INTERVAL_HOURS`, `DOWNSAMPLE_OFFSET`, `LEN_REALP`, and `POINTS_PER_DAY`.
- Add a single helper, conceptually `downsample_records(records, interval, offset)`, applied after sorting and after year split, so 2022 support and 2023 test do not share boundary assumptions.
- Apply downsampling to main, normal, and all extreme-weather sheets before clustering/window counting.
- Recompute conventional weather KMeans clusters on downsampled normal weather, not on original 1h normal weather.
- Save protocol metadata with interval/window fields and per-station counts.
- Either write protocol-specific files such as `58wf_2h_6p_train.mat`, or write into an isolated run directory where files are named `58wf_4_train.mat` for compatibility.

Minimum-compatible output path option:

```text
protocol_data/2h_6p/58wf_4_train.mat
protocol_data/2h_6p/59wf_4_train.mat
protocol_data/2h_6p/60wf_4_train.mat
protocol_data/2h_6p/protocol_metadata.json
```

This avoids overwriting the existing 1h mats.

### `DemoModelTraining.py`

This is where the old 1h assumptions currently matter most.

Required changes:

- Replace hardcoded `len_realp=12` with a protocol-driven value, defaulting to `12` for existing runs.
- Replace hardcoded `d=24` with protocol-driven `POINTS_PER_DAY`, defaulting to `24`.
- Add an env var such as `PROTOCOL_DATA_DIR`; if set, load `58wf_4_train.mat`, `59wf_4_train.mat`, and `60wf_4_train.mat` from that directory instead of the current working directory.
- Log the active protocol at startup: sample interval, `len_realp`, points/day, data dir, and metadata path.
- Keep `dem_realc=5` and `dem_realp=1` unchanged.
- Keep the model architecture unchanged for the minimum experiment. The TCN/linear head can accept a different sequence length because there is no fixed-size dense layer over the time dimension.

Important model note: state dicts are technically loadable across 1h/12 and 2h/6 because convolution weights do not encode sequence length. They are still semantically incompatible for final comparison. Do not initialize final 2h runs from 1h checkpoints.

### `generate_multi_station_results.py`

Evaluation must use the same protocol metadata as training.

Required changes:

- Replace hardcoded `len_realp=12` with metadata/env-driven `LEN_REALP=6` for 2h protocol.
- Load `.mat` files from `PROTOCOL_DATA_DIR`.
- Include protocol fields in `multi_station_performance.csv` and `multi_station_performance_task_level.csv`, for example `Protocol`, `Sample_Interval_Hours`, `Window_Points`, `Window_Span_Hours`.
- Avoid silently loading stale root-level `58wf_4_train.mat` if `PROTOCOL_DATA_DIR` is set.
- Keep evaluation within the same protocol. A 2h/6 model should be evaluated on 2h/6 windows.

Formal-version evaluation should prefer explicit 2023 extreme test keys, if available, instead of evaluating on the same `p_extre_class*` arrays used for target adaptation. The migrated yearly builder already has the concept of `p_test_extre_class*` and `nwp_test_extre_class*_`; those should be wired into evaluation before paper-grade reporting.

### `run_three_station_yearly_protocol.py`

This launcher should expose separate presets for 2h/6-point experiments.

Required changes:

- Add protocol env vars to `PREVIEW_ENV_KEYS`.
- Add a `2h-6p` preset or CLI flag that sets `SAMPLE_INTERVAL_HOURS=2`, `DOWNSAMPLE_OFFSET=1`, `LEN_REALP=6`, `POINTS_PER_DAY=12`, and `PROTOCOL_DATA_DIR=protocol_data/2h_6p`.
- Add reduced epoch presets based on observed convergence behavior instead of preserving the old 80000/70000 defaults.

### `2223jilin_58-60.py`

Do not use this as the main formal builder unless necessary. If kept, only refactor it to call shared downsampling utilities or mark it as legacy.

The danger is that this script can generate root-level `58wf_4_train.mat` files with no protocol metadata, making it easy to mix 1h and 2h results accidentally.

## Epoch Strategy

The previous full run used:

```text
PRETRAIN_EPOCHS=80000
PROPOSED_META_EPOCHS=70000
FEW_SHOT_EPOCHS=500
EXTREME_TARGET_REFINEMENT_EPOCHS=500
```

Based on prior convergence checks, these are too expensive for protocol development and likely excessive for the final 2h experiment. The 2h protocol changes the data distribution enough that we still need fresh pilot curves, but we do not need to start with 80000/70000.

Recommended stages:

| Stage | Purpose | Pretrain | Proposed meta | Few-shot | Target refinement |
| --- | --- | ---: | ---: | ---: | ---: |
| Smoke | Verify shape/data path only | 1-5 | 1-5 | 1-2 | 1-2 |
| Pilot-1k | Check loss behavior quickly | 1000 | 1000 | 50 | 50 |
| Pilot-5k | Check ranking stability | 5000 | 5000 | 200 | 200 |
| Final candidate | Paper-grade candidate if curves stabilize | 10000-20000 | 10000-20000 | 500 | 500 |

The final candidate should be chosen from pilot convergence evidence, not from the old constants. If pretrain/meta losses flatten early under 2h/6, use the smaller value. If they are still decreasing materially, extend only the stage that still has useful improvement.

Suggested initial pilot command style:

```bash
PROTOCOL_DATA_DIR=protocol_data/2h_6p \
SAMPLE_INTERVAL_HOURS=2 DOWNSAMPLE_OFFSET=1 LEN_REALP=6 POINTS_PER_DAY=12 \
PRETRAIN_EPOCHS=1000 PROPOSED_META_EPOCHS=1000 \
FEW_SHOT_EPOCHS=50 EXTREME_TARGET_REFINEMENT_EPOCHS=50 \
python -u DemoModelTraining.py
```

Do not run this as the first step. Run builder smoke and shape smoke first.

## What Can And Cannot Be Compared Directly

Can compare directly:

- LMT vs Extreme-FedAvg vs Proposed-A within the same 2h/6 protocol.
- Different 2h/6 epoch settings, if the data, split, seed, and evaluation keys are identical.
- 2h/6 smoke/pilot/final convergence curves within the same protocol, as training diagnostics.

Cannot directly compare as the same table row:

- Absolute nMAE/nRMSE from 1h/12 vs absolute nMAE/nRMSE from 2h/6 as if only the algorithm changed.
- 1h/12 checkpoints evaluated against 2h/6 data.
- 2h/6 checkpoints evaluated against 1h/12 data unless the experiment is explicitly framed as cross-resolution transfer.
- Results generated from root-level stale `58wf_4_train.mat` against results generated from `protocol_data/2h_6p`.

Acceptable paper framing:

- Main 2h/6 table: “Under reduced 2-hour sampling resolution, Proposed-A improves over LMT and Extreme-FedAvg by X%.”
- Optional appendix stress test: 1h/12 vs 2h/6 shows that lower temporal resolution increases task difficulty, but algorithm rankings must be discussed within each protocol.

## Risks And Controls

Risk: The experiment name says 2h/6, but training/evaluation still use 1h/12 root mats.

Control: Print protocol metadata at build, train, and eval startup. Add tests that fail if `LEN_REALP=6` but `POINTS_PER_DAY=24`, or if `PROTOCOL_DATA_DIR` is set but the script loads root mats.

Risk: Downsampling after concatenating years changes the first sample phase of 2023 based on 2022 length.

Control: Split by year first, then downsample each year/sheet with the same fixed offset.

Risk: KMeans conventional classes are accidentally inherited from 1h data.

Control: Run KMeans after downsampling normal weather.

Risk: 2024 data accidentally enters training.

Control: Treat 2024 as external validation only. Do not include it in `p_1h`, `p_conven`, `p_conven_class`, or extreme support unless a separate protocol explicitly says so.

Risk: Final epoch counts are still oversized.

Control: Save and inspect `training_convergence_report.json` for each pilot. Select final epochs based on plateau behavior, with 10000-20000 as the expected upper range unless the pilot contradicts it.

## Minimum Test Plan

1. Builder smoke: generate `protocol_data/2h_6p/*.mat` and metadata only.
2. Shape checks: verify each station has `len_realp=6`, `points_per_day=12`, and the 2023 test reshape gives 730 windows.
3. Extreme window checks: verify each class has nonzero support/test windows and no accidental empty class after downsampling.
4. Training smoke: run 1-5 epochs only and confirm no shape error in pretrain, meta, LMT, Extreme-FedAvg, Proposed-A.
5. Eval smoke: generate CSV and confirm all rows include protocol columns and `Samples` are derived from 6-point windows.
6. Pilot-1k: run 1000/1000/50/50 and inspect convergence and ranking.
7. Pilot-5k: run only if Pilot-1k ranking or loss curves are unstable.
8. Final candidate: run reduced final epochs selected from pilot evidence.

## Initial Recommendation

Proceed with the 2h/6 protocol, but do not treat it as a direct continuation of the existing 1h/12 checkpoints. The strongest design is:

1. Generate isolated 2h/6 protocol data from source xlsx.
2. Parameterize `DemoModelTraining.py` and `generate_multi_station_results.py` so they cannot silently use 12-point assumptions.
3. Run smoke, then 1k pilot, then 5k pilot only if needed.
4. Choose final epochs from convergence evidence, likely far below `80000/70000`.

