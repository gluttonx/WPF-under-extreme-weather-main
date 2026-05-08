# 2h/6-Point Protocol Runbook

## Worktree

Use the isolated worktree:

```bash
cd /tmp/wpf-worktrees/restore-raw-data-83688a3/WPF-under-extreme-weather-main
```

## Build 2h/6p Data

```bash
PROTOCOL_NAME=three_station_2h_6point_protocol \
PROTOCOL_DATA_DIR=protocol_data/2h_6p \
PROTOCOL_METADATA_PATH=protocol_data/2h_6p/protocol_metadata.json \
SAMPLE_INTERVAL_HOURS=2 DOWNSAMPLE_OFFSET=1 LEN_REALP=6 POINTS_PER_DAY=12 \
python -u build_three_station_extreme_yearly_protocol.py
```

## Smoke Train

Use only for shape validation. Do not interpret metrics.

```bash
PROTOCOL_NAME=three_station_2h_6point_protocol \
PROTOCOL_DATA_DIR=protocol_data/2h_6p \
PROTOCOL_METADATA_PATH=protocol_data/2h_6p/protocol_metadata.json \
SAMPLE_INTERVAL_HOURS=2 DOWNSAMPLE_OFFSET=1 LEN_REALP=6 POINTS_PER_DAY=12 \
PRETRAIN_EPOCHS=1 PROPOSED_META_EPOCHS=1 META_ONLY_META_EPOCHS=1 \
FEW_SHOT_EPOCHS=1 EXTREME_TARGET_REFINEMENT_EPOCHS=1 \
PRETRAIN_LOG_INTERVAL=1 META_LOG_INTERVAL=1 FEW_SHOT_LOG_INTERVAL=1 \
python -u DemoModelTraining.py
```

## Pilot-1k On 4090

```bash
cd /tmp/wpf-worktrees/restore-raw-data-83688a3/WPF-under-extreme-weather-main
mkdir -p logs

PROTOCOL_NAME=three_station_2h_6point_protocol \
PROTOCOL_DATA_DIR=protocol_data/2h_6p \
PROTOCOL_METADATA_PATH=protocol_data/2h_6p/protocol_metadata.json \
SAMPLE_INTERVAL_HOURS=2 DOWNSAMPLE_OFFSET=1 LEN_REALP=6 POINTS_PER_DAY=12 \
PRETRAIN_EPOCHS=1000 PROPOSED_META_EPOCHS=1000 META_ONLY_META_EPOCHS=1000 \
FEW_SHOT_EPOCHS=50 EXTREME_TARGET_REFINEMENT_EPOCHS=50 \
PRETRAIN_LOG_INTERVAL=50 META_LOG_INTERVAL=50 FEW_SHOT_LOG_INTERVAL=10 \
python -u DemoModelTraining.py 2>&1 | tee logs/2h_6p_pilot1k_train_$(date +%Y%m%d_%H%M%S).log

PROTOCOL_NAME=three_station_2h_6point_protocol \
PROTOCOL_DATA_DIR=protocol_data/2h_6p \
PROTOCOL_METADATA_PATH=protocol_data/2h_6p/protocol_metadata.json \
SAMPLE_INTERVAL_HOURS=2 DOWNSAMPLE_OFFSET=1 LEN_REALP=6 POINTS_PER_DAY=12 \
python -u generate_multi_station_results.py 2>&1 | tee logs/2h_6p_pilot1k_eval_$(date +%Y%m%d_%H%M%S).log
```

## Pilot-5k On 4090

Run only if Pilot-1k ranking or convergence is not stable.

```bash
cd /tmp/wpf-worktrees/restore-raw-data-83688a3/WPF-under-extreme-weather-main
mkdir -p logs

PROTOCOL_NAME=three_station_2h_6point_protocol \
PROTOCOL_DATA_DIR=protocol_data/2h_6p \
PROTOCOL_METADATA_PATH=protocol_data/2h_6p/protocol_metadata.json \
SAMPLE_INTERVAL_HOURS=2 DOWNSAMPLE_OFFSET=1 LEN_REALP=6 POINTS_PER_DAY=12 \
PRETRAIN_EPOCHS=5000 PROPOSED_META_EPOCHS=5000 META_ONLY_META_EPOCHS=5000 \
FEW_SHOT_EPOCHS=200 EXTREME_TARGET_REFINEMENT_EPOCHS=200 \
PRETRAIN_LOG_INTERVAL=100 META_LOG_INTERVAL=100 FEW_SHOT_LOG_INTERVAL=20 \
python -u DemoModelTraining.py 2>&1 | tee logs/2h_6p_pilot5k_train_$(date +%Y%m%d_%H%M%S).log

PROTOCOL_NAME=three_station_2h_6point_protocol \
PROTOCOL_DATA_DIR=protocol_data/2h_6p \
PROTOCOL_METADATA_PATH=protocol_data/2h_6p/protocol_metadata.json \
SAMPLE_INTERVAL_HOURS=2 DOWNSAMPLE_OFFSET=1 LEN_REALP=6 POINTS_PER_DAY=12 \
python -u generate_multi_station_results.py 2>&1 | tee logs/2h_6p_pilot5k_eval_$(date +%Y%m%d_%H%M%S).log
```

## Final Candidate On 4090

Choose `10000` or `20000` after inspecting `training_convergence_report.json`. Start with `10000`.

```bash
cd /tmp/wpf-worktrees/restore-raw-data-83688a3/WPF-under-extreme-weather-main
mkdir -p logs

PROTOCOL_NAME=three_station_2h_6point_protocol \
PROTOCOL_DATA_DIR=protocol_data/2h_6p \
PROTOCOL_METADATA_PATH=protocol_data/2h_6p/protocol_metadata.json \
SAMPLE_INTERVAL_HOURS=2 DOWNSAMPLE_OFFSET=1 LEN_REALP=6 POINTS_PER_DAY=12 \
PRETRAIN_EPOCHS=10000 PROPOSED_META_EPOCHS=10000 META_ONLY_META_EPOCHS=10000 \
FEW_SHOT_EPOCHS=500 EXTREME_TARGET_REFINEMENT_EPOCHS=500 \
PRETRAIN_LOG_INTERVAL=200 META_LOG_INTERVAL=200 FEW_SHOT_LOG_INTERVAL=50 \
python -u DemoModelTraining.py 2>&1 | tee logs/2h_6p_final10k_train_$(date +%Y%m%d_%H%M%S).log

PROTOCOL_NAME=three_station_2h_6point_protocol \
PROTOCOL_DATA_DIR=protocol_data/2h_6p \
PROTOCOL_METADATA_PATH=protocol_data/2h_6p/protocol_metadata.json \
SAMPLE_INTERVAL_HOURS=2 DOWNSAMPLE_OFFSET=1 LEN_REALP=6 POINTS_PER_DAY=12 \
python -u generate_multi_station_results.py 2>&1 | tee logs/2h_6p_final10k_eval_$(date +%Y%m%d_%H%M%S).log
```

Do not compare 2h/6p absolute metrics directly against 1h/12p as the same protocol. Compare LMT, Extreme-FedAvg, and Proposed-A within the same 2h/6p run.

## Six-Station Phase-Augmented Pilot-5k On 4090

This is the current next experiment. It treats the complementary 2h sampling phase as added stations `61 / 62 / 63`, and allows all non-self source stations.

```bash
cd /tmp/wpf-worktrees/restore-raw-data-83688a3/WPF-under-extreme-weather-main
mkdir -p logs

python -u run_three_station_yearly_protocol.py build --preset pilot-5k --six-station \
  2>&1 | tee logs/2h6p_six_station_build_$(date +%Y%m%d_%H%M%S).log

python -u run_three_station_yearly_protocol.py train --preset pilot-5k --six-station \
  2>&1 | tee logs/2h6p_six_station_pilot5k_train_$(date +%Y%m%d_%H%M%S).log

python -u run_three_station_yearly_protocol.py eval --preset pilot-5k --six-station \
  2>&1 | tee logs/2h6p_six_station_pilot5k_eval_all6_$(date +%Y%m%d_%H%M%S).log

EVAL_STATION_IDS=58,59,60 \
RESULTS_OUTPUT_PATH=artifacts/2h6p_six_station/pilot-5k/results/multi_station_performance_orig3.csv \
TASK_RESULTS_OUTPUT_PATH=artifacts/2h6p_six_station/pilot-5k/results/multi_station_performance_task_level_orig3.csv \
python -u run_three_station_yearly_protocol.py eval --preset pilot-5k --six-station \
  2>&1 | tee logs/2h6p_six_station_pilot5k_eval_orig3_$(date +%Y%m%d_%H%M%S).log
```

Primary readout:

- `artifacts/2h6p_six_station/pilot-5k/results/multi_station_performance.csv`: all six stations.
- `artifacts/2h6p_six_station/pilot-5k/results/multi_station_performance_orig3.csv`: original target subset, useful for comparison against the previous three-station `2h6p_pilot_5k`.
- `artifacts/2h6p_six_station/pilot-5k/models/`: isolated model outputs, so this run does not overwrite root `.pth` files.
