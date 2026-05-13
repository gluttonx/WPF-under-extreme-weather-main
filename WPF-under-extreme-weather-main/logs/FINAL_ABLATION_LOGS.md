# Final Ablation Log Manifest

These logs are tied to the final high-wind ablation experiments and should be
kept with the paper results.

## Shared Baseline Checkpoints

- `high_wind_four_client_selective_fed_meta_noft_pilot_1k.log`
  - Source of the early four-client high-wind pilot artifacts.
- `high_wind_spring_noft_selective_fed_meta_noft_pilot_1k.log`
  - Early high-wind spring no-FT pilot record.

## FedPFT

Final table row:
`FedPFT (Federated Pre-training with Personalized Fine-Tuning)`

- `high_wind_fedtl_ft_sweep_train.log`
  - FedTL/FedPFT training and FT sweep setup.
- `high_wind_fedtl_ft_sweep_eval.log`
  - FedTL/FedPFT evaluation log.
- `high_wind_fedtl_ft_sweep_plot.log`
  - FT sweep plot generation log.

Final selected artifacts:

- `artifacts/high_wind_spring_noft_four_client/ablation-fedtl-fedavg-ft50-selected/`
- FT selection evidence:
  - `artifacts/high_wind_spring_noft_four_client/ablation-fedtl-fedavg-ft-sweep-1k/ft_sweep_fedtl.csv`
  - `artifacts/high_wind_spring_noft_four_client/ablation-fedtl-fedavg-ft-sweep-1k/ft_sweep_fedtl.png`

## CDC-FSL

Final table row:
`CDC-FSL (Cross-Domain Consistent Few-Shot Learning)`

- `high_wind_original_local_ft_sweep_train.log`
  - Original local FT sweep up to 100 epochs.
- `high_wind_original_local_ft_sweep_eval.log`
  - Evaluation for the 100-epoch sweep.
- `high_wind_original_local_ft_sweep_plot.log`
  - Sweep plot generation.
- `high_wind_original_local_ft_sweep500_train.log`
  - Extended original local FT sweep up to 500 epochs.
- `high_wind_original_local_ft_sweep500_eval.log`
  - Evaluation for the 500-epoch sweep.
- `high_wind_original_local_ft_sweep500_plot.log`
  - Extended sweep plot generation.

Final selected artifacts:

- `artifacts/high_wind_spring_noft_four_client/ablation-original-local-ft100-selected/`
- FT selection evidence:
  - `artifacts/high_wind_spring_noft_four_client/ablation-original-local-ft-sweep-1k/ft_sweep_original_local.csv`
  - `artifacts/high_wind_spring_noft_four_client/ablation-original-local-ft-sweep-1k/ft_sweep_original_local.png`
  - `artifacts/high_wind_spring_noft_four_client/ablation-original-local-ft-sweep-500/ft_sweep_original_local_500.csv`
  - `artifacts/high_wind_spring_noft_four_client/ablation-original-local-ft-sweep-500/ft_sweep_original_local_500.png`

## TA-F2SL w/o FL

Final table row:
`TA-F2SL w/o FL`

- `high_wind_target_aware_meta_noft_pilot_1k.log`
  - Target-aware pretraining and target-aware local meta training.

Final artifacts:

- `artifacts/high_wind_spring_noft_four_client/pilot-1k-target-aware-meta/`

## TA-F2SL Proposed

Final table row:
`TA-F2SL (Proposed: Target-Aware Federated Few-Shot Learning)`

- `high_wind_target_aware_selective_fed_meta_noft_pilot_1k_train.log`
  - Target-aware selective federated meta-training.
- `high_wind_target_aware_selective_fed_meta_noft_pilot_1k_eval.log`
  - No-calibration evaluation.
- `high_wind_target_aware_selective_fed_meta_noft_biascal_pilot_1k_eval.log`
  - Final bias-calibrated evaluation.
- `high_wind_ta_selective_fed_1k_meta1k_loss_plot.log`
  - Final 1k/1k loss curve plot generation.

Final artifacts:

- `artifacts/high_wind_spring_noft_four_client/pilot-1k-target-aware-selective-fed/`
- `artifacts/high_wind_spring_noft_four_client/pilot-1k-target-aware-selective-fed-biascal/`

## Paper Figures

- `plot_high_wind_power_distribution_raw_base.log`
  - Power distribution figure generation log.

Figure artifacts:

- `artifacts/high_wind_spring_noft_four_client/power_distribution/`

