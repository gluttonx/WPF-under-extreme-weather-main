# Hybrid 4h3p Normal 2h6p Extreme Design

Goal: reuse finite 4h/3p normal-stage checkpoints while rerunning only extreme-weather few-shot and eval on 2h/6p data.

Design:
- Keep `local_pretrain`, `local_meta`, and `Fed-Normal-Meta` checkpoints from `artifacts/4h3p_six_station/fed-normal-meta-self08-best-5k/models`.
- Run the training script with 2h/6p six-station protocol data for the extreme stage, and skip all normal-stage training.
- Add `BASE_MODEL_OUTPUT_DIR` so base checkpoints can be read from the existing 4h/3p artifact while new LMT / Extreme-FedAvg / Proposed-A extreme checkpoints are written to a separate hybrid artifact directory.
- Add launcher flag `--hybrid-extreme-2h` to make the mixed data protocol explicit in logs and results.
- Fix one-window extreme splits so one 12h window remains a full temporal sequence rather than becoming `T=1`.
- Add non-finite loss/state guards in few-shot adaptation to avoid saving NaN-poisoned checkpoints.

Validation:
- AST tests cover launcher env, base checkpoint path separation, one-window split behavior, and non-finite guards.
- Dry-run must show `PROTOCOL_NAME=hybrid_4h3p_normal_2h6p_extreme_protocol`, 2h/6p extreme runtime config, `BASE_MODEL_OUTPUT_DIR` pointing at the 4h/3p model dir, and skip flags enabled.
