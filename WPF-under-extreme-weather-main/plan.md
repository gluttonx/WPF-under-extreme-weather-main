# SFML-lite Execution Plan

## Goal
Promote a new strict true-federated mainline for extreme-weather few-shot forecasting:
- `strict FL pre-train`
- `strict federated meta-train`
- `local few-shot`

The server still aggregates only the shared backbone. Each station keeps `LWP + fore_baselearner + few-shot params` local. Episodic support/query tasks are sampled only from each station's own conventional-weather data.

## Steps
1. Add AST coverage for `ENABLE_STRICT_FED_META_TRAIN`, `STRICT_META_USE_SECOND_ORDER`, `client_local_meta_round(...)`, and `run_strict_federated_meta_training(...)`.
2. Add an env override layer so formal 4090 experiments can be launched without hand-editing constants.
3. Demote the old global-task-pool `ENABLE_FED_META_TRAIN` path to legacy pseudo-FL semantics.
4. Reuse `sample_local_meta_task(...)` and local task cache for strict per-station episodic sampling.
5. Implement first-order `SFML-lite` with support adaptation and query-driven client updates, but only upload shared backbone updates.
6. Save per-station pretrain checkpoints and separate per-station strict-meta checkpoints.
7. Route Proposed few-shot initialization to `model_fore_meta_station{station_id}_personalized.pth` when strict federated meta-train is enabled.
8. Run AST tests, syntax checks, and a `1/1/1` CPU-only smoke run plus evaluation.
9. Hand off formal `20/5` and longer 4090 commands to the user for real performance comparison.
10. Only if first-order `SFML-lite` shows stable gains should the project upgrade to full second-order MAML.

## Risks
- Accidentally mixing strict federated meta-train with the old global task pool.
- Uploading or aggregating local/head parameters instead of shared backbone only.
- Confusing pretrain-only checkpoints with strict-meta checkpoints and invalidating evaluation semantics.
- Over-interpreting CPU-only smoke runs as performance evidence.

## Verification
- `python -m unittest tests.test_sfml_lite_ast tests.test_sfml_no_global_pool_ast tests.test_strict_federated_baseline_ast tests.test_fedrep_lite_ast tests.test_pfedfsl_lite_ast tests.test_pfedfsl_routing_utils tests.test_generate_multi_station_results_ast tests.test_f2l_phase2_ast tests.test_device_selection_ast`
- `python -m py_compile DemoModelTraining.py generate_multi_station_results.py pfedfsl_lite_utils.py`
- isolated smoke training with `PRETRAIN_EPOCHS=1 STRICT_META_EPOCHS=1 FEW_SHOT_EPOCHS=1`
- isolated evaluation run with `STRICT_PAPER_ORDER=0`
