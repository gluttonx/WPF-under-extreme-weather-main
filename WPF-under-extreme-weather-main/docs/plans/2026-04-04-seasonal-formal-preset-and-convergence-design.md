# Seasonal Formal Preset and Convergence Detection Design

**Goal:** Add a reusable formal seasonal training preset and stage-level convergence detection without changing existing optimization behavior.

## Scope

1. Extend `run_six_client_seasonal_protocol.py` with `--preset formal-v1`
2. Keep `--smoke` support for fast health checks
3. Add convergence detection across all training phases in `DemoModelTraining.py`
4. Persist convergence summaries to a JSON report and print concise summaries
5. Record the convergence design in `AGENTS.md` and long-term memory

## Recommended approach

### Launcher preset
Use a named preset instead of hand-written environment variables.

`formal-v1` will inject:
- `SEASONAL_PROTOCOL_ENABLED=1`
- `SEASONAL_PROTOCOL_METADATA_PATH=seasonal_protocol_data/seasonal_protocol_metadata.json`
- `META_SUPPORT_SHOTS=5`
- `META_QUERY_SHOTS=5`
- `PROPOSED_META_SAMPLER_MODE=balanced`
- `PRETRAIN_EPOCHS=4000`
- `PROPOSED_META_EPOCHS=3000`
- `META_ONLY_META_EPOCHS=3000`
- `FEW_SHOT_EPOCHS=20`
- `CONVENTIONAL_RATIO=1.0`
- `REGIME_MISSING_MODE=none`

### Convergence detection
Use non-invasive plateau detection:
- do not early stop
- monitor scalar loss already produced by each loop
- declare convergence when loss fails to improve by `min_delta` for `patience` epochs after `min_epochs`
- record first detected `convergence_epoch`

### Report structure
Persist one JSON file, e.g. `training_convergence_report.json`, containing:
- `run_config`
- one record per training stage with:
  - `stage_type`
  - `stage_id`
  - `total_epochs`
  - `converged`
  - `convergence_epoch`
  - `best_epoch`
  - `best_loss`
  - `final_loss`
  - `patience`
  - `min_delta`

## Why this approach
- Named presets eliminate command drift.
- Convergence monitoring stays orthogonal to the method: it observes training, it does not change optimization.
- JSON is easier to compare across runs than ad-hoc console logs.
