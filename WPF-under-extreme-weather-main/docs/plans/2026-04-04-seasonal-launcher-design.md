# Six-Client Seasonal Launcher Design

**Goal:** Add a single entrypoint that can reproducibly build seasonal assets, launch training with the approved six-client seasonal scarcity preset, and run evaluation without hand-editing environment variables.

**Context:** `DemoModelTraining.py` and `generate_multi_station_results.py` already support seasonal mode through environment variables. What is missing is an operational entrypoint that assembles the right preset, exposes a smoke budget, and keeps build/train/eval steps consistent.

## Recommended approach

Add one Python launcher, `run_six_client_seasonal_protocol.py`, with four stages: `build`, `train`, `eval`, and `all`.

The launcher should:
- inject the seasonal protocol env toggles (`SEASONAL_PROTOCOL_ENABLED`, `SEASONAL_PROTOCOL_METADATA_PATH`)
- inject the approved meta-task defaults (`META_SUPPORT_SHOTS=5`, `META_QUERY_SHOTS=5`)
- optionally inject a bounded smoke preset (`PRETRAIN_EPOCHS=1`, `PROPOSED_META_EPOCHS=1`, `META_ONLY_META_EPOCHS=1`, `FEW_SHOT_EPOCHS=1`, `STRICT_PAPER_ORDER=0`)
- support `--dry-run` so commands can be inspected without starting long training
- keep subprocess execution explicit rather than hiding work inside shell heredocs

## Alternatives considered

### A. Shell script wrapper
Pros: shortest.
Cons: weaker argument handling, harder to keep cross-stage env logic readable, less testable.

### B. Modify `DemoModelTraining.py` to self-bootstrap build/eval
Pros: fewer files.
Cons: mixes orchestration into training logic and makes later experiment management worse.

### C. Python launcher (recommended)
Pros: minimal behavioral surface, testable, readable, preserves current training/eval scripts as-is.
Cons: one extra file.

## Data flow

1. `build` runs `build_six_client_seasonal_protocol.py`
2. `train` runs `DemoModelTraining.py` with seasonal preset env
3. `eval` runs `generate_multi_station_results.py` with seasonal preset env
4. `all` runs the three stages in order

## Validation

- Add an AST test asserting the launcher exposes the required stages and seasonal env names.
- Add a `--dry-run` smoke check so the launcher can be exercised quickly without starting long training.
