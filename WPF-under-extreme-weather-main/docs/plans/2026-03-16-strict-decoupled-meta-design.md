# Strict Decoupled Meta Design

**Date:** 2026-03-16

**Context**

Current `strict` meta-learning is semantically cleaner than the legacy global-task-pool route, but it still trails legacy badly at matched pre-train/meta/fine-tune budgets. Two issues are now clear:

1. `strict` currently updates local meta state during client rounds, then discards most of that gain by rebuilding the final meta initialization from `best shared + pretrain local`.
2. `strict` lets both support and query phases update both shared and local parameters, so shared/local roles are not clearly separated.

The next design direction is a decoupled strict meta-learning variant inspired by the role split in F2L, but without importing F2L's server-visible mutual-information / distillation machinery.

## Goal

Define a `strict` federated meta-learning algorithm where:

- support updates are `local-only`
- query updates are `shared-only`
- server aggregation remains `shared-only`
- final `Proposed` initialization uses `final shared + final local`
- the main algorithm no longer depends on `BEST_ONLY` or `EARLY_STOP` semantics

## Non-Goals

- Do not implement server-visible feature/logit exchange.
- Do not introduce MI/KD losses into the server path.
- Do not change the pre-train or few-shot loss definitions in this phase.
- Do not introduce second-order MAML yet.

## Architecture

### Parameters

Server maintains:

- `phi_t`: global shared backbone

Client `i` maintains locally:

- `psi_i_t`: local meta state, consisting of `LWP + fore_baselearner`
- local meta-task cache sampled from station-local conventional-weather data
- local extreme-weather few-shot state

### Client Round

For a sampled local support/query task on client `i`:

1. Start from `(phi_t, psi_i_t)`.
2. Support phase:
   - freeze shared parameters
   - update local parameters only
   - produce `psi_i_sup`
3. Query phase:
   - freeze local parameters at `psi_i_sup`
   - update shared parameters only
   - produce `phi_i_prime`
4. Client keeps:
   - `psi_i_{t+1} = psi_i_sup`
5. Client sends only:
   - `phi_i_prime`

### Server Round

Server aggregates only shared states:

`phi_{t+1} = sum_i alpha_i * phi_i_prime`

No local state, support/query features, logits, or KD/MI intermediate signals are transmitted to the server.

## Why This Design

### What it keeps from strict

- data stays local
- server remains shared-only
- no global task pool
- no server-visible local parameters

### What it fixes from current strict

- local meta gains are no longer trained and then discarded
- shared and local parameter roles are explicitly separated
- shared updates are forced to represent query-side transferable structure

### What it does not copy from F2L

- no server-visible mutual information proxy
- no server-visible knowledge distillation
- no dual-model representation exchange between client and server

This keeps the design in the user's intended strict federated regime.

## Final Checkpoint Semantics

Mainline algorithm semantics should use fixed training lengths and final checkpoints:

- strict pre-train runs fixed `PRETRAIN_EPOCHS`
- strict meta-train runs fixed `STRICT_META_EPOCHS`
- few-shot runs fixed `FEW_SHOT_EPOCHS`

At the end of strict meta-train:

- save final global shared backbone `phi_T`
- save final client-local meta state `psi_i_T`
- compose each station meta initialization as `combine(phi_T, psi_i_T)`

`STRICT_META_SAVE_BEST_ONLY` and `STRICT_META_EARLY_STOP_PATIENCE` remain diagnostic tools only, not mainline semantics.

## Expected Benefits

- stronger local initialization than current strict, because local meta gains are preserved
- cleaner pressure on shared backbone to absorb cross-client transferable signal
- closer comparison to legacy without relaxing server visibility boundaries

## Main Risks

1. Local-only support updates may dominate, leaving too little learning pressure on shared parameters.
2. Query-only shared updates may be too weak if local adaptation already explains most support/query improvement.
3. Final local-state retention may improve strict substantially, but still remain meaningfully behind legacy because strict still lacks a global task pool.

## Evaluation Criteria

The design should be considered promising if, at fixed budgets:

1. `Proposed` remains consistently better than `Pre_Training`.
2. strict decoupled meta moves materially closer to legacy on `Overall_Average / Proposed`.
3. final-checkpoint training works without relying on `best-only` or `early-stop`.

## Files Expected to Change

- `DemoModelTraining.py`
- `generate_multi_station_results.py` if `Meta_Learning` semantics need explicit alignment
- `tests/test_sfml_lite_ast.py`
- new strict-decoupled tests

## Open Follow-Up Questions

1. Should local state be updated once per task or accumulated across all local tasks in the round?
2. Should query phase update all shared params or only a subset of shared params?
3. Should strict decoupled meta eventually re-introduce a client-local KD/MI-style auxiliary loss, while still keeping the server shared-only?
