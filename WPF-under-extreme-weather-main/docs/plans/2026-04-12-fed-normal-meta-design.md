# Fed-Normal-Meta Proposed-A Design

## Goal

Increase the six-station `2h / 6-point / 12h` Proposed-A advantage over LMT from the pilot-5k result of `+1.87%` toward the hard `>=5%` target by moving cross-station information earlier, into the normal-weather meta-learning stage.

## Current Problem

The current six-station pilot already weakens the target-only LMT baseline by reducing temporal information. The remaining gap is that Proposed-A only receives cross-station information in the final extreme-weather adaptation stage. That stage is correct for same-class extreme few-shot transfer, but it is too late and too small to reshape the normal-to-extreme task prior.

Code evidence:

- Local normal meta currently uses target-only tasks via `sample_station_ids=[station_id]`.
- LMT, Extreme-FedAvg, and Proposed-A currently share the same target local-meta initialization before extreme adaptation.

## Proposed Design

Add a Proposed-A-only target-conditioned federated normal-meta stage.

For each target station `s`:

1. Keep the target local pretrain initialization:
   `theta_s^0 = model_fore_pre_station{s}_local.pth`.
2. Train a new `fed_normal_meta_station{s}` checkpoint using normal-weather meta tasks from all stations `58/59/60/61/62/63`.
3. Preserve target identity with self-floor FedAvg:
   `alpha_s >= FED_NORMAL_META_SELF_FLOOR`.
4. Allocate the remaining weight to source stations by normal-weather task-window counts.
5. Use the resulting checkpoint only as the Proposed-A base for extreme aggregation and target refinement.

Output checkpoints:

- `model_fore_train_task_support_fed_normal_meta_station{s}.pth`
- `model_fore_train_task_query_fed_normal_meta_station{s}.pth`

## Method Separation

The first experiment deliberately changes only Proposed-A initialization:

- `LMT`: `local_pretrain_s -> local_meta_s -> target extreme fine-tune`
- `Extreme-FedAvg`: `local_pretrain_s -> local_meta_s -> extreme uniform aggregation`
- `Proposed-A`: `local_pretrain_s -> fed_normal_meta_s -> extreme reliability-aware aggregation`

This keeps the comparison interpretable. If Extreme-FedAvg also uses fed-normal-meta in the first run, the experiment can no longer isolate the Proposed-A mechanism.

## Why This Fits The Current Background

Extreme-stage federation is still appropriate because extreme-weather target windows are scarce and same-class source windows can help. However, the pilot result shows that extreme-stage federation alone is not enough for the `>=5%` target.

Fed-normal-meta is better aligned with the current bottleneck because normal-weather task pools are larger and can learn a stronger cross-station task prior before the extreme few-shot stage begins. The six-station phase augmentation also gives the normal-meta stage more task variation than the original three-station protocol.

## Minimal Experiment

Use the same `pilot-5k` budget and same six-station 2h/6p protocol, but enable:

`ENABLE_FED_NORMAL_META_PROPOSED=1`

Expected artifact root:

`artifacts/2h6p_six_station/fed-normal-meta-pilot-5k/`

Primary go/no-go metric:

- all-6 `Overall_Average` Mean nMAE
- target threshold: `Proposed-A <= 27.4073`
- this equals `>=5%` relative improvement over the pilot LMT Mean nMAE `28.8498`

If this run remains near `2%-3%`, the next high-priority path should be stronger effective-information reduction such as `3h / 4-point / 12h`, not minor final-stage tuning.

## Risks

- Normal-meta FedAvg may over-smooth target-specific phase behavior. The self-floor weight limits this.
- The run adds six extra meta-training stages, so pilot runtime increases.
- If fed-normal-meta helps Extreme-FedAvg equally in a later ablation, the Proposed-A-specific claim must be narrowed. The first run avoids this by keeping Extreme-FedAvg unchanged.

## Verification

Local verification is limited to contract tests, compilation, launcher dry-run, and CPU smoke. Formal evidence must come from a 4090 pilot run.

