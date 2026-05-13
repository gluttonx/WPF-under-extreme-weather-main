# High-Temperature Fed-Meta Mainline Design

## Goal

Determine where federation should sit in the new `High temperature only` summer protocol so the next mainline has a realistic chance to beat the corrected `LMT` baseline by `>=10%` on `Overall_Average`, without relying on fallback or overly aggressive target-side gating.

## Current Evidence

The current high-temperature protocol has a strong stage imbalance.

- Normal/meta data are relatively abundant:
  - each phase station has about `175-178` normal windows
  - the six-station meta task pool is `36` tasks in total
- Extreme fine-tune data are very scarce:
  - `58/61`: `5` support windows
  - `59/62`: `7` support windows
  - `60/63`: `8` support windows

The current pilot also shows that the useful gains are happening before the extreme fine-tune stage:

- prior stage ablation gave sample-weighted `nMAE`
  - `pretrain = 18.7641`
  - `local-meta = 15.5911`
  - `LMT = 22.6703`
- `local-meta` was better than current `LMT` on `5/6` stations
- this means the present target-side `T` stage is high variance and can easily undo a good meta initialization

There is also one known baseline drift that must later be corrected:

- current `LMT` splits tiny target extreme support into `adapt/val`
- the paper-consistent fine-tune should use all available target shots
- this baseline correction should be done before any final claim, but it does not change the stage-selection conclusion above

## Three Candidate Routes

### Route A: Fine-Tune Federation Only

Keep normal pretrain and normal meta local. Introduce federation only in the extreme-weather fine-tune stage.

This is the current family:

- `LMT`
- `Extreme-FedAvg`
- `Proposed-A`

#### Pros

- minimal conceptual change
- easiest to explain relative to the current code
- directly targets same-class source-to-target transfer

#### Cons

- federation acts only on `5/7/8` support windows per station
- the stage is already the noisiest part of the pipeline
- repeated experiments have not produced a large, stable gain over `LMT`
- any apparent gain is likely to be fragile because the available target support is too small

#### Assessment

Not recommended as the main innovation path.

It can remain as an ablation family, but it is no longer the highest-value place to spend algorithmic complexity.

### Route B: Federated Meta-Learning + Local Fine-Tune

Move federation into the normal-weather meta stage, where task volume is much larger and the learned prior is more stable. Keep the final high-temperature fine-tune local and conservative.

Pipeline:

- `local pretrain`
- `fed normal meta`
- `target-only high-temp fine-tune`

#### Pros

- federation happens in the stage with the most data
- directly improves the initialization and task prior, which is where the current gains are already coming from
- keeps the final target adaptation interpretable
- lower risk of negative transfer than extreme-stage-only federation
- easiest route to a real, not cosmetic, `>=10%` improvement target

#### Cons

- requires re-centering the paper narrative around meta rather than T-stage transfer
- adds a second normal-meta branch to the training graph
- demands a stronger baseline set, especially `Local-Meta-NoFT`

#### Assessment

Recommended as the new mainline.

If the objective is to reach a real `10%+` gain over corrected `LMT`, the phase with enough statistical leverage is meta, not T-stage.

### Route C: Federated Meta-Learning + Federated Fine-Tune

Use federation in both normal meta and extreme fine-tune.

Pipeline:

- `local pretrain`
- `fed normal meta`
- `federated high-temp fine-tune`

#### Pros

- highest theoretical ceiling
- can test whether meta-stage federation and extreme-stage federation are complementary

#### Cons

- too many moving parts at once
- if the result improves, attribution is weak
- if the result fails, diagnosis is hard
- most likely to reintroduce the same high-variance behavior already seen in current T-stage federation

#### Assessment

Do not use as the first corrected mainline.

This should be a second-stage enhancement or ablation only after Route B is stable.

## Decision

Use **Route B: Federated Meta-Learning + Local Fine-Tune** as the next mainline.

Do not keep fine-tune-only federation as the main proposed method.

Do not begin with dual-stage federation.

## Proposed Mainline

### Baselines

The corrected comparison set should become:

1. `Pretrain`
2. `Local-Meta-NoFT`
3. `Corrected LMT`

Where:

- `Pretrain`: local summer-normal pretrain only
- `Local-Meta-NoFT`: local summer-normal meta checkpoint evaluated directly on high-temperature test without target fine-tune
- `Corrected LMT`: paper-consistent target fine-tune from local meta, using all available target high-temperature shots

`Local-Meta-NoFT` must be in the main table because current evidence shows it is not a weak side baseline; it is one of the strongest signals in the pipeline.

### Main Proposed Method

The next proposed method should be:

- `Fed-Normal-Meta + Local FT`

Meaning:

- initialization starts from each target station's local pretrain checkpoint
- normal-weather meta tasks are then federated across the six protocol stations
- the resulting target-conditioned fed-meta checkpoint becomes the base model for the final target high-temperature fine-tune
- the final high-temperature fine-tune remains local and conservative

This keeps the innovation concentrated at the phase where data are sufficient.

### Optional Later Ablations

Only after the mainline is working:

1. `Fed-Normal-Meta + Fed-FT`
2. `Local-Meta + Fed-FT`
3. `Fed-Normal-Meta + selective Fed-FT`

These should not be introduced before the fed-meta-only mainline has been validated.

## Federated Meta Design Choice

For the first fed-meta mainline, reuse the existing target-conditioned `Fed-Normal-Meta` skeleton rather than inventing a brand-new federated optimizer.

The preferred semantics are:

- target station keeps a non-trivial self weight
- remaining weight is allocated across source stations
- weighting stays simple and interpretable in the first run

The current self-floor weighted aggregation is acceptable as the first implementation vehicle because the main question right now is **which stage should federate**, not **which federated optimizer is theoretically best**.

## Fine-Tune Design Choice

The final high-temperature stage should be made more conservative than the current implementation.

That means:

- correct the current `adapt/val` split drift and use all target shots for baseline `LMT`
- keep final fine-tune local in the first fed-meta mainline
- do not add extreme-stage federation, fallback, or heavy gating into the first corrected run

This is important because the purpose of the first fed-meta run is to test whether a better prior alone already lifts target adaptation.

## Evaluation Matrix

The first corrected experimental matrix should be:

1. `Pretrain`
2. `Local-Meta-NoFT`
3. `Corrected LMT`
4. `Fed-Normal-Meta + Local FT`

Optional later ablation set:

5. `Local-Meta + Extreme-FedAvg`
6. `Local-Meta + Proposed-A`
7. `Fed-Normal-Meta + Extreme-FedAvg`
8. `Fed-Normal-Meta + Proposed-A`

The first four are the main table. The rest are secondary.

## Success Criteria

### Primary

- `Overall_Average` `HighTemperature_nMAE_%`
- target: proposed mainline beats corrected `LMT` by `>=10%`

### Secondary

- `Overall_Average` `HighTemperature_nRMSE_%`
- `Overall_SampleWeighted` diagnostics
- per-station results for `58-63`

### Safety Condition

- the new mainline must not be worse than `Local-Meta-NoFT` on both overall rows

If a method cannot beat `Local-Meta-NoFT`, it is not a defensible mainline for this protocol.

## Why This Is The Best Bet

This route matches all three empirical realities of the current protocol:

1. the data-rich phase is normal meta
2. the current useful gains already come from meta
3. the target-side high-temperature fine-tune is too data-poor to carry the full innovation burden

So the correct strategic move is not to design a more elaborate extreme-stage federation mechanism first. It is to move the main innovation earlier, where the protocol actually has enough signal to learn from.

## Recommended Next Step

Implement the corrected baseline set and the fed-meta mainline in this order:

1. fix `LMT` target-shot usage
2. expose `Local-Meta-NoFT` in the exported result table
3. run `Fed-Normal-Meta + Local FT` under the current high-temperature-only protocol
4. only then decide whether extreme-stage federation deserves to stay as an ablation family
