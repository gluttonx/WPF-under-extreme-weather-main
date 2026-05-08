# High-Temperature Selective Fed-Meta Design

## Goal

Define the real innovation point of the new `High temperature only` mainline inside the federated normal-meta stage, so the proposed method is not just a mild `FedAvg` variant but a target-aware cross-station transfer mechanism with a defensible reason to outperform corrected `LMT`.

## Positioning

The method family is no longer:

- local normal meta
- federation only in extreme fine-tune

The new proposed family is:

- local pretrain
- selective federated normal meta-learning
- local high-temperature fine-tune

The core claim is therefore:

> Cross-station transfer should happen in the data-rich normal-meta stage, and source stations should be admitted into aggregation only when their current meta update is validated to help the target station on target-side normal proxy tasks.

## Why Vanilla Fed-Meta Is Not Enough

The existing `Fed-Normal-Meta` skeleton already provides:

- personalized or target-conditioned training
- self-floor protection for the target station
- weighted aggregation by source data volume

That is useful as a baseline, but it does **not** solve the main risk:

- a source station may have enough normal data to receive non-trivial weight
- but its current meta update may still be harmful to the target station because the source-to-target normal regime mismatch is too large

So the missing innovation is **not** more federation intensity.  
The missing innovation is **target-side validation of source usefulness before aggregation**.

## Method Name

Use the following working method name:

`Target-Proxy-Validated Selective Federated Meta-Learning`

Short name:

`Selective Fed-Meta`

If the final method name needs to stay close to the current paper language, use:

`Selective Fed-Normal-Meta`

## Method Definition

For each target station `s`, train a separate personalized federated meta model:

- `fed_normal_meta_station58`
- `fed_normal_meta_station59`
- `fed_normal_meta_station60`
- `fed_normal_meta_station61`
- `fed_normal_meta_station62`
- `fed_normal_meta_station63`

This is still personalized or target-conditioned federated meta-learning.

The new innovation is that **source participation is not unconditional**.

In each federated meta round:

1. every client computes a local meta update from the same target-specific server model
2. the server evaluates each returned client update on target-side normal proxy tasks
3. the server rejects harmful sources
4. the server softly reweights the remaining helpful sources
5. the server aggregates only the accepted updates

## Data Partition Inside The Normal-Meta Stage

For each station `s`, the available `2023-06~08 normal_weather` windows are split once into:

- `D_s^meta-train`
- `Q_s^proxy`

### `D_s^meta-train`

Used by client `s` for local federated meta updates.

### `Q_s^proxy`

Held out for target-side proxy validation only.

It is **not** used for the local client update in the same round.

## Recommended Split

Use a fixed station-wise split:

- `80%` of normal windows for `meta-train`
- `20%` of normal windows for `proxy`

Rules:

- split at the window level, not the raw point level
- use a fixed random seed
- keep the partition fixed during the whole run

Reason:

- the proxy set must be stable enough to compare candidate source updates across rounds
- the summer normal pool is large enough to afford this holdout

## Model State And Trainable Parameters

The server model for target station `s` at federated meta round `r` is:

- `θ_s^r`

This is the **complete** `model_fore` parameter state.

It includes:

- TCN backbone parameters
- LWP parameters
- final prediction head parameters

However, during local meta updates, the client only optimizes the current meta-trainable subset:

- all LWP parameters
- the final `fore_baselearner` parameters

So:

- the server aggregates full `state_dict`s
- the local meta gradient updates only touch `LWP + head`

## Communication Flow

For a fixed target station `s`, one federated meta round works as follows.

### Step 1: Server broadcast

The server holds the current target-specific model:

- `θ_s^r`

The server broadcasts this same full model state to all six clients:

- `58`
- `59`
- `60`
- `61`
- `62`
- `63`

### Step 2: Client local meta update

Each client `c`:

- receives `θ_s^r`
- samples meta tasks only from its own `D_c^meta-train`
- runs one local support-to-query meta update
- returns an updated model:
  - `φ_{s,c}^r`

This local update remains episode-level and lightweight.

The client also returns its own local query loss for logging, but that loss is **not** the main admission criterion.

### Step 3: Target-side proxy evaluation

Before aggregation, the server evaluates each returned model `φ_{s,c}^r` on the target station proxy tasks `Q_s^proxy`.

This produces:

- `L_proxy(φ_{s,58}^r)`
- `L_proxy(φ_{s,59}^r)`
- ...
- `L_proxy(φ_{s,63}^r)`

### Step 4: Source gain computation

Use the target self-update as the reference:

```text
gain_{s,c}^r = L_proxy(φ_{s,s}^r) - L_proxy(φ_{s,c}^r)
```

Interpretation:

- `gain > 0`: source `c` is better than the target self-update on target proxy tasks
- `gain = 0`: no extra benefit relative to the self-update
- `gain < 0`: source `c` is harmful in this round

This is the key target-aware transferability signal.

### Step 5: Hard rejection

Define the accepted source set:

```text
K_s^r = { c != s | gain_{s,c}^r > m }
```

Recommended first-round margin:

- `m = 0`

Meaning:

- if a source cannot beat the target self-update on target proxy tasks, it is rejected from aggregation for this round

### Step 6: Soft weighting

For accepted sources only, define:

```text
w_{s,c}^r ∝ max(gain_{s,c}^r, 0)^γ
```

Recommended first-round exponent:

- `γ = 1`

Then combine this with self-floor:

```text
α_{s,s} = ρ
α_{s,c} = (1 - ρ) * w_{s,c}^r / Σ_{j∈K_s^r} w_{s,j}^r
```

Recommended first-round self-floor:

- `ρ = 0.5`

If no source passes rejection:

- `α_{s,s} = 1`
- all source weights are `0`

### Step 7: Server aggregation

The server updates the target model by weighted averaging:

```text
θ_s^{r+1} = α_{s,s} φ_{s,s}^r + Σ_{c∈K_s^r} α_{s,c} φ_{s,c}^r
```

This becomes the server state for the next round.

## What The Innovation Is And Is Not

### The innovation is

- target-side proxy validation of each source update
- round-wise source rejection based on measured target benefit
- soft reweighting among only the accepted sources

### The innovation is not

- personalization by itself
- self-floor by itself
- source weighting purely by data volume
- more aggressive extreme-stage federation

So the contribution should be framed as:

> A target-proxy-validated selective aggregation rule embedded into personalized federated normal meta-learning.

## Why Normal Proxy Instead Of Extreme Proxy

The selective criterion should be attached to the normal-meta stage, not the high-temperature stage, because:

- the normal-weather pool is much larger and more stable
- the high-temperature support windows are too few to support reliable source admission every round
- the current empirical evidence already shows that the meta stage is where the useful gains are coming from

So the proxy is deliberately chosen from the same stage where the federation acts.

## Primary Baselines

The main comparison table should become:

1. `Pretrain`
2. `Local-Meta-NoFT`
3. `Corrected LMT`
4. `Vanilla Fed-Normal-Meta + Local FT`
5. `Selective Fed-Normal-Meta + Local FT`

This is important because:

- row 4 shows whether moving federation to meta helps at all
- row 5 shows whether selective source validation is the real extra innovation

## Secondary Diagnostics

Keep the following internal diagnostics even if they do not all enter the main paper table:

1. `Fed-Normal-Meta-NoFT`
2. number of accepted sources per target per round
3. acceptance frequency of each source station
4. average proxy gain per source-target pair

These diagnostics are crucial for showing that the selective mechanism is actually doing something meaningful.

## Recommended First-Round Hyperparameters

Use the simplest defensible setting first:

- proxy split ratio: `20%`
- self-floor: `ρ = 0.5`
- gain margin: `m = 0`
- gain exponent: `γ = 1`
- no top-k filtering
- no exponential moving average smoothing
- no cluster-level or task-level routing inside a station

This keeps the method interpretable.

## Optional Later Extensions

Only after the first selective version works:

1. `top-k` accepted sources
2. `γ > 1` to emphasize stronger sources
3. gain smoothing across rounds
4. station-cluster-level selective aggregation
5. later coupling with extreme-stage federation

These should not be part of the first proposed method.

## Research Recommendation

### Method definition

Define the final proposed method directly as:

- `Selective Fed-Normal-Meta + Local FT`

### Engineering sequence

The implementation can still proceed in two stages:

1. run vanilla `Fed-Normal-Meta + Local FT` first as a smoke-proof intermediate step
2. then add target-proxy-validated selective aggregation

But for the actual main experiment narrative:

- vanilla fed-meta is a baseline
- selective fed-meta is the proposed method

## Success Criterion

The main success criterion remains:

- `Overall_Average HighTemperature_nMAE_%`
- proposed selective fed-meta must beat corrected `LMT` by `>=10%`

Additional safety rule:

- the proposed method must not underperform `Local-Meta-NoFT`

If it cannot beat `Local-Meta-NoFT`, then the final target fine-tune stage is still too destructive and must be revisited.
