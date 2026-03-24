# 2026-03-23 Session Summary: Federated + Meta-Learning WPF Progress

## 1. Research Goal and Current Framing

This session focused on a multi-station extension of the root paper's extreme-weather few-shot forecasting pipeline. The intended target problem is not generic federation, but privacy-preserving sharing of source-domain conventional-weather knowledge across stations, followed by station-local few-shot adaptation for extreme weather.

The core comparison remained:
- `Proposed = Federated Pre-Training + Local Meta-Training + Fine-Tune`
- `Local_Meta_Transfer (LMT) = Local Pre-Training + Local Meta-Training + Fine-Tune`

The central question evolved from “can Proposed beat LMT?” to two more precise questions:
1. Under what training regime can Proposed outperform LMT?
2. Under what experimental protocol is federated learning actually necessary rather than just appended to the original single-station paper?

## 2. Two Main Innovation Points

### 2.1 Main Innovation 1: Regime-Aware Federated Pre-Training

A generic FedAvg pre-training stage was judged insufficient. The improved Phase 1 design assigns higher weight to hard/rare/boundary conventional segments and aggregates client updates with a regime-aware client weight.

Client-side weighting:
- Let the raw regime score for segment `i` be composed of:
  - ramp score
  - volatility score
  - rarity score
- After normalization, the sample weight is:

`w_i = 1 + alpha * s_i_tilde`

where `alpha = FED_PRETRAIN_REGIME_ALPHA` and `s_i_tilde` is the normalized regime score.

In code, the score is operationalized as:
- `ramp_score`
- `volatility_score`
- `rarity_score`

and the weighted pre-train loss is a weighted MSE.

Server-side aggregation:
- Each client returns a `regime_factor`
- Aggregation weight is:

`rho_s ∝ n_s * clip(1 + gamma * (regime_factor_s - 1), 0.5, 2.0)`

where `gamma = FED_PRETRAIN_AGGREGATION_GAMMA`.

Implementation locations:
- `DemoModelTraining.py:51`
- `DemoModelTraining.py:612`
- `DemoModelTraining.py:629`
- `DemoModelTraining.py:693`

### 2.2 Main Innovation 2: Prior-Preserving Local Meta-Training

A critical diagnosis from the pilot experiments was that local meta-training can overwrite the federated prior. Therefore, Phase 2 was modified so that Proposed preserves the federated shared prior during station-local episodic learning.

The meta objective becomes:

`L_meta = L_query + beta * ||theta_shared - theta_fed||_2^2`

where:
- `theta_fed` is the federated prior from Phase 1
- `beta = PROPOSED_META_SHARED_ANCHOR_BETA`

At the optimizer level, shared parameters also use a scaled learning rate:

`lr_shared = shared_lr_scale * lr_base`

where `shared_lr_scale = PROPOSED_META_SHARED_LR_SCALE`.

This makes Proposed different from LMT not only in initialization source, but in how local meta-training uses and preserves that initialization.

Implementation locations:
- `DemoModelTraining.py:53`
- `DemoModelTraining.py:877`
- `DemoModelTraining.py:906`
- `DemoModelTraining.py:981`
- `DemoModelTraining.py:1149`

## 3. New Phase-2 Sampling Rule

A later key refinement was the Proposed-only balanced Phase-2 sampler. This turned out to be the most important structural improvement in the session.

### 3.1 Motivation

The remaining bottleneck after the first two innovations was not obviously “Frost as a category”, but poor local task coverage / neighborhood quality in Phase 2, especially under the low-resource setting.

### 3.2 Rule

For Proposed only, keep the original root-paper episodic structure but replace uniform class sampling with weighted sampling without replacement.

For each local conventional class `c`, define:

`coverage_bonus(c) = 1 / (1 + exposure_c)`

where `exposure_c` counts how many of the last `W=4` meta epochs included that class.

`size_bonus(c) = sqrt(mean_class_size / class_size_c)`

Then:

`weight_c = coverage_bonus(c) * size_bonus(c)`

The sampler draws classes without replacement according to `weight_c`.

### 3.3 Why `W=4`

The original root-paper task space is `k=10`, and each epoch samples `k*=5`, i.e. about half the task space. Under uniform sampling, the probability that a class is missed for four consecutive epochs is approximately:

`(1 - 0.5)^4 = 0.0625`

This makes `W=4` a short-window detector for persistent local under-coverage rather than a reaction to normal one-epoch randomness.

Implementation locations:
- `DemoModelTraining.py:50`
- `DemoModelTraining.py:709`
- `DemoModelTraining.py:787`
- `DemoModelTraining.py:794`
- `DemoModelTraining.py:981`
- `DemoModelTraining.py:1149`

## 4. Key Experimental Timeline

### 4.1 Early pilots A/B/C/D

The initial pilot series established the following:
- Proposed can beat LMT, but not stably under plain FedAvg-style federation.
- `Pilot C` was the earlier sweet spot.
- Increasing pre-train and meta budgets naively was non-monotonic.
- This supported the diagnosis that both Phase 1 and Phase 2 needed redesign.

### 4.2 First coupled-innovation 4090 result

With the two main innovation points enabled and full conventional data, Proposed improved and remained stronger than LMT, but still not decisively enough to settle the necessity question.

### 4.3 Conventional-ratio necessity experiments

A controlled `CONVENTIONAL_RATIO` protocol was added with ratios `{1.0, 0.7, 0.5, 0.3}`. Importantly, this protocol preserved all local conventional classes and only reduced sample density.

Main finding:
- As local conventional data became less complete in sample count, Proposed gained relative to LMT.
- This was strongest under `R30`.

Representative trend on `Overall_Average`:
- `R100`: Proposed vs LMT edge `-1.9873`
- `R70`: edge `-2.6114`
- `R50`: edge `-2.3560`
- `R30`: edge `-2.7728`

Interpretation:
- This supported a weak necessity story: federated prior becomes more valuable when local conventional knowledge is incomplete in density.
- However, this still did **not** remove any regime from a station, so it was a density stress test rather than a regime-missing stress test.

### 4.4 Multi-seed low-resource check

`R30` was re-run with multiple subsampling seeds. The trend remained favorable to Proposed relative to `R100`, which gave additional robustness to the low-resource finding.

### 4.5 Strongest positive result of the session

Configuration:
- `CONVENTIONAL_RATIO=0.3`
- `CONVENTIONAL_SUBSAMPLE_SEED_OFFSET=2`
- `PRETRAIN_EPOCHS=4000`
- `PROPOSED_META_EPOCHS=3000`
- `META_ONLY_META_EPOCHS=3000`
- balanced Proposed sampler enabled

Result file:
- `sampler_runs/R30_seed2_balanced_M3000/multi_station_performance.csv`

Main result:
- `Overall_Average`: Proposed vs LMT = `8 wins / 0 losses`
- Mean edge: `-4.2729`
- `station58 = 8/0`
- `station59 = 7/1`
- `station60 = 8/0`

This was the strongest evidence that the new Phase-2 sampling rule solved the remaining low-resource bottleneck without a Frost-specific algorithm.

### 4.6 High-budget reversal

Configuration:
- `R30-seed2`
- balanced sampler
- `PRETRAIN_EPOCHS=8000`
- `PROPOSED_META_EPOCHS=8000`
- `META_ONLY_META_EPOCHS=8000`

Result file:
- `final_runs/R30_seed2_balanced_M8000_P8000/multi_station_performance.csv`

Main result:
- `Overall_Average`: Proposed vs LMT = `0 wins / 8 losses`
- Mean edge: `+4.3742`

Interpretation:
- High budget is strongly non-monotonic under the current coupled setup.
- Proposed worsened moderately, but LMT improved dramatically.
- Therefore, the best low-resource point remained `R30-seed2 balanced M3000`, not the larger-budget configuration.

## 5. Data Diagnosis Work

### 5.1 Frost frequency correction

A mistaken extrapolation from the root paper's single-station table was corrected. In the current three-station data, Frost is still low-frequency:
- station58: `3.4247%`
- station59: `2.1918%`
- station60: `2.6027%`

So the earlier explanation “Frost is not really few-shot” does not hold in the current multi-station data.

### 5.2 Station60-specific bottleneck diagnosis

The remaining issue looked more like station-specific local task-neighborhood weakness than a universal Frost problem. This diagnosis motivated the balanced sampler. Once the balanced sampler was added under `R30-seed2`, station60 moved from being the bottleneck to `8/0` on the main metrics.

## 6. Regime-Missing / Class-Dropout Protocol

To test a stronger necessity story, a new full-conventional-data stress protocol was designed and implemented.

### 6.1 Protocol

- `CONVENTIONAL_RATIO=1.0`
- `REGIME_MISSING_MODE=class_dropout`
- fixed complementary class map:
  - `station58`: drop `{1,2,3,4}`
  - `station59`: drop `{4,5,6,7}`
  - `station60`: drop `{7,8,9,10}`
- no class is missing in all three stations simultaneously
- dropout applies to:
  - Phase 1 local conventional pretrain pool
  - Phase 2 local conventional meta task pool
- extreme few-shot and test remain untouched

After `4-class dropout`, each station retains `6` classes, so local task count is reduced to half the retained pool:

`k*_local = floor(6 / 2) = 3`

Implementation locations:
- `DemoModelTraining.py:51`
- `DemoModelTraining.py:66`
- `DemoModelTraining.py:217`
- `DemoModelTraining.py:249`
- `DemoModelTraining.py:583`
- `DemoModelTraining.py:588`
- `DemoModelTraining.py:858`

### 6.2 Result

Configuration:
- `CONVENTIONAL_RATIO=1.0`
- `REGIME_MISSING_MODE=class_dropout`
- `PRETRAIN_EPOCHS=4000`
- `PROPOSED_META_EPOCHS=3000`
- `META_ONLY_META_EPOCHS=3000`
- balanced Proposed sampler

Result file:
- `regime_missing_runs/fullconv_classdrop_m3000/multi_station_performance.csv`

Main result:
- `Overall_Average`: Proposed vs LMT = `2 wins / 6 losses`
- Mean edge: `+2.2512`
- `station58 = 1/7`
- `station59 = 2/6`
- `station60 = 4/4`

Interpretation:
- This protocol did **not** support federated necessity.
- The most plausible explanation is that the fixed complementary class-dropout simplified the local task space, and LMT benefited from that simplification more than Proposed did.
- Therefore, the new regime-missing protocol currently acts more like a cleaner local-task protocol than a convincing “federation is necessary” protocol.

## 7. Current Best Technical State

### 7.1 Best-performing positive result

For strong low-resource performance, the current best point is:
- `R30-seed2`
- balanced Proposed sampler
- `PRETRAIN=4000`
- `META=3000`

This is the strongest evidence in favor of the new coupled method.

### 7.2 Strongest negative result

For necessity, the strongest current negative evidence is:
- full conventional data
- complementary `4-class dropout`
- balanced Proposed sampler
- `PRETRAIN=4000`
- `META=3000`

This result weakens the claim that federation is necessary in the way currently protocolized.

## 8. Honest Current Conclusion

The session produced a clear split conclusion:

1. **Method improvement conclusion:**
   - The session did produce real method-level progress.
   - The two main innovation points plus the new balanced sampler created a strong low-resource configuration where Proposed clearly beats LMT.

2. **Necessity conclusion:**
   - The current evidence does **not** yet justify a strong “federation is necessary” claim.
   - The class-dropout regime-missing protocol currently weakens rather than strengthens that claim.

Therefore, if a new session resumes from here, it should remember that:
- the coupled method itself has a strong positive configuration,
- but the broader necessity story is currently unresolved and partially contradicted by the newest stress test.

## 9. Files Added/Changed During This Session

Key files:
- `DemoModelTraining.py`
- `tests/test_regime_prior_coupling_ast.py`
- `tests/test_balanced_meta_sampler_ast.py`
- `tests/test_conventional_ratio_necessity_ast.py`
- `tests/test_regime_missing_stress_ast.py`
- `docs/plans/2026-03-22-regime-missing-design.md`

Key AGENTS summaries were appended in nested `AGENTS.md` and should be treated as the canonical short-form experiment log.

## 10. CDRM Formula vs. Code Reality

### 10.1 Historical fact from earliest git versions

A direct inspection of the earliest git history shows that the repository never implemented the paper's CDRM equations as a literal symbol-by-symbol translation. From `first commit 80e6266`, the code already used a proxy penalty:

- define `penalty(logits, y)` by introducing an auxiliary scalar `scale = 1.0`,
- split a batch into two halves by odd/even indexing,
- compute the gradient of each half-loss with respect to `scale`,
- return the gradient inner product `sum(grad_batch1 * grad_batch2)`.

Then both pre-train and meta-train use the form:

- `loss_en = k * penalty(...) + mse(...)`

So the earliest code already chose an engineering proxy for CDRM rather than a literal implementation of paper equations `(18)(19)(20)(21)`.

### 10.2 Why this is only "mechanism-aligned" rather than "formula-identical"

The code and paper align in optimization intent, but not in exact formula form.

Main mismatches:

1. The paper writes gradients with respect to the abstract base learner `eta` under `eta = 1.0`, while the code uses an auxiliary scalar `scale` as a proxy variable.
2. The paper defines the penalty over sampled tasks and two mini-batches `m,n` within a task, while the code simply splits the current tensor batch into odd/even halves.
3. The paper writes CDRM through update equations `(19)(20)(21)`, while the code realizes it as a trainable proxy objective `loss_en = k * penalty + mse`.

Therefore the correct statement is:

- the code is aligned with the paper at the level of optimization mechanism (gradient consistency / invariant-feature pressure),
- but it is not a strict formula-isomorphic implementation.

### 10.3 Warm-up is also historical, not a later artifact

The pre-train CDRM warm-up schedule was already present in the earliest version:

- `<10000 -> k = 0`
- `<20000 -> k = 1`
- `<30000 -> k = 5`
- `>=30000 -> k = 10`

So this schedule is not a late regression. It is part of the original engineering design.

### 10.4 Implication for current experiments

This matters because most experiments in the current session used `PRETRAIN_EPOCHS = 4000` or `8000`. Under the inherited warm-up schedule, Phase-1 CDRM is effectively inactive in those runs.

So many conclusions reached in this session about Phase-1 behavior should be interpreted as conclusions about:

- regime-aware pre-training without active Phase-1 CDRM,

rather than about a fully activated `Regime-aware + CDRM` pre-train design.
