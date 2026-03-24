# Findings

## Core technical outcomes
- Main innovation 1: regime-aware federated pre-training with hard/rare/boundary conventional weighting and regime-aware client aggregation.
- Main innovation 2: prior-preserving local meta-training with shared-parameter anchor loss and reduced shared learning rate.
- Later key refinement: Proposed-only balanced Phase-2 sampler with
  - `coverage_bonus(c) = 1 / (1 + exposure_c)`
  - `size_bonus(c) = sqrt(mean_class_size / class_size_c)`
  - `weight_c = coverage_bonus(c) * size_bonus(c)`
  - fixed `W=4` short-window memory.

## Strongest positive evidence
- `R30-seed2 + balanced sampler + PRETRAIN=4000 + META=3000` achieved `Overall_Average Proposed vs LMT = 8 wins / 0 losses`.

## Strongest negative evidence
- High-budget `R30-seed2 + balanced + 8000/8000` fully reversed to `0 wins / 8 losses`.
- Full-conventional complementary 4-class dropout stress test gave `2 wins / 6 losses`, weakening the broad federated-necessity claim.

## Current best interpretation
- The coupled method and new sampler clearly help under the low-resource `R30` protocol.
- The broader claim that federation is necessary is not currently established under the newer class-dropout stress test.
