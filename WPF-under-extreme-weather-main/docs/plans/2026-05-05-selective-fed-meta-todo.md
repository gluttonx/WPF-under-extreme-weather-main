# Selective Fed-Meta TODO

## Goal

Improve `Selective Fed-Normal-Meta-NoFT` over the strong `Local-Meta-NoFT`
baseline by reducing negative transfer while preserving beneficial
cross-station transfer.

## Candidate Changes

1. Raise `SELECTIVE_FED_META_GAIN_MARGIN`.
   - Purpose: reject tiny positive proxy gains that are likely validation noise.
   - First range: `1e-4` to `2e-4`.

2. Add `SELECTIVE_FED_META_TOP_K`.
   - Purpose: keep only the best source stations per target per round.
   - First range: `top_k=1` or `top_k=2`.

3. Raise `SELECTIVE_FED_META_SELF_FLOOR`.
   - Purpose: cap external source influence and make aggregation more conservative.
   - First range: `0.6` to `0.7`.

## First Candidate

Use `gain_margin=1e-4`, `top_k=1`, and `self_floor=0.6`.

This should suppress weak-source accumulation on stations that showed negative
transfer while preserving the strong positive-transfer cases observed on
stations 58 and 60.
