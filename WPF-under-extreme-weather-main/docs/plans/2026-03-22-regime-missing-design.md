# Regime-Missing Stress Test Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a controlled full-conventional-data regime-missing protocol that removes complementary conventional classes from all three stations, so we can test whether federated prior is necessary when each station loses part of its local regime coverage.

**Architecture:** Keep the main `Proposed` / `Local_Meta_Transfer` algorithms unchanged. Add a data-protocol layer before Phase 1 and Phase 2 that drops specified conventional classes per station from both the local pretrain pool and the local meta task pool, while leaving extreme few-shot and test data untouched. Adjust local meta task count to stay proportional to remaining classes.

**Tech Stack:** Python, NumPy, PyTorch, MATLAB `.mat` inputs, existing AST/unittest test suite

---

## Design Summary

### Why this protocol

The current `CONVENTIONAL_RATIO` protocol reduces sample density but preserves all conventional classes for every station. That weakens the necessity test, because `Local_Meta_Transfer` still sees a full local regime set. The new protocol instead removes whole conventional classes from each station so that local regime coverage becomes incomplete while federated stations still retain complementary regime knowledge.

### Experiment definition

- Use full conventional data (`CONVENTIONAL_RATIO=1.0`)
- Apply complementary `4-class dropout` to all three stations
- No class may be missing from all three stations simultaneously
- Apply the dropout to:
  - Phase 1 local conventional pretrain data
  - Phase 2 local conventional meta task pool
- Do **not** apply the dropout to:
  - Extreme few-shot data
  - Test data

### Initial fixed pattern

- `station58`: drop classes `{1, 2, 3, 4}`
- `station59`: drop classes `{4, 5, 6, 7}`
- `station60`: drop classes `{7, 8, 9, 10}`

This pattern creates overlap, ensures all stations lose four classes, and keeps every class available in at least one station.

### Meta protocol adjustment

After `4-class dropout`, each station has `6` local conventional classes left. To preserve the original “sample about half the task space per epoch” logic from `k=10, k*=5`, set local meta task count to `k*=3` under regime-missing mode.

### Scope

The first implementation should be minimal:

- add runtime env switches for regime-missing mode and pattern
- build reduced local pretrain conventional pools from retained class data
- build reduced local meta task pools from retained class data
- keep `Proposed` balanced sampler behavior unchanged
- keep `Local_Meta_Transfer` uniform sampler behavior unchanged

Do not add random pattern generation or multiple stress protocols in the first patch.

### Success criteria

- Runtime with `REGIME_MISSING_MODE=class_dropout` completes on CPU smoke
- Full-data/no-dropout behavior remains unchanged when the new mode is disabled
- Under the new protocol, `Proposed` and `Local_Meta_Transfer` are compared under identical local missing-regime conditions
- Extreme few-shot and test pipelines are unaffected

