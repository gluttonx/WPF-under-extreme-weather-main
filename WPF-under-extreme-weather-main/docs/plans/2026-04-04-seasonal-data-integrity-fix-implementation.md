# Seasonal Data Integrity Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix seasonal asset corruption caused by repeated normalization and regenerate all six seasonal `.mat` files with verified correct values.

**Architecture:** Add a detaching copy step at workbook merge time so client serialization works on isolated sheet records. Lock the bug with one aliasing unit test and one sequential-build integrity test using the real seasonal protocol configuration.

**Tech Stack:** Python 3.8, `unittest`, `scipy.io`, existing `build_six_client_seasonal_protocol.py` pipeline.

---

### Task 1: Add failing tests for aliasing and corrupted scale

**Files:**
- Create: `WPF-under-extreme-weather-main/tests/test_seasonal_data_integrity.py`
- Modify: `WPF-under-extreme-weather-main/build_six_client_seasonal_protocol.py`

**Step 1: Write the failing test**
- Test that `merge_workbooks_by_sheet(...)` returns detached `SheetRecord` copies.
- Test that sequential serialization of all six clients does not crush `client63` conventional targets below realistic scale.

**Step 2: Run test to verify it fails**
- Run: `python -m unittest WPF-under-extreme-weather-main/tests/test_seasonal_data_integrity.py -v`

**Step 3: Write minimal implementation**
- Add a record-cloning helper and use it inside merge.

**Step 4: Run test to verify it passes**
- Run the same unittest command.

### Task 2: Rebuild seasonal assets and compare all six clients

**Files:**
- Modify: `WPF-under-extreme-weather-main/build_six_client_seasonal_protocol.py`

**Step 1: Rebuild**
- Run: `cd /root/WPF-under-extreme-weather-main/WPF-under-extreme-weather-main && python build_six_client_seasonal_protocol.py`

**Step 2: Verify all six client means against fresh single-pass expectations**
- Run a one-off verification script comparing `.mat` conventional values to `.xlsx`-derived expectations.

### Task 3: Record the fix

**Files:**
- Modify: `WPF-under-extreme-weather-main/AGENTS.md`

**Step 1: Append a `##` section**
- Document the root cause, fix, and requirement that workbook cache objects must never be mutated across client serialization.

**Step 2: Save long-term memory**
- Save a high-priority decision summary of the data-integrity fix.
