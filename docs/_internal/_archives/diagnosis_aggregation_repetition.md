# Diagnosis: Aggregation / Repetition Feature

**Date**: 2026-03-24
**Status**: Bug confirmed — aggregation silently disabled when only `repetition` is set

---

## 1. Feature Overview

The repetition/aggregation feature addresses a fundamental NIRS problem: multiple spectral measurements (repetitions) of the same physical sample. It has two related but distinct sub-features:

**A. Repetition (fold splitting):** Ensures all spectra from the same physical sample stay in the same fold during cross-validation, preventing data leakage. Configured via `DatasetConfigs(repetition="column_name")` or `dataset.set_repetition("column_name")`.

**B. Aggregation (prediction reporting):** Combines predictions from multiple repetitions into one prediction per physical sample, then reports both raw and aggregated metrics in a secondary tab. Configured via `dataset.set_aggregate("column_name")` or config dict `"aggregate": "column_name"`.

These are stored as **separate properties** on `SpectroDataset`:
- `_repetition` (str|None) — for fold grouping
- `_aggregate_column` / `_aggregate_by_y` — for prediction aggregation in reports

---

## 2. Current Data Flow

### 2.1 Repetition flow (fold grouping) — WORKS

```
DatasetConfigs(repetition="sample_id")
  → _get_dataset_with_types() → dataset.set_repetition("sample_id")
  → CrossValidatorController.execute()
    → compute_effective_groups(dataset)  // reads dataset.repetition
    → GroupedSplitterWrapper(splitter) wraps the inner splitter
    → Folds correctly group samples by repetition column
  → Predictions.set_repetition_column("sample_id")  // orchestrator.py:548-549
    → predictions.top(by_repetition=True) works correctly
```

### 2.2 Aggregation flow (prediction reports) — BROKEN

```
DatasetConfigs(repetition="sample_id")  // sets repetition but NOT aggregate
  → _get_dataset_with_types() → dataset.set_repetition("sample_id")
  → dataset.aggregate → returns None !!
  → orchestrator.py:686: self.last_aggregate_column = dataset.aggregate  // gets None
  → runner.last_aggregate → None
  → Tab reports: aggregate_column=None → no aggregated rows in output
  → PredictionAnalyzer(predictions, default_aggregate=runner.last_aggregate) → None
```

---

## 3. Bug Found

### Primary Bug: `repetition` does not automatically enable `aggregate` for reporting

**Location:** `nirs4all/data/config.py:316-318`

When `repetition="sample_id"` is set via the `DatasetConfigs` constructor, `set_repetition()` is called but `set_aggregate()` is NOT called. The dataset's `aggregate` property remains `None`, so:

1. **Tab reports** don't show aggregated rows (the `*` rows are missing)
2. **`runner.last_aggregate`** returns `None` (`pipeline/runner.py:462`)
3. **`PredictionAnalyzer`** doesn't aggregate by default

There IS a one-directional link at `config.py:128-129`:

```python
# If repetition not in config but aggregate is a string, use it as repetition
if config_repetition is None and isinstance(config_aggregate, str):
    config_repetition = config_aggregate
```

This maps `aggregate → repetition` (setting aggregate also enables fold grouping), but the **reverse** mapping (`repetition → aggregate`) does NOT exist. This asymmetry is the root cause.

### Secondary Issue: `DatasetConfigs` constructor lacks `aggregate` parameter

`DatasetConfigs.__init__` (`config.py:24`) accepts `repetition` as a constructor parameter but NOT `aggregate`. The `aggregate` value can only be set via config dict (`"aggregate": "sample_id"`) or by calling `dataset.set_aggregate()` after construction. This is a design inconsistency.

---

## 4. What Still Works

| Component | Status | Notes |
|-----------|--------|-------|
| Fold grouping via repetition | ✅ Works | `GroupedSplitterWrapper` correctly groups samples |
| `predictions.repetition_column` | ✅ Works | Set at `orchestrator.py:548-549` |
| `predictions.top(by_repetition=True)` | ✅ Works | Sorts and filters correctly |
| `Predictions.aggregate()` static method | ✅ Works | Aggregation logic is correct |
| Tab report aggregation (when given column) | ✅ Works | `TabReportManager` handles it properly |
| Rep-to-sources / rep-to-pp transforms | ✅ Works | `RepetitionController` functions correctly |

The feature itself is intact — the bug is purely in the **config wiring** between `repetition` and `aggregate`.

---

## 5. Proposed Fix

In `nirs4all/data/config.py`, method `_get_dataset_with_types()` at lines 316-318, auto-enable aggregate when repetition is set and aggregate is not explicitly configured:

```python
# Apply repetition setting if specified
if repetition is not None:
    dataset.set_repetition(repetition)
    # Auto-enable aggregation with same column if not explicitly set
    if aggregate is None:
        dataset.set_aggregate(repetition)
```

This creates the symmetric link: `repetition → aggregate` (to match the existing `aggregate → repetition` at line 128-129).

Additionally, add `aggregate` as an explicit constructor parameter to `DatasetConfigs.__init__()` for API parity.

---

## 6. Documentation Assessment

### Current docs: `docs/source/user_guide/data/aggregation.md`

- **Lines 30-34**: Quick Start uses `aggregate="sample_id"` — correct for getting aggregated reports, but doesn't mention the connection to `repetition`
- **Lines 262-288**: Repetition Handling section shows `set_repetition('Sample_ID')` separately from aggregation — confusing
- **Lines 402-450**: Complete Example correctly shows BOTH `set_repetition()` AND `set_aggregate()` as separate calls — but a user setting only `repetition` would naturally expect aggregation to work too

### CLAUDE.md instructions

- List `dataset.set_repetition(column)` and `dataset.set_aggregate(value)` as separate methods
- Don't clarify that both need to be set for full functionality

### Example file: `examples/user/05_cross_validation/U04_aggregation.py`

- Uses `DatasetConfigs(repetition="sample_id")` only, NOT `aggregate="sample_id"` — this demonstrates the bug in action. The example silently fails to produce aggregated reports.

### Recommendations

1. After the fix, clarify in docs that `repetition` controls both fold grouping AND prediction aggregation
2. Simplify Quick Start to show `repetition="sample_id"` as the single entry point
3. Document that `aggregate` can be set independently for cases where aggregation differs from fold grouping

---

## 7. Test Coverage Assessment

### Existing Tests (all pass — 86/86)

| Test File | Count | What it covers |
|-----------|-------|---------------|
| `tests/unit/controllers/data/test_repetition.py` | 41 | `RepetitionConfig`, reshape operations, controllers |
| `tests/unit/visualization/test_tab_report_aggregation.py` | 17 | Tab report with aggregation parameter (passed directly) |
| `tests/unit/data/test_predictions_aggregate.py` | 13 | `Predictions.aggregate()` method |
| `tests/integration/pipeline/test_aggregation_integration.py` | 11 | Config propagation with `aggregate=` in config dict |

### Why tests didn't catch the bug

The integration tests use `aggregate="sample_id"` in the config dict — not `repetition="sample_id"` via the constructor. The tab report tests pass `aggregate_column` directly to the method. No test verifies the **config wiring path** from `repetition` → `aggregate`.

### Missing Tests (Required)

1. **Config wiring test**: Set `DatasetConfigs(repetition="sample_id")` and assert `dataset.aggregate == "sample_id"` after `_get_dataset_with_types()`
2. **End-to-end aggregation via repetition**: Set only `repetition="sample_id"`, run a pipeline, assert `runner.last_aggregate == "sample_id"` and tab reports contain aggregated (`*`) rows
3. **Fold grouping integration test**: Set `repetition="sample_id"`, run pipeline, verify that same-sample spectra never appear in both train and validation within a fold (data leakage test)
4. **`predictions.top(by_repetition=True)` integration test**: Verify it returns different scores from `by_repetition=None` when repetitions exist
5. **Symmetric config test**: Verify that `aggregate="col"` in config dict also sets `repetition="col"` (the existing forward link), and that `repetition="col"` also sets `aggregate="col"` (the new reverse link)

### Existing Tests That Should Be Updated

- `tests/integration/pipeline/test_aggregation_integration.py` — Add test variants that use `repetition=` instead of `aggregate=` in config
- `examples/user/05_cross_validation/U04_aggregation.py` — Fix the example to verify it actually produces aggregated output

---

## 8. Improvement Recommendations

1. **Unify the concepts**: Consider making `repetition` the single user-facing entry point. Setting `repetition` should automatically enable both fold grouping and prediction aggregation. A separate `aggregate` should only be needed when the aggregation column differs from the repetition column (rare edge case).

2. **Add `aggregate` to `DatasetConfigs` constructor**: For explicit control when needed.

3. **Validate repetition column exists**: Currently, setting a nonexistent column name silently fails. Add validation that the column exists in metadata at pipeline execution time.

4. **Improve aggregation reporting**: When aggregation is active, the tab report should clearly indicate "Aggregated by: sample_id (N unique samples from M total spectra)" in the header.
