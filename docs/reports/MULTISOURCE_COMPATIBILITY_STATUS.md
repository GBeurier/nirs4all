# Multisource + Complex Features Compatibility Report

**Date**: December 14, 2025
**Test File**: `tests/integration/pipeline/test_multisource_branching_stacking.py`

---

## Executive Summary

| Feature Combination | Status | Issue |
|---------------------|--------|-------|
| **Branching + Multisource (training)** | ⚠️ Partial | Predictions missing metrics (NaN rmse) |
| **Branching + Multisource (reload)** | ❌ Broken | Binary loading fails |
| **Sklearn Stacking + Multisource (training)** | ✅ Works | - |
| **Sklearn Stacking + Multisource (reload)** | ❌ Broken | Binary loading fails |
| **Branching + Stacking + Multisource (training)** | ✅ Works | - |
| **In-Branch Models + Multisource (training)** | ✅ Works | - |
| **MetaModel + Multisource** | ❌ Broken | `_persist_model()` signature mismatch |
| **MetaModel + Branches + Multisource** | ❌ Broken | Same as above |

---

## Detailed Findings

### 1. Branching + Multisource

**Training**: ⚠️ PARTIAL
- Pipeline runs and produces predictions
- **Issue**: Predictions have `NaN` for rmse metric in some branches
- Branch contexts are created correctly, models train, but metric computation fails

**Reload/Predict**: ❌ BROKEN
```
ValueError: Binary for MinMaxScaler_1 not found. Available: []
```
- Artifact loading fails when reloading a branched multisource pipeline
- The minimal pipeline execution cannot find saved transformers

---

### 2. Sklearn Stacking + Multisource

**Training**: ✅ WORKS
- `StackingRegressor` trains correctly on concatenated multisource features
- Predictions are valid with good metrics (R² 0.99+)

**Reload/Predict**: ❌ BROKEN
```
ValueError: Binary for MinMaxScaler_1 not found. Available: []
```
- Same artifact loading issue as branching

---

### 3. Branching + Stacking + Multisource

**Training**: ✅ WORKS
- Stacking models train in each branch
- Each branch produces valid predictions

---

### 4. In-Branch Models + Multisource

**Training**: ✅ WORKS
- Different models can be trained inside different branches
- All branch/model combinations produce valid predictions

---

### 5. MetaModel + Multisource

**Training**: ❌ BROKEN
```
TypeError: MetaModelController._persist_model() got an unexpected keyword argument 'custom_name'
```
- Method signature mismatch between `BaseModelController` and `MetaModelController`
- The `_persist_model()` method in `MetaModelController` doesn't accept `custom_name` parameter

---

## Root Causes Identified

### Issue 1: Artifact Loading for Multisource
**Location**: `nirs4all/pipeline/predictor.py`, `nirs4all/controllers/transforms/transformer.py`

The minimal pipeline artifact provider doesn't correctly map transformer binaries when:
- Multiple sources are present
- Branches modify the operation counter

### Issue 2: Metrics Computation in Branches
**Location**: Branch context handling

Some branch predictions have `NaN` metrics, suggesting the Y values or prediction arrays may have issues during branch execution.

### Issue 3: MetaModelController Signature
**Location**: `nirs4all/controllers/models/meta_model.py`

The `_persist_model()` method override is missing the `custom_name` parameter that was added to the base class.

---

## Recommended Fixes (Priority Order)

1. **Fix MetaModelController._persist_model()** - Add missing `custom_name` parameter
2. **Fix artifact loading for multisource** - Review MinimalArtifactProvider handling
3. **Investigate NaN metrics in branches** - Debug metric computation in branch contexts

---

## Test Commands

```bash
# Run the multisource integration tests
pytest tests/integration/pipeline/test_multisource_branching_stacking.py -v

# Run just the working tests
pytest tests/integration/pipeline/test_multisource_branching_stacking.py -v -k "not reload and not metamodel and not basic_branching"
```

---

## Fix Roadmap

### Phase 1: Quick Win - MetaModelController Signature (Est: 15 min)

**Priority**: HIGH - Blocks all MetaModel + Multisource functionality

**File**: `nirs4all/controllers/models/meta_model.py`

**Problem**: `_persist_model()` override is missing `custom_name` parameter added to base class.

**Fix**:
```python
# Line 1721 - Change from:
def _persist_model(
    self,
    runtime_context: 'RuntimeContext',
    model: Any,
    model_id: str,
    branch_id: Optional[int] = None,
    branch_name: Optional[str] = None,
    branch_path: Optional[List[int]] = None,
    fold_id: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None
) -> Optional[Any]:

# To:
def _persist_model(
    self,
    runtime_context: 'RuntimeContext',
    model: Any,
    model_id: str,
    branch_id: Optional[int] = None,
    branch_name: Optional[str] = None,
    branch_path: Optional[List[int]] = None,
    fold_id: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
    custom_name: Optional[str] = None  # ADD THIS
) -> Optional[Any]:
```

**Tests to pass after fix**: `test_metamodel_multisource`, `test_metamodel_with_branches_multisource`

---

### Phase 2: Artifact Loading for Multisource Reload (Est: 2-4 hours)

**Priority**: HIGH - Blocks all reload/predict with multisource

**Files**:
- `nirs4all/pipeline/minimal_predictor.py` - MinimalArtifactProvider
- `nirs4all/pipeline/storage/artifacts/artifact_loader.py`
- `nirs4all/controllers/transforms/transformer.py`

**Problem**: When reloading a multisource pipeline, transformers can't find their binaries:
```
ValueError: Binary for MinMaxScaler_1 not found. Available: []
```

**Root Cause Analysis**:
1. `MinimalArtifactProvider.get_artifacts_for_step()` returns empty list
2. The minimal pipeline's artifact map doesn't correctly index multi-source artifacts
3. Operation counters may differ between training (per-source) and prediction

**Investigation Steps**:
1. Add debug logging to `MinimalArtifactProvider.get_artifacts_for_step()`
2. Check how artifacts are indexed in `MinimalPipeline.artifact_map`
3. Verify operation counter consistency between train and predict modes

**Potential Fixes**:
1. **Option A**: Fix artifact indexing to handle multi-source operation counters
2. **Option B**: Use class-name based fallback lookup (already exists in `_find_transformer_by_class`)
3. **Option C**: Store source index in artifact metadata and use for lookup

**Code locations to investigate**:
```python
# nirs4all/pipeline/minimal_predictor.py:84
def get_artifacts_for_step(self, step_index, branch_path, branch_id):
    step_artifacts = self.minimal_pipeline.get_artifacts_for_step(step_index)
    # This returns None - why?

# nirs4all/controllers/transforms/transformer.py:228
raise ValueError(f"Binary for {expected_name} not found. Available: {list(binaries_dict.keys())}")
# binaries_dict is empty - artifacts not loaded
```

**Tests to pass after fix**: `test_branching_multisource_reload`, `test_sklearn_stacking_multisource_reload`

---

### Phase 3: Branch Metrics NaN Issue (Est: 1-2 hours)

**Priority**: MEDIUM - Training works but metrics display is broken

**Symptom**: Branch predictions have `NaN` for rmse metric

**Investigation**:
1. Check if y_pred or y_true have invalid values in branch context
2. Verify Y processing is correctly applied per branch
3. Check if y_test extraction works correctly with branch selectors

**Code locations**:
```python
# nirs4all/controllers/data/branch.py - Branch context creation
branch_context.selector.processing = copy.deepcopy(initial_processing)

# nirs4all/core/metrics.py - Metric computation
# May need NaN handling for edge cases
```

**Tests to pass after fix**: `test_basic_branching_multisource`

---

### Phase 4: Sample Augmentation Multi-source (Est: 1 hour)

**Priority**: LOW - Documented limitation, not blocking

**File**: `nirs4all/controllers/data/sample_augmentation.py`

**Problem** (Line 550-551):
```python
# For multi-source, we'd need to handle differently
# For now, use first source
batch_data = transformed_per_source[0]
```

**Fix**: Either:
1. Apply augmentation to all sources independently
2. Concatenate augmented data across sources
3. Document as intentional behavior for spectroscopic data

---

## Implementation Order

```
┌─────────────────────────────────────────────────────────────────┐
│ Week 1: Critical Fixes                                          │
├─────────────────────────────────────────────────────────────────┤
│ Day 1: Phase 1 - MetaModel signature fix (15 min)               │
│        Run tests, verify MetaModel tests pass                   │
│                                                                 │
│ Day 2-3: Phase 2 - Artifact loading investigation & fix         │
│          Debug MinimalArtifactProvider                          │
│          Fix operation counter / artifact mapping               │
│          Run reload tests                                       │
├─────────────────────────────────────────────────────────────────┤
│ Week 2: Polish                                                  │
├─────────────────────────────────────────────────────────────────┤
│ Day 4: Phase 3 - Branch metrics NaN fix                         │
│                                                                 │
│ Day 5: Phase 4 - Sample augmentation (optional)                 │
│        Update documentation                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Validation Checklist

After fixes, all these tests should pass:

```bash
pytest tests/integration/pipeline/test_multisource_branching_stacking.py -v
```

Expected: **9/9 passing**

| Test | Current | Target |
|------|---------|--------|
| test_basic_branching_multisource | ❌ FAIL | ✅ PASS |
| test_named_branching_multisource | ✅ PASS | ✅ PASS |
| test_branching_multisource_reload | ❌ FAIL | ✅ PASS |
| test_sklearn_stacking_multisource | ✅ PASS | ✅ PASS |
| test_sklearn_stacking_multisource_reload | ❌ FAIL | ✅ PASS |
| test_branching_with_stacking_multisource | ✅ PASS | ✅ PASS |
| test_in_branch_models_multisource | ✅ PASS | ✅ PASS |
| test_metamodel_multisource | ❌ FAIL | ✅ PASS |
| test_metamodel_with_branches_multisource | ❌ FAIL | ✅ PASS |
