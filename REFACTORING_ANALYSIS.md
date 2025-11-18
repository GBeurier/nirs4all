# Refactoring Analysis Report
**Date:** November 18, 2025
**Project:** nirs4all
**Scope:** PipelineRunner Refactoring Impact Assessment

## Executive Summary

The recent refactoring of the `PipelineRunner` class has introduced **86 test failures** out of 886 total tests (~9.7% failure rate). The good news is that **798 tests still pass** (90.3%), indicating that core functionality remains intact. However, the refactoring has created significant API breaking changes and removed several features that existing tests depend on.

### Key Findings
- **Environment Issue (Not Refactoring):** 6 SHAP tests fail due to numpy 2.x / SHAP compatibility (fixed by conditional import)
- **API Breaking Changes:** ~50 tests fail due to removed/renamed methods and attributes
- **Data Flow Issues:** ~15 tests fail due to partition/split handling problems
- **State Management Changes:** ~15 tests fail due to altered internal state tracking

---

## Test Failure Categories

### 1. Environment/Dependency Issues (6 failures) ✅ FIXED
**Impact:** Low - Not related to refactoring
**Status:** Already fixed during analysis

- **All SHAP integration tests** (6 tests)
  - **Root Cause:** NumPy 2.x incompatibility with installed SHAP version
  - **Files:** `test_shap_integration.py` (all 6 tests)
  - **Fix Applied:** Made SHAP import conditional in `__init__.py`
  - **Status:** ✅ Fixed

---

### 2. API Breaking Changes (50 failures) ⚠️ UPDATE TESTS
**Impact:** Medium - Tests need updating to new API
**Status:** Update tests, don't restore old code
**Strategy:** Clean break - update all test code to use new architecture

#### 2.1 Removed Runner Attributes (30+ failures)
Tests expect attributes that were removed during refactoring:

**Missing Attributes:**
- `max_workers` - Parallel execution configuration
- `parallel` - Flag to enable parallel execution
- `backend` - Parallel backend selection ('threading', 'loky')
- `load_existing_predictions` - Prediction loading behavior
- Various internal state attributes

**Affected Test Files:**
- `test_runner_comprehensive.py::TestRunnerInitialization` (all initialization tests)
- `test_runner_comprehensive.py::TestParallelExecution` (all parallel tests)
- `test_runner_comprehensive.py::TestControllerSelection` (controller routing tests)
- `test_runner_state.py` (all state tracking tests)

**Example Failures:**
```python
# Test expects:
assert runner.max_workers == -1
# But raises:
AttributeError: 'PipelineRunner' object has no attribute 'max_workers'
```

**Root Cause:** The orchestration refactoring moved these concerns to internal components. This is intentional and correct.

**Resolution Strategy:** Update tests to use new architecture:
- Tests for parallel execution → test `orchestrator` directly or remove if feature removed
- Tests for initialization → update to check new structure
- Tests for state → test at appropriate component level (executor, orchestrator)

#### 2.2 Removed/Renamed Methods (15 failures)
Methods that tests call but no longer exist:

**Missing Public Methods:**
- `prepare_replay()` → now `_prepare_replay()` (private)
- `normalize_pipeline()` → delegated to orchestrator
- `normalize_dataset()` → delegated to orchestrator
- `select_controller()` → delegated to router
- `execute_controller()` → delegated to executor
- `print_best_predictions()` → removed

**Affected Test Files:**
- `test_runner_predict.py` (all predict workflow tests)
- `test_runner_normalization.py` (all normalization tests)
- `test_runner_comprehensive.py::TestPipelineNormalization`
- `test_runner_comprehensive.py::TestDatasetNormalization`

**Example Failures:**
```python
# Test calls:
steps = runner.prepare_replay(selection_obj, dataset_config)
# But raises:
AttributeError: 'PipelineRunner' object has no attribute 'prepare_replay'.
Did you mean: '_prepare_replay'?
```

**Root Cause:** Methods were refactored into internal implementation (_private methods) or moved to other components. This is correct architecture.

**Resolution Strategy:** Update tests to:
- Use public `predict()` API instead of `prepare_replay()`
- Test normalization through high-level `run()` API
- Test controller selection through integration tests, not unit tests of internal routing
- Remove tests of internal implementation details that are now private

#### 2.3 Changed Method Signatures (5 failures)
Methods that exist but have different signatures:

**Changed APIs:**
- `run_step()` - Different parameter order/names
- `run_steps()` - Changed context handling

**Affected Tests:**
- `test_runner_predict.py::test_run_steps_with_context_list`
- `test_runner_comprehensive.py::TestStepExecution`

---

### 3. Data Flow / Partition Handling Issues (15 failures) 🔴 HIGH PRIORITY
**Impact:** High - Core functionality broken
**Status:** Functional regression

#### 3.1 Empty Validation Set Problem (5 failures)
**Symptom:** Models receive 0 samples in validation set during training

**Example Error:**
```
ValueError: Found array with 0 sample(s) (shape=(0, 50)) while a
minimum of 1 is required by Ridge.
```

**Affected Tests:**
- `test_flexible_inputs_integration.py::test_direct_list_and_tuple_approach`
- `test_flexible_inputs_integration.py::test_dict_pipeline_with_tuple_dataset`
- `test_flexible_inputs_integration.py::test_numpy_arrays_without_partition_info`
- `test_flexible_inputs_integration.py::test_cross_validation_with_numpy`
- `test_flexible_inputs_integration.py::test_minimal_numpy_input`

**Root Cause:** Dataset normalization or partition creation logic changed, resulting in improper train/test/validation split. The validation set is being created empty, which causes sklearn models to fail during prediction on validation data.

**Likely Location:**
- `DatasetConfigs` normalization in orchestrator
- Partition index calculation in dataset
- Context selector initialization

#### 3.2 Prediction Workflow Failures (7 failures)
**Symptom:** Saved model prediction workflow broken

**Affected Tests:**
- All tests in `test_prediction_reuse_integration.py` (7 failures)
- `test_multisource_integration.py::test_multi_source_model_reuse`

**Example Scenario:**
```python
# Train model
runner.run(pipeline, dataset)
# Save best model reference
best = predictions.get_best()
# Predict on new data - FAILS
runner.predict(best, new_dataset)
```

**Root Cause:** Multiple factors:
1. `prepare_replay()` is now private (`_prepare_replay()`)
2. Binary loading mechanism may have changed
3. Manifest manager integration altered
4. Context reconstruction from saved state broken

#### 3.3 Dataset Cache Access (3 failures)
Tests that access `raw_data` or `pp_data` snapshots fail.

**Affected Tests:**
- `test_runner_comprehensive.py::TestDatasetNormalization::test_extract_dataset_cache`

**Root Cause:** These attributes are now on orchestrator but not properly exposed/synced on runner.

---

### 4. State Management Changes (15 failures) 🟡 MEDIUM PRIORITY
**Impact:** Medium - Test infrastructure more than user-facing
**Status:** Architectural change

#### 4.1 Step Number Tracking
Tests expect specific step numbering behavior that changed.

**Affected Tests:**
- `test_runner_state.py::TestStepNumbering` (all tests)
- `test_runner_comprehensive.py::TestStepExecution::test_step_number_tracking`

**Root Cause:** Step numbering logic moved to executor, tracking behavior may differ.

#### 4.2 Binary Management
Tests that check binary artifact tracking fail.

**Affected Tests:**
- `test_runner_comprehensive.py::TestBinaryManagement`
- `test_runner_state.py::TestStepBinariesTracking`

**Root Cause:** Binary management refactored into separate components, internal structure changed.

---

## Technical Debt Assessment

### Severity Classification

#### 🔴 **CRITICAL - Functional Regressions** (15 failures)
**Business Impact:** High - Core features broken
- Empty validation set issue (breaks normal training workflow)
- Prediction workflow broken (breaks model reuse)
- These must be fixed in refactored code

#### ⚠️ **HIGH - Test Updates Needed** (50 failures)
**Business Impact:** Low - Tests need updating, not production code
- Tests use old internal APIs
- Tests check internal state that moved to other components
- Action: Update tests to use new public API and architecture
- **Important:** Ensure feature coverage is maintained when updating tests

#### 🟡 **MEDIUM - Test Infrastructure** (15 failures)
**Business Impact:** Low - Test utilities need updates
- Internal state tracking changed
- Test patterns need updating
- Won't affect end users

---

## Drift from Previous Functionality

### What Was Intentionally Removed/Refactored
1. **Parallel Execution Configuration (from Runner)**
   - `max_workers`, `parallel`, `backend` parameters
   - Status: Moved to orchestrator (correct separation of concerns)
   - Test Update: Test orchestrator directly if feature still exists

2. **Public Normalization APIs**
   - `normalize_pipeline()`, `normalize_dataset()`
   - Status: Moved to orchestrator (correct separation of concerns)
   - Test Update: Use high-level `run()` API or test orchestrator directly

3. **Controller Selection API**
   - `select_controller()` method
   - Status: Moved to router (correct separation of concerns)
   - Test Update: Test through integration tests, not unit tests of routing

4. **Prediction Control Flags**
   - `load_existing_predictions` flag
   - Status: Moved to orchestrator configuration
   - Test Update: Test through orchestrator if feature still exists

5. **Internal Replay Preparation**
   - `prepare_replay()` → `_prepare_replay()` (private)
   - Status: Internal implementation detail
   - Test Update: Use public `predict()` API in tests

### What Changed Behavior (Architecture Improvements)
1. **Step Numbering**
   - Counter management moved to executor (correct location)
   - Test Update: Check executor state or test through integration

2. **Binary Management**
   - Artifact tracking restructured into dedicated components (cleaner)
   - Test Update: Test through manifest manager or integration tests

3. **Dataset Snapshots**
   - `raw_data`, `pp_data` now on orchestrator (correct ownership)
   - Test Update: Access through orchestrator

4. **Context Flow**
   - Execution context handling refactored (better separation)
   - Test Update: Test context propagation through integration tests

### What Broke Silently
1. **Partition Creation**
   - Validation sets created empty in some scenarios
   - No error at dataset creation, fails during training
   - **CRITICAL:** This is a silent regression

2. **Prediction Workflow**
   - Binary loading from manifest may be broken
   - Context reconstruction incomplete
   - **HIGH:** Saved model reuse broken

---

## Risk Analysis

### Critical Risks 🔴
1. **Data Leakage Risk**
   - Empty validation sets suggest partition logic issues
   - Could potentially create train/test leakage if not careful
   - **Recommendation:** Audit entire partition creation flow

2. **Production Model Reuse Broken**
   - Users cannot load and use saved models reliably
   - This is a showstopper for ML pipelines
   - **Recommendation:** Priority #1 fix

### High Risks ⚠️
1. **Test Coverage Loss**
   - When updating tests, must ensure features remain covered
   - Risk: Updating a test might hide a bug by removing coverage
   - **Mitigation:**
     - Map each old test to feature being tested
     - Ensure new test still covers that feature
     - Add integration tests if unit tests become redundant
     - Document what each test is actually verifying

2. **Hidden Functional Issues**
   - Some test failures might indicate real bugs, not just API changes
   - **Mitigation:** Carefully analyze each test failure before updating

### Medium Risks 🟡
1. **Test Maintenance**
   - Large number of tests to update
   - Must ensure updates are correct
   - **Recommendation:** Update incrementally, verify each batch

---

## Positive Aspects ✅

Despite the issues, the refactoring has benefits:

1. **Better Separation of Concerns**
   - Orchestrator, Executor, StepRunner are cleaner
   - Easier to understand each component's role

2. **Most Tests Still Pass**
   - 90% pass rate indicates core logic intact
   - Data processing, transformations, models all work

3. **Cleaner Architecture**
   - Less tangled dependencies
   - Better testability of individual components

4. **Active Development**
   - Integration tests comprehensive
   - Good test coverage to catch regressions

---

## Recommendations

### Immediate Actions (Before Any Release)
1. ✅ Fix SHAP import (already done)
2. 🔴 Fix empty validation set issue
3. 🔴 Fix prediction workflow
4. 🔴 Run all examples with `run.ps1 -l` to verify real-world usage

### Short-Term Actions (This Sprint)
1. ⚠️ **Test Update Strategy:**
   - Create test coverage mapping document
   - For each failing test, document:
     - What feature it's testing
     - How to test that feature with new API
   - Update tests in batches, verify each batch
   - Ensure no feature coverage is lost

2. 🟡 Update test infrastructure for new architecture
3. 📝 Document new architecture and APIs
4. 🧪 Add integration tests where unit tests are no longer appropriate

### Long-Term Actions (Next Release)
1. 📚 Create architecture guide for new component structure
2. 🎯 Improve integration test coverage
3. 🏗️ Document testing patterns for new architecture
4. 📊 Measure and optimize performance

---

## Conclusion

The refactoring is **recoverable** and the architecture improvements are sound. Focus on:

1. **Fix validation set creation** (highest priority - functional bug)
2. **Fix prediction workflow** (highest priority - functional bug)
3. **Update tests systematically** (ensure feature coverage maintained)

The architecture improvements are worthwhile and should be kept. The test failures are mostly due to tests checking old APIs and internal implementation details. By updating tests to use the new public APIs and test at appropriate architectural levels, we'll have a cleaner, more maintainable test suite that better reflects the actual usage patterns.

**Key Principle:** When updating tests, always ask "What feature is this test verifying?" and ensure the updated test still verifies that feature, just using the new architecture.

**Estimated Recovery Effort:**
- 1-2 days for critical bug fixes
- 2-3 days for systematic test updates
- **Total: 1 week for full recovery with clean codebase**
