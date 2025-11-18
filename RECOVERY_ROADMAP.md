# Recovery Roadmap
**Project:** nirs4all Pipeline Runner Refactoring
**Status:** 86 test failures, 798 passing (90.3% pass rate)
**Goal:** Restore full functionality with clean architecture (NO deprecated code)
**Strategy:** Fix bugs + Update tests to new API (maintain feature coverage)
## Phase 0: Environment Setup ✅ COMPLETE

**Status:** ✅ Done
**Time:** 10 minutes
**Results:** SHAP import issue fixed

- [x] Fix SHAP/NumPy 2.x compatibility
  - Made SHAP import conditional in `__init__.py`
  - 6 SHAP tests now loadable (still fail for SHAP functionality, but won't block other tests)

---

## Phase 1: Critical Bug Fixes 🔴 HIGHEST PRIORITY

**Estimated Time:** 2-3 days
**Impact:** Restores core functionality
**Tests Fixed:** ~15 failures

### 1.1 Fix Empty Validation Set Issue
**Priority:** 🔴 CRITICAL
**Affected:** 5 flexible input tests + potential silent bugs
**Time Estimate:** 4-8 hours

**Problem:**
```python
# Current behavior:
partition_info = {"train": 80}  # First 80 samples for training
# Results in: val_set.shape = (0, 50) ← EMPTY!
```

**Investigation Steps:**
1. Trace dataset normalization in `orchestrator._normalize_dataset()`
2. Check `DatasetConfigs` partition creation logic
3. Verify `SpectroDataset` index assignment
4. Examine context selector initialization

**Files to Check:**
- `nirs4all/pipeline/execution/orchestrator.py` (dataset normalization)
- `nirs4all/data/config.py` (DatasetConfigs)
- `nirs4all/data/dataset.py` (partition indices)
- `nirs4all/data/indexer.py` (index filtering)

**Test Command:**
```powershell
pytest tests/integration_tests/test_flexible_inputs_integration.py::TestFlexibleInputsIntegration::test_direct_list_and_tuple_approach -xvs
```

**Success Criteria:**
- [ ] Test passes with proper train/val/test split
- [ ] All 5 flexible input tests pass
- [ ] Verify with manual dataset inspection that splits are correct

---

### 1.2 Fix Prediction/Model Reuse Workflow
**Priority:** 🔴 CRITICAL
**Affected:** 7 prediction reuse tests + 1 multisource test
**Time Estimate:** 6-12 hours

**Problem:**
```python
# Workflow broken:
runner.run(pipeline, dataset)  # ✅ Works
best = predictions.get_best()   # ✅ Works
runner.predict(best, new_data)  # ❌ FAILS - prepare_replay issues
```

**Root Causes:**
1. `prepare_replay()` → `_prepare_replay()` (now private)
2. Binary loader initialization may be incomplete
3. Context reconstruction from manifest broken
4. Pipeline UID tracking issues

**Investigation Steps:**
1. Trace `predict()` method flow in runner
2. Check `_prepare_replay()` implementation
3. Verify manifest loading and binary extraction
4. Test context reconstruction manually

**Files to Fix:**
- `nirs4all/pipeline/runner.py` (`predict()`, `_prepare_replay()`)
- `nirs4all/pipeline/binary_loader.py` (artifact loading)
- `nirs4all/pipeline/manifest_manager.py` (manifest reading)

**Test Command:**
```powershell
pytest tests/integration_tests/test_prediction_reuse_integration.py::TestPredictionReuseIntegration::test_model_persistence_and_prediction_with_entry -xvs
```

**Success Criteria:**
- [ ] Can save model and reload for prediction
- [ ] Context properly reconstructed from manifest
- [ ] Binary artifacts correctly loaded
- [ ] All 7 prediction reuse tests pass

---

### 1.3 Verify Real-World Usage with Examples
**Priority:** 🔴 CRITICAL
**Time Estimate:** 2-4 hours

**Command:**
```powershell
cd examples
.\run.ps1 -l  # Run all examples and log output
```

**Check log.txt for:**
- Any tracebacks (indicates failures)
- Proper completion of all examples Q1-Q14
- Correct results in generated outputs

**If examples fail:**
- [ ] Identify which examples break
- [ ] Fix underlying issues
- [ ] Re-run until all examples pass

**Success Criteria:**
- [ ] All examples run without traceback
- [ ] `log.txt` shows successful completions
- [ ] Generated outputs look correct

---

## Phase 2: Update Tests to New Architecture ⚠️ HIGH PRIORITY

**Estimated Time:** 2-3 days
**Impact:** Tests align with new architecture
**Tests Fixed:** ~65 failures (50 API + 15 internal)

**CRITICAL PRINCIPLE:** When updating tests, ensure feature coverage is maintained!

### Strategy: Map-Update-Verify Pattern

For each test file:
1. **Map:** Document what features each test is verifying
2. **Update:** Rewrite test to verify same feature using new API
3. **Verify:** Confirm feature is still tested

**DO NOT:**
- ❌ Simply delete failing tests
- ❌ Comment out assertions
- ❌ Skip tests without understanding what they verify

**DO:**
- ✅ Understand what feature the test protects
- ✅ Test the same feature through appropriate API level
- ✅ Add integration tests if unit tests are no longer appropriate
- ✅ Document test changes in commit messages

---

### 2.1 Create Test Coverage Mapping
**Priority:** ⚠️ HIGH (PREREQUISITE)
**Time Estimate:** 2-3 hours

**Purpose:** Document what each test verifies to prevent losing coverage

**Create:** `TEST_COVERAGE_MAP.md`

**Format:**
```markdown
## test_runner_comprehensive.py

### TestRunnerInitialization::test_default_initialization
**Feature Tested:** Runner initializes with correct default values
**Current Approach:** Checks runner.max_workers, runner.parallel, etc.
**New Approach:**
  - Check runner.orchestrator configuration
  - Or remove if testing internal implementation details
**Reason:** max_workers moved to orchestrator

### TestRunnerInitialization::test_custom_initialization
**Feature Tested:** Runner accepts and uses custom configuration
**Current Approach:** Passes max_workers, parallel to __init__
**New Approach:**
  - Pass configuration through orchestrator parameters
  - Or test at integration level (does pipeline execute correctly?)
**Reason:** Configuration delegated to components
```

**Implementation:**
```powershell
# For each failing test file, create mapping
pytest tests/unit/pipeline/test_runner_comprehensive.py --collect-only -q > test_list.txt
# Analyze each test and document in mapping
```

**Success Criteria:**
- [ ] All failing tests mapped to features
- [ ] Update strategy documented for each test
- [ ] Identified tests that should be removed vs updated

---

### 2.2 Update test_runner_comprehensive.py
**Priority:** ⚠️ HIGH
**Time Estimate:** 4-6 hours
**Tests to Fix:** ~25 failures

**Approach:** Update tests to check appropriate components, not removed APIs

**Batch 1: TestRunnerInitialization (~10 tests)**
- Remove checks for `max_workers`, `parallel`, `backend` (moved to orchestrator)
- Test runner's actual responsibilities: workspace setup, mode, verbose
- If parallel feature is important, create `test_orchestrator.py` to test it

**Batch 2: TestPipelineNormalization (~5 tests)**
- Remove calls to `normalize_pipeline()` (internal method)
- Test at integration level: does `run()` handle various pipeline formats?

**Batch 3: TestDatasetNormalization (~7 tests)**
- Remove calls to `normalize_dataset()` (internal method)
- Test at integration level: does `run()` handle various dataset formats?

**Batch 4: TestControllerSelection (~5 tests)**
- Move to `tests/unit/pipeline/test_router.py`
- Test router component directly, not through runner

**Files to Update:**
- `tests/unit/pipeline/test_runner_comprehensive.py`
- Create `tests/unit/pipeline/test_router.py` (if doesn't exist)
- Create `tests/unit/pipeline/test_orchestrator.py` (if doesn't exist)

**Success Criteria:**
- [ ] Tests check appropriate components
- [ ] No deprecated API calls
- [ ] Feature coverage maintained (mapped in 2.1)

---

### 2.3 Update test_runner_predict.py
**Priority:** ⚠️ HIGH
**Time Estimate:** 3-4 hours
**Tests to Fix:** ~8 failures

**Issue:** Tests call `prepare_replay()` which is now private `_prepare_replay()`

**Approach:** Update tests to use public `predict()` API

**Example Current Test:**
```python
def test_prepare_replay_loads_pipeline_and_sets_state(tmp_path):
    runner, run_dir, pipeline_steps, selection_obj = _create_runner_with_pipeline(tmp_path)
    dataset_config = SimpleNamespace(configs=[({}, "dataset")])

    runner.saver = SimulationSaver(run_dir, save_files=False)
    steps = runner.prepare_replay(selection_obj.copy(), dataset_config)  # ❌

    assert steps is not None
    assert runner.pipeline_uid == "test_uid"
```

**Updated Test:**
```python
def test_predict_workflow_end_to_end(tmp_path):
    """Test complete prediction workflow using public API."""
    # Train model
    runner = PipelineRunner(workspace_path=tmp_path, save_files=True)
    pipeline = [StandardScaler(), Ridge()]
    X_train, y_train = make_regression(n_samples=50, n_features=10)

    predictions, _ = runner.run(pipeline, (X_train, y_train, {"train": 40}))
    best = predictions.get_best()

    # Predict on new data
    X_new = np.random.randn(10, 10)
    y_pred, _ = runner.predict(best, X_new)

    assert y_pred is not None
    assert len(y_pred) == 10
```

**Files to Update:**
- `tests/unit/pipeline/test_runner_predict.py`

**Success Criteria:**
- [ ] Tests use public `predict()` API
- [ ] Tests verify actual prediction workflow
- [ ] No calls to internal `_prepare_replay()`

---

### 2.4 Update test_runner_normalization.py
**Priority:** ⚠️ HIGH
**Time Estimate:** 2-3 hours
**Tests to Fix:** ~15 failures

**Issue:** Tests call internal normalization methods

**Approach:** Test normalization through integration (does `run()` accept various formats?)

**Example Update:**
```python
# Instead of testing normalization directly:
# def test_normalize_pipeline_list(self):
#     result = runner.normalize_pipeline([StandardScaler(), Ridge()])

# Test that runner accepts and processes the format:
def test_runner_accepts_pipeline_as_list(self):
    """Test runner handles pipeline as list."""
    runner = PipelineRunner(verbose=0, save_files=False)
    pipeline_list = [StandardScaler(), Ridge()]
    X, y = make_regression(n_samples=50, n_features=10)

    predictions, _ = runner.run(pipeline_list, (X, y, {"train": 40}))

    assert len(predictions) > 0  # Successfully processed
```

**Files to Update:**
- `tests/unit/pipeline/test_runner_normalization.py`

**Success Criteria:**
- [ ] Tests verify runner handles various input formats
- [ ] No direct calls to internal normalization methods
- [ ] Feature coverage maintained

---

### 2.5 Update test_runner_state.py
**Priority:** 🟡 MEDIUM
**Time Estimate:** 2-3 hours
**Tests to Fix:** ~8 failures

**Issue:** Tests check internal state management that moved to executor

**Approach:** Decide what's worth testing

**Options:**
1. Test state through integration (does pipeline execute in correct order?)
2. Test executor state directly (if state management is critical feature)
3. Remove tests if they're testing implementation details

**Example Decision Tree:**
```
Is step numbering a user-facing feature?
├─ No → Remove test (implementation detail)
└─ Yes → Test at integration level or test executor directly
```

**Files to Update:**
- `tests/unit/pipeline/test_runner_state.py`

---

## Phase 3: Update Remaining Tests 🟡 MEDIUM PRIORITY

**Estimated Time:** 1-2 days
**Impact:** Clean up remaining test failures
**Tests Fixed:** ~7 failures

**Files to Update:**
- `tests/unit/pipeline/test_runner_comprehensive.py::TestControllerSelection`

---

## Phase 4: Documentation & Quality 📚 MEDIUM PRIORITY

**Estimated Time:** 2-3 days
**Impact:** Long-term maintenance

### 4.1 Document Test Update Decisions
**Priority:** 📚 DOCUMENTATION
**Time Estimate:** 1 day

**Create:** `TEST_UPDATE_SUMMARY.md`

**Contents:**
- List of tests updated vs removed
- Rationale for each decision
- Feature coverage verification
- New test patterns for new architecture

**Purpose:** Document why tests were changed to prevent future confusion

---

### 4.2 Add Regression Tests
**Priority:** 🧪 QUALITY
**Time Estimate:** 1 day

**Focus Areas:**
1. Partition creation edge cases (from Issue #1)
2. Prediction workflow scenarios (from Issue #2)
3. Context propagation

**Files to Create:**
- `tests/regression/test_partition_creation.py`
- `tests/regression/test_prediction_workflow.py`

---

### 4.3 Update Architecture Documentation
**Priority:** 📚 DOCUMENTATION
**Time Estimate:** 1 day

**Documents to Create/Update:**
- `docs/developer/architecture.md` - New component structure
- `docs/developer/testing_guide.md` - How to test new architecture
- Component responsibility matrix

---

## Phase 5: Stabilization & Release 🚀

**Estimated Time:** 2-3 days
**Impact:** Production readiness

### 5.1 Full Test Suite Run
```powershell
pytest tests/ -v --cov=nirs4all --cov-report=html
```

**Success Criteria:**
- [ ] All tests pass
- [ ] Code coverage > 80%
- [ ] No unexpected warnings

---

### 5.2 Example Suite Validation
```powershell
cd examples
.\run.ps1 -l
# Check log.txt for any issues
```

**Success Criteria:**
- [ ] All examples complete successfully
- [ ] No warnings or errors in log
- [ ] Generated outputs verified

---

### 5.3 Performance Regression Check
**Create benchmark script:**
```python
# benchmark.py
import time
from nirs4all.pipeline.runner import PipelineRunner

# Test pipeline execution time
start = time.time()
runner.run(pipeline, dataset)
elapsed = time.time() - start

# Compare with baseline
assert elapsed < BASELINE * 1.1  # Max 10% slower
```

---

### 5.4 Release Preparation
- [ ] Update CHANGELOG.md with all changes
- [ ] Bump version number (following semver)
- [ ] Create release notes
- [ ] Tag release in git

---

## Timeline Summary

| Phase | Priority | Time Estimate | Tests Fixed | Focus |
|-------|----------|---------------|-------------|-------|
| 0. Environment | ✅ Done | 10 min | 6 (loadable) | SHAP fix |
| 1. Critical Fixes | 🔴 Critical | 2-3 days | 15 | Fix bugs |
| 2. Update Tests | ⚠️ High | 2-3 days | 65 | Update to new API |
| 3. Remaining Tests | 🟡 Medium | 1-2 days | 7 | Cleanup |
| 4. Documentation | 📚 Medium | 2-3 days | 0 | Document changes |
| 5. Stabilization | 🚀 Release | 2-3 days | 0 | Verify & release |
| **TOTAL** | | **2-3 weeks** | **86** | **Clean codebase** |

---

## Fast Track: Minimum Viable Fix

**If you need quick results (1 week):**

1. **Day 1-2:** Phase 1 - Critical bug fixes
   - Fix empty validation set
   - Fix prediction workflow
   - Verify examples work

2. **Day 3-5:** Phase 2 - Update failing tests
   - Focus on high-value tests
   - Remove tests of implementation details
   - Ensure feature coverage maintained

3. **Day 6-7:** Phase 5 - Basic validation
   - Run full test suite
   - Run example suite
   - Fix any new issues

**Result:** 878+ tests passing (>99%), clean codebase, no deprecated code

---

## Success Metrics

### Phase 1 Complete (Critical)
- [ ] 798 → 813 tests passing (15 more)
- [ ] All examples run successfully
- [ ] No silent data bugs

### Phase 2 Complete (High Priority)
- [ ] 813 → 863 tests passing (50 more)
- [ ] Backward compatibility maintained
- [ ] Deprecation warnings in place

### Phase 3 Complete (All Tests)
- [ ] 863 → 878 tests passing (15 more)
- [ ] Clean test suite
- [ ] Good coverage

### Production Ready
- [ ] 878+ tests passing (>99%)
- [ ] Examples validated
- [ ] Documentation complete
- [ ] Performance validated
- [ ] No deprecated code
- [ ] Clean architecture
- [ ] Release tagged

---

## Risk Mitigation

### If Phase 1 Takes Too Long (>3 days)
**Option:** Focus on validation set bug first (highest impact)
- Get examples working before fixing all prediction tests
- Validate real-world usage early

### If Test Updates Reveal Hidden Bugs
**Critical:** Stop and fix the bug immediately
- This is why we update tests - to find hidden issues
- Don't paper over bugs by changing assertions
- Fix the underlying issue, then update test

### If Test Coverage is Unclear
**Option:** Add integration test first
- Before removing unit test, add integration test
- Verify feature still works end-to-end
- Then safe to remove unit test of internal detail

---

## Key Principles for Test Updates

### ✅ DO:
1. **Understand the feature** being tested before changing code
2. **Map tests to features** using TEST_COVERAGE_MAP
3. **Add integration tests** when removing unit tests
4. **Document decisions** about why tests were removed/updated
5. **Verify feature coverage** is maintained
6. **Test at appropriate level** (integration vs unit)

### ❌ DON'T:
1. **Delete tests** without understanding what they protect
2. **Comment out assertions** to make tests pass
3. **Skip tests** without documenting why
4. **Test internal implementation** details
5. **Lose feature coverage** when updating tests
6. **Add deprecated code** to make old tests pass

---

## Daily Progress Tracking

**Suggested Format:**
```markdown
## Day 1 - Nov 18, 2025
- [x] SHAP import fixed
- [x] Analysis report created
- [ ] Started empty validation set investigation
- **Blockers:** None
- **Tomorrow:** Continue validation set fix

## Day 2 - Nov 19, 2025
- [ ] ...
```

---

## Communication Plan

### Stakeholder Updates
**Frequency:** End of each phase
**Format:** Summary of:
- Tests fixed
- Issues resolved
- Remaining work
- Timeline adjustment

### Developer Team
**Frequency:** Daily standups
**Focus:**
- Yesterday's progress
- Today's plan
- Blockers

---

## Conclusion

The refactoring is **sound and should be kept**. The architecture improvements (orchestrator, executor, step runner separation) are correct. Focus on:

1. **Fix 2 critical bugs** (validation set + prediction workflow)
2. **Update tests systematically** (maintain feature coverage)
3. **No deprecated code** (clean break, clean codebase)

**Critical Success Factor:** When updating tests, always ensure the feature being tested is still verified, just through the appropriate API level (public API, not internal implementation).

**Key Insight:** Most failures aren't bugs - they're tests checking internal details that moved. Update tests to verify features through public APIs and you'll have a cleaner, more maintainable test suite.

**Estimated Effort:**
- 2-3 days for critical bug fixes
- 2-3 days for systematic test updates
- **Total: 1 week for clean, working codebase with no technical debt**

Good luck! The refactoring is worth keeping. 🚀
