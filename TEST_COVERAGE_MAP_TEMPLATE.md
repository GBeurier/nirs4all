# Test Coverage Mapping Template

**Purpose:** Document what each failing test protects and how to update it without losing coverage.

**Instructions:**
1. For each failing test, fill out this template
2. Determine update strategy before modifying code
3. Verify feature still covered after update
4. Check off when complete

---

## Test File: `tests/unit/pipeline/test_runner_comprehensive.py`

### ✅ TestRunnerInitialization::test_default_initialization
**Feature Tested:** Runner initializes with correct default configuration
**What it checks now:** `runner.max_workers`, `runner.parallel`, `runner.backend`
**Why it fails:** These attributes moved to orchestrator
**Is feature still available?** Yes - configuration still exists, just in different location
**Update strategy:**
- Option A: Check `runner.orchestrator.max_workers` (if parallel config is user-facing)
- Option B: Remove test (if testing internal implementation detail)
- Option C: Test at integration level (does runner execute pipelines correctly?)
**Chosen:** Option B - Remove, this is internal implementation
**Replacement test:** None needed - integration tests cover pipeline execution
**Status:** ⬜ Not started

---

### ⬜ TestRunnerInitialization::test_custom_initialization
**Feature Tested:** Runner accepts and applies custom configuration
**What it checks now:** Passes `max_workers=4, parallel=True` to `__init__`, checks they're set
**Why it fails:** Configuration parameters removed from runner.__init__
**Is feature still available?** Need to verify - where does parallel config go now?
**Update strategy:**
- Research: Check if orchestrator accepts these params
- If yes: Test through orchestrator
- If no: Was parallel execution removed? Check with team
**Chosen:** TBD - need investigation
**Replacement test:** TBD
**Status:** ⬜ Not started

---

### ⬜ TestPipelineNormalization::test_normalize_pipeline_list
**Feature Tested:** Pipeline normalization handles list format
**What it checks now:** Calls `runner.normalize_pipeline([step1, step2])`
**Why it fails:** Method removed (now internal)
**Is feature still available?** Yes - runner.run() still accepts list format
**Update strategy:**
- Test through integration: does `runner.run()` accept pipeline as list?
- Verify it executes correctly
**Chosen:** Integration test
**Replacement test:** `test_runner_accepts_pipeline_list`
**Status:** ⬜ Not started

---

## Test File: `tests/unit/pipeline/test_runner_predict.py`

### ⬜ test_prepare_replay_loads_pipeline_and_sets_state
**Feature Tested:** Prediction workflow can load saved pipeline and set up state
**What it checks now:** Calls `runner.prepare_replay()`, checks internal state
**Why it fails:** Method is now private `_prepare_replay()`
**Is feature still available?** Yes - `runner.predict()` handles this internally
**Update strategy:**
- Test through public API: `runner.predict(best_model, new_data)`
- Test end-to-end workflow, not internal state
**Chosen:** Integration test through predict()
**Replacement test:** `test_predict_workflow_end_to_end`
**Status:** ⬜ Not started

---

## Test File: `tests/unit/pipeline/test_runner_normalization.py`

### ⬜ test_normalize_dataset_configs
**Feature Tested:** Dataset normalization handles DatasetConfigs passthrough
**What it checks now:** Calls `runner.normalize_dataset(DatasetConfigs(...))`
**Why it fails:** Method removed (now internal)
**Is feature still available?** Yes - runner.run() accepts DatasetConfigs
**Update strategy:**
- Test through integration: does `runner.run()` accept DatasetConfigs?
- Verify it processes correctly
**Chosen:** Integration test
**Replacement test:** `test_runner_accepts_dataset_configs`
**Status:** ⬜ Not started

---

### ⬜ test_normalize_numpy_array_x_only
**Feature Tested:** Dataset normalization handles numpy array (X only)
**What it checks now:** Calls `runner.normalize_dataset(X_array)`
**Why it fails:** Method removed
**Is feature still available?** Yes - runner.run() accepts numpy arrays
**Update strategy:**
- Test through integration: does `runner.run(pipeline, X_array)` work?
**Chosen:** Integration test
**Replacement test:** `test_runner_accepts_numpy_array`
**Status:** ⬜ Not started

---

## Test File: `tests/unit/pipeline/test_runner_state.py`

### ⬜ test_step_number_increments
**Feature Tested:** Step counter increments correctly during execution
**What it checks now:** `runner.step_number == 1` after first step
**Why it fails:** Step numbering moved to executor
**Is feature still available?** Yes, but internal implementation detail
**Update strategy:**
- Option A: Remove - this is internal implementation
- Option B: Test through executor if step ordering is critical feature
- Option C: Test at integration level (does pipeline execute steps in order?)
**Chosen:** Option A - Remove, covered by integration tests
**Replacement test:** None needed
**Status:** ⬜ Not started

---

## Summary Template

After mapping all tests, create summary:

### Tests to Update (with replacement)
- [ ] `test_X` → `test_X_new` (integration approach)
- [ ] `test_Y` → `test_Y_updated` (test orchestrator instead)

### Tests to Remove (covered by integration)
- [ ] `test_Z` - internal implementation detail
- [ ] `test_W` - redundant with integration tests

### Tests to Move (to different test file)
- [ ] `test_controller_selection` → `tests/unit/pipeline/test_router.py`

### New Tests to Add
- [ ] `test_partition_edge_cases` - regression test for Issue #1
- [ ] `test_prediction_workflow_complete` - regression test for Issue #2

### Feature Coverage Verification
- [ ] Parallel execution (if still supported)
- [ ] Pipeline format handling (list, dict, configs)
- [ ] Dataset format handling (numpy, tuple, dict, configs)
- [ ] Prediction workflow
- [ ] Context propagation

---

## Notes

**Common Patterns:**

1. **Testing moved internal methods:**
   - ❌ Call internal method directly
   - ✅ Test through public API at integration level

2. **Testing configuration:**
   - ❌ Check runner attributes that moved
   - ✅ Test through appropriate component (orchestrator, executor)
   - ✅ OR test that functionality works (integration)

3. **Testing implementation details:**
   - ❌ Check internal state counters
   - ✅ Remove if covered by integration tests
   - ✅ OR move to test appropriate component

4. **Testing workflows:**
   - ❌ Test internal steps of workflow
   - ✅ Test complete workflow end-to-end

**Remember:** The goal is to maintain feature coverage, not preserve old test code.
