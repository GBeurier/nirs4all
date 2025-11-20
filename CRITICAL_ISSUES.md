# Critical Issues Quick Reference
**Strategy:** Fix bugs + Update tests (NO backward compatibility code)

## 🔴 ISSUE #1: Empty Validation Set
**Priority:** CRITICAL
**Symptoms:** `ValueError: Found array with 0 sample(s)` during training
**Impact:** 5 tests fail, potential silent data bugs

### Investigation Checklist
- [ ] Check `orchestrator._normalize_dataset()`
- [ ] Verify partition creation in `DatasetConfigs`
- [ ] Inspect index assignment in `SpectroDataset`
- [ ] Test context selector initialization

### Test Command
```powershell
pytest tests/integration_tests/test_flexible_inputs_integration.py::TestFlexibleInputsIntegration::test_direct_list_and_tuple_approach -xvs
```

### Debug Code
```python
# Add to orchestrator after dataset normalization:
print(f"Train indices: {dataset.indexer.indices['train']}")
print(f"Val indices: {dataset.indexer.indices.get('val', 'MISSING')}")
print(f"Test indices: {dataset.indexer.indices['test']}")

# Check actual data:
X_train = dataset.x({'partition': 'train'})
X_val = dataset.x({'partition': 'val'})
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")  # Should NOT be (0, n_features)
```

### Success Criteria
- [ ] Validation set has > 0 samples
- [ ] Train + Val + Test = Total samples
- [ ] No overlap between sets

---

## 🔴 ISSUE #2: Prediction Workflow Broken
**Priority:** CRITICAL
**Symptoms:** `AttributeError: 'PipelineRunner' object has no attribute 'prepare_replay'`
**Impact:** 8 tests fail, saved models unusable

### Investigation Checklist
- [ ] Make `prepare_replay()` public (remove underscore)
- [ ] Verify manifest loading in `_prepare_replay()`
- [ ] Check binary loader initialization
- [ ] Test context reconstruction

### Test Command
```powershell
pytest tests/integration_tests/test_prediction_reuse_integration.py::TestPredictionReuseIntegration::test_model_persistence_and_prediction_with_entry -xvs
```

### Debug Code
```python
# Add to runner.predict() before _prepare_replay:
print(f"Prediction object: {prediction_obj}")
print(f"Run dir: {self.saver.base_path}")
print(f"Manifest path: {self.saver.base_path / pipeline_uid / 'manifest.yaml'}")

# In _prepare_replay:
print(f"Pipeline UID: {pipeline_uid}")
print(f"Manifest exists: {manifest_path.exists()}")
print(f"Binary loader: {self.binary_loader}")
```

### Quick Fix Option
```python
# In runner.py, add public wrapper (no deprecation warning):
def prepare_replay(self, selection_obj, dataset_config, verbose=0):
    """Public API for prepare_replay."""
    return self._prepare_replay(selection_obj, dataset_config, verbose)
```

**NOTE:** Only if prediction workflow (Issue #2) isn't fixed. Better to update tests to use `predict()` API.

### Success Criteria
- [ ] Can save model during training
- [ ] Can load model for prediction
- [ ] Predictions match original test predictions
- [ ] All 8 prediction tests pass

---

## ⚠️ ISSUE #3: Tests Using Old APIs
**Priority:** HIGH (but NOT bugs - test updates needed)
**Symptoms:** `AttributeError: 'PipelineRunner' object has no attribute 'max_workers'`
**Impact:** 50+ tests fail, need updating to new architecture

### Strategy: UPDATE TESTS, DON'T RESTORE CODE

**Key Principle:** These aren't bugs - the refactoring is correct. Tests need updating.

### Quick Decision Tree
```
For each failing test, ask:
1. What feature is this test protecting?
2. Is that feature still available?
   ├─ Yes → Update test to use new API
   └─ No → Was feature removed intentionally?
       ├─ Yes → Remove test
       └─ No → BUG - fix the code
```

### Example: max_workers Attribute

**DON'T:**
```python
# DON'T add deprecated property
@property
def max_workers(self):
    warnings.warn("deprecated")
    return self.orchestrator.max_workers
```

**DO:**
```python
# DO update test to check appropriate component
def test_orchestrator_configuration(self):
    runner = PipelineRunner()
    # If parallel config is important, test orchestrator:
    assert runner.orchestrator.max_workers == -1
    # OR test at integration level:
    # "Does pipeline execute correctly?"
```

### Files Needing Test Updates
- `tests/unit/pipeline/test_runner_comprehensive.py` (~25 tests)
- `tests/unit/pipeline/test_runner_predict.py` (~8 tests)
- `tests/unit/pipeline/test_runner_normalization.py` (~15 tests)
- `tests/unit/pipeline/test_runner_state.py` (~8 tests)

### Test Update Commands
```powershell
# Create mapping of what each test protects
pytest tests/unit/pipeline/test_runner_comprehensive.py --collect-only -q

# Update tests one file at a time
# After each file, verify:
pytest tests/unit/pipeline/test_runner_comprehensive.py -v
---

## Verification Commands

### Run All Tests
```powershell
pytest tests/ -v --tb=short
```

### Run Just Failing Tests
```powershell
pytest tests/ -v --lf  # Last failed
```

### Run With Coverage
```powershell
pytest tests/ --cov=nirs4all --cov-report=html
```

### Run Examples (Real World Test)
```powershell
cd examples
.\run.ps1 -l
# Check log.txt for tracebacks
```

### Quick Smoke Test
```powershell
pytest tests/integration_tests/test_basic_pipeline.py -v
```

---

## Progress Tracking

### Phase 1: Critical Bug Fixes
- [ ] Issue #1: Empty validation set fixed
- [ ] Issue #2: Prediction workflow fixed
- [ ] Examples run successfully
- [ ] 813+ tests passing (91.8%)

### Phase 2: Update Tests (No Deprecated Code)
- [ ] Issue #3: Tests updated to new API
- [ ] Test coverage mapping complete
- [ ] Feature coverage verified
- [ ] 878+ tests passing (>99%)

### Phase 3: Full Recovery
- [ ] All tests updated
- [ ] Regression tests added
- [ ] Documentation complete
- [ ] Clean codebase (no deprecated code)

---

## Guiding Principles

**When updating tests:**
1. ✅ Ask "What feature does this test protect?"
2. ✅ Ensure that feature is still tested (just differently)
3. ✅ Test through public APIs, not internal implementation
4. ✅ Add integration tests if removing unit tests
5. ❌ Never delete tests without understanding their purpose
6. ❌ Never add deprecated code to make old tests pass

**The goal:** Clean codebase + Full feature coverage

---

## Quick Status Check

Run this after each fix:
```powershell
# Count passing tests
pytest tests/ -q 2>&1 | Select-String "passed"

# Check critical workflows
pytest tests/integration_tests/test_flexible_inputs_integration.py -v
pytest tests/integration_tests/test_prediction_reuse_integration.py -v

# Verify examples
cd examples; .\run.ps1 -n Q1_regression.py -l; cat log.txt | Select-String "Traceback"
```

Expected progression:
- Start: 798 passed (90.3%)
- After Phase 1: 813 passed (91.8%)
- After Phase 2: 863 passed (97.4%)
- Target: 878+ passed (>99%)

---

## Emergency Contacts

### If Stuck on Issue #1 (Empty Validation)
1. Check if `partition_info` dict is correctly parsed
2. Verify indexer creates proper train/val/test splits
3. Look at how old code handled this (git diff)
4. Create minimal reproduction case

### If Stuck on Issue #2 (Prediction)
1. Check manifest.yaml exists and is readable
2. Verify pipeline UID is saved during training
3. Test binary loading separately
4. Look at old prepare_replay implementation

### General Debugging
1. Add verbose=2 to see detailed logs
2. Use pytest -xvs to stop on first failure with full output
3. Insert breakpoint() in code for interactive debugging
4. Check git history for how it worked before

---

*Quick reference for critical issues - See RECOVERY_ROADMAP.md for complete plan*
