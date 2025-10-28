# PipelineRunner Refactoring Checklist

## Phase 0: Preparation ✅

- [x] Create comprehensive test suite
  - [x] `test_pipeline_runner_comprehensive.py` (100+ tests)
  - [x] `test_pipeline_runner_state.py` (40+ tests)
  - [x] `test_pipeline_runner_output.py` (30+ tests)
  - [x] `test_pipeline_runner_regression_prevention.py` (20+ tests)
- [x] Create test documentation
  - [x] `PIPELINE_RUNNER_TESTS_README.md`
  - [x] `RUNNER_TEST_SUITE_SUMMARY.md`
- [x] Create test runner script
  - [x] `run_runner_tests.py`

## Phase 1: Baseline Establishment

- [ ] Run all tests and verify they pass
  ```bash
  python tests/run_runner_tests.py all
  ```
  - [ ] Expected: 190+ tests pass
  - [ ] No failures
  - [ ] No warnings

- [ ] Create baseline for comparison
  ```bash
  python tests/run_runner_tests.py baseline
  ```
  - [ ] Baseline file created
  - [ ] Metadata saved

- [ ] Generate coverage report
  ```bash
  python tests/run_runner_tests.py coverage
  ```
  - [ ] Coverage ≥ 95%
  - [ ] Report generated in `htmlcov/`

## Phase 2: Create Configuration Dataclass

### 2.1 Create `RunnerConfig`
- [ ] Create new file: `nirs4all/pipeline/runner_config.py`
- [ ] Define dataclass with all 17 parameters
  ```python
  @dataclass
  class RunnerConfig:
      max_workers: Optional[int] = None
      continue_on_error: bool = False
      backend: str = 'threading'
      verbose: int = 0
      parallel: bool = False
      workspace_path: Optional[Path] = None
      save_files: bool = True
      mode: str = "train"
      load_existing_predictions: bool = True
      show_spinner: bool = True
      enable_tab_reports: bool = True
      random_state: Optional[int] = None
      plots_visible: bool = False
      keep_datasets: bool = True
  ```

### 2.2 Update PipelineRunner.__init__
- [ ] Accept `config: Optional[RunnerConfig]` parameter
- [ ] Keep backward compatibility (all individual parameters)
- [ ] Internal usage: `self.config = config or RunnerConfig(...)`

### 2.3 Test
- [ ] Run tests: `pytest tests/test_pipeline_runner*.py -v`
- [ ] All tests should still pass
- [ ] No behavioral changes

## Phase 3: Extract DatasetNormalizer

### 3.1 Create `DatasetNormalizer` class
- [ ] Create new file: `nirs4all/pipeline/dataset_normalizer.py`
- [ ] Move methods:
  - [ ] `_normalize_dataset()` → `normalize_dataset()`
  - [ ] `_normalize_pipeline()` → `normalize_pipeline()`
  - [ ] `_extract_dataset_cache()` → `extract_dataset_cache()`

### 3.2 Update PipelineRunner
- [ ] Add `self.normalizer = DatasetNormalizer()`
- [ ] Update calls to use `self.normalizer.normalize_dataset()`
- [ ] Update calls to use `self.normalizer.normalize_pipeline()`

### 3.3 Test
- [ ] Run tests: `pytest tests/test_pipeline_runner_comprehensive.py::TestDatasetNormalization -v`
- [ ] Run tests: `pytest tests/test_pipeline_runner_comprehensive.py::TestPipelineNormalization -v`
- [ ] All normalization tests should pass

## Phase 4: Extract PipelineExecutor

### 4.1 Create `PipelineExecutor` class
- [ ] Create new file: `nirs4all/pipeline/pipeline_executor.py`
- [ ] Move methods:
  - [ ] `run_steps()` → `execute_steps()`
  - [ ] `run_step()` → `execute_step()`
  - [ ] `_select_controller()` → `select_controller()`
  - [ ] `_execute_controller()` → `execute_controller()`

### 4.2 Update PipelineExecutor dependencies
- [ ] Pass necessary state to executor
- [ ] Handle step numbering
- [ ] Handle context management

### 4.3 Update PipelineRunner
- [ ] Add `self.executor = PipelineExecutor(self.config)`
- [ ] Update calls to use `self.executor.execute_steps()`

### 4.4 Test
- [ ] Run tests: `pytest tests/test_pipeline_runner_comprehensive.py::TestStepExecution -v`
- [ ] Run tests: `pytest tests/test_pipeline_runner_comprehensive.py::TestControllerSelection -v`
- [ ] All execution tests should pass

## Phase 5: Extract BinaryManager

### 5.1 Create `BinaryManager` class
- [ ] Create new file: `nirs4all/pipeline/binary_manager.py`
- [ ] Move binary-related functionality
- [ ] Integration with ManifestManager

### 5.2 Update PipelineRunner
- [ ] Add `self.binary_manager = BinaryManager()`
- [ ] Update binary handling calls

### 5.3 Test
- [ ] Run tests: `pytest tests/test_pipeline_runner_comprehensive.py::TestBinaryManagement -v`
- [ ] Binary tests should pass

## Phase 6: Refactor PipelineRunner Main Methods

### 6.1 Simplify `run()` method
- [ ] Extract run directory setup
- [ ] Extract saver initialization
- [ ] Extract manifest management
- [ ] Delegate to normalizer, executor, binary_manager

### 6.2 Simplify `predict()` method
- [ ] Extract prediction setup
- [ ] Delegate to normalizer and executor

### 6.3 Simplify `explain()` method
- [ ] Extract explanation setup
- [ ] Delegate to normalizer and executor

### 6.4 Test
- [ ] Run tests: `pytest tests/test_pipeline_runner_comprehensive.py::TestRunMethod -v`
- [ ] All run method tests should pass

## Phase 7: Cleanup and Documentation

### 7.1 Remove dead code
- [ ] Remove commented-out code
- [ ] Remove unused imports
- [ ] Remove redundant methods

### 7.2 Add type hints
- [ ] Add type hints to all public methods
- [ ] Add type hints to all parameters
- [ ] Add return type hints

### 7.3 Update docstrings
- [ ] Update PipelineRunner docstring
- [ ] Add docstrings to new classes
- [ ] Update method docstrings

### 7.4 Test
- [ ] Run all tests: `python tests/run_runner_tests.py all`
- [ ] All 190+ tests should pass

## Phase 8: Final Validation

### 8.1 Run comprehensive test suite
- [ ] `pytest tests/test_pipeline_runner*.py -v --tb=short`
- [ ] Expected: **190+ passed**
- [ ] Zero failures

### 8.2 Run critical regression tests
- [ ] `pytest tests/test_pipeline_runner_regression_prevention.py -v`
- [ ] Expected: **20+ passed**
- [ ] Zero failures

### 8.3 Compare with baseline
- [ ] `python tests/run_runner_tests.py compare`
- [ ] Expected: "SUCCESS: All tests pass"
- [ ] Expected: "Test count matches baseline"

### 8.4 Verify coverage
- [ ] `python tests/run_runner_tests.py coverage`
- [ ] Expected: Coverage ≥ 95%
- [ ] No decrease from baseline

### 8.5 Performance check
- [ ] Record test execution time
- [ ] Compare with baseline
- [ ] Should be within ±10%

## Phase 9: Code Review

### 9.1 Self review
- [ ] Check all classes follow SRP
- [ ] Check all methods < 50 lines
- [ ] Check no deep nesting (max 3 levels)
- [ ] Check consistent naming

### 9.2 Static analysis
- [ ] Run pylint: `pylint nirs4all/pipeline/runner.py`
- [ ] Run mypy: `mypy nirs4all/pipeline/runner.py`
- [ ] Fix any issues

### 9.3 Documentation review
- [ ] README updated if needed
- [ ] CHANGELOG.md updated
- [ ] API documentation accurate

## Phase 10: Integration Testing

### 10.1 Run existing integration tests
- [ ] `pytest tests/test_comprehensive_integration.py -v`
- [ ] All integration tests pass

### 10.2 Test with real datasets
- [ ] Run example scripts
- [ ] Verify predictions identical to before refactoring

### 10.3 Test backward compatibility
- [ ] Old code using PipelineRunner works unchanged
- [ ] All parameters still supported
- [ ] All methods still available

## Phase 11: Final Sign-off

### 11.1 Checklist verification
- [ ] All phases completed
- [ ] All tests passing
- [ ] Coverage maintained
- [ ] Performance acceptable
- [ ] Documentation updated

### 11.2 Create summary
- [ ] Document changes made
- [ ] Document new class structure
- [ ] Document any breaking changes (should be none)

### 11.3 Commit and PR
- [ ] Commit with descriptive message
- [ ] Create pull request
- [ ] Link to this checklist
- [ ] Request review

---

## Success Metrics

✅ **All 190+ tests pass**
✅ **Coverage ≥ 95%**
✅ **Performance within 10% of baseline**
✅ **Zero breaking changes**
✅ **Code more maintainable**
✅ **Classes follow SRP**

## Notes

- Mark items as [x] when complete
- Run tests after each phase
- Do not proceed if tests fail
- Keep commits atomic (one phase per commit)
- Update this checklist as needed

## Estimated Time

- Phase 0: ✅ Complete (4 hours)
- Phase 1: 30 minutes
- Phase 2: 1 hour
- Phase 3: 2 hours
- Phase 4: 3 hours
- Phase 5: 2 hours
- Phase 6: 2 hours
- Phase 7: 1 hour
- Phase 8: 30 minutes
- Phase 9: 1 hour
- Phase 10: 1 hour
- Phase 11: 30 minutes

**Total estimated time: ~15 hours**

---

**Date Started:** _____________
**Date Completed:** _____________
**Refactored By:** _____________
