# Error Management Refactoring Report

**Document Version:** 1.0
**Date:** December 17, 2025
**Author:** Development Team
**Status:** Proposal

---

## Table of Contents

1. [Objectives](#objectives)
2. [Current State Analysis](#current-state-analysis)
3. [Proposal](#proposal)
4. [Roadmap](#roadmap)
5. [Appendix](#appendix)

---

## Objectives

### Primary Goals

1. **Uniform Error Handling**
   - Establish a single, consistent exception hierarchy across all nirs4all modules
   - Enable predictable error handling patterns for library consumers
   - Reduce cognitive load when debugging and handling errors

2. **Production-Ready Error Reporting**
   - Provide rich error context (error codes, categories, details, suggestions)
   - Enable programmatic error handling via error codes rather than string matching
   - Support machine-readable error formats for integration with monitoring systems

3. **Developer Experience**
   - Clear, actionable error messages with resolution suggestions
   - Easy discoverability of exception types through centralized imports
   - Comprehensive documentation of error scenarios

4. **Maintainability**
   - Reduce code duplication in error handling logic
   - Simplify testing of error conditions
   - Enable consistent logging integration

### Success Criteria

| Metric | Target |
|--------|--------|
| Exception imports from single location | 100% of custom exceptions |
| Error codes defined | All error types |
| Bare `except` clauses eliminated | 100% |
| Tests for exception classes | 90%+ coverage |
| Documentation coverage | All public exceptions |

---

## Current State Analysis

### Existing Error Patterns

#### 1. Stacking Module Exceptions

**Location:** `nirs4all/controllers/models/stacking/exceptions.py`

```python
# Current implementation
class CrossPartitionStackingError(Exception):
    """Raised when stacking across incompatible partitions."""
    pass

class NestedBranchStackingError(Exception):
    """Raised for invalid nested branch structures."""
    pass
```

**Issues:**
- No common base class
- Minimal context information
- No error codes or categories

#### 2. Schema Validation Errors

**Location:** `nirs4all/pipeline/config/_generator/validators/schema.py`

```python
# Current pattern with severity
class ValidationError:
    def __init__(self, message: str, path: str, severity: str = "error"):
        self.message = message
        self.path = path
        self.severity = severity
```

**Issues:**
- Not a proper Exception subclass
- Inconsistent with other error patterns
- Severity as string instead of enum

#### 3. Data Loader Error Reporting

**Location:** `nirs4all/data/loaders/csv_loader.py`

```python
# Current pattern using dict
report = {
    'status': 'error',
    'error': str(e),
    'path': file_path
}
return report
```

**Issues:**
- Returns dict instead of raising exception
- Caller must check status field
- No structured error information

#### 4. Metrics Calculation

**Location:** `nirs4all/core/metrics.py`

```python
# Current bare except pattern
try:
    result = calculate_metric(...)
except:
    return None
```

**Issues:**
- Bare `except` catches all exceptions including KeyboardInterrupt
- Silent failure with None return
- No logging or error context

#### 5. Reconstructor Validation

**Location:** `nirs4all/controllers/models/stacking/reconstructor.py`

```python
# Current ValidationResult pattern
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
```

**Issues:**
- Simple string lists instead of structured errors
- No error codes or categories
- Cannot be raised as exception

### Summary of Issues

| Category | Count | Severity |
|----------|-------|----------|
| Scattered exception definitions | 3+ locations | Medium |
| Bare `except` clauses | Multiple | High |
| Dict-based error reporting | Several modules | Medium |
| Missing error context | All exceptions | Medium |
| Inconsistent severity handling | 2+ patterns | Low |
| No error code system | All modules | Medium |

---

## Proposal

### 1. Centralized Exception Hierarchy

Create `nirs4all/core/exceptions.py` with a unified exception hierarchy:

```
Nirs4allError (base)
├── DataError
│   ├── DataLoadError
│   ├── DataValidationError
│   ├── MissingDataError
│   └── NAHandlingError
├── PipelineError
│   ├── PipelineConfigError
│   ├── PipelineExecutionError
│   └── StepNotFoundError
├── ModelError
│   ├── ModelTrainingError
│   ├── ModelPredictionError
│   ├── ModelNotFittedError
│   └── StackingError
│       ├── CrossPartitionStackingError
│       ├── NestedBranchStackingError
│       ├── FoldMismatchError
│       └── DisjointSampleSetsError
├── ValidationError
│   └── SchemaValidationError
├── IOError
│   ├── FileNotFoundError
│   └── FileWriteError
└── DependencyError
```

### 2. Rich Error Context

Each exception includes structured context:

```python
@dataclass
class ErrorContext:
    code: str                    # Machine-readable code (e.g., "DATA_LOAD_FAILED")
    category: ErrorCategory      # Enum: DATA, PIPELINE, MODEL, etc.
    severity: ErrorSeverity      # Enum: WARNING, ERROR, CRITICAL
    details: Dict[str, Any]      # Arbitrary context data
    suggestions: List[str]       # Resolution hints
    cause: Optional[Exception]   # Original exception if wrapped
```

### 3. Error Result Container

For batch operations that shouldn't fail on first error:

```python
@dataclass
class ErrorResult:
    errors: List[ErrorContext]
    warnings: List[ErrorContext]
    data: Optional[Any]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def raise_if_errors(self) -> None:
        """Convert to exception if errors present."""
```

### 4. Logging Integration

Seamless integration with existing `nirs4all.core.logging`:

```python
def log_and_raise(exception: Nirs4allError, logger: Optional[Logger] = None) -> None:
    """Log exception with appropriate level and raise."""
```

### 5. Error Codes Convention

Structured error code format: `{CATEGORY}_{ACTION}_{RESULT}`

Examples:
- `DATA_LOAD_FAILED`
- `PIPELINE_CONFIG_INVALID`
- `MODEL_NOT_FITTED`
- `STACKING_CROSS_PARTITION`

### Benefits Comparison

| Aspect | Current | Proposed |
|--------|---------|----------|
| Import location | Multiple modules | Single `nirs4all.core` |
| Error context | Minimal/none | Rich structured context |
| Programmatic handling | String matching | Error codes + categories |
| Debugging | Basic messages | Details + suggestions |
| Batch operations | Ad-hoc patterns | `ErrorResult` container |
| Logging | Manual | Built-in integration |
| Testing | Complex setup | Simple exception assertions |

---

## Roadmap

### Phase 1: Infrastructure (Week 1)

**Objective:** Create the foundation for the new error system

| Task | Description | Effort |
|------|-------------|--------|
| 1.1 | Create `nirs4all/core/exceptions.py` with base classes | 4h |
| 1.2 | Implement `ErrorContext`, `ErrorSeverity`, `ErrorCategory` | 2h |
| 1.3 | Implement `ErrorResult` container | 2h |
| 1.4 | Update `nirs4all/core/__init__.py` exports | 1h |
| 1.5 | Write unit tests for exception classes | 4h |
| 1.6 | Add logging integration helper | 2h |

**Deliverables:**
- [ ] `nirs4all/core/exceptions.py` module
- [ ] Updated `nirs4all/core/__init__.py`
- [ ] Test file `tests/core/test_exceptions.py`

### Phase 2: Migrate Stacking Errors (Week 1-2)

**Objective:** Update stacking module to use new exceptions

| Task | Description | Effort |
|------|-------------|--------|
| 2.1 | Create new stacking exceptions extending hierarchy | 2h |
| 2.2 | Update `meta_model.py` error handling | 3h |
| 2.3 | Update `reconstructor.py` ValidationResult | 3h |
| 2.4 | Deprecate old exception locations | 1h |
| 2.5 | Update stacking tests | 2h |

**Deliverables:**
- [ ] Updated `nirs4all/controllers/models/stacking/` modules
- [ ] Deprecation warnings in old locations
- [ ] Updated tests

### Phase 3: Migrate Data Errors (Week 2)

**Objective:** Standardize data loading and validation errors

| Task | Description | Effort |
|------|-------------|--------|
| 3.1 | Update `csv_loader.py` - replace dict reports | 4h |
| 3.2 | Update `loader.py` base class | 2h |
| 3.3 | Update `spectrum.py` data validation | 2h |
| 3.4 | Implement `ErrorResult` for batch loading | 3h |
| 3.5 | Update data module tests | 3h |

**Deliverables:**
- [ ] Updated `nirs4all/data/loaders/` modules
- [ ] Proper exception raising instead of dict returns
- [ ] Updated tests

### Phase 4: Migrate Pipeline Errors (Week 2-3)

**Objective:** Unify pipeline configuration and execution errors

| Task | Description | Effort |
|------|-------------|--------|
| 4.1 | Update schema validators | 3h |
| 4.2 | Update pipeline execution error handling | 4h |
| 4.3 | Update generator error handling | 3h |
| 4.4 | Add pipeline-specific error codes | 2h |
| 4.5 | Update pipeline tests | 3h |

**Deliverables:**
- [ ] Updated `nirs4all/pipeline/` modules
- [ ] Unified validation error pattern
- [ ] Updated tests

### Phase 5: Cleanup & Documentation (Week 3)

**Objective:** Remove legacy patterns and document the system

| Task | Description | Effort |
|------|-------------|--------|
| 5.1 | Remove bare `except` clauses in metrics.py | 2h |
| 5.2 | Audit all modules for legacy patterns | 3h |
| 5.3 | Write exception handling guide | 3h |
| 5.4 | Update API documentation | 2h |
| 5.5 | Add error code reference | 2h |
| 5.6 | Final integration testing | 4h |

**Deliverables:**
- [ ] No bare `except` clauses
- [ ] Documentation in `docs/explanations/error_handling.md`
- [ ] Error code reference in `docs/reference/error_codes.md`
- [ ] All examples passing

### Timeline Summary

```
Week 1:  [████████████████████████████████████████] Phase 1 + Phase 2 start
Week 2:  [████████████████████████████████████████] Phase 2 complete + Phase 3 + Phase 4 start
Week 3:  [████████████████████████████████████████] Phase 4 complete + Phase 5
```

### Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking changes for users | Medium | High | Deprecation warnings, migration guide |
| Increased code verbosity | Low | Low | Provide factory functions for common errors |
| Performance overhead | Low | Low | Context creation is lazy, minimal impact |
| Incomplete migration | Medium | Medium | Prioritize high-impact modules first |

### Success Metrics

Post-implementation verification:

1. **Code Quality**
   - [ ] Zero bare `except` clauses
   - [ ] All custom exceptions inherit from `Nirs4allError`
   - [ ] Error codes defined for all exception types

2. **Test Coverage**
   - [ ] Exception tests at 90%+ coverage
   - [ ] All examples run without errors
   - [ ] Integration tests pass

3. **Documentation**
   - [ ] Exception handling guide complete
   - [ ] Error code reference complete
   - [ ] API docs updated

---

## Appendix

### A. Error Code Reference (Draft)

| Code | Exception | Description |
|------|-----------|-------------|
| `DATA_LOAD_FAILED` | `DataLoadError` | Failed to load data from file |
| `DATA_VALIDATION_FAILED` | `DataValidationError` | Data structure validation failed |
| `DATA_MISSING` | `MissingDataError` | Required data fields missing |
| `NA_HANDLING_FAILED` | `NAHandlingError` | NA value handling failed |
| `PIPELINE_CONFIG_INVALID` | `PipelineConfigError` | Invalid pipeline configuration |
| `PIPELINE_EXECUTION_FAILED` | `PipelineExecutionError` | Pipeline step execution failed |
| `STEP_NOT_FOUND` | `StepNotFoundError` | Pipeline step not found |
| `MODEL_TRAINING_FAILED` | `ModelTrainingError` | Model training failed |
| `MODEL_PREDICTION_FAILED` | `ModelPredictionError` | Model prediction failed |
| `MODEL_NOT_FITTED` | `ModelNotFittedError` | Model used before fitting |
| `CROSS_PARTITION_STACKING` | `CrossPartitionStackingError` | Cross-partition stacking attempted |
| `NESTED_BRANCH_STACKING` | `NestedBranchStackingError` | Invalid nested branch for stacking |
| `FOLD_MISMATCH` | `FoldMismatchError` | Fold counts don't align |
| `DISJOINT_SAMPLES` | `DisjointSampleSetsError` | Sample sets don't overlap |
| `VALIDATION_ERROR` | `ValidationError` | Generic validation error |
| `SCHEMA_VALIDATION_FAILED` | `SchemaValidationError` | Schema validation failed |
| `FILE_NOT_FOUND` | `FileNotFoundError` | File does not exist |
| `FILE_WRITE_FAILED` | `FileWriteError` | Failed to write file |
| `DEPENDENCY_MISSING` | `DependencyError` | Required dependency unavailable |

### B. Migration Example

**Before:**
```python
# In data loader
try:
    data = load_csv(path)
except Exception as e:
    return {'status': 'error', 'error': str(e)}
```

**After:**
```python
# In data loader
from nirs4all.core import DataLoadError

try:
    data = load_csv(path)
except FileNotFoundError as e:
    raise DataLoadError(
        "CSV file not found",
        path=path,
        cause=e,
        suggestions=["Check file path", "Verify file exists"]
    )
except csv.Error as e:
    raise DataLoadError(
        "Invalid CSV format",
        path=path,
        cause=e,
        suggestions=["Verify CSV format", "Check for encoding issues"]
    )
```

### C. Usage Examples

**Catching specific errors:**
```python
from nirs4all.core import DataLoadError, PipelineError

try:
    pipeline.run(data)
except DataLoadError as e:
    print(f"Data error [{e.code}]: {e.message}")
    for suggestion in e.context.suggestions:
        print(f"  → {suggestion}")
except PipelineError as e:
    print(f"Pipeline error: {e}")
```

**Batch processing with ErrorResult:**
```python
from nirs4all.core import ErrorResult, ErrorCategory

result = ErrorResult()

for file in files:
    try:
        process(file)
    except Exception as e:
        result.add_error(
            code="PROCESS_FAILED",
            message=str(e),
            category=ErrorCategory.DATA,
            file=file
        )

if not result.is_valid:
    print(f"Failed: {len(result.errors)} errors")
    result.raise_if_errors()  # Raises if any errors
```

---

*Document prepared for nirs4all error management refactoring initiative.*
