# Task Type Detection Refactoring Plan

## Overview
Centralize task type detection across nirs4all using only the modern `ModelUtils.detect_task_type()` and `TaskType` enum, removing all deprecated code and inconsistent implementations.

## Goals
1. **Single source of truth**: Only `ModelUtils.detect_task_type()` returns `TaskType` enum
2. **Consistent types**: Use `TaskType` enum everywhere (no strings like "classification")
3. **Remove deprecated code**: Delete all `deprec_*` methods
4. **Clean dataset integration**: Dataset stores and uses `TaskType` enum
5. **Backward compatibility**: None - breaking change

---

## Current State Analysis

### Detection Functions (5 implementations)
1. ✅ **ModelUtils.detect_task_type()** → `TaskType` enum (KEEP - modern)
2. ❌ **ModelUtils.deprec_detect_task_type()** → "classification"/"regression" (DELETE)
3. ❌ **Dataset._detect_task_type()** → string (REPLACE with ModelUtils call)
4. ❌ **evaluator.detect_task_type()** → string (DELETE - duplicate)
5. ❌ **TabReportManager._detect_task_type_from_entry()** → string (REPLACE)

### Return Value Inconsistencies
- **TaskType enum**: REGRESSION, BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION
- **Strings**: "regression", "binary_classification", "multiclass_classification", "classification"
- **String checks**: `'classification' in dataset.task_type`

### Deprecated Methods to Remove
- `ModelUtils.deprec_detect_task_type()`
- `ModelUtils.deprec_calculate_scores()`
- `ModelUtils.deprec_format_scores()`
- `ModelUtils.deprec_get_best_metric_for_task()`

---

## Refactoring Steps

### Phase 1: Core ModelUtils Enhancement

**File: `nirs4all/utils/model_utils.py`**

1. **Add TaskType helper methods**:
```python
class TaskType(str, Enum):
    """Enumeration of machine learning task types."""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"

    @property
    def is_classification(self) -> bool:
        """Check if this is a classification task."""
        return self in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)

    @property
    def is_regression(self) -> bool:
        """Check if this is a regression task."""
        return self == TaskType.REGRESSION
```

2. **Replace deprecated score methods**:
   - Rename `calculate_scores()` → primary method (remove modern suffix)
   - Rename `format_scores()` → primary method
   - Rename `get_best_score_metric()` → primary method

3. **Delete deprecated methods**:
   - Remove `deprec_detect_task_type()`
   - Remove `deprec_calculate_scores()`
   - Remove `deprec_format_scores()`
   - Remove `deprec_get_best_metric_for_task()`

### Phase 2: Dataset Layer

**File: `nirs4all/dataset/dataset.py`**

1. **Change internal storage**:
```python
# Before:
self._task_type: Optional[str] = None

# After:
self._task_type: Optional[TaskType] = None
```

2. **Replace _detect_task_type()**:
```python
# DELETE the internal method entirely

# In add_numeric_targets():
from nirs4all.utils.model_utils import ModelUtils, TaskType
self._task_type = ModelUtils.detect_task_type(y)
```

3. **Update properties**:
```python
@property
def task_type(self) -> Optional[TaskType]:
    """Get the detected task type."""
    return self._task_type

@property
def is_regression(self) -> bool:
    """Check if dataset is for regression task."""
    return self._task_type == TaskType.REGRESSION if self._task_type else False

@property
def is_classification(self) -> bool:
    """Check if dataset is for classification task."""
    return self._task_type.is_classification if self._task_type else False
```

4. **Update from_dict()**: Parse string → TaskType enum

**File: `nirs4all/dataset/evaluator.py`**

1. **Delete standalone function**: Remove `detect_task_type()` entirely
2. **Import and use**: `from nirs4all.utils.model_utils import ModelUtils, TaskType`

### Phase 3: Controllers

**File: `nirs4all/controllers/models/base_model_controller.py`**

1. **Delete _detect_task_type() method** - use `ModelUtils.detect_task_type()` directly
2. **Update all method calls**:
   - Replace `deprec_get_best_metric_for_task()` → `get_best_score_metric()`
   - Replace `deprec_calculate_scores()` → `calculate_scores()`
   - Replace `deprec_format_scores()` → `format_scores()`

3. **Update string checks**:
```python
# Before:
if dataset.task_type and 'classification' in dataset.task_type:

# After:
if dataset.task_type and dataset.task_type.is_classification:
```

**File: `nirs4all/controllers/sklearn/op_model.py`**

1. **Update _detect_task_type() calls**:
```python
# Before:
task_type = self._detect_task_type(y_train)

# After:
task_type = ModelUtils.detect_task_type(y_train)
```

2. **Update method signature expectations** (TaskType enum vs string)

**File: `nirs4all/controllers/tensorflow/op_model.py`**

1. **Simplify task type detection**:
```python
# Before: string detection + manual conversion
task_type_str = self._detect_task_type(y_train)
# ... complex if/elif chain ...

# After:
task_type = ModelUtils.detect_task_type(y_train)
```

2. **Remove manual enum conversion logic**

### Phase 4: Utilities

**File: `nirs4all/utils/tab_report_manager.py`**

1. **Replace _detect_task_type_from_entry()**:
```python
@staticmethod
def _detect_task_type_from_entry(entry: Dict[str, Any]) -> TaskType:
    """Detect task type from a prediction entry."""
    y_true = np.array(entry.get('y_true', []))
    if len(y_true) == 0:
        return TaskType.REGRESSION
    return ModelUtils.detect_task_type(y_true)
```

2. **Update all usages** to expect `TaskType` enum

**File: `nirs4all/utils/shap_analyzer.py`**

1. **Update method signatures**:
```python
# Before:
task_type: str = "regression"

# After:
task_type: TaskType = TaskType.REGRESSION
```

2. **Update string comparisons**:
```python
# Before:
if task_type == 'classification':

# After:
if task_type.is_classification:
```

**File: `nirs4all/dataset/prediction_analyzer.py`**

1. Already imports `ModelUtils, TaskType` ✅
2. Verify usage is consistent with enum types

### Phase 5: Pipeline Runner

**File: `nirs4all/pipeline/runner.py`**

1. **Simplify task type extraction**:
```python
# Before:
task_type = 'classification' if dataset.task_type and 'classification' in dataset.task_type else 'regression'

# After:
task_type = 'classification' if dataset.task_type and dataset.task_type.is_classification else 'regression'
```

*Note: SHAP analyzer may still need string for legacy reasons - keep string conversion*

### Phase 6: Serialization

**Files: Anywhere task_type is saved/loaded**

1. **Serialize as string**:
```python
# Save:
data['task_type'] = task_type.value if task_type else None

# Load:
task_type_str = data.get('task_type')
task_type = TaskType(task_type_str) if task_type_str else None
```

---

## Implementation Order

1. ✅ **Phase 1**: Update `model_utils.py` (core)
2. ✅ **Phase 2**: Update `dataset.py` (data layer)
3. ✅ **Phase 3**: Update controllers (model layer)
4. ✅ **Phase 4**: Update utilities (support)
5. ✅ **Phase 5**: Update pipeline runner
6. ✅ **Phase 6**: Fix serialization

---

## Breaking Changes

### API Changes
- `Dataset.task_type` returns `TaskType` enum instead of `str`
- All model controller methods expect `TaskType` instead of strings
- Deprecated methods removed (no fallback)

### Migration Guide for Users
```python
# OLD CODE:
if dataset.task_type == "classification":
    ...

# NEW CODE:
from nirs4all.utils.model_utils import TaskType

if dataset.task_type and dataset.task_type.is_classification:
    ...

# Or:
if dataset.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
    ...
```

---

## Testing Strategy

### Unit Tests to Update
1. `tests/test_model_utils.py` - update all task type tests
2. `tests/test_dataset.py` - update task_type property tests
3. `tests/test_controllers.py` - update model controller tests

### Integration Tests
1. Run all examples (Q1-Q5)
2. Test serialization/deserialization
3. Test pipeline execution with various task types

### Validation
- [ ] All examples run without errors
- [ ] Task type correctly detected for all datasets
- [ ] Metrics calculation works for all task types
- [ ] Serialization preserves task type
- [ ] No deprecated method calls remain

---

## Files to Modify

### Core (Priority 1)
- [x] `nirs4all/utils/model_utils.py` - Add helpers, remove deprecated
- [x] `nirs4all/dataset/dataset.py` - Change storage type, remove duplication

### Controllers (Priority 2)
- [x] `nirs4all/controllers/models/base_model_controller.py` - Update all calls
- [x] `nirs4all/controllers/sklearn/op_model.py` - Simplify detection
- [x] `nirs4all/controllers/tensorflow/op_model.py` - Remove conversion logic

### Utilities (Priority 3)
- [x] `nirs4all/utils/tab_report_manager.py` - Use ModelUtils
- [x] `nirs4all/utils/shap_analyzer.py` - Update signatures
- [x] `nirs4all/dataset/evaluator.py` - Remove duplicate function
- [x] `nirs4all/dataset/prediction_analyzer.py` - Verify consistency

### Pipeline (Priority 4)
- [x] `nirs4all/pipeline/runner.py` - Simplify string checks

### Other (Priority 5)
- [ ] Any serialization/deserialization code
- [ ] Any remaining string comparisons

---

## Estimated Effort

- **Phase 1 (Core)**: 1-2 hours
- **Phase 2 (Dataset)**: 1 hour
- **Phase 3 (Controllers)**: 2-3 hours
- **Phase 4 (Utilities)**: 1-2 hours
- **Phase 5 (Pipeline)**: 30 min
- **Phase 6 (Serialization)**: 1 hour
- **Testing**: 2-3 hours

**Total**: ~10-14 hours

---

## Success Criteria

✅ **Completed when**:
1. Only one detection function exists: `ModelUtils.detect_task_type()`
2. All code uses `TaskType` enum (no string comparisons)
3. All deprecated methods removed
4. All tests pass
5. All examples run successfully
6. Code is cleaner and more maintainable
