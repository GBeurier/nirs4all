# Chained Y-Processing Analysis Report

**Date:** December 11, 2025
**Author:** GitHub Copilot
**Status:** ✅ IMPLEMENTED

---

## Objectives

Enable users to apply **sequential/chained transformers** to target variables (y) using a simple list syntax:

```python
{"y_processing": [StandardScaler, QuantileTransformer(n_quantiles=30, output_distribution='normal')]}
```

This would:
1. First apply `StandardScaler` to the target values
2. Then apply `QuantileTransformer` to the already-scaled targets
3. Track both transformers for proper inverse transformation during predictions

---

## Current State

### What Works ✅

Single transformer y_processing works correctly:
```python
{"y_processing": StandardScaler()}
{"y_processing": QuantileTransformer(n_quantiles=30)}
```

The current implementation in [y_transformer.py](../../nirs4all/controllers/transforms/y_transformer.py) handles this via `YTransformerMixinController`:
- Clones and fits transformer on training targets
- Transforms all targets
- Adds new processing to `dataset._targets` with proper ancestry tracking
- Persists fitted transformer for prediction mode

### What Fails ❌

List syntax for chained transformers:
```python
{"y_processing": [StandardScaler, QuantileTransformer(n_quantiles=30)]}
```

**Root Cause Analysis:**

1. **Parser Stage** ([parser.py#L146](../../nirs4all/pipeline/steps/parser.py#L146))
   ```python
   operator = self._deserialize_operator(step[key])
   ```
   The `_deserialize_operator` method doesn't handle lists - it returns the list as-is.

2. **Controller Matching** ([y_transformer.py#L27-L30](../../nirs4all/controllers/transforms/y_transformer.py#L27-L30))
   ```python
   @classmethod
   def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
       return (keyword == "y_processing" and
               (isinstance(operator, TransformerMixin) or
                issubclass(operator.__class__, TransformerMixin)))
   ```
   A Python `list` is not a `TransformerMixin`, so this returns `False`.

3. **Fallback** ([dummy.py](../../nirs4all/controllers/flow/dummy.py))
   `DummyController` catches the unhandled step (priority=1000) and prints debug output but performs no actual processing.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     Pipeline Definition                          │
│  {"y_processing": [StandardScaler, QuantileTransformer(...)]}   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      StepParser                                   │
│  _parse_dict_step() → finds "y_processing" keyword               │
│  _deserialize_operator() → returns list as-is (no special case)  │
└──────────────────────────┬───────────────────────────────────────┘
                           │ ParsedStep(operator=[...], keyword="y_processing")
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                   ControllerRouter                                │
│  For each controller in CONTROLLER_REGISTRY (sorted by priority):│
│    - YTransformerMixinController.matches() → False (list ≠ TM)   │
│    - ... other controllers ...                                   │
│    - DummyController.matches() → True (always)                   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                   DummyController                                 │
│  Prints debug info but performs NO transformation                 │
│  Target data remains unchanged                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Suggested Solutions

### Option A: Modify YTransformerMixinController (Recommended)

Extend the existing controller to handle both single transformers and lists:

**Pros:**
- Minimal code changes
- Keeps all y_processing logic in one place
- Maintains existing API and serialization

**Cons:**
- Adds complexity to one controller

**Implementation:**

```python
# In y_transformer.py

@classmethod
def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
    """Match if keyword is 'y_processing' and operator is TransformerMixin or list thereof."""
    if keyword != "y_processing":
        return False

    # Single transformer
    if isinstance(operator, TransformerMixin):
        return True
    if hasattr(operator, '__class__') and issubclass(operator.__class__, TransformerMixin):
        return True

    # List of transformers
    if isinstance(operator, list) and len(operator) > 0:
        return all(
            isinstance(t, TransformerMixin) or
            (isinstance(t, type) and issubclass(t, TransformerMixin))
            for t in operator
        )

    return False

def execute(self, ...):
    operator = step_info.operator

    # Normalize to list
    operators = operator if isinstance(operator, list) else [operator]

    # Instantiate any class types (e.g., StandardScaler vs StandardScaler())
    instantiated = []
    for op in operators:
        if isinstance(op, type):
            instantiated.append(op())
        else:
            instantiated.append(op)

    # Chain execution: apply each transformer sequentially
    current_context = context
    all_artifacts = []

    for transformer in instantiated:
        current_context, artifacts = self._execute_single_transformer(
            transformer, dataset, current_context, runtime_context, mode, loaded_binaries
        )
        all_artifacts.extend(artifacts)

    return current_context, all_artifacts
```

### Option B: Use sklearn.pipeline.Pipeline Wrapper

Automatically wrap lists in `sklearn.pipeline.Pipeline`:

```python
from sklearn.pipeline import Pipeline, make_pipeline

# In parser.py _deserialize_operator:
if isinstance(value, list):
    # Convert list of transformers to sklearn Pipeline
    steps = []
    for i, t in enumerate(value):
        if isinstance(t, type):
            t = t()  # Instantiate class
        steps.append((f"step_{i}", t))
    return Pipeline(steps)
```

**Pros:**
- Leverages sklearn's battle-tested Pipeline
- Automatic inverse_transform support
- Familiar to ML practitioners

**Cons:**
- sklearn Pipeline has specific requirements (e.g., last step must be a model or transformer)
- May not play well with nirs4all's processing ancestry tracking
- Single fitted pipeline object vs individual transformer artifacts

### Option C: Create ChainedYProcessingController

A dedicated controller for list-based y_processing:

```python
@register_controller
class ChainedYProcessingController(OperatorController):
    """Controller for chained y transformations."""
    priority = 4  # Higher priority than single Y transformer (5)

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword == "y_processing" and isinstance(operator, list)

    def execute(self, ...):
        # Process each transformer in sequence
        pass
```

**Pros:**
- Clean separation of concerns
- No modification to existing controller

**Cons:**
- Code duplication with YTransformerMixinController
- Two places to maintain y_processing logic

---

## Recommended Approach: Option A

**Modify YTransformerMixinController** for the following reasons:

1. **Single source of truth** - All y_processing logic stays in one place
2. **Minimal changes** - Only `matches()` and `execute()` need updates
3. **Backward compatible** - Existing single-transformer usage continues to work
4. **Proper ancestry** - Each transformer creates a processing node in the chain:
   ```
   raw → numeric → StandardScaler_1 → QuantileTransformer_2
   ```
5. **Proper inverse transform** - Each transformer is saved separately, enabling correct inverse transforms for predictions

---

## Roadmap

### Phase 1: Core Implementation (1-2 days)

1. **Update `YTransformerMixinController.matches()`**
   - Accept lists of TransformerMixin instances or classes
   - Handle mixed lists (classes and instances)

2. **Update `YTransformerMixinController.execute()`**
   - Normalize input to list
   - Instantiate class types
   - Loop through transformers, executing sequentially
   - Chain context updates between transformers

3. **Update parser (optional enhancement)**
   - Improve `_deserialize_operator()` to instantiate class types in lists
   - Handle nested dict params: `[StandardScaler, {"class": "QuantileTransformer", "params": {...}}]`

### Phase 2: Testing (0.5 day)

1. **Unit tests** for `YTransformerMixinController`:
   - Single transformer (regression test)
   - List of instances: `[StandardScaler(), MinMaxScaler()]`
   - List of classes: `[StandardScaler, MinMaxScaler]`
   - Mixed: `[StandardScaler, MinMaxScaler()]`

2. **Integration test** in examples:
   - Create `Q29_chained_y_processing.py`
   - Test inverse transform on predictions

### Phase 3: Documentation (0.5 day)

1. Update [writing_pipelines.md](../reference/writing_pipelines.md) with chained y_processing examples
2. Add docstrings to updated controller methods

### Phase 4: Prediction Mode Validation (0.5 day)

1. Verify that multiple transformer artifacts are loaded correctly in prediction mode
2. Ensure inverse transforms work through the full chain
3. Test with `PipelinePredictor` on saved pipelines

---

## Estimated Effort

| Phase | Duration | Priority |
|-------|----------|----------|
| Core Implementation | 1-2 days | High |
| Testing | 0.5 day | High |
| Documentation | 0.5 day | Medium |
| Prediction Mode Validation | 0.5 day | High |
| **Total** | **~3-4 days** | |

---

## Files to Modify

| File | Changes |
|------|---------|
| [nirs4all/controllers/transforms/y_transformer.py](../../nirs4all/controllers/transforms/y_transformer.py) | Update `matches()`, refactor `execute()` |
| [nirs4all/pipeline/steps/parser.py](../../nirs4all/pipeline/steps/parser.py) | Optional: enhance `_deserialize_operator()` |
| [docs/reference/writing_pipelines.md](../reference/writing_pipelines.md) | Document new syntax |
| [examples/](../../examples/) | Add `Q29_chained_y_processing.py` example |
| [tests/](../../tests/) | Add unit tests for chained y_processing |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing single-transformer usage | High | Extensive regression testing |
| Inverse transform order bugs | High | Unit test with known transformations |
| Serialization issues with lists | Medium | Test JSON/YAML round-trip |
| Prediction mode binary loading mismatch | Medium | Integration test with saved pipelines |

---

## Example: Final Desired Usage

```python
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer

pipeline = [
    # Transform y: first standardize, then quantile transform
    {"y_processing": [
        StandardScaler,  # Class or instance both work
        QuantileTransformer(n_quantiles=30, output_distribution='normal', random_state=42)
    ]},

    # Alternative with all instances:
    # {"y_processing": [
    #     StandardScaler(),
    #     PowerTransformer(method='yeo-johnson')
    # ]},

    # X transformations...
    StandardScaler(),
    PCA(n_components=50),

    # Model
    {"model": RandomForestRegressor(n_estimators=100)}
]
```

Processing chain visualization after execution:
```
targets.processing_ids → ['raw', 'numeric', 'numeric_StandardScaler_1', 'numeric_StandardScaler_1_QuantileTransformer_2']
targets.get_processing_ancestry('numeric_StandardScaler_1_QuantileTransformer_2')
  → ['raw', 'numeric', 'numeric_StandardScaler_1', 'numeric_StandardScaler_1_QuantileTransformer_2']
```

---

## Conclusion

The feature has been **successfully implemented**. The changes include:

### Files Modified

1. **[nirs4all/controllers/transforms/y_transformer.py](../../nirs4all/controllers/transforms/y_transformer.py)**
   - Added `_is_transformer_like()` helper function to check for TransformerMixin instances or classes
   - Updated `matches()` to accept lists of transformers
   - Refactored `execute()` to process transformers sequentially
   - Added `_normalize_operators()` to handle class instantiation
   - Added `_execute_single_transformer()` for single transformer execution with proper artifact naming

2. **[nirs4all/pipeline/steps/parser.py](../../nirs4all/pipeline/steps/parser.py)**
   - Updated `_deserialize_operator()` to recursively handle lists/tuples

### Verified Functionality

- ✅ Single transformer syntax still works: `{"y_processing": StandardScaler()}`
- ✅ Chained transformer syntax now works: `{"y_processing": [StandardScaler, QuantileTransformer(...)]}`
- ✅ Class types are auto-instantiated: `StandardScaler` → `StandardScaler()`
- ✅ Each transformer is saved as a separate artifact with unique naming
- ✅ Processing ancestry is properly tracked in `Targets` class
- ✅ Manifest correctly records all artifacts

### Example Usage

```python
from sklearn.preprocessing import StandardScaler, QuantileTransformer

pipeline = [
    {"y_processing": [
        StandardScaler,  # Class or instance both work
        QuantileTransformer(n_quantiles=30, output_distribution='normal', random_state=42)
    ]},
    # ... rest of pipeline
]
```

### Resulting Processing Chain

```
targets.processing_ids → ['raw', 'numeric', 'numeric_StandardScaler1', 'numeric_StandardScaler1_QuantileTransformer2']
```
