# Group-Based Splitting - Quick Reference

**Status**: 📋 Design Complete - Ready for Implementation
**Full Document**: `GROUP_SPLIT_IMPLEMENTATION.md`

---

## What's Being Added

Support for group-aware cross-validation using metadata columns:

```python
{"split": GroupKFold(n_splits=5), "group": "batch_id"}
```

---

## Key Changes Summary

### 1. Runner (`runner.py`, line 50)
```python
# ADD "split" to workflow operators
WORKFLOW_OPERATORS = [..., "y_processing", "y_chart", "split"]  # ← ADD
```

### 2. Controller Matching (`op_split.py`, matches method)
```python
@classmethod
def matches(cls, step, operator, keyword):
    # Priority 1: Explicit keyword
    if keyword == "split":
        return True

    # Priority 2: Dict with "split" key
    if isinstance(step, dict) and "split" in step:
        return True

    # Priority 3: Object with split() method (existing)
    # ... existing logic ...
```

### 3. Controller Execution (`op_split.py`, execute method)
```python
def execute(self, step, operator, dataset, context, ...):
    # Extract group column specification
    group_column = None
    if isinstance(step, dict) and "group" in step:
        group_column = step["group"]

    # Get groups from metadata if needed
    if needs_g:
        if group_column is None:
            # Default to first metadata column with warning
            group_column = dataset.metadata_columns[0]

        groups = dataset.metadata_column(group_column, local_context)

    # Pass groups to split()
    kwargs = {"y": y, "groups": groups}
    folds = list(operator.split(X, **kwargs))
    # ... rest of logic ...
```

---

## Implementation Checklist

### Must Have (Core - 2-3 hours)
- [ ] Add `"split"` to `WORKFLOW_OPERATORS` in runner.py
- [ ] Enhance `matches()` in `CrossValidatorController`
- [ ] Enhance `execute()` in `CrossValidatorController`
- [ ] Add unit tests (`test_group_split.py`)

### Must Have (Docs - 1-2 hours)
- [ ] Update `WRITING_A_PIPELINE.md`
- [ ] Update `METADATA_USAGE.md`
- [ ] Add examples

### Should Have (Integration - 1 hour)
- [ ] End-to-end test (`test_group_split_e2e.py`)
- [ ] Serialization test
- [ ] Prediction mode test

### Nice to Have (Polish - 1 hour)
- [ ] Edge case tests
- [ ] UI component update
- [ ] Error message refinement

**Total Estimated Time**: 5-7 hours

---

## Example Usage

### Basic Syntax
```python
from sklearn.model_selection import GroupKFold

pipeline = [
    StandardScaler(),
    {"split": GroupKFold(n_splits=5), "group": "batch"},
    PLSRegression(10)
]
```

### With Serialization
```json
{
  "split": {
    "class": "sklearn.model_selection._split.GroupKFold",
    "params": {"n_splits": 5}
  },
  "group": "batch"
}
```

### Default Group Column
```python
# Uses first metadata column (e.g., column 0)
{"split": GroupKFold(n_splits=5)}
```

---

## Supported Splitters

| Splitter | Use Case |
|----------|----------|
| `GroupKFold` | General group-aware CV |
| `GroupShuffleSplit` | Fast approximate CV |
| `LeaveOneGroupOut` | Test each group separately |
| `LeavePGroupsOut` | Leave multiple groups out |
| `StratifiedGroupKFold` | Stratified + grouped |

---

## Error Messages

### Missing Metadata
```
❌ ValueError: Dataset has no metadata columns.
   Please add metadata or use a non-grouped splitter.
   Syntax: {'split': GroupKFold(n_splits=5), 'group': 'column_name'}
```

### Invalid Column
```
❌ ValueError: Group column 'invalid' not found in metadata.
   Available columns: ['batch', 'location', 'sample_id']
```

### Wrong Type
```
❌ TypeError: Group column must be a string, got <class 'int'>
```

---

## Testing Strategy

### Unit Tests
```python
# Test matching
assert controller.matches({"split": GroupKFold()}, None, "split")

# Test execution
step = {"split": GroupKFold(n_splits=4), "group": "batch"}
context, binaries = controller.execute(step, step["split"], dataset, ...)

# Test no leakage
for train_idx, val_idx in dataset._folds:
    train_groups = dataset.metadata_column("batch")[train_idx]
    val_groups = dataset.metadata_column("batch")[val_idx]
    assert len(set(train_groups) & set(val_groups)) == 0
```

### Integration Tests
```python
# Full pipeline
pipeline = [
    StandardScaler(),
    {"split": GroupKFold(n_splits=5), "group": "batch"},
    PLSRegression(10)
]

runner = PipelineRunner(verbose=1)
predictions, _ = runner.run(pipeline_config, dataset_config)

assert predictions.num_predictions > 0
```

---

## Backward Compatibility

✅ **All existing syntaxes continue to work**:

```python
# Direct operator (old)
GroupKFold(n_splits=5)

# Dict format (old)
{"class": "sklearn.model_selection.GroupKFold", "params": {"n_splits": 5}}

# New format
{"split": GroupKFold(n_splits=5), "group": "batch"}
```

---

## Files Modified

| File | Changes |
|------|---------|
| `nirs4all/pipeline/runner.py` | Add `"split"` to `WORKFLOW_OPERATORS` |
| `nirs4all/controllers/sklearn/op_split.py` | Update `matches()` and `execute()` |
| `tests/test_group_split.py` | New unit tests |
| `examples/test_group_split_e2e.py` | New integration test |
| `docs/WRITING_A_PIPELINE.md` | Document new syntax |
| `docs/METADATA_USAGE.md` | Add CV examples |

---

## Key Design Decisions

1. **Keyword**: `"split"` follows existing patterns (`"model"`, `"y_processing"`)
2. **Default**: First metadata column with warning (fail-safe behavior)
3. **Validation**: Comprehensive error messages with available options
4. **Serialization**: Follows existing component serialization rules
5. **Backward Compatibility**: No breaking changes to existing pipelines

---

## Risk Assessment

- ✅ **Low Risk**: Well-isolated changes, follows existing patterns
- ✅ **High Value**: Prevents data leakage, essential for proper CV
- ✅ **Fully Scoped**: Clear requirements, comprehensive tests
- ✅ **Backward Compatible**: Existing pipelines unaffected

---

**Ready to implement!** 🚀

For detailed implementation guide, see `GROUP_SPLIT_IMPLEMENTATION.md`.
