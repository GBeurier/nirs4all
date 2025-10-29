# Processing List Migration: String to Polars List Type

## Overview

Migration from string-based processing list storage to native Polars `List(Utf8)` type.

**Estimated effort:** ~25 lines of code changes, no performance regression.

---

## Current Implementation (String-based)

```python
# Schema
"processings": pl.Series([], dtype=pl.Utf8)

# Storage example
"['raw', 'msc', 'savgol']"  # Stored as string

# Retrieval
proc_str = df["processings"][0]  # "['raw', 'msc', 'savgol']"
proc_list = eval(proc_str)        # ['raw', 'msc', 'savgol'] ⚠️ eval()!
```

**Issues:**
- `eval()` security risk (though mitigated by controlled input)
- Error-prone parsing
- String manipulation for replace/add operations
- Not type-safe

---

## New Implementation (Polars List Type)

```python
# Schema
"processings": pl.Series([], dtype=pl.List(pl.Utf8))

# Storage example
[['raw', 'msc', 'savgol']]  # Native Polars list!

# Retrieval
proc_list = df["processings"][0]  # ['raw', 'msc', 'savgol'] ✅ Direct access
```

**Benefits:**
- No `eval()` needed
- Type-safe
- Native Polars operations
- Cleaner, more maintainable

---

## Required Changes

### 1. Schema Initialization (indexer.py)

```python
# BEFORE
self.df = pl.DataFrame({
    # ... other columns ...
    "processings": pl.Series([], dtype=pl.Utf8),
    # ... other columns ...
})

# AFTER
self.df = pl.DataFrame({
    # ... other columns ...
    "processings": pl.Series([], dtype=pl.List(pl.Utf8)),
    # ... other columns ...
})
```

**Lines changed:** 1 line

---

### 2. Processing List Preparation (_prepare_processings method)

```python
# BEFORE
def _prepare_processings(self, processings, count):
    if processings is None:
        return [str(self.default_values["processings"])] * count
    elif isinstance(processings, str):
        return [processings] * count
    elif isinstance(processings, list) and len(processings) > 0:
        if isinstance(processings[0], str) and processings[0].startswith("["):
            # Already string representation
            return processings
        elif isinstance(processings[0], str):
            # List of processing names
            return [str(processings)] * count
        elif isinstance(processings[0], list):
            # List of lists
            return [str(p) for p in processings]
    return [str(processings)] * count

# AFTER
def _prepare_processings(self, processings, count):
    if processings is None:
        return [self.default_values["processings"]] * count
    elif isinstance(processings, list) and len(processings) > 0:
        if isinstance(processings[0], str):
            # Single list for all samples
            return [processings] * count
        elif isinstance(processings[0], list):
            # List of lists, one per sample
            if len(processings) != count:
                raise ValueError(f"processings length ({len(processings)}) must match count ({count})")
            return processings
    # Fallback: single list for all
    return [processings if isinstance(processings, list) else [str(processings)]] * count
```

**Lines changed:** ~10 lines (simplified logic, no string conversion)

---

### 3. Replace Processings (replace_processings method)

```python
# BEFORE
def replace_processings(self, source_processings: List[str], new_processings: List[str]):
    if not source_processings or not new_processings:
        return

    def replace_proc(proc_str: str) -> str:
        try:
            proc_list = eval(proc_str)  # ⚠️ eval()
            if not isinstance(proc_list, list):
                return proc_str

            replacement_map = {old: new for old, new in zip(source_processings, new_processings)}
            updated = [replacement_map.get(proc, proc) for proc in proc_list]
            return str(updated)
        except Exception:
            return proc_str

    self.df = self.df.with_columns(
        pl.col("processings").map_elements(replace_proc, return_dtype=pl.Utf8)
    )

# AFTER
def replace_processings(self, source_processings: List[str], new_processings: List[str]):
    if not source_processings or not new_processings:
        return

    replacement_map = {old: new for old, new in zip(source_processings, new_processings)}

    # Use native Polars list operations
    self.df = self.df.with_columns(
        pl.col("processings").list.eval(
            pl.when(pl.element().is_in(list(replacement_map.keys())))
            .then(pl.element().replace(replacement_map))
            .otherwise(pl.element())
        )
    )
```

**Alternative simpler approach (more explicit):**
```python
def replace_processings(self, source_processings: List[str], new_processings: List[str]):
    if not source_processings or not new_processings:
        return

    replacement_map = {old: new for old, new in zip(source_processings, new_processings)}

    def replace_in_list(proc_list):
        return [replacement_map.get(p, p) for p in proc_list]

    self.df = self.df.with_columns(
        pl.col("processings").map_elements(replace_in_list, return_dtype=pl.List(pl.Utf8))
    )
```

**Lines changed:** ~8 lines (cleaner, no eval)

---

### 4. Add Processings (add_processings method)

```python
# BEFORE
def add_processings(self, new_processings: List[str]):
    if not new_processings:
        return

    def append_processings(proc_str: str) -> str:
        try:
            proc_list = eval(proc_str)  # ⚠️ eval()
            if not isinstance(proc_list, list):
                proc_list = [proc_str]
            updated_list = proc_list + new_processings
            return str(updated_list)
        except Exception:
            return str(new_processings)

    self.df = self.df.with_columns(
        pl.col("processings").map_elements(append_processings, return_dtype=pl.Utf8)
    )

# AFTER
def add_processings(self, new_processings: List[str]):
    if not new_processings:
        return

    # Use native Polars list concatenation
    self.df = self.df.with_columns(
        pl.col("processings").list.concat(pl.lit(new_processings))
    )
```

**Lines changed:** ~3 lines (much simpler!)

---

### 5. String Representation (__str__ method)

```python
# BEFORE
# Processing strings already formatted, just use them
for row in combinations.to_dicts():
    value = row[col]
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        parts.append(f"{col} - {value}")

# AFTER
# Convert list to string representation for display
for row in combinations.to_dicts():
    value = row[col]
    if col == "processings" and isinstance(value, list):
        parts.append(f"{col} - {value}")  # List prints nicely as ['raw', 'msc']
    elif isinstance(value, str):
        parts.append(f'{col} - "{value}"')
```

**Lines changed:** ~2 lines

---

## Migration Checklist

- [ ] Update schema in `__init__()` - change `pl.Utf8` to `pl.List(pl.Utf8)`
- [ ] Simplify `_prepare_processings()` - remove string conversion logic
- [ ] Update `replace_processings()` - use native list operations, remove eval()
- [ ] Simplify `add_processings()` - use `list.concat()`
- [ ] Update `__str__()` - handle list display
- [ ] Run existing tests - should all pass
- [ ] Add test for list type handling
- [ ] Update docstrings to mention list type

**Total changes:** ~25 lines across 5-6 methods

---

## Testing Strategy

### 1. Unit Tests (No changes needed - API stays same)

```python
def test_add_samples_with_processings():
    indexer = Indexer()
    indexer.add_samples(5, processings=["raw", "msc"])
    # Should work exactly as before
```

### 2. New Test for List Type

```python
def test_processings_stored_as_list():
    """Verify processings are stored as native Polars list."""
    indexer = Indexer()
    indexer.add_samples(1, processings=["raw", "msc"])

    # Access directly from DataFrame
    processings = indexer.df["processings"][0]
    assert isinstance(processings, list)
    assert processings == ["raw", "msc"]

    # No eval() needed!
    assert "raw" in processings
    assert "msc" in processings
```

### 3. Test Processing Operations

```python
def test_replace_processings_with_list_type():
    indexer = Indexer()
    indexer.add_samples(5, processings=["raw", "old_proc", "msc"])

    indexer.replace_processings(["old_proc"], ["new_proc"])

    # Verify replacement
    processings = indexer.df["processings"][0]
    assert processings == ["raw", "new_proc", "msc"]

def test_add_processings_with_list_type():
    indexer = Indexer()
    indexer.add_samples(5, processings=["raw", "msc"])

    indexer.add_processings(["normalize", "scale"])

    # Verify addition
    processings = indexer.df["processings"][0]
    assert processings == ["raw", "msc", "normalize", "scale"]
```

---

## Backward Compatibility

### API Level (Public)
✅ **No changes** - All public methods accept and return the same types

```python
# Still works
indexer.add_samples(5, processings=["raw", "msc"])
indexer.replace_processings(["old"], ["new"])
indexer.add_processings(["extra"])
```

### Data Level (Internal)
⚠️ **Breaking change for direct DataFrame access**

```python
# BEFORE (string-based)
proc_str = indexer.df["processings"][0]  # "['raw', 'msc']"
proc_list = eval(proc_str)               # ['raw', 'msc']

# AFTER (list-based)
proc_list = indexer.df["processings"][0]  # ['raw', 'msc'] directly!
```

**Impact:** Only affects code that directly accesses `indexer.df["processings"]`
- This is internal usage only
- Easy to identify with grep: `indexer.df\["processings"\]`
- Update ~3-5 locations in tests/internal code

---

## Performance Impact

**Expected:** No performance regression, possibly slight improvement

### Benchmark Plan

```python
import timeit

def benchmark_string_based():
    indexer_old = OldIndexer()
    indexer_old.add_samples(10000, processings=["raw", "msc", "savgol"])
    indexer_old.replace_processings(["raw"], ["raw_v2"])
    indexer_old.add_processings(["normalize"])

def benchmark_list_based():
    indexer_new = NewIndexer()
    indexer_new.add_samples(10000, processings=["raw", "msc", "savgol"])
    indexer_new.replace_processings(["raw"], ["raw_v2"])
    indexer_new.add_processings(["normalize"])

# Run benchmarks
old_time = timeit.timeit(benchmark_string_based, number=100)
new_time = timeit.timeit(benchmark_list_based, number=100)

print(f"String-based: {old_time:.3f}s")
print(f"List-based: {new_time:.3f}s")
print(f"Improvement: {(old_time - new_time) / old_time * 100:.1f}%")
```

**Expected result:** 0-10% improvement (no eval overhead, native operations)

---

## Rollback Plan

If issues arise, rollback is simple:

1. Revert schema: `pl.List(pl.Utf8)` → `pl.Utf8`
2. Restore string conversion in `_prepare_processings()`
3. Restore eval() in `replace_processings()` and `add_processings()`
4. Run tests to confirm

**Time to rollback:** < 10 minutes

---

## Implementation Order

1. **Update schema** in `__init__()` ✅ 5 min
2. **Update `_prepare_processings()`** ✅ 10 min
3. **Update `add_processings()`** ✅ 5 min
4. **Update `replace_processings()`** ✅ 10 min
5. **Update `__str__()`** ✅ 5 min
6. **Run existing tests** ✅ 2 min
7. **Add new tests** ✅ 15 min
8. **Update docstrings** ✅ 10 min
9. **Code review** ✅ 15 min

**Total time:** ~1 hour of focused work

---

## Conclusion

This migration is:
- ✅ **Low risk** - ~25 lines, clear scope
- ✅ **High value** - Eliminates eval(), cleaner code
- ✅ **Easy to test** - Existing tests should pass
- ✅ **Easy to rollback** - Simple revert if needed
- ✅ **No public API changes** - Full backward compatibility

**Recommendation:** Implement as part of Phase 1 of the refactoring.

---

*Document Version: 1.0*
*Date: 2025-10-29*
