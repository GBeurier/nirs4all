# Force Group Splitting: Universal Group Support for Any Splitter

## Executive Summary

This report proposes a `force_group` mechanism that enables **any sklearn-compatible splitter** (e.g., `KFold`, `ShuffleSplit`, `StratifiedKFold`) to work with grouped samples. The mechanism aggregates samples by group, provides "virtual samples" to the splitter, retrieves fold indices, and expands them back to the original dataset.

---

## 1. Objectives

### 1.1 Problem Statement

Currently, nirs4all requires users to choose splitters based on whether they need group awareness:

| Need Groups? | Available Splitters | Limitation |
|--------------|---------------------|------------|
| No | `KFold`, `ShuffleSplit`, `StratifiedKFold`, etc. | Rich sklearn ecosystem |
| Yes | `GroupKFold`, `GroupShuffleSplit`, `SPXYGFold` | Limited options |

This creates friction:
1. Users must remember which splitters support groups
2. `ShuffleSplit` silently ignores the `group` parameter (sklearn warning)
3. No way to use stratified splitting with groups on continuous targets
4. Custom splitters require explicit group-awareness implementation

### 1.2 Proposed Solution

A **`force_group`** parameter in the split step configuration that:
1. Intercepts any splitter before execution
2. Aggregates samples by group into "virtual samples"
3. Passes virtual samples to the splitter
4. Maps fold indices back to original sample indices

### 1.3 Expected Syntax

```python
# Current approach (only works with group-aware splitters):
{"split": GroupKFold(n_splits=3), "group": "ID"}

# Proposed approach (works with ANY splitter):
{"split": KFold(n_splits=3), "force_group": "ID"}
{"split": ShuffleSplit(test_size=0.2), "force_group": "ID"}
{"split": StratifiedKFold(n_splits=5), "force_group": "y"}  # Stratify by target
```

### 1.4 Key Benefits

1. **Universal compatibility**: Any sklearn splitter works with groups
2. **Backward compatibility**: `group` parameter for native group-aware splitters unchanged
3. **Stratification support**: `StratifiedKFold` can stratify on group-level y values
4. **User-friendly**: Single syntax for all use cases
5. **Prevents data leakage**: Groups are never split across train/test

---

## 2. Current State Analysis

### 2.1 Existing Implementations

#### SPXYGFold (Complete Group Support)

The `SPXYGFold` splitter already implements the aggregation pattern:

```python
# In nirs4all/operators/splitters/splitters.py (lines 682-745)
def _aggregate_groups(self, X, y, groups):
    """Aggregate samples by group, returning representatives and index mapping."""
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    X_rep = np.zeros((n_groups, X.shape[1]))
    group_indices = []

    for i, g in enumerate(unique_groups):
        mask = groups == g
        indices = np.where(mask)[0].tolist()
        group_indices.append(indices)

        if self.aggregation == "mean":
            X_rep[i] = X[mask].mean(axis=0)
        else:  # median
            X_rep[i] = np.median(X[mask], axis=0)

    # ... y aggregation ...

    return X_rep, y_rep, group_indices, unique_groups
```

And the fold expansion:

```python
# In split() method (lines 983-986)
# Map back to sample indices
train_indices = np.concatenate([group_indices[u] for u in train_units])
test_indices = np.concatenate([group_indices[u] for u in test_units])
```

#### CrossValidatorController (Group Extraction)

The controller already extracts groups from metadata:

```python
# In nirs4all/controllers/splitters/split.py (lines 131-200)
if group_column is not None:
    groups = dataset.metadata_column(group_column, local_context, include_augmented=False)
```

But it only passes groups to splitters that natively support them (via `_needs()` function).

### 2.2 Known Issues

1. **`ShuffleSplit` ignores groups**: sklearn explicitly warns "The groups parameter is ignored"
2. **`group` parameter is silently ignored**: When provided with non-group splitters
3. **No stratification with groups**: Can't combine group-awareness with target stratification on continuous data
4. **Confusing syntax**: Users expect `group` to work universally

### 2.3 Roadmap Reference

From `Roadmap.md` (line 22):
> [Split] add force_group (mean, median, max, min, etc.) before split to be able to split with groups seamlessly. ie: {"split": KFold, "force_group":"ID"} or reuse group if possible to detect that group is not required (I think it's not detectable)

---

## 3. Implementation Proposal

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CrossValidatorController                         │
├─────────────────────────────────────────────────────────────────────┤
│  1. Check for "force_group" parameter                               │
│  2. If present:                                                      │
│     a. Extract groups from metadata or y                             │
│     b. Create GroupedSplitterWrapper(original_splitter, groups, agg) │
│     c. Execute wrapper instead of original                           │
│  3. If not: Execute original splitter (current behavior)            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GroupedSplitterWrapper                            │
├─────────────────────────────────────────────────────────────────────┤
│  split(X, y=None, groups=None):                                      │
│    1. X_rep, y_rep, group_indices = aggregate(X, y, groups)          │
│    2. For each (train_rep, test_rep) from inner.split(X_rep, y_rep): │
│       a. train_idx = expand(train_rep, group_indices)                │
│       b. test_idx = expand(test_rep, group_indices)                  │
│       c. yield train_idx, test_idx                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Proposed Code Structure

#### 3.2.1 New Wrapper Class

```python
# nirs4all/operators/splitters/grouped_wrapper.py

class GroupedSplitterWrapper:
    """
    Wraps any sklearn-compatible splitter to add group-awareness.

    This wrapper aggregates samples by group into "virtual samples",
    passes them to the inner splitter, and expands the fold indices
    back to the original sample space.

    Parameters
    ----------
    splitter : BaseCrossValidator
        Any sklearn-compatible cross-validator.

    aggregation : str, default="mean"
        Method for aggregating samples within groups:
        - "mean": Use group centroid (average of all samples)
        - "median": Use group median (robust to outliers)
        - "first": Use first sample in each group (fast, no aggregation)

    y_aggregation : str or None, default=None
        Method for aggregating y values. If None, inferred from splitter needs:
        - "mean": For regression (continuous y)
        - "mode": For classification (categorical y)
        - "first": Use first y value in group

    Examples
    --------
    >>> from sklearn.model_selection import KFold
    >>> wrapper = GroupedSplitterWrapper(KFold(n_splits=3), aggregation="mean")
    >>> for train_idx, test_idx in wrapper.split(X, y, groups=sample_ids):
    ...     # train_idx and test_idx are original sample indices
    ...     # All samples from the same group are in the same fold
    """

    def __init__(self, splitter, aggregation="mean", y_aggregation=None):
        self.splitter = splitter
        self.aggregation = aggregation
        self.y_aggregation = y_aggregation

    def _aggregate(self, X, y, groups):
        """Aggregate samples by group into representative virtual samples."""
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        X_rep = np.zeros((n_groups, X.shape[1]))
        y_rep = None if y is None else np.zeros(n_groups)
        group_indices = []

        for i, g in enumerate(unique_groups):
            mask = groups == g
            indices = np.where(mask)[0].tolist()
            group_indices.append(indices)

            # Aggregate X
            if self.aggregation == "mean":
                X_rep[i] = X[mask].mean(axis=0)
            elif self.aggregation == "median":
                X_rep[i] = np.median(X[mask], axis=0)
            elif self.aggregation == "first":
                X_rep[i] = X[mask][0]

            # Aggregate y
            if y is not None:
                y_agg = self.y_aggregation or self._infer_y_aggregation()
                if y_agg == "mean":
                    y_rep[i] = y[mask].mean()
                elif y_agg == "mode":
                    from scipy import stats
                    y_rep[i] = stats.mode(y[mask], keepdims=False).mode
                elif y_agg == "first":
                    y_rep[i] = y[mask][0]

        return X_rep, y_rep, group_indices, unique_groups

    def _infer_y_aggregation(self):
        """Infer y aggregation method from splitter type."""
        splitter_name = self.splitter.__class__.__name__
        if "Stratified" in splitter_name:
            return "mode"  # Classification
        return "mean"  # Default for regression

    def _expand_indices(self, rep_indices, group_indices):
        """Expand representative indices to original sample indices."""
        return np.concatenate([group_indices[i] for i in rep_indices])

    def split(self, X, y=None, groups=None):
        """Generate train/test indices with group-awareness."""
        if groups is None:
            # No groups - delegate to original splitter
            yield from self.splitter.split(X, y)
            return

        # Aggregate by groups
        X_rep, y_rep, group_indices, unique_groups = self._aggregate(X, y, groups)

        # Split on representative samples
        for train_rep, test_rep in self.splitter.split(X_rep, y_rep):
            train_indices = self._expand_indices(train_rep, group_indices)
            test_indices = self._expand_indices(test_rep, group_indices)
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits from inner splitter."""
        return self.splitter.get_n_splits()
```

#### 3.2.2 Controller Modifications

```python
# In nirs4all/controllers/splitters/split.py

def execute(self, step_info, dataset, context, runtime_context, ...):
    op = step_info.operator

    # Check for force_group parameter
    force_group = None
    aggregation = "mean"

    if isinstance(step_info.original_step, dict):
        force_group = step_info.original_step.get("force_group")
        aggregation = step_info.original_step.get("aggregation", "mean")

    # If force_group is specified, wrap the splitter
    if force_group is not None:
        # Extract groups
        if force_group == "y":
            # Use y values as groups (for stratification on continuous targets)
            groups = self._bin_y_for_groups(dataset, local_context)
        else:
            # Use metadata column
            groups = dataset.metadata_column(force_group, local_context, include_augmented=False)

        # Wrap the splitter
        from nirs4all.operators.splitters import GroupedSplitterWrapper
        op = GroupedSplitterWrapper(op, aggregation=aggregation)
        kwargs["groups"] = groups

    # Continue with existing logic...
```

### 3.3 Syntax Options

#### Option A: Separate `force_group` Parameter (Recommended)

```python
# Distinct from native group support
{"split": KFold(n_splits=3), "force_group": "ID"}
{"split": ShuffleSplit(test_size=0.2), "force_group": "ID", "aggregation": "median"}
```

**Pros**: Clear distinction, backward compatible, explicit intent
**Cons**: New keyword to learn

#### Option B: Unified `group` Parameter with Auto-Detection

```python
# Single parameter, behavior depends on splitter
{"split": KFold(n_splits=3), "group": "ID"}  # Auto-wraps KFold
{"split": GroupKFold(n_splits=3), "group": "ID"}  # Uses native support
```

**Pros**: Simpler API, automatic selection
**Cons**: Less explicit, may have edge cases

#### Option C: Explicit Wrapper in Pipeline

```python
# User explicitly wraps
{"split": GroupedSplitterWrapper(KFold(n_splits=3)), "group": "ID"}
```

**Pros**: Very explicit, no magic
**Cons**: Verbose, requires import

### 3.4 Special Case: Stratification on Continuous Y

One powerful use case is stratifying on binned continuous targets:

```python
# Stratify groups by their average y value
{"split": StratifiedKFold(n_splits=5), "force_group": "ID", "y_aggregation": "mean"}
```

This would:
1. Aggregate samples by group (mean of X and y)
2. Bin aggregated y values for stratification
3. Ensure each fold has balanced y distribution at group level
4. Expand indices back to original samples

---

## 4. Implementation Roadmap

### Phase 1: Core Wrapper (Priority: High)

**Timeline**: 1-2 days

1. Create `GroupedSplitterWrapper` class in `nirs4all/operators/splitters/grouped_wrapper.py`
2. Implement aggregation logic (reuse from SPXYGFold)
3. Implement index expansion
4. Add comprehensive unit tests

**Deliverables**:
- `GroupedSplitterWrapper` class
- Unit tests for wrapper
- Documentation

### Phase 2: Controller Integration (Priority: High)

**Timeline**: 1 day

1. Modify `CrossValidatorController.execute()` to detect `force_group`
2. Implement automatic wrapping logic
3. Handle edge cases (no metadata, invalid column, etc.)
4. Add integration tests

**Deliverables**:
- Modified controller
- Integration tests
- Updated docstrings

### Phase 3: Extended Features (Priority: Medium)

**Timeline**: 1-2 days

1. Implement `force_group: "y"` for target-based grouping
2. Add y-binning for stratification on continuous targets
3. Implement `aggregation` parameter (mean, median, first)
4. Add `y_aggregation` parameter for classification/regression

**Deliverables**:
- Extended wrapper capabilities
- Additional tests
- User guide documentation

### Phase 4: Cleanup and Unification (Priority: Low)

**Timeline**: 1 day

1. Consider deprecating separate `group` parameter for non-group splitters
2. Add warnings when `group` is provided to non-group splitters
3. Update examples and documentation
4. Performance benchmarks

**Deliverables**:
- Deprecation warnings (if applicable)
- Updated examples
- Performance report

---

## 5. Testing Strategy

### 5.1 Unit Tests

```python
class TestGroupedSplitterWrapper:

    def test_kfold_with_groups(self):
        """Test KFold respects groups through wrapper."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        groups = np.repeat(np.arange(20), 5)  # 20 groups, 5 samples each

        wrapper = GroupedSplitterWrapper(KFold(n_splits=5))

        for train_idx, test_idx in wrapper.split(X, y, groups=groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            # No group overlap
            assert len(train_groups & test_groups) == 0

    def test_shuffle_split_with_groups(self):
        """Test ShuffleSplit respects groups through wrapper."""
        # Similar test for ShuffleSplit

    def test_stratified_kfold_with_groups(self):
        """Test StratifiedKFold works with group aggregation."""
        # Verify stratification on aggregated y values

    def test_aggregation_methods(self):
        """Test mean, median, first aggregation."""

    def test_backward_compatibility(self):
        """Test wrapper is transparent when no groups provided."""
```

### 5.2 Integration Tests

```python
class TestForceGroupIntegration:

    def test_full_pipeline_with_force_group(self):
        """Test force_group in complete pipeline."""
        pipeline = [
            StandardScaler(),
            {"split": ShuffleSplit(n_splits=1, test_size=0.2), "force_group": "ID"},
            {"split": KFold(n_splits=3), "force_group": "ID"},
            PLSRegression(n_components=5)
        ]
        # Run and verify no group leakage
```

---

## 6. Appendix

### A. Comparison with Existing Approaches

| Approach | Groups Respected | Stratification | Works with Any Splitter |
|----------|------------------|----------------|-------------------------|
| `group` + GroupKFold | ✓ | ✗ | ✗ |
| `group` + GroupShuffleSplit | ✓ | ✗ | ✗ |
| `group` + SPXYGFold | ✓ | ✗ | ✗ |
| `group` + StratifiedGroupKFold | ✓ | ✓ (classification) | ✗ |
| **`force_group` (proposed)** | ✓ | ✓ (any) | ✓ |

### B. Pseudocode Summary

```
FORCE_GROUP SPLITTING ALGORITHM

Input:
  - X: features (n_samples × n_features)
  - y: targets (n_samples)
  - groups: group labels (n_samples)
  - splitter: any sklearn cross-validator
  - aggregation: "mean" | "median" | "first"

Algorithm:
  1. AGGREGATE BY GROUPS
     unique_groups = unique(groups)
     for each group g in unique_groups:
       X_rep[g] = aggregate(X[groups == g])
       y_rep[g] = aggregate(y[groups == g])
       indices[g] = where(groups == g)

  2. SPLIT ON REPRESENTATIVES
     for train_rep, test_rep in splitter.split(X_rep, y_rep):

  3. EXPAND TO ORIGINAL INDICES
       train_idx = concat([indices[g] for g in train_rep])
       test_idx = concat([indices[g] for g in test_rep])
       yield train_idx, test_idx

Output:
  - Folds where all samples from same group are together
```

### C. References

1. SPXYGFold implementation: `nirs4all/operators/splitters/splitters.py`
2. CrossValidatorController: `nirs4all/controllers/splitters/split.py`
3. Roadmap entry: `Roadmap.md` line 22
4. sklearn splitters: https://scikit-learn.org/stable/modules/cross_validation.html
