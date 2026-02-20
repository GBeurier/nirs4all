# Workflow Operators v2 - Migration Guide

This guide helps you migrate from the old workflow keywords to the new v2 syntax.

---

## Overview of Changes

### Keywords Removed

| Keyword | Replacement |
|---------|-------------|
| `sample_filter` | `exclude` |
| `source_branch` | `branch` with `by_source` |
| `merge_sources` | `merge` with `sources` key |

### Keywords Added

| Keyword | Purpose |
|---------|---------|
| `tag` | Mark samples without removal (for analysis/branching) |
| `exclude` | Mark samples AND exclude from training |

### Controllers Deleted

The following controllers have been removed from the codebase:

- `SampleFilterController` - replaced by `ExcludeController`
- `SourceBranchController` - absorbed into `BranchController`
- `OutlierExcluderController` - use `branch` with `exclude` steps
- `SamplePartitionerController` - use `branch` with `by_tag`
- `MetadataPartitionerController` - use `branch` with `by_metadata`

---

## Quick Reference Table

| Old Syntax | New Syntax |
|------------|------------|
| `{"sample_filter": Filter()}` | `{"exclude": Filter()}` |
| `{"sample_filter": {"filters": [F1, F2], "mode": "any"}}` | `{"exclude": [F1, F2], "mode": "any"}` |
| `{"source_branch": {"NIR": [...], "UV": [...]}}` | `{"branch": {"by_source": True, "steps": {"NIR": [...], "UV": [...]}}}` |
| `{"merge_sources": "concat"}` | `{"merge": {"sources": "concat"}}` |
| `{"branch": {"by": "metadata_partitioner", "column": "site"}}` | `{"branch": {"by_metadata": "site"}}` |
| `{"branch": {"by": "outlier_excluder", "strategies": [...]}}` | Use `{"tag": ...}` then `{"branch": {"by_tag": ...}}` |

---

## Detailed Migration Examples

### 1. Sample Filtering

**Before (v1):**
```python
from nirs4all.operators.filters import YOutlierFilter, XOutlierFilter

pipeline = [
    {"sample_filter": {
        "filters": [YOutlierFilter(method="iqr"), XOutlierFilter(method="mahalanobis")],
        "mode": "any",
        "report": True
    }},
    MinMaxScaler(),
    PLSRegression(10),
]
```

**After (v2):**
```python
from nirs4all.operators.filters import YOutlierFilter, XOutlierFilter

pipeline = [
    {"exclude": [YOutlierFilter(method="iqr"), XOutlierFilter(method="mahalanobis")], "mode": "any"},
    MinMaxScaler(),
    PLSRegression(10),
]
```

### 2. Source Branch

**Before (v1):**
```python
pipeline = [
    {"source_branch": {
        "NIR": [SNV(), SavitzkyGolay(window=11, polyorder=2)],
        "UV": [MinMaxScaler()],
    }},
    {"merge_sources": "concat"},
    PLSRegression(10),
]
```

**After (v2):**
```python
pipeline = [
    {"branch": {
        "by_source": True,
        "steps": {
            "NIR": [SNV(), SavitzkyGolay(window=11, polyorder=2)],
            "UV": [MinMaxScaler()],
        }
    }},
    {"merge": {"sources": "concat"}},
    PLSRegression(10),
]
```

### 3. Metadata Partitioning

**Before (v1):**
```python
pipeline = [
    {"branch": {
        "by": "metadata_partitioner",
        "column": "site",
    }},
    {"merge": "predictions"},
    Ridge(),
]
```

**After (v2):**
```python
pipeline = [
    {"branch": {"by_metadata": "site"}},
    {"merge": "concat"},  # Separation branches use concat
    Ridge(),
]
```

### 4. Outlier-Based Branching

**Before (v1):**
```python
pipeline = [
    {"branch": {
        "by": "outlier_excluder",
        "strategies": [None, {"method": "mahalanobis"}, {"method": "isolation_forest"}]
    }},
    PLSRegression(10),
    {"merge": "predictions"},
]
```

**After (v2) - Using tag then branch:**
```python
from nirs4all.operators.filters import XOutlierFilter

pipeline = [
    # First, tag samples (computes outlier flags without removing)
    {"tag": XOutlierFilter(method="mahalanobis", tag_name="mahal_outlier")},

    # Then branch by tag
    {"branch": {
        "by_tag": "mahal_outlier",
        "values": {
            "clean": False,    # Non-outliers
            "outliers": True,  # Outliers
        }
    }},
    PLSRegression(10),
    {"merge": "concat"},  # Reassemble samples
]
```

**Alternative (v2) - Separate pipelines with exclude:**
```python
# If you want different models for clean vs outlier-filtered data,
# use duplication branches with exclude in each:
pipeline = [
    {"branch": [
        [PLSRegression(10)],  # All samples
        [{"exclude": XOutlierFilter(method="mahalanobis")}, PLSRegression(10)],  # Filtered
    ]},
    {"merge": "predictions"},  # Stacking
    Ridge(),
]
```

### 5. Sample Partitioning by Tag

**Before (v1):** Not directly available - required custom controller

**After (v2):**
```python
from nirs4all.operators.filters import YOutlierFilter

pipeline = [
    # Tag samples first
    {"tag": YOutlierFilter(method="zscore", threshold=2.5, tag_name="extreme_y")},

    # Branch by tag - disjoint sample subsets
    {"branch": {
        "by_tag": "extreme_y",
        "values": {
            "normal": False,
            "extreme": True,
        }
    }},

    # Each branch gets its samples processed
    PLSRegression(10),

    # Reassemble all samples
    {"merge": "concat"},
]
```

---

## Value Mapping Syntax

The new `by_tag` and `by_metadata` branches support user-friendly value conditions:

```python
# Boolean conditions
{"by_tag": "is_outlier", "values": {"clean": False, "outliers": True}}

# List conditions
{"by_metadata": "site", "values": {"group_a": ["site1", "site2"], "group_b": ["site3"]}}

# Comparison conditions
{"by_metadata": "temperature", "values": {"cold": "< 20", "warm": ">= 20"}}

# Range conditions
{"by_metadata": "age", "values": {"young": "0..30", "middle": "30..60", "senior": "60.."}}

# Lambda conditions
{"by_metadata": "quality", "values": {"high": lambda x: x > 0.8}}
```

---

## Breaking Changes

### No Deprecation Period

Old keywords will error immediately. There is no backward compatibility layer.

**If you see these errors, migrate your syntax:**

- `KeyError: 'sample_filter'` - Use `exclude` instead
- `KeyError: 'source_branch'` - Use `branch` with `by_source`
- `KeyError: 'merge_sources'` - Use `merge` with `sources` key
- `ValueError: Unknown branch mode 'metadata_partitioner'` - Use `by_metadata`
- `ValueError: Unknown branch mode 'outlier_excluder'` - Use `tag` + `branch` with `by_tag`

### Behavior Changes

1. **`exclude` never runs during prediction** - Tags are computed fresh on prediction data, but samples are never excluded during prediction.

2. **Separation branches require `concat` merge** - When using `by_tag`, `by_metadata`, or `by_filter`, use `{"merge": "concat"}` to reassemble samples.

3. **Tags persist in dataset** - Tags created with the `tag` keyword are stored in the dataset's indexer and available for analysis.

---

## New Features

### Tag Keyword

Mark samples for analysis without removing them:

```python
from nirs4all.operators.filters import YOutlierFilter, XOutlierFilter, SpectralQualityFilter

# Single filter
{"tag": YOutlierFilter(method="iqr")},

# Multiple filters - each creates its own tag
{"tag": [
    YOutlierFilter(method="iqr"),
    XOutlierFilter(method="mahalanobis"),
    SpectralQualityFilter(),
]},

# Named filters
{"tag": {
    "y_extreme": YOutlierFilter(method="zscore", threshold=3),
    "x_extreme": XOutlierFilter(method="robust_mahalanobis"),
}},
```

Tags are stored in the dataset and can be accessed:

```python
result = nirs4all.run(pipeline, dataset)
# Tags available via dataset.get_tag("y_outlier_iqr")
```

### Separation Branches

Create disjoint sample subsets with guaranteed non-overlap:

```python
# By metadata column
{"branch": {"by_metadata": "instrument"}},

# By tag value
{"branch": {"by_tag": "quality_tier", "values": {"high": "> 0.8", "medium": "0.5..0.8", "low": "< 0.5"}}},

# By filter result
{"branch": {"by_filter": XOutlierFilter(method="mahalanobis")}},

# By source (multi-source datasets)
{"branch": {"by_source": True, "steps": {...}}},
```

### Unified Merge Syntax

Source merging is now unified under the `merge` keyword:

```python
# Old syntax
{"merge_sources": "concat"}
{"merge_sources": "stack"}

# New syntax
{"merge": {"sources": "concat"}}
{"merge": {"sources": "stack"}}
```

---

## Migration Checklist

- [ ] Replace all `sample_filter` with `exclude`
- [ ] Replace all `source_branch` with `branch` using `by_source`
- [ ] Replace all `merge_sources` with `merge` using `sources` key
- [ ] Replace `branch: {by: "metadata_partitioner"}` with `branch: {by_metadata: ...}`
- [ ] Replace `branch: {by: "outlier_excluder"}` patterns with `tag` + `branch: {by_tag: ...}`
- [ ] Verify separation branches use `{"merge": "concat"}`
- [ ] Run tests: `pytest tests/`
- [ ] Run examples: `cd examples && ./run.sh -q`

---

*Migration guide version: 1.0*
*Created: 2026-01-20*
*Associated roadmap: ROADMAP_workflow_v2.md*
