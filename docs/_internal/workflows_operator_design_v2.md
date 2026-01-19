# Workflow Operators Design v2

## Executive Summary

This document redesigns nirs4all's workflow operators around **four unified concepts**: Filters, Exclude, Branches, and Merge. The goal is to simplify the mental model, reduce keyword proliferation, and provide clear semantics for data flow through pipelines.

**Core Principles:**
1. **Filters tag, they don't remove** - General tagging without data modification
2. **Exclude removes with explicit intent** - Tag + remove samples from training
3. **Branches have two explicit modes** - Duplication (parallel paths) vs Separation (data splitting)
4. **Merge is the only exit** - All branches must eventually merge

---

## 1. Core Concepts

### 1.1 FILTERS (General Tagging)

**Purpose:** Populate sample indices with arbitrary tags without removing data.

**Key Characteristics:**
- Filters ONLY ADD TAGS - they do NOT exclude/remove data
- Tags are stored as additional columns in the Indexer
- Tag types: boolean, categorical (string), numeric
- Tags are non-destructive and reversible
- Can filter on: X values, Y values, metadata, partitions, existing tags

**Keyword:** `tag`

**Prediction Mode Behavior:** Tags are computed fresh on prediction data (same filters applied to new samples for consistent behavior).

**Syntax:**
```python
# Single tagger
{"tag": YOutlierFilter(method="iqr", threshold=1.5)}

# Multiple taggers
{"tag": [
    YOutlierFilter(method="iqr", tag_name="y_outlier_iqr"),
    SpectralQualityFilter(tag_name="low_quality"),
    ClusteringFilter(n_clusters=3, tag_name="cluster_id"),
]}

# Shorthand with inline tag name
{"tag": {"y_extreme": YOutlierFilter(method="zscore", threshold=3.0)}}
```

**Tag Storage (IndexStore):**
```
| sample | partition | excluded | tag:y_outlier_iqr | tag:cluster_id |
|--------|-----------|----------|-------------------|----------------|
| 0      | train     | False    | False             | "A"            |
| 1      | train     | False    | True              | "B"            |
| 2      | test      | False    | False             | "A"            |
```

**Mental Model:** Tags are like metadata columns that you compute from the data itself.

---

### 1.2 EXCLUDE (Tag + Remove from Training)

**Purpose:** Tag samples AND remove them from training data.

**Key Characteristics:**
- Explicit exclusion with clear intent ("exclude" means remove)
- Tagged samples are removed from training
- Configurable: `remove=False` to tag-only (keep in training)
- Prediction mode: exclusion rules never applied (samples pass through)
- Uses existing `excluded` + `exclusion_reason` indexer columns

**Keyword:** `exclude`

**Syntax:**
```python
# Standard: tag + remove from training
{"exclude": YOutlierFilter(method="iqr")}

# Multiple detectors (exclude if ANY flags)
{"exclude": [
    YOutlierFilter(method="iqr", threshold=1.5),
    XOutlierFilter(method="mahalanobis", threshold=3.0),
], "mode": "any"}

# Tag-only mode (no removal)
{"exclude": YOutlierFilter(method="iqr"), "remove": False}

# Exclude only if ALL methods agree
{"exclude": [...], "mode": "all"}
```

**Distinction from Filters:**

| Aspect | `tag` (Filter) | `exclude` |
|--------|----------------|-----------|
| Default behavior | Tag only | Tag + remove |
| Training data | Unchanged | Excluded samples removed |
| Prediction mode | Tags applied | No removal applied |
| Use case | Grouping, analysis | Data cleaning, outlier removal |

---

### 1.3 BRANCHES (Path Creation)

**Purpose:** Create multiple execution paths with either duplicated or partitioned data.

**Two Explicit Modes:**

#### Mode A: Duplication Branches
- Same data appears in ALL branches
- Each branch processes identical samples
- Used for: comparing preprocessing, ensemble building

```python
# List syntax (anonymous branches)
{"branch": [
    [SNV(), PCA(10)],
    [MSC(), FirstDerivative()],
    [Detrend(), SavitzkyGolay()],
]}

# Dict syntax (named branches)
{"branch": {
    "snv_path": [SNV(), PCA(10)],
    "msc_path": [MSC(), FirstDerivative()],
}}

# Generator syntax (creates variants)
{"branch": {"_or_": [SNV, MSC, Detrend]}}
```

#### Mode B: Separation Branches
- Data is SPLIT into non-overlapping subsets
- Each sample appears in exactly ONE branch
- Used for: per-group models, stratified processing
- **Two sub-modes:** shared steps OR per-branch steps

```python
# By existing tag (shared steps follow after)
{"branch": {"by_tag": "cluster_id"}}
# Creates one branch per unique value, post-branch steps shared

# By metadata column
{"branch": {"by_metadata": "site"}}
# Creates one branch per site value

# By inline filter (tags + splits in one step)
{"branch": {"by_filter": ClusteringFilter(n_clusters=3)}}

# By exclusion tag (binary split)
{"branch": {"by_tag": "y_outlier_iqr"}}
# Creates: branch_true (excluded), branch_false (kept)

# Value mapping - USER-FRIENDLY SYNTAX
# String operators for common comparisons (no lambda required)
{"branch": {
    "by_tag": "quality_score",
    "values": {"high": "> 0.8", "low": "<= 0.8"}
}}

# Explicit list of values
{"branch": {
    "by_metadata": "site",
    "values": {"group_a": ["site_1", "site_2"], "group_b": ["site_3", "site_4"]}
}}

# Range-based grouping
{"branch": {
    "by_tag": "y_value",
    "values": {"low": "0..50", "medium": "50..100", "high": "100.."}
}}

# Boolean shorthand
{"branch": {
    "by_tag": "is_outlier",
    "values": {"outliers": True, "inliers": False}
}}

# DEVELOPER-LEVEL: Lambda for complex conditions
{"branch": {
    "by_tag": "quality_score",
    "values": {"high": lambda x: x > 0.8, "low": lambda x: x <= 0.8}
}}

# By source (multi-source datasets)
{"branch": {"by_source": True}}
# Creates one branch per data source

# Per-branch specific steps (when 'steps' key provided)
{"branch": {
    "by_tag": "cluster_id",
    "steps": {
        "cluster_A": [SNV(), PCA(10)],
        "cluster_B": [MSC(), PCA(5)],
        "cluster_C": [Detrend()],
    }
}}
```

**Post-Branch Behavior:**
- Operators after branches are duplicated to apply separately on each branch
- Model after branches without merge: model applied N times (one per branch)
- Branches multiply per source: 3 branches × 2 sources = 6 paths

```python
pipeline = [
    {"branch": [[SNV()], [MSC()]]},  # 2 branches
    PLSRegression(10),  # Applied 2 times (once per branch)
    # Still in branch mode until merge!
]
```

---

### 1.4 MERGE (Branch Combination)

**Purpose:** Combine branches back into unified data flow.

**Keyword:** `merge`

**Merge Strategies:**

| Strategy | Use Case | Description |
|----------|----------|-------------|
| `features` | Duplication branches | Concatenate feature matrices horizontally |
| `predictions` | Branches with models | Stack OOF predictions for meta-learning |
| `concat` | Separation branches | Vertically concatenate (reassemble samples) |
| `average` | Ensemble | Average predictions across branches |
| `voting` | Classification | Majority voting |
| `best` | Model selection | Select best branch by validation metric |

**Syntax:**
```python
# Feature concatenation (duplication branches)
{"merge": "features"}
# Result: [branch_0_features | branch_1_features | ...]

# Prediction stacking (for meta-models)
{"merge": "predictions"}
# Result: OOF predictions as new features

# Sample reassembly (separation branches)
{"merge": "concat"}
# Result: All samples back together with their branch-specific predictions

# Advanced configuration
{"merge": {
    "mode": "predictions",
    "selection": "top_k",  # Select top K models per branch
    "k": 2,
    "aggregation": "weighted_mean",
    "weights": "validation_score"
}}

# Per-branch selection
{"merge": {
    "predictions": [
        {"branch": 0, "select": "best", "metric": "rmse"},
        {"branch": 1, "select": "all"},
    ],
    "features": [2],  # Also include features from branch 2
}}
```

**Error Handling:**
- `on_asymmetric`: "error" | "pad" | "truncate" | "warn"
- `on_missing`: "error" | "fill_nan" | "exclude"
- `alignment`: "strict" | "sample_id" | "index"

---

## 2. Multi-Source Behavior

### Default Behavior
By default, the same pipeline applies to all sources separately:

```python
# Data: (Source_NIR, Source_Raman)
pipeline = [MSC()]
# Equivalent to:
# Source_NIR -> MSC_instance_1
# Source_Raman -> MSC_instance_2
```

### Before Model: Auto-Aggregation
When a model step is encountered, sources are automatically aggregated:

```python
pipeline = [
    MSC(),                    # Applied per-source
    PLSRegression(10),        # Sources auto-concatenated before model
]
# Flow: [NIR_msc | Raman_msc] -> PLS
```

Aggregation method is configurable:
- Default: horizontal concatenation
- Model can request specific layout via `source_layout` parameter

### With Branches
Branches multiply per source unless explicitly source-specific:

```python
# 2 sources x 3 duplication branches = 6 paths
pipeline = [
    {"branch": [[SNV()], [MSC()], [Detrend()]]},
    PLSRegression(10),  # Applied 6 times!
]

# Source-specific branching (no multiplication)
pipeline = [
    {"branch": {"by_source": True}},  # 1 branch per source
    # NIR -> branch_0, Raman -> branch_1
]
```

### Source Merge
```python
{"merge": {"sources": "concat"}}  # Default before model
{"merge": {"sources": "stack"}}   # 3D tensor
{"merge": {"sources": "dict"}}    # Keep separate for multi-input models
```

---

## 3. Current State Analysis

### 3.1 Existing Keywords and Controllers

| Current Keyword | Controller | What It Does |
|-----------------|------------|--------------|
| `sample_filter` | SampleFilterController | Tags + excludes (conflated) |
| `branch: [[...]]` | BranchController | Duplication branches |
| `branch: {by: "outlier_excluder"}` | OutlierExcluderController | Compare outlier strategies as branches |
| `branch: {by: "sample_partitioner"}` | SamplePartitionerController | Split by filter result |
| `branch: {by: "metadata_partitioner"}` | MetadataPartitionerController | Split by metadata column |
| `source_branch` | SourceBranchController | Per-source pipelines |
| `merge` | MergeController | Combine branches |
| `merge_sources` | MergeController | Combine sources |
| `merge_predictions` | MergeController | Stack predictions |

### 3.2 Identified Discrepancies

| Issue | Current State | New Design |
|-------|---------------|------------|
| **No general tagging** | Only `excluded` (boolean) | Dynamic tag columns |
| **Overloaded `branch`** | 5+ different `{by: "..."}` modes | Two explicit modes: duplication vs separation |
| **Conflated filter/outlier** | `sample_filter` does both | Separate `tag` and `outlier` keywords |
| **Scattered source handling** | `source_branch` separate from `branch` | Unified under `branch: {by_source: ...}` |
| **Implicit multi-source** | Behavior not always clear | Explicit rules documented |
| **Multiple merge keywords** | `merge`, `merge_sources`, `merge_predictions` | Single `merge` with configuration |

### 3.3 Controller Priority Conflicts

```
Current priorities:
- SamplePartitionerController: 3
- MetadataPartitionerController: 3
- OutlierExcluderController: 4
- BranchController: 5
- SourceBranchController: 5
- MergeController: 5
- SampleFilterController: 5
```

The overlapping priorities with different `matches()` logic creates confusing routing behavior.

---

## 4. Migration Requirements

### 4.1 New Components

| Component | Type | Location | Purpose |
|-----------|------|----------|---------|
| `TagController` | Controller | `controllers/data/tag.py` | Handle `tag` keyword |
| `ExcludeController` | Controller | `controllers/data/exclude.py` | Handle `exclude` keyword |
| Tag column support | Enhancement | `data/_indexer/index_store.py` | Store arbitrary tags |
| Tag filtering | Enhancement | `pipeline/config/context.py` | Filter by tags in DataSelector |

### 4.2 Modified Components

| Component | Changes |
|-----------|---------|
| `BranchController` | Unify all branch types, add `by_tag`, `by_metadata`, `by_filter`, `by_source` |
| `MergeController` | Add `concat` strategy, consolidate `merge_sources`/`merge_predictions` |
| `SampleFilter` base class | Add `tag_name` parameter |
| `IndexStore` | Dynamic column creation for tags |
| `DataSelector` | Add `tag_filters` field |

### 4.3 Components to Remove (Clean Break)

| Component | Replacement |
|-----------|-------------|
| `SampleFilterController` | `ExcludeController` + `TagController` |
| `OutlierExcluderController` | `branch: [...]` with `exclude` steps |
| `SamplePartitionerController` | `branch: {by_filter: ...}` |
| `MetadataPartitionerController` | `branch: {by_metadata: ...}` |
| `SourceBranchController` | `branch: {by_source: ...}` |

**Note:** No deprecation shims - these controllers will be deleted immediately.

### 4.4 Files Impacted

**Core Infrastructure:**
- `nirs4all/data/_indexer/index_store.py` - Tag column support
- `nirs4all/data/_indexer/query_builder.py` - Tag filtering queries
- `nirs4all/pipeline/config/context.py` - DataSelector tag filters

**Controllers:**
- `nirs4all/controllers/data/tag.py` - NEW
- `nirs4all/controllers/data/exclude.py` - NEW
- `nirs4all/controllers/data/branch.py` - Major refactor
- `nirs4all/controllers/data/merge.py` - Enhancements
- `nirs4all/controllers/data/sample_filter.py` - DELETE
- `nirs4all/controllers/data/outlier_excluder.py` - DELETE
- `nirs4all/controllers/data/source_branch.py` - DELETE

**Operators:**
- `nirs4all/operators/filters/base.py` - Add `tag_name` parameter

**Tests:**
- `tests/unit/controllers/test_tag_controller.py` - NEW
- `tests/unit/controllers/test_exclude_controller.py` - NEW
- `tests/unit/data/indexer/test_tag_columns.py` - NEW
- `tests/integration/pipeline/test_new_branch_modes.py` - NEW

**Documentation:**
- `examples/user/` - Update all filtering/branching examples
- `docs/source/` - Update API documentation

---

## 5. High-Level Roadmap

### Phase 1: Foundation - Indexer Tag Infrastructure
**Goal:** Enable storing arbitrary tags in the indexer

**Tasks:**
1. Extend `IndexStore` with `add_tag_column()` method
2. Implement `set_tags()` and `get_tags()` methods
3. Add tag column support to `QueryBuilder`
4. Unit tests for tag operations
5. Update `SpectroDataset` to expose tag operations

**Estimated complexity:** Medium

### Phase 2: Tag Controller
**Goal:** Implement general tagging via `tag` keyword

**Tasks:**
1. Create `TagController` class
2. Implement tag propagation through branches
3. Add `tag_filters` to `DataSelector`
4. Ensure tags persist through prediction mode
5. Integration tests

**Estimated complexity:** Medium

### Phase 3: Exclude Controller
**Goal:** Implement explicit exclusion via `exclude` keyword

**Tasks:**
1. Create `ExcludeController` class
2. Implement `remove` config option (default: True)
3. Delete `SampleFilterController` (clean break)
4. Migration tests for new syntax

**Estimated complexity:** Medium

### Phase 4: Unified Branch Controller
**Goal:** Single branch controller handling all modes

**Tasks:**
1. Refactor `BranchController` with mode detection
2. Implement `by_tag` separation mode (shared + per-branch steps)
3. Implement `by_metadata` separation mode
4. Implement `by_filter` separation mode
5. Absorb `SourceBranchController` as `by_source`
6. Delete old controllers: `OutlierExcluderController`, `SamplePartitionerController`, `MetadataPartitionerController`, `SourceBranchController`
7. Comprehensive branch tests

**Estimated complexity:** High

### Phase 5: Merge Enhancements
**Goal:** Unified merge with all strategies

**Tasks:**
1. Add `concat` strategy for separation branches
2. Auto-detect branch type for merge strategy validation
3. Consolidate `merge_sources` into main `merge`
4. Add advanced configuration options
5. Error handling for asymmetric merges

**Estimated complexity:** Medium

### Phase 6: Documentation and Examples
**Goal:** Update all documentation and examples

**Tasks:**
1. Update all examples in `examples/user/`
2. Write migration guide (old syntax -> new syntax)
3. Update CLAUDE.md with new keywords
4. Update Sphinx documentation
5. Update any internal docs

**Estimated complexity:** Low

---

## 6. Example Pipelines: Before and After

### Example 1: Basic Outlier Exclusion

**Before:**
```python
pipeline = [
    {"sample_filter": {
        "filters": [YOutlierFilter(method="iqr", threshold=1.5)],
        "mode": "any"
    }},
    SNV(),
    PLSRegression(10)
]
```

**After:**
```python
pipeline = [
    {"exclude": YOutlierFilter(method="iqr", threshold=1.5)},
    SNV(),
    PLSRegression(10)
]
```

### Example 2: Tag for Analysis (No Removal)

**Before:** Not directly supported - had to use sample_filter with workarounds

**After:**
```python
pipeline = [
    {"tag": [
        YOutlierFilter(method="iqr", tag_name="y_outlier"),
        SpectralQualityFilter(tag_name="low_quality"),
    ]},
    SNV(),
    PLSRegression(10)
]
# Tags available in results for analysis, data not removed
```

### Example 3: Compare Exclusion Strategies

**Before:**
```python
pipeline = [
    ShuffleSplit(n_splits=3),
    {"branch": {
        "by": "outlier_excluder",
        "strategies": [
            None,
            {"method": "mahalanobis", "threshold": 3.0},
            {"method": "isolation_forest", "contamination": 0.05},
        ]
    }},
    PLSRegression(10)
]
```

**After:**
```python
pipeline = [
    ShuffleSplit(n_splits=3),
    {"branch": [
        [],  # baseline - no exclusion
        [{"exclude": XOutlierFilter(method="mahalanobis", threshold=3.0)}],
        [{"exclude": XOutlierFilter(method="isolation_forest", contamination=0.05)}],
    ]},
    PLSRegression(10)
]
```

### Example 4: Per-Cluster Models

**Before:**
```python
pipeline = [
    MinMaxScaler(),
    {"branch": {
        "by": "metadata_partitioner",
        "column": "cluster"
    }},
    PLSRegression(10),
    {"merge": "predictions"}
]
```

**After:**
```python
pipeline = [
    MinMaxScaler(),
    {"branch": {"by_metadata": "cluster"}},  # Separation branch
    PLSRegression(10),
    {"merge": "concat"}  # Reassemble samples
]
```

### Example 5: Tag-Based Separation (New Capability)

**After:**
```python
pipeline = [
    # First, tag samples by clustering
    {"tag": ClusteringFilter(n_clusters=3, tag_name="cluster")},

    # Then split by that tag
    {"branch": {"by_tag": "cluster"}},

    # Each cluster gets specialized preprocessing + model
    [SNV(), PLSRegression(5)],

    # Reassemble predictions
    {"merge": "concat"}
]
```

### Example 6: Stacking Ensemble

**Before & After (unchanged):**
```python
pipeline = [
    KFold(n_splits=5),
    {"branch": [
        [SNV(), PLSRegression(10)],
        [MSC(), RandomForestRegressor(n_estimators=100)],
    ]},
    {"merge": "predictions"},
    Ridge()
]
```

### Example 7: Multi-Source with Per-Source Preprocessing

**Before:**
```python
pipeline = [
    {"source_branch": {
        "NIR": [SNV(), SavitzkyGolay()],
        "Raman": [BaselineCorrection(), Normalize()],
    }},
    {"merge_sources": "concat"},
    PLSRegression(10)
]
```

**After:**
```python
pipeline = [
    {"branch": {
        "by_source": True,
        "steps": {
            "NIR": [SNV(), SavitzkyGolay()],
            "Raman": [BaselineCorrection(), Normalize()],
        }
    }},
    {"merge": {"sources": "concat"}},
    PLSRegression(10)
]
```

---

## 7. Design Decisions

1. **Backward compatibility:** No deprecation period - clean break
   - Old keywords removed immediately for cleaner codebase
   - Users must update pipelines when upgrading

2. **Tag prediction mode:** Tags computed fresh on prediction data
   - Same filters applied to new samples for consistent behavior
   - Enables tag-based routing in production

3. **Separation branch steps:** Both modes supported
   - Shared steps: `{"branch": {"by_tag": "..."}}` followed by post-branch steps
   - Per-branch steps: `{"branch": {"by_tag": "...", "steps": {...}}}`

## 8. Open Questions

1. **Tag persistence:** Should tags persist in exported bundles (.n4a)?
   - Recommendation: Yes, include in manifest for reproducibility

2. **Tag naming conflicts:** What if user creates tag with reserved name?
   - Recommendation: Prefix system tags with `_` (e.g., `_excluded`)

3. **Separation branch merge validation:** Error if wrong merge strategy?
   - Recommendation: Warning by default, error with `strict=True`

---

## Appendix: Keyword Reference

### New Keywords

| Keyword | Purpose | Example |
|---------|---------|---------|
| `tag` | General tagging | `{"tag": Filter()}` |
| `exclude` | Tag + remove from training | `{"exclude": Filter()}` |
| `branch` | Path creation | `{"branch": [...]}` or `{"branch": {"by_tag": "..."}}` |
| `merge` | Combine branches | `{"merge": "features"}` |

### Branch Modifiers

| Modifier | Mode | Example |
|----------|------|---------|
| (list) | Duplication | `{"branch": [[...], [...]]}` |
| `by_tag` | Separation | `{"branch": {"by_tag": "cluster"}}` |
| `by_metadata` | Separation | `{"branch": {"by_metadata": "site"}}` |
| `by_filter` | Separation | `{"branch": {"by_filter": Filter()}}` |
| `by_source` | Separation | `{"branch": {"by_source": True}}` |

### Merge Strategies

| Strategy | Use Case |
|----------|----------|
| `features` | Concat feature matrices (duplication) |
| `predictions` | Stack OOF predictions (meta-learning) |
| `concat` | Reassemble samples (separation) |
| `average` | Average predictions (ensemble) |
| `voting` | Majority vote (classification) |
| `best` | Select best branch |

### Value Mapping Syntax (for `values` parameter)

The `values` parameter in separation branches supports multiple user-friendly formats:

| Format | Example | Description |
|--------|---------|-------------|
| **String comparison** | `"> 0.8"`, `"<= 50"` | Comparison operators: `>`, `>=`, `<`, `<=`, `==`, `!=` |
| **Range** | `"0..50"`, `"50..100"`, `"100.."` | Inclusive range with `..`, open-ended with trailing `..` |
| **List of values** | `["a", "b", "c"]` | Match any value in list |
| **Boolean** | `True`, `False` | Direct boolean match |
| **Lambda** (advanced) | `lambda x: x > 0.8` | Custom Python function for complex conditions |

**Examples:**
```python
# Numeric thresholds (user-friendly)
"values": {"high": "> 80", "medium": "40..80", "low": "< 40"}

# Categorical grouping
"values": {"europe": ["FR", "DE", "IT"], "asia": ["JP", "CN", "KR"]}

# Boolean split
"values": {"outliers": True, "inliers": False}

# Complex conditions (developer-level)
"values": {"special": lambda x: x % 2 == 0 and x > 10}
```

---

*Document version: 2.0*
*Last updated: 2026-01-19*
