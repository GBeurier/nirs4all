# Branching and Merging Analysis Report

**Version**: 1.0.0
**Date**: December 21, 2025
**Author**: Pipeline Architecture Analysis
**Status**: Analysis Complete

---

## Executive Summary

This report analyzes the branching and merging capabilities in nirs4all, identifies gaps between current implementation and desired features, and proposes a path forward. The analysis reveals that **branching is well-implemented** for preprocessing and model comparison, but **merging is essentially non-existent** as a first-class feature, and **source-aware branching is not yet implemented**.

---

## 1. Objectives Analysis

### 1.1 Requested Features (Reformulated)

The user has requested three key capabilities:

| Feature | Description | Current Status |
|---------|-------------|----------------|
| **Source Branching** | Create one branch per data source in multi-source datasets | ❌ Not implemented |
| **Feature Index Branching** | Create branches based on wavelength/feature ranges | ❌ Not implemented |
| **Flexible Merge** | Merge on features, predictions, or both; support hybrid strategies | ⚠️ Partial (stacking only) |

### 1.2 Discussion of Objectives

#### Source Branching (1 source per branch)

**Use Case**: In sensor fusion scenarios (e.g., NIR + Raman + Markers), users want to apply different preprocessing pipelines to each source independently, then combine results.

**Key Insight**: This is fundamentally different from current branching which clones the entire dataset. Source branching requires *slicing* the dataset by source index and routing each slice through a dedicated pipeline branch.

**Complexity**: Medium-high. Requires:
- Source-aware branch creation
- Tracking which source each branch operates on
- Merge logic to recombine sources after branch processing

#### Feature Index Branching (wavelength ranges)

**Use Case**: Different spectral regions may benefit from different preprocessing. For example:
- 400-700nm: Use derivative
- 700-1100nm: Use SNV
- 1100-2500nm: Use MSC

**Key Insight**: This is essentially a generalization of source branching at the feature level. It can work two ways:
1. **Within a source**: Split features by index ranges
2. **Across sources**: When multi-source is concatenated, allow splitting the concatenated feature space

**Complexity**: High. Requires:
- Feature slicing in branch creation
- Tracking original feature indices through transformations
- Merge logic to recombine feature ranges

**Challenge**: After transformations like PCA, the original feature indices lose meaning. Need clear semantics for when index branching is applicable.

#### Flexible Merge Strategies

**Use Case**: Combine branch outputs in different ways:
1. **Feature merge**: Concatenate transformed features → single model
2. **Prediction merge**: Average/vote on branch predictions → final prediction
3. **Hybrid merge**: Use branch predictions as features for a meta-model (stacking)

**Current State**: Only (3) is partially supported via `MetaModel` stacking, but there's no explicit merge controller or feature concatenation after branches.

---

## 2. Current State Analysis

### 2.1 Implemented Branching Controllers

| Controller | Location | Purpose | Maturity |
|------------|----------|---------|----------|
| `BranchController` | [controllers/data/branch.py](../../nirs4all/controllers/data/branch.py) | General preprocessing branching | ✅ Production |
| `OutlierExcluderController` | [controllers/data/outlier_excluder.py](../../nirs4all/controllers/data/outlier_excluder.py) | Branch by outlier exclusion strategy | ✅ Production |
| `SamplePartitionerController` | [controllers/data/sample_partitioner.py](../../nirs4all/controllers/data/sample_partitioner.py) | Branch by sample subset (outlier/inlier) | ✅ Production |
| `SourceBranchController` | N/A | Branch by data source | ❌ Not implemented |
| `FeatureSlicerController` | N/A | Branch by feature index range | ❌ Not implemented |

### 2.2 Branching Architecture

The current branching system works as follows:

```
Pipeline Start
      │
      ▼
┌─────────────────┐
│ Shared Steps    │  ← Cross-validation split, initial preprocessing
└────────┬────────┘
         │
    {"branch": [...]}
         │
    ┌────┴────┬────────┐
    ▼         ▼        ▼
┌───────┐ ┌───────┐ ┌───────┐
│Branch0│ │Branch1│ │Branch2│  ← Independent contexts, cloned dataset features
└───┬───┘ └───┬───┘ └───┬───┘
    │         │        │
    └────┬────┴────────┘
         │
    Post-branch steps
         │
         ▼
   (Each branch continues independently)
```

**Key Design Decisions**:

1. **Feature Snapshotting**: Each branch receives a deep copy of features via `_snapshot_features()` and `_restore_features()`. This ensures isolation but is memory-intensive.

2. **Context Propagation**: Branches create independent `ExecutionContext` copies with unique `branch_id`, `branch_name`, and `branch_path`.

3. **Post-branch Iteration**: Steps after a branch block iterate over `branch_contexts` list, executing on each branch independently.

4. **Nested Branching**: Supported via `_multiply_branch_contexts()` which creates Cartesian product of parent × child branches.

### 2.3 Merge Situation

**Critical Finding**: The file [controllers/data/merge.py](../../nirs4all/controllers/data/merge.py) **exists but is completely empty**.

Current "merge" behaviors:

| Behavior | How It Works | Limitation |
|----------|--------------|------------|
| **Implicit iteration** | Post-branch steps run on each branch separately | No actual merging—branches stay independent |
| **Stacking (MetaModel)** | Collects OOF predictions from source models | Prediction-level merge only, specific to stacking |
| **Cross-branch stacking** | `crossbranch.py` validates and collects across branches | Still prediction-level, no feature merge |

### 2.4 Multi-Source Handling

The `Features` class manages multi-source data:

```python
class Features:
    sources: List[FeatureSource]  # One per source

    def x(self, indices, layout="2d", concat_source=True):
        # If concat_source=True: concatenate along feature axis
        # If concat_source=False: return list of arrays
```

**How Transformers Handle Multi-Source**:

The `TransformerMixinController` iterates over sources:

```python
for sd_idx, (fit_x, all_x) in enumerate(zip(fit_data, all_data)):
    for processing_idx in range(fit_x.shape[1]):
        # Fit and transform per source, per processing
```

This means the same transformer is applied to each source. There's no mechanism to apply *different* transformers to different sources.

### 2.5 Identified Problems, Flaws, and Antipatterns

#### 2.5.1 Architecture Issues

| Issue | Description | Severity |
|-------|-------------|----------|
| **Empty merge.py** | File exists but has no implementation | 🔴 Critical |
| **No explicit merge step** | Branches never formally merge—they just continue separately | 🔴 Critical |
| **Implicit iteration anti-pattern** | Post-branch behavior assumes all branches want the same downstream steps | 🟡 Medium |
| **Memory-intensive snapshotting** | Deep-copying features for each branch is wasteful | 🟡 Medium |
| **No source-aware branching** | Cannot route different sources to different pipelines | 🔴 Critical |

#### 2.5.2 Implementation Gaps

| Gap | Description | Impact |
|-----|-------------|--------|
| **No feature-level merge** | Cannot concatenate branch features before a model | High |
| **No prediction averaging without stacking** | Simple branch prediction averaging requires MetaModel | Medium |
| **No partial source selection** | Cannot select subset of sources for a step | Medium |
| **3D layout with asymmetric sources** | Fails with cryptic numpy error | High |

#### 2.5.3 Loopholes and Edge Cases

| Edge Case | Current Behavior | Expected Behavior |
|-----------|------------------|-------------------|
| Branch with no steps | Silent no-op | Should warn |
| Nested branches × sources | Creates explosion of contexts | Should validate feasibility |
| Sample partitioner + cross-branch stacking | Silently broken (disjoint samples) | Should error early |
| Asymmetric processings in 3D concat | Numpy broadcasting error | Clear error with options |

---

## 3. Gap Analysis

### 3.1 Feature Gap Summary

| Desired Feature | Status | Gap Description |
|-----------------|--------|-----------------|
| Source branching | ❌ Missing | No `SourceBranchController` |
| Feature index branching | ❌ Missing | No `FeatureSlicerController` |
| Feature merge (concat) | ❌ Missing | No merge controller |
| Prediction merge (average/vote) | ⚠️ Partial | Only via MetaModel, not explicit |
| Hybrid merge (predictions → features) | ⚠️ Partial | Stacking supports this but requires MetaModel |
| Explicit merge syntax | ❌ Missing | No `{"merge": ...}` step type |

### 3.2 Documentation Redundancy

Multiple documents discuss branching/merging in overlapping ways:

| Document | Focus | Issue |
|----------|-------|-------|
| [branching.md](../reference/branching.md) | Pipeline branching reference | Good, focused |
| [asymmetric_sources_design.md](../specifications/asymmetric_sources_design.md) | Multi-source + source branching design | Mixes dataset loading with runtime branching |
| [dataset_config_roadmap.md](../specifications/dataset_config_roadmap.md) | Dataset loading implementation | Contains merge strategies that belong in pipeline docs |
| [dataset_config_specification.md](../specifications/dataset_config_specification.md) | Dataset configuration | Contains aspirational merge features |

**Recommendation**: Dataset config documents should focus purely on data **loading**. Branching and merging are **runtime pipeline concerns** and should be in separate documents.

---

## 4. Recommendations

### 4.1 Proposed Pipeline Syntax Extensions

#### Source Branching

```python
# Route each source to its own branch
pipeline = [
    ShuffleSplit(n_splits=5),
    {"source_branch": {
        "NIR": [SNV(), PCA(n_components=20)],
        "Raman": [Baseline(), Normalize()],
        "Markers": [VarianceThreshold(threshold=0.01)],
    }},
    {"merge": "features"},  # Concatenate transformed features
    PLSRegression(n_components=10),
]

# Alternative: automatic one-branch-per-source
pipeline = [
    {"source_branch": "auto"},  # Creates one branch per source
    [SNV()],  # Applied to each source branch
    {"merge": "features"},
]
```

#### Feature Index Branching

```python
# Branch by wavelength ranges (for single-source or concatenated multi-source)
pipeline = [
    ShuffleSplit(n_splits=5),
    {"feature_branch": {
        "visible": {"range": [0, 300], "steps": [FirstDerivative()]},
        "nir_short": {"range": [300, 600], "steps": [SNV()]},
        "nir_long": {"range": [600, 1000], "steps": [MSC()]},
    }},
    {"merge": "features"},
    PLSRegression(n_components=10),
]
```

#### Explicit Merge Step

```python
# Feature merge (concatenate along feature axis)
{"merge": "features"}

# Prediction merge (average predictions from models in branches)
{"merge": "predictions", "method": "mean"}  # or "weighted_mean", "vote", "best"

# Hybrid merge (use predictions as features for meta-model)
{"merge": "predictions", "as_features": True}
```

### 4.2 Implementation Approach

#### Phase 1: MergeController Foundation

1. Implement `MergeController` in [controllers/data/merge.py](../../nirs4all/controllers/data/merge.py)
2. Support three merge modes:
   - `features`: Concatenate branch feature matrices
   - `predictions`: Aggregate branch predictions
   - `predictions_as_features`: Route predictions to next step as features

3. Handle incompatible dimensions:
   - **2D features with different sizes**: Pad with zeros or error based on config
   - **Different sample counts**: Error (sample partitioner branches can't merge features)

#### Phase 2: SourceBranchController

1. Implement `SourceBranchController` with syntax:
   ```python
   {"source_branch": {
       "source_name": [steps],
       # or
       0: [steps],  # by index
   }}
   ```

2. Each branch receives only its assigned source(s)
3. After merge, recombine into multi-source structure

#### Phase 3: FeatureSlicerController

1. Implement `FeatureSlicerController` with syntax:
   ```python
   {"feature_branch": {
       "name": {"range": [start, end], "steps": [...]},
   }}
   ```

2. Track original indices for recombination
3. Validate that transformers don't change feature count unexpectedly

### 4.3 Handling Impossible Situations

| Situation | Recommendation |
|-----------|----------------|
| **Asymmetric processing counts in 3D layout** | Error with clear message listing options: flatten first, get as list, or use 2D |
| **Feature merge with different sample counts** | Error: sample partitioner branches cannot feature-merge |
| **Prediction merge with missing models** | Warn and use available predictions |
| **Nested source_branch within branch** | Error: source branching should be top-level |

### 4.4 Proposed Error Messages

```python
# Asymmetric source concat
SourceConcatError: Cannot concatenate sources in 3D layout.

Sources have different processing counts:
  - Source 0 (NIR): 3 processings
  - Source 1 (Raman): 1 processing

Resolution options:
  1. Use layout="2d" to flatten processings first
  2. Use concat_source=False to get sources as list
  3. Apply same number of preprocessings to all sources
  4. Use source_branch to process sources independently then merge

# Feature merge incompatible
BranchMergeError: Cannot merge features from branches with different sample counts.

Branch samples:
  - Branch 0 (inliers): 450 samples
  - Branch 1 (outliers): 50 samples

sample_partitioner branches produce disjoint sample sets and cannot
be feature-merged. Use prediction merge instead:

  {"merge": "predictions", "method": "mean"}
```

---

## 5. What Could Be Done Better Overall

### 5.1 Architecture Improvements

1. **Explicit Merge Step**: Branching without merging is like opening a file without closing it. Every branch should have a corresponding merge.

2. **Branch Context as First-Class Object**: Instead of storing branch contexts in `context.custom["branch_contexts"]`, create a `BranchContext` class that manages lifecycle, validation, and merging.

3. **Lazy Feature Snapshotting**: Instead of deep-copying all features for each branch, use copy-on-write or reference counting. Only copy when a branch modifies features.

4. **Branch Validation Layer**: Add upfront validation before executing branches:
   - Check if merge is possible
   - Warn about memory implications
   - Validate source/feature references

### 5.2 API Improvements

1. **Unified Branch Syntax**: All branch types should use the same base pattern:
   ```python
   {"branch": {...}, "by": "preprocessing|source|feature|outlier|partition"}
   ```

2. **Explicit Branch Names**: Require named branches for complex pipelines to aid debugging and visualization.

3. **Branch Diagram Generation**: Automatically generate Mermaid/graphviz diagrams showing branch structure.

### 5.3 Documentation Improvements

1. **Separation of Concerns**: Keep dataset loading docs (dataset_config_*) separate from runtime pipeline docs (branching, merging).

2. **Decision Tree for Users**: Create a flowchart helping users choose between:
   - Preprocessing branch vs source branch vs feature branch
   - Feature merge vs prediction merge vs stacking

3. **Examples for Each Pattern**: Create Q-examples demonstrating:
   - Q32 or similar: Source branching with merge
   - Q33 or similar: Feature range branching with merge
   - Q34 or similar: Hybrid merge (predictions → meta-model)

---

## 6. Conclusion

The nirs4all branching system is well-designed for preprocessing comparison but lacks critical features for advanced multi-source and hybrid ensemble workflows:

1. **Source branching** is the top priority—users with multi-instrument data need this.
2. **Explicit merge** is essential to complete the branching story.
3. **Feature index branching** is useful but lower priority than the above.

The recommended implementation order is:
1. `MergeController` (foundation for all merge operations)
2. `SourceBranchController` (most requested feature)
3. `FeatureSlicerController` (nice-to-have)

Document cleanup should separate dataset configuration (loading) from pipeline runtime (branching/merging) to reduce confusion.

---

## Appendix: File Reference

| File | Current State | Proposed Action |
|------|---------------|-----------------|
| [controllers/data/merge.py](../../nirs4all/controllers/data/merge.py) | Empty | Implement MergeController |
| [controllers/data/branch.py](../../nirs4all/controllers/data/branch.py) | Complete | No change needed |
| [asymmetric_sources_design.md](../specifications/asymmetric_sources_design.md) | Mixes concerns | Extract merge content to pipeline docs |
| [dataset_config_roadmap.md](../specifications/dataset_config_roadmap.md) | Contains merge | Remove merge sections |
| [dataset_config_specification.md](../specifications/dataset_config_specification.md) | Contains merge aspirations | Remove merge sections |
