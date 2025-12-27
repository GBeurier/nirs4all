# Asymmetric Multi-Source Dataset Design

**Version**: 1.1.0-draft
**Status**: Draft Design Document
**Date**: December 2025
**Related Spec**: [dataset_config_specification.md](dataset_config_specification.md)

---

## Table of Contents

1. [Objective](#objective)
2. [Current State](#current-state)
3. [Problem Statement](#problem-statement)
4. [Proposition](#proposition)
5. [Source Branching & Merging Study](#source-branching--merging-study)
6. [Implementation Roadmap](#implementation-roadmap)

---

## Objective

Enable nirs4all to load and process **asymmetric multi-source datasets** where each source can have:

- Different feature dimensions (e.g., 500 vs 300 vs 50,000 features)
- Different numbers of preprocessing variations per source
- Different data modalities (spectral, time series, markers)

### Example Use Case

```
Source 1: NIRS spectra     (500 samples × 500 features)  - 3 preprocessings (raw, SNV, derivative)
Source 2: Time series      (500 samples × 300 features)  - 1 preprocessing (raw only)
Source 3: Genetic markers  (500 samples × 50,000 features) - 2 preprocessings (raw, scaled)
```

All sources share the same samples (rows aligned) but differ in their feature space and preprocessing history.

---

## Current State

### Data Structure

The current architecture supports multi-source datasets via the `Features` class:

```
Features
├── sources: List[FeatureSource]
│   ├── FeatureSource[0]  # Source 1
│   │   └── _storage: 3D array (samples × processings × features)
│   ├── FeatureSource[1]  # Source 2
│   │   └── _storage: 3D array (samples × processings × features)
│   └── ...
```

### Current Capabilities

| Feature | Status | Notes |
|---------|--------|-------|
| Multiple sources | ✅ Supported | Via list of paths in `train_x` |
| Different feature dims per source | ✅ Supported | Each `FeatureSource` has independent shape |
| Same processings across sources | ✅ Supported | All sources get same preprocessing pipeline |
| Different processings per source | ❌ Not supported | All sources share processing list |
| Asymmetric source concatenation | ⚠️ Partial | Works for 2D horizontal concat only |

### Execution Behaviors

When operators are applied to multi-source datasets, there are two dimensions of variation:

#### 1. Processing Dimension (within a source)

| Mode | Behavior | Use Case |
|------|----------|----------|
| **2D (concat)** | Flatten processings: `(samples, processings × features)` | sklearn models, most regressors |
| **3D (separate)** | Keep shape: `(samples, processings, features)` | Apply transformer to each processing separately |

#### 2. Source Dimension (across sources)

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Concatenate** | Merge horizontally: `(samples, Σ features)` | Single-input models |
| **Separate** | List of arrays: `[source1, source2, ...]` | Multi-head models, per-source transforms |

### Current Controller Behaviors

```python
# TransformerController: Iterates over sources, then processings
for sd_idx, (fit_x, all_x) in enumerate(zip(fit_data, all_data)):
    for processing_idx in range(fit_x.shape[1]):
        # Apply transformer to each (source, processing) pair
        transformer.fit(fit_2d)
        transformed = transformer.transform(all_2d)

# ModelController: Gets concatenated data
X_train = dataset.x(train_context.selector, layout="2d")  # Concatenates all
```

### Existing Branching Infrastructure

| Component | Status | Description |
|-----------|--------|-------------|
| `BranchController` | ✅ Implemented | Generic pipeline branching with `{"branch": [...]}` |
| `OutlierExcluderController` | ✅ Implemented | Specialized branch for outlier exclusion |
| `SamplePartitionerController` | ✅ Implemented | Specialized branch for sample partitioning |
| `SourceBranchController` | ❌ Not implemented | Branch by source (proposed) |
| `MergeController` | ❌ Empty file | Branch merging (not implemented) |

---

## Problem Statement

### Core Issue: 3D Concatenation with Asymmetric Preprocessing Counts

The **primary problem** is when requesting 3D layout with source concatenation when sources have different preprocessing counts:

```python
# Source 1: (500 samples, 3 processings, 500 features)
# Source 2: (500 samples, 1 processing, 300 features)

# Request:
X = dataset.x(selector, layout="3d", concat_source=True)
# FAILS: Cannot concatenate along feature axis when axis 1 (processings) differs
```

**Note**: Neural network input shape mismatch is NOT the core problem because nirs4all configures NN input shapes at runtime based on actual data dimensions. The real issue is numpy's inability to concatenate arrays with incompatible shapes.

### Failure Scenarios Table

| Scenario | Request | Sources | Result |
|----------|---------|---------|--------|
| A | `layout="2d", concat_source=True` | (500,500), (500,300) | ✅ Works: (500, 800) |
| B | `layout="3d", concat_source=True` | (500,2,500), (500,2,300) | ✅ Works: (500, 2, 800) |
| C | `layout="3d", concat_source=True` | (500,3,500), (500,1,300) | ❌ **Fails: incompatible pp counts** |
| D | `layout="2d", concat_source=True` | (500,3,500), (500,1,300) | ✅ Works: flattens then concats |
| E | `layout="3d", concat_source=False` | (500,3,500), (500,1,300) | ✅ Works: returns list |

### Resolution Strategies for Scenario C

When 3D concat is requested but preprocessing counts differ:

| Strategy | Behavior | Pros | Cons |
|----------|----------|------|------|
| **Error** | Raise clear error with resolution options | Explicit, no surprises | Requires user intervention |
| **Force 2D** | Flatten each source to 2D, then concat | Always works | Loses processing structure |
| **Separate** | Return list instead of single array | Preserves structure | Changes return type |

**Recommendation**: Default to **Error** with clear message, allow user to specify fallback behavior via `on_incompatible` parameter.

### Current Error Handling

Currently, these failures result in numpy broadcasting errors which are confusing:

```
ValueError: all input arrays must have the same shape except for the concatenation axis
```

---

## Proposition

### Design Principles

1. **Dataset config describes data, not processing** - Fusion strategy is a pipeline decision, not dataset config
2. **Runtime shape configuration** - NN models configure their input shapes at runtime, no pre-declaration needed
3. **Early validation with clear errors** - Detect incompatible configurations and explain resolution options
4. **Flexible execution strategies** - Support multiple approaches via pipeline syntax

### Dataset Configuration (Data-Only)

Dataset config should only describe the data structure, not how it's processed:

```yaml
sources:
  - name: "NIR"
    train_x: data/nir_train.csv
    params:
      header_unit: cm-1
    # Optional metadata (informational, not prescriptive)
    source_type: spectral          # spectral | timeseries | tabular | image
    allow_preprocessing: true      # Hint for pipeline validation

  - name: "markers"
    train_x: data/markers_train.csv
    source_type: tabular
    allow_preprocessing: false     # Pre-processed externally

targets:
  path: data/targets.csv

task_type: regression
```

**Removed from dataset config**: `source_fusion`, `shape_hint` - these are pipeline concerns.

### Pipeline-Level Fusion Control

Fusion strategy is controlled in the pipeline, not dataset config:

```python
pipeline = [
    # Per-source preprocessing
    {"preprocessing": SNV(), "sources": ["NIR", 0]},  # By name OR index
    {"preprocessing": MinMaxScaler(), "sources": "all"},

    # Source branching for complex per-source pipelines
    {"source_branch": {
        "NIR": [SNV(), SavitzkyGolay()],
        "markers": [VarianceThreshold()],
    }},

    # Merge branches (for features) or use multi-head model (for late fusion)
    {"merge_sources": "concat"},  # or "stack", "dict"

    # Model
    {"model": PLSRegression(n_components=10)},
]
```

### API Changes

#### Dataset Query API

```python
# Current
X = dataset.x(selector, layout="2d", concat_source=True)

# Extended
X = dataset.x(
    selector,
    layout="2d",
    concat_source=True,
    sources=["NIR", 0, 2],           # Select by name OR index
    on_incompatible="error"          # "error" | "flatten" | "separate"
)
```

#### Source Selection by Name or Index

```python
# All equivalent:
dataset.x(selector, sources=["NIR"])      # By name
dataset.x(selector, sources=[0])          # By index
dataset.x(selector, sources=["NIR", 1])   # Mixed
dataset.x(selector, sources="all")        # All sources (default)
```

#### Source Introspection

```python
dataset.source_info("NIR")  # or dataset.source_info(0)
# Returns: {"name": "NIR", "index": 0, "shape": (500, 3, 500), "type": "spectral", ...}

dataset.sources_compatible(layout="3d", concat=True)
# Returns: (False, "Processing count mismatch: NIR=3, markers=1")

dataset.num_sources  # 3
dataset.source_names  # ["NIR", "timeseries", "markers"]
```

### Error Messages

```
SourceConcatError: Cannot concatenate sources in 3D layout.

Sources have different processing counts:
  - Source 0 (NIR): 3 processings
  - Source 1 (timeseries): 1 processing
  - Source 2 (markers): 2 processings

To resolve, choose one:
  1. Use layout="2d" to flatten processings before concatenation
  2. Use concat_source=False to get sources as a list
  3. Use on_incompatible="flatten" to auto-flatten to 2D
  4. Use source_branch to process sources separately then merge

Example:
  X = dataset.x(selector, layout="2d", concat_source=True)  # Flatten first
  X = dataset.x(selector, layout="3d", concat_source=False)  # Get as list
```

---

## Source Branching & Merging Study

This section analyzes the requirements for source-aware branching and merging.

### Current Branching System

The existing `BranchController` supports:

```python
{"branch": [
    [SNV(), PCA(n_components=10)],      # Branch 0
    [MSC(), FirstDerivative()],         # Branch 1
]}
```

Branches execute **all sources** through each branch independently. There's no mechanism to route specific sources to specific branches.

### Proposed: Source Branching

Route different sources through different pipeline branches:

```python
{"source_branch": {
    "NIR": [SNV(), SavitzkyGolay(), PCA(n_components=50)],
    "timeseries": [StandardScaler()],
    "markers": [VarianceThreshold(threshold=0.01)],
}}
# OR by index:
{"source_branch": {
    0: [SNV(), SavitzkyGolay()],
    1: [StandardScaler()],
    2: [VarianceThreshold()],
}}
```

**Semantics**:
- Each source is isolated during branch execution
- Transformers in each branch only see their assigned source
- After branch completion, sources remain separate until explicitly merged

### Merging Strategies

After branching (source_branch or regular branch), results must be merged. The complexity depends on **what was applied in branches**:

#### Case 1: Transformers Only (Feature-Level Merge)

If branches contain only `TransformerMixin` operators (no models):

```python
pipeline = [
    {"source_branch": {
        "NIR": [SNV(), MinMaxScaler()],
        "markers": [VarianceThreshold()],
    }},
    {"merge_sources": "concat"},  # Merge transformed features
    {"model": PLSRegression()},   # Single model on merged features
]
```

**Merge options**:
- `concat`: Horizontal concatenation `(samples, Σ features)`
- `stack`: Stack as 3D `(samples, n_sources, max_features)` with padding
- `dict`: Return `{"NIR": X_nir, "markers": X_markers}` for multi-head models

#### Case 2: Models in Branches (Prediction-Level Merge / Late Fusion)

If branches contain models, merge operates on **predictions**:

```python
pipeline = [
    {"source_branch": {
        "NIR": [SNV(), {"model": PLSRegression(10)}],
        "markers": [{"model": RandomForestRegressor()}],
    }},
    {"merge_predictions": "average"},  # Late fusion
]
```

**Merge options**:
- `average`: Mean of predictions (regression)
- `weighted_average`: Weighted by validation score
- `vote`: Majority voting (classification)
- `stack`: Use predictions as features for meta-model
- `best`: Select best-performing branch

#### Case 3: Multiple Models in Branch

When a branch contains multiple sequential models:

```python
{"source_branch": {
    "NIR": [
        SNV(),
        {"model": PLSRegression(10)},      # Model 1
        {"model": RandomForestRegressor()}, # Model 2
    ],
}}
```

**Question**: Which model's predictions to use for merge?

**Options**:
1. **Last**: Use predictions from last model in branch (default)
2. **Best**: Use predictions from best-scoring model in branch
3. **All**: Ensemble all models within branch, then merge across branches
4. **Explicit**: User specifies which model via index or name

**Recommendation**: Default to **Last**, with `merge_model_selection: "best" | "all" | int` parameter.

### Merge Controller Design

```python
@register_controller
class MergeController(OperatorController):
    """
    Merge results from branched execution.

    Handles:
    - Feature merge (after transformer-only branches)
    - Prediction merge (after model branches)
    - Source merge (after source_branch)

    Keywords:
    - merge_sources: Merge multi-source features
    - merge_branches: Merge regular branch features/predictions
    - merge_predictions: Explicitly merge predictions for late fusion
    """

    priority = 5  # Same as BranchController

    @classmethod
    def matches(cls, step, operator, keyword):
        return keyword in ("merge_sources", "merge_branches", "merge_predictions")
```

### Complexity Analysis

| Scenario | Branch Type | Content | Merge Target | Complexity |
|----------|-------------|---------|--------------|------------|
| A | source_branch | Transformers only | Features | Low |
| B | source_branch | Single model per source | Predictions | Medium |
| C | source_branch | Multiple models per source | Predictions (which?) | High |
| D | branch | Transformers only | Features (per branch) | Low |
| E | branch | Models | Predictions | Medium |
| F | Mixed | source_branch + branch | Both | Very High |

**Recommendation**: Implement scenarios A and B first, document limitations for C and F.

### Pipeline Examples

#### Early Fusion (Feature Concatenation)

```python
pipeline = [
    # Per-source preprocessing
    {"source_branch": {
        "NIR": [SNV(), SavitzkyGolay()],
        "MIR": [MSC()],
    }},
    {"merge_sources": "concat"},  # → (samples, NIR_features + MIR_features)
    ShuffleSplit(n_splits=5),
    {"model": PLSRegression(10)},
]
```

#### Late Fusion (Prediction Averaging)

```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"source_branch": {
        "NIR": [SNV(), {"model": PLSRegression(10)}],
        "MIR": [MSC(), {"model": PLSRegression(15)}],
    }},
    {"merge_predictions": "weighted_average"},
]
```

#### Multi-Head Model (Separate Inputs)

```python
pipeline = [
    {"source_branch": {
        "NIR": [SNV(), MinMaxScaler()],
        "markers": [VarianceThreshold()],
    }},
    # No merge - multi-head model receives dict of sources
    {"model": MultiHeadNN(), "source_inputs": ["NIR", "markers"]},
]
```

---

## Implementation Roadmap

> **Note**: This roadmap is the single source of truth for Phase 9 implementation.
> See [dataset_config_roadmap.md](dataset_config_roadmap.md) for reference to this phase.

### Phase 9A: Validation & Error Handling (Foundation)

**Goal**: Clear errors when asymmetric sources cause concat failures.

#### Tasks

- [ ] Implement `sources_compatible(layout, concat)` method on `SpectroDataset`
- [ ] Add `on_incompatible` parameter to `FeatureAccessor.x()`: `"error" | "flatten" | "separate"`
- [ ] Create `SourceConcatError` exception with clear resolution suggestions
- [ ] Add validation in `x()` before attempting numpy concat
- [ ] Write unit tests for all failure scenarios

### Phase 9B: Selective Source Retrieval

**Goal**: Allow source selection by name or index.

#### Tasks

- [ ] Add `sources` parameter to `FeatureAccessor.x()`: accepts names, indices, or mixed
- [ ] Implement source name→index resolution via `source_names` property
- [ ] Add `source_info(name_or_index)` introspection method
- [ ] Update controllers to pass source selection through context
- [ ] Support `{"preprocessing": op, "sources": [...]}` pipeline syntax

### Phase 9C: Source-Aware Operators

**Goal**: Allow operators to target specific sources.

#### Tasks

- [ ] Parse `sources` key in step configuration
- [ ] Add source filtering in `TransformerMixinController.execute()`
- [ ] Track preprocessing history per source independently
- [ ] Update artifact naming to include source index
- [ ] Ensure predict mode correctly routes to source-specific artifacts

### Phase 9D: Source Branching

**Goal**: Implement `source_branch` keyword for per-source pipelines.

#### Tasks

- [ ] Create `SourceBranchController` extending pattern from `BranchController`
- [ ] Parse source_branch config (by name and by index)
- [ ] Execute branch steps with source isolation
- [ ] Store branch contexts per source
- [ ] Support predict mode reconstruction

### Phase 9E: Merge Controller

**Goal**: Implement merge strategies for branched execution.

#### Tasks

- [ ] Implement `MergeController` for `merge_sources` keyword
- [ ] Feature merge strategies: `concat`, `stack`, `dict`
- [ ] Add `merge_branches` for regular branch merging
- [ ] Add `merge_predictions` for prediction-level late fusion
- [ ] Prediction merge strategies: `average`, `weighted_average`, `vote`, `best`
- [ ] Handle "which model" selection for multi-model branches

### Phase 9F: Multi-Head Model Support

**Goal**: Enable models that accept multiple source inputs.

#### Tasks

- [ ] Define `MultiInputModel` protocol/interface
- [ ] Support `source_inputs` parameter in model configuration
- [ ] Create adapters for TensorFlow functional API multi-input
- [ ] Create adapters for PyTorch multi-input
- [ ] Integration with prediction storage

### Phase 9G: Documentation & Examples

**Goal**: Complete documentation and working examples.

#### Tasks

- [ ] Update dataset_config_specification.md with source metadata fields
- [ ] Add asymmetric sources example in examples/
- [ ] Document all error messages and resolutions
- [ ] Add troubleshooting guide for shape mismatches
- [ ] Update user guide with multi-source best practices
- [ ] Document merge controller strategies

### Dependencies

```
Phase 9A (Validation)
    ↓
Phase 9B (Selection) → Phase 9C (Operators)
    ↓                      ↓
Phase 9D (Source Branch) → Phase 9E (Merge)
                              ↓
                        Phase 9F (Multi-Head)
                              ↓
                        Phase 9G (Documentation)
```

### Testing Strategy

| Test Category | Coverage |
|---------------|----------|
| Unit: Validation | `sources_compatible()`, error messages |
| Unit: Selection | Source by name/index, mixed selection |
| Unit: Accessor | `on_incompatible` behaviors |
| Integration: Operators | Source-aware transformers |
| Integration: Branching | `source_branch` execution |
| Integration: Merging | Feature and prediction merge |
| E2E: Examples | Real asymmetric datasets |

---

## Appendix: Use Case Examples

### Use Case 1: Spectral + Genetic Markers (Early Fusion)

```python
# Dataset config (data only)
config = {
    "sources": [
        {"name": "NIR", "train_x": "nir.csv", "source_type": "spectral"},
        {"name": "SNPs", "train_x": "snps.csv", "source_type": "tabular"},
    ],
    "train_y": "phenotypes.csv",
}

# Pipeline (fusion strategy)
pipeline = [
    {"source_branch": {
        "NIR": [SNV(), MinMaxScaler()],
        "SNPs": [],  # No preprocessing
    }},
    {"merge_sources": "concat"},
    ShuffleSplit(n_splits=5),
    {"model": PLSRegression(10)},
]
```

### Use Case 2: Multi-Sensor Late Fusion

```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"source_branch": {
        0: [SNV(), {"model": PLSRegression(10)}],
        1: [MSC(), {"model": PLSRegression(15)}],
        2: [{"model": RandomForestRegressor()}],
    }},
    {"merge_predictions": "weighted_average"},
]
```

### Use Case 3: Multi-Head Neural Network

```python
pipeline = [
    {"source_branch": {
        "spectral": [SNV(), MinMaxScaler()],
        "tabular": [StandardScaler()],
    }},
    # No merge - model receives sources separately
    {
        "model": build_multihead_nn,  # Function that builds multi-input model
        "source_inputs": ["spectral", "tabular"],
    },
]
```
