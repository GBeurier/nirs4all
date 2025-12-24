# Disjoint Sample Branch Merging Specification

**Status**: Draft
**Date**: 2024-12-24
**Author**: Design Discussion

---

## 1. Overview

This specification defines the behavior for merging branches that partition samples into **disjoint subsets**. Unlike copy branches (where all branches see all samples), disjoint branches create non-overlapping sample sets, requiring special merge semantics.

### Scope

Applies to any branch that partitions samples:
- `metadata_partitioner` (by metadata column value: site, variety, instrument, etc.)
- `sample_partitioner` (by outlier status)
- Any future partitioner producing non-overlapping sample sets

**Key property**: Each sample exists in exactly ONE branch.

---

## 2. Syntax

### 2.1 Metadata Partitioner

```python
# Branch by metadata column with per-branch CV and models
{
    "branch": [PLS(5), RF(100), XGB()],
    "by": "metadata_partitioner",
    "column": "site",
    "cv": ShuffleSplit(n_splits=3),
    "min_samples": 20,  # Optional: skip branches with < 20 samples
    "group_values": {   # Optional: group rare values
        "others": ["C", "D", "E"],
    },
}
```

### 2.2 Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `branch` | `list` | Yes | - | Models/steps to run in each branch |
| `by` | `str` | Yes | - | Partitioner type: `"metadata_partitioner"` or `"sample_partitioner"` |
| `column` | `str` | Yes* | - | Metadata column name (*required for metadata_partitioner) |
| `cv` | Splitter | No | `None` | Per-branch cross-validation strategy |
| `min_samples` | `int` | No | `1` | Minimum samples per branch; branches with fewer are skipped |
| `group_values` | `dict` | No | `None` | Map of branch_name → list of values to group together |

### 2.3 Merge Syntax

```python
# Default: auto-detect N columns from branches
{"merge": "predictions"}

# Override: force specific column count
{"merge": "predictions", "n_columns": 2}

# Override: selection criterion for top-N
{"merge": "predictions", "select_by": "mse"}   # default: lowest MSE
{"merge": "predictions", "select_by": "r2"}    # highest R²
{"merge": "predictions", "select_by": "mae"}   # lowest MAE
{"merge": "predictions", "select_by": "order"} # first N in definition order
```

---

## 3. Feature Merge Behavior

### 3.1 Rule

```
IF all branches produce the SAME feature dimension:
    → Concatenate rows by sample_id
    → Result: (n_total_samples, n_features)

ELSE (asymmetric feature dimensions):
    → ERROR
```

### 3.2 Valid Example

```python
pipeline = [
    {
        "branch": [SNV(), PCA(n_components=20)],
        "by": "metadata_partitioner",
        "column": "site",
        "cv": ShuffleSplit(n_splits=3),
    },
    {"merge": "features"},  # OK: all branches → 20 features
    PLSRegression(n_components=10),
]
```

```
Branch "site_A" (300 samples): SNV → PCA(20) → 300 × 20
Branch "site_B" (120 samples): SNV → PCA(20) → 120 × 20
Branch "site_C" (30 samples):  SNV → PCA(20) → 30 × 20

Merge: 450 × 20 ✓
```

### 3.3 Invalid Example

```
Branch "site_A": SNV → PCA(20) → 300 × 20
Branch "site_B": SNV → PCA(10) → 120 × 10

Merge: ERROR ✗
  "Cannot merge features from disjoint branches with different feature
   dimensions: {'site_A': 20, 'site_B': 10}. Ensure all branches apply
   identical transformations."
```

---

## 4. Prediction Merge Behavior

### 4.1 Algorithm

```python
def merge_disjoint_predictions(branches, n_columns=None, select_by="mse"):
    """
    Merge OOF predictions from disjoint sample branches.

    Args:
        branches: List of branch results with predictions
        n_columns: Force output column count (None = auto)
        select_by: Criterion for selecting top N models

    Returns:
        Merged OOF matrix (n_total_samples, N)
    """
    # Step 1: Determine N (output column count)
    model_counts = [len(branch.predictions) for branch in branches]

    if n_columns is not None:
        N = n_columns
        if N > min(model_counts):
            raise ValueError(
                f"n_columns={N} exceeds minimum model count ({min(model_counts)}) "
                f"across branches. Model counts: {dict(zip(branch_names, model_counts))}"
            )
    else:
        N = min(model_counts)

    # Step 2: Select top N models per branch
    selected_per_branch = {}
    for branch in branches:
        if len(branch.predictions) == N:
            selected = branch.predictions
        else:
            scored = [(model, get_score(model, select_by)) for model in branch.predictions]
            # Sort: ascending for mse/mae, descending for r2
            scored.sort(key=lambda x: x[1], reverse=(select_by == "r2"))
            selected = [model for model, score in scored[:N]]

        selected_per_branch[branch.name] = selected

    # Step 3: Validate no data leakage
    validate_no_leakage(selected_per_branch)

    # Step 4: Validate trainability
    validate_trainable(merged)

    # Step 5: Reconstruct OOF matrix by sample_id
    merged = np.zeros((n_total_samples, N))
    for branch in branches:
        sample_ids = branch.sample_ids
        for col_idx, model in enumerate(selected_per_branch[branch.name]):
            merged[sample_ids, col_idx] = model.oof_predictions

    return merged, metadata
```

### 4.2 Example

```
Branch "red" (50 samples):
    Models: PLS (mse=0.15), RF (mse=0.12), XGB (mse=0.18)
    OOF shape: 50 × 3

Branch "blue" (100 samples):
    Models: PLS (mse=0.20), RF (mse=0.16)
    OOF shape: 100 × 2

N = min(3, 2) = 2

Branch "red": Top 2 by MSE → RF (0.12), PLS (0.15)
Branch "blue": All 2 models → PLS (0.20), RF (0.16)

Merged OOF: 150 × 2
    Column 0: RF (red samples), PLS (blue samples)
    Column 1: PLS (red samples), RF (blue samples)
```

**Note**: Columns represent DIFFERENT models for different samples. The downstream meta-model learns sample-specific weights.

---

## 5. Validation

### 5.1 Leakage Validation

```python
def validate_no_leakage(selected_per_branch):
    """
    Ensure no sample was used to train a model that predicts on it.

    For disjoint branches with per-branch CV, this is guaranteed:
    - Each branch has its own folds
    - OOF predictions are always from held-out folds
    - Samples never cross branches
    """
    for branch_name, models in selected_per_branch.items():
        for model in models:
            train_samples = set(model.train_sample_ids)
            pred_samples = set(model.oof_sample_ids)

            overlap = train_samples & pred_samples
            if overlap:
                raise ValueError(
                    f"Data leakage detected in branch '{branch_name}', "
                    f"model '{model.name}': {len(overlap)} samples used in both "
                    f"training and OOF prediction."
                )
```

### 5.2 Trainability Validation

```python
def validate_trainable(merged, sample_ids):
    """Ensure merged predictions can train a meta-model."""
    # Check for NaN/Inf
    if np.any(~np.isfinite(merged)):
        nan_count = np.sum(~np.isfinite(merged))
        raise ValueError(
            f"Merged predictions contain {nan_count} non-finite values. "
            f"Cannot train meta-model on invalid data."
        )

    # Check minimum samples
    MIN_SAMPLES = 10
    if len(sample_ids) < MIN_SAMPLES:
        raise ValueError(
            f"Merged predictions have only {len(sample_ids)} samples. "
            f"Minimum {MIN_SAMPLES} required for meta-model training."
        )
```

---

## 6. Metadata Output

```python
merged_metadata = {
    "merge_type": "disjoint_samples",
    "n_columns": 2,
    "select_by": "mse",
    "branches": {
        "red": {
            "n_samples": 50,
            "sample_ids": [0, 1, 2, ...],
            "n_models_original": 3,
            "n_models_selected": 2,
            "selected_models": [
                {"name": "RF", "score": 0.12, "column": 0},
                {"name": "PLS", "score": 0.15, "column": 1},
            ],
            "dropped_models": [
                {"name": "XGB", "score": 0.18},
            ],
        },
        "blue": {
            "n_samples": 100,
            "sample_ids": [50, 51, ...],
            "n_models_original": 2,
            "n_models_selected": 2,
            "selected_models": [
                {"name": "PLS", "score": 0.16, "column": 0},
                {"name": "RF", "score": 0.20, "column": 1},
            ],
            "dropped_models": [],
        },
    },
    "column_mapping": {
        0: {"red": "RF", "blue": "PLS"},
        1: {"red": "PLS", "blue": "RF"},
    },
}
```

---

## 7. Logging

### Info Level
```
Merging 2 disjoint branches: 'red' (50 samples), 'blue' (100 samples)
Output: 150 samples × 2 columns
```

### Warning Level
```
Model count differs across branches. Using N=2 columns (minimum).
  Branch 'red': selected RF, PLS; dropped XGB
  Branch 'blue': all models selected
Column mapping is heterogeneous:
  Column 0: RF (red), PLS (blue)
  Column 1: PLS (red), RF (blue)
```

---

## 8. Error Conditions

| Condition | Error Message |
|-----------|---------------|
| Feature dimension mismatch | `"Cannot merge features from disjoint branches with different feature dimensions: {'red': 20, 'blue': 10}"` |
| n_columns exceeds minimum | `"n_columns=5 exceeds minimum model count (2) across branches"` |
| Branch has 0 models | `"Branch 'X' has no predictions to merge"` |
| Data leakage detected | `"Data leakage detected in branch 'X': N samples in both train and OOF"` |
| Non-finite values in merged | `"Merged predictions contain N non-finite values"` |
| Insufficient samples | `"Merged predictions have only N samples. Minimum 10 required"` |

---

## 9. Complete Examples

### 9.1 Basic Metadata Partition with Stacking

```python
pipeline = [
    MinMaxScaler(),
    {
        "branch": [PLS(5), RF(100), XGB()],
        "by": "metadata_partitioner",
        "column": "site",
        "cv": ShuffleSplit(n_splits=3),
    },
    {"merge": "predictions"},
    Ridge(),
]
```

### 9.2 Force Column Count with R² Selection

```python
pipeline = [
    MinMaxScaler(),
    {
        "branch": [PLS(5), RF(100), XGB(), LightGBM()],
        "by": "metadata_partitioner",
        "column": "site",
        "cv": ShuffleSplit(n_splits=3),
        "min_samples": 30,
    },
    {"merge": "predictions", "n_columns": 2, "select_by": "r2"},
    Ridge(),
]
```

### 9.3 Group Rare Values

```python
pipeline = [
    MinMaxScaler(),
    {
        "branch": [PLS(10)],
        "by": "metadata_partitioner",
        "column": "percentage",
        "cv": ShuffleSplit(n_splits=3),
        "group_values": {
            "zero": [0],
            "low": [5, 10, 15],
            "high": [20, 25, 30, 35, 40],
        },
    },
    {"merge": "predictions"},
]
# Creates 3 branches: "zero", "low", "high"
```

### 9.4 Feature Merge (Symmetric Transforms)

```python
pipeline = [
    {
        "branch": [SNV(), PCA(n_components=20)],
        "by": "metadata_partitioner",
        "column": "site",
        "cv": ShuffleSplit(n_splits=3),
    },
    {"merge": "features"},
    PLSRegression(n_components=10),
]
```

---

## 10. Summary Table

| Merge Type | Symmetric Case | Asymmetric Case |
|------------|----------------|-----------------|
| **Features** | Concatenate rows → (n_total, n_features) | ERROR |
| **Predictions** | Concatenate rows → (n_total, M) | Select top N per branch → (n_total, N) |

---

## 11. Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)

- [ ] **P1.1** Add `MetadataPartitionerController`
  - Parse syntax: `{"branch": [...], "by": "metadata_partitioner", "column": ...}`
  - Extract unique values from metadata column
  - Create per-branch sample sets
  - Handle `group_values` parameter
  - Handle `min_samples` parameter (skip small branches)

- [ ] **P1.2** Implement per-branch CV
  - Apply `cv` parameter within each branch
  - Generate independent folds per partition
  - Store fold info in branch context

- [ ] **P1.3** Add `is_disjoint_branch()` detection
  - Check `context.custom.get("sample_partition")`
  - Check `context.custom.get("metadata_partition")`
  - Add `BranchType.METADATA_PARTITIONER` enum

### Phase 2: Merge Logic (Week 2-3)

- [ ] **P2.1** Update MergeController for disjoint detection
  - Detect disjoint branches before merge
  - Route to appropriate merge strategy

- [ ] **P2.2** Implement disjoint feature merge
  - Validate feature dimension equality
  - Concatenate by sample_id
  - Generate clear error for dimension mismatch

- [ ] **P2.3** Implement disjoint prediction merge
  - Parse `n_columns` and `select_by` parameters
  - Implement top-N selection logic
  - Implement score-based model ranking
  - Reconstruct OOF by sample_id

- [ ] **P2.4** Implement validation checks
  - Leakage validation
  - Trainability validation (non-finite, min samples)

### Phase 3: Metadata & Logging (Week 3)

- [x] **P3.1** Generate merge metadata
  - Column mapping
  - Selected/dropped models per branch
  - Branch statistics
  - Added `DisjointBranchInfo` and `DisjointMergeMetadata` dataclasses

- [x] **P3.2** Implement logging
  - Info: basic merge summary via `log_summary()`
  - Warning: heterogeneous columns, dropped models via `log_warnings()`

### Phase 4: Prediction Mode (Week 4)

- [x] **P4.1** Implement prediction routing
  - Load metadata value for new samples
  - Route to correct branch model
  - Handle missing metadata gracefully

- [x] **P4.2** Bundle support
  - Export per-branch models
  - Store metadata column info for routing

### Phase 5: Testing & Documentation (Week 4-5)

- [x] **P5.1** Unit tests
  - Symmetric feature merge
  - Asymmetric feature merge (error case)
  - Prediction merge with equal model counts
  - Prediction merge with unequal model counts
  - All selection criteria (mse, r2, mae, order)
  - n_columns override
  - Leakage detection
  - Trainability validation
  - Edge cases (single branch, empty branch, etc.)

- [x] **P5.2** Integration tests
  - Full pipeline with metadata_partitioner
  - Stacking after disjoint merge
  - Prediction mode with routing

- [x] **P5.3** Example script
  - Add `Q35_metadata_branching.py` to examples/

- [x] **P5.4** Documentation
  - Update merge_syntax.md
  - Add to pipeline_syntax.md
  - Update user guide

### Dependencies

```
P1.1 ──┬── P1.2 ── P2.1 ──┬── P2.2
       │                   │
       └── P1.3 ───────────┼── P2.3 ── P2.4 ── P3.1 ── P3.2
                           │
                           └── P4.1 ── P4.2

P5.* depends on P1-P4 completion
```

### Estimated Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1 | 1.5 weeks | 1.5 weeks |
| Phase 2 | 1.5 weeks | 3 weeks |
| Phase 3 | 0.5 week | 3.5 weeks |
| Phase 4 | 1 week | 4.5 weeks |
| Phase 5 | 1 week | 5.5 weeks |

---

## Appendix A: Visual Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    DISJOINT SAMPLE MERGE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Branch A (50 samples)          Branch B (100 samples)          │
│  ┌─────────────────┐            ┌─────────────────┐             │
│  │ Model1: 50×1    │            │ Model1: 100×1   │             │
│  │ Model2: 50×1    │            │ Model2: 100×1   │             │
│  │ Model3: 50×1    │            └─────────────────┘             │
│  └─────────────────┘                   ↓                        │
│         ↓                         N = min(3,2) = 2              │
│    Top 2 by score                      ↓                        │
│         ↓                              │                        │
│  ┌─────────────────┐            ┌─────────────────┐             │
│  │ Best1:  50×1    │            │ Model1: 100×1   │             │
│  │ Best2:  50×1    │            │ Model2: 100×1   │             │
│  └─────────────────┘            └─────────────────┘             │
│         │                              │                        │
│         └──────────┬───────────────────┘                        │
│                    ↓                                            │
│           ┌─────────────────┐                                   │
│           │ Merged: 150 × 2 │                                   │
│           │ (by sample_id)  │                                   │
│           └─────────────────┘                                   │
│                    ↓                                            │
│              Meta-model                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix B: Comparison with Copy Branches

| Aspect | Copy Branches | Disjoint Branches |
|--------|---------------|-------------------|
| Samples | All branches see all samples | Each sample in exactly one branch |
| OOF shape | (n_samples, n_models) per branch | (n_branch_samples, n_models) per branch |
| Feature merge | Horizontal concat (n, f1+f2+...) | Row concat (n1+n2+..., f) if symmetric |
| Prediction merge | Stack for meta-model | Reconstruct by sample_id |
| Cross-validation | Global folds, shared | Per-branch folds, independent |
| Use case | Compare preprocessing variants | Per-site/variety/instrument models |
