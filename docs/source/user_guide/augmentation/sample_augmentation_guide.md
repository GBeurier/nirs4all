# Sample Augmentation Guide

Sample augmentation creates synthetic variations of your training samples to improve model robustness and generalization. This guide covers both quick reference and detailed usage.

## Overview

Sample augmentation is useful for:
- **Increasing dataset size** when training samples are limited
- **Balancing class distributions** in imbalanced datasets
- **Improving model generalization** through data diversity
- **Preventing overfitting** by introducing controlled variations

:::{important}
**Leak Prevention**: nirs4all automatically ensures augmented samples never appear in validation folds during cross-validation.
:::

## Quick Start

### Standard Mode (Count-Based)

```yaml
sample_augmentation:
  transformers:
    - StandardScaler: {}
    - MinMaxScaler: {}
  count: 2  # Augmentations per sample
  selection: "random"  # or "all"
  random_state: 42
```

### Balanced Mode (Class-Aware)

```yaml
sample_augmentation:
  transformers:
    - StandardScaler: {}
  balance: "y"  # Balance on targets
  target_size: 100  # Samples per class
  random_state: 42
```

## Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `transformers` | List | Required | List of sklearn transformers to apply |
| `count` | int | 1 | Number of augmentations per sample (standard mode) |
| `balance` | str | None | Column to balance on: "y" or metadata column (balanced mode) |
| `target_size` | int | None | Target samples per class (balanced mode) |
| `max_factor` | float | None | Maximum augmentation multiplier (balanced mode) |
| `ref_percentage` | float | None | Target as percentage of majority class |
| `selection` | str | "random" | "random" or "all" - how to assign transformers |
| `random_state` | int | None | Random seed for reproducibility |

## Usage Modes

### Standard Mode (Count-Based)

Creates a fixed number of augmented samples per base sample:

```yaml
sample_augmentation:
  transformers:
    - StandardScaler: {}
    - MinMaxScaler: {}
  count: 3  # 3 augmentations per sample
  selection: "random"  # or "all"
  random_state: 42
```

**Selection options:**
- `"random"`: Randomly assign transformers (default)
- `"all"`: Cycle through transformers systematically

**Example with 100 samples, count=3:**
- Original: 100 samples
- After augmentation: 400 samples (100 base + 300 augmented)

### Balanced Mode Strategies

Three balancing strategies are available. Choose ONE:

#### Strategy 1: Fixed Target Size

```yaml
sample_augmentation:
  transformers:
    - StandardScaler: {}
  balance: "y"
  target_size: 100  # Each class to exactly 100 samples
```

**Example:**
```
Initial: Class 0: 150, Class 1: 30, Class 2: 50
Result:  Class 0: 150 (unchanged), Class 1: 100 (+70), Class 2: 100 (+50)
```

#### Strategy 2: Multiplier Factor

```yaml
sample_augmentation:
  transformers:
    - StandardScaler: {}
  balance: "y"
  max_factor: 3.0  # Multiply each class by 3, capped at majority
```

**Example:**
```
Initial: Class 0: 100 (majority), Class 1: 20, Class 2: 50
Result:  Class 0: 100, Class 1: 60 (20×3), Class 2: 100 (50×3 capped)
```

#### Strategy 3: Reference Percentage

```yaml
sample_augmentation:
  transformers:
    - StandardScaler: {}
  balance: "y"
  ref_percentage: 0.8  # Target 80% of majority class
```

**Example:**
```
Initial: Class 0: 100 (majority), Class 1: 30, Class 2: 20
Result:  Class 0: 100, Class 1: 80, Class 2: 80
```

### Binning for Regression

When balancing continuous targets, nirs4all automatically bins values:

```yaml
sample_augmentation:
  transformers:
    - StandardScaler: {}
  balance: "y"
  bins: 10  # Create 10 virtual classes
  binning_strategy: "equal_width"  # or "quantile"
  max_factor: 2.0
```

**Binning strategies:**
- `"equal_width"`: Uniform bin spacing (default)
- `"quantile"`: Equal samples per bin

## Pipeline Integration

### Full Pipeline Example

```yaml
pipeline:
  # 1. Preprocessing
  - preprocessing:
      - SNV: {}
      - SavitzkyGolay: {window_length: 11}

  # 2. Sample Augmentation (before splitting)
  - sample_augmentation:
      transformers:
        - StandardScaler: {}
      count: 2
      random_state: 42

  # 3. Cross-Validation (leak-free)
  - split:
      - StratifiedKFold:
          n_splits: 5
          shuffle: true
          random_state: 42

  # 4. Model Training
  - model:
      - PLSRegression:
          n_components: 10
```

### Sequential Augmentation

Multiple augmentation rounds target only base samples:

```yaml
pipeline:
  - sample_augmentation:
      transformers:
        - StandardScaler: {}
      count: 1

  - sample_augmentation:
      transformers:
        - MinMaxScaler: {}
      count: 1
# Result: 100 base → 300 total (100 + 100 + 100)
```

## Recommended Transformers

| Transformer | Use Case | Pros | Cons |
|-------------|----------|------|------|
| `StandardScaler` | General purpose | Stable, well-tested | Assumes normality |
| `MinMaxScaler` | Bounded features | [0,1] range | Sensitive to outliers |
| `RobustScaler` | Noisy data | Outlier-resistant | Slower |
| `MaxAbsScaler` | Sparse data | Preserves sparsity | Less common |

## Best Practices

### ✅ DO

- Set `random_state` for reproducibility
- Start with count=1 or 2
- Use balanced mode for imbalanced data
- Test different transformers
- Monitor validation performance

### ❌ DON'T

- Over-augment (count > 5 usually unnecessary)
- Mix incompatible transformers
- Forget to validate results
- Augment test/validation data
- Ignore computation cost

### Augmentation Count Guidelines

| Dataset Size | Recommended Count |
|--------------|-------------------|
| Small (<100 samples) | 2-5 |
| Medium (100-1000) | 1-3 |
| Large (>1000) | 1-2 or balanced mode |

## Dataset API

### Python API

```python
# Augment samples manually
dataset.augment_samples(
    data=transformed_data,
    processings=["proc_name"],
    augmentation_id="unique_id",
    selector={"partition": "train"},  # Optional
    count=2
)

# Get data with/without augmented
X_all = dataset.x({}, include_augmented=True)
X_base = dataset.x({}, include_augmented=False)

# Get augmented samples for origins
aug_indices = dataset._indexer.get_augmented_for_origins([0, 1, 2])

# Get origin for augmented sample
origin_idx = dataset._indexer.get_origin_for_sample(10)
```

## How It Works

### Architecture

```
Base Samples (n samples)
    ↓
SampleAugmentationController
    ↓ (delegates to)
TransformerMixinController (applies transformations)
    ↓
Dataset.augment_samples() (stores with origin tracking)
    ↓
Augmented Dataset (n base + m augmented samples)
    ↓
Split Controller (uses only base samples for splitting)
    ↓
Training Folds (can access augmented samples)
Validation Folds (only base samples, leak-free!)
```

### Leak Prevention

The system prevents augmented samples from leaking into validation:

1. **Origin Tracking**: Every augmented sample stores its origin sample index
2. **Two-Phase Selection**: CV splits use `include_augmented=False`
3. **Metadata Inheritance**: Augmented samples inherit all metadata from origins

### Memory Formula

```
Memory ≈ base_samples_size + (augmentation_count × features_size)
```

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| "Processing 'X' not found" | Wrong processing name | Check transformer output names |
| High memory usage | Too many augmented samples | Reduce count or use max_factor |
| Poor performance | Over-augmentation | Reduce count, try different transformers |
| Inconsistent results | No random_state | Set random_state parameter |
| Validation too optimistic | Data leakage | Verify CV splits use include_augmented=False |

## See Also

- {doc}`augmentations` - Overview of augmentation methods
- {doc}`synthetic_nirs_generator` - Generate synthetic NIRS data
- {doc}`/reference/operator_catalog` - Complete operator reference
