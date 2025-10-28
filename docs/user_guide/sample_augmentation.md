# Sample Augmentation Quick Reference

## Quick Syntax

### Standard Mode
```yaml
sample_augmentation:
  transformers:
    - StandardScaler: {}
    - MinMaxScaler: {}
  count: 2  # Augmentations per sample
  selection: "random"  # or "all"
  random_state: 42
```

### Balanced Mode
```yaml
sample_augmentation:
  transformers:
    - StandardScaler: {}
  balance: "y"  # Balance on targets
  target_size: 100  # Samples per class
  max_factor: 3.0  # Max augmentation limit
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
| `selection` | str | "random" | "random" or "all" - how to assign transformers |
| `random_state` | int | None | Random seed for reproducibility |

## Common Patterns

### Pattern 1: Double Your Data
```yaml
sample_augmentation:
  transformers:
    - StandardScaler: {}
  count: 1
```
Result: n base + n augmented = 2n total

### Pattern 2: Balance Two-Class Dataset
```yaml
sample_augmentation:
  transformers:
    - StandardScaler: {}
    - MinMaxScaler: {}
  balance: "y"
  target_size: majority_class_size
```
Result: Equal class distribution

### Pattern 3: Conservative Augmentation
```yaml
sample_augmentation:
  transformers:
    - RobustScaler: {}
  count: 1
  max_factor: 2.0  # Prevent over-augmentation
```
Result: Controlled augmentation

### Pattern 4: Train-Only Augmentation
```yaml
# In context
context:
  partition: "train"

sample_augmentation:
  transformers:
    - StandardScaler: {}
  count: 2
```
Result: Only training samples augmented

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

## Recommended Transformers

| Transformer | Use Case | Pros | Cons |
|-------------|----------|------|------|
| StandardScaler | General purpose | Stable, well-tested | Assumes normality |
| MinMaxScaler | Bounded features | [0,1] range | Sensitive to outliers |
| RobustScaler | Noisy data | Outlier-resistant | Slower |
| MaxAbsScaler | Sparse data | Preserves sparsity | Less common |

## Integration with Pipeline

### Full Pipeline Example
```yaml
pipeline:
  # 1. Preprocessing
  - preprocessing:
      - SNV: {}
      - SavitzkyGolay: {window_length: 11}

  # 2. Sample Augmentation
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

## Dataset API Quick Reference

```python
# Augment samples manually
dataset.augment_samples(
    data=transformed_data,
    processings=["proc_name"],
    augmentation_id="unique_id",
    selector={"partition": "train"},  # Optional
    count=2  # or [2, 1, 0, 3, ...] per sample
)

# Get data with/without augmented
X_all = dataset.x({}, include_augmented=True)
X_base = dataset.x({}, include_augmented=False)

# Get augmented samples for origins
aug_indices = dataset._indexer.get_augmented_for_origins([0, 1, 2])

# Get origin for augmented sample
origin_idx = dataset._indexer.get_origin_for_sample(10)
```

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| "Processing 'X' not found" | Wrong processing name | Check transformer output names |
| High memory usage | Too many augmented samples | Reduce count or use max_factor |
| Poor performance | Over-augmentation | Reduce count, try different transformers |
| Inconsistent results | No random_state | Set random_state parameter |
| Validation too optimistic | Data leakage | Verify CV splits use include_augmented=False |

## Performance Tips

1. **Memory**: `Memory ≈ base_size + (count × feature_size)`
2. **Speed**: More transformers = longer pipeline
3. **Quality**: Test augmentation impact on validation set
4. **Balance**: Use balanced mode instead of manual balancing

## Links

- Full Guide: `docs/SAMPLE_AUGMENTATION.md`
- Examples: `examples/sample_augmentation_examples.py`
- Tests: `tests/unit/test_sample_augmentation_controller.py`
- Design: Design document (see project docs)
