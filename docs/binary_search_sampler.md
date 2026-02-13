# Binary Search Sampler for Optuna

## Overview

The `BinarySearchSampler` is a custom Optuna sampler optimized for **unimodal integer parameters** (parameters with a single peak and monotonic gradients on both sides).

**Primary use case**: PLS/PCR `n_components` optimization

**Efficiency**: Reduces optimization from ~30-50 trials (TPE) to ~10-15 trials (binary search)

## When to Use

### ✅ Best For:
- **PLS/PCR n_components** (most common)
- KNN n_neighbors
- Polynomial degree
- Any integer parameter with clear unimodal behavior

### ❌ Not Suitable For:
- Multi-modal parameters (multiple peaks)
- Continuous float parameters (use TPE or CMA-ES)
- Categorical parameters (use Grid)
- Non-monotonic parameters

## Basic Usage

```python
import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

pipeline = [
    ShuffleSplit(n_splits=3, random_state=42),
    {
        "model": PLSRegression(),
        "finetune_params": {
            "n_trials": 12,              # 10-15 trials sufficient
            "sampler": "binary",         # Binary search sampler
            "seed": 42,                  # Reproducibility
            "model_params": {
                "n_components": ('int', 1, 30),
            },
        }
    },
]

result = nirs4all.run(pipeline=pipeline, dataset="data/", verbose=1)
```

## Advanced Patterns

### Multi-Phase Optimization (Recommended)

Combine binary search for coarse optimization with TPE for fine-tuning:

```python
finetune_params = {
    "seed": 42,
    "phases": [
        {"n_trials": 8, "sampler": "binary"},  # Coarse search
        {"n_trials": 5, "sampler": "tpe"},     # Fine-tuning
    ],
    "model_params": {
        "n_components": ('int', 1, 30),
    }
}
```

### Grouped Approach with Preprocessing

Binary search per preprocessing variant:

```python
pipeline = [
    {"feature_augmentation": [SNV, MSC, Detrend], "action": "extend"},
    ShuffleSplit(n_splits=3),
    {
        "model": PLSRegression(),
        "finetune_params": {
            "n_trials": 12,
            "sampler": "binary",
            "approach": "grouped",  # Binary search per preprocessing
            "model_params": {
                "n_components": ('int', 1, 30),
            }
        }
    }
]
```

### Mixed Parameters

Binary search handles integers; other types fall back to random:

```python
finetune_params = {
    "n_trials": 12,
    "sampler": "binary",
    "model_params": {
        "n_components": ('int', 1, 30),  # Binary search
        "scale": [True, False],          # Categorical (exhaustive)
    }
}
```

## Algorithm Details

### Strategy

1. **Initial Phase** (3 trials):
   - Test: low, high, midpoint

2. **Binary Search Phase**:
   - Find best value from completed trials
   - Narrow boundaries based on neighbors
   - Test midpoint of larger unexplored region

3. **Refinement Phase**:
   - Test immediate neighbors of best value
   - Fine-tune around the optimum

### Example Search Path

For `n_components` in range [1, 30]:

```
Trial 1: Test 1 (low)       → RMSE: 15.2
Trial 2: Test 30 (high)     → RMSE: 18.5
Trial 3: Test 15 (mid)      → RMSE: 12.3  ← Best so far
Trial 4: Test 7             → RMSE: 13.1
Trial 5: Test 22            → RMSE: 14.8
Trial 6: Test 11            → RMSE: 11.9  ← New best
Trial 7: Test 10            → RMSE: 12.0
Trial 8: Test 12            → RMSE: 12.1
Trial 9: Test 9             → RMSE: 11.8  ← Final best
...
```

Result: Found optimal n_components = 9 in ~10 trials (vs ~30-50 for TPE)

## Performance Guidelines

| Range Size | Recommended Trials |
|------------|-------------------|
| 1-30       | 10-15             |
| 1-50       | 12-18             |
| 1-100      | 15-20             |

**Rule of thumb**: `trials ≈ log₂(range_size) + 5`

## Integration with Your Code

Your existing code in `full_run.py:136` already uses the binary sampler:

```python
{
    "model": PLSRegression(scale=False),
    "name": "PLS",
    "finetune_params": {
        "n_trials": CFG["pls_finetune_trials"],  # 25 in full mode
        "sampler": "binary",                      # ✓ Binary search
        "n_jobs": 10,                             # Parallel trials
        "model_params": {
            "n_components": ('int', 1, 25),
        },
    },
}
```

## Examples

- **Basic**: `examples/developer/06_internals/D05_binary_search_sampler.py`
- **Tests**: `tests/unit/optimization/test_binary_search_sampler.py`

## API Reference

### BinarySearchSampler

```python
from nirs4all.optimization.optuna import BinarySearchSampler

sampler = BinarySearchSampler(seed=42)
```

**Parameters**:
- `seed` (int, optional): Random seed for reproducibility

**Methods**:
- `sample_independent(study, trial, param_name, param_distribution)`: Sample parameter value
- `infer_relative_search_space(study, trial)`: Not used (returns empty dict)
- `sample_relative(study, trial, search_space)`: Not used (returns empty dict)

## Comparison with Other Samplers

| Sampler | Best For | Trials Needed | Speed |
|---------|----------|---------------|-------|
| **binary** | Unimodal integers (PLS n_components) | 10-15 | ⚡⚡⚡ |
| **tpe** | General-purpose, mixed parameters | 30-50 | ⚡⚡ |
| **grid** | Small categorical spaces | All combos | ⚡ |
| **random** | Quick baseline | 20-30 | ⚡⚡⚡ |
| **cmaes** | Continuous parameters | 50-100 | ⚡ |

## References

- Optuna documentation: https://optuna.readthedocs.io/
- Binary search algorithm: https://en.wikipedia.org/wiki/Binary_search_algorithm
- nirs4all Optuna integration: `nirs4all/optimization/optuna.py`
