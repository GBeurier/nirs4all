# Using Generators for Hyperparameter Exploration

## Overview

Generators allow you to explore multiple pipeline variants without writing repetitive code. Instead of manually creating dozens of similar pipelines, you use special keywords to express the search space compactly.

nirs4all supports **8 generator keywords** for different exploration patterns:

| Generator | Purpose | Example Count |
|-----------|---------|---------------|
| `_or_` | Simple alternatives | 3 options → 3 variants |
| `_range_` | Linear numeric sweep | [1, 10] → 10 variants |
| `_log_range_` | Logarithmic sweep | [0.001, 1, 4] → 4 variants |
| `_grid_` | Full grid search (Cartesian product) | 3×2 → 6 variants |
| `_cartesian_` | Pipeline stage combinations | [A,B] × [X,Y] → 4 pipelines |
| `_zip_` | Parallel iteration (paired) | 3 pairs → 3 variants |
| `_chain_` | Sequential ordered choices | 5 configs → 5 variants |
| `_sample_` | Random sampling | 100 space → 20 samples |

**When to use generators:**
- Exploring preprocessing combinations (SNV vs MSC vs Detrend)
- Hyperparameter optimization (n_components, learning rates, regularization)
- Ablation studies (progressive feature addition)
- Random search in large spaces

---

## Basic Generators

### `_or_` - Simple Alternatives

**Use case**: Try different preprocessing methods, models, or discrete options.

```python
import nirs4all
from nirs4all.operators.transforms import SNV, MSC, Detrend
from sklearn.cross_decomposition import PLSRegression

# Instead of manually creating 3 pipelines...
pipeline = [
    {"_or_": [SNV(), MSC(), Detrend()]},  # Try each option
    PLSRegression(n_components=10)
]

result = nirs4all.run(pipeline, dataset="sample_data/regression")
print(f"Tested {result.num_predictions} variants")
# → 3 variants: SNV, MSC, Detrend
```

**Key features:**
- Each item in the list becomes a separate variant
- Can use classes, instances, or dicts
- Order doesn't matter (may be shuffled if using `count`)

### `_range_` - Parameter Sweep

**Use case**: Sweep numeric parameters linearly (n_components, window sizes, polynomial orders).

```python
# Linear sweep from 5 to 20 with step 5
pipeline = [
    SNV(),
    {"n_components": {"_range_": [5, 20, 5]}, "model": PLSRegression}
]
# → 4 variants: n_components = 5, 10, 15, 20

# Alternative dict syntax
pipeline = [
    SNV(),
    {"n_components": {"_range_": {"from": 5, "to": 20, "step": 5}},
     "model": PLSRegression}
]
```

**Key features:**
- Inclusive range: `[1, 5]` → `[1, 2, 3, 4, 5]`
- Default step is 1
- Use for: n_components, window sizes, polynomial orders

---

## Advanced Generators

### `_log_range_` - Logarithmic Sweep

**Use case**: Sweep parameters spanning multiple orders of magnitude (learning rates, regularization, decay).

```python
from sklearn.linear_model import Ridge

# Regularization search over 4 orders of magnitude
pipeline = [
    SNV(),
    {"alpha": {"_log_range_": [0.0001, 10, 8]}, "model": Ridge}
]
# → 8 variants: alpha ≈ 0.0001, 0.001, 0.01, 0.1, 1, 10

# Learning rate search (base 10, default)
pipeline = [
    {"learning_rate": {"_log_range_": [1e-5, 1e-1, 9]}}
]
# → 9 variants logarithmically spaced from 0.00001 to 0.1
```

**Why log scale?**
- Linear `_range_`: `[0.0001, 10, 8]` → [0.0001, 1.43, 2.86, ...] (poor coverage at small values)
- Log `_log_range_`: `[0.0001, 10, 8]` → [0.0001, 0.001, 0.01, 0.1, 1, 10] (even coverage)

**Key features:**
- Default base is 10 (can customize: `{"base": 2}`)
- Use for: learning rates, regularization (alpha, lambda), decay factors

### `_cartesian_` - Pipeline Stage Combinations

**Use case**: Generate all combinations of sequential preprocessing stages, then optionally select a subset.

```python
from nirs4all.operators.transforms import SNV, MSC, EMSC, FirstDerivative, Detrend

# Generate all 3×2×2 = 12 complete preprocessing pipelines
pipeline = [
    {"_cartesian_": [
        {"_or_": [SNV(), MSC(), EMSC()]},           # Stage 1: Scatter correction
        {"_or_": [FirstDerivative(), None]},        # Stage 2: Derivative
        {"_or_": [Detrend(order=1), None]}          # Stage 3: Detrending
    ]},
    PLSRegression(n_components=10)
]
# → 12 variants: all combinations of [stage1, stage2, stage3]

# Select subset: pick 5 random complete pipelines
pipeline = [
    {"_cartesian_": [
        {"_or_": [SNV(), MSC(), None]},
        {"_or_": [FirstDerivative(), None]},
    ], "pick": 1, "count": 5},  # Pick 1 pipeline per selection, 5 total
    PLSRegression(n_components=10)
]
# → 5 randomly selected complete preprocessing pipelines
```

**Difference from `_grid_`:**
- `_cartesian_`: Produces **lists** (pipeline stages) → `[SNV(), Deriv1()]`
- `_grid_`: Produces **dicts** (parameters) → `{"alpha": 0.1, "beta": 2}`

**Key features:**
- First generates all stage combinations
- Then applies `pick` or `arrange` to select subset
- Ideal for preprocessing pipeline exploration

### `_grid_` - Full Grid Search

**Use case**: Exhaustive search over all parameter combinations (like sklearn's `ParameterGrid`).

```python
# 3×2 = 6 parameter combinations
pipeline = [
    {"_grid_": {
        "n_components": [5, 10, 15],
        "scale": [True, False]
    }, "model": PLSRegression}
]
# → 6 variants: all (n_components, scale) pairs

# Nested generators in grid
pipeline = [
    {"_grid_": {
        "n_components": {"_range_": [5, 20, 5]},  # [5, 10, 15, 20]
        "preprocessing": ["SNV", "MSC", None]
    }, "model": PLSRegression}
]
# → 4 × 3 = 12 variants
```

**Caution**: Exponential explosion!
- 3 params with 5 values each → 5³ = 125 variants
- Use `count` to limit: `{"_grid_": {...}, "count": 50}`

### `_zip_` - Parallel Iteration

**Use case**: Parameters that should vary **together**, not independently. Avoids combinatorial explosion.

```python
from sklearn.linear_model import ElasticNet

# Paired hyperparameters (3 pairs, not 3×3=9 combinations)
pipeline = [
    {"_zip_": {
        "alpha": [0.1, 1.0, 10.0],
        "l1_ratio": [0.2, 0.5, 0.8]  # Paired with alpha by position
    }, "model": ElasticNet}
]
# → 3 variants: (0.1, 0.2), (1.0, 0.5), (10.0, 0.8)
```

**Comparison with `_grid_`:**

```python
# _zip_: 3 paired configurations
{"_zip_": {"x": [1, 2, 3], "y": ["A", "B", "C"]}}
# → [{"x": 1, "y": "A"}, {"x": 2, "y": "B"}, {"x": 3, "y": "C"}]

# _grid_: 3×3 = 9 all-combinations
{"_grid_": {"x": [1, 2, 3], "y": ["A", "B", "C"]}}
# → [{"x": 1, "y": "A"}, {"x": 1, "y": "B"}, ..., {"x": 3, "y": "C"}]
```

**Key features:**
- Pairs values at same index (like Python's `zip()`)
- Shortest list determines number of variants
- Use when parameters have semantic relationship

### `_chain_` - Sequential Choices

**Use case**: Preserve exact order of configurations. Ideal for ablation studies or progressive experiments.

```python
# Progressive complexity study (order matters!)
pipeline = [
    {"_chain_": [
        {},                                         # Baseline: no preprocessing
        {"transform": SNV()},                       # Add SNV
        {"transform": SNV(), "derivative": 1},      # Add derivative
        {"transform": SNV(), "derivative": 2}       # Add 2nd derivative
    ]},
    PLSRegression(n_components=10)
]
# → 4 variants in exact order (not randomized)
```

**vs `_or_`:**
- `_or_`: May shuffle order if using `count`
- `_chain_`: **Always** preserves order

**Key features:**
- Guaranteed sequential order
- Useful for ablation studies
- Readable progressive experiments

### `_sample_` - Random Sampling

**Use case**: Random search in large spaces. More efficient than grid search for high-dimensional problems.

```python
from sklearn.linear_model import Ridge

# Uniform sampling: 20 random values between 0.1 and 10
pipeline = [
    {"alpha": {"_sample_": {
        "distribution": "uniform",
        "from": 0.1,
        "to": 10.0,
        "num": 20
    }}, "model": Ridge}
]
# → 20 variants with random alpha values

# Log-uniform sampling (better for learning rates)
pipeline = [
    {"learning_rate": {"_sample_": {
        "distribution": "log_uniform",
        "from": 1e-5,
        "to": 1e-1,
        "num": 20
    }}}
]
# → 20 variants with log-uniformly distributed learning rates

# Random choice from categorical options
pipeline = [
    {"preprocessing": {"_sample_": {
        "distribution": "choice",
        "values": ["SNV", "MSC", "Detrend", "EMSC"],
        "num": 10
    }}}
]
# → 10 random selections from 4 options (with replacement)
```

**Distributions:**
- `uniform`: Even probability across range
- `log_uniform`: Even probability on log scale (for learning rates)
- `normal`/`gaussian`: Normal distribution
- `choice`: Random selection from list

**Reproducibility**: Use `_seed_` modifier for deterministic results:
```python
{"_sample_": {...}, "_seed_": 42}  # Same results every time
```

---

## Combining Generators

### Multiple Generators = Cartesian Product

```python
# 3 preprocessing × 4 n_components = 12 variants
pipeline = [
    {"_or_": [SNV(), MSC(), Detrend()]},           # 3 options
    {"n_components": {"_range_": [5, 20, 5]},      # 4 values
     "model": PLSRegression}
]
# → 12 total variants
```

### Nested Generators

```python
# Grid with nested _or_
pipeline = [
    {"_grid_": {
        "preprocessing": {"_or_": [SNV(), MSC()]},  # Expanded first
        "n_components": [5, 10, 15]
    }, "model": PLSRegression}
]
# → 2 × 3 = 6 variants
```

### Complex Pipeline with Multiple Generator Levels

```python
# Realistic hyperparameter search
pipeline = [
    # Preprocessing: 3 options
    {"_or_": [SNV(), MSC(), None]},

    # Smoothing: 2 options
    {"_or_": [
        {"window": 11, "polyorder": 2, "deriv": 0},  # Savitzky-Golay
        None  # No smoothing
    ]},

    # Model grid: 3×4 = 12 combinations
    {"_grid_": {
        "n_components": {"_range_": [5, 20, 5]},
        "scale": [True, False]
    }, "model": PLSRegression}
]
# → 3 preprocessing × 2 smoothing × 12 model configs = 72 variants
```

---

## Performance Considerations

### Combinatorial Explosion Warning

```python
# Bad: 10 × 10 × 10 × 10 = 10,000 variants!
{"_grid_": {
    "param1": {"_range_": [1, 10]},
    "param2": {"_range_": [1, 10]},
    "param3": {"_range_": [1, 10]},
    "param4": {"_range_": [1, 10]}
}}
```

**Solutions:**
1. **Use `count` to limit**: `{"_grid_": {...}, "count": 100}`
2. **Use `_sample_` instead**: Random search 100 points
3. **Use `_zip_` for correlated parameters**
4. **Progressive refinement**: Coarse grid → fine grid around best

### Memory Management with Cache

Generator expansion can create many variants. Use cache configuration to optimize memory:

```python
from nirs4all.config.cache_config import CacheConfig

result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset,
    cache=CacheConfig(
        step_cache_enabled=True,      # Reuse preprocessing across variants
        use_cow_snapshots=True,        # Efficient branch copies
        step_cache_max_mb=2048,        # 2 GB cache budget
    ),
)
```

See [Cache Optimization Guide](../advanced/cache_optimization.md) for details.

### Counting Variants Before Execution

```python
from nirs4all.pipeline.config.generator import count_combinations

spec = {
    "_grid_": {
        "n_components": {"_range_": [5, 20, 5]},
        "alpha": {"_log_range_": [0.01, 1, 5]}
    }
}

count = count_combinations(spec)
print(f"This will generate {count} variants")  # → 20
```

---

## Best Practices

### 1. Start Small, Scale Up

```python
# Phase 1: Quick exploration (few variants)
pipeline = [
    {"_or_": [SNV(), MSC()]},  # 2 options
    {"n_components": {"_or_": [5, 10, 15]}}  # 3 options
]
# → 6 variants, runs fast

# Phase 2: Fine-tune best region
best_preprocessing = SNV()  # From Phase 1
pipeline = [
    best_preprocessing,
    {"n_components": {"_range_": [8, 12]}}  # Fine sweep
]
# → 5 variants
```

### 2. Use Appropriate Generator for the Task

| Task | Use | Don't Use |
|------|-----|-----------|
| Discrete options | `_or_` | `_range_` |
| Linear sweep | `_range_` | `_log_range_` |
| Exponential sweep | `_log_range_` | `_range_` |
| All combinations | `_grid_` | Nested `_or_` |
| Paired parameters | `_zip_` | `_grid_` |
| Large spaces | `_sample_` | `_grid_` |
| Ordered experiments | `_chain_` | `_or_` |

### 3. Use Constraints to Prune Invalid Combinations

```python
# Only test combinations where SNV is not paired with MSC
pipeline = [
    {"_or_": ["SNV", "MSC", "Detrend", "None"],
     "_mutex_": [["SNV", "MSC"]],  # Mutual exclusion
     "pick": 2}
]
# Excludes [SNV, MSC] combination
```

See [generator_keywords.md](../../reference/generator_keywords.md#phase-4-production-keywords) for constraint keywords.

### 4. Progressive Complexity

```python
# Ablation study: baseline → +feature1 → +feature2
pipeline = [
    {"_chain_": [
        {},                                      # Baseline
        {"feature": "A"},                        # +A
        {"feature": "A", "feature2": "B"},       # +A+B
        {"feature": "A", "feature2": "B", "feature3": "C"}  # +A+B+C
    ]},
    PLSRegression(n_components=10)
]
```

### 5. Use Presets for Reusable Patterns

```python
from nirs4all.pipeline.config.generator import register_preset, resolve_presets_recursive

# Define preset
register_preset("pls_search", {
    "_grid_": {
        "n_components": {"_range_": [5, 20, 5]},
        "scale": [True, False]
    }
})

# Use in multiple pipelines
pipeline = [
    SNV(),
    {"_preset_": "pls_search", "model": PLSRegression}
]

# Resolve before execution
resolved_pipeline = resolve_presets_recursive(pipeline)
```

---

## Complete Example: Realistic Hyperparameter Search

```python
import nirs4all
from nirs4all.operators.transforms import SNV, MSC, FirstDerivative
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

# Define search space
pipeline = [
    # Cross-validation
    KFold(n_splits=5, shuffle=True, random_state=42),

    # Preprocessing: 3 scatter corrections
    {"_or_": [SNV(), MSC(), None]},

    # Optional derivative
    {"_or_": [FirstDerivative(), None]},

    # Model: grid search over n_components and scaling
    {"_grid_": {
        "n_components": {"_range_": [5, 20, 5]},  # [5, 10, 15, 20]
        "scale": [True, False]
    }, "model": PLSRegression}
]

# Count total variants
# 3 scatter × 2 derivative × (4 n_components × 2 scale) = 48 variants

# Run with cache optimization
from nirs4all.config.cache_config import CacheConfig

result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    cache=CacheConfig(
        step_cache_enabled=True,  # Reuse preprocessing
        step_cache_max_mb=2048
    ),
    verbose=1
)

# Analyze results
print(f"Tested {result.num_predictions} variants")
print(f"Best RMSE: {result.best_rmse:.4f}")
print(f"Best config: {result.best_config}")

# Export top 3 models
for i, pred in enumerate(result.top(3), 1):
    pred.export(f"model_top{i}.n4a")
```

---

## See Also

- {doc}`../../reference/generator_keywords` - Complete keyword reference
- {doc}`../advanced/cache_optimization` - Memory management for large searches
- {doc}`../../examples/developer` - Advanced generator examples (D01, D02)
- [examples/developer/02_generators/](../../examples/developer/02_generators/) - Runnable examples

```{seealso}
**Related Examples:**
- [D01: Generator Syntax](../../../examples/developer/02_generators/D01_generator_syntax.py) - Basic generator syntax: `_or_`, `_range_`, `pick`, `arrange`
- [D02: Generator Advanced](../../../examples/developer/02_generators/D02_generator_advanced.py) - Constraints, `_grid_`, `_zip_`, `_chain_`, `_sample_`
- [D03: Generator Iterators](../../../examples/developer/02_generators/D03_generator_iterators.py) - Lazy iteration and batch processing
- [U01: Multi-Model](../../../examples/user/04_models/U01_multi_model.py) - Simple generator usage for model comparison
```
