# Combination Generator Documentation

## Overview

The Combination Generator is a powerful Python utility that expands configuration specifications into all possible combinations based on flexible syntax rules. It supports basic combinations, size constraints, second-order permutations, and stochastic sampling for efficient exploration of large configuration spaces.

## Core Concepts

### Basic "_or_" Combinations

The fundamental building block is the `"_or_"` key that defines a set of choices:

```python
{"_or_": ["A", "B", "C"]}
# Generates: ["A", "B", "C"]
```

### Size Constraints

Control the number of elements selected from choices:

```python
# Single size
{"_or_": ["A", "B", "C"], "size": 2}
# Generates: [["A", "B"], ["A", "C"], ["B", "C"]]

# Size range
{"_or_": ["A", "B", "C"], "size": (1, 2)}
# Generates: [["A"], ["B"], ["C"], ["A", "B"], ["A", "C"], ["B", "C"]]
```

### Numeric Ranges

Generate sequences of numeric values using the `"_range_"` key:

```python
# Array syntax: [from, to, step] - step defaults to 1
{"_range_": [1, 5]}
# Generates: [1, 2, 3, 4, 5]

{"_range_": [0, 10, 2]}
# Generates: [0, 2, 4, 6, 8, 10]

# Dictionary syntax: {"from": start, "to": end, "step": step}
{"_range_": {"from": 1, "to": 5, "step": 2}}
# Generates: [1, 3, 5]

# With count sampling for large ranges
{"_range_": [1, 1000], "count": 10}
# Generates: 10 random values from 1 to 1000
```

### Second-Order Combinations

Use array syntax `[outer, inner]` for hierarchical combinations where:
- **Inner**: Uses **permutations** (order matters within sub-arrays)
- **Outer**: Uses **combinations** (order doesn't matter for selection)

```python
{"_or_": ["A", "B"], "size": [1, 2]}
# Generates: [["A", "B"], ["B", "A"]]
# Note: Both permutations appear because order matters within inner arrays
```

### Stochastic Sampling

Use `"count"` to randomly sample from large result sets:

```python
{"_or_": ["A", "B", "C", "D"], "size": (2, 3), "count": 5}
# Generates: Random 5 combinations from all possible size 2-3 combinations
```

## Complete Syntax Reference

### Basic Features

| Syntax | Description | Example Output |
|--------|-------------|----------------|
| `{"_or_": ["A", "B", "C"]}` | All individual choices | `["A", "B", "C"]` |
| `{"_or_": ["A", "B", "C"], "size": 2}` | Combinations of exactly 2 elements | `[["A", "B"], ["A", "C"], ["B", "C"]]` |
| `{"_or_": ["A", "B", "C"], "size": (1, 3)}` | Combinations of 1 to 3 elements | `[["A"], ["B"], ["C"], ["A", "B"], ...]` |
| `{"_or_": ["A", "B", "C"], "count": 2}` | Random 2 choices | `["A", "C"]` (random) |

### Numeric Range Features

| Syntax | Description | Example Output |
|--------|-------------|----------------|
| `{"_range_": [1, 5]}` | Range from 1 to 5 (inclusive) | `[1, 2, 3, 4, 5]` |
| `{"_range_": [0, 10, 2]}` | Range with step=2 | `[0, 2, 4, 6, 8, 10]` |
| `{"_range_": {"from": 1, "to": 5}}` | Dictionary syntax | `[1, 2, 3, 4, 5]` |
| `{"_range_": {"from": 0, "to": 20, "step": 5}}` | Dictionary with step | `[0, 5, 10, 15, 20]` |
| `{"_range_": [1, 100], "count": 5}` | Random sampling from range | `[23, 67, 12, 89, 45]` (random) |

### Second-Order Syntax

| Syntax | Description | Key Behavior |
|--------|-------------|--------------|
| `[outer, inner]` | Array notation for second-order | Inner uses permutations, outer uses combinations |
| `[2, 2]` | Select 2 arrangements of 2 elements each | `[["A", "B"], ["B", "A"]]` are different |
| `[(1,3), 2]` | Select 1-3 arrangements of exactly 2 elements | Variable outer selection |
| `[2, (1,3)]` | Select exactly 2 arrangements of 1-3 elements | Variable inner arrangements |

### Advanced Combinations

| Syntax | Description | Use Case |
|--------|-------------|----------|
| `[2, 2, "count": 4]` | Random 4 from second-order combinations | Large space sampling |
| `{"_or_": [...], "size": [...], "count": N}` | Any combination with count limit | Efficient exploration |

## Key Behavioral Rules

### 1. First-Order vs Second-Order

- **First-Order**: `["A", "B"]` = `["B", "A"]` (combinations - order doesn't matter)
- **Second-Order**: `["A", "B"]` ≠ `["B", "A"]` (permutations - order matters within inner arrays)

### 2. Permutation Logic

In second-order combinations:
- `[A, [B, C]]` ≠ `[A, [C, B]]` ✅ (different inner permutations)
- `[A, [B, C]]` = `[[B, C], A]` ✅ (same outer selection, different order doesn't matter)

### 3. Count Sampling

- Always applies **after** all combinations are generated
- Uses random sampling without replacement
- Works with any size or second-order configuration

## Implementation Details

### Core Functions

#### `expand_spec_fixed(node)`
Main recursive expansion function that handles:
- Lists: Cartesian product expansion
- Dictionaries: OR nodes, range nodes, size constraints, count limits
- Scalars: Direct return

#### `count_combinations(node)`
**NEW**: Calculate total number of combinations without generating them:
- Returns exact count that `expand_spec_fixed` would produce
- Uses mathematical formulas (combinations, permutations, factorials)
- Supports both `_or_` and `_range_` nodes
- Extremely fast even for large configuration spaces
- Essential for performance planning and safety checks

#### `_handle_nested_combinations(choices, nested_size)`
Handles second-order array syntax `[outer, inner]`:
1. Generates all inner permutations (order matters)
2. Selects outer combinations (order doesn't matter)
3. Returns nested structure

#### `_count_nested_combinations(choices, nested_size)`
**NEW**: Counts second-order combinations using mathematical formulas:
- Calculates P(n,k) for inner permutations
- Calculates C(P,m) for outer combinations
- Avoids expensive enumeration

#### `_generate_range(range_spec)`
**NEW**: Generate numeric sequences from range specifications:
- Supports array syntax: `[from, to]` or `[from, to, step]`
- Supports dict syntax: `{"from": start, "to": end, "step": step}`
- Handles positive and negative steps
- End value is inclusive

#### `_count_range(range_spec)`
**NEW**: Count elements in numeric ranges without generating them:
- Uses mathematical formula: `(end - start) // step + 1`
- Handles edge cases (negative steps, invalid ranges)
- Extremely efficient for large ranges

#### `_expand_combination(combo)`
Expands specific combinations using Cartesian products.

#### `_expand_value_fixed(v)`
Handles value-position OR nodes with size and count constraints.

### Dependencies

```python
from itertools import product, combinations, permutations
from collections.abc import Mapping
from math import comb, factorial
import random
```

## Performance Planning

### Count Before Generate

**Always estimate first** for unknown configuration spaces:

```python
# Safe workflow
config = [{"_or_": ["A", "B", "C", "D"], "size": [(1, 3), (1, 4)]}]

# Step 1: Estimate without generating
estimated_count = count_combinations(config)
print(f"Would generate {estimated_count:,} combinations")

# Step 2: Decide based on count
if estimated_count > 10000:
    # Add count limit for large spaces
    config[0]["count"] = 1000
    print("Added count limit for safe sampling")

# Step 3: Generate safely
results = expand_spec_fixed(config)
```

### Smart Generation Utility

```python
def estimate_and_generate(config, max_safe=1000):
    estimated = count_combinations(config)
    if estimated <= max_safe:
        return expand_spec_fixed(config)
    else:
        print(f"Large space: {estimated:,}. Add count limit!")
        return None
```

## Usage Examples

### Basic Pipeline Configuration

```python
pipeline = [
    {"_or_": ["normalize", "standardize"]},
    {"model": {"_or_": ["svm", "rf", "xgb"], "size": 2}},
    {"features": {"_or_": ["pca", "lda"], "count": 1}}
]

results = expand_spec_fixed(pipeline)
# Generates all combinations of preprocessing, 2 models, and 1 random feature method
```

### Hyperparameter Range Exploration

```python
# Systematic hyperparameter search
hyperparams = {
    "model_params": {
        "n_estimators": {"_range_": [50, 200, 25]},      # [50, 75, 100, 125, 150, 175, 200]
        "max_depth": {"_range_": {"from": 3, "to": 15, "step": 2}},  # [3, 5, 7, 9, 11, 13, 15]
        "learning_rate": {"_or_": [0.01, 0.1, 0.2]}
    }
}

results = expand_spec_fixed(hyperparams)
# Generates: 7 × 7 × 3 = 147 hyperparameter combinations
```

### Mixed Range and Choice Combinations

```python
# Combine different generation strategies
config = [
    {"preprocessing": {"_or_": ["minmax", "standard", "robust"]}},
    {"batch_size": {"_range_": [16, 128, 16]}},  # [16, 32, 48, 64, 80, 96, 112, 128]
    {"optimizer": {"_or_": ["adam", "sgd"]}},
    {"epochs": {"_range_": [10, 50, 10], "count": 3}}  # Random 3 values from [10, 20, 30, 40, 50]
]

results = expand_spec_fixed(config)
# Generates: 3 × 8 × 2 × 3 = 144 training configurations
```

### Complex second-Order Example

```python
config = [{"_or_": ["A", "B", "C", "D"], "size": [(1, 3), (2, 4)]}]
results = expand_spec_fixed(config)
# Generates:
# - Inner: all permutations of 2-4 elements
# - Outer: select 1-3 of those inner arrangements
```

### Numeric Range Integration

```python
# Combine ranges with other features
pipeline = [
    {"preprocessing": {"_or_": ["normalize", "standardize"]}},
    {"n_estimators": {"_range_": [10, 100, 10]}},  # [10, 20, 30, ..., 100]
    {"max_depth": {"_range_": {"from": 3, "to": 10, "step": 2}}}  # [3, 5, 7, 9]
]

results = expand_spec_fixed(pipeline)
# Generates all combinations of preprocessing × n_estimators × max_depth
```

### Stochastic Exploration

```python
config = [{"_or_": ["method1", "method2", "method3", "method4"],
           "size": [3, (1, 4)],
           "count": 10}]
results = expand_spec_fixed(config)
# Random 10 samples from potentially thousands of combinations
```

## Performance Considerations

### Complexity Analysis

- **Basic OR**: O(n) where n = number of choices
- **Size constraints**: O(C(n,k)) where k = size, n = choices
- **Second-order**: O(P(n,k) × C(P,m)) where P = inner permutations, m = outer size
- **Numeric ranges**: O((end-start)/step + 1) - very efficient even for large ranges
- **Count sampling**: O(min(count, total_combinations))

### Memory Optimization

- Use `count` parameter for large combination spaces
- Consider tuple ranges instead of generating all intermediate sizes
- Second-order combinations can grow exponentially - use count limits

### Best Practices

1. **Estimate first**: Always use `count_combinations()` before generating
2. **Use count limits**: For spaces >1000 combinations, use count sampling
3. **Profile configurations**: Build up complexity incrementally
4. **Smart generation**: Use estimation utilities for safe workflows## Error Handling

### Common Issues

- **Memory errors**: Use count limits for large spaces
- **Type errors**: Ensure OR values are compatible with context
- **Size errors**: Size cannot exceed number of available choices
- **Empty results**: Check that size constraints are satisfiable

### Debug Tips

```python
# Check combination count first
config = [{"_or_": ["A", "B", "C"], "size": [2, 2]}]
results = expand_spec_fixed(config)
print(f"Total combinations: {len(results)}")

# Test range counting vs generation
range_config = {"_range_": [1, 1000]}
estimated = count_combinations(range_config)
print(f"Range 1-1000 has {estimated} values")  # Should be 1000

# Use count for large spaces
if len(results) > 100:
    config[0]["count"] = 10  # Limit to 10 samples
```

## Version History

- **v1.0**: Basic OR combinations and size constraints
- **v1.1**: Tuple range support for size constraints
- **v1.2**: Second-order combinations with array syntax
- **v1.3**: Stochastic count sampling
- **v1.4**: Fixed permutation logic for inner arrays
- **v1.5**: Added `_range_` keyword for numeric sequences with tuple and dict syntax

## Related Documentation

- [Configuration Format](config_format.md)
- [Nested Cross Validation](nested_cross_validation.md)
- [Usage Examples](../examples/)

---

*This generator is designed for flexible configuration space exploration in machine learning pipelines, hyperparameter optimization, and systematic experimentation.*