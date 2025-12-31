# Combination Generator Documentation

> **Last Updated**: December 8, 2025 | **Version**: Post Phase 1.5

## Overview

The Combination Generator is a powerful Python utility that expands configuration specifications into all possible combinations based on flexible syntax rules. It supports basic combinations, size constraints, explicit selection semantics (`pick`/`arrange`), second-order permutations (`then_pick`/`then_arrange`), and stochastic sampling for efficient exploration of large configuration spaces.

## Architecture

The generator module follows a modular architecture:

```
nirs4all/pipeline/config/
├── generator.py              # Main API: expand_spec(), count_combinations()
└── _generator/
    ├── __init__.py           # Package exports
    ├── keywords.py           # Keyword constants and detection utilities
    └── utils/
        ├── __init__.py
        ├── combinatorics.py  # Combination/permutation generators and counters
        └── sampling.py       # Deterministic random sampling with seed support
```

## Core Concepts

### Basic "_or_" Combinations

The fundamental building block is the `"_or_"` key that defines a set of choices:

```python
{"_or_": ["A", "B", "C"]}
# Generates: ["A", "B", "C"]
```

### Selection Semantics: `pick` vs `arrange`

Two explicit keywords control how items are selected:

| Keyword | Meaning | Mathematical Basis | When to Use |
|---------|---------|-------------------|-------------|
| `pick` | Select N items, order doesn't matter | Combinations C(n, k) | Parallel/independent operations |
| `arrange` | Arrange N items in sequence, order matters | Permutations P(n, k) | Sequential/chained operations |

```python
# Unordered selection (combinations) - order doesn't matter
{"_or_": ["A", "B", "C"], "pick": 2}
# Generates: [["A", "B"], ["A", "C"], ["B", "C"]]
# C(3, 2) = 3 variants

# Ordered arrangement (permutations) - order matters
{"_or_": ["A", "B", "C"], "arrange": 2}
# Generates: [["A", "B"], ["A", "C"], ["B", "A"], ["B", "C"], ["C", "A"], ["C", "B"]]
# P(3, 2) = 6 variants
```

### Legacy `size` Parameter

The `size` parameter is maintained for backward compatibility and behaves like `pick` (combinations):

```python
# Legacy syntax (equivalent to pick)
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

### Second-Order Selection with `then_pick` / `then_arrange`

For hierarchical selection, use `then_pick` or `then_arrange` after a primary selection:

```python
# Step 1: pick=2 generates C(3,2) = 3 combinations: [A,B], [A,C], [B,C]
# Step 2: then_arrange=2 generates P(3,2) = 6 arrangements of those 3 items
{"_or_": ["A", "B", "C"], "pick": 2, "then_arrange": 2}

# Step 1: arrange=2 generates P(3,2) = 6 permutations
# Step 2: then_pick=2 generates C(6,2) = 15 combinations of those
{"_or_": ["A", "B", "C"], "arrange": 2, "then_pick": 2}
```

### Legacy Second-Order Array Syntax

The array syntax `[outer, inner]` is supported for backward compatibility:

```python
{"_or_": ["A", "B"], "size": [1, 2]}
# Inner: permutations (order matters within sub-arrays)
# Outer: combinations (order doesn't matter for selection)
```

### Stochastic Sampling

Use `"count"` to randomly sample from large result sets:

```python
{"_or_": ["A", "B", "C", "D"], "pick": (2, 3), "count": 5}
# Generates: Random 5 combinations from all possible size 2-3 combinations
```

## Complete Syntax Reference

### Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `_or_` | Choice between alternatives | `{"_or_": ["A", "B", "C"]}` |
| `_range_` | Numeric sequence generation | `{"_range_": [1, 10, 2]}` |
| `size` | Number of items to select (legacy, uses combinations) | `{"_or_": [...], "size": 2}` |
| `pick` | Unordered selection (combinations) | `{"_or_": [...], "pick": 2}` |
| `arrange` | Ordered arrangement (permutations) | `{"_or_": [...], "arrange": 2}` |
| `then_pick` | Apply combinations to primary results | `{"_or_": [...], "pick": 2, "then_pick": 2}` |
| `then_arrange` | Apply permutations to primary results | `{"_or_": [...], "pick": 2, "then_arrange": 2}` |
| `count` | Limit number of generated variants | `{"_or_": [...], "count": 10}` |

### Basic Features

| Syntax | Description | Example Output |
|--------|-------------|----------------|
| `{"_or_": ["A", "B", "C"]}` | All individual choices | `["A", "B", "C"]` |
| `{"_or_": ["A", "B", "C"], "pick": 2}` | Combinations of exactly 2 elements | `[["A", "B"], ["A", "C"], ["B", "C"]]` |
| `{"_or_": ["A", "B", "C"], "pick": (1, 3)}` | Combinations of 1 to 3 elements | `[["A"], ["B"], ["C"], ["A", "B"], ...]` |
| `{"_or_": ["A", "B", "C"], "arrange": 2}` | Permutations of exactly 2 elements | `[["A", "B"], ["A", "C"], ["B", "A"], ...]` |
| `{"_or_": ["A", "B", "C"], "count": 2}` | Random 2 choices | `["A", "C"]` (random) |

### Numeric Range Features

| Syntax | Description | Example Output |
|--------|-------------|----------------|
| `{"_range_": [1, 5]}` | Range from 1 to 5 (inclusive) | `[1, 2, 3, 4, 5]` |
| `{"_range_": [0, 10, 2]}` | Range with step=2 | `[0, 2, 4, 6, 8, 10]` |
| `{"_range_": {"from": 1, "to": 5}}` | Dictionary syntax | `[1, 2, 3, 4, 5]` |
| `{"_range_": {"from": 0, "to": 20, "step": 5}}` | Dictionary with step | `[0, 5, 10, 15, 20]` |
| `{"_range_": [1, 100], "count": 5}` | Random sampling from range | `[23, 67, 12, 89, 45]` (random) |

### Second-Order Selection

| Syntax | Description | Key Behavior |
|--------|-------------|--------------|
| `pick: 2, then_pick: 2` | Pick 2, then pick 2 from results | Primary: C(n,2), Secondary: C(primary,2) |
| `pick: 2, then_arrange: 2` | Pick 2, then arrange 2 from results | Primary: C(n,2), Secondary: P(primary,2) |
| `arrange: 2, then_pick: 2` | Arrange 2, then pick 2 from results | Primary: P(n,2), Secondary: C(primary,2) |
| `arrange: 2, then_arrange: 2` | Arrange 2, then arrange 2 from results | Primary: P(n,2), Secondary: P(primary,2) |

### Legacy Second-Order Syntax (Array Notation)

| Syntax | Description | Key Behavior |
|--------|-------------|--------------|
| `size: [outer, inner]` | Array notation for second-order | Inner uses permutations, outer uses combinations |
| `size: [2, 2]` | Select 2 arrangements of 2 elements each | `[["A", "B"], ["B", "A"]]` are different |
| `size: [(1,3), 2]` | Select 1-3 arrangements of exactly 2 elements | Variable outer selection |
| `size: [2, (1,3)]` | Select exactly 2 arrangements of 1-3 elements | Variable inner arrangements |

### Advanced Combinations

| Syntax | Description | Use Case |
|--------|-------------|----------|
| `{"_or_": [...], "pick": 2, "count": 4}` | Random 4 from combinations | Large space sampling |
| `{"_or_": [...], "arrange": 2, "then_pick": 2, "count": N}` | Second-order with count limit | Efficient exploration |

## Key Behavioral Rules

### 1. `pick` vs `arrange`

- **`pick`**: `["A", "B"]` = `["B", "A"]` (combinations - order doesn't matter)
- **`arrange`**: `["A", "B"]` ≠ `["B", "A"]` (permutations - order matters)

**Use case guidance:**
- Use `pick` for `concat_transform` (feature order doesn't matter)
- Use `pick` for `feature_augmentation` (parallel channels)
- Use `arrange` for sequential preprocessing steps
- Use `arrange` when the order of operations affects the result

### 2. Second-Order Selection Logic

With `then_pick` / `then_arrange`:
- Primary selection (`pick` or `arrange`) happens first
- Secondary selection (`then_pick` or `then_arrange`) is applied to primary results
- Each step can use int or tuple (from, to) for size specification

```python
# Example: pick 2 combinations, then arrange 2 of those
{"_or_": ["A", "B", "C"], "pick": 2, "then_arrange": 2}
# Step 1: C(3,2) = 3 combinations: [A,B], [A,C], [B,C]
# Step 2: P(3,2) = 6 arrangements of those 3 items
```

### 3. Legacy `size` with Array Notation

In the legacy `size=[outer, inner]` second-order combinations:
- `[A, [B, C]]` ≠ `[A, [C, B]]` ✅ (different inner permutations)
- `[A, [B, C]]` = `[[B, C], A]` ✅ (same outer selection, different order doesn't matter)

### 4. Count Sampling

- Always applies **after** all combinations are generated
- Uses random sampling without replacement
- Works with any selection configuration

## Implementation Details

### Module Structure

The generator is organized into modular components:

```
nirs4all/pipeline/config/
├── generator.py                    # Main API
│   ├── expand_spec(node)          # Expand to all combinations
│   ├── count_combinations(node)   # Count without generating
│   ├── _expand_with_pick()        # Handle pick keyword
│   ├── _expand_with_arrange()     # Handle arrange keyword
│   ├── _handle_pick_then_*()      # Second-order with pick primary
│   ├── _handle_arrange_then_*()   # Second-order with arrange primary
│   └── _generate_range()          # Numeric range generation
│
└── _generator/
    ├── keywords.py                 # Keyword constants and utilities
    │   ├── OR_KEYWORD, RANGE_KEYWORD
    │   ├── PICK_KEYWORD, ARRANGE_KEYWORD
    │   ├── THEN_PICK_KEYWORD, THEN_ARRANGE_KEYWORD
    │   ├── is_generator_node()
    │   ├── is_pure_or_node()
    │   ├── extract_modifiers()
    │   └── extract_base_node()
    │
    └── utils/
        ├── combinatorics.py       # Mathematical operations
        │   ├── generate_combinations()
        │   ├── generate_permutations()
        │   ├── count_combinations()
        │   ├── count_permutations()
        │   └── normalize_size_spec()
        │
        └── sampling.py            # Deterministic random sampling
            ├── sample_with_seed()
            ├── shuffle_with_seed()
            └── random_choice_with_seed()
```

### Core Functions

#### `expand_spec(node)`
Main recursive expansion function that handles:
- Lists: Cartesian product expansion
- Dictionaries: OR nodes, range nodes, pick/arrange constraints, count limits
- Scalars: Direct return

#### `count_combinations(node)`
Calculate total number of combinations without generating them:
- Returns exact count that `expand_spec` would produce
- Uses mathematical formulas (combinations, permutations, factorials)
- Supports `_or_`, `_range_`, `pick`, `arrange`, `then_pick`, `then_arrange`
- Extremely fast even for large configuration spaces
- Essential for performance planning and safety checks

#### `_expand_with_pick(choices, pick_spec, count, then_pick, then_arrange)`
Handles the `pick` keyword (combinations):
- Generates C(n, k) combinations where order doesn't matter
- Supports int or tuple (from, to) range specification
- Handles second-order with `then_pick` or `then_arrange`
- Legacy `size=[outer, inner]` array notation for backward compatibility

#### `_expand_with_arrange(choices, arrange_spec, count, then_pick, then_arrange)`
Handles the `arrange` keyword (permutations):
- Generates P(n, k) permutations where order matters
- Supports int or tuple (from, to) range specification
- Handles second-order with `then_pick` or `then_arrange`

#### `_generate_range(range_spec)`
Generate numeric sequences from range specifications:
- Supports array syntax: `[from, to]` or `[from, to, step]`
- Supports dict syntax: `{"from": start, "to": end, "step": step}`
- Handles positive and negative steps
- End value is inclusive

### Keywords Module

Centralized keyword constants and detection utilities:

```python
from nirs4all.pipeline.config.generator import (
    # Constants
    OR_KEYWORD,           # "_or_"
    RANGE_KEYWORD,        # "_range_"
    PICK_KEYWORD,         # "pick"
    ARRANGE_KEYWORD,      # "arrange"
    THEN_PICK_KEYWORD,    # "then_pick"
    THEN_ARRANGE_KEYWORD, # "then_arrange"
    SIZE_KEYWORD,         # "size" (legacy)
    COUNT_KEYWORD,        # "count"

    # Detection functions
    is_generator_node,     # Check if node has _or_ or _range_
    is_pure_or_node,       # Check if node is purely an OR node
    is_pure_range_node,    # Check if node is purely a range node
    extract_modifiers,     # Extract size, count, pick, arrange modifiers
    extract_base_node,     # Extract non-keyword keys
)
```

### Sampling Utilities

Deterministic random sampling with optional seed support:

```python
from nirs4all.pipeline.config._generator.utils import sample_with_seed

# Deterministic sampling with seed
result = sample_with_seed(["A", "B", "C", "D"], k=2, seed=42)
# → Same result every time with seed=42

# Without seed (non-deterministic)
result = sample_with_seed(["A", "B", "C", "D"], k=2)
# → Random each time
```

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
from nirs4all.pipeline.config.generator import expand_spec, count_combinations

# Safe workflow
config = [{"_or_": ["A", "B", "C", "D"], "pick": [(1, 3), (1, 4)]}]

# Step 1: Estimate without generating
estimated_count = count_combinations(config)
print(f"Would generate {estimated_count:,} combinations")

# Step 2: Decide based on count
if estimated_count > 10000:
    # Add count limit for large spaces
    config[0]["count"] = 1000
    print("Added count limit for safe sampling")

# Step 3: Generate safely
results = expand_spec(config)
```

### Smart Generation Utility

```python
def estimate_and_generate(config, max_safe=1000):
    from nirs4all.pipeline.config.generator import expand_spec, count_combinations

    estimated = count_combinations(config)
    if estimated <= max_safe:
        return expand_spec(config)
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

- {doc}`/reference/pipeline_syntax` - Pipeline syntax reference
- {doc}`/examples/index` - Working examples

---

*This generator is designed for flexible configuration space exploration in machine learning pipelines, hyperparameter optimization, and systematic experimentation.*