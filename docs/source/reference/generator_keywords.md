# Generator Keywords Reference

This document provides a comprehensive reference for all generator keywords used in nirs4all pipeline configuration expansion.

## Table of Contents

1. [Overview](#overview)
2. [Phase 1-2: Core Keywords](#phase-1-2-core-keywords)
   - [_or_](#_or_)
   - [_range_](#_range_)
   - [size](#size)
   - [pick](#pick)
   - [arrange](#arrange)
   - [then_pick](#then_pick)
   - [then_arrange](#then_arrange)
   - [count](#count)
3. [Phase 3: Advanced Keywords](#phase-3-advanced-keywords)
   - [_log_range_](#_log_range_)
   - [_grid_](#_grid_)
   - [_zip_](#_zip_)
   - [_chain_](#_chain_)
   - [_sample_](#_sample_)
   - [_tags_](#_tags_)
   - [_metadata_](#_metadata_)
4. [Phase 4: Production Keywords](#phase-4-production-keywords)
   - [_cartesian_](#_cartesian_)
   - [_mutex_](#_mutex_)
   - [_requires_](#_requires_)
   - [_depends_on_](#_depends_on_)
   - [_exclude_](#_exclude_)
   - [_preset_](#_preset_)
5. [Modifier Keywords](#modifier-keywords)
   - [_seed_](#_seed_)
   - [_weights_](#_weights_)
6. [API Functions](#api-functions)
7. [Selection Semantics: pick vs arrange](#selection-semantics-pick-vs-arrange)
8. [Common Patterns and Examples](#common-patterns-and-examples)

---

## Overview

The generator module expands pipeline configuration specifications into concrete pipeline variants. It takes a single configuration with combinatorial keywords and generates all possible combinations.

### Basic Import

```python
from nirs4all.pipeline.config.generator import (
    # Core API
    expand_spec,
    expand_spec_with_choices,
    count_combinations,

    # Iterator API
    expand_spec_iter,
    batch_iter,
    iter_with_progress,

    # Validation
    validate_spec,
    validate_config,
    validate_expanded_configs,

    # Presets
    PRESET_KEYWORD,
    register_preset,
    unregister_preset,
    get_preset,
    get_preset_info,
    list_presets,
    clear_presets,
    has_preset,
    is_preset_reference,
    resolve_preset,
    resolve_presets_recursive,
    export_presets,
    import_presets,
    register_builtin_presets,

    # Constraints
    apply_mutex_constraint,
    apply_requires_constraint,
    apply_exclude_constraint,
    apply_all_constraints,
    parse_constraints,
    validate_constraints,

    # Export utilities
    to_dataframe,
    diff_configs,
    summarize_configs,
    get_expansion_tree,
    print_expansion_tree,
    format_config_table,
    ExpansionTreeNode,

    # Keyword constants
    OR_KEYWORD,
    RANGE_KEYWORD,
    LOG_RANGE_KEYWORD,
    GRID_KEYWORD,
    ZIP_KEYWORD,
    CHAIN_KEYWORD,
    SAMPLE_KEYWORD,
    CARTESIAN_KEYWORD,
    SIZE_KEYWORD,
    COUNT_KEYWORD,
    SEED_KEYWORD,
    WEIGHTS_KEYWORD,
    PICK_KEYWORD,
    ARRANGE_KEYWORD,
    THEN_PICK_KEYWORD,
    THEN_ARRANGE_KEYWORD,
    TAGS_KEYWORD,
    METADATA_KEYWORD,
    MUTEX_KEYWORD,
    REQUIRES_KEYWORD,
    DEPENDS_ON_KEYWORD,
    EXCLUDE_KEYWORD,

    # Detection functions
    is_generator_node,
    is_pure_or_node,
    is_pure_range_node,
    is_pure_log_range_node,
    is_pure_grid_node,
    is_pure_zip_node,
    is_pure_chain_node,
    is_pure_sample_node,
    is_pure_cartesian_node,

    # Extraction functions
    extract_modifiers,
    extract_base_node,
    extract_or_choices,
    extract_range_spec,
    extract_tags,
    extract_metadata,
    extract_constraints,

    # Strategies (advanced usage)
    ExpansionStrategy,
    get_strategy,
    register_strategy,
    RangeStrategy,
    OrStrategy,
    LogRangeStrategy,
    GridStrategy,
    ZipStrategy,
    ChainStrategy,
    SampleStrategy,
    CartesianStrategy,
)
```

---

## Phase 1-2: Core Keywords

### `_or_`

Select from a list of alternatives. Each choice becomes a separate configuration variant.

**Syntax:**
```python
{"_or_": [choice1, choice2, ...]}
```

**Examples:**
```python
# Simple string choices
{"_or_": ["StandardScaler", "MinMaxScaler", "RobustScaler"]}
# → ["StandardScaler", "MinMaxScaler", "RobustScaler"]

# Dictionary choices
{"_or_": [
    {"class": "PCA", "n_components": 10},
    {"class": "SVD", "n_components": 10},
]}
# → [{"class": "PCA", "n_components": 10}, {"class": "SVD", "n_components": 10}]

# Mixed types
{"_or_": [None, 5, {"window": 11}]}
# → [None, 5, {"window": 11}]
```

**Modifiers:** `size`, `pick`, `arrange`, `then_pick`, `then_arrange`, `count`

---

### `_range_`

Generate a sequence of numeric values.

**Syntax:**
```python
# Array syntax
{"_range_": [start, end]}              # Inclusive, step=1
{"_range_": [start, end, step]}        # With custom step

# Dict syntax
{"_range_": {"from": start, "to": end, "step": step}}
```

**Examples:**
```python
{"_range_": [1, 5]}
# → [1, 2, 3, 4, 5]

{"_range_": [0, 20, 5]}
# → [0, 5, 10, 15, 20]

{"_range_": {"from": 10, "to": 50, "step": 10}}
# → [10, 20, 30, 40, 50]
```

---

### `size`

**(Legacy)** Select combinations of N items from `_or_` choices. Equivalent to `pick`.

**Syntax:**
```python
{"_or_": [...], "size": n}           # Fixed size
{"_or_": [...], "size": (min, max)}  # Range of sizes
{"_or_": [...], "size": [outer, inner]}  # Second-order (nested)
```

**Examples:**
```python
# Select 2 from 4 items → C(4,2) = 6 combinations
{"_or_": ["A", "B", "C", "D"], "size": 2}
# → [["A", "B"], ["A", "C"], ["A", "D"], ["B", "C"], ["B", "D"], ["C", "D"]]

# Size range
{"_or_": ["A", "B", "C"], "size": (1, 2)}
# → [["A"], ["B"], ["C"], ["A", "B"], ["A", "C"], ["B", "C"]]
```

---

### `pick`

**(Explicit)** Unordered selection - combinations where order doesn't matter.

**Syntax:**
```python
{"_or_": [...], "pick": n}           # Fixed size
{"_or_": [...], "pick": (min, max)}  # Range of sizes
```

**Mathematical formula:** C(n, k) = n! / (k! × (n-k)!)

**Examples:**
```python
# Pick 2 from 3 → C(3,2) = 3
{"_or_": ["A", "B", "C"], "pick": 2}
# → [["A", "B"], ["A", "C"], ["B", "C"]]
```

**Use cases:**
- `concat_transform` where feature order doesn't matter
- `feature_augmentation` for parallel channels
- Any scenario where [A, B] and [B, A] should be treated as equivalent

---

### `arrange`

**(Explicit)** Ordered arrangement - permutations where order matters.

**Syntax:**
```python
{"_or_": [...], "arrange": n}           # Fixed size
{"_or_": [...], "arrange": (min, max)}  # Range of sizes
```

**Mathematical formula:** P(n, k) = n! / (n-k)!

**Examples:**
```python
# Arrange 2 from 3 → P(3,2) = 6
{"_or_": ["A", "B", "C"], "arrange": 2}
# → [["A", "B"], ["A", "C"], ["B", "A"], ["B", "C"], ["C", "A"], ["C", "B"]]
```

**Use cases:**
- Sequential preprocessing pipelines
- Any scenario where order of operations affects results
- When [A, B] and [B, A] should be treated as different configurations

---

### `then_pick`

Second-order operation: apply combinations to the results of a primary selection.

**Syntax:**
```python
{"_or_": [...], "pick": n1, "then_pick": n2}
{"_or_": [...], "arrange": n1, "then_pick": n2}
```

**Example:**
```python
# Pick 2, then pick 2 from those 3 results
{"_or_": ["A", "B", "C"], "pick": 2, "then_pick": 2}
# Step 1: pick=2 → C(3,2) = 3 combos: [A,B], [A,C], [B,C]
# Step 2: then_pick=2 → C(3,2) = 3 selections of those combos
```

---

### `then_arrange`

Second-order operation: apply permutations to the results of a primary selection.

**Syntax:**
```python
{"_or_": [...], "pick": n1, "then_arrange": n2}
{"_or_": [...], "arrange": n1, "then_arrange": n2}
```

**Example:**
```python
# Pick 2, then arrange 2 from those results
{"_or_": ["A", "B", "C"], "pick": 2, "then_arrange": 2}
# Step 1: pick=2 → 3 combos: [A,B], [A,C], [B,C]
# Step 2: then_arrange=2 → P(3,2) = 6 arrangements
```

---

### `count`

Limit the number of results returned. With a seed, results are deterministic.

**Syntax:**
```python
{"_or_": [...], "count": n}
{"_or_": [...], "size": k, "count": n}
```

**Example:**
```python
# Get 2 random items from 5
{"_or_": ["A", "B", "C", "D", "E"], "count": 2}
# → 2 randomly selected items

# With seed for reproducibility
expand_spec({"_or_": ["A", "B", "C", "D", "E"], "count": 2}, seed=42)
# → Same 2 items every time with seed=42
```

---

## Phase 3: Advanced Keywords

### `_log_range_`

**Purpose**: Generate logarithmically-spaced numeric sequences. Useful for hyperparameter optimization over values spanning multiple orders of magnitude (e.g., learning rates, regularization).

**Syntax:**
```python
# Array syntax: [from, to, num_values]
{"_log_range_": [start, end, num]}

# Dict syntax
{"_log_range_": {"from": start, "to": end, "num": n}}
{"_log_range_": {"from": start, "to": end, "base": b}}  # Custom base (default: 10)
```

**Result**: Generates `num` values logarithmically spaced from `start` to `end`.

**Examples:**
```python
# 4 values from 0.001 to 1 (base 10)
{"_log_range_": [0.001, 1, 4]}
# → [0.001, 0.01, 0.1, 1.0]

# Learning rate search spanning 4 orders of magnitude
{"_log_range_": [0.0001, 0.1, 5]}
# → [0.0001, 0.001, 0.01, 0.1]  (approximately)

# Base 2 powers for depth/width parameters
{"_log_range_": {"from": 1, "to": 256, "num": 9, "base": 2}}
# → [1, 2, 4, 8, 16, 32, 64, 128, 256]

# Pipeline example: regularization search
pipeline = [
    {"model": PLSRegression},
    {"alpha": {"_log_range_": [0.0001, 10, 8]}}
]
# Generates 8 pipeline variants with exponentially spaced alpha values
```

**See also**: `_range_` (linear spacing), `_sample_` (random sampling)

**Example**: See [examples/developer/02_generators/D01_generator_syntax.py](../../examples/developer/02_generators/D01_generator_syntax.py)

---

### `_grid_`

**Purpose**: Generate Cartesian product of parameter spaces (full grid search). Similar to sklearn's `ParameterGrid` - creates all possible combinations of parameters.

**Syntax:**
```python
{"_grid_": {"param1": [v1, v2, ...], "param2": [v3, v4, ...]}}
```

**Result**: Generates all combinations (param1 × param2 × ...). Returns list of dicts.

**Examples:**
```python
# 2×3 = 6 combinations
{"_grid_": {"learning_rate": [0.01, 0.1], "batch_size": [16, 32, 64]}}
# → [{"learning_rate": 0.01, "batch_size": 16},
#    {"learning_rate": 0.01, "batch_size": 32},
#    {"learning_rate": 0.01, "batch_size": 64},
#    {"learning_rate": 0.1, "batch_size": 16},
#    {"learning_rate": 0.1, "batch_size": 32},
#    {"learning_rate": 0.1, "batch_size": 64}]

# Grid with nested generator
{"_grid_": {
    "n_components": {"_range_": [5, 15, 5]},  # [5, 10, 15]
    "scale": [True, False]
}}
# → 3 × 2 = 6 combinations

# Pipeline example: full grid search
pipeline = [
    {"_grid_": {
        "preprocessing": ["SNV", "MSC", None],
        "n_components": [5, 10, 15]
    }, "model": PLSRegression}
]
# Generates 3 × 3 = 9 pipeline variants
```

**See also**: `_zip_` (paired iteration), `_cartesian_` (for lists, not dicts), `_sample_` (random subset)

**Example**: See [examples/developer/02_generators/D01_generator_syntax.py](../../examples/developer/02_generators/D01_generator_syntax.py)

---

### `_zip_`

**Purpose**: Parallel iteration - pair values at the same index (like Python's `zip()`). Use when parameters should vary together, not independently.

**Syntax:**
```python
{"_zip_": {"param1": [v1, v2, ...], "param2": [v3, v4, ...]}}
```

**Result**: Generates N configurations where N = min(len(param1), len(param2), ...). Pairs by position.

**Examples:**
```python
# 3 paired configurations
{"_zip_": {"x": [1, 2, 3], "y": ["A", "B", "C"]}}
# → [{"x": 1, "y": "A"}, {"x": 2, "y": "B"}, {"x": 3, "y": "C"}]

# Mismatched lengths: shortest list wins
{"_zip_": {"x": [1, 2, 3, 4], "y": ["A", "B"]}}
# → [{"x": 1, "y": "A"}, {"x": 2, "y": "B"}]  # Only 2 configurations

# Pipeline example: paired hyperparameters
pipeline = [
    {"_zip_": {
        "alpha": [0.1, 1.0, 10.0],
        "l1_ratio": [0.2, 0.5, 0.8]  # Paired with alpha
    }, "model": ElasticNet}
]
# Generates 3 variants: (0.1, 0.2), (1.0, 0.5), (10.0, 0.8)
```

**Comparison with `_grid_`:**
```python
# _zip_: 2 paired configurations (N pairs)
{"_zip_": {"x": [1, 2], "y": ["A", "B"]}}
# → [{"x": 1, "y": "A"}, {"x": 2, "y": "B"}]

# _grid_: 4 all-combinations (N × M)
{"_grid_": {"x": [1, 2], "y": ["A", "B"]}}
# → [{"x": 1, "y": "A"}, {"x": 1, "y": "B"},
#    {"x": 2, "y": "A"}, {"x": 2, "y": "B"}]
```

**See also**: `_grid_` (all combinations), `_chain_` (sequential)

**Example**: See [examples/developer/02_generators/D01_generator_syntax.py](../../examples/developer/02_generators/D01_generator_syntax.py)

---

### `_chain_`

**Purpose**: Sequential ordered choices. Preserves exact order of configurations (unlike `_or_` which may randomize with `count`). Ideal for progressive experiments.

**Syntax:**
```python
{"_chain_": [config1, config2, config3, ...]}
```

**Result**: Generates configurations in the exact order provided. Behaves like `_or_` but with guaranteed order.

**Examples:**
```python
# 3 configurations in exact order
{"_chain_": [
    {"model": "baseline", "complexity": "low"},
    {"model": "improved", "complexity": "medium"},
    {"model": "best", "complexity": "high"}
]}
# → Returns in that exact order

# Pipeline example: progressive preprocessing complexity
pipeline = [
    {"_chain_": [
        {},                    # No preprocessing (baseline)
        {"transform": "SNV"},  # Simple SNV
        {"transform": "SNV", "derivative": 1},  # SNV + 1st derivative
        {"transform": "SNV", "derivative": 2}   # SNV + 2nd derivative
    ]},
    PLSRegression(n_components=10)
]
# Runs 4 variants in order of increasing complexity
```

**Use cases:**
- Progressive experiments: baseline → improved → best
- Ablation studies with defined order
- When configuration order has semantic meaning
- Avoiding randomization from `count`

**See also**: `_or_` (unordered choices), `_zip_` (paired parameters)

**Example**: See [examples/developer/02_generators/D02_generator_advanced.py](../../examples/developer/02_generators/D02_generator_advanced.py)

---

### `_sample_`

**Purpose**: Statistical sampling from probability distributions. Use for random search in large hyperparameter spaces instead of exhaustive grid search.

**Syntax:**
```python
{"_sample_": {"distribution": "uniform|log_uniform|normal|choice", ...}}
```

**Distributions:**

| Distribution | Parameters | Description | Use Case |
|-------------|------------|-------------|----------|
| `uniform` | `from`, `to`, `num` | Uniform distribution | Regularization, dropout rates |
| `log_uniform` | `from`, `to`, `num` | Log-uniform (logarithmic scale) | Learning rates, decay factors |
| `normal`/`gaussian` | `mean`, `std`, `num` | Normal/Gaussian distribution | Random initialization |
| `choice` | `values`, `num` | Random selection with replacement | Categorical parameters |

**Result**: Generates `num` random values from specified distribution. Use `_seed_` modifier for reproducibility.

**Examples:**
```python
# Uniform sampling (0 to 1)
{"_sample_": {"distribution": "uniform", "from": 0.1, "to": 1.0, "num": 5}}
# → 5 random values uniformly distributed: [0.32, 0.67, 0.15, 0.89, 0.44]

# Log-uniform (learning rate search)
{"_sample_": {"distribution": "log_uniform", "from": 0.0001, "to": 0.1, "num": 5}}
# → 5 values log-uniformly distributed: [0.0003, 0.002, 0.015, 0.0008, 0.05]

# Normal distribution
{"_sample_": {"distribution": "normal", "mean": 0, "std": 1, "num": 5}}
# → 5 values from standard normal: [-0.32, 1.15, 0.05, -0.89, 0.67]

# Random choice from list
{"_sample_": {"distribution": "choice", "values": ["A", "B", "C", "D"], "num": 3}}
# → 3 randomly selected values: ["B", "A", "B"]  (with replacement)

# Reproducible sampling with seed
{"_sample_": {"distribution": "uniform", "from": 0, "to": 1, "num": 10}, "_seed_": 42}
# → Same 10 values every time with seed=42

# Pipeline example: random regularization search
pipeline = [
    {"alpha": {"_sample_": {
        "distribution": "log_uniform",
        "from": 0.0001,
        "to": 10.0,
        "num": 20
    }}, "model": Ridge}
]
# Generates 20 pipeline variants with random alpha values
```

**See also**: `_log_range_` (deterministic log scale), `_or_` with `count` (random subset)

**Example**: See [examples/developer/02_generators/D02_generator_advanced.py](../../examples/developer/02_generators/D02_generator_advanced.py)

---

### `_tags_`

Add tags to configurations for filtering and categorization.

**Syntax:**
```python
{"_or_": [...], "_tags_": ["tag1", "tag2"]}
```

---

### `_metadata_`

Attach arbitrary metadata to configurations.

**Syntax:**
```python
{"_or_": [...], "_metadata_": {"key": "value", ...}}
```

---

## Phase 4: Production Keywords

### `_cartesian_`

**Purpose**: Generate the Cartesian product of multiple stages (each with `_or_` choices), then optionally apply pick/arrange selection on the resulting complete pipelines. Key pattern for preprocessing pipeline generation.

**Syntax:**
```python
{"_cartesian_": [stage1, stage2, ...]}
{"_cartesian_": [stage1, stage2, ...], "pick": N}
{"_cartesian_": [stage1, stage2, ...], "arrange": N}
{"_cartesian_": [stage1, stage2, ...], "pick": (min, max), "count": N}
```

**Result**:
- First expands all stages → generates stage1 × stage2 × ... complete pipeline lists
- Then optionally selects subset using `pick` (combinations) or `arrange` (permutations)
- Returns list of lists (each inner list is a complete pipeline)

**Examples:**
```python
# Generate all 2×2 = 4 complete pipelines
{"_cartesian_": [
    {"_or_": ["SNV", "MSC"]},
    {"_or_": ["Deriv1", "Deriv2"]}
]}
# → [["SNV", "Deriv1"], ["SNV", "Deriv2"],
#    ["MSC", "Deriv1"], ["MSC", "Deriv2"]]

# Generate 3×3×3 = 27 pipelines, then pick 2-combinations
{"_cartesian_": [
    {"_or_": ["MSC", "SNV", "EMSC"]},        # Scatter correction
    {"_or_": ["SavGol", "Gaussian", None]},  # Smoothing
    {"_or_": [None, "Deriv1", "Deriv2"]}     # Derivative
], "pick": 2}
# → C(27, 2) = 351 pairs of complete pipelines

# Pick 1-3 complete pipelines with count limit
{"_cartesian_": [
    {"_or_": ["A", "B"]},
    {"_or_": ["X", "Y"]}
], "pick": (1, 3), "count": 20}
# → Up to 20 random selections from all 1-, 2-, and 3-combinations

# Pipeline example: preprocessing pipeline search
pipeline = [
    {"_cartesian_": [
        {"_or_": [SNV(), MSC(), None]},         # 3 scatter corrections
        {"_or_": [FirstDerivative(), None]},    # 2 derivative options
        {"_or_": [Detrend(order=1), None]}      # 2 detrending options
    ], "pick": 1, "count": 10},  # Select 10 random complete pipelines
    PLSRegression(n_components=10)
]
# From 3×2×2=12 possible pipelines, randomly select 10
```

**Difference from `_grid_`:**

| Feature | `_grid_` | `_cartesian_` |
|---------|----------|---------------|
| Output | Dict (parameters) | List of lists (stages) |
| Use case | Hyperparameters | Pipeline stages |
| Example | `{"x": 1, "y": 2}` | `[step1, step2, step3]` |
| Selection | Not applicable | Can use `pick`/`arrange` |

```python
# _grid_ for parameters
{"_grid_": {"alpha": [0.1, 1.0], "beta": [2, 3]}}
# → [{"alpha": 0.1, "beta": 2}, {"alpha": 0.1, "beta": 3}, ...]

# _cartesian_ for pipeline stages
{"_cartesian_": [{"_or_": ["A", "B"]}, {"_or_": ["X", "Y"]}]}
# → [["A", "X"], ["A", "Y"], ["B", "X"], ["B", "Y"]]
```

**Use cases:**
- Preprocessing pipeline generation (scatter correction → smoothing → derivative)
- Any staged pipeline where order matters
- Selecting subset of complete pipeline variants
- Exploring combinations of sequential transformations

**See also**: `_grid_` (for dicts), `_or_` with `arrange` (for single stage)

**Example**: See [examples/developer/02_generators/D01_generator_syntax.py](../../examples/developer/02_generators/D01_generator_syntax.py)

---

### `_mutex_`

Mutual exclusion constraint - certain items cannot appear together.

**Syntax:**
```python
{"_or_": [...], "pick": n, "_mutex_": [[item1, item2], [item3, item4]]}
```

**Example:**
```python
# A and B cannot appear together
{"_or_": ["A", "B", "C", "D"], "pick": 2, "_mutex_": [["A", "B"]]}
# All combinations: [A,B], [A,C], [A,D], [B,C], [B,D], [C,D]
# After _mutex_:    [A,C], [A,D], [B,C], [B,D], [C,D]  (A,B excluded)
```

---

### `_requires_`

Dependency constraint - if item A is selected, item B must also be selected.

**Syntax:**
```python
{"_or_": [...], "pick": n, "_requires_": [[trigger, required1, required2]]}
```

**Example:**
```python
# If A is selected, C must also be selected
{"_or_": ["A", "B", "C", "D"], "pick": 2, "_requires_": [["A", "C"]]}
# Valid: [A,C], [B,C], [B,D], [C,D]
# Invalid: [A,B], [A,D] (A without C)
```

---

### `_depends_on_`

Conditional expansion - expansion depends on the value of another parameter.

**Syntax:**
```python
{"_or_": [...], "_depends_on_": "other_param"}
```

**Use cases:**
- Conditional hyperparameter spaces
- Parameters that only apply when another parameter has a certain value

---

### `_exclude_`

Exclude specific combinations from results.

**Syntax:**
```python
{"_or_": [...], "pick": n, "_exclude_": [[combo1], [combo2]]}
```

**Example:**
```python
# Exclude specific combinations [A,C] and [B,D]
{"_or_": ["A", "B", "C", "D"], "pick": 2, "_exclude_": [["A", "C"], ["B", "D"]]}
# Remaining: [A,B], [A,D], [B,C], [C,D]
```

---

### `_preset_`

Reference a named preset configuration.

**Syntax:**
```python
{"_preset_": "preset_name"}
```

**Usage:**
```python
from nirs4all.pipeline.config.generator import register_preset, resolve_presets_recursive

# Register presets
register_preset(
    "spectral_transforms",
    {"_or_": ["SNV", "MSC", "Detrend"], "pick": (1, 2)},
    description="Common spectral preprocessing"
)

register_preset(
    "pls_components",
    {"_range_": [2, 15]}
)

# Use in configuration
config = {
    "transforms": {"_preset_": "spectral_transforms"},
    "model": {
        "class": "PLSRegression",
        "n_components": {"_preset_": "pls_components"}
    }
}

# Resolve presets before expansion
resolved = resolve_presets_recursive(config)
results = expand_spec(resolved)
```

---

## Modifier Keywords

### `_seed_`

Provide a deterministic seed for random operations within a node. This ensures reproducible generation when using `count` or random sampling.

**Syntax:**
```python
{"_or_": [...], "count": N, "_seed_": 42}
{"_sample_": {...}, "_seed_": 42}
```

**Examples:**
```python
# Reproducible random selection
{"_or_": ["A", "B", "C", "D", "E"], "count": 2, "_seed_": 42}
# → Same 2 items every time

# Reproducible sampling
{"_sample_": {"distribution": "uniform", "from": 0, "to": 1, "num": 5}, "_seed_": 123}
# → Same 5 values every time
```

---

### `_weights_`

Provide weights for weighted random selection when using `count`.

**Syntax:**
```python
{"_or_": [...], "count": N, "_weights_": [w1, w2, ...]}
```

**Examples:**
```python
# Weighted random selection (A is 3x more likely than others)
{"_or_": ["A", "B", "C", "D"], "count": 2, "_weights_": [3, 1, 1, 1]}
```

---

## API Functions

### Core Functions

```python
# Expand a specification to all variants
results = expand_spec(spec, seed=None)

# Expand with choice tracking (returns configs and choice paths)
results, choices = expand_spec_with_choices(spec, seed=None)

# Count variants without generating
count = count_combinations(spec)
```

### Iterator Functions

```python
# Lazy iteration for large spaces
for config in expand_spec_iter(spec, seed=None):
    process(config)

# With sampling (uses reservoir sampling for uniform distribution)
configs = list(expand_spec_iter(spec, seed=42, sample_size=100))

# Batch processing
for batch in batch_iter(spec, batch_size=10):
    process_batch(batch)

# With progress reporting
for i, config in iter_with_progress(spec, report_every=1000):
    process(config)
```

### Preset Functions

```python
# Register a preset
register_preset(name, spec, description=None, tags=None, overwrite=False)

# Get preset specification
spec = get_preset(name)

# Get preset info (spec, description, tags)
info = get_preset_info(name)

# List and manage presets
names = list_presets(tags=None)  # Filter by tags optionally
has_preset(name)
unregister_preset(name)
clear_presets()

# Resolve presets in a config (handles circular reference detection)
resolved = resolve_presets_recursive(config)

# Check if a node is a preset reference
is_preset_reference(node)

# Export/import presets
presets_dict = export_presets()
count = import_presets(presets_dict, overwrite=False)

# Register built-in presets (standard_scalers, pls_components, learning_rates)
register_builtin_presets()
```

### Constraint Functions

```python
# Apply individual constraints
filtered = apply_mutex_constraint(results, mutex_groups)
filtered = apply_requires_constraint(results, requires_groups)
filtered = apply_exclude_constraint(results, exclude_combos)

# Apply all constraints at once
filtered = apply_all_constraints(results, mutex_groups, requires_groups, exclude_combos)

# Parse and validate constraints
parsed = parse_constraints(constraint_spec)
errors = validate_constraints(constraint_spec)
```

### Export Functions

```python
# Convert to pandas DataFrame
df = to_dataframe(configs, flatten=True, prefix_sep=".", include_index=True)

# Compare configurations
diff = diff_configs(config1, config2)

# Summary statistics
summary = summarize_configs(configs, max_unique=10)

# Tree visualization
tree_str = print_expansion_tree(spec, indent="  ", show_counts=True, max_depth=None)
tree_node = get_expansion_tree(spec)

# ASCII table formatting
table_str = format_config_table(configs, columns=None, max_rows=20)
```

### Validation Functions

```python
# Validate a specification
result = validate_spec(spec)
if not result.is_valid:
    print(result.errors)

# Validate a config dict
result = validate_config(config, schema=None)

# Validate expanded configs
results = validate_expanded_configs(configs, schema=None)
```

### Detection Functions

```python
# Check if a node contains any generator keywords
is_generator_node(node)  # True if has _or_, _range_, etc.

# Check for specific node types
is_pure_or_node(node)       # Only OR-related keys
is_pure_range_node(node)    # Only range-related keys
is_pure_log_range_node(node)
is_pure_grid_node(node)
is_pure_zip_node(node)
is_pure_chain_node(node)
is_pure_sample_node(node)
is_pure_cartesian_node(node)

# Check for specific keywords
has_or_keyword(node)
has_range_keyword(node)
has_log_range_keyword(node)
has_grid_keyword(node)
has_zip_keyword(node)
has_chain_keyword(node)
has_sample_keyword(node)
has_cartesian_keyword(node)
```

### Extraction Functions

```python
# Extract modifiers (size, count, pick, arrange, etc.)
modifiers = extract_modifiers(node)

# Extract non-keyword keys
base = extract_base_node(node)

# Extract specific elements
choices = extract_or_choices(node)      # From _or_ node
range_spec = extract_range_spec(node)   # From _range_ node
tags = extract_tags(node)               # From _tags_
metadata = extract_metadata(node)       # From _metadata_
constraints = extract_constraints(node) # From _mutex_, _requires_, etc.
```

---

## Selection Semantics: pick vs arrange

| Aspect | `pick` (Combinations) | `arrange` (Permutations) |
|--------|----------------------|--------------------------|
| Order matters? | No | Yes |
| [A, B] vs [B, A] | Same | Different |
| Formula | C(n,k) = n!/(k!(n-k)!) | P(n,k) = n!/(n-k)! |
| Count for 3 choose 2 | 3 | 6 |
| Use case | Feature sets | Processing pipelines |

**When to use `pick`:**
- `concat_transform` where feature order doesn't matter
- `feature_augmentation` for parallel channels
- Any unordered collection

**When to use `arrange`:**
- Sequential preprocessing steps
- When operation order affects results
- Pipeline stages with dependencies

---

## Common Patterns and Examples

### 1. Hyperparameter Grid Search

```python
{
    "_grid_": {
        "model": ["PLS", "RF", "SVR"],
        "n_components": {"_range_": [5, 20, 5]},
        "preprocessing": ["StandardScaler", "MinMaxScaler", None]
    }
}
```

### 2. Learning Rate Search

```python
{
    "optimizer": "Adam",
    "learning_rate": {"_log_range_": [0.0001, 0.1, 10]},
    "batch_size": {"_or_": [16, 32, 64, 128]}
}
```

### 3. Preprocessing Pipeline Combinations

```python
{
    "feature_augmentation": {
        "_or_": [
            {"class": "SNV"},
            {"class": "MSC"},
            {"class": "Detrend", "order": {"_or_": [1, 2]}},
            {"class": "SavitzkyGolay", "window": {"_or_": [5, 11, 21]}}
        ],
        "pick": (1, 3)  # 1 to 3 transforms
    }
}
```

### 4. Constrained Combinations

```python
{
    "_or_": ["PCA", "ICA", "NMF", "UMAP"],
    "pick": 2,
    "_mutex_": [["PCA", "ICA"]],  # PCA and ICA can't be together
    "_requires_": [["UMAP", "NMF"]]  # If UMAP selected, NMF required
}
```

### 5. Progressive Experiments with Chain

```python
{
    "_chain_": [
        {"model": "baseline", "transforms": []},
        {"model": "baseline", "transforms": ["SNV"]},
        {"model": "improved", "transforms": ["SNV", "Detrend"]},
        {"model": "best", "transforms": ["SNV", "Detrend", "SavGol"]}
    ]
}
```

### 6. Using Presets for Reusable Patterns

```python
# Define presets
register_preset("standard_preprocessing", {
    "_or_": [
        {"class": "StandardScaler"},
        {"class": "MinMaxScaler"},
        None
    ]
})

register_preset("pls_search", {
    "_grid_": {
        "class": ["PLSRegression"],
        "n_components": {"_range_": [2, 20]}
    }
})

# Use in pipeline
config = [
    {"preprocessing": {"_preset_": "standard_preprocessing"}},
    {"model": {"_preset_": "pls_search"}}
]
```

### 7. Memory-Efficient Large Space Processing

```python
from itertools import islice

large_spec = {
    "_grid_": {
        "param1": {"_range_": [1, 100]},
        "param2": {"_range_": [1, 100]},
        "param3": {"_range_": [1, 100]}
    }
}

# Don't do this! (1M configurations in memory)
# all_configs = expand_spec(large_spec)

# Do this instead (lazy iteration)
for config in expand_spec_iter(large_spec):
    process(config)

# Or sample
sample = list(expand_spec_iter(large_spec, seed=42, sample_size=1000))
```

### 8. Preprocessing Pipeline with Cartesian

```python
# Generate all stage combinations, then select complete pipelines
{
    "_cartesian_": [
        # Stage 1: Scatter correction
        {"_or_": ["MSC", "SNV", "EMSC", None]},
        # Stage 2: Smoothing
        {"_or_": [
            {"class": "SavitzkyGolay", "window": 11},
            {"class": "Gaussian", "sigma": 2},
            None
        ]},
        # Stage 3: Derivative
        {"_or_": [
            {"class": "FirstDerivative"},
            {"class": "SecondDerivative"},
            None
        ]}
    ],
    "pick": (1, 3),  # Select 1-3 complete pipelines
    "count": 50       # Limit to 50 variants
}
```

### 9. Reproducible Random Search

```python
# Use _seed_ for reproducible random selection
{
    "_or_": [
        {"class": "PLS", "n_components": {"_range_": [2, 20]}},
        {"class": "RF", "n_estimators": {"_or_": [100, 200, 500]}},
        {"class": "SVR", "C": {"_log_range_": [0.1, 100, 10]}}
    ],
    "count": 10,
    "_seed_": 42  # Same 10 configs every time
}
```

---

## See Also

- {doc}`/examples/index` - Working examples organized by topic
- {doc}`/reference/pipeline_syntax` - Pipeline syntax reference
- {doc}`/reference/combination_generator` - Combination generator syntax

---

*Document updated: December 27, 2025*
*Version: Phase 4+ Complete*
