# Generator Module Analysis and Improvement Proposals

**Date**: December 8, 2025
**Status**: 🔍 Analysis Complete
**Module**: `nirs4all/pipeline/config/generator.py`

---

## Table of Contents

1. [Current Status and Functioning](#1-current-status-and-functioning)
2. [Known Issues, Bugs, and Edge Cases](#2-known-issues-bugs-and-edge-cases)
3. [Proposed Improvements, Extensions, and Fixes](#3-proposed-improvements-extensions-and-fixes)
4. [Modularization Strategy and Architecture](#4-modularization-strategy-and-architecture)
5. [New Features and API Proposals for Production](#5-new-features-and-api-proposals-for-production)

---

## 1. Current Status and Functioning

### 1.1 Overview

The generator module (`generator.py`) is responsible for expanding pipeline configuration specifications into concrete pipeline variants. It takes a single configuration with combinatorial keywords (`_or_`, `_range_`, `size`, `count`) and generates all possible combinations.

### 1.2 Core Functions

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `expand_spec(node)` | Main recursive expansion | Any node (dict/list/scalar) | List of expanded variants |
| `count_combinations(node)` | Calculate total without generating | Any node | Integer count |
| `_expand_value(v)` | Handle value-position expansion | Value from dict | List of expanded values |
| `_count_value(v)` | Count value-position combinations | Value from dict | Integer count |
| `_handle_nested_combinations(choices, nested_size)` | Second-order permutation/combination | Choices list, [outer, inner] | List of nested lists |
| `_count_nested_combinations(choices, nested_size)` | Count second-order | Choices list, [outer, inner] | Integer count |
| `_generate_range(range_spec)` | Generate numeric sequences | [from, to, step] or dict | List of numbers |
| `_count_range(range_spec)` | Count range elements | [from, to, step] or dict | Integer count |
| `_expand_combination(combo)` | Expand Cartesian product of choices | Tuple of choices | List of expanded lists |

### 1.3 Supported Syntax

#### 1.3.1 Basic `_or_` Choices
```python
{"_or_": ["A", "B", "C"]}
# → ["A", "B", "C"]
```

#### 1.3.2 `_or_` with `size` (Combinations)
```python
{"_or_": ["A", "B", "C"], "size": 2}
# → [["A", "B"], ["A", "C"], ["B", "C"]]

{"_or_": ["A", "B", "C"], "size": (1, 2)}  # Range of sizes
# → [["A"], ["B"], ["C"], ["A", "B"], ["A", "C"], ["B", "C"]]
```

#### 1.3.3 `_or_` with `count` (Random Sampling)
```python
{"_or_": ["A", "B", "C", "D"], "count": 2}
# → Random 2 from the choices
```

#### 1.3.4 Second-Order with `size=[outer, inner]` (Permutations inside, Combinations outside)
```python
{"_or_": ["A", "B", "C"], "size": [2, 2]}
# Inner: permutations of 2 elements (order matters)
# Outer: combinations of 2 inner arrangements
```

#### 1.3.5 Numeric Ranges with `_range_`
```python
{"_range_": [1, 10]}        # → [1, 2, ..., 10]
{"_range_": [0, 20, 5]}     # → [0, 5, 10, 15, 20]
{"_range_": {"from": 1, "to": 10, "step": 2}}  # Dict syntax
```

### 1.4 Data Flow

```
Pipeline Definition (Python/YAML/JSON)
           ↓
    PipelineConfigs.__init__()
           ↓
    serialize_component()  → Normalizes all objects to serializable form
           ↓
    _has_gen_keys()  → Checks for _or_ or _range_ keywords
           ↓
    count_combinations()  → Validates count doesn't exceed limit
           ↓
    expand_spec()  → Generates all concrete configurations
           ↓
    List[Dict] → Individual pipeline configurations
```

### 1.5 Integration Points

The generator is used by:
- `PipelineConfigs` class in `pipeline_config.py`
- Controllers: `FeatureAugmentationController`, `ConcatAugmentationController`, `SampleAugmentationController`
- Step types: `feature_augmentation`, `concat_transform`, `sample_augmentation`, `y_processing`

### 1.6 Current Test Coverage

The generator has **minimal test coverage**:
- `tests/unit/pipeline/config/test_generator.py` contains only a placeholder
- Some serialization tests in `test_serialization.py` exercise `_or_` and `_range_` syntax
- Examples serve as integration tests

---

## 2. Known Issues, Bugs, and Edge Cases

### 2.1 Critical Issues

#### 2.1.1 ❌ `count=1` Edge Case Inconsistency

**Problem**: When `count=1`, the behavior varies inconsistently depending on context.

```python
# Basic _or_ with count=1 - returns unwrapped item
{"_or_": ["A", "B", "C"], "count": 1}
# → ["B"] (single item, but random each time)

# With size=2 and count=1 - returns wrapped list
{"_or_": ["A", "B", "C"], "size": 2, "count": 1}
# → [["B", "C"]] (list containing the single combination)
```

**Impact**: Users expecting deterministic behavior for `count=1` get random results. No seed parameter exists.

**Root Cause**: `random.sample()` is used without a seed, making results non-reproducible.

#### 2.1.2 ⚠️ Non-Deterministic Generation

**Problem**: The `count` parameter uses `random.sample()` without any seed, making pipeline generation non-reproducible.

```python
# Running the same config twice gives different results:
{"_or_": ["A", "B", "C", "D"], "count": 2}
# Run 1: ["A", "D"]
# Run 2: ["B", "C"]
```

**Impact**: Cannot reproduce experiments, debugging is difficult, hash uniqueness is compromised.

#### 2.1.3 ⚠️ Empty `_or_` Returns Empty List

```python
{"_or_": []}
# → [] (empty list, count=0)
```

**Impact**: Downstream code may not handle empty pipeline lists gracefully.

### 2.2 Structural Issues

#### 2.2.1 Monolithic Code Structure

The file is 509 lines with multiple responsibilities:
- Recursive expansion logic
- Counting logic (duplicates expansion logic)
- Range generation
- Nested combination handling
- Value-position handling

**Problems**:
- DRY violation between `expand_spec` and `count_combinations`
- Hard to test individual components
- Complex branching logic (multiple if/elif chains)

#### 2.2.2 No Type Annotations

The module lacks type hints, making it hard to understand expected inputs/outputs:

```python
# Current
def expand_spec(node):
    ...

# Should be
def expand_spec(node: Union[Dict, List, Any]) -> List[Any]:
    ...
```

#### 2.2.3 Missing Docstrings

Several internal functions lack documentation:
- `_expand_combination`
- `_expand_value`
- `_count_value`

### 2.3 Behavioral Edge Cases

#### 2.3.1 `size=0` Generates Empty List Wrapper

```python
{"_or_": ["A", "B", "C"], "size": 0}
# → [[]] (list containing empty list)
```

**Question**: Is this intentional? Should `size=0` return `[]` instead?

#### 2.3.2 `count=0` Returns Empty List

```python
{"_or_": ["A", "B", "C"], "count": 0}
# → []
```

**Impact**: May cause issues if downstream code expects at least one configuration.

#### 2.3.3 Mixed Types in `_or_` Work but Are Risky

```python
{"_or_": ["string", 42, {"class": "MyClass"}, ["nested", "list"]]}
# → All items returned as-is
```

**Problem**: No type validation. May cause runtime errors when expanded values are used.

#### 2.3.4 Nested `_or_` Inside `_or_` Flattens

```python
{"_or_": [{"_or_": ["A1", "A2"]}, "B"]}
# → ["A1", "A2", "B"] (inner _or_ expanded and flattened)
```

**Question**: Is this the expected behavior? Users might expect hierarchical selection.

### 2.4 Missing Features

#### 2.4.1 No Support for Step-Type-Specific Keywords

The generator is generic and doesn't understand step types like:
- `concat_transform`
- `feature_augmentation`
- `sample_augmentation`
- `split`
- `model`

This means all step types are treated identically during expansion, which may not be semantically correct.

#### 2.4.2 No Validation of Generated Configurations

The generator doesn't validate that generated configurations are valid:
- No schema validation
- No step type validation
- No parameter validation

#### 2.4.3 No Support for Weighted Choices

No way to specify that some choices should be more likely than others:
```python
# Not supported:
{"_or_": ["A", "B", "C"], "weights": [0.5, 0.3, 0.2]}
```

#### 2.4.4 No Support for Exclusion Rules

No way to specify that certain combinations should be excluded:
```python
# Not supported:
{"_or_": ["A", "B", "C"], "exclude": [["A", "C"]]}
```

#### 2.4.5 No Lazy/Iterator Generation

All combinations are generated eagerly in memory:
```python
# For large spaces, this uses lots of memory:
{"_or_": [...100 items...], "size": (1, 5)}
# Generates all C(100,1) + C(100,2) + ... + C(100,5) = 79,375,496 items
```

### 2.5 API Design Issues

#### 2.5.1 Inconsistent Return Types

| Input | Output |
|-------|--------|
| Scalar | `[scalar]` |
| `{"_or_": [...]}` | `[item1, item2, ...]` |
| `{"_or_": [...], "size": 2}` | `[[item1, item2], [item1, item3], ...]` |
| List | `[[combo1], [combo2], ...]` (Cartesian product) |

The output structure changes based on input type, making it hard to process uniformly.

#### 2.5.2 Magic Keywords Not Documented in Code

The keywords `_or_`, `_range_`, `size`, `count` are scattered throughout:
```python
# No central definition of keywords
if "_or_" in node:
    ...
if set(node.keys()).issubset({"_or_", "size", "count"}):
    ...
```

---

## 3. Proposed Improvements, Extensions, and Fixes

### 3.1 Critical Fixes (Priority: High)

#### 3.1.1 Add Deterministic Mode with Seed Support

```python
def expand_spec(node, seed: Optional[int] = None) -> List[Any]:
    """
    Expand specification to all combinations.

    Args:
        node: Specification to expand
        seed: Random seed for reproducible sampling when using 'count'
              If None, uses current random state (non-deterministic)
    """
    if seed is not None:
        random.seed(seed)
    ...
```

**Alternative**: Use a `_seed_` keyword in the spec:
```python
{"_or_": ["A", "B", "C"], "count": 2, "_seed_": 42}
```

#### 3.1.2 Consistent Return Types

Normalize all outputs to the same structure:

```python
# Option A: Always return list of configurations
expand_spec({"_or_": ["A", "B"]})          → [{"value": "A"}, {"value": "B"}]
expand_spec({"_or_": ["A", "B"], "size": 1}) → [{"value": ["A"]}, {"value": ["B"]}]

# Option B: Always return raw values (current behavior, but documented)
# Document clearly when nesting levels change
```

#### 3.1.3 Validate Empty Inputs

```python
def expand_spec(node):
    if isinstance(node, dict) and "_or_" in node:
        choices = node["_or_"]
        if not choices:
            raise ValueError("_or_ cannot have empty choices list")
```

### 3.2 Code Quality Improvements (Priority: Medium)

#### 3.2.1 Add Type Annotations

```python
from typing import Any, Dict, List, Union, Optional, Tuple

GeneratorNode = Union[Dict[str, Any], List[Any], str, int, float, bool, None]
ExpandedResult = List[Any]

def expand_spec(node: GeneratorNode, seed: Optional[int] = None) -> ExpandedResult:
    """..."""

def count_combinations(node: GeneratorNode) -> int:
    """..."""
```

#### 3.2.2 Add Comprehensive Docstrings

Every function should have:
- Purpose description
- Args with types and descriptions
- Returns description
- Examples
- Raises description

#### 3.2.3 Extract Constants

```python
# At module level
GENERATOR_KEYWORDS = frozenset({"_or_", "_range_", "size", "count", "_seed_"})
PURE_OR_KEYS = frozenset({"_or_", "size", "count"})
PURE_RANGE_KEYS = frozenset({"_range_", "count"})
```

### 3.3 New Features (Priority: Medium)

#### 3.3.1 Weighted Random Selection

```python
{"_or_": ["A", "B", "C"], "_weights_": [0.5, 0.3, 0.2], "count": 2}
# "A" is 50% likely, "B" is 30%, "C" is 20%
```

Implementation:
```python
if "_weights_" in node:
    weights = node["_weights_"]
    selected = random.choices(choices, weights=weights, k=count)
```

#### 3.3.2 Exclusion Rules

```python
{"_or_": ["A", "B", "C", "D"], "size": 2, "_exclude_": [["A", "B"], ["C", "D"]]}
# Skip combinations ["A", "B"] and ["C", "D"]
```

#### 3.3.3 Conditional Expansion

```python
{
    "_or_": ["PCA", "SVD"],
    "_when_": {"preprocessing": "StandardScaler"}  # Only expand when condition met
}
```

#### 3.3.4 Float Range Support

```python
{"_range_": [0.1, 1.0, 0.1]}  # Currently only integers supported
# → [0.1, 0.2, 0.3, ..., 1.0]
```

#### 3.3.5 Logarithmic Range

```python
{"_log_range_": [1, 1000, 10]}  # Logarithmic scale
# → [1, 10, 100, 1000]

{"_log_range_": {"from": 0.001, "to": 1, "base": 10}}
# → [0.001, 0.01, 0.1, 1]
```

### 3.4 Performance Improvements (Priority: Low)

#### 3.4.1 Lazy/Iterator-Based Generation

```python
def expand_spec_iter(node) -> Iterator[Any]:
    """Lazy generator for memory-efficient expansion."""
    if isinstance(node, list):
        for combo in product(*[expand_spec_iter(elem) for elem in node]):
            yield list(combo)
    ...

# Usage:
for config in expand_spec_iter(large_spec):
    # Process one at a time, memory efficient
    run_pipeline(config)
```

#### 3.4.2 Memoization for Repeated Nodes

```python
@functools.lru_cache(maxsize=1024)
def _cached_expand(frozen_node):
    return expand_spec(frozen_node)
```

#### 3.4.3 Parallel Expansion

For very large spaces, expand in parallel:
```python
from concurrent.futures import ProcessPoolExecutor

def parallel_expand(nodes: List[Dict]) -> List[List[Any]]:
    with ProcessPoolExecutor() as executor:
        return list(executor.map(expand_spec, nodes))
```

### 3.5 Validation Features (Priority: Medium)

#### 3.5.1 Schema Validation for Generated Configs

```python
def validate_generated_config(config: Dict, schema: Dict) -> bool:
    """Validate that generated config matches expected schema."""
    ...

# Integration with PipelineConfigs:
expanded = expand_spec(node)
for config in expanded:
    if not validate_generated_config(config, PIPELINE_SCHEMA):
        raise InvalidConfigError(f"Generated config is invalid: {config}")
```

#### 3.5.2 Semantic Validation Hooks

```python
def expand_spec(node, validator: Optional[Callable] = None):
    """
    Args:
        validator: Optional function(config) -> bool to filter invalid configs
    """
    results = _expand_internal(node)
    if validator:
        results = [r for r in results if validator(r)]
    return results
```

---

## 4. Modularization Strategy and Architecture

### 4.1 Current Structure Problems

```
generator.py (509 lines)
├── expand_spec()           - Main expansion logic
├── count_combinations()    - Counting logic (duplicates expand_spec pattern)
├── _handle_nested_combinations() - Second-order logic
├── _count_nested_combinations()  - Duplicates nested logic
├── _expand_combination()   - Cartesian product helper
├── _expand_value()         - Value-position expansion
├── _count_value()          - Value-position counting
├── _generate_range()       - Range generation
└── _count_range()          - Range counting
```

**Issues**:
1. DRY violations: `expand_*` and `count_*` functions duplicate logic
2. Single file with multiple responsibilities
3. No clear separation between generation strategies
4. Hardcoded keyword handling

### 4.2 Proposed Module Structure

```
nirs4all/pipeline/config/
├── __init__.py
├── generator/
│   ├── __init__.py              # Public API exports
│   ├── core.py                  # Main expand_spec and count_combinations
│   ├── keywords.py              # Keyword constants and detection
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base strategy
│   │   ├── or_strategy.py       # _or_ handling
│   │   ├── range_strategy.py    # _range_ handling
│   │   ├── nested_strategy.py   # Second-order combinations
│   │   └── registry.py          # Strategy registry
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── combinatorics.py     # combinations, permutations, etc.
│   │   └── sampling.py          # Random sampling with seed support
│   └── validators/
│       ├── __init__.py
│       └── schema.py            # Generated config validation
├── component_serialization.py
├── context.py
└── pipeline_config.py
```

### 4.3 Strategy Pattern for Keywords

#### 4.3.1 Abstract Base Strategy

```python
# generator/strategies/base.py
from abc import ABC, abstractmethod
from typing import Any, List, Optional

class ExpansionStrategy(ABC):
    """Base class for expansion strategies."""

    @property
    @abstractmethod
    def keywords(self) -> frozenset:
        """Keywords this strategy handles."""
        pass

    @abstractmethod
    def matches(self, node: dict) -> bool:
        """Check if this strategy should handle the node."""
        pass

    @abstractmethod
    def expand(self, node: dict, seed: Optional[int] = None) -> List[Any]:
        """Expand the node to all combinations."""
        pass

    @abstractmethod
    def count(self, node: dict) -> int:
        """Count combinations without generating."""
        pass
```

#### 4.3.2 OR Strategy Implementation

```python
# generator/strategies/or_strategy.py
from .base import ExpansionStrategy

class OrStrategy(ExpansionStrategy):
    """Handles _or_ keyword expansion."""

    @property
    def keywords(self) -> frozenset:
        return frozenset({"_or_", "size", "count"})

    def matches(self, node: dict) -> bool:
        return "_or_" in node and set(node.keys()).issubset(self.keywords)

    def expand(self, node: dict, seed: Optional[int] = None) -> List[Any]:
        choices = node["_or_"]
        size = node.get("size")
        count = node.get("count")

        if size is not None:
            return self._expand_with_size(choices, size, count, seed)
        return self._expand_simple(choices, count, seed)

    def count(self, node: dict) -> int:
        choices = node["_or_"]
        size = node.get("size")
        count_limit = node.get("count")

        total = self._count_internal(choices, size)
        return min(total, count_limit) if count_limit else total
```

#### 4.3.3 Strategy Registry

```python
# generator/strategies/registry.py
from typing import List, Type
from .base import ExpansionStrategy
from .or_strategy import OrStrategy
from .range_strategy import RangeStrategy
from .nested_strategy import NestedStrategy

STRATEGY_REGISTRY: List[Type[ExpansionStrategy]] = [
    OrStrategy,
    RangeStrategy,
    NestedStrategy,
]

def get_strategy(node: dict) -> ExpansionStrategy:
    """Find matching strategy for a node."""
    for strategy_cls in STRATEGY_REGISTRY:
        strategy = strategy_cls()
        if strategy.matches(node):
            return strategy
    return None
```

### 4.4 Core Module Refactoring

#### 4.4.1 Simplified Core

```python
# generator/core.py
from typing import Any, List, Optional, Union
from .strategies.registry import get_strategy
from .keywords import is_generator_node

def expand_spec(
    node: Union[dict, list, Any],
    seed: Optional[int] = None
) -> List[Any]:
    """
    Expand a specification node to all possible combinations.

    Args:
        node: Specification to expand (dict, list, or scalar)
        seed: Random seed for reproducible sampling

    Returns:
        List of all expanded combinations
    """
    # Handle lists
    if isinstance(node, list):
        return _expand_list(node, seed)

    # Handle non-dict scalars
    if not isinstance(node, dict):
        return [node]

    # Find matching strategy
    strategy = get_strategy(node)
    if strategy:
        return strategy.expand(node, seed)

    # Default dict expansion (Cartesian product of keys)
    return _expand_dict(node, seed)

def count_combinations(node: Union[dict, list, Any]) -> int:
    """
    Count total combinations without generating them.

    Args:
        node: Specification to count

    Returns:
        Total number of combinations
    """
    if isinstance(node, list):
        return _count_list(node)

    if not isinstance(node, dict):
        return 1

    strategy = get_strategy(node)
    if strategy:
        return strategy.count(node)

    return _count_dict(node)
```

### 4.5 Keywords Module

```python
# generator/keywords.py
"""Generator keyword definitions and utilities."""

# Core generation keywords
OR_KEYWORD = "_or_"
RANGE_KEYWORD = "_range_"
LOG_RANGE_KEYWORD = "_log_range_"  # Future

# Modifier keywords
SIZE_KEYWORD = "size"
COUNT_KEYWORD = "count"
SEED_KEYWORD = "_seed_"
WEIGHTS_KEYWORD = "_weights_"
EXCLUDE_KEYWORD = "_exclude_"

# Keyword groups
GENERATION_KEYWORDS = frozenset({OR_KEYWORD, RANGE_KEYWORD, LOG_RANGE_KEYWORD})
MODIFIER_KEYWORDS = frozenset({SIZE_KEYWORD, COUNT_KEYWORD, SEED_KEYWORD, WEIGHTS_KEYWORD, EXCLUDE_KEYWORD})
ALL_KEYWORDS = GENERATION_KEYWORDS | MODIFIER_KEYWORDS

def is_generator_node(node: dict) -> bool:
    """Check if a dict node contains any generator keywords."""
    if not isinstance(node, dict):
        return False
    return bool(GENERATION_KEYWORDS & set(node.keys()))

def extract_modifiers(node: dict) -> dict:
    """Extract modifier values from a node."""
    return {k: node[k] for k in MODIFIER_KEYWORDS if k in node}

def extract_base_node(node: dict) -> dict:
    """Extract non-keyword keys from a node."""
    return {k: v for k, v in node.items() if k not in ALL_KEYWORDS}
```

### 4.6 Utilities Module

```python
# generator/utils/sampling.py
import random
from typing import List, TypeVar, Optional

T = TypeVar('T')

def sample_with_seed(
    population: List[T],
    k: int,
    seed: Optional[int] = None,
    weights: Optional[List[float]] = None
) -> List[T]:
    """
    Sample from population with optional seed and weights.

    Args:
        population: Items to sample from
        k: Number of items to sample
        seed: Random seed for reproducibility
        weights: Optional weights for weighted sampling

    Returns:
        Sampled items
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    if weights:
        return rng.choices(population, weights=weights, k=k)
    return rng.sample(population, k=min(k, len(population)))
```

### 4.7 Migration Path

1. **Phase 1**: Extract keywords and utilities (non-breaking)
2. **Phase 2**: Implement strategy pattern alongside existing code
3. **Phase 3**: Migrate expand_spec to use strategies
4. **Phase 4**: Remove old code
5. **Phase 5**: Add new features using strategy pattern

---

## 5. New Features and API Proposals for Production

### 5.1 Production-Ready API Design

#### 5.1.1 High-Level API

```python
from nirs4all.pipeline.config import Generator

# Create generator with configuration
gen = Generator(
    seed=42,                    # Global seed for reproducibility
    max_combinations=10000,     # Safety limit
    validate=True,              # Enable schema validation
    lazy=False                  # Eager vs lazy generation
)

# Generate configurations
configs = gen.expand(pipeline_spec)

# Preview without generating
preview = gen.preview(pipeline_spec)
print(f"Would generate {preview.count} configurations")
print(f"Estimated memory: {preview.memory_estimate_mb} MB")

# Lazy iteration for large spaces
for config in gen.iterate(pipeline_spec):
    run_pipeline(config)
```

#### 5.1.2 Preview Object

```python
@dataclass
class ExpansionPreview:
    """Preview of what expansion would generate."""
    count: int
    keywords_used: Set[str]
    depth: int
    has_random_sampling: bool
    memory_estimate_mb: float
    estimated_time_seconds: float
    warnings: List[str]

    def is_safe(self, max_count: int = 10000) -> bool:
        return self.count <= max_count
```

### 5.2 New Generation Keywords

#### 5.2.1 `_grid_` - Grid Search Style

```python
{
    "_grid_": {
        "n_components": [5, 10, 15],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5, 7]
    }
}
# Generates 3 × 2 × 3 = 18 configurations with all parameter combinations
```

#### 5.2.2 `_zip_` - Parallel Iteration

```python
{
    "_zip_": {
        "name": ["model_a", "model_b", "model_c"],
        "n_components": [5, 10, 15]
    }
}
# Generates 3 configurations: (model_a, 5), (model_b, 10), (model_c, 15)
```

#### 5.2.3 `_chain_` - Sequential Choices (Ordered Pipeline Variations)

```python
{
    "_chain_": [
        {"preprocessing": "A"},
        {"preprocessing": "B"},
        {"preprocessing": "C"}
    ]
}
# Generates 3 configurations in order (unlike _or_ which is unordered)
```

#### 5.2.4 `_sample_` - Statistical Sampling

```python
{
    "_sample_": {
        "distribution": "uniform",
        "min": 0.001,
        "max": 1.0,
        "n": 10
    }
}
# Generates 10 values uniformly sampled from [0.001, 1.0]

{
    "_sample_": {
        "distribution": "log_uniform",
        "min": 0.0001,
        "max": 1.0,
        "n": 20
    }
}
# Log-uniform sampling (common for learning rates)
```

#### 5.2.5 `_each_` - Apply to Each in Context

```python
{
    "feature_augmentation": {
        "_each_": ["SNV", "Detrend", "FirstDerivative"],
        "_apply_": {"window_size": {"_range_": [3, 11, 2]}}
    }
}
# Generates: SNV with window 3,5,7,9,11 + Detrend with window 3,5,7,9,11 + ...
```

#### 5.2.6 `_conditional_` - Dependent Expansion

```python
{
    "model": {"_or_": ["PLS", "RandomForest"]},
    "n_components": {
        "_conditional_": {
            "when": {"model": "PLS"},
            "then": {"_range_": [5, 20]},
            "else": None  # Don't generate n_components for RF
        }
    }
}
```

### 5.3 Tagging and Metadata System

#### 5.3.1 Configuration Tags

```python
{
    "_or_": [
        {"_tag_": "baseline", "preprocessing": None},
        {"_tag_": "standard", "preprocessing": "StandardScaler"},
        {"_tag_": "robust", "preprocessing": "RobustScaler"}
    ]
}

# Generated configs include metadata:
[
    {"preprocessing": None, "_metadata_": {"tags": ["baseline"]}},
    {"preprocessing": "StandardScaler", "_metadata_": {"tags": ["standard"]}},
    ...
]
```

#### 5.3.2 Hierarchical Tags

```python
{
    "_tags_": ["experiment:v2", "category:preprocessing"],
    "_or_": [
        {"_tag_": "method:snv", "transform": "SNV"},
        {"_tag_": "method:msc", "transform": "MSC"}
    ]
}
# Tags are inherited: ["experiment:v2", "category:preprocessing", "method:snv"]
```

#### 5.3.3 Filtering by Tags

```python
# After generation
all_configs = gen.expand(spec)
snv_only = gen.filter_by_tag(all_configs, "method:snv")
baseline_configs = gen.filter_by_tag(all_configs, "baseline")
```

### 5.4 Constraint System

#### 5.4.1 Mutual Exclusion

```python
{
    "_or_": ["A", "B", "C", "D"],
    "size": 2,
    "_mutex_": [["A", "B"], ["C", "D"]]  # A and B can't be together, same for C and D
}
# Generates: [A,C], [A,D], [B,C], [B,D] - but not [A,B] or [C,D]
```

#### 5.4.2 Required Combinations

```python
{
    "_or_": ["A", "B", "C", "D"],
    "size": 3,
    "_requires_": [["A", "B"]]  # If A is selected, B must also be selected
}
# Only generates combinations that include both A and B, or neither
```

#### 5.4.3 Dependency Constraints

```python
{
    "preprocessing": {"_or_": ["Standard", "Robust", "None"]},
    "model": {
        "_or_": ["PLS", "CNN"],
        "_depends_on_": {
            "CNN": {"preprocessing": ["Standard", "Robust"]}  # CNN requires preprocessing
        }
    }
}
# Won't generate (preprocessing=None, model=CNN)
```

### 5.5 Named Configurations (Presets)

```python
# Define named presets
PRESETS = {
    "spectral_preprocessing": {
        "_or_": [SNV, MSC, Detrend, FirstDerivative],
        "size": (1, 2)
    },
    "standard_models": {
        "_or_": [PLSRegression, RandomForest, GradientBoosting]
    }
}

# Use in pipeline
pipeline = [
    {"feature_augmentation": "_preset_:spectral_preprocessing"},
    ShuffleSplit(n_splits=5),
    {"model": "_preset_:standard_models"}
]
```

### 5.6 Integration with Step Types

#### 5.6.1 Step-Aware Expansion

```python
class StepAwareGenerator:
    """Generator that understands pipeline step semantics."""

    STEP_CONSTRAINTS = {
        "concat_transform": {
            "min_items": 1,
            "max_items": 10,
            "item_types": ["transformer"]
        },
        "feature_augmentation": {
            "min_items": 1,
            "item_types": ["transformer", "concat_transform"]
        },
        "model": {
            "count": 1,  # Exactly one model per pipeline
        }
    }

    def expand_step(self, step: dict, step_type: str) -> List[dict]:
        """Expand a step with type-specific constraints."""
        constraints = self.STEP_CONSTRAINTS.get(step_type, {})
        return self._expand_with_constraints(step, constraints)
```

#### 5.6.2 Validation per Step Type

```python
# Validate that concat_transform contains only transformers
{
    "concat_transform": {
        "_or_": [PCA, SVD, LDA],  # Valid: all are transformers
        "size": 2,
        "_validate_": "transformer_only"
    }
}
```

### 5.7 Export and Inspection

#### 5.7.1 Expansion Tree Visualization

```python
gen = Generator()
tree = gen.get_expansion_tree(spec)
tree.print()

# Output:
# ├── _or_ [3 choices]
# │   ├── "A"
# │   ├── "B"
# │   └── "C"
# └── params
#     └── n_components: _range_ [1, 10] → 10 values
#
# Total: 30 combinations
```

#### 5.7.2 Export to DataFrame

```python
import pandas as pd

configs = gen.expand(spec)
df = gen.to_dataframe(configs)

#    preprocessing  n_components  model
# 0  StandardScaler           5    PLS
# 1  StandardScaler          10    PLS
# 2  StandardScaler           5    RF
# ...
```

#### 5.7.3 Configuration Diff

```python
config1, config2 = configs[0], configs[1]
diff = gen.diff(config1, config2)
# → {"n_components": (5, 10)}  # Shows only differences
```

### 5.8 Error Handling and Recovery

#### 5.8.1 Graceful Degradation

```python
gen = Generator(
    on_invalid="skip",  # or "raise", "warn"
    max_retries=3
)

# If a generated config is invalid, skip it instead of failing
```

#### 5.8.2 Partial Expansion

```python
# For very large spaces, expand partially
partial_configs = gen.expand(spec, limit=100, strategy="random")
# Get 100 random configs from the full space
```

#### 5.8.3 Checkpoint and Resume

```python
# For long-running generations
gen.expand(
    spec,
    checkpoint_file="generation_checkpoint.json",
    checkpoint_interval=1000
)

# Resume from checkpoint
gen.resume("generation_checkpoint.json")
```

---

## 6. Testing Strategy

### 6.1 Unit Tests Required

```python
# tests/unit/pipeline/config/generator/
├── test_core.py              # expand_spec, count_combinations
├── test_or_strategy.py       # _or_ keyword tests
├── test_range_strategy.py    # _range_ keyword tests
├── test_nested_strategy.py   # Second-order tests
├── test_keywords.py          # Keyword detection
├── test_sampling.py          # Random sampling with seeds
├── test_edge_cases.py        # Empty inputs, size=0, count=0, etc.
└── test_integration.py       # Full pipeline expansion tests
```

### 6.2 Property-Based Testing

```python
from hypothesis import given, strategies as st

@given(st.lists(st.text(), min_size=1, max_size=10))
def test_expand_or_returns_all_choices(choices):
    spec = {"_or_": choices}
    result = expand_spec(spec)
    assert set(result) == set(choices)

@given(st.integers(min_value=1, max_value=100))
def test_range_count_matches_expand_length(end):
    spec = {"_range_": [1, end]}
    assert count_combinations(spec) == len(expand_spec(spec))
```

### 6.3 Regression Tests

```python
# Test known edge cases that have caused issues
def test_count_equals_one():
    """count=1 should return exactly one result."""
    spec = {"_or_": ["A", "B", "C"], "count": 1}
    result = expand_spec(spec, seed=42)
    assert len(result) == 1
    # With same seed, should be deterministic
    result2 = expand_spec(spec, seed=42)
    assert result == result2
```

---

## 7. Implementation Roadmap

### Phase 1: Critical Fixes (Week 1-2)
- [ ] Add seed support to `expand_spec`
- [ ] Add type annotations
- [ ] Add comprehensive docstrings
- [ ] Create test suite for edge cases

### Phase 2: Modularization (Week 3-4)
- [ ] Extract keywords.py
- [ ] Extract utils/ module
- [ ] Implement strategy pattern
- [ ] Migrate existing code to strategies

### Phase 3: New Features (Week 5-8)
- [ ] Add `_log_range_` support
- [ ] Add weighted sampling (`_weights_`)
- [ ] Add exclusion rules (`_exclude_`)
- [ ] Add tagging system

### Phase 4: Production Features (Week 9-12)
- [ ] Add lazy/iterator generation
- [ ] Add constraint system
- [ ] Add preset system
- [ ] Add export/visualization tools

### Phase 5: Documentation & Examples (Week 13-14)
- [ ] Update reference documentation
- [ ] Create tutorial examples
- [ ] Add migration guide

---

## 8. Summary

The generator module is functional but has grown organically into a monolithic file with several issues:

**Critical Issues**:
1. Non-deterministic behavior with `count` parameter
2. Inconsistent return types
3. DRY violations between expand and count functions
4. Missing type safety

**Recommended Immediate Actions**:
1. Add `seed` parameter for reproducibility
2. Add type annotations
3. Create comprehensive test suite
4. Document edge cases

**Long-term Vision**:
- Modular, extensible architecture using strategy pattern
- Rich keyword vocabulary for complex generation patterns
- Production-ready API with validation, constraints, and tagging
- Memory-efficient lazy generation for large spaces

---

*Document created: December 8, 2025*
*Author: GitHub Copilot Analysis*


