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

The generator module (`generator.py`) is responsible for expanding pipeline configuration specifications into concrete pipeline variants. It takes a single configuration with combinatorial keywords (`_or_`, `_range_`, `pick`, `count`) and generates all possible combinations.

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

#### 1.3.2 `_or_` with `pick` (Combinations)
```python
{"_or_": ["A", "B", "C"], "pick": 2}
# → [["A", "B"], ["A", "C"], ["B", "C"]]

{"_or_": ["A", "B", "C"], "pick": (1, 2)}  # Range of picks
# → [["A"], ["B"], ["C"], ["A", "B"], ["A", "C"], ["B", "C"]]
```

#### 1.3.3 `_or_` with `count` (Random Sampling)
```python
{"_or_": ["A", "B", "C", "D"], "count": 2}
# → Random 2 from the choices
```

#### 1.3.4 Second-Order with `pick=[outer, inner]` (Permutations inside, Combinations outside)
```python
{"_or_": ["A", "B", "C"], "pick": [2, 2]}
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

# With pick=2 and count=1 - returns wrapped list
{"_or_": ["A", "B", "C"], "pick": 2, "count": 1}
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

#### 2.3.1 `pick=0` Generates Empty List Wrapper

```python
{"_or_": ["A", "B", "C"], "pick": 0}
# → [[]] (list containing empty list)
```

**Question**: Is this intentional? Should `pick=0` return `[]` instead?

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
{"_or_": [...100 items...], "pick": (1, 5)}
# Generates all C(100,1) + C(100,2) + ... + C(100,5) = 79,375,496 items
```

### 2.5 API Design Issues

#### 2.5.1 Inconsistent Return Types

| Input | Output |
|-------|--------|
| Scalar | `[scalar]` |
| `{"_or_": [...]}` | `[item1, item2, ...]` |
| `{"_or_": [...], "pick": 2}` | `[[item1, item2], [item1, item3], ...]` |
| List | `[[combo1], [combo2], ...]` (Cartesian product) |

The output structure changes based on input type, making it hard to process uniformly.

#### 2.5.2 Magic Keywords Not Documented in Code

The keywords `_or_`, `_range_`, `pick`, `count` are scattered throughout:
```python
# No central definition of keywords
if "_or_" in node:
    ...
if set(node.keys()).issubset({"_or_", "pick", "count"}):
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
expand_spec({"_or_": ["A", "B"], "pick": 1}) → [{"value": ["A"]}, {"value": ["B"]}]

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
GENERATOR_KEYWORDS = frozenset({"_or_", "_range_", "pick", "count", "_seed_"})
PURE_OR_KEYS = frozenset({"_or_", "pick", "count"})
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
{"_or_": ["A", "B", "C", "D"], "pick": 2, "_exclude_": [["A", "B"], ["C", "D"]]}
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

### 4.1 Phase 1.5 Completion Status ✅

Phase 1.5 (Selection Semantics) has been fully implemented. The following was delivered:

**Implemented Features:**
- ✅ `pick` keyword for combinations (unordered selection)
- ✅ `arrange` keyword for permutations (ordered arrangement)
- ✅ `then_pick` and `then_arrange` for second-order operations
- ✅ Modular `_generator/` subpackage structure started
- ✅ `keywords.py` with centralized constants and utilities
- ✅ `utils/sampling.py` with seed-aware random functions
- ✅ `utils/combinatorics.py` with helper functions
- ✅ Comprehensive test coverage for pick/arrange

**Current Structure (Post Phase 1.5):**

```
nirs4all/pipeline/config/
├── generator.py                 # Main module (905 lines - still monolithic)
└── _generator/
    ├── __init__.py              # Package exports
    ├── keywords.py              # Keyword constants and detection ✅
    └── utils/
        ├── __init__.py          # Utility exports ✅
        ├── sampling.py          # Random sampling with seed ✅
        └── combinatorics.py     # Combination/permutation helpers ✅
```

### 4.2 Current Structure Analysis

While Phase 1.5 created the foundation, `generator.py` remains monolithic at ~900 lines:

```
generator.py (905 lines)
├── expand_spec()                      # Main expansion (lines 47-135)
├── count_combinations()               # Counting logic (lines 138-209)
├── _expand_with_pick()                # Pick expansion (lines 212-266)
├── _expand_with_arrange()             # Arrange expansion (lines 269-319)
├── _handle_nested_arrangements()      # Nested with permutations (lines 322-361)
├── _handle_nested_combinations()      # Nested with combinations (lines 364-405)
├── _normalize_spec()                  # Spec normalization (lines 408-415)
├── _handle_pick_then_pick()           # Second-order pick→pick (lines 418-449)
├── _handle_pick_then_arrange()        # Second-order pick→arrange (lines 452-483)
├── _handle_arrange_then_pick()        # Second-order arrange→pick (lines 486-517)
├── _handle_arrange_then_arrange()     # Second-order arrange→arrange (lines 520-551)
├── _count_nested_combinations()       # Count nested (lines 554-582)
├── _count_with_pick()                 # Count pick (lines 585-620)
├── _count_with_arrange()              # Count arrange (lines 623-658)
├── _count_nested_arrangements()       # Count nested arrange (lines 661-693)
├── _count_pick_then_*()               # Count second-order (lines 696-789)
├── _expand_combination()              # Cartesian product (lines 792-805)
├── _expand_value()                    # Value-position expansion (lines 808-845)
├── _count_value()                     # Value-position counting (lines 848-859)
├── _generate_range()                  # Range generation (lines 862-886)
└── _count_range()                     # Range counting (lines 889-905)
```

**Remaining Issues:**
1. ❌ DRY violations: `_expand_*` and `_count_*` duplicate logic patterns
2. ❌ No strategy pattern for keyword handling
3. ❌ 30+ functions in single file
4. ❌ Complex branching in `expand_spec` and `count_combinations`
5. ⚠️ Utils exist but not fully utilized by main module

### 4.3 Phase 2 Target Module Structure

```
nirs4all/pipeline/config/
├── __init__.py
├── generator.py                      # Thin wrapper - public API only (~100 lines)
└── _generator/
    ├── __init__.py                   # Package exports
    ├── core.py                       # NEW: Main expand/count orchestration
    ├── keywords.py                   # ✅ Exists - keyword constants
    ├── strategies/                   # NEW: Strategy pattern implementation
    │   ├── __init__.py
    │   ├── base.py                   # Abstract base strategy
    │   ├── or_strategy.py            # _or_ keyword handling
    │   ├── range_strategy.py         # _range_ keyword handling
    │   ├── pick_strategy.py          # pick/arrange and second-order
    │   └── registry.py               # Strategy discovery and dispatch
    ├── utils/                        # ✅ Exists - helper utilities
    │   ├── __init__.py
    │   ├── combinatorics.py          # ✅ Exists - math helpers
    │   └── sampling.py               # ✅ Exists - random with seed
    └── validators/                   # NEW: Validation layer
        ├── __init__.py
        └── schema.py                 # Config validation
```

### 4.4 Strategy Pattern Implementation

#### 4.4.1 Abstract Base Strategy

```python
# _generator/strategies/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

SizeSpec = Union[int, Tuple[int, int], List[int]]

class ExpansionStrategy(ABC):
    """Base class for generator expansion strategies.

    Each strategy handles a specific type of generator node (e.g., _or_, _range_).
    Strategies are responsible for both expansion and counting.
    """

    @property
    @abstractmethod
    def primary_keyword(self) -> str:
        """The main keyword this strategy handles (e.g., '_or_', '_range_')."""
        pass

    @property
    @abstractmethod
    def allowed_keys(self) -> FrozenSet[str]:
        """All keys allowed in nodes this strategy handles."""
        pass

    @abstractmethod
    def matches(self, node: Dict[str, Any]) -> bool:
        """Check if this strategy should handle the given node.

        Args:
            node: A dictionary node from the configuration.

        Returns:
            True if this strategy can handle the node.
        """
        pass

    @abstractmethod
    def expand(
        self,
        node: Dict[str, Any],
        expand_child: callable,
        seed: Optional[int] = None
    ) -> List[Any]:
        """Expand the node to all combinations.

        Args:
            node: The node to expand.
            expand_child: Function to recursively expand child nodes.
            seed: Optional random seed for reproducibility.

        Returns:
            List of all expanded variants.
        """
        pass

    @abstractmethod
    def count(
        self,
        node: Dict[str, Any],
        count_child: callable
    ) -> int:
        """Count combinations without generating them.

        Args:
            node: The node to count.
            count_child: Function to recursively count child nodes.

        Returns:
            Total number of combinations.
        """
        pass
```

#### 4.4.2 OR Strategy Implementation

```python
# _generator/strategies/or_strategy.py
from typing import Any, Dict, FrozenSet, List, Optional
from itertools import combinations, permutations

from .base import ExpansionStrategy
from ..keywords import (
    OR_KEYWORD, COUNT_KEYWORD,
    PICK_KEYWORD, ARRANGE_KEYWORD,
    THEN_PICK_KEYWORD, THEN_ARRANGE_KEYWORD,
    PURE_OR_KEYS
)
from ..utils import sample_with_seed

class OrStrategy(ExpansionStrategy):
    """Handles _or_ nodes with pick/arrange semantics."""

    @property
    def primary_keyword(self) -> str:
        return OR_KEYWORD

    @property
    def allowed_keys(self) -> FrozenSet[str]:
        return PURE_OR_KEYS

    def matches(self, node: Dict[str, Any]) -> bool:
        if not isinstance(node, dict):
            return False
        return set(node.keys()).issubset(self.allowed_keys) and OR_KEYWORD in node

    def expand(
        self,
        node: Dict[str, Any],
        expand_child: callable,
        seed: Optional[int] = None
    ) -> List[Any]:
        choices = node[OR_KEYWORD]
        pick = node.get(PICK_KEYWORD)
        arrange = node.get(ARRANGE_KEYWORD)
        then_pick = node.get(THEN_PICK_KEYWORD)
        then_arrange = node.get(THEN_ARRANGE_KEYWORD)
        count = node.get(COUNT_KEYWORD)

        # Dispatch to appropriate handler
        if arrange is not None:
            result = self._expand_arrange(choices, arrange, then_pick, then_arrange, expand_child)
        elif pick is not None:
            result = self._expand_pick(choices, pick, then_pick, then_arrange, expand_child)
        else:
            # Simple expansion: all choices
            result = []
            for choice in choices:
                result.extend(expand_child(choice))

        # Apply count limit
        if count is not None and len(result) > count:
            result = sample_with_seed(result, count, seed=seed)

        return result

    def count(self, node: Dict[str, Any], count_child: callable) -> int:
        # Similar structure to expand, but returns counts
        ...

    def _expand_pick(self, choices, spec, then_pick, then_arrange, expand_child):
        """Handle pick (combinations) with optional second-order."""
        ...

    def _expand_arrange(self, choices, spec, then_pick, then_arrange, expand_child):
        """Handle arrange (permutations) with optional second-order."""
        ...
```

#### 4.4.3 Range Strategy Implementation

```python
# _generator/strategies/range_strategy.py
from typing import Any, Dict, FrozenSet, List, Optional, Union

from .base import ExpansionStrategy
from ..keywords import RANGE_KEYWORD, COUNT_KEYWORD, PURE_RANGE_KEYS
from ..utils import sample_with_seed

class RangeStrategy(ExpansionStrategy):
    """Handles _range_ nodes for numeric sequence generation."""

    @property
    def primary_keyword(self) -> str:
        return RANGE_KEYWORD

    @property
    def allowed_keys(self) -> FrozenSet[str]:
        return PURE_RANGE_KEYS

    def matches(self, node: Dict[str, Any]) -> bool:
        if not isinstance(node, dict):
            return False
        return set(node.keys()).issubset(self.allowed_keys) and RANGE_KEYWORD in node

    def expand(
        self,
        node: Dict[str, Any],
        expand_child: callable,
        seed: Optional[int] = None
    ) -> List[Any]:
        range_spec = node[RANGE_KEYWORD]
        count = node.get(COUNT_KEYWORD)

        values = self._generate_range(range_spec)

        if count is not None and len(values) > count:
            values = sample_with_seed(values, count, seed=seed)

        return values

    def count(self, node: Dict[str, Any], count_child: callable) -> int:
        range_spec = node[RANGE_KEYWORD]
        count_limit = node.get(COUNT_KEYWORD)

        total = self._count_range(range_spec)
        return min(total, count_limit) if count_limit else total

    def _generate_range(self, range_spec: Union[list, dict]) -> List[int]:
        """Generate numeric range from specification."""
        if isinstance(range_spec, list):
            if len(range_spec) == 2:
                start, end = range_spec
                step = 1
            else:
                start, end, step = range_spec
        else:  # dict
            start = range_spec["from"]
            end = range_spec["to"]
            step = range_spec.get("step", 1)

        if step > 0:
            return list(range(start, end + 1, step))
        else:
            return list(range(start, end - 1, step))

    def _count_range(self, range_spec: Union[list, dict]) -> int:
        """Count range elements without generating."""
        ...
```

#### 4.4.4 Strategy Registry

```python
# _generator/strategies/registry.py
from typing import Dict, List, Optional, Type
from .base import ExpansionStrategy
from .or_strategy import OrStrategy
from .range_strategy import RangeStrategy

# Ordered list of strategies (first match wins)
_STRATEGIES: List[Type[ExpansionStrategy]] = [
    RangeStrategy,  # Check range first (simpler node structure)
    OrStrategy,     # Then OR nodes
]

# Cached strategy instances
_strategy_instances: Dict[Type[ExpansionStrategy], ExpansionStrategy] = {}


def get_strategy(node: dict) -> Optional[ExpansionStrategy]:
    """Find matching strategy for a node.

    Args:
        node: A dictionary node from the configuration.

    Returns:
        Matching strategy instance, or None if no strategy matches.
    """
    for strategy_cls in _STRATEGIES:
        if strategy_cls not in _strategy_instances:
            _strategy_instances[strategy_cls] = strategy_cls()

        strategy = _strategy_instances[strategy_cls]
        if strategy.matches(node):
            return strategy

    return None


def register_strategy(strategy_cls: Type[ExpansionStrategy], priority: int = -1):
    """Register a custom strategy.

    Args:
        strategy_cls: The strategy class to register.
        priority: Position in the strategy list (-1 = append at end).
    """
    if priority < 0:
        _STRATEGIES.append(strategy_cls)
    else:
        _STRATEGIES.insert(priority, strategy_cls)
```

### 4.5 Core Module Refactoring

With strategies in place, the core module becomes a thin orchestration layer:

```python
# _generator/core.py
from typing import Any, Dict, List, Optional, Union
from collections.abc import Mapping
from itertools import product

from .strategies.registry import get_strategy
from .keywords import has_or_keyword, extract_base_node

GeneratorNode = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


def expand_spec(
    node: GeneratorNode,
    seed: Optional[int] = None
) -> List[Any]:
    """Expand a specification node to all possible combinations.

    Args:
        node: Specification to expand (dict, list, or scalar).
        seed: Random seed for reproducible sampling when using 'count'.

    Returns:
        List of all expanded combinations.

    Examples:
        >>> expand_spec({"_or_": ["A", "B", "C"]})
        ['A', 'B', 'C']
        >>> expand_spec({"_or_": ["A", "B"], "pick": 2})
        [['A', 'B']]
    """
    return _expand_internal(node, seed)


def _expand_internal(node: GeneratorNode, seed: Optional[int] = None) -> List[Any]:
    """Internal recursive expansion."""
    # Handle lists: Cartesian product of expanded elements
    if isinstance(node, list):
        if not node:
            return [[]]
        expanded = [_expand_internal(elem, seed) for elem in node]
        return [list(combo) for combo in product(*expanded)]

    # Handle scalars
    if not isinstance(node, Mapping):
        return [node]

    # Try to find a matching strategy
    strategy = get_strategy(node)
    if strategy:
        return strategy.expand(node, _expand_internal, seed)

    # Mixed node with _or_ and other keys
    if has_or_keyword(node):
        return _expand_mixed_or_node(node, seed)

    # Regular dict: Cartesian product over values
    return _expand_dict(node, seed)


def _expand_mixed_or_node(node: Dict[str, Any], seed: Optional[int]) -> List[Any]:
    """Expand dict containing _or_ mixed with other keys."""
    base = extract_base_node(node)
    base_expanded = _expand_internal(base, seed)

    # Build OR-only node for separate expansion
    or_node = {k: v for k, v in node.items() if k not in base}
    or_expanded = _expand_internal(or_node, seed)

    # Merge results
    results = []
    for b in base_expanded:
        for c in or_expanded:
            if isinstance(c, Mapping):
                results.append({**b, **c})
            else:
                raise ValueError("Top-level _or_ choices must be dicts")
    return results


def _expand_dict(node: Dict[str, Any], seed: Optional[int]) -> List[Dict]:
    """Expand regular dict by taking Cartesian product of values."""
    if not node:
        return [{}]

    keys = list(node.keys())
    value_options = [_expand_value(node[k], seed) for k in keys]

    results = []
    for combo in product(*value_options):
        results.append(dict(zip(keys, combo)))
    return results


def _expand_value(v: Any, seed: Optional[int]) -> List[Any]:
    """Expand a single value, handling nested generators."""
    if isinstance(v, Mapping):
        strategy = get_strategy(v)
        if strategy:
            return strategy.expand(v, _expand_internal, seed)
        return _expand_internal(v, seed)
    elif isinstance(v, list):
        return _expand_internal(v, seed)
    else:
        return [v]


def count_combinations(node: GeneratorNode) -> int:
    """Count total combinations without generating them.

    Args:
        node: Specification to count.

    Returns:
        Total number of combinations that expand_spec would produce.
    """
    return _count_internal(node)


def _count_internal(node: GeneratorNode) -> int:
    """Internal recursive counting."""
    if isinstance(node, list):
        if not node:
            return 1
        total = 1
        for elem in node:
            total *= _count_internal(elem)
        return total

    if not isinstance(node, Mapping):
        return 1

    strategy = get_strategy(node)
    if strategy:
        return strategy.count(node, _count_internal)

    if has_or_keyword(node):
        return _count_mixed_or_node(node)

    return _count_dict(node)


def _count_mixed_or_node(node: Dict[str, Any]) -> int:
    """Count mixed OR node."""
    base = extract_base_node(node)
    or_node = {k: v for k, v in node.items() if k not in base}
    return _count_internal(base) * _count_internal(or_node)


def _count_dict(node: Dict[str, Any]) -> int:
    """Count regular dict."""
    if not node:
        return 1
    total = 1
    for v in node.values():
        total *= _count_value(v)
    return total


def _count_value(v: Any) -> int:
    """Count value-position combinations."""
    if isinstance(v, (Mapping, list)):
        return _count_internal(v)
    return 1
```

### 4.6 Simplified Public API

After Phase 2, `generator.py` becomes a thin wrapper:

```python
# generator.py (becomes thin wrapper ~50 lines)
"""Generator module for pipeline configuration expansion.

This module expands pipeline configuration specifications into concrete
pipeline variants. It handles combinatorial keywords (_or_, _range_,
pick, arrange, count) and generates all possible combinations.

Main Functions:
    expand_spec(node): Expand a configuration node into all variants
    count_combinations(node): Count variants without generating them

Keywords:
    _or_: Choice between alternatives
    _range_: Numeric sequence generation
    pick: Unordered selection (combinations)
    arrange: Ordered arrangement (permutations)
    then_pick: Second-order combination selection
    then_arrange: Second-order permutation selection
    count: Limit number of generated variants
"""

# Re-export core API
from ._generator.core import expand_spec, count_combinations

# Re-export keywords for external use
from ._generator.keywords import (
    OR_KEYWORD, RANGE_KEYWORD, COUNT_KEYWORD,
    PICK_KEYWORD, ARRANGE_KEYWORD, THEN_PICK_KEYWORD, THEN_ARRANGE_KEYWORD,
    PURE_OR_KEYS, PURE_RANGE_KEYS,
    is_generator_node, is_pure_or_node, is_pure_range_node,
    has_or_keyword, has_range_keyword,
    extract_modifiers, extract_base_node,
)

__all__ = [
    "expand_spec", "count_combinations",
    # Keywords
    "OR_KEYWORD", "RANGE_KEYWORD", "COUNT_KEYWORD",
    "PICK_KEYWORD", "ARRANGE_KEYWORD", "THEN_PICK_KEYWORD", "THEN_ARRANGE_KEYWORD",
    "PURE_OR_KEYS", "PURE_RANGE_KEYS",
    # Utilities
    "is_generator_node", "is_pure_or_node", "is_pure_range_node",
    "has_or_keyword", "has_range_keyword",
    "extract_modifiers", "extract_base_node",
]
```

### 4.7 Validators Module (New)

```python
# _generator/validators/__init__.py
"""Validation utilities for generated configurations."""

from .schema import validate_config, ValidationError

__all__ = ["validate_config", "ValidationError"]
```

```python
# _generator/validators/schema.py
"""Schema validation for generated configurations."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class ValidationError(Exception):
    """Raised when a generated configuration is invalid."""
    def __init__(self, message: str, path: str = "", value: Any = None):
        self.message = message
        self.path = path
        self.value = value
        super().__init__(f"{path}: {message}" if path else message)


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    valid: bool
    errors: List[ValidationError]
    warnings: List[str]

    @property
    def error_messages(self) -> List[str]:
        return [str(e) for e in self.errors]


def validate_config(
    config: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
    strict: bool = False
) -> ValidationResult:
    """Validate a generated configuration.

    Args:
        config: The configuration to validate.
        schema: Optional schema definition for validation.
        strict: If True, unknown keys are errors; otherwise warnings.

    Returns:
        ValidationResult with valid flag, errors, and warnings.

    Examples:
        >>> result = validate_config({"model": "PLS"})
        >>> result.valid
        True
    """
    errors = []
    warnings = []

    # Check for empty config
    if not config:
        warnings.append("Empty configuration")

    # If schema provided, validate against it
    if schema:
        errors.extend(_validate_against_schema(config, schema, strict))

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def _validate_against_schema(
    config: Dict[str, Any],
    schema: Dict[str, Any],
    strict: bool
) -> List[ValidationError]:
    """Validate config against schema definition."""
    errors = []

    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in config:
            errors.append(ValidationError(
                f"Missing required field: {field}",
                path=field
            ))

    # Check types
    properties = schema.get("properties", {})
    for key, value in config.items():
        if key in properties:
            expected_type = properties[key].get("type")
            if expected_type and not _check_type(value, expected_type):
                errors.append(ValidationError(
                    f"Expected {expected_type}, got {type(value).__name__}",
                    path=key,
                    value=value
                ))
        elif strict:
            errors.append(ValidationError(
                f"Unknown field: {key}",
                path=key
            ))

    return errors


def _check_type(value: Any, expected: str) -> bool:
    """Check if value matches expected type string."""
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected_types = type_map.get(expected, object)
    return isinstance(value, expected_types)
```

### 4.8 Migration Path: Phase 2 Implementation Plan

#### 4.8.1 Week 1: Strategy Infrastructure

**Tasks:**
1. Create `_generator/strategies/` directory
2. Implement `base.py` with abstract `ExpansionStrategy`
3. Implement `registry.py` with strategy dispatch
4. Add unit tests for registry

**Deliverables:**
- [ ] `_generator/strategies/__init__.py`
- [ ] `_generator/strategies/base.py`
- [ ] `_generator/strategies/registry.py`
- [ ] `tests/unit/pipeline/config/generator/test_registry.py`

#### 4.8.2 Week 2: Range Strategy

**Tasks:**
1. Implement `RangeStrategy` in `range_strategy.py`
2. Move `_generate_range` and `_count_range` logic to strategy
3. Update registry to use RangeStrategy
4. Ensure backward compatibility with tests

**Deliverables:**
- [ ] `_generator/strategies/range_strategy.py`
- [ ] `tests/unit/pipeline/config/generator/test_range_strategy.py`

#### 4.8.3 Week 3: OR Strategy (Core)

**Tasks:**
1. Implement basic `OrStrategy` without pick/arrange
2. Handle simple `_or_` expansion
3. Handle `pick` parameter
4. Handle `count` parameter
5. Add comprehensive tests

**Deliverables:**
- [ ] `_generator/strategies/or_strategy.py` (basic)
- [ ] `tests/unit/pipeline/config/generator/test_or_strategy.py`

#### 4.8.4 Week 4: OR Strategy (Selection Semantics)

**Tasks:**
1. Add `pick` and `arrange` handling to OrStrategy
2. Add `then_pick` and `then_arrange` second-order
3. Handle nested array syntax `[outer, inner]`
4. Ensure all existing tests pass

**Deliverables:**
- [ ] `_generator/strategies/or_strategy.py` (complete)
- [ ] Updated tests for pick/arrange

#### 4.8.5 Week 5: Core Module Refactoring

**Tasks:**
1. Create `_generator/core.py` with simplified expand_spec
2. Integrate strategy dispatch into core
3. Update `generator.py` to be thin wrapper
4. Run full test suite, fix regressions

**Deliverables:**
- [ ] `_generator/core.py`
- [ ] Simplified `generator.py`
- [ ] All tests passing

#### 4.8.6 Week 6: Validators & Polish

**Tasks:**
1. Create `_generator/validators/` module
2. Implement basic schema validation
3. Add validation integration tests
4. Documentation updates
5. Clean up old code (remove duplicates)

**Deliverables:**
- [ ] `_generator/validators/__init__.py`
- [ ] `_generator/validators/schema.py`
- [ ] Updated documentation
- [ ] Final cleanup

### 4.9 Phase 2 Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Main file size | < 100 lines | `wc -l generator.py` |
| Test coverage | > 90% | `pytest --cov` |
| All existing tests | Pass | `pytest tests/` |
| DRY compliance | No duplicate expand/count | Code review |
| Type coverage | 100% public API | `mypy --strict` |
| Documentation | All functions documented | `pydocstyle` |

### 4.10 Backward Compatibility Guarantees

Phase 2 maintains full backward compatibility:

1. **Public API unchanged**: `expand_spec()` and `count_combinations()` signatures preserved
2. **Keyword behavior unchanged**: All existing keywords work identically
3. **Import paths stable**: `from nirs4all.pipeline.config.generator import expand_spec` continues to work
4. **No deprecation warnings**: Phase 2 adds no new deprecation warnings
5. **Test compatibility**: All existing tests pass without modification

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
    "pick": 2,
    "_mutex_": [["A", "B"], ["C", "D"]]  # A and B can't be together, same for C and D
}
# Generates: [A,C], [A,D], [B,C], [B,D] - but not [A,B] or [C,D]
```

#### 5.4.2 Required Combinations

```python
{
    "_or_": ["A", "B", "C", "D"],
    "pick": 3,
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
        "pick": (1, 2)
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
        "pick": 2,
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
├── test_edge_cases.py        # Empty inputs, pick=0, count=0, etc.
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

### Phase 1: Critical Fixes ✅ COMPLETE
- [x] Add seed support via `sample_with_seed` utility
- [x] Add type annotations in `keywords.py` and `utils/`
- [x] Add comprehensive docstrings
- [x] Create test suite for edge cases

### Phase 1.5: Selection Semantics ✅ COMPLETE
- [x] Add `pick` keyword for combinations (unordered selection)
- [x] Add `arrange` keyword for permutations (ordered arrangement)
- [x] Add `then_pick` and `then_arrange` for second-order operations
- [x] Create `_generator/` subpackage structure
- [x] Extract `keywords.py` with centralized constants
- [x] Extract `utils/sampling.py` and `utils/combinatorics.py`
- [x] Add comprehensive tests for pick/arrange in `test_generator_pick_arrange.py`

### Phase 2: Modularization (Current - 6 weeks)
- [x] Week 1: Create strategy infrastructure (`base.py`, `registry.py`)
- [x] Week 2: Implement `RangeStrategy`
- [x] Week 3: Implement basic `OrStrategy`
- [x] Week 4: Add pick/arrange handling to `OrStrategy`
- [x] Week 5: Refactor core module, thin wrapper
- [x] Week 6: Add validators, polish, cleanup

**See Section 4.8 for detailed implementation plan.**

### Phase 3: New Features (Week 7-12)
- [x] Add `_log_range_` support for logarithmic ranges
- [x] Add weighted sampling (`_weights_`) using existing `sample_with_seed`
- [x] Add exclusion rules (`_exclude_`) for forbidden combinations
- [x] Add tagging system (`_tag_`) for configuration metadata
- [x] Add duplicates removal or generation skip

### Phase 4: Production Features (Week 13-18) ✅ COMPLETED
- [x] Add lazy/iterator generation via `expand_spec_iter()` - iterator.py
- [x] Add constraint system (`_mutex_`, `_requires_`, `_exclude_`) - constraints.py
- [x] Add preset system (`_preset_`) for named configurations - presets.py
- [x] Add export/visualization tools (DataFrame, tree view) - utils/export.py

### Phase 5: Documentation & Examples (Week 19-20)
- [ ] Update reference documentation in `docs/reference/`
- [ ] Create tutorial examples for new features
- [ ] Create a complete pipeline with PLS (regression dataset) using a very complex and nested generation pipeline
- [ ] Add migration guide for deprecated patterns
- [ ] Update `Q23_generator_syntax.py` example

---

## 8. Summary

### Current Status (Post Phase 4)

The generator module has completed its major refactoring phases:

**Completed ✅**:
1. `pick` and `arrange` keywords for explicit selection semantics
2. `then_pick` and `then_arrange` for second-order operations
3. Modular `_generator/` subpackage with strategies, constraints, presets
4. Seed-aware random sampling for reproducibility
5. Strategy pattern for all keyword types (`_or_`, `_range_`, `_log_range_`, `_grid_`, `_zip_`, `_chain_`, `_sample_`)
6. Constraint system (`_mutex_`, `_requires_`, `_exclude_`) for filtering combinations
7. Preset system (`_preset_`) for reusable named configurations
8. Iterator-based expansion (`expand_spec_iter()`) for memory efficiency
9. Export utilities (DataFrame export, tree visualization, config diff)
10. Comprehensive test coverage (261+ tests passing)

**Remaining** (Phase 5):
- Documentation updates in `docs/reference/`
- Tutorial examples for new features
- Migration guide for deprecated patterns

### Recommended Next Actions

1. **Update reference documentation** (Phase 5)
   - Document constraint keywords in `docs/reference/`
   - Document preset system and usage
   - Document iterator API

2. **Create tutorial examples** (Phase 5)
   - Example with presets for common preprocessing patterns
   - Example with constraints for valid transform combinations
   - Example using iterator for large configuration spaces

3. **Update Q23_generator_syntax.py** (Phase 5)
   - Add Phase 4 features to the examples
   - Show constraint and preset usage

### Long-term Vision

- ✅ Modular, extensible architecture using strategy pattern
- ✅ Rich keyword vocabulary for complex generation patterns
- ✅ Production-ready API with validation, constraints, and tagging
- ✅ Memory-efficient lazy generation for large configuration spaces
- ✅ Full backward compatibility throughout evolution

---

*Document created: December 8, 2025*
*Last updated: Phase 4 Complete*
*Author: GitHub Copilot Analysis*


