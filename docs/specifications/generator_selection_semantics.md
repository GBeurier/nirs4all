# Generator Selection Semantics: `pick` vs `arrange`

**Date**: December 8, 2025
**Status**: ✅ Implemented
**Phase**: 1.5 (Complete)
**Module**: `nirs4all/pipeline/config/generator.py`

---

## Implementation Status

> **Phase 1.5 has been fully implemented.** All features described in this specification
> are now available in the generator module:
>
> - ✅ `pick` keyword for combinations (unordered selection)
> - ✅ `arrange` keyword for permutations (ordered arrangement)
> - ✅ `then_pick` and `then_arrange` for second-order operations
> - ✅ `_generator/keywords.py` with centralized constants
> - ✅ `_generator/utils/` with sampling and combinatorics helpers
> - ✅ Comprehensive test coverage in `test_generator_pick_arrange.py`
>
> See [generator_analysis.md](generator_analysis.md) for Phase 2 planning.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Root Cause Analysis](#2-root-cause-analysis)
3. [Proposed Solution](#3-proposed-solution)
4. [Syntax Reference](#4-syntax-reference)
5. [Controller Context Defaults](#5-controller-context-defaults)
6. [Migration and Backward Compatibility](#6-migration-and-backward-compatibility)
7. [Implementation Roadmap (Phase 1.5)](#7-implementation-roadmap-phase-15)
8. [Examples](#8-examples)

---

## 1. Problem Statement

### 1.1 The Current Situation

The generator module uses the `size` parameter with `_or_` to select multiple items:

```python
{"_or_": ["A", "B", "C"], "size": 2}
# Currently generates: [["A", "B"], ["A", "C"], ["B", "C"]]
# Uses combinations: C(3, 2) = 3 variants
```

### 1.2 The Problem

**The `size` parameter always uses combinations (order doesn't matter), but this is semantically incorrect in some contexts.**

Consider these two use cases:

#### Use Case 1: concat_transform (Order DOESN'T Matter)
```python
{"concat_transform": {"_or_": [PCA(30), SVD(20), ICA(25)], "size": 2}}
```

Here, selecting `[PCA, SVD]` means concatenating their outputs horizontally:
- `[PCA_output | SVD_output]` → 50 features
- `[SVD_output | PCA_output]` → 50 features (same information, different column order)

For ML models, feature order typically doesn't matter. **Combinations are correct.**

#### Use Case 2: Sequential Pipeline Transforms (Order MATTERS)
```python
# If someone wanted to generate sequences of 2 transforms from a pool
{"preprocessing": {"_or_": [StandardScaler, Normalizer, RobustScaler], "size": 2}}
```

If this represents a **sequence** of transforms applied one after another:
- `[StandardScaler, Normalizer]` → StandardScaler then Normalizer
- `[Normalizer, StandardScaler]` → Normalizer then StandardScaler

These produce **different results!** **Permutations are needed.**

### 1.3 The Fundamental Confusion

The `size` parameter conflates two distinct concepts:

| Concept | Mathematical Term | When to Use |
|---------|-------------------|-------------|
| "Select N items, order irrelevant" | Combination C(n, k) | Parallel/independent operations |
| "Arrange N items in sequence" | Permutation P(n, k) | Sequential/chained operations |

**Users cannot express which behavior they want.**

### 1.4 Current Workaround (Second-Order)

The `size=[outer, inner]` notation partially addresses this:
- Inner uses permutations (order matters within each sub-sequence)
- Outer uses combinations (selecting which sub-sequences to include)

But this is:
1. Not intuitive
2. Only for second-order scenarios
3. Not documented clearly
4. Doesn't solve first-order selection needs

---

## 2. Root Cause Analysis

### 2.1 Context-Dependent Semantics

The correct behavior depends on **where** the generator is used:

| Context | Order Matters? | Should Use |
|---------|----------------|------------|
| `concat_transform` | No | Combinations |
| `feature_augmentation` | No | Combinations |
| `sample_augmentation` | No | Combinations |
| Pipeline-level list | Yes | Permutations |
| Chained transforms `[[A, B]]` | Yes | Permutations |

### 2.2 Generator is Context-Agnostic

The generator module processes specifications without knowing the context:

```python
# Generator only sees this:
{"_or_": [A, B, C], "size": 2}

# It doesn't know if this is for:
# - concat_transform (should use combinations)
# - a preprocessing sequence (should use permutations)
```

### 2.3 Why This Wasn't a Problem Before

1. Most uses were for `feature_augmentation` and `concat_transform` where combinations are correct
2. Second-order `size=[outer, inner]` handled the common case of "create sequences, pick some"
3. Pipeline-level generation typically uses Cartesian product (separate `_or_` nodes), not multi-selection

### 2.4 When This Becomes a Problem

As the generator syntax becomes more powerful and used in more contexts:
- Users may want to generate sequences of preprocessing steps
- Custom controllers may have different semantics
- The implicit assumption of "combinations" will cause confusion

---

## 3. Proposed Solution

### 3.1 Core Concept: Explicit Keywords

Replace the overloaded `size` with two explicit keywords that clearly communicate intent:

| Keyword | Meaning | Mathematical Basis |
|---------|---------|-------------------|
| `pick` | Select N items, order doesn't matter | Combinations C(n, k) |
| `arrange` | Arrange N items in sequence, order matters | Permutations P(n, k) |

### 3.2 Why These Names?

- **`pick`**: Evokes "picking items from a bag" - you don't care about the order you pick them
- **`arrange`**: Evokes "arranging items in a line" - the sequence matters

Alternative names considered:
- `choose` / `order` - "order" is ambiguous (sorting? sequencing?)
- `select` / `sequence` - both are longer
- `combo` / `perm` - too mathematical, not user-friendly

### 3.3 Syntax Overview

```python
# Unordered selection (combinations)
{"_or_": ["A", "B", "C"], "pick": 2}
# → [["A", "B"], ["A", "C"], ["B", "C"]]
# C(3, 2) = 3 variants

# Ordered arrangement (permutations)
{"_or_": ["A", "B", "C"], "arrange": 2}
# → [["A", "B"], ["A", "C"], ["B", "A"], ["B", "C"], ["C", "A"], ["C", "B"]]
# P(3, 2) = 6 variants
```

### 3.4 Range Syntax

Both keywords support ranges:

```python
# Pick 1 to 3 items (unordered)
{"_or_": ["A", "B", "C", "D"], "pick": (1, 3)}
# C(4,1) + C(4,2) + C(4,3) = 4 + 6 + 4 = 14 variants

# Arrange 1 to 3 items (ordered)
{"_or_": ["A", "B", "C", "D"], "arrange": (1, 3)}
# P(4,1) + P(4,2) + P(4,3) = 4 + 12 + 24 = 40 variants
```

### 3.5 Controller-Aware Defaults

Controllers can translate legacy `size` to the appropriate keyword:

```python
# In ConcatAugmentationController
def _normalize_spec(self, spec):
    if "size" in spec and "pick" not in spec and "arrange" not in spec:
        # concat_transform uses combinations by default
        spec = {**spec, "pick": spec.pop("size")}
    return spec

# In a hypothetical SequentialController
def _normalize_spec(self, spec):
    if "size" in spec and "pick" not in spec and "arrange" not in spec:
        # Sequential transforms use permutations by default
        spec = {**spec, "arrange": spec.pop("size")}
    return spec
```

---

## 4. Syntax Reference

### 4.1 First-Order Selection

#### `pick` - Unordered Selection (Combinations)

```python
# Basic: pick exactly 2 items
{"_or_": ["A", "B", "C"], "pick": 2}
# Result: [["A", "B"], ["A", "C"], ["B", "C"]]
# Count: C(3, 2) = 3

# Range: pick 1 to 2 items
{"_or_": ["A", "B", "C"], "pick": (1, 2)}
# Result: [["A"], ["B"], ["C"], ["A", "B"], ["A", "C"], ["B", "C"]]
# Count: C(3,1) + C(3,2) = 6

# With count limit (random sampling)
{"_or_": ["A", "B", "C", "D"], "pick": 2, "count": 3}
# Result: 3 randomly sampled combinations from C(4,2)=6
```

#### `arrange` - Ordered Arrangement (Permutations)

```python
# Basic: arrange exactly 2 items
{"_or_": ["A", "B", "C"], "arrange": 2}
# Result: [["A", "B"], ["A", "C"], ["B", "A"], ["B", "C"], ["C", "A"], ["C", "B"]]
# Count: P(3, 2) = 6

# Range: arrange 1 to 2 items
{"_or_": ["A", "B", "C"], "arrange": (1, 2)}
# Result: [["A"], ["B"], ["C"], ["A", "B"], ["A", "C"], ["B", "A"], ...]
# Count: P(3,1) + P(3,2) = 9

# With count limit (random sampling)
{"_or_": ["A", "B", "C", "D"], "arrange": 2, "count": 5}
# Result: 5 randomly sampled permutations from P(4,2)=12
```

### 4.2 Second-Order Selection (Nested)

For complex scenarios where you need hierarchical selection:

#### Current Syntax (to be evolved)
```python
# size=[outer, inner] - inner uses permutations, outer uses combinations
{"_or_": ["A", "B", "C"], "size": [2, 2]}
```

#### New Explicit Syntax

```python
# Pick 2 combinations first, then arrange those results
{"_or_": ["A", "B", "C"], "pick": 2, "then_arrange": 2}
# Step 1: Pick C(3,2) = 3 combinations: [A,B], [A,C], [B,C]
# Step 2: Arrange P(3,2) = 6 orderings of those 3 items

# Arrange 2 permutations first, then pick from those results
{"_or_": ["A", "B", "C"], "arrange": 2, "then_pick": 2}
# Step 1: Create P(3,2) = 6 permutations: [A,B], [A,C], [B,A], [B,C], [C,A], [C,B]
# Step 2: Pick C(6,2) = 15 combinations of those 6 items
```

#### Alternative Nested Syntax

```python
# Using a nested object for clarity
{
    "_or_": ["A", "B", "C"],
    "nested": {
        "inner": {"arrange": 2},   # First: create arrangements
        "outer": {"pick": 2}       # Then: pick groups
    }
}
```

### 4.3 Combining with Other Modifiers

```python
# pick with count (random sample)
{"_or_": [...], "pick": 2, "count": 5}

# arrange with seed (deterministic)
{"_or_": [...], "arrange": 2, "count": 5, "_seed_": 42}

# Nested values with pick
{
    "n_components": {"_or_": [10, 20, 30], "pick": 2},
    "method": {"_or_": ["pca", "svd"]}
}
# Creates all combinations of: 2-item picks × methods
```

---

## 5. Controller Context Defaults

### 5.1 Default Behavior Matrix

| Controller | Default for `size` | Rationale |
|------------|-------------------|-----------|
| `concat_transform` | → `pick` | Horizontal concatenation, feature order irrelevant |
| `feature_augmentation` | → `pick` | Parallel processing channels |
| `sample_augmentation` | → `pick` | Transformer selection for samples |
| Pipeline-level list | → `arrange` | Sequential execution order matters |
| User-specified `pick` | `pick` | Explicit choice honored |
| User-specified `arrange` | `arrange` | Explicit choice honored |

### 5.2 Implementation in Controllers

```python
# nirs4all/controllers/data/concat_transform.py

class ConcatAugmentationController(OperatorController):

    @staticmethod
    def normalize_generator_spec(spec: dict) -> dict:
        """Normalize generator spec for concat_transform context.

        In concat_transform context, multi-selection should use combinations
        by default since the order of concatenated features doesn't matter.
        """
        if not isinstance(spec, dict):
            return spec

        # If explicit pick/arrange specified, honor it
        if "pick" in spec or "arrange" in spec:
            return spec

        # Convert legacy size to pick (combinations)
        if "size" in spec:
            result = dict(spec)
            result["pick"] = result.pop("size")
            return result

        return spec
```

### 5.3 Generator Context Parameter (Phase 2)

For more advanced use cases, the generator can accept a context hint:

```python
def expand_spec(node, context: str = None, seed: int = None):
    """
    Args:
        node: Specification to expand
        context: Hint for default behavior
            - "parallel": size → pick (combinations)
            - "sequential": size → arrange (permutations)
            - None: use spec as-is
        seed: Random seed for reproducibility
    """
    if context and isinstance(node, dict) and "size" in node:
        if "pick" not in node and "arrange" not in node:
            node = dict(node)
            if context == "parallel":
                node["pick"] = node.pop("size")
            elif context == "sequential":
                node["arrange"] = node.pop("size")
    # ... rest of expansion logic
```

---

## 6. Migration and Backward Compatibility

### 6.1 Phased Approach

| Phase | `size` Behavior | `pick`/`arrange` | Notes |
|-------|-----------------|------------------|-------|
| 1 (Current) | Uses combinations | N/A | Status quo |
| 1.5 | Uses combinations | Fully supported | Both work, no warnings |
| 2 | Uses combinations + warning | Fully supported | Deprecation warning for `size` |
| 3 | Error | Required | `size` removed |

### 6.2 Phase 1.5 Details (This Proposal)

1. **Add `pick` keyword**: Works exactly like current `size` (combinations)
2. **Add `arrange` keyword**: New functionality (permutations)
3. **Keep `size` working**: No changes to existing behavior, no warnings
4. **Controllers normalize internally**: Can translate `size` to `pick` if desired
5. **Documentation**: Recommend `pick`/`arrange` for new code

### 6.3 Phase 2 Migration (Future)

```python
# In generator.py
import warnings

def expand_spec(node, ...):
    if isinstance(node, dict) and "size" in node:
        if "pick" not in node and "arrange" not in node:
            warnings.warn(
                "The 'size' parameter is deprecated. "
                "Use 'pick' for combinations or 'arrange' for permutations.",
                DeprecationWarning
            )
            # Continue using size as pick for backward compat
```

### 6.4 Code Update Examples

```python
# Before (works but deprecated in Phase 2)
{"_or_": [A, B, C], "size": 2}

# After - for unordered selection
{"_or_": [A, B, C], "pick": 2}

# After - for ordered sequences
{"_or_": [A, B, C], "arrange": 2}
```

---

## 7. Implementation Roadmap (Phase 1.5)

### 7.1 Overview

Phase 1.5 sits between the existing Phase 1 (Critical Fixes) and Phase 2 (Modularization) from the main generator analysis document.

**Duration**: ~4 weeks
**Dependencies**: Builds on Phase 1 completion (seed support, type hints)

### 7.2 Week 1: Core Implementation

**Tasks**:
1. Add `PICK_KEYWORD = "pick"` and `ARRANGE_KEYWORD = "arrange"` to keywords.py
2. Update `PURE_OR_KEYS` to include `pick` and `arrange`
3. Implement `_expand_pick()` function (combinations) - can reuse existing code
4. Implement `_expand_arrange()` function (permutations)
5. Update `expand_spec()` to handle both keywords
6. Update `count_combinations()` for new keywords

**Files Modified**:
- `nirs4all/pipeline/config/_generator/keywords.py`
- `nirs4all/pipeline/config/generator.py`

### 7.3 Week 2: Counting and Edge Cases

**Tasks**:
1. Implement `_count_pick()` and `_count_arrange()`
2. Handle range syntax: `pick: (from, to)`, `arrange: (from, to)`
3. Handle edge cases: `pick: 0`, `arrange: 0`, empty choices
4. Ensure `count` modifier works with both
5. Ensure `_seed_` works with both

**Files Modified**:
- `nirs4all/pipeline/config/generator.py`

### 7.4 Week 3: Controller Integration

**Tasks**:
1. Add `normalize_generator_spec()` to `ConcatAugmentationController`
2. Add `normalize_generator_spec()` to `FeatureAugmentationController`
3. Add `normalize_generator_spec()` to `SampleAugmentationController`
4. Ensure controllers call normalization before passing to generator
5. Add tests for controller normalization

**Files Modified**:
- `nirs4all/controllers/data/concat_transform.py`
- `nirs4all/controllers/data/feature_augmentation.py`
- `nirs4all/controllers/data/sample_augmentation.py`

### 7.5 Week 4: Testing and Documentation

**Tasks**:
1. Add unit tests for `pick` keyword
2. Add unit tests for `arrange` keyword
3. Add integration tests with controllers
4. Update Q23_generator_syntax.py example
5. Update user documentation
6. Update API reference

**Files Created/Modified**:
- `tests/unit/pipeline/config/test_generator_pick_arrange.py`
- `examples/Q23_generator_syntax.py`
- `docs/user_guide/generator_syntax.md`

### 7.6 Deliverables

| Deliverable | Description |
|-------------|-------------|
| `pick` keyword | Combinations-based selection |
| `arrange` keyword | Permutations-based selection |
| Controller normalization | Context-aware defaults |
| Test suite | Comprehensive coverage |
| Updated examples | Q23 with new syntax |
| Documentation | User guide updates |

---

## 8. Examples

### 8.1 concat_transform with pick (Recommended)

```python
# Using pick explicitly (recommended for clarity)
pipeline = [
    {
        "concat_transform": {
            "_or_": [PCA(30), SVD(20), ICA(25)],
            "pick": 2  # Select 2 transformers, order doesn't matter
        }
    }
]
# Generates 3 pipeline variants: PCA+SVD, PCA+ICA, SVD+ICA
# Each produces concatenated features (50-55 features)
```

### 8.2 feature_augmentation with pick

```python
pipeline = [
    {
        "feature_augmentation": {
            "_or_": [SNV, MSC, FirstDerivative, SecondDerivative],
            "pick": (1, 3)  # Add 1 to 3 processing channels
        }
    }
]
# Generates C(4,1) + C(4,2) + C(4,3) = 14 pipeline variants
# Each has different combinations of preprocessing channels
```

### 8.3 Sequential Transforms with arrange

```python
# Hypothetical: generating preprocessing sequences
preprocessing_pool = [StandardScaler, Normalizer, RobustScaler, MinMaxScaler]

# Generate all 2-step preprocessing sequences
pipeline_template = [
    {
        "preprocessing_sequence": {
            "_or_": preprocessing_pool,
            "arrange": 2  # Order matters: A→B ≠ B→A
        }
    },
    PLSRegression(n_components=10)
]
# Generates P(4,2) = 12 pipeline variants
# Each has a different 2-step preprocessing sequence
```

### 8.4 Comparing pick vs arrange

```python
from nirs4all.pipeline.config.generator import expand_spec

choices = ["A", "B", "C"]

# Using pick (combinations)
pick_result = expand_spec({"_or_": choices, "pick": 2})
print("pick: 2 →", pick_result)
# Output: [['A', 'B'], ['A', 'C'], ['B', 'C']]
# Count: 3

# Using arrange (permutations)
arrange_result = expand_spec({"_or_": choices, "arrange": 2})
print("arrange: 2 →", arrange_result)
# Output: [['A', 'B'], ['A', 'C'], ['B', 'A'], ['B', 'C'], ['C', 'A'], ['C', 'B']]
# Count: 6
```

### 8.5 Second-Order: Groups of Sequences

```python
# Pick combinations first, then arrange those results
{
    "_or_": [PCA, SVD, ICA, NMF],
    "pick": 2,           # Pick 2-item combinations first
    "then_arrange": 2    # Then arrange 2 of those results
}

# Step 1: Pick C(4,2) = 6 combinations
#   [PCA,SVD], [PCA,ICA], [PCA,NMF], [SVD,ICA], [SVD,NMF], [ICA,NMF]

# Step 2: Arrange P(6,2) = 30 orderings of those 6 items
#   [[PCA,SVD], [PCA,ICA]], [[PCA,ICA], [PCA,SVD]], ...
```

### 8.6 Real-World Pipeline Example

```python
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

from nirs4all.operators.transforms import SNV, MSC, FirstDerivative

# Generate pipelines exploring preprocessing combinations
pipeline_spec = [
    # Y scaling
    {"y_processing": MinMaxScaler()},

    # X scaling (simple choice)
    {"_or_": [StandardScaler(), MinMaxScaler()]},

    # Feature augmentation: pick 1-2 spectral preprocessings
    {
        "feature_augmentation": {
            "_or_": [SNV, MSC, FirstDerivative],
            "pick": (1, 2)  # Combinations: order doesn't matter
        }
    },

    # Dimensionality reduction via concat
    {
        "concat_transform": {
            "_or_": [PCA(30), TruncatedSVD(30), FastICA(30)],
            "pick": 2  # Combinations: order doesn't matter
        }
    },

    # Cross-validation
    ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),

    # Model
    PLSRegression(n_components={"_range_": [5, 15]})
]

# Total variants:
# - Y scaling: 1
# - X scaling: 2
# - Feature augmentation: C(3,1) + C(3,2) = 6
# - Concat transform: C(3,2) = 3
# - PLS components: 11 (5 to 15)
# Total: 2 × 6 × 3 × 11 = 396 pipeline variants
```

---

## 9. Summary

### 9.1 Key Changes

1. **New `pick` keyword**: Explicit unordered selection (combinations)
2. **New `arrange` keyword**: Explicit ordered arrangement (permutations)
3. **Controller defaults**: Context-aware translation of legacy `size`
4. **Backward compatible**: `size` continues to work (as `pick`)

### 9.2 Benefits

- **Clarity**: Users explicitly state their intent
- **Flexibility**: Both behaviors available in any context
- **Correctness**: Sequential transforms can now use permutations
- **No breakage**: Existing code continues to work

### 9.3 Next Steps

1. Review and approve this specification
2. Implement Phase 1.5 tasks
3. Update examples and documentation
4. Proceed to Phase 2 (Modularization) in main roadmap

---

*Document created: December 8, 2025*
*Author: GitHub Copilot Analysis*
*Related: [generator_analysis.md](generator_analysis.md)*
