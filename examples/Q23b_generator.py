"""
Q23: Generator Syntax Examples
==============================
This example demonstrates the generator syntax for creating multiple pipeline
configurations from a single specification. No models are run - this is purely
to illustrate and validate the generation mechanisms.

Generator Keywords:
- _or_: Choose between alternatives
- _range_: Generate numeric sequences
- size: Number of items to select from _or_ choices (legacy)
- pick: Unordered selection (combinations) - explicit intent
- arrange: Ordered arrangement (permutations) - explicit intent
- count: Limit number of generated variants

This example covers:
1. Basic _or_ expansion
2. _or_ with size (combinations) - legacy
3. pick keyword (explicit combinations)
4. arrange keyword (explicit permutations)
5. pick vs arrange comparison
6. _range_ for numeric sequences
7. Nested _or_ in dictionaries
8. Complex nested structures
9. count for limiting results
10. Second-order combinations with [outer, inner] size/pick/arrange

Usage:
    python Q23_generator_syntax.py
"""

from nirs4all.pipeline.config.generator import (
    expand_spec,
    count_combinations,
    is_generator_node,
    sample_with_seed,
    PICK_KEYWORD,
    ARRANGE_KEYWORD,
)

import json


def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()


def show_expansion(name: str, spec, show_all: bool = True, max_show: int = 10):
    """Show expansion results for a specification."""
    print(f"--- {name} ---")
    print(f"Spec: {json.dumps(spec, indent=2, default=str)}")
    print()

    count = count_combinations(spec)
    print(f"Count (without generating): {count}")

    results = expand_spec(spec)
    print(f"Actual results count: {len(results)}")

    if show_all or len(results) <= max_show:
        print("Results:")
        for i, r in enumerate(results):
            print(f"  [{i}] {r}")
    else:
        print(f"First {max_show} results:")
        for i, r in enumerate(results[:max_show]):
            print(f"  [{i}] {r}")
        print(f"  ... ({len(results) - max_show} more)")

    print()
    return results


# ============================================================
# Example 1: Basic _or_ expansion
# ============================================================
print_section("EXAMPLE 1: Basic _or_ expansion")
print("The _or_ keyword creates variants from a list of choices.")
print("Each choice becomes a separate configuration.")

# Simple string choices
show_expansion(
    "String choices",
    {"_or_": ["StandardScaler", "MinMaxScaler", "RobustScaler"], "pick": 2, "then_arrange": (1, 2)}
)

# # Dictionary choices (typical for pipeline steps)
# show_expansion(
#     "Dictionary choices",
#     {"_or_": [
#         {"class": "PCA", "n_components": 10},
#         {"class": "TruncatedSVD", "n_components": 10},
#         {"class": "FastICA", "n_components": 10},
#     ]}
# )

# # Mixed types (scalars, dicts)
# show_expansion(
#     "Mixed types",
#     {"_or_": [None, 5, {"window": 11}]}
# )


# # ============================================================
# # Example 2: _or_ with size (combinations)
# # ============================================================
# print_section("EXAMPLE 2: _or_ with size (combinations)")
# print("The 'size' parameter selects combinations of N items from choices.")
# print("Uses mathematical combinations C(n, k).")

# # Size = 2: pick 2 from 3 choices -> C(3,2) = 3 combinations
# show_expansion(
#     "size=2 from 3 items (C(3,2)=3)",
#     {"_or_": ["A", "B", "C"], "size": 2}
# )

# # Size = 2 from 4 choices -> C(4,2) = 6 combinations
# show_expansion(
#     "size=2 from 4 items (C(4,2)=6)",
#     {"_or_": ["PCA", "SVD", "ICA", "NMF"], "size": 2}
# )

# # Size = 3 from 4 choices -> C(4,3) = 4 combinations
# show_expansion(
#     "size=3 from 4 items (C(4,3)=4)",
#     {"_or_": ["A", "B", "C", "D"], "size": 3}
# )


# # ============================================================
# # Example 3: _or_ with size tuple (range of sizes)
# # ============================================================
# print_section("EXAMPLE 3: _or_ with size tuple (range of sizes)")
# print("size=(from, to) generates combinations for all sizes in range.")
# print("Example: size=(1,3) generates C(n,1) + C(n,2) + C(n,3).")

# # size=(1,2) from 3 items: C(3,1) + C(3,2) = 3 + 3 = 6
# show_expansion(
#     "size=(1,2) from 3 items",
#     {"_or_": ["A", "B", "C"], "size": (1, 2)}
# )

# # size=(1,3) from 4 items: C(4,1) + C(4,2) + C(4,3) = 4 + 6 + 4 = 14
# show_expansion(
#     "size=(1,3) from 4 items",
#     {"_or_": ["PCA", "SVD", "ICA", "NMF"], "size": (1, 3)}
# )


# # ============================================================
# # Example 3.5: pick keyword (explicit combinations)
# # ============================================================
# print_section("EXAMPLE 3.5: pick keyword (explicit combinations)")
# print("The 'pick' keyword is the explicit version of 'size'.")
# print("It clearly indicates unordered selection (combinations).")
# print(f"Keyword constant: PICK_KEYWORD = '{PICK_KEYWORD}'")

# # pick = 2: same as size = 2
# show_expansion(
#     "pick=2 from 3 items (same as size=2)",
#     {"_or_": ["A", "B", "C"], "pick": 2}
# )

# # pick with range
# show_expansion(
#     "pick=(1,2) from 3 items",
#     {"_or_": ["A", "B", "C"], "pick": (1, 2)}
# )

# # pick in nested dict
# show_expansion(
#     "pick in nested dict",
#     {"transforms": {"_or_": ["PCA", "SVD", "ICA"], "pick": 2}, "model": "PLS"}
# )


# # ============================================================
# # Example 3.6: arrange keyword (explicit permutations)
# # ============================================================
# print_section("EXAMPLE 3.6: arrange keyword (explicit permutations)")
# print("The 'arrange' keyword selects items where ORDER MATTERS.")
# print("Uses mathematical permutations P(n, k).")
# print(f"Keyword constant: ARRANGE_KEYWORD = '{ARRANGE_KEYWORD}'")

# # arrange = 2: P(3,2) = 6 permutations
# show_expansion(
#     "arrange=2 from 3 items (P(3,2)=6)",
#     {"_or_": ["A", "B", "C"], "arrange": 2}
# )

# # arrange with range
# show_expansion(
#     "arrange=(1,2) from 3 items (P(3,1)+P(3,2)=9)",
#     {"_or_": ["A", "B", "C"], "arrange": (1, 2)}
# )


# # ============================================================
# # Example 3.7: pick vs arrange comparison
# # ============================================================
# print_section("EXAMPLE 3.7: pick vs arrange comparison")
# print("Comparing pick (combinations) vs arrange (permutations).")
# print("Key difference: arrange includes both [A,B] and [B,A] as separate results.")

# choices = ["A", "B", "C"]

# print("--- pick=2 (combinations) ---")
# pick_result = expand_spec({"_or_": choices, "pick": 2})
# print(f"Count: {len(pick_result)}")
# print(f"Results: {pick_result}")
# print("Note: [A,B] appears, but [B,A] does NOT (order doesn't matter)")
# print()

# print("--- arrange=2 (permutations) ---")
# arrange_result = expand_spec({"_or_": choices, "arrange": 2})
# print(f"Count: {len(arrange_result)}")
# print(f"Results: {arrange_result}")
# print("Note: BOTH [A,B] AND [B,A] appear (order matters)")
# print()

# print("Use case guidance:")
# print("- Use 'pick' for concat_transform (feature order doesn't matter)")
# print("- Use 'pick' for feature_augmentation (parallel channels)")
# print("- Use 'arrange' for sequential preprocessing steps")
# print("- Use 'arrange' when the order of operations affects the result")
# print()


# # ============================================================
# # Example 4: _range_ for numeric sequences
# # ============================================================
# print_section("EXAMPLE 4: _range_ for numeric sequences")
# print("The _range_ keyword generates a sequence of numbers.")
# print("Supports [start, end] or [start, end, step] syntax.")

# # Basic range [from, to] - inclusive
# show_expansion(
#     "Range [1, 5] (inclusive)",
#     {"_range_": [1, 5]}
# )

# # Range with step
# show_expansion(
#     "Range [0, 20, 5] (step=5)",
#     {"_range_": [0, 20, 5]}
# )

# # Range in dict syntax
# show_expansion(
#     "Range dict syntax",
#     {"_range_": {"from": 10, "to": 50, "step": 10}}
# )


# # ============================================================
# # Example 5: Nested _or_ in dictionaries
# # ============================================================
# print_section("EXAMPLE 5: Nested _or_ in dictionaries")
# print("When _or_ is a value in a dictionary, it expands that key only.")
# print("The Cartesian product is taken across all keys.")

# # Single key with _or_
# show_expansion(
#     "Single key with _or_",
#     {"n_components": {"_or_": [5, 10, 20]}}
# )

# # Multiple keys with _or_ -> Cartesian product
# show_expansion(
#     "Two keys with _or_ (Cartesian product)",
#     {
#         "n_components": {"_or_": [5, 10]},
#         "random_state": {"_or_": [0, 42]}
#     }
# )

# # Mix of _or_ and fixed values
# show_expansion(
#     "Mix of _or_ and fixed values",
#     {
#         "class": "PCA",
#         "n_components": {"_or_": [5, 10, 20]},
#         "whiten": True
#     }
# )

# # _range_ in value position
# show_expansion(
#     "_range_ in value position",
#     {
#         "class": "PLSRegression",
#         "n_components": {"_range_": [2, 10, 2]}
#     }
# )


# # ============================================================
# # Example 6: Complex nested structures
# # ============================================================
# print_section("EXAMPLE 6: Complex nested structures")
# print("Generator syntax works recursively in complex structures.")

# # Nested dicts
# show_expansion(
#     "Nested dictionary structure",
#     {
#         "scaler": {"class": "StandardScaler"},
#         "reducer": {
#             "_or_": [
#                 {"class": "PCA", "n_components": {"_or_": [5, 10]}},
#                 {"class": "SVD", "n_components": {"_or_": [5, 10]}},
#             ]
#         }
#     }
# )

# # List of items with _or_
# show_expansion(
#     "List with _or_ elements",
#     [
#         {"_or_": ["A", "B"]},
#         {"_or_": ["X", "Y"]}
#     ]
# )


# # ============================================================
# # Example 7: count for limiting results
# # ============================================================
# print_section("EXAMPLE 7: count for limiting results")
# print("The 'count' parameter limits the number of results returned.")
# print("Note: Results are randomly sampled, so may vary between runs.")

# # count=2 from many choices
# print("--- count=2 from 5 choices (random sampling) ---")
# spec = {"_or_": ["A", "B", "C", "D", "E"], "count": 2}
# print(f"Spec: {spec}")
# print(f"Count: {count_combinations(spec)}")

# # Run 3 times to show randomness
# for i in range(3):
#     results = expand_spec(spec)
#     print(f"  Run {i+1}: {results}")
# print()

# # count with size
# print("--- count=3 from size=2 combinations ---")
# spec = {"_or_": ["A", "B", "C", "D"], "size": 2, "count": 3}
# print(f"Spec: {spec}")
# print("All combinations would be: C(4,2) = 6")
# print("With count=3, only 3 are returned:")
# results = expand_spec(spec)
# print(f"  Results: {results}")
# print()


# # ============================================================
# # Example 8: Second-order combinations with [outer, inner]
# # ============================================================
# print_section("EXAMPLE 8: Second-order combinations [outer, inner]")
# print("size=[outer, inner] creates nested combinations:")
# print("- inner: permutations (order matters within sub-arrays)")
# print("- outer: combinations (selecting which sub-arrays)")
# print("- Both outer and inner can be int or tuple (range)")

# # [2, 1] from 3 items
# # Inner: P(3,1) = 3 single-element permutations: A, B, C
# # Outer: C(3,2) = 3 ways to pick 2: [A,B], [A,C], [B,C]
# show_expansion(
#     "size=[2, 1] from 3 items",
#     {"_or_": ["A", "B", "C"], "size": [2, 1]}
# )

# # [2, 2] from 3 items
# # Inner: P(3,2) = 6 permutations: [A,B], [A,C], [B,A], [B,C], [C,A], [C,B]
# # Outer: C(6,2) = 15 ways to pick 2 arrangements
# show_expansion(
#     "size=[2, 2] from 3 items",
#     {"_or_": ["A", "B", "C"], "size": [2, 2]},
#     max_show=15
# )

# # [(1,2), 2] - outer range, inner fixed
# # Inner: P(4,2) = 12 permutations of 2 from 4 items
# # Outer: C(12,1) + C(12,2) = 12 + 66 = 78 ways to pick 1 or 2
# print("--- size=[(1,2), 2] - outer range (1-2), inner fixed (2) ---")
# print("Inner: P(4,2) = 12 permutations of size 2")
# print("Outer: pick 1 or 2 of those -> C(12,1) + C(12,2) = 12 + 66 = 78")
# show_expansion(
#     "size=[(1,2), 2] from 4 items",
#     {"_or_": ["A", "B", "C", "D"], "size": [(1, 2), 2]},
#     show_all=False,
#     max_show=10
# )

# # [2, (1,2)] - outer fixed, inner range
# # Inner: P(3,1) + P(3,2) = 3 + 6 = 9 arrangements (sizes 1 or 2)
# # Outer: C(9,2) = 36 ways to pick 2 arrangements
# print("--- size=[2, (1,2)] - outer fixed (2), inner range (1-2) ---")
# print("Inner: P(3,1) + P(3,2) = 3 + 6 = 9 arrangements")
# print("Outer: pick 2 of those -> C(9,2) = 36")
# show_expansion(
#     "size=[2, (1,2)] from 3 items",
#     {"_or_": ["A", "B", "C"], "size": [2, (1, 2)]},
#     show_all=False,
#     max_show=10
# )

# # [(1,2), (1,2)] - both ranges!
# # Inner: P(3,1) + P(3,2) = 3 + 6 = 9 arrangements
# # Outer: C(9,1) + C(9,2) = 9 + 36 = 45 ways to pick 1 or 2
# print("--- size=[(1,2), (1,2)] - both outer and inner are ranges ---")
# print("Inner: P(3,1) + P(3,2) = 3 + 6 = 9 arrangements (A, B, C, [A,B], [A,C], ...)")
# print("Outer: pick 1 or 2 of those -> C(9,1) + C(9,2) = 9 + 36 = 45")
# show_expansion(
#     "size=[(1,2), (1,2)] from 3 items",
#     {"_or_": ["A", "B", "C"], "size": [(1, 2), (1, 2)]},
#     show_all=False,
#     max_show=15
# )


# # ============================================================
# # Example 9: Pipeline-like structures
# # ============================================================
# print_section("EXAMPLE 9: Pipeline-like structures")
# print("Demonstrating generator syntax in realistic pipeline configurations.")

# # Feature augmentation with multiple transformers
# pipeline_spec = {
#     "preprocessing": {
#         "_or_": [
#             {"class": "StandardScaler"},
#             {"class": "MinMaxScaler"},
#         ]
#     },
#     "feature_extraction": {
#         "_or_": [
#             {"class": "PCA", "n_components": {"_or_": [10, 20, 30]}},
#             {"class": "TruncatedSVD", "n_components": {"_or_": [10, 20]}},
#         ]
#     }
# }

# show_expansion("Pipeline with nested choices", pipeline_spec, max_show=15)

# # Concat transform pool
# concat_pool = {
#     "_or_": [
#         {"class": "PCA", "n_components": 10},
#         {"class": "TruncatedSVD", "n_components": 10},
#         {"class": "FastICA", "n_components": 10},
#     ],
#     "size": (1, 2)
# }

# show_expansion(
#     "Concat pool: single or pairs of transformers",
#     concat_pool
# )


# # ============================================================
# # Example 10: Utility functions
# # ============================================================
# print_section("EXAMPLE 10: Utility functions")
# print("Testing the new utility functions from the refactored module.")

# # is_generator_node
# print("--- is_generator_node() ---")
# test_cases = [
#     {"_or_": ["A", "B"]},
#     {"_range_": [1, 10]},
#     {"class": "PCA"},
#     {"n_components": 10},
#     {"_or_": ["A"], "size": 1},
# ]
# for tc in test_cases:
#     print(f"  {tc} -> {is_generator_node(tc)}")
# print()

# # sample_with_seed - deterministic sampling
# print("--- sample_with_seed() with seed (deterministic) ---")
# items = ["A", "B", "C", "D", "E", "F"]
# print(f"Items: {items}")
# for seed in [42, 42, 99]:
#     result = sample_with_seed(items, 3, seed=seed)
#     print(f"  seed={seed}: {result}")
# print()

# # count_combinations for various specs
# print("--- count_combinations() ---")
# specs_to_count = [
#     {"_or_": ["A", "B", "C"]},
#     {"_or_": ["A", "B", "C"], "size": 2},
#     {"_or_": ["A", "B", "C", "D"], "size": (1, 3)},
#     {"_range_": [1, 100]},
#     {"x": {"_or_": [1, 2]}, "y": {"_or_": [3, 4, 5]}},
# ]
# for spec in specs_to_count:
#     count = count_combinations(spec)
#     print(f"  {spec}")
#     print(f"    -> count: {count}")
# print()


# # ============================================================
# # Summary
# # ============================================================
# print_section("SUMMARY")
# print("Generator syntax allows creating multiple configurations from one spec:")
# print()
# print("  _or_: ['A', 'B', 'C']           -> ['A', 'B', 'C'] (3 variants)")
# print("  _or_: [...], size=2             -> combinations of 2 (legacy)")
# print("  _or_: [...], pick=2             -> combinations of 2 (explicit)")
# print("  _or_: [...], arrange=2          -> permutations of 2 (explicit)")
# print("  _or_: [...], size=(1,3)         -> combinations of sizes 1-3")
# print("  _or_: [...], pick=(1,3)         -> combinations of sizes 1-3")
# print("  _or_: [...], arrange=(1,3)      -> permutations of sizes 1-3")
# print("  _range_: [1, 10]                -> [1, 2, ..., 10]")
# print("  _range_: [0, 100, 10]           -> [0, 10, 20, ..., 100]")
# print("  count=N                         -> limit to N random samples")
# print("  size=[outer, inner]             -> second-order nested combinations")
# print()
# print("Selection semantics:")
# print("  pick   -> combinations C(n,k)   -> order doesn't matter")
# print("  arrange -> permutations P(n,k)  -> order matters")
# print()
# print("Nested in dicts: Cartesian product across all keys with generators.")
# print("Nested in lists: Cartesian product across all list elements.")
# print()
# print("Use count_combinations() to get count without generating.")
# print("Use sample_with_seed() for deterministic random sampling.")
# print()
# print("Example completed successfully!")
