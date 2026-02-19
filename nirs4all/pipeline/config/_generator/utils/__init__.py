"""Utilities package for generator module.

This package provides utility functions for sampling, combinatorics,
export, and other helper operations used in pipeline configuration generation.
"""

from .combinatorics import (
    count_combinations,
    count_combinations_range,
    count_permutations,
    count_permutations_range,
    expand_combination_cartesian,
    generate_cartesian_product,
    generate_combinations,
    generate_combinations_range,
    generate_permutations,
    is_nested_size_spec,
    normalize_size_spec,
)
from .export import (
    ExpansionTreeNode,
    diff_configs,
    format_config_table,
    get_expansion_tree,
    print_expansion_tree,
    summarize_configs,
    to_dataframe,
)
from .sampling import (
    random_choice_with_seed,
    sample_with_seed,
    shuffle_with_seed,
)

__all__ = [
    # Sampling utilities
    "sample_with_seed",
    "shuffle_with_seed",
    "random_choice_with_seed",
    # Combinatorics utilities
    "generate_combinations",
    "generate_combinations_range",
    "generate_permutations",
    "generate_cartesian_product",
    "count_combinations",
    "count_combinations_range",
    "count_permutations",
    "count_permutations_range",
    "normalize_size_spec",
    "is_nested_size_spec",
    "expand_combination_cartesian",
    # Export utilities (Phase 4)
    "to_dataframe",
    "diff_configs",
    "summarize_configs",
    "get_expansion_tree",
    "print_expansion_tree",
    "format_config_table",
    "ExpansionTreeNode",
]
