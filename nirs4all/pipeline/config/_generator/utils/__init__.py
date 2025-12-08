"""Utilities package for generator module.

This package provides utility functions for sampling, combinatorics,
export, and other helper operations used in pipeline configuration generation.
"""

from .sampling import (
    sample_with_seed,
    shuffle_with_seed,
    random_choice_with_seed,
)
from .combinatorics import (
    generate_combinations,
    generate_combinations_range,
    generate_permutations,
    generate_cartesian_product,
    count_combinations,
    count_combinations_range,
    count_permutations,
    count_permutations_range,
    normalize_size_spec,
    is_nested_size_spec,
    expand_combination_cartesian,
)
from .export import (
    to_dataframe,
    diff_configs,
    summarize_configs,
    get_expansion_tree,
    print_expansion_tree,
    format_config_table,
    ExpansionTreeNode,
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
