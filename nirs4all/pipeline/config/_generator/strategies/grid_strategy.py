"""Grid strategy for grid-search style parameter expansion.

This module handles _grid_ nodes that generate Cartesian products of
parameter spaces - useful for hyperparameter grid search.

Syntax:
    {"_grid_": {"param1": [...], "param2": [...]}}

Examples:
    {"_grid_": {"lr": [0.01, 0.1], "batch_size": [16, 32]}}
    -> [{"lr": 0.01, "batch_size": 16}, {"lr": 0.01, "batch_size": 32},
        {"lr": 0.1, "batch_size": 16}, {"lr": 0.1, "batch_size": 32}]
"""

from itertools import product
from typing import Any, Dict, FrozenSet, List, Optional

from .base import ExpansionStrategy, GeneratorNode, ExpandedResult
from .registry import register_strategy
from ..keywords import GRID_KEYWORD, COUNT_KEYWORD, SEED_KEYWORD, PURE_GRID_KEYS
from ..utils.sampling import sample_with_seed


@register_strategy
class GridStrategy(ExpansionStrategy):
    """Strategy for handling _grid_ nodes.

    Generates all combinations (Cartesian product) of parameter values.
    Similar to sklearn's ParameterGrid.

    Supported formats:
        - Dict: {"param1": [v1, v2], "param2": [v3, v4]}
        - With count: Limits output to n random samples

    Attributes:
        keywords: {_grid_, count}
        priority: 30 (checked early due to specific structure)
    """

    keywords: FrozenSet[str] = PURE_GRID_KEYS
    priority: int = 30  # High priority

    @classmethod
    def handles(cls, node: GeneratorNode) -> bool:
        """Check if node is a pure grid node.

        Args:
            node: Dictionary node to check.

        Returns:
            True if node contains _grid_ and only grid-related keys.
        """
        if not isinstance(node, dict):
            return False
        return GRID_KEYWORD in node and set(node.keys()).issubset(PURE_GRID_KEYS)

    def expand(
        self,
        node: GeneratorNode,
        seed: Optional[int] = None,
        expand_nested: Optional[callable] = None
    ) -> ExpandedResult:
        """Expand a grid node to list of parameter combinations.

        Args:
            node: Grid specification node.
            seed: Optional seed for random sampling when count is used.
            expand_nested: Callback to expand nested generator nodes.

        Returns:
            List of dicts with all parameter combinations.

        Examples:
            >>> strategy.expand({"_grid_": {"x": [1, 2], "y": ["A", "B"]}})
            [{"x": 1, "y": "A"}, {"x": 1, "y": "B"}, {"x": 2, "y": "A"}, {"x": 2, "y": "B"}]
        """
        grid_spec = node[GRID_KEYWORD]
        count = node.get(COUNT_KEYWORD)
        node_seed = node.get(SEED_KEYWORD, seed)

        if not isinstance(grid_spec, dict):
            raise ValueError(
                f"_grid_ must be a dict of param: values, got {type(grid_spec).__name__}"
            )

        # Handle empty grid
        if not grid_spec:
            return [{}]

        # Expand nested generators in values first
        expanded_grid = {}
        for key, values in grid_spec.items():
            if expand_nested and isinstance(values, (dict, list)):
                # If value is a generator node or list, expand it
                if isinstance(values, dict):
                    expanded_values = expand_nested(values)
                else:
                    # For lists, we assume they're the parameter values directly
                    # unless they contain generator nodes
                    if any(isinstance(v, dict) for v in values):
                        expanded_values = []
                        for v in values:
                            if isinstance(v, dict):
                                expanded_values.extend(expand_nested(v))
                            else:
                                expanded_values.append(v)
                    else:
                        expanded_values = values
            else:
                if not isinstance(values, list):
                    values = [values]
                expanded_values = values
            expanded_grid[key] = expanded_values

        # Generate Cartesian product
        keys = list(expanded_grid.keys())
        value_lists = [expanded_grid[k] for k in keys]

        results = []
        for combo in product(*value_lists):
            result_dict = dict(zip(keys, combo))
            results.append(result_dict)

        # Apply count limit if specified
        if count is not None and len(results) > count:
            results = sample_with_seed(results, count, seed=node_seed)

        return results

    def count(self, node: GeneratorNode, count_nested: Optional[callable] = None) -> int:
        """Count grid combinations without generating them.

        Args:
            node: Grid specification node.
            count_nested: Callback to count nested nodes.

        Returns:
            Number of parameter combinations.
        """
        grid_spec = node[GRID_KEYWORD]
        count_limit = node.get(COUNT_KEYWORD)

        if not isinstance(grid_spec, dict):
            return 0

        if not grid_spec:
            return 1  # Empty grid produces one empty result

        # Count total combinations
        total = 1
        for key, values in grid_spec.items():
            if count_nested and isinstance(values, dict):
                val_count = count_nested(values)
            elif isinstance(values, list):
                val_count = len(values)
            else:
                val_count = 1
            total *= val_count

        # Apply count limit
        if count_limit is not None:
            return min(count_limit, total)
        return total

    def validate(self, node: GeneratorNode) -> List[str]:
        """Validate grid node specification.

        Args:
            node: Grid node to validate.

        Returns:
            List of error messages. Empty if valid.
        """
        errors = []
        grid_spec = node.get(GRID_KEYWORD)

        if grid_spec is None:
            errors.append("Missing _grid_ key")
            return errors

        if not isinstance(grid_spec, dict):
            errors.append(
                f"_grid_ must be a dict, got {type(grid_spec).__name__}"
            )
            return errors

        # Validate each parameter
        for key, values in grid_spec.items():
            if not isinstance(key, str):
                errors.append(f"Grid keys must be strings, got {type(key).__name__}")
            if not isinstance(values, (list, dict)):
                # Allow scalar values (treated as single-item list)
                pass

        # Validate count
        count = node.get(COUNT_KEYWORD)
        if count is not None and not isinstance(count, int):
            errors.append(f"count must be an integer, got {type(count).__name__}")

        return errors
