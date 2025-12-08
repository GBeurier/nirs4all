"""Zip strategy for parallel iteration.

This module handles _zip_ nodes that iterate over multiple parameter lists
in parallel (like Python's zip function).

Syntax:
    {"_zip_": {"param1": [v1, v2], "param2": [v3, v4]}}

Examples:
    {"_zip_": {"lr": [0.01, 0.1], "batch_size": [16, 32]}}
    -> [{"lr": 0.01, "batch_size": 16}, {"lr": 0.1, "batch_size": 32}]

Unlike _grid_ which generates all combinations, _zip_ pairs values by position.
"""

from typing import Any, Dict, FrozenSet, List, Optional

from .base import ExpansionStrategy, GeneratorNode, ExpandedResult
from .registry import register_strategy
from ..keywords import ZIP_KEYWORD, COUNT_KEYWORD, SEED_KEYWORD, PURE_ZIP_KEYS
from ..utils.sampling import sample_with_seed


@register_strategy
class ZipStrategy(ExpansionStrategy):
    """Strategy for handling _zip_ nodes.

    Generates configurations by pairing values at the same index
    from multiple parameter lists (like Python's zip).

    Supported formats:
        - Dict: {"param1": [v1, v2], "param2": [v3, v4]}
        - With count: Limits output to n random samples

    Attributes:
        keywords: {_zip_, count}
        priority: 28 (between grid and log_range)
    """

    keywords: FrozenSet[str] = PURE_ZIP_KEYS
    priority: int = 28

    @classmethod
    def handles(cls, node: GeneratorNode) -> bool:
        """Check if node is a pure zip node.

        Args:
            node: Dictionary node to check.

        Returns:
            True if node contains _zip_ and only zip-related keys.
        """
        if not isinstance(node, dict):
            return False
        return ZIP_KEYWORD in node and set(node.keys()).issubset(PURE_ZIP_KEYS)

    def expand(
        self,
        node: GeneratorNode,
        seed: Optional[int] = None,
        expand_nested: Optional[callable] = None
    ) -> ExpandedResult:
        """Expand a zip node to list of paired parameter values.

        Args:
            node: Zip specification node.
            seed: Optional seed for random sampling when count is used.
            expand_nested: Callback to expand nested generator nodes.

        Returns:
            List of dicts with paired parameter values.

        Examples:
            >>> strategy.expand({"_zip_": {"x": [1, 2, 3], "y": ["A", "B", "C"]}})
            [{"x": 1, "y": "A"}, {"x": 2, "y": "B"}, {"x": 3, "y": "C"}]
        """
        zip_spec = node[ZIP_KEYWORD]
        count = node.get(COUNT_KEYWORD)
        node_seed = node.get(SEED_KEYWORD, seed)

        if not isinstance(zip_spec, dict):
            raise ValueError(
                f"_zip_ must be a dict of param: values, got {type(zip_spec).__name__}"
            )

        # Handle empty zip
        if not zip_spec:
            return [{}]

        # Expand nested generators and normalize values
        expanded_zip = {}
        for key, values in zip_spec.items():
            if expand_nested and isinstance(values, dict):
                expanded_values = expand_nested(values)
            elif isinstance(values, list):
                expanded_values = values
            else:
                expanded_values = [values]
            expanded_zip[key] = expanded_values

        # Get minimum length (zip stops at shortest list)
        min_len = min(len(v) for v in expanded_zip.values())

        if min_len == 0:
            return [{}]

        # Generate zipped results
        keys = list(expanded_zip.keys())
        results = []
        for i in range(min_len):
            result_dict = {k: expanded_zip[k][i] for k in keys}
            results.append(result_dict)

        # Apply count limit if specified
        if count is not None and len(results) > count:
            results = sample_with_seed(results, count, seed=node_seed)

        return results

    def count(self, node: GeneratorNode, count_nested: Optional[callable] = None) -> int:
        """Count zip pairs without generating them.

        Args:
            node: Zip specification node.
            count_nested: Callback to count nested nodes.

        Returns:
            Number of zipped pairs (minimum list length).
        """
        zip_spec = node[ZIP_KEYWORD]
        count_limit = node.get(COUNT_KEYWORD)

        if not isinstance(zip_spec, dict):
            return 0

        if not zip_spec:
            return 1

        # Count based on shortest list
        lengths = []
        for key, values in zip_spec.items():
            if count_nested and isinstance(values, dict):
                lengths.append(count_nested(values))
            elif isinstance(values, list):
                lengths.append(len(values))
            else:
                lengths.append(1)

        total = min(lengths) if lengths else 0

        # Apply count limit
        if count_limit is not None:
            return min(count_limit, total)
        return total

    def validate(self, node: GeneratorNode) -> List[str]:
        """Validate zip node specification.

        Args:
            node: Zip node to validate.

        Returns:
            List of error messages. Empty if valid.
        """
        errors = []
        zip_spec = node.get(ZIP_KEYWORD)

        if zip_spec is None:
            errors.append("Missing _zip_ key")
            return errors

        if not isinstance(zip_spec, dict):
            errors.append(f"_zip_ must be a dict, got {type(zip_spec).__name__}")
            return errors

        # Check for consistent lengths (warning, not error)
        lengths = {}
        for key, values in zip_spec.items():
            if isinstance(values, list):
                lengths[key] = len(values)

        if lengths and len(set(lengths.values())) > 1:
            # Different lengths - warn
            pass  # Could add warning here

        # Validate count
        count = node.get(COUNT_KEYWORD)
        if count is not None and not isinstance(count, int):
            errors.append(f"count must be an integer, got {type(count).__name__}")

        return errors
