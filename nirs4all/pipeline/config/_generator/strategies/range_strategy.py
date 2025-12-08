"""Range strategy for numeric sequence generation.

This module handles _range_ nodes that generate numeric sequences.

Syntax:
    {"_range_": [from, to]}           -> [from, from+1, ..., to]
    {"_range_": [from, to, step]}     -> [from, from+step, ..., <=to]
    {"_range_": {"from": f, "to": t, "step": s}}
    {"_range_": ..., "count": n}      -> Limit to n random samples
"""

from typing import Any, Dict, FrozenSet, List, Optional, Union

from .base import ExpansionStrategy, GeneratorNode, ExpandedResult
from .registry import register_strategy
from ..keywords import RANGE_KEYWORD, COUNT_KEYWORD, PURE_RANGE_KEYS
from ..utils.sampling import sample_with_seed


@register_strategy
class RangeStrategy(ExpansionStrategy):
    """Strategy for handling _range_ nodes.

    Generates numeric sequences based on range specifications.

    Supported formats:
        - Array: [from, to] or [from, to, step]
        - Dict: {"from": start, "to": end, "step": step}
        - With count: Limits output to n random samples

    Attributes:
        keywords: {_range_, count}
        priority: 20 (checked before OrStrategy)
    """

    keywords: FrozenSet[str] = PURE_RANGE_KEYS
    priority: int = 20  # Higher priority - check range before or

    @classmethod
    def handles(cls, node: GeneratorNode) -> bool:
        """Check if node is a pure range node.

        A pure range node contains only _range_ and optionally count.

        Args:
            node: Dictionary node to check.

        Returns:
            True if node contains _range_ and only range-related keys.
        """
        if not isinstance(node, dict):
            return False
        return RANGE_KEYWORD in node and set(node.keys()).issubset(PURE_RANGE_KEYS)

    def expand(
        self,
        node: GeneratorNode,
        seed: Optional[int] = None,
        expand_nested: Optional[callable] = None
    ) -> ExpandedResult:
        """Expand a range node to list of numeric values.

        Args:
            node: Range specification node.
            seed: Optional seed for random sampling when count is used.
            expand_nested: Not used for range nodes (no nesting).

        Returns:
            List of numeric values.

        Raises:
            ValueError: If range specification is invalid.

        Examples:
            >>> strategy.expand({"_range_": [1, 5]})
            [1, 2, 3, 4, 5]
            >>> strategy.expand({"_range_": [0, 10, 2]})
            [0, 2, 4, 6, 8, 10]
        """
        range_spec = node[RANGE_KEYWORD]
        count = node.get(COUNT_KEYWORD)

        # Generate the full range
        range_values = self._generate_range(range_spec)

        # Apply count limit if specified
        if count is not None and len(range_values) > count:
            range_values = sample_with_seed(range_values, count, seed=seed)

        return range_values

    def count(self, node: GeneratorNode, count_nested: Optional[callable] = None) -> int:
        """Count range elements without generating them.

        Args:
            node: Range specification node.
            count_nested: Not used for range nodes.

        Returns:
            Number of values in the range.

        Raises:
            ValueError: If range specification is invalid.
        """
        range_spec = node[RANGE_KEYWORD]
        count_limit = node.get(COUNT_KEYWORD)

        # Calculate range size
        range_size = self._count_range(range_spec)

        # Apply count limit if specified
        if count_limit is not None:
            return min(count_limit, range_size)
        return range_size

    def validate(self, node: GeneratorNode) -> List[str]:
        """Validate range node specification.

        Args:
            node: Range node to validate.

        Returns:
            List of error messages. Empty if valid.
        """
        errors = []
        range_spec = node.get(RANGE_KEYWORD)

        if range_spec is None:
            errors.append("Missing _range_ key")
            return errors

        # Validate array syntax
        if isinstance(range_spec, list):
            if len(range_spec) not in (2, 3):
                errors.append(
                    f"Range array must have 2 or 3 elements, got {len(range_spec)}"
                )
            elif not all(isinstance(x, (int, float)) for x in range_spec):
                errors.append("Range array elements must be numeric")
        # Validate dict syntax
        elif isinstance(range_spec, dict):
            required = {"from", "to"}
            missing = required - set(range_spec.keys())
            if missing:
                errors.append(f"Range dict missing required keys: {missing}")
        else:
            errors.append(
                f"Range spec must be array or dict, got {type(range_spec).__name__}"
            )

        # Validate count if present
        count = node.get(COUNT_KEYWORD)
        if count is not None and not isinstance(count, int):
            errors.append(f"count must be an integer, got {type(count).__name__}")

        return errors

    def _generate_range(
        self, range_spec: Union[list, Dict[str, Any]]
    ) -> List[Union[int, float]]:
        """Generate numeric range from specification.

        Args:
            range_spec: Range specification (list or dict).

        Returns:
            List of numeric values.

        Raises:
            ValueError: If specification format is invalid.
        """
        if isinstance(range_spec, list):
            if len(range_spec) == 2:
                start, end = range_spec
                step = 1
            elif len(range_spec) == 3:
                start, end, step = range_spec
            else:
                raise ValueError(
                    "Range array must be [from, to] or [from, to, step]"
                )
        elif isinstance(range_spec, dict):
            start = range_spec["from"]
            end = range_spec["to"]
            step = range_spec.get("step", 1)
        else:
            raise ValueError(
                "Range specification must be array [from, to, step] or "
                "dict {'from': start, 'to': end, 'step': step}"
            )

        # Handle float ranges
        if any(isinstance(x, float) for x in (start, end, step)):
            return self._generate_float_range(start, end, step)

        # Generate integer range - end is inclusive
        if step > 0:
            return list(range(start, end + 1, step))
        else:
            # For negative steps, ensure end is included
            return list(range(start, end - 1, step))

    def _generate_float_range(
        self, start: float, end: float, step: float
    ) -> List[float]:
        """Generate float range with inclusive end.

        Args:
            start: Start value.
            end: End value (inclusive).
            step: Step size.

        Returns:
            List of float values.
        """
        result = []
        current = start

        if step > 0:
            while current <= end + 1e-10:  # Small epsilon for float comparison
                result.append(round(current, 10))  # Round to avoid float precision issues
                current += step
        else:
            while current >= end - 1e-10:
                result.append(round(current, 10))
                current += step

        return result

    def _count_range(self, range_spec: Union[list, Dict[str, Any]]) -> int:
        """Count elements in a numeric range without generating them.

        Args:
            range_spec: Range specification.

        Returns:
            Number of elements in the range.
        """
        if isinstance(range_spec, list):
            if len(range_spec) == 2:
                start, end = range_spec
                step = 1
            elif len(range_spec) == 3:
                start, end, step = range_spec
            else:
                raise ValueError(
                    "Range array must be [from, to] or [from, to, step]"
                )
        elif isinstance(range_spec, dict):
            start = range_spec["from"]
            end = range_spec["to"]
            step = range_spec.get("step", 1)
        else:
            raise ValueError(
                "Range specification must be array [from, to, step] or "
                "dict {'from': start, 'to': end, 'step': step}"
            )

        # Calculate count: (end - start) / step + 1 (end is inclusive)
        if step > 0 and end >= start:
            return int((end - start) / step) + 1
        elif step < 0 and end <= start:
            return int((start - end) / abs(step)) + 1
        else:
            return 0
