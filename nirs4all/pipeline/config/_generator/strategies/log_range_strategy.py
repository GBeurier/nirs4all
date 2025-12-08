"""Log-range strategy for logarithmic sequence generation.

This module handles _log_range_ nodes that generate logarithmically-spaced
numeric sequences - useful for hyperparameter optimization (learning rates, etc.)

Syntax:
    {"_log_range_": [from, to, num]}           -> num log-spaced values from from to to
    {"_log_range_": {"from": f, "to": t, "num": n}}
    {"_log_range_": {"from": f, "to": t, "base": b}}
    {"_log_range_": ..., "count": n}           -> Limit to n random samples

Examples:
    {"_log_range_": [0.001, 1, 4]}     -> [0.001, 0.01, 0.1, 1.0]
    {"_log_range_": [1, 1000, 4]}      -> [1, 10, 100, 1000]
"""

import math
from typing import Any, Dict, FrozenSet, List, Optional, Union

from .base import ExpansionStrategy, GeneratorNode, ExpandedResult
from .registry import register_strategy
from ..keywords import LOG_RANGE_KEYWORD, COUNT_KEYWORD, SEED_KEYWORD, PURE_LOG_RANGE_KEYS
from ..utils.sampling import sample_with_seed


@register_strategy
class LogRangeStrategy(ExpansionStrategy):
    """Strategy for handling _log_range_ nodes.

    Generates logarithmically-spaced numeric sequences. Useful for
    hyperparameter search over values that span multiple orders of magnitude.

    Supported formats:
        - Array: [from, to, num] - num values from from to to
        - Dict: {"from": start, "to": end, "num": n}
        - Dict: {"from": start, "to": end, "base": b} - explicit base
        - With count: Limits output to n random samples

    Attributes:
        keywords: {_log_range_, count}
        priority: 25 (checked before range and or strategies)
    """

    keywords: FrozenSet[str] = PURE_LOG_RANGE_KEYS
    priority: int = 25  # High priority

    @classmethod
    def handles(cls, node: GeneratorNode) -> bool:
        """Check if node is a pure log range node.

        A pure log range node contains only _log_range_ and optionally count/seed.

        Args:
            node: Dictionary node to check.

        Returns:
            True if node contains _log_range_ and only log-range-related keys.
        """
        if not isinstance(node, dict):
            return False
        return LOG_RANGE_KEYWORD in node and set(node.keys()).issubset(PURE_LOG_RANGE_KEYS)

    def expand(
        self,
        node: GeneratorNode,
        seed: Optional[int] = None,
        expand_nested: Optional[callable] = None
    ) -> ExpandedResult:
        """Expand a log range node to list of numeric values.

        Args:
            node: Log range specification node.
            seed: Optional seed for random sampling when count is used.
            expand_nested: Not used for log range nodes (no nesting).

        Returns:
            List of logarithmically-spaced numeric values.

        Raises:
            ValueError: If log range specification is invalid.

        Examples:
            >>> strategy.expand({"_log_range_": [0.001, 1, 4]})
            [0.001, 0.01, 0.1, 1.0]
            >>> strategy.expand({"_log_range_": [1, 1000, 4]})
            [1.0, 10.0, 100.0, 1000.0]
        """
        log_range_spec = node[LOG_RANGE_KEYWORD]
        count = node.get(COUNT_KEYWORD)
        node_seed = node.get(SEED_KEYWORD, seed)

        # Generate the full log range
        log_values = self._generate_log_range(log_range_spec)

        # Apply count limit if specified
        if count is not None and len(log_values) > count:
            log_values = sample_with_seed(log_values, count, seed=node_seed)

        return log_values

    def count(self, node: GeneratorNode, count_nested: Optional[callable] = None) -> int:
        """Count log range elements without generating them.

        Args:
            node: Log range specification node.
            count_nested: Not used for log range nodes.

        Returns:
            Number of values in the log range.
        """
        log_range_spec = node[LOG_RANGE_KEYWORD]
        count_limit = node.get(COUNT_KEYWORD)

        # Calculate log range size
        log_range_size = self._count_log_range(log_range_spec)

        # Apply count limit if specified
        if count_limit is not None:
            return min(count_limit, log_range_size)
        return log_range_size

    def validate(self, node: GeneratorNode) -> List[str]:
        """Validate log range node specification.

        Args:
            node: Log range node to validate.

        Returns:
            List of error messages. Empty if valid.
        """
        errors = []
        log_range_spec = node.get(LOG_RANGE_KEYWORD)

        if log_range_spec is None:
            errors.append("Missing _log_range_ key")
            return errors

        # Validate array syntax
        if isinstance(log_range_spec, list):
            if len(log_range_spec) != 3:
                errors.append(
                    f"Log range array must have 3 elements [from, to, num], got {len(log_range_spec)}"
                )
            elif not all(isinstance(x, (int, float)) for x in log_range_spec):
                errors.append("Log range array elements must be numeric")
            else:
                start, end, num = log_range_spec
                if start <= 0 or end <= 0:
                    errors.append("Log range start and end must be positive")
                if not isinstance(num, int) or num < 1:
                    errors.append("Log range num must be a positive integer")

        # Validate dict syntax
        elif isinstance(log_range_spec, dict):
            required = {"from", "to"}
            missing = required - set(log_range_spec.keys())
            if missing:
                errors.append(f"Log range dict missing required keys: {missing}")

            for key in ("from", "to"):
                if key in log_range_spec:
                    val = log_range_spec[key]
                    if not isinstance(val, (int, float)):
                        errors.append(f"Log range '{key}' must be numeric")
                    elif val <= 0:
                        errors.append(f"Log range '{key}' must be positive")

            if "num" in log_range_spec:
                num = log_range_spec["num"]
                if not isinstance(num, int) or num < 1:
                    errors.append("Log range 'num' must be a positive integer")

            if "base" in log_range_spec:
                base = log_range_spec["base"]
                if not isinstance(base, (int, float)) or base <= 0 or base == 1:
                    errors.append("Log range 'base' must be a positive number != 1")
        else:
            errors.append(
                f"Log range spec must be array or dict, got {type(log_range_spec).__name__}"
            )

        # Validate count if present
        count = node.get(COUNT_KEYWORD)
        if count is not None and not isinstance(count, int):
            errors.append(f"count must be an integer, got {type(count).__name__}")

        return errors

    def _generate_log_range(
        self, log_range_spec: Union[list, Dict[str, Any]]
    ) -> List[float]:
        """Generate logarithmically-spaced values from specification.

        Args:
            log_range_spec: Log range specification (list or dict).

        Returns:
            List of logarithmically-spaced values.

        Raises:
            ValueError: If specification format is invalid.
        """
        if isinstance(log_range_spec, list):
            if len(log_range_spec) != 3:
                raise ValueError(
                    "Log range array must be [from, to, num]"
                )
            start, end, num = log_range_spec
            base = 10  # Default base
        elif isinstance(log_range_spec, dict):
            start = log_range_spec["from"]
            end = log_range_spec["to"]
            num = log_range_spec.get("num", 10)  # Default 10 values
            base = log_range_spec.get("base", 10)  # Default base 10
        else:
            raise ValueError(
                "Log range specification must be array [from, to, num] or "
                "dict {'from': start, 'to': end, 'num': n}"
            )

        if start <= 0 or end <= 0:
            raise ValueError("Log range start and end must be positive")

        if num < 1:
            return []

        if num == 1:
            return [float(start)]

        # Generate logarithmically-spaced values
        log_start = math.log(start, base)
        log_end = math.log(end, base)
        step = (log_end - log_start) / (num - 1)

        result = []
        for i in range(num):
            log_val = log_start + i * step
            val = base ** log_val
            # Round to reasonable precision
            result.append(round(val, 10))

        return result

    def _count_log_range(self, log_range_spec: Union[list, Dict[str, Any]]) -> int:
        """Count elements in a log range without generating them.

        Args:
            log_range_spec: Log range specification.

        Returns:
            Number of elements in the log range.
        """
        if isinstance(log_range_spec, list):
            if len(log_range_spec) != 3:
                raise ValueError("Log range array must be [from, to, num]")
            return int(log_range_spec[2])
        elif isinstance(log_range_spec, dict):
            return log_range_spec.get("num", 10)
        else:
            raise ValueError("Log range specification must be array or dict")
