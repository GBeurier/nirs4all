"""
Branch utilities for value mapping and condition parsing.

This module provides utilities for parsing user-friendly value conditions
into callable predicates for branch separation logic.

Supported condition formats:
    - Boolean: True, False
    - List: ["a", "b", "c"] (value in list)
    - Comparison strings: "> 0.8", ">=10", "< 50", "<=100", "== 5", "!= 3"
    - Range strings: "0..50", "50..", "..50" (inclusive on both ends)
    - Lambda/callable: lambda x: x > 0.8

Example usage:
    >>> from nirs4all.controllers.data.branch_utils import parse_value_condition
    >>> pred = parse_value_condition("> 0.8")
    >>> pred(0.9)  # True
    >>> pred(0.5)  # False
    >>> pred = parse_value_condition("0..50")
    >>> pred(25)   # True
    >>> pred(75)   # False
"""

import re
from collections.abc import Callable
from typing import Any, Optional, Union


def parse_value_condition(condition: Any) -> Callable[[Any], bool]:
    """Parse user-friendly value conditions into callable predicates.

    Args:
        condition: The condition to parse. Supported formats:
            - callable: Used directly
            - bool: Equality check (x == condition)
            - list/tuple/set: Membership check (x in condition)
            - str: Parsed as comparison or range (see _parse_string_condition)
            - int/float: Equality check (x == condition)

    Returns:
        A callable that takes a single value and returns bool.

    Raises:
        ValueError: If condition format is not recognized.

    Examples:
        >>> parse_value_condition(True)(True)
        True
        >>> parse_value_condition([1, 2, 3])(2)
        True
        >>> parse_value_condition("> 0.5")(0.8)
        True
        >>> parse_value_condition("10..20")(15)
        True
    """
    if callable(condition):
        return condition

    if isinstance(condition, bool):
        return lambda x: x == condition

    if isinstance(condition, (list, tuple, set)):
        condition_set = set(condition)
        return lambda x: x in condition_set

    if isinstance(condition, str):
        return _parse_string_condition(condition)

    if isinstance(condition, (int, float)):
        return lambda x: x == condition

    raise ValueError(
        f"Unknown condition format: {type(condition).__name__}. "
        f"Expected callable, bool, list, str, int, or float."
    )

def _parse_string_condition(s: str) -> Callable[[Any], bool]:
    """Parse string conditions like '> 0.8', '0..50', etc.

    Supports:
        - Comparison operators: >, >=, <, <=, ==, !=
        - Range syntax: start..end, start.., ..end

    Args:
        s: The string condition to parse.

    Returns:
        A callable predicate.

    Raises:
        ValueError: If the string format is not recognized.
    """
    s = s.strip()

    # Try range syntax first: "0..50", "50..", "..50"
    range_match = re.match(r'^(-?\d*\.?\d*)\s*\.\.\s*(-?\d*\.?\d*)$', s)
    if range_match:
        start_str, end_str = range_match.groups()
        return _create_range_predicate(start_str, end_str)

    # Try comparison syntax: ">= 0.8", "> 10", "< 5", "<= 3", "== 1", "!= 0"
    comparison_match = re.match(r'^(>=|<=|>|<|==|!=)\s*(-?\d+\.?\d*)$', s)
    if comparison_match:
        operator, value_str = comparison_match.groups()
        return _create_comparison_predicate(operator, value_str)

    # Try simple equality (just a number)
    try:
        value = _parse_number(s)
        return lambda x: x == value
    except ValueError:
        pass

    # Try as a literal string value for equality
    return lambda x: x == s

def _create_range_predicate(
    start_str: str,
    end_str: str
) -> Callable[[Any], bool]:
    """Create a predicate for range conditions.

    Ranges are inclusive on both ends (if specified).

    Args:
        start_str: Start of range (empty string for open start)
        end_str: End of range (empty string for open end)

    Returns:
        Range predicate function.

    Raises:
        ValueError: If neither start nor end is specified.
    """
    has_start = bool(start_str.strip())
    has_end = bool(end_str.strip())

    if not has_start and not has_end:
        raise ValueError("Range must have at least a start or end value: '..' is invalid")

    start = _parse_number(start_str) if has_start else None
    end = _parse_number(end_str) if has_end else None

    if has_start and has_end:
        # Closed range: start..end (inclusive)
        return lambda x: start <= x <= end
    elif has_start:
        # Open-ended range: start.. (start and above)
        return lambda x: x >= start
    else:
        # Open-start range: ..end (up to and including end)
        return lambda x: x <= end

def _create_comparison_predicate(
    operator: str,
    value_str: str
) -> Callable[[Any], bool]:
    """Create a predicate for comparison conditions.

    Args:
        operator: One of >, >=, <, <=, ==, !=
        value_str: The value to compare against (as string)

    Returns:
        Comparison predicate function.
    """
    value = _parse_number(value_str)

    operators = {
        '>': lambda x: x > value,
        '>=': lambda x: x >= value,
        '<': lambda x: x < value,
        '<=': lambda x: x <= value,
        '==': lambda x: x == value,
        '!=': lambda x: x != value,
    }

    return operators[operator]

def _parse_number(s: str) -> int | float:
    """Parse a string as int or float.

    Args:
        s: String to parse.

    Returns:
        Parsed number (int if no decimal, float otherwise).

    Raises:
        ValueError: If string cannot be parsed as a number.
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty string cannot be parsed as number")

    if '.' in s:
        return float(s)
    return int(s)

def group_samples_by_value_mapping(
    values: list[Any],
    value_mapping: dict[str, Any]
) -> dict[str, list[int]]:
    """Group sample indices by value mapping conditions.

    Given a list of values and a mapping of branch names to conditions,
    returns a dict mapping branch names to lists of sample indices.

    Args:
        values: List of values (one per sample).
        value_mapping: Dict mapping branch names to conditions.
            Each condition is parsed via parse_value_condition().

    Returns:
        Dict mapping branch names to lists of sample indices.

    Raises:
        ValueError: If samples appear in multiple branches (overlapping conditions).

    Example:
        >>> values = [0.2, 0.5, 0.8, 0.9, 0.3]
        >>> mapping = {"low": "< 0.5", "high": ">= 0.5"}
        >>> group_samples_by_value_mapping(values, mapping)
        {'low': [0, 4], 'high': [1, 2, 3]}
    """
    # Parse all conditions
    predicates = {
        name: parse_value_condition(condition)
        for name, condition in value_mapping.items()
    }

    # Group samples
    result: dict[str, list[int]] = {name: [] for name in value_mapping}
    assigned_samples: dict[int, str] = {}

    for idx, value in enumerate(values):
        for branch_name, predicate in predicates.items():
            if predicate(value):
                if idx in assigned_samples:
                    raise ValueError(
                        f"Sample index {idx} (value={value}) matches multiple branches: "
                        f"'{assigned_samples[idx]}' and '{branch_name}'. "
                        f"Separation branches require disjoint value mappings."
                    )
                result[branch_name].append(idx)
                assigned_samples[idx] = branch_name

    return result

def validate_disjoint_conditions(
    value_mapping: dict[str, Any],
    sample_values: list[Any] | None = None
) -> bool:
    """Validate that value mapping conditions are disjoint.

    This is a best-effort check. For complex conditions (lambdas),
    it will only detect overlaps when sample_values is provided.

    Args:
        value_mapping: Dict mapping branch names to conditions.
        sample_values: Optional sample values to test against.

    Returns:
        True if conditions appear disjoint, False if overlap detected.

    Raises:
        ValueError: If obvious overlap is detected.
    """
    if sample_values is not None:
        # Test with actual samples
        try:
            group_samples_by_value_mapping(sample_values, value_mapping)
            return True
        except ValueError:
            return False

    # Without sample values, we can only do basic checks
    # For now, assume disjoint - actual validation happens at runtime
    return True

__all__ = [
    "parse_value_condition",
    "group_samples_by_value_mapping",
    "validate_disjoint_conditions",
]
