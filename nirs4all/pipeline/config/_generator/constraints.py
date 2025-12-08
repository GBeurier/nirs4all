"""Constraint handling for generator expansion.

This module provides constraint evaluation for filtering generated combinations
based on mutual exclusion (_mutex_) and dependency requirements (_requires_).

Constraint Types:
    _mutex_: Mutual exclusion - certain items cannot appear together
    _requires_: Dependencies - if item A is selected, item B must also be

Usage:
    # Items A and B cannot appear together in the same combination
    {"_or_": ["A", "B", "C", "D"], "pick": 2, "_mutex_": [["A", "B"]]}

    # If A is selected, B must also be selected
    {"_or_": ["A", "B", "C", "D"], "pick": 2, "_requires_": [["A", "B"]]}

    # Complex constraints
    {"_or_": ["A", "B", "C", "D"], "pick": 3,
     "_mutex_": [["A", "C"]],
     "_requires_": [["B", "D"]]}
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


def apply_mutex_constraint(
    combinations: List[List[Any]],
    mutex_groups: List[List[Any]]
) -> List[List[Any]]:
    """Filter combinations that violate mutual exclusion constraints.

    A mutex constraint [A, B] means A and B cannot both be present
    in the same combination.

    Args:
        combinations: List of combinations to filter.
        mutex_groups: List of mutex groups. Each group is a list of items
            that cannot appear together in the same combination.

    Returns:
        Filtered list of combinations that satisfy all mutex constraints.

    Examples:
        >>> combos = [["A", "B"], ["A", "C"], ["B", "C"]]
        >>> apply_mutex_constraint(combos, [["A", "B"]])
        [['A', 'C'], ['B', 'C']]

        >>> apply_mutex_constraint(combos, [["A", "B"], ["B", "C"]])
        [['A', 'C']]
    """
    if not mutex_groups:
        return combinations

    result = []
    for combo in combinations:
        if _satisfies_mutex(combo, mutex_groups):
            result.append(combo)
    return result


def _satisfies_mutex(
    combo: List[Any],
    mutex_groups: List[List[Any]]
) -> bool:
    """Check if a combination satisfies all mutex constraints.

    Args:
        combo: A single combination to check.
        mutex_groups: List of mutex groups.

    Returns:
        True if combo satisfies all mutex constraints.
    """
    combo_set = set(_normalize_item(item) for item in combo)

    for mutex_group in mutex_groups:
        mutex_set = set(_normalize_item(item) for item in mutex_group)
        # If all items from mutex group are in combo, it violates the constraint
        if mutex_set.issubset(combo_set):
            return False
    return True


def apply_requires_constraint(
    combinations: List[List[Any]],
    requires_groups: List[List[Any]]
) -> List[List[Any]]:
    """Filter combinations that violate dependency requirements.

    A requires constraint [A, B] means if A is present, B must also be present.
    This is a one-directional dependency from A to B.

    Args:
        combinations: List of combinations to filter.
        requires_groups: List of requirement pairs. Each pair [A, B] means
            if A is selected, B must also be selected.

    Returns:
        Filtered list of combinations that satisfy all requires constraints.

    Examples:
        >>> combos = [["A", "B"], ["A", "C"], ["B", "C"]]
        >>> apply_requires_constraint(combos, [["A", "B"]])
        [['A', 'B'], ['B', 'C']]  # "A, C" removed because A requires B

        >>> # B and C without A is OK because no constraint on B or C
    """
    if not requires_groups:
        return combinations

    result = []
    for combo in combinations:
        if _satisfies_requires(combo, requires_groups):
            result.append(combo)
    return result


def _satisfies_requires(
    combo: List[Any],
    requires_groups: List[List[Any]]
) -> bool:
    """Check if a combination satisfies all requires constraints.

    Args:
        combo: A single combination to check.
        requires_groups: List of requirement pairs [A, B] where A requires B.

    Returns:
        True if combo satisfies all requires constraints.
    """
    combo_set = set(_normalize_item(item) for item in combo)

    for requires_pair in requires_groups:
        if len(requires_pair) < 2:
            continue
        # First item requires subsequent items
        trigger = _normalize_item(requires_pair[0])
        required = set(_normalize_item(item) for item in requires_pair[1:])

        if trigger in combo_set:
            # Trigger is present, check if all required items are present
            if not required.issubset(combo_set):
                return False
    return True


def apply_exclude_constraint(
    combinations: List[List[Any]],
    exclude_combos: List[List[Any]]
) -> List[List[Any]]:
    """Filter specific combinations from results.

    Args:
        combinations: List of combinations to filter.
        exclude_combos: Specific combinations to exclude.

    Returns:
        Filtered list excluding specified combinations.

    Examples:
        >>> combos = [["A", "B"], ["A", "C"], ["B", "C"]]
        >>> apply_exclude_constraint(combos, [["A", "B"]])
        [['A', 'C'], ['B', 'C']]
    """
    if not exclude_combos:
        return combinations

    # Normalize exclude patterns for comparison
    exclude_normalized = [
        frozenset(_normalize_item(item) for item in exc)
        for exc in exclude_combos
    ]

    result = []
    for combo in combinations:
        combo_normalized = frozenset(_normalize_item(item) for item in combo)
        if combo_normalized not in exclude_normalized:
            result.append(combo)
    return result


def apply_all_constraints(
    combinations: List[List[Any]],
    mutex_groups: Optional[List[List[Any]]] = None,
    requires_groups: Optional[List[List[Any]]] = None,
    exclude_combos: Optional[List[List[Any]]] = None
) -> List[List[Any]]:
    """Apply all constraints in sequence.

    Args:
        combinations: List of combinations to filter.
        mutex_groups: Mutual exclusion groups.
        requires_groups: Dependency requirement pairs.
        exclude_combos: Specific combinations to exclude.

    Returns:
        Filtered list satisfying all constraints.
    """
    result = combinations

    if mutex_groups:
        result = apply_mutex_constraint(result, mutex_groups)

    if requires_groups:
        result = apply_requires_constraint(result, requires_groups)

    if exclude_combos:
        result = apply_exclude_constraint(result, exclude_combos)

    return result


def count_with_constraints(
    n: int,
    k: int,
    mutex_groups: Optional[List[List[Any]]] = None,
    requires_groups: Optional[List[List[Any]]] = None
) -> int:
    """Estimate count of combinations after constraint filtering.

    Note: This is an approximation. For exact count, generate and filter.

    Args:
        n: Total number of items.
        k: Size of combinations to select.
        mutex_groups: Mutual exclusion groups.
        requires_groups: Dependency requirement pairs.

    Returns:
        Estimated count of valid combinations.
    """
    from math import comb

    base_count = comb(n, k)

    if not mutex_groups and not requires_groups:
        return base_count

    # For complex constraints, we'd need to use inclusion-exclusion
    # This is a simplified estimate
    # For now, we return the base count as an upper bound
    return base_count


def _normalize_item(item: Any) -> Any:
    """Normalize an item for comparison.

    Converts dicts to frozensets of items and lists to tuples
    so they can be used in set operations.

    Args:
        item: Item to normalize.

    Returns:
        Hashable normalized item.
    """
    if isinstance(item, dict):
        # Convert dict to frozenset of (key, normalized_value) tuples
        return frozenset((k, _normalize_item(v)) for k, v in item.items())
    elif isinstance(item, list):
        return tuple(_normalize_item(i) for i in item)
    elif isinstance(item, set):
        return frozenset(_normalize_item(i) for i in item)
    else:
        return item


def parse_constraints(node: Dict[str, Any]) -> Dict[str, List[List[Any]]]:
    """Extract constraint specifications from a node.

    Args:
        node: Node containing constraint keywords.

    Returns:
        Dict with 'mutex', 'requires', 'exclude' lists.
    """
    return {
        'mutex': node.get('_mutex_', []),
        'requires': node.get('_requires_', []),
        'exclude': node.get('_exclude_', []),
    }


def validate_constraints(
    constraints: Dict[str, List[List[Any]]],
    choices: List[Any]
) -> List[str]:
    """Validate constraint specifications against available choices.

    Args:
        constraints: Constraint dict from parse_constraints.
        choices: Available choice items.

    Returns:
        List of validation error messages.
    """
    errors = []
    normalized_choices = set(_normalize_item(c) for c in choices)

    # Validate mutex groups
    for i, group in enumerate(constraints.get('mutex', [])):
        if not isinstance(group, list):
            errors.append(f"_mutex_[{i}] must be a list")
            continue
        for j, item in enumerate(group):
            if _normalize_item(item) not in normalized_choices:
                errors.append(f"_mutex_[{i}][{j}]: '{item}' not in choices")

    # Validate requires groups
    for i, group in enumerate(constraints.get('requires', [])):
        if not isinstance(group, list):
            errors.append(f"_requires_[{i}] must be a list")
            continue
        if len(group) < 2:
            errors.append(f"_requires_[{i}] must have at least 2 items")
            continue
        for j, item in enumerate(group):
            if _normalize_item(item) not in normalized_choices:
                errors.append(f"_requires_[{i}][{j}]: '{item}' not in choices")

    # Validate exclude patterns
    for i, pattern in enumerate(constraints.get('exclude', [])):
        if not isinstance(pattern, list):
            errors.append(f"_exclude_[{i}] must be a list")
            continue
        for j, item in enumerate(pattern):
            if _normalize_item(item) not in normalized_choices:
                errors.append(f"_exclude_[{i}][{j}]: '{item}' not in choices")

    return errors
