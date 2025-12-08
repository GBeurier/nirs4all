"""Combinatorics utilities for generator module.

This module provides helper functions for computing combinations,
permutations, and related operations used in pipeline generation.
"""

from itertools import combinations, permutations, product
from math import comb, factorial
from typing import Any, Iterator, List, Optional, Tuple, TypeVar, Union

T = TypeVar('T')


def generate_combinations(
    items: List[T],
    size: int
) -> Iterator[Tuple[T, ...]]:
    """Generate all combinations of items with given size.

    Args:
        items: List of items to combine.
        size: Size of each combination.

    Yields:
        Tuples of items representing each combination.

    Examples:
        >>> list(generate_combinations(["A", "B", "C"], 2))
        [('A', 'B'), ('A', 'C'), ('B', 'C')]
    """
    if size > len(items):
        return
    yield from combinations(items, size)


def generate_combinations_range(
    items: List[T],
    size_range: Tuple[int, int]
) -> Iterator[Tuple[T, ...]]:
    """Generate combinations for all sizes in a range.

    Args:
        items: List of items to combine.
        size_range: Tuple of (from_size, to_size), inclusive on both ends.

    Yields:
        Tuples of items representing each combination for all sizes.

    Examples:
        >>> list(generate_combinations_range(["A", "B", "C"], (1, 2)))
        [('A',), ('B',), ('C',), ('A', 'B'), ('A', 'C'), ('B', 'C')]
    """
    from_size, to_size = size_range
    for size in range(from_size, to_size + 1):
        if size <= len(items):
            yield from combinations(items, size)


def generate_permutations(
    items: List[T],
    size: Optional[int] = None
) -> Iterator[Tuple[T, ...]]:
    """Generate all permutations of items.

    Args:
        items: List of items to permute.
        size: Size of each permutation. If None, uses full length.

    Yields:
        Tuples of items representing each permutation.

    Examples:
        >>> list(generate_permutations(["A", "B"], 2))
        [('A', 'B'), ('B', 'A')]
    """
    if size is None:
        size = len(items)
    if size > len(items):
        return
    yield from permutations(items, size)


def generate_cartesian_product(*iterables: List[Any]) -> Iterator[Tuple[Any, ...]]:
    """Generate cartesian product of multiple iterables.

    Args:
        *iterables: Variable number of iterables to combine.

    Yields:
        Tuples representing each combination from the cartesian product.

    Examples:
        >>> list(generate_cartesian_product([1, 2], ["A", "B"]))
        [(1, 'A'), (1, 'B'), (2, 'A'), (2, 'B')]
    """
    yield from product(*iterables)


def count_combinations(n: int, k: int) -> int:
    """Count combinations C(n, k) = n! / (k! * (n-k)!).

    Args:
        n: Total number of items.
        k: Size of each combination.

    Returns:
        Number of possible combinations.

    Examples:
        >>> count_combinations(5, 2)
        10
        >>> count_combinations(3, 3)
        1
    """
    if k > n or k < 0:
        return 0
    return comb(n, k)


def count_combinations_range(n: int, size_range: Tuple[int, int]) -> int:
    """Count total combinations for all sizes in a range.

    Args:
        n: Total number of items.
        size_range: Tuple of (from_size, to_size), inclusive.

    Returns:
        Total number of combinations across all sizes.

    Examples:
        >>> count_combinations_range(4, (1, 2))
        10  # C(4,1) + C(4,2) = 4 + 6
    """
    from_size, to_size = size_range
    total = 0
    for size in range(from_size, to_size + 1):
        if size <= n:
            total += comb(n, size)
    return total


def count_permutations(n: int, k: int) -> int:
    """Count permutations P(n, k) = n! / (n-k)!.

    Args:
        n: Total number of items.
        k: Size of each permutation.

    Returns:
        Number of possible permutations.

    Examples:
        >>> count_permutations(5, 2)
        20  # 5 * 4
        >>> count_permutations(3, 3)
        6  # 3!
    """
    if k > n or k < 0:
        return 0
    return factorial(n) // factorial(n - k)


def count_permutations_range(n: int, size_range: Tuple[int, int]) -> int:
    """Count total permutations for all sizes in a range.

    Args:
        n: Total number of items.
        size_range: Tuple of (from_size, to_size), inclusive.

    Returns:
        Total number of permutations across all sizes.

    Examples:
        >>> count_permutations_range(3, (1, 2))
        9  # P(3,1) + P(3,2) = 3 + 6
    """
    from_size, to_size = size_range
    total = 0
    for size in range(from_size, to_size + 1):
        if size <= n:
            total += count_permutations(n, size)
    return total


def normalize_size_spec(
    size: Union[int, Tuple[int, int], List[int]]
) -> Tuple[int, int]:
    """Normalize size specification to (from, to) tuple.

    Handles various size specifications and normalizes them to
    a consistent tuple format.

    Args:
        size: Size specification - can be:
            - int: Single size (from=size, to=size)
            - tuple: (from, to) range
            - list: [from, to] for first-order, [outer, inner] handled separately

    Returns:
        Tuple of (from_size, to_size).

    Raises:
        ValueError: If size specification is invalid.

    Examples:
        >>> normalize_size_spec(3)
        (3, 3)
        >>> normalize_size_spec((1, 3))
        (1, 3)
    """
    if isinstance(size, int):
        return (size, size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        return (size[0], size[1])
    else:
        raise ValueError(f"Invalid size specification: {size}")


def is_nested_size_spec(size: Any) -> bool:
    """Check if size specification is for nested combinations [outer, inner].

    The nested size spec uses a list with two elements where each element
    represents the outer and inner size specifications.

    Args:
        size: Size specification to check.

    Returns:
        True if this is a nested size specification (list of length 2),
        False otherwise.

    Examples:
        >>> is_nested_size_spec([2, 3])
        True
        >>> is_nested_size_spec((2, 3))
        False  # Tuple is range, not nested
        >>> is_nested_size_spec(2)
        False
    """
    return isinstance(size, list) and len(size) == 2


def expand_combination_cartesian(
    combo: Tuple[Any, ...],
    expand_func: callable
) -> List[List[Any]]:
    """Expand a combination by taking cartesian product of expanded elements.

    This is used when each element of a combination can itself expand
    into multiple values.

    Args:
        combo: Tuple of items to expand.
        expand_func: Function to expand each item, returns list of values.

    Returns:
        List of lists, each being one expanded combination.

    Examples:
        >>> expand_combination_cartesian(
        ...     ({"_or_": ["A", "B"]}, "C"),
        ...     lambda x: [x] if isinstance(x, str) else ["A", "B"]
        ... )
        [['A', 'C'], ['B', 'C']]
    """
    expanded_elements = [expand_func(item) for item in combo]
    results = []
    for expanded_combo in product(*expanded_elements):
        results.append(list(expanded_combo))
    return results
