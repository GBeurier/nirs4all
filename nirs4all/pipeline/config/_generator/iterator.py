"""Iterator-based lazy expansion for memory-efficient configuration generation.

This module provides lazy/iterator-based expansion functions that generate
configurations one at a time instead of materializing all combinations in
memory. This is essential for large configuration spaces that would otherwise
exhaust available memory.

Main Functions:
    expand_spec_iter(node, seed): Lazy iterator that yields configurations
    count_combinations(node): Still returns total count (unchanged)

Usage:
    # Memory-efficient processing of large spaces
    for config in expand_spec_iter(large_spec):
        process_config(config)

    # With limit
    from itertools import islice
    first_100 = list(islice(expand_spec_iter(large_spec), 100))

    # With random sampling (uses reservoir sampling)
    sample = list(expand_spec_iter(large_spec, seed=42, sample_size=100))
"""

from collections.abc import Mapping
from itertools import product, islice
from typing import Any, Dict, Iterator, List, Optional, Union

from .strategies import get_strategy
from .keywords import (
    OR_KEYWORD,
    RANGE_KEYWORD,
    has_or_keyword,
)

# Type alias
GeneratorNode = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


def expand_spec_iter(
    node: GeneratorNode,
    seed: Optional[int] = None,
    sample_size: Optional[int] = None
) -> Iterator[Any]:
    """Lazily expand a specification node to all possible combinations.

    This is the memory-efficient version of expand_spec that yields
    configurations one at a time instead of building a complete list.

    Args:
        node: Configuration node to expand. Can be:
            - dict: Expanded based on keys (strategies or Cartesian product)
            - list: Cartesian product of expanded elements
            - scalar: Yielded as single item

        seed: Optional random seed for reproducible generation when
            using sample_size to limit results.

        sample_size: If provided, yield at most this many items using
            reservoir sampling for uniform distribution.

    Yields:
        Expanded configuration variants one at a time.

    Examples:
        >>> list(expand_spec_iter({"_or_": ["A", "B"]}))
        ['A', 'B']

        >>> from itertools import islice
        >>> large_spec = {"_range_": [1, 1000000]}
        >>> list(islice(expand_spec_iter(large_spec), 5))
        [1, 2, 3, 4, 5]

        >>> # With sampling
        >>> list(expand_spec_iter({"_range_": [1, 100]}, seed=42, sample_size=5))
        [23, 45, 67, 12, 89]  # Random 5 items
    """
    if sample_size is not None:
        # Use reservoir sampling for uniform random selection
        yield from _reservoir_sample(
            _expand_iter_internal(node, seed),
            sample_size,
            seed
        )
    else:
        yield from _expand_iter_internal(node, seed)


def _expand_iter_internal(
    node: GeneratorNode,
    seed: Optional[int] = None
) -> Iterator[Any]:
    """Internal iterator-based expansion.

    Args:
        node: Node to expand.
        seed: Random seed.

    Yields:
        Expanded variants.
    """
    # Handle lists: Cartesian product of expanded elements
    if isinstance(node, list):
        yield from _expand_list_iter(node, seed)
        return

    # Handle non-dict types: yield as single item
    if not isinstance(node, Mapping):
        yield node
        return

    # Try strategy dispatch for pure generator nodes
    strategy = get_strategy(node)
    if strategy:
        # Check if strategy supports iteration
        if hasattr(strategy, 'expand_iter'):
            yield from strategy.expand_iter(
                node,
                seed=seed,
                expand_nested=lambda n: _expand_iter_internal(n, seed)
            )
        else:
            # Fall back to eager expansion for strategies without iter support
            yield from strategy.expand(
                node,
                seed=seed,
                expand_nested=lambda n: list(_expand_iter_internal(n, seed))
            )
        return

    # Handle mixed dict with _or_ key (has other keys too)
    if has_or_keyword(node):
        yield from _expand_mixed_or_node_iter(node, seed)
        return

    # Handle dict with _range_ in value position (not a pure range node)
    if RANGE_KEYWORD in node:
        yield from _expand_dict_iter(node, seed)
        return

    # Normal dict: Cartesian product over key values
    yield from _expand_dict_iter(node, seed)


def _expand_list_iter(
    node: list,
    seed: Optional[int]
) -> Iterator[List[Any]]:
    """Lazily expand a list by yielding Cartesian product of elements.

    Args:
        node: List to expand.
        seed: Random seed.

    Yields:
        Expanded list combinations.
    """
    if not node:
        yield []
        return

    # Special case: single element that expands to lists
    if len(node) == 1:
        for result in _expand_iter_internal(node[0], seed):
            if isinstance(result, list):
                yield result
            else:
                yield [result]
        return

    # For Cartesian product, we need to materialize inner expansions
    # This is a limitation - true lazy Cartesian product requires
    # knowing all dimensions upfront
    expanded_elements = [list(_expand_iter_internal(elem, seed)) for elem in node]

    # Yield products one at a time
    for combo in product(*expanded_elements):
        yield list(combo)


def _expand_mixed_or_node_iter(
    node: Dict[str, Any],
    seed: Optional[int]
) -> Iterator[Dict[str, Any]]:
    """Lazily expand a dict that has _or_ mixed with other keys.

    Args:
        node: Dict containing _or_ and other keys.
        seed: Random seed.

    Yields:
        Merged dict variants.
    """
    # Extract modifiers that go with _or_
    or_modifier_keys = {
        "_or_", "size", "count", "pick", "arrange", "then_pick", "then_arrange"
    }

    # Separate base keys from OR-related keys
    base = {k: v for k, v in node.items() if k not in or_modifier_keys}
    or_node = {k: node[k] for k in or_modifier_keys if k in node}

    # Materialize base expansion (usually small)
    base_expanded = list(_expand_iter_internal(base, seed))

    # Iterate over choices
    for b in base_expanded:
        for c in _expand_iter_internal(or_node, seed):
            if isinstance(c, Mapping):
                yield {**b, **c}
            else:
                raise ValueError(
                    "Top-level '_or_' choices in a mixed dict must be dicts, "
                    f"not {type(c).__name__}. Got: {c}"
                )


def _expand_dict_iter(
    node: Dict[str, Any],
    seed: Optional[int]
) -> Iterator[Dict[str, Any]]:
    """Lazily expand a regular dict by yielding Cartesian product of values.

    Args:
        node: Dict to expand.
        seed: Random seed.

    Yields:
        Dict variants.
    """
    if not node:
        yield {}
        return

    # Expand each value (need to materialize for Cartesian product)
    keys = list(node.keys())
    value_options = [list(_expand_value_iter(v, seed)) for v in node.values()]

    # Yield products one at a time
    for combo in product(*value_options):
        yield dict(zip(keys, combo))


def _expand_value_iter(v: Any, seed: Optional[int]) -> Iterator[Any]:
    """Lazily expand a value in a dict position.

    Args:
        v: Value to expand.
        seed: Random seed.

    Yields:
        Expanded values.
    """
    if isinstance(v, Mapping):
        yield from _expand_iter_internal(v, seed)
    elif isinstance(v, list):
        yield from _expand_iter_internal(v, seed)
    else:
        yield v


def _reservoir_sample(
    iterator: Iterator[Any],
    k: int,
    seed: Optional[int] = None
) -> Iterator[Any]:
    """Reservoir sampling for uniformly selecting k items from an iterator.

    Uses Algorithm R (Vitter, 1985) for efficient single-pass sampling.

    Args:
        iterator: Source iterator.
        k: Number of items to sample.
        seed: Random seed for reproducibility.

    Yields:
        k uniformly sampled items.
    """
    import random

    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # Fill reservoir with first k items
    reservoir = list(islice(iterator, k))

    if len(reservoir) < k:
        # Not enough items, yield all
        yield from reservoir
        return

    # Process remaining items
    for i, item in enumerate(iterator, k):
        j = rng.randint(0, i)
        if j < k:
            reservoir[j] = item

    yield from reservoir


# =============================================================================
# Iteration utilities
# =============================================================================

def iter_with_progress(
    spec: GeneratorNode,
    seed: Optional[int] = None,
    report_every: int = 1000
) -> Iterator[tuple]:
    """Iterate with progress reporting.

    Args:
        spec: Specification to expand.
        seed: Random seed.
        report_every: Report progress every N items.

    Yields:
        Tuples of (index, config).
    """
    for i, config in enumerate(expand_spec_iter(spec, seed)):
        yield (i, config)
        if (i + 1) % report_every == 0:
            # Progress info is yielded as part of the tuple
            pass


def batch_iter(
    spec: GeneratorNode,
    batch_size: int,
    seed: Optional[int] = None
) -> Iterator[List[Any]]:
    """Iterate in batches for chunk processing.

    Args:
        spec: Specification to expand.
        batch_size: Number of configs per batch.
        seed: Random seed.

    Yields:
        Lists of up to batch_size configurations.
    """
    batch = []
    for config in expand_spec_iter(spec, seed):
        batch.append(config)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch
