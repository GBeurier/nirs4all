"""Core expansion logic for the generator module.

This module provides the main expansion and counting functions using the
Strategy pattern for node-specific handling. It serves as the orchestration
layer that dispatches to appropriate strategies.

Main Functions:
    expand_spec(node, seed): Expand a configuration node into all variants
    count_combinations(node): Count variants without generating them

The core module handles:
    - Recursive expansion of nested structures
    - Delegation to strategies for _or_ and _range_ nodes
    - Cartesian product expansion for dicts and lists
    - Mixed nodes (dict with both _or_ and other keys)
"""

from collections.abc import Mapping
from itertools import product
from typing import Any, Dict, List, Optional, Union

from .strategies import get_strategy
from .strategies.base import ExpandedResult
from .keywords import (
    OR_KEYWORD,
    SIZE_KEYWORD,
    COUNT_KEYWORD,
    PICK_KEYWORD,
    ARRANGE_KEYWORD,
    THEN_PICK_KEYWORD,
    THEN_ARRANGE_KEYWORD,
    RANGE_KEYWORD,
    has_or_keyword,
)
from .utils.sampling import sample_with_seed

# Type alias
GeneratorNode = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

# Type for expansion with choices: list of (config, choices) tuples
ExpandedWithChoices = List[tuple]  # List[Tuple[Any, List[Dict[str, Any]]]]


def expand_spec(node: GeneratorNode, seed: Optional[int] = None) -> ExpandedResult:
    """Expand a specification node to all possible combinations.

    This is the main entry point for configuration expansion. It handles
    all node types and delegates to appropriate strategies for special
    generator nodes.

    Args:
        node: Configuration node to expand. Can be:
            - dict: Expanded based on keys (strategies or Cartesian product)
            - list: Cartesian product of expanded elements
            - scalar: Wrapped in a list

        seed: Optional random seed for reproducible generation when
            using 'count' to limit results.

    Returns:
        List of expanded variants.

    Examples:
        >>> expand_spec({"_or_": ["A", "B"]})
        ['A', 'B']
        >>> expand_spec({"_range_": [1, 3]})
        [1, 2, 3]
        >>> expand_spec({"x": {"_or_": [1, 2]}, "y": "fixed"})
        [{'x': 1, 'y': 'fixed'}, {'x': 2, 'y': 'fixed'}]
    """
    return _expand_internal(node, seed)


def expand_spec_with_choices(
    node: GeneratorNode,
    seed: Optional[int] = None
) -> List[tuple]:
    """Expand a specification node and track generator choices.

    Like expand_spec, but also returns the choices made at each generator
    node (_or_, _range_, etc.) for each expanded variant. This is useful
    for tracking which specific values were selected to produce each
    pipeline configuration.

    Args:
        node: Configuration node to expand.
        seed: Optional random seed for reproducible generation.

    Returns:
        List of (expanded_config, generator_choices) tuples.
        Each generator_choices is a list of dicts like:
        [{"_or_": selected_value}, {"_range_": 18}, ...]
        in the order they were encountered during expansion.

    Examples:
        >>> results = expand_spec_with_choices({"_or_": ["A", "B"]})
        >>> results
        [('A', [{'_or_': 'A'}]), ('B', [{'_or_': 'B'}])]

        >>> results = expand_spec_with_choices({"x": {"_or_": [1, 2]}, "y": 3})
        >>> results
        [({'x': 1, 'y': 3}, [{'_or_': 1}]), ({'x': 2, 'y': 3}, [{'_or_': 2}])]
    """
    return _expand_with_choices_internal(node, seed)


def _expand_internal(node: GeneratorNode, seed: Optional[int] = None) -> ExpandedResult:
    """Internal recursive expansion with seed propagation.

    Args:
        node: Node to expand.
        seed: Random seed.

    Returns:
        List of expanded variants.
    """
    # Handle lists: Cartesian product of expanded elements
    if isinstance(node, list):
        return _expand_list(node, seed)

    # Handle non-dict types: wrap as single-element list
    if not isinstance(node, Mapping):
        return [node]

    # Try strategy dispatch for pure generator nodes
    strategy = get_strategy(node)
    if strategy:
        return strategy.expand(
            node,
            seed=seed,
            expand_nested=lambda n: _expand_internal(n, seed)
        )

    # Handle mixed dict with _or_ key (has other keys too)
    if has_or_keyword(node):
        return _expand_mixed_or_node(node, seed)

    # Handle dict with _range_ in value position (not a pure range node)
    if RANGE_KEYWORD in node:
        # This shouldn't happen for pure range nodes (handled by strategy)
        # But handle it defensively
        return _expand_dict(node, seed)

    # Normal dict: Cartesian product over key values
    return _expand_dict(node, seed)


def _expand_list(node: list, seed: Optional[int]) -> ExpandedResult:
    """Expand a list by taking Cartesian product of elements.

    Args:
        node: List to expand.
        seed: Random seed.

    Returns:
        List of expanded list combinations.

    Examples:
        >>> _expand_list([{"_or_": ["A", "B"]}, "C"], None)
        [['A', 'C'], ['B', 'C']]
    """
    if not node:
        return [[]]  # Empty list -> single empty result

    # Special case: single element that expands to lists
    if len(node) == 1:
        element_result = _expand_internal(node[0], seed)
        # If result contains lists (combinations), return directly
        if element_result and isinstance(element_result[0], list):
            return element_result
        # Otherwise, fall through to normal processing

    # Expand each element
    expanded_elements = [_expand_internal(element, seed) for element in node]

    # Take Cartesian product
    results = []
    for combo in product(*expanded_elements):
        results.append(list(combo))
    return results


def _expand_mixed_or_node(node: Dict[str, Any], seed: Optional[int]) -> ExpandedResult:
    """Expand a dict that has _or_ mixed with other keys.

    The strategy is:
    1. Separate the base dict (non-generator keys) from the OR node
    2. Expand both independently
    3. Merge each base variant with each OR variant

    Args:
        node: Dict containing _or_ and other keys.
        seed: Random seed.

    Returns:
        List of merged dict variants.
    """
    # Extract modifiers that go with _or_
    or_modifier_keys = {
        "_or_", "size", "count", "pick", "arrange", "then_pick", "then_arrange"
    }

    # Separate base keys from OR-related keys
    base = {k: v for k, v in node.items() if k not in or_modifier_keys}
    or_node = {k: node[k] for k in or_modifier_keys if k in node}

    # Expand both parts
    base_expanded = _expand_internal(base, seed)  # list[dict]
    choice_expanded = _expand_internal(or_node, seed)  # list[dict or scalar]

    # Merge results
    results = []
    for b in base_expanded:
        for c in choice_expanded:
            if isinstance(c, Mapping):
                merged = {**b, **c}
                results.append(merged)
            else:
                # Scalar choices require top-level merge with a key
                raise ValueError(
                    "Top-level '_or_' choices in a mixed dict must be dicts, "
                    f"not {type(c).__name__}. Got: {c}"
                )

    return results


def _expand_dict(node: Dict[str, Any], seed: Optional[int]) -> ExpandedResult:
    """Expand a regular dict by taking Cartesian product of values.

    Args:
        node: Dict to expand.
        seed: Random seed.

    Returns:
        List of dict variants.

    Examples:
        >>> _expand_dict({"a": {"_or_": [1, 2]}, "b": 3}, None)
        [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]
    """
    if not node:
        return [{}]

    # Expand each value
    keys = []
    value_options = []
    for k, v in node.items():
        keys.append(k)
        value_options.append(_expand_value(v, seed))

    # Take Cartesian product over values
    results = []
    for combo in product(*value_options):
        result_dict = dict(zip(keys, combo))
        results.append(result_dict)

    return results


def _expand_value(v: Any, seed: Optional[int]) -> ExpandedResult:
    """Expand a value in a dict position.

    Handles nested generator nodes in value positions.

    Args:
        v: Value to expand.
        seed: Random seed.

    Returns:
        List of expanded values.
    """
    if isinstance(v, Mapping):
        # Check for value-level _or_ or _range_
        return _expand_internal(v, seed)
    elif isinstance(v, list):
        # Handle lists in value positions
        return _expand_internal(v, seed)
    else:
        # Scalar value
        return [v]


# =============================================================================
# Expansion with Choice Tracking
# =============================================================================

# Type for a single result with its choices
ResultWithChoices = tuple  # Tuple[Any, List[Dict[str, Any]]]


def _expand_with_choices_internal(
    node: GeneratorNode,
    seed: Optional[int] = None
) -> List[ResultWithChoices]:
    """Internal recursive expansion that tracks generator choices.

    Args:
        node: Node to expand.
        seed: Random seed.

    Returns:
        List of (expanded_value, choices_list) tuples.
    """
    # Handle lists: Cartesian product of expanded elements with merged choices
    if isinstance(node, list):
        return _expand_list_with_choices(node, seed)

    # Handle non-dict types: wrap as single-element list with no choices
    if not isinstance(node, Mapping):
        return [(node, [])]

    # Try strategy dispatch for pure generator nodes
    strategy = get_strategy(node)
    if strategy:
        return _expand_strategy_with_choices(node, strategy, seed)

    # Handle mixed dict with _or_ key (has other keys too)
    if has_or_keyword(node):
        return _expand_mixed_or_with_choices(node, seed)

    # Normal dict: Cartesian product over key values with merged choices
    return _expand_dict_with_choices(node, seed)


def _expand_strategy_with_choices(
    node: Dict[str, Any],
    strategy: Any,
    seed: Optional[int]
) -> List[ResultWithChoices]:
    """Expand a generator node using its strategy and track the choice.

    Args:
        node: Generator node (_or_, _range_, etc.).
        strategy: The strategy to use for expansion.
        seed: Random seed.

    Returns:
        List of (value, choices) tuples.
    """
    # Determine the keyword for this generator
    keyword = _get_generator_keyword(node)

    # Expand the node
    expanded = strategy.expand(
        node,
        seed=seed,
        expand_nested=lambda n: _expand_internal(n, seed)
    )

    # Each expanded value gets recorded as a choice
    results = []
    for value in expanded:
        # The choice records the keyword and the selected value
        choice = {keyword: value}
        results.append((value, [choice]))

    return results


def _get_generator_keyword(node: Dict[str, Any]) -> str:
    """Get the primary generator keyword from a node.

    Args:
        node: Generator node.

    Returns:
        The keyword string (e.g., "_or_", "_range_").
    """
    from .keywords import (
        OR_KEYWORD, RANGE_KEYWORD, LOG_RANGE_KEYWORD,
        GRID_KEYWORD, ZIP_KEYWORD, CHAIN_KEYWORD,
        SAMPLE_KEYWORD, CARTESIAN_KEYWORD
    )

    keyword_priority = [
        OR_KEYWORD, RANGE_KEYWORD, LOG_RANGE_KEYWORD,
        GRID_KEYWORD, ZIP_KEYWORD, CHAIN_KEYWORD,
        SAMPLE_KEYWORD, CARTESIAN_KEYWORD
    ]

    for kw in keyword_priority:
        if kw in node:
            return kw

    return "_unknown_"


def _expand_list_with_choices(
    node: list,
    seed: Optional[int]
) -> List[ResultWithChoices]:
    """Expand a list with choice tracking.

    Args:
        node: List to expand.
        seed: Random seed.

    Returns:
        List of (list_value, merged_choices) tuples.
    """
    if not node:
        return [([], [])]

    # Special case: single element
    if len(node) == 1:
        element_results = _expand_with_choices_internal(node[0], seed)
        # If results contain lists, return directly
        if element_results and isinstance(element_results[0][0], list):
            return element_results

    # Expand each element with choices
    expanded_elements = [_expand_with_choices_internal(element, seed) for element in node]

    # Cartesian product with merged choices
    results = []
    for combo in product(*expanded_elements):
        # combo is tuple of (value, choices) pairs
        values = [item[0] for item in combo]
        # Merge all choices in order
        merged_choices = []
        for item in combo:
            merged_choices.extend(item[1])
        results.append((values, merged_choices))

    return results


def _expand_mixed_or_with_choices(
    node: Dict[str, Any],
    seed: Optional[int]
) -> List[ResultWithChoices]:
    """Expand a mixed OR node with choice tracking.

    Args:
        node: Dict containing _or_ and other keys.
        seed: Random seed.

    Returns:
        List of (dict_value, merged_choices) tuples.
    """
    or_modifier_keys = {
        "_or_", "size", "count", "pick", "arrange", "then_pick", "then_arrange"
    }

    base = {k: v for k, v in node.items() if k not in or_modifier_keys}
    or_node = {k: node[k] for k in or_modifier_keys if k in node}

    # Expand both parts with choices
    base_expanded = _expand_dict_with_choices(base, seed)
    choice_expanded = _expand_with_choices_internal(or_node, seed)

    # Merge results
    results = []
    for b_val, b_choices in base_expanded:
        for c_val, c_choices in choice_expanded:
            if isinstance(c_val, Mapping):
                merged_val = {**b_val, **c_val}
                merged_choices = b_choices + c_choices
                results.append((merged_val, merged_choices))
            else:
                raise ValueError(
                    "Top-level '_or_' choices in a mixed dict must be dicts, "
                    f"not {type(c_val).__name__}. Got: {c_val}"
                )

    return results


def _expand_dict_with_choices(
    node: Dict[str, Any],
    seed: Optional[int]
) -> List[ResultWithChoices]:
    """Expand a dict with choice tracking.

    Args:
        node: Dict to expand.
        seed: Random seed.

    Returns:
        List of (dict_value, merged_choices) tuples.
    """
    if not node:
        return [({}, [])]

    # Expand each value with choices
    keys = []
    value_options = []
    for k, v in node.items():
        keys.append(k)
        value_options.append(_expand_value_with_choices(v, seed))

    # Cartesian product with merged choices
    results = []
    for combo in product(*value_options):
        # combo is tuple of (value, choices) pairs
        values = [item[0] for item in combo]
        result_dict = dict(zip(keys, values))
        # Merge all choices in order
        merged_choices = []
        for item in combo:
            merged_choices.extend(item[1])
        results.append((result_dict, merged_choices))

    return results


def _expand_value_with_choices(
    v: Any,
    seed: Optional[int]
) -> List[ResultWithChoices]:
    """Expand a value in a dict position with choice tracking.

    Args:
        v: Value to expand.
        seed: Random seed.

    Returns:
        List of (value, choices) tuples.
    """
    if isinstance(v, Mapping):
        return _expand_with_choices_internal(v, seed)
    elif isinstance(v, list):
        return _expand_with_choices_internal(v, seed)
    else:
        # Scalar value - no choices
        return [(v, [])]


# =============================================================================
# Counting Functions
# =============================================================================


def count_combinations(node: GeneratorNode) -> int:
    """Calculate total number of combinations without generating them.

    This is more efficient than generating all combinations when you only
    need to know the count.

    Args:
        node: Configuration node to count.

    Returns:
        Number of variants that expand_spec would produce.

    Examples:
        >>> count_combinations({"_or_": ["A", "B", "C"]})
        3
        >>> count_combinations({"_or_": ["A", "B", "C"], "pick": 2})
        3  # C(3,2)
        >>> count_combinations({"_range_": [1, 10]})
        10
    """
    return _count_internal(node)


def _count_internal(node: GeneratorNode) -> int:
    """Internal recursive counting.

    Args:
        node: Node to count.

    Returns:
        Number of variants.
    """
    # Handle lists: product of counts
    if isinstance(node, list):
        if not node:
            return 1  # Empty list -> single empty result
        total = 1
        for element in node:
            total *= _count_internal(element)
        return total

    # Scalars return 1
    if not isinstance(node, Mapping):
        return 1

    # Try strategy dispatch for pure generator nodes
    strategy = get_strategy(node)
    if strategy:
        return strategy.count(node, count_nested=_count_internal)

    # Handle mixed dict with _or_ key
    if has_or_keyword(node):
        return _count_mixed_or_node(node)

    # Normal dict: product over key values
    return _count_dict(node)


def _count_mixed_or_node(node: Dict[str, Any]) -> int:
    """Count mixed OR node.

    Args:
        node: Dict containing _or_ and other keys.

    Returns:
        Number of variants.
    """
    or_modifier_keys = {
        "_or_", "size", "count", "pick", "arrange", "then_pick", "then_arrange"
    }

    base = {k: v for k, v in node.items() if k not in or_modifier_keys}
    or_node = {k: node[k] for k in or_modifier_keys if k in node}

    base_count = _count_internal(base)
    choice_count = _count_internal(or_node)

    return base_count * choice_count


def _count_dict(node: Dict[str, Any]) -> int:
    """Count regular dict.

    Args:
        node: Dict to count.

    Returns:
        Number of variants.
    """
    if not node:
        return 1

    total = 1
    for v in node.values():
        total *= _count_value(v)
    return total


def _count_value(v: Any) -> int:
    """Count value-position combinations.

    Args:
        v: Value to count.

    Returns:
        Number of variants.
    """
    if isinstance(v, Mapping):
        return _count_internal(v)
    elif isinstance(v, list):
        return _count_internal(v)
    else:
        return 1
