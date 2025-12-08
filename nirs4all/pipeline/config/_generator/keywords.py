"""Generator keyword definitions and utilities.

This module centralizes all generator-related keywords and provides
utility functions for detecting and extracting generator-specific
elements from configuration nodes.

Phase 3 additions:
- New generation keywords: _grid_, _zip_, _chain_, _sample_, _log_range_
- Metadata keywords: _tags_, _metadata_
- Constraint keywords: _mutex_, _requires_, _depends_on_
"""

from typing import Any, Dict, FrozenSet

# =============================================================================
# Core Generation Keywords
# =============================================================================

OR_KEYWORD: str = "_or_"
RANGE_KEYWORD: str = "_range_"
LOG_RANGE_KEYWORD: str = "_log_range_"  # Logarithmic range generation
GRID_KEYWORD: str = "_grid_"  # Grid search style parameter combinations
ZIP_KEYWORD: str = "_zip_"  # Parallel iteration (zip behavior)
CHAIN_KEYWORD: str = "_chain_"  # Sequential ordered choices
SAMPLE_KEYWORD: str = "_sample_"  # Statistical sampling (uniform, log-uniform)
CARTESIAN_KEYWORD: str = "_cartesian_"  # Cartesian product of stages with pick/arrange

# =============================================================================
# Modifier Keywords
# =============================================================================

SIZE_KEYWORD: str = "size"
COUNT_KEYWORD: str = "count"
SEED_KEYWORD: str = "_seed_"  # Deterministic generation with seed
WEIGHTS_KEYWORD: str = "_weights_"  # Weighted random selection

# =============================================================================
# Selection Semantics Keywords
# =============================================================================

PICK_KEYWORD: str = "pick"  # Unordered selection (combinations)
ARRANGE_KEYWORD: str = "arrange"  # Ordered arrangement (permutations)

# Second-order selection keywords (alternative to [outer, inner] array syntax)
# Semantics: pick/arrange happens first, then then_* is applied to the result
THEN_PICK_KEYWORD: str = "then_pick"  # Then select from result using combinations
THEN_ARRANGE_KEYWORD: str = "then_arrange"  # Then select from result using permutations

# =============================================================================
# Metadata Keywords (Phase 3)
# =============================================================================

TAGS_KEYWORD: str = "_tags_"  # Configuration tags for filtering/grouping
METADATA_KEYWORD: str = "_metadata_"  # Arbitrary metadata attached to configs

# =============================================================================
# Constraint Keywords (Phase 3)
# =============================================================================

MUTEX_KEYWORD: str = "_mutex_"  # Mutual exclusion constraints
REQUIRES_KEYWORD: str = "_requires_"  # Dependency requirements
DEPENDS_ON_KEYWORD: str = "_depends_on_"  # Conditional expansion
EXCLUDE_KEYWORD: str = "_exclude_"  # Exclusion rules

# =============================================================================
# Keyword Groups
# =============================================================================

GENERATION_KEYWORDS: FrozenSet[str] = frozenset({
    OR_KEYWORD,
    RANGE_KEYWORD,
    LOG_RANGE_KEYWORD,
    GRID_KEYWORD,
    ZIP_KEYWORD,
    CHAIN_KEYWORD,
    SAMPLE_KEYWORD,
    CARTESIAN_KEYWORD,
})

SELECTION_KEYWORDS: FrozenSet[str] = frozenset({
    PICK_KEYWORD,
    ARRANGE_KEYWORD,
    THEN_PICK_KEYWORD,
    THEN_ARRANGE_KEYWORD,
})

MODIFIER_KEYWORDS: FrozenSet[str] = frozenset({
    SIZE_KEYWORD,
    COUNT_KEYWORD,
    SEED_KEYWORD,
    WEIGHTS_KEYWORD,
    PICK_KEYWORD,
    ARRANGE_KEYWORD,
    THEN_PICK_KEYWORD,
    THEN_ARRANGE_KEYWORD,
})

METADATA_KEYWORDS: FrozenSet[str] = frozenset({
    TAGS_KEYWORD,
    METADATA_KEYWORD,
})

CONSTRAINT_KEYWORDS: FrozenSet[str] = frozenset({
    MUTEX_KEYWORD,
    REQUIRES_KEYWORD,
    DEPENDS_ON_KEYWORD,
    EXCLUDE_KEYWORD,
})

ALL_KEYWORDS: FrozenSet[str] = (
    GENERATION_KEYWORDS | MODIFIER_KEYWORDS | METADATA_KEYWORDS | CONSTRAINT_KEYWORDS
)

# Subsets for specific node type detection
PURE_OR_KEYS: FrozenSet[str] = frozenset({
    OR_KEYWORD, SIZE_KEYWORD, COUNT_KEYWORD, PICK_KEYWORD, ARRANGE_KEYWORD,
    THEN_PICK_KEYWORD, THEN_ARRANGE_KEYWORD, SEED_KEYWORD, WEIGHTS_KEYWORD,
    MUTEX_KEYWORD, REQUIRES_KEYWORD, EXCLUDE_KEYWORD,
    TAGS_KEYWORD, METADATA_KEYWORD,
})

PURE_RANGE_KEYS: FrozenSet[str] = frozenset({
    RANGE_KEYWORD, COUNT_KEYWORD, SEED_KEYWORD,
    TAGS_KEYWORD, METADATA_KEYWORD,
})

PURE_LOG_RANGE_KEYS: FrozenSet[str] = frozenset({
    LOG_RANGE_KEYWORD, COUNT_KEYWORD, SEED_KEYWORD,
    TAGS_KEYWORD, METADATA_KEYWORD,
})

PURE_GRID_KEYS: FrozenSet[str] = frozenset({
    GRID_KEYWORD, COUNT_KEYWORD, SEED_KEYWORD,
    TAGS_KEYWORD, METADATA_KEYWORD,
})

PURE_ZIP_KEYS: FrozenSet[str] = frozenset({
    ZIP_KEYWORD, COUNT_KEYWORD, SEED_KEYWORD,
    TAGS_KEYWORD, METADATA_KEYWORD,
})

PURE_CHAIN_KEYS: FrozenSet[str] = frozenset({
    CHAIN_KEYWORD, COUNT_KEYWORD, SEED_KEYWORD,
    TAGS_KEYWORD, METADATA_KEYWORD,
})

PURE_SAMPLE_KEYS: FrozenSet[str] = frozenset({
    SAMPLE_KEYWORD, COUNT_KEYWORD, SEED_KEYWORD,
    TAGS_KEYWORD, METADATA_KEYWORD,
})

PURE_CARTESIAN_KEYS: FrozenSet[str] = frozenset({
    CARTESIAN_KEYWORD, PICK_KEYWORD, ARRANGE_KEYWORD,
    COUNT_KEYWORD, SEED_KEYWORD,
    MUTEX_KEYWORD, REQUIRES_KEYWORD, EXCLUDE_KEYWORD,
    TAGS_KEYWORD, METADATA_KEYWORD,
})


def is_generator_node(node: Dict[str, Any]) -> bool:
    """Check if a dict node contains any generator keywords.

    Args:
        node: A dictionary node from the configuration.

    Returns:
        True if the node contains any generation keywords (_or_, _range_, etc.),
        False otherwise.

    Examples:
        >>> is_generator_node({"_or_": ["A", "B"]})
        True
        >>> is_generator_node({"class": "MyClass"})
        False
        >>> is_generator_node({"_range_": [1, 10]})
        True
    """
    if not isinstance(node, dict):
        return False
    return bool(GENERATION_KEYWORDS & set(node.keys()))


def is_pure_or_node(node: Dict[str, Any]) -> bool:
    """Check if a node is a pure OR node (only _or_, size, count keys).

    Args:
        node: A dictionary node from the configuration.

    Returns:
        True if the node contains only OR-related keys, False otherwise.

    Examples:
        >>> is_pure_or_node({"_or_": ["A", "B"], "size": 2})
        True
        >>> is_pure_or_node({"_or_": ["A", "B"], "class": "X"})
        False
    """
    if not isinstance(node, dict):
        return False
    return set(node.keys()).issubset(PURE_OR_KEYS)


def is_pure_range_node(node: Dict[str, Any]) -> bool:
    """Check if a node is a pure range node (only _range_, count keys).

    Args:
        node: A dictionary node from the configuration.

    Returns:
        True if the node contains only range-related keys, False otherwise.

    Examples:
        >>> is_pure_range_node({"_range_": [1, 10]})
        True
        >>> is_pure_range_node({"_range_": [1, 10], "count": 5})
        True
        >>> is_pure_range_node({"_range_": [1, 10], "size": 2})
        False
    """
    if not isinstance(node, dict):
        return False
    return set(node.keys()).issubset(PURE_RANGE_KEYS)


def has_or_keyword(node: Dict[str, Any]) -> bool:
    """Check if a node contains the _or_ keyword.

    Args:
        node: A dictionary node from the configuration.

    Returns:
        True if the node contains _or_, False otherwise.
    """
    if not isinstance(node, dict):
        return False
    return OR_KEYWORD in node


def has_range_keyword(node: Dict[str, Any]) -> bool:
    """Check if a node contains the _range_ keyword.

    Args:
        node: A dictionary node from the configuration.

    Returns:
        True if the node contains _range_, False otherwise.
    """
    if not isinstance(node, dict):
        return False
    return RANGE_KEYWORD in node


def extract_modifiers(node: Dict[str, Any]) -> Dict[str, Any]:
    """Extract modifier values from a node.

    Extracts all modifier keywords (size, count, _seed_, _weights_, _exclude_)
    from a node and returns them as a dictionary.

    Args:
        node: A dictionary node from the configuration.

    Returns:
        A dictionary containing only the modifier key-value pairs found in the node.

    Examples:
        >>> extract_modifiers({"_or_": ["A", "B"], "size": 2, "count": 1})
        {"size": 2, "count": 1}
        >>> extract_modifiers({"_or_": ["A", "B"]})
        {}
    """
    if not isinstance(node, dict):
        return {}
    return {k: node[k] for k in MODIFIER_KEYWORDS if k in node}


def extract_base_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """Extract non-keyword keys from a node.

    Returns a copy of the node with all generator and modifier keywords removed.

    Args:
        node: A dictionary node from the configuration.

    Returns:
        A dictionary containing only the non-keyword key-value pairs.

    Examples:
        >>> extract_base_node({"_or_": ["A", "B"], "class": "MyClass", "size": 2})
        {"class": "MyClass"}
        >>> extract_base_node({"class": "MyClass", "params": {"n": 5}})
        {"class": "MyClass", "params": {"n": 5}}
    """
    if not isinstance(node, dict):
        return {}
    return {k: v for k, v in node.items() if k not in ALL_KEYWORDS}


def extract_or_choices(node: Dict[str, Any]) -> list:
    """Extract the choices list from an OR node.

    Args:
        node: A dictionary node containing the _or_ keyword.

    Returns:
        The list of choices, or an empty list if _or_ is not present.

    Examples:
        >>> extract_or_choices({"_or_": ["A", "B", "C"]})
        ["A", "B", "C"]
        >>> extract_or_choices({"class": "MyClass"})
        []
    """
    if not isinstance(node, dict):
        return []
    return node.get(OR_KEYWORD, [])


def extract_range_spec(node: Dict[str, Any]) -> Any:
    """Extract the range specification from a range node.

    Args:
        node: A dictionary node containing the _range_ keyword.

    Returns:
        The range specification (list or dict), or None if not present.

    Examples:
        >>> extract_range_spec({"_range_": [1, 10, 2]})
        [1, 10, 2]
        >>> extract_range_spec({"_range_": {"from": 1, "to": 10}})
        {"from": 1, "to": 10}
    """
    if not isinstance(node, dict):
        return None
    return node.get(RANGE_KEYWORD)


# =============================================================================
# Phase 3: New Detection Functions
# =============================================================================

def has_log_range_keyword(node: Dict[str, Any]) -> bool:
    """Check if a node contains the _log_range_ keyword."""
    if not isinstance(node, dict):
        return False
    return LOG_RANGE_KEYWORD in node


def has_grid_keyword(node: Dict[str, Any]) -> bool:
    """Check if a node contains the _grid_ keyword."""
    if not isinstance(node, dict):
        return False
    return GRID_KEYWORD in node


def has_zip_keyword(node: Dict[str, Any]) -> bool:
    """Check if a node contains the _zip_ keyword."""
    if not isinstance(node, dict):
        return False
    return ZIP_KEYWORD in node


def has_chain_keyword(node: Dict[str, Any]) -> bool:
    """Check if a node contains the _chain_ keyword."""
    if not isinstance(node, dict):
        return False
    return CHAIN_KEYWORD in node


def has_sample_keyword(node: Dict[str, Any]) -> bool:
    """Check if a node contains the _sample_ keyword."""
    if not isinstance(node, dict):
        return False
    return SAMPLE_KEYWORD in node


def is_pure_log_range_node(node: Dict[str, Any]) -> bool:
    """Check if a node is a pure log range node."""
    if not isinstance(node, dict):
        return False
    return LOG_RANGE_KEYWORD in node and set(node.keys()).issubset(PURE_LOG_RANGE_KEYS)


def is_pure_grid_node(node: Dict[str, Any]) -> bool:
    """Check if a node is a pure grid node."""
    if not isinstance(node, dict):
        return False
    return GRID_KEYWORD in node and set(node.keys()).issubset(PURE_GRID_KEYS)


def is_pure_zip_node(node: Dict[str, Any]) -> bool:
    """Check if a node is a pure zip node."""
    if not isinstance(node, dict):
        return False
    return ZIP_KEYWORD in node and set(node.keys()).issubset(PURE_ZIP_KEYS)


def is_pure_chain_node(node: Dict[str, Any]) -> bool:
    """Check if a node is a pure chain node."""
    if not isinstance(node, dict):
        return False
    return CHAIN_KEYWORD in node and set(node.keys()).issubset(PURE_CHAIN_KEYS)


def is_pure_sample_node(node: Dict[str, Any]) -> bool:
    """Check if a node is a pure sample node."""
    if not isinstance(node, dict):
        return False
    return SAMPLE_KEYWORD in node and set(node.keys()).issubset(PURE_SAMPLE_KEYS)


def has_cartesian_keyword(node: Dict[str, Any]) -> bool:
    """Check if a node contains the _cartesian_ keyword."""
    if not isinstance(node, dict):
        return False
    return CARTESIAN_KEYWORD in node


def is_pure_cartesian_node(node: Dict[str, Any]) -> bool:
    """Check if a node is a pure cartesian node."""
    if not isinstance(node, dict):
        return False
    return CARTESIAN_KEYWORD in node and set(node.keys()).issubset(PURE_CARTESIAN_KEYS)


def extract_tags(node: Dict[str, Any]) -> list:
    """Extract tags from a node.

    Args:
        node: A dictionary node.

    Returns:
        List of tags, or empty list if no tags.

    Examples:
        >>> extract_tags({"_tags_": ["baseline", "v2"]})
        ["baseline", "v2"]
    """
    if not isinstance(node, dict):
        return []
    tags = node.get(TAGS_KEYWORD, [])
    if isinstance(tags, str):
        return [tags]
    return list(tags) if tags else []


def extract_metadata(node: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from a node.

    Args:
        node: A dictionary node.

    Returns:
        Metadata dict, or empty dict if no metadata.

    Examples:
        >>> extract_metadata({"_metadata_": {"author": "user1"}})
        {"author": "user1"}
    """
    if not isinstance(node, dict):
        return {}
    return node.get(METADATA_KEYWORD, {})


def extract_constraints(node: Dict[str, Any]) -> Dict[str, Any]:
    """Extract constraint specifications from a node.

    Args:
        node: A dictionary node.

    Returns:
        Dict containing constraint specifications (_mutex_, _requires_, etc.)

    Examples:
        >>> extract_constraints({"_mutex_": [["A", "B"]], "_requires_": [["C", "D"]]})
        {"_mutex_": [["A", "B"]], "_requires_": [["C", "D"]]}
    """
    if not isinstance(node, dict):
        return {}
    return {k: node[k] for k in CONSTRAINT_KEYWORDS if k in node}

