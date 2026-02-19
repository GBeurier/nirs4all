"""Strategy registry for generator expansion.

This module provides registration and lookup functionality for expansion
strategies. It maintains an ordered list of strategies and provides
dispatch functionality to find the appropriate strategy for a given node.
"""

from typing import Optional

from .base import ExpansionStrategy, GeneratorNode

# Registry of strategy classes, ordered by priority
_strategy_classes: list[type[ExpansionStrategy]] = []

# Cache of strategy instances (singletons)
_strategy_instances: dict[type[ExpansionStrategy], ExpansionStrategy] = {}

def register_strategy(
    strategy_cls: type[ExpansionStrategy],
    priority: int | None = None
) -> type[ExpansionStrategy]:
    """Register a strategy class.

    Can be used as a decorator or called directly.

    Args:
        strategy_cls: The strategy class to register.
        priority: Optional priority override. If None, uses class priority.

    Returns:
        The strategy class (for decorator usage).

    Examples:
        >>> @register_strategy
        ... class MyStrategy(ExpansionStrategy):
        ...     priority = 10
        ...     ...

        >>> register_strategy(MyStrategy, priority=5)
    """
    if priority is not None:
        strategy_cls.priority = priority

    # Remove if already registered (for re-registration with new priority)
    _strategy_classes[:] = [c for c in _strategy_classes if c is not strategy_cls]

    # Insert at correct position based on priority (higher priority first)
    inserted = False
    for i, existing in enumerate(_strategy_classes):
        if strategy_cls.priority > existing.priority:
            _strategy_classes.insert(i, strategy_cls)
            inserted = True
            break
    if not inserted:
        _strategy_classes.append(strategy_cls)

    # Clear instance cache if exists
    _strategy_instances.pop(strategy_cls, None)

    return strategy_cls

def get_strategy(node: GeneratorNode) -> ExpansionStrategy | None:
    """Find the appropriate strategy for a node.

    Iterates through registered strategies (in priority order) and
    returns the first one that can handle the node.

    Args:
        node: A dictionary node from the configuration.

    Returns:
        An ExpansionStrategy instance if one handles the node, None otherwise.

    Examples:
        >>> strategy = get_strategy({"_or_": ["A", "B"]})
        >>> strategy
        OrStrategy(priority=10)
        >>> result = strategy.expand(node)
    """
    if not isinstance(node, dict):
        return None

    for strategy_cls in _strategy_classes:
        if strategy_cls.handles(node):
            # Return cached instance or create new one
            if strategy_cls not in _strategy_instances:
                _strategy_instances[strategy_cls] = strategy_cls()
            return _strategy_instances[strategy_cls]

    return None

def clear_registry() -> None:
    """Clear all registered strategies and cached instances.

    Primarily for testing purposes.
    """
    _strategy_classes.clear()
    _strategy_instances.clear()
