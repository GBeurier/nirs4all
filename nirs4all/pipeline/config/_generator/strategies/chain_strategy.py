"""Chain strategy for sequential ordered choices.

This module handles _chain_ nodes that produce configurations in a specific
order (unlike _or_ which is unordered).

Syntax:
    {"_chain_": [config1, config2, config3]}

Examples:
    {"_chain_": [{"model": "baseline"}, {"model": "improved"}, {"model": "best"}]}
    -> Generates configs in that exact order: baseline, improved, best

Unlike _or_ which might be randomized with count, _chain_ preserves order.
"""

from typing import Any, Dict, FrozenSet, List, Optional

from .base import ExpansionStrategy, GeneratorNode, ExpandedResult
from .registry import register_strategy
from ..keywords import CHAIN_KEYWORD, COUNT_KEYWORD, SEED_KEYWORD, PURE_CHAIN_KEYS
from ..utils.sampling import sample_with_seed


@register_strategy
class ChainStrategy(ExpansionStrategy):
    """Strategy for handling _chain_ nodes.

    Generates configurations in sequential order. Each item in the chain
    is expanded and added to the result list in order.

    Supported formats:
        - Array: [config1, config2, ...]
        - With count: Limits output to first n items (not random)

    Attributes:
        keywords: {_chain_, count}
        priority: 26 (between log_range and range)
    """

    keywords: FrozenSet[str] = PURE_CHAIN_KEYS
    priority: int = 26

    @classmethod
    def handles(cls, node: GeneratorNode) -> bool:
        """Check if node is a pure chain node.

        Args:
            node: Dictionary node to check.

        Returns:
            True if node contains _chain_ and only chain-related keys.
        """
        if not isinstance(node, dict):
            return False
        return CHAIN_KEYWORD in node and set(node.keys()).issubset(PURE_CHAIN_KEYS)

    def expand(
        self,
        node: GeneratorNode,
        seed: Optional[int] = None,
        expand_nested: Optional[callable] = None
    ) -> ExpandedResult:
        """Expand a chain node to list of sequential configurations.

        Args:
            node: Chain specification node.
            seed: Optional seed for random sampling when count is used.
            expand_nested: Callback to expand nested generator nodes.

        Returns:
            List of configurations in order.

        Examples:
            >>> strategy.expand({"_chain_": [{"x": 1}, {"x": 2}, {"x": 3}]})
            [{"x": 1}, {"x": 2}, {"x": 3}]
        """
        chain_spec = node[CHAIN_KEYWORD]
        count = node.get(COUNT_KEYWORD)
        node_seed = node.get(SEED_KEYWORD, seed)

        if not isinstance(chain_spec, list):
            raise ValueError(
                f"_chain_ must be a list, got {type(chain_spec).__name__}"
            )

        # Handle empty chain
        if not chain_spec:
            return []

        # Expand each item in order
        results = []
        for item in chain_spec:
            if expand_nested and isinstance(item, (dict, list)):
                expanded = expand_nested(item)
                results.extend(expanded)
            else:
                results.append(item)

        # Apply count limit (takes first n, not random)
        if count is not None and len(results) > count:
            # For chain, count takes first n items (ordered), not random
            # Unless seed is specified, then we sample randomly
            if node_seed is not None:
                results = sample_with_seed(results, count, seed=node_seed)
            else:
                results = results[:count]

        return results

    def count(self, node: GeneratorNode, count_nested: Optional[callable] = None) -> int:
        """Count chain items without generating them.

        Args:
            node: Chain specification node.
            count_nested: Callback to count nested nodes.

        Returns:
            Number of items in the chain.
        """
        chain_spec = node[CHAIN_KEYWORD]
        count_limit = node.get(COUNT_KEYWORD)

        if not isinstance(chain_spec, list):
            return 0

        # Count items, expanding nested generators
        total = 0
        for item in chain_spec:
            if count_nested and isinstance(item, (dict, list)):
                total += count_nested(item)
            else:
                total += 1

        # Apply count limit
        if count_limit is not None:
            return min(count_limit, total)
        return total

    def validate(self, node: GeneratorNode) -> List[str]:
        """Validate chain node specification.

        Args:
            node: Chain node to validate.

        Returns:
            List of error messages. Empty if valid.
        """
        errors = []
        chain_spec = node.get(CHAIN_KEYWORD)

        if chain_spec is None:
            errors.append("Missing _chain_ key")
            return errors

        if not isinstance(chain_spec, list):
            errors.append(f"_chain_ must be a list, got {type(chain_spec).__name__}")
            return errors

        # Validate count
        count = node.get(COUNT_KEYWORD)
        if count is not None and not isinstance(count, int):
            errors.append(f"count must be an integer, got {type(count).__name__}")

        return errors
