"""Base class for generator expansion strategies.

This module defines the abstract base class for all expansion strategies.
Each strategy handles a specific type of generator node (e.g., _or_, _range_).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, FrozenSet, List, Optional, Union

# Type aliases for clarity
GeneratorNode = Dict[str, Any]
ExpandedResult = List[Any]
SizeSpec = Union[int, tuple, list]


class ExpansionStrategy(ABC):
    """Abstract base class for generator expansion strategies.

    Each strategy is responsible for:
    1. Detecting if it can handle a specific node type
    2. Expanding the node into all possible variants
    3. Counting the variants without generating them

    Subclasses must implement:
        - handles(node): Check if strategy can handle this node
        - expand(node, seed): Expand node to list of variants
        - count(node): Count variants without generating

    Attributes:
        keywords: Set of keywords this strategy recognizes.
        priority: Higher priority strategies are checked first.
    """

    # Keywords this strategy recognizes (to be overridden by subclasses)
    keywords: FrozenSet[str] = frozenset()

    # Priority for strategy matching (higher = checked first)
    priority: int = 0

    @classmethod
    @abstractmethod
    def handles(cls, node: GeneratorNode) -> bool:
        """Check if this strategy can handle the given node.

        Args:
            node: A dictionary node from the configuration.

        Returns:
            True if this strategy can expand the node, False otherwise.
        """

    @abstractmethod
    def expand(
        self,
        node: GeneratorNode,
        seed: Optional[int] = None,
        expand_nested: Optional[callable] = None
    ) -> ExpandedResult:
        """Expand a node into all possible variants.

        Args:
            node: A dictionary node to expand.
            seed: Optional random seed for reproducible generation.
            expand_nested: Callback to expand nested nodes recursively.
                This allows strategies to delegate back to the main
                expansion logic for nested structures.

        Returns:
            List of expanded variants.
        """

    @abstractmethod
    def count(self, node: GeneratorNode, count_nested: Optional[callable] = None) -> int:
        """Count the number of variants without generating them.

        Args:
            node: A dictionary node to count.
            count_nested: Callback to count nested nodes recursively.

        Returns:
            Number of variants that would be generated.
        """

    def validate(self, node: GeneratorNode) -> List[str]:
        """Validate a node and return any errors.

        Args:
            node: A dictionary node to validate.

        Returns:
            List of error messages. Empty list if valid.
        """
        return []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(priority={self.priority})"
