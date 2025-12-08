"""Cartesian strategy for staged pipeline expansion with selection.

This module handles _cartesian_ nodes that first compute the Cartesian product
of nested stages (each with _or_ choices), then apply pick/arrange selection
on the resulting complete pipelines.

This is the key pattern for preprocessing pipeline generation:
- Define stages (scatter, smooth, derivative, etc.)
- Each stage has multiple options via _or_
- Generate all combinations (Cartesian product)
- Then pick/arrange from the complete pipelines

Syntax:
    {"_cartesian_": [stage1, stage2, ...], "pick": N}
    {"_cartesian_": [stage1, stage2, ...], "arrange": N}

Examples:
    # Generate all pipeline combinations, then pick 2
    {"_cartesian_": [
        {"_or_": ["MSC", "SNV", "EMSC"]},
        {"_or_": ["SavGol", "Gaussian", None]},
        {"_or_": [None, "Deriv1", "Deriv2"]}
    ], "pick": 2}

    -> Generates all 27 pipelines (3×3×3), then picks 2-combinations
    -> Result: [["MSC", "SavGol", None], ["SNV", "Gaussian", "Deriv1"]], ...

    # Pick 1-3 complete pipelines
    {"_cartesian_": [...], "pick": (1, 3), "count": 20}
"""

from itertools import combinations, permutations
from math import comb, factorial
from typing import Any, FrozenSet, List, Optional, Tuple

from .base import ExpansionStrategy, GeneratorNode, ExpandedResult, SizeSpec
from .registry import register_strategy
from ..keywords import (
    COUNT_KEYWORD,
    SEED_KEYWORD,
    PICK_KEYWORD,
    ARRANGE_KEYWORD,
    MUTEX_KEYWORD,
    REQUIRES_KEYWORD,
    EXCLUDE_KEYWORD,
    TAGS_KEYWORD,
    METADATA_KEYWORD,
)
from ..utils.sampling import sample_with_seed

# Define the keyword
CARTESIAN_KEYWORD: str = "_cartesian_"

# Valid keys for a pure cartesian node
PURE_CARTESIAN_KEYS: FrozenSet[str] = frozenset({
    CARTESIAN_KEYWORD,
    PICK_KEYWORD,
    ARRANGE_KEYWORD,
    COUNT_KEYWORD,
    SEED_KEYWORD,
    MUTEX_KEYWORD,
    REQUIRES_KEYWORD,
    EXCLUDE_KEYWORD,
    TAGS_KEYWORD,
    METADATA_KEYWORD,
})


@register_strategy
class CartesianStrategy(ExpansionStrategy):
    """Strategy for handling _cartesian_ nodes.

    Generates the Cartesian product of all stages first (each stage being
    an _or_ node or list of options), then applies pick or arrange
    selection to the complete pipelines.

    This differs from _grid_ which produces dicts. _cartesian_ produces
    lists (ordered stages) which is ideal for preprocessing pipelines.

    Supported formats:
        - Array of stages: [stage1, stage2, ...]
        - With pick: Select N combinations of complete pipelines
        - With arrange: Select N permutations of complete pipelines
        - With count: Limit number of results
        - With constraints: Filter invalid combinations

    Attributes:
        keywords: {_cartesian_, pick, arrange, count, ...}
        priority: 35 (high priority, checked before grid)
    """

    keywords: FrozenSet[str] = PURE_CARTESIAN_KEYS
    priority: int = 35  # High priority

    @classmethod
    def handles(cls, node: GeneratorNode) -> bool:
        """Check if node is a pure cartesian node.

        Args:
            node: Dictionary node to check.

        Returns:
            True if node contains _cartesian_ and only cartesian-related keys.
        """
        if not isinstance(node, dict):
            return False
        return CARTESIAN_KEYWORD in node and set(node.keys()).issubset(PURE_CARTESIAN_KEYS)

    def expand(
        self,
        node: GeneratorNode,
        seed: Optional[int] = None,
        expand_nested: Optional[callable] = None
    ) -> ExpandedResult:
        """Expand a cartesian node to list of pipeline combinations.

        The process:
        1. Expand each stage to get its options
        2. Compute Cartesian product of all stages -> complete pipelines
        3. If pick/arrange specified, select from complete pipelines
        4. Apply constraints if specified
        5. Apply count limit if specified

        Args:
            node: Cartesian specification node.
            seed: Optional seed for random sampling when count is used.
            expand_nested: Callback to expand nested generator nodes.

        Returns:
            List of pipeline combinations.

        Examples:
            >>> strategy.expand({
            ...     "_cartesian_": [
            ...         {"_or_": ["A", "B"]},
            ...         {"_or_": ["X", "Y"]}
            ...     ],
            ...     "pick": 2
            ... })
            [[["A", "X"], ["A", "Y"]], [["A", "X"], ["B", "X"]], ...]
        """
        stages = node[CARTESIAN_KEYWORD]
        pick = node.get(PICK_KEYWORD)
        arrange = node.get(ARRANGE_KEYWORD)
        count = node.get(COUNT_KEYWORD)
        node_seed = node.get(SEED_KEYWORD, seed)

        # Extract constraints
        mutex_groups = node.get(MUTEX_KEYWORD, [])
        requires_groups = node.get(REQUIRES_KEYWORD, [])
        exclude_combos = node.get(EXCLUDE_KEYWORD, [])

        if not isinstance(stages, list):
            raise ValueError(
                f"_cartesian_ must be a list of stages, got {type(stages).__name__}"
            )

        # Handle empty stages
        if not stages:
            return [[]]

        # Step 1: Expand each stage to get its options
        expanded_stages = []
        for stage in stages:
            if expand_nested and isinstance(stage, (dict, list)):
                stage_options = expand_nested(stage)
            else:
                stage_options = [stage]
            expanded_stages.append(stage_options)

        # Step 2: Compute Cartesian product -> complete pipelines
        from itertools import product as cartesian_product
        all_pipelines = [list(combo) for combo in cartesian_product(*expanded_stages)]

        # Step 3: Apply pick or arrange selection
        if pick is not None:
            result = self._apply_pick(all_pipelines, pick)
        elif arrange is not None:
            result = self._apply_arrange(all_pipelines, arrange)
        else:
            # No selection - return all pipelines as-is
            result = all_pipelines

        # Step 4: Apply constraints if specified
        if mutex_groups or requires_groups or exclude_combos:
            result = self._apply_constraints(
                result, mutex_groups, requires_groups, exclude_combos
            )

        # Step 5: Apply count limit
        if count is not None and len(result) > count:
            result = sample_with_seed(result, count, seed=node_seed)

        return result

    def _apply_pick(
        self,
        pipelines: List[List[Any]],
        pick_spec: SizeSpec
    ) -> ExpandedResult:
        """Apply pick (combinations) to the list of pipelines.

        Args:
            pipelines: List of complete pipelines.
            pick_spec: Size specification (int or tuple).

        Returns:
            List of pipeline combinations.
        """
        from_size, to_size = self._normalize_spec(pick_spec)
        result = []

        for size in range(from_size, to_size + 1):
            if size > len(pipelines):
                continue
            if size == 0:
                result.append([])
                continue
            for combo in combinations(pipelines, size):
                result.append(list(combo))

        return result

    def _apply_arrange(
        self,
        pipelines: List[List[Any]],
        arrange_spec: SizeSpec
    ) -> ExpandedResult:
        """Apply arrange (permutations) to the list of pipelines.

        Args:
            pipelines: List of complete pipelines.
            arrange_spec: Size specification (int or tuple).

        Returns:
            List of pipeline permutations.
        """
        from_size, to_size = self._normalize_spec(arrange_spec)
        result = []

        for size in range(from_size, to_size + 1):
            if size > len(pipelines):
                continue
            if size == 0:
                result.append([])
                continue
            for perm in permutations(pipelines, size):
                result.append(list(perm))

        return result

    def _normalize_spec(self, spec: SizeSpec) -> Tuple[int, int]:
        """Normalize size specification to (from, to) tuple."""
        if isinstance(spec, int):
            return (spec, spec)
        elif isinstance(spec, (tuple, list)) and len(spec) == 2:
            return (spec[0], spec[1])
        else:
            raise ValueError(f"Invalid size spec: {spec}. Must be int or (from, to).")

    def _apply_constraints(
        self,
        results: ExpandedResult,
        mutex_groups: List[List[Any]],
        requires_groups: List[List[Any]],
        exclude_combos: List[List[Any]]
    ) -> ExpandedResult:
        """Apply constraint filters to expanded results.

        Note: For _cartesian_, constraints apply to the selected pipelines,
        not the items within pipelines.
        """
        from ..constraints import apply_all_constraints

        if not results:
            return results

        return apply_all_constraints(
            results,
            mutex_groups=mutex_groups,
            requires_groups=requires_groups,
            exclude_combos=exclude_combos
        )

    def count(self, node: GeneratorNode, count_nested: Optional[callable] = None) -> int:
        """Count cartesian combinations without generating them.

        Args:
            node: Cartesian specification node.
            count_nested: Callback to count nested nodes.

        Returns:
            Number of pipeline combinations.
        """
        stages = node[CARTESIAN_KEYWORD]
        pick = node.get(PICK_KEYWORD)
        arrange = node.get(ARRANGE_KEYWORD)
        count_limit = node.get(COUNT_KEYWORD)

        if not isinstance(stages, list):
            return 0

        if not stages:
            return 1

        # Count total pipelines (Cartesian product)
        total_pipelines = 1
        for stage in stages:
            if count_nested and isinstance(stage, (dict, list)):
                stage_count = count_nested(stage)
            else:
                stage_count = 1
            total_pipelines *= stage_count

        # Apply pick/arrange selection count
        if pick is not None:
            total = self._count_pick(total_pipelines, pick)
        elif arrange is not None:
            total = self._count_arrange(total_pipelines, arrange)
        else:
            total = total_pipelines

        # Apply count limit
        if count_limit is not None:
            return min(count_limit, total)
        return total

    def _count_pick(self, n: int, pick_spec: SizeSpec) -> int:
        """Count pick combinations."""
        from_size, to_size = self._normalize_spec(pick_spec)
        total = 0
        for size in range(from_size, to_size + 1):
            if size <= n:
                total += comb(n, size)
        return total

    def _count_arrange(self, n: int, arrange_spec: SizeSpec) -> int:
        """Count arrange permutations."""
        from_size, to_size = self._normalize_spec(arrange_spec)
        total = 0
        for size in range(from_size, to_size + 1):
            if size <= n:
                total += factorial(n) // factorial(n - size)
        return total

    def validate(self, node: GeneratorNode) -> List[str]:
        """Validate cartesian node specification.

        Args:
            node: Cartesian node to validate.

        Returns:
            List of error messages. Empty if valid.
        """
        errors = []
        stages = node.get(CARTESIAN_KEYWORD)

        if stages is None:
            errors.append("Missing _cartesian_ key")
            return errors

        if not isinstance(stages, list):
            errors.append(
                f"_cartesian_ must be a list, got {type(stages).__name__}"
            )
            return errors

        # Validate pick/arrange
        pick = node.get(PICK_KEYWORD)
        arrange = node.get(ARRANGE_KEYWORD)

        if pick is not None and arrange is not None:
            errors.append("Cannot specify both 'pick' and 'arrange'")

        for key, spec in [(PICK_KEYWORD, pick), (ARRANGE_KEYWORD, arrange)]:
            if spec is not None and not self._is_valid_size_spec(spec):
                errors.append(
                    f"{key} must be int or tuple (from, to), "
                    f"got {type(spec).__name__}"
                )

        # Validate count
        count = node.get(COUNT_KEYWORD)
        if count is not None and not isinstance(count, int):
            errors.append(f"count must be an integer, got {type(count).__name__}")

        return errors

    def _is_valid_size_spec(self, spec: Any) -> bool:
        """Check if a size specification is valid."""
        if isinstance(spec, int):
            return True
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            return all(isinstance(x, int) for x in spec)
        return False
