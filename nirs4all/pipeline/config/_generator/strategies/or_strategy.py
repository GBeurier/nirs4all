"""OR strategy for choice-based expansion.

This module handles _or_ nodes that define choices with various selection modes:
- Basic choice: Pick one from alternatives
- pick: Unordered selection (combinations)
- arrange: Ordered arrangement (permutations)
- size: Legacy alias for pick
- Second-order: then_pick, then_arrange, or [outer, inner] syntax
- Constraints: _mutex_, _requires_, _exclude_ for filtering combinations

Syntax examples:
    {"_or_": ["A", "B", "C"]}                    -> "A", "B", "C"
    {"_or_": ["A", "B", "C"], "pick": 2}         -> ["A", "B"], ["A", "C"], ["B", "C"]
    {"_or_": ["A", "B", "C"], "arrange": 2}      -> ["A", "B"], ["B", "A"], ...
    {"_or_": ["A", "B", "C"], "pick": (1, 2)}    -> Pick 1 or 2 items
    {"_or_": [...], "pick": 1, "then_pick": 2}   -> Second-order selection
    {"_or_": [...], "pick": 2, "_mutex_": [["A", "B"]]} -> A and B can't be together
"""

from itertools import combinations, permutations, product
from math import comb, factorial
from typing import Any, FrozenSet, List, Optional, Tuple, Union

from .base import ExpansionStrategy, GeneratorNode, ExpandedResult, SizeSpec
from .registry import register_strategy
from ..keywords import (
    OR_KEYWORD, SIZE_KEYWORD, COUNT_KEYWORD,
    PICK_KEYWORD, ARRANGE_KEYWORD,
    THEN_PICK_KEYWORD, THEN_ARRANGE_KEYWORD,
    MUTEX_KEYWORD, REQUIRES_KEYWORD, EXCLUDE_KEYWORD,
    PURE_OR_KEYS
)
from ..utils.sampling import sample_with_seed


@register_strategy
class OrStrategy(ExpansionStrategy):
    """Strategy for handling _or_ nodes with selection semantics.

    Supports:
        - Basic choice expansion (each alternative becomes a variant)
        - pick: Unordered selection using combinations
        - arrange: Ordered arrangement using permutations
        - size: Legacy alias for pick (backward compatibility)
        - Second-order selection via then_pick/then_arrange or [outer, inner]
        - count: Limit number of generated variants
        - Constraints: _mutex_, _requires_, _exclude_ for filtering (Phase 4)

    Attributes:
        keywords: {_or_, size, count, pick, arrange, then_pick, then_arrange,
                   _mutex_, _requires_, _exclude_}
        priority: 10 (standard priority)
    """

    keywords: FrozenSet[str] = PURE_OR_KEYS
    priority: int = 10

    @classmethod
    def handles(cls, node: GeneratorNode) -> bool:
        """Check if node is a pure OR node.

        A pure OR node contains _or_ and only OR-related modifier keys.

        Args:
            node: Dictionary node to check.

        Returns:
            True if node is a pure OR node.
        """
        if not isinstance(node, dict):
            return False
        return OR_KEYWORD in node and set(node.keys()).issubset(PURE_OR_KEYS)

    def expand(
        self,
        node: GeneratorNode,
        seed: Optional[int] = None,
        expand_nested: Optional[callable] = None
    ) -> ExpandedResult:
        """Expand an OR node to list of variants.

        Args:
            node: OR specification node.
            seed: Optional seed for random sampling.
            expand_nested: Callback to expand nested generator nodes.

        Returns:
            List of expanded variants.
        """
        choices = node[OR_KEYWORD]
        size = node.get(SIZE_KEYWORD)
        pick = node.get(PICK_KEYWORD)
        arrange = node.get(ARRANGE_KEYWORD)
        then_pick = node.get(THEN_PICK_KEYWORD)
        then_arrange = node.get(THEN_ARRANGE_KEYWORD)
        count = node.get(COUNT_KEYWORD)

        # Extract constraint specifications (Phase 4)
        mutex_groups = node.get(MUTEX_KEYWORD, [])
        requires_groups = node.get(REQUIRES_KEYWORD, [])
        exclude_combos = node.get(EXCLUDE_KEYWORD, [])

        # Determine selection mode: arrange > pick > size (backward compat)
        if arrange is not None:
            result = self._expand_with_arrange(
                choices, arrange, then_pick, then_arrange, expand_nested, seed
            )
        elif pick is not None:
            result = self._expand_with_pick(
                choices, pick, then_pick, then_arrange, expand_nested, seed
            )
        elif size is not None:
            # Legacy size behaves like pick (combinations)
            result = self._expand_with_pick(
                choices, size, then_pick, then_arrange, expand_nested, seed
            )
        else:
            # Basic expansion: each choice becomes a variant
            result = self._expand_basic(choices, expand_nested)

        # Apply constraints if specified (Phase 4)
        if mutex_groups or requires_groups or exclude_combos:
            result = self._apply_constraints(
                result, mutex_groups, requires_groups, exclude_combos
            )

        # Apply count limit if specified
        if count is not None and len(result) > count:
            result = sample_with_seed(result, count, seed=seed)

        return result

    def count(self, node: GeneratorNode, count_nested: Optional[callable] = None) -> int:
        """Count OR node variants without generating them.

        Args:
            node: OR specification node.
            count_nested: Callback to count nested nodes.

        Returns:
            Number of variants.
        """
        choices = node[OR_KEYWORD]
        size = node.get(SIZE_KEYWORD)
        pick = node.get(PICK_KEYWORD)
        arrange = node.get(ARRANGE_KEYWORD)
        then_pick = node.get(THEN_PICK_KEYWORD)
        then_arrange = node.get(THEN_ARRANGE_KEYWORD)
        count_limit = node.get(COUNT_KEYWORD)

        # Determine selection mode
        if arrange is not None:
            total = self._count_with_arrange(
                choices, arrange, then_pick, then_arrange, count_nested
            )
        elif pick is not None:
            total = self._count_with_pick(
                choices, pick, then_pick, then_arrange, count_nested
            )
        elif size is not None:
            total = self._count_with_pick(
                choices, size, then_pick, then_arrange, count_nested
            )
        else:
            # Basic count: sum of each choice's count
            total = 0
            for choice in choices:
                if count_nested:
                    total += count_nested(choice)
                else:
                    total += 1

        # Apply count limit
        if count_limit is not None:
            return min(count_limit, total)
        return total

    def validate(self, node: GeneratorNode) -> List[str]:
        """Validate OR node specification.

        Args:
            node: OR node to validate.

        Returns:
            List of error messages. Empty if valid.
        """
        errors = []
        choices = node.get(OR_KEYWORD)

        if choices is None:
            errors.append("Missing _or_ key")
            return errors

        if not isinstance(choices, list):
            errors.append(f"_or_ must be a list, got {type(choices).__name__}")

        # Validate pick/arrange specs
        for key in (PICK_KEYWORD, ARRANGE_KEYWORD, SIZE_KEYWORD):
            if key in node:
                spec = node[key]
                if not self._is_valid_size_spec(spec):
                    errors.append(
                        f"{key} must be int, tuple (from, to), or list [outer, inner], "
                        f"got {type(spec).__name__}"
                    )

        # Validate count
        count = node.get(COUNT_KEYWORD)
        if count is not None and not isinstance(count, int):
            errors.append(f"count must be an integer, got {type(count).__name__}")

        return errors

    # -------------------------------------------------------------------------
    # Basic Expansion
    # -------------------------------------------------------------------------

    def _expand_basic(
        self,
        choices: List[Any],
        expand_nested: Optional[callable]
    ) -> ExpandedResult:
        """Expand basic OR (each choice is a variant).

        Args:
            choices: List of choice values.
            expand_nested: Callback to expand nested nodes.

        Returns:
            List of expanded variants.
        """
        result = []
        for choice in choices:
            if expand_nested:
                expanded = expand_nested(choice)
                result.extend(expanded)
            else:
                result.append(choice)
        return result

    # -------------------------------------------------------------------------
    # Pick Expansion (Combinations)
    # -------------------------------------------------------------------------

    def _expand_with_pick(
        self,
        choices: List[Any],
        pick_spec: SizeSpec,
        then_pick: Optional[SizeSpec],
        then_arrange: Optional[SizeSpec],
        expand_nested: Optional[callable],
        seed: Optional[int]
    ) -> ExpandedResult:
        """Expand using pick (combinations).

        Args:
            choices: List of choices.
            pick_spec: Size specification for pick.
            then_pick: Optional second-order pick.
            then_arrange: Optional second-order arrange.
            expand_nested: Callback for nested expansion.
            seed: Random seed.

        Returns:
            List of combinations.
        """
        # Handle second-order with then_pick
        if then_pick is not None:
            return self._handle_pick_then_pick(choices, pick_spec, then_pick)

        # Handle second-order with then_arrange
        if then_arrange is not None:
            return self._handle_pick_then_arrange(choices, pick_spec, then_arrange)

        # Standard pick expansion
        # pick_spec can be: int (exact), tuple/list of 2 ints (range from, to)
        from_size, to_size = self._normalize_spec(pick_spec)
        result = []

        for s in range(from_size, to_size + 1):
            if s > len(choices):
                continue
            if s == 0:
                result.append([])
                continue
            for combo in combinations(choices, s):
                combo_results = self._expand_combination(combo, expand_nested)
                result.extend(combo_results)

        return result

    def _count_with_pick(
        self,
        choices: List[Any],
        pick_spec: SizeSpec,
        then_pick: Optional[SizeSpec],
        then_arrange: Optional[SizeSpec],
        count_nested: Optional[callable]
    ) -> int:
        """Count pick (combinations) variants.

        Args:
            choices: List of choices.
            pick_spec: Size specification.
            then_pick: Optional second-order pick.
            then_arrange: Optional second-order arrange.
            count_nested: Callback for nested counting.

        Returns:
            Number of combinations.
        """
        n = len(choices)

        # Handle second-order with then_pick
        if then_pick is not None:
            return self._count_pick_then_pick(n, pick_spec, then_pick)

        # Handle second-order with then_arrange
        if then_arrange is not None:
            return self._count_pick_then_arrange(n, pick_spec, then_arrange)

        # Standard count
        # pick_spec can be: int (exact), tuple/list of 2 ints (range from, to)
        from_size, to_size = self._normalize_spec(pick_spec)
        total = 0
        for s in range(from_size, to_size + 1):
            if s <= n:
                total += comb(n, s)
        return total

    # -------------------------------------------------------------------------
    # Arrange Expansion (Permutations)
    # -------------------------------------------------------------------------

    def _expand_with_arrange(
        self,
        choices: List[Any],
        arrange_spec: SizeSpec,
        then_pick: Optional[SizeSpec],
        then_arrange: Optional[SizeSpec],
        expand_nested: Optional[callable],
        seed: Optional[int]
    ) -> ExpandedResult:
        """Expand using arrange (permutations).

        Args:
            choices: List of choices.
            arrange_spec: Size specification for arrange.
            then_pick: Optional second-order pick.
            then_arrange: Optional second-order arrange.
            expand_nested: Callback for nested expansion.
            seed: Random seed.

        Returns:
            List of permutations.
        """
        # Handle second-order with then_pick
        if then_pick is not None:
            return self._handle_arrange_then_pick(choices, arrange_spec, then_pick)

        # Handle second-order with then_arrange
        if then_arrange is not None:
            return self._handle_arrange_then_arrange(choices, arrange_spec, then_arrange)

        # Standard arrange expansion
        # arrange_spec can be: int (exact), tuple/list of 2 ints (range from, to)
        from_size, to_size = self._normalize_spec(arrange_spec)
        result = []

        for s in range(from_size, to_size + 1):
            if s > len(choices):
                continue
            if s == 0:
                result.append([])
                continue
            for perm in permutations(choices, s):
                perm_results = self._expand_combination(perm, expand_nested)
                result.extend(perm_results)

        return result

    def _count_with_arrange(
        self,
        choices: List[Any],
        arrange_spec: SizeSpec,
        then_pick: Optional[SizeSpec],
        then_arrange: Optional[SizeSpec],
        count_nested: Optional[callable]
    ) -> int:
        """Count arrange (permutations) variants.

        Args:
            choices: List of choices.
            arrange_spec: Size specification.
            then_pick: Optional second-order pick.
            then_arrange: Optional second-order arrange.
            count_nested: Callback for nested counting.

        Returns:
            Number of permutations.
        """
        n = len(choices)

        # Handle second-order with then_pick
        if then_pick is not None:
            return self._count_arrange_then_pick(n, arrange_spec, then_pick)

        # Handle second-order with then_arrange
        if then_arrange is not None:
            return self._count_arrange_then_arrange(n, arrange_spec, then_arrange)

        # Standard count: P(n, k) = n! / (n-k)!
        # arrange_spec can be: int (exact), tuple/list of 2 ints (range from, to)
        from_size, to_size = self._normalize_spec(arrange_spec)
        total = 0
        for s in range(from_size, to_size + 1):
            if s <= n:
                total += factorial(n) // factorial(n - s)
        return total
    # -------------------------------------------------------------------------
    # Second-Order (then_pick / then_arrange)
    # -------------------------------------------------------------------------

    def _handle_pick_then_pick(
        self,
        choices: List[Any],
        primary_spec: SizeSpec,
        then_spec: SizeSpec
    ) -> ExpandedResult:
        """Pick from choices, then pick from results."""
        primary_from, primary_to = self._normalize_spec(primary_spec)
        then_from, then_to = self._normalize_spec(then_spec)

        # Step 1: Generate primary combinations
        primary_items = []
        for s in range(primary_from, primary_to + 1):
            if s > len(choices):
                continue
            for combo in combinations(choices, s):
                if len(combo) == 1:
                    primary_items.append(combo[0])
                else:
                    primary_items.append(list(combo))

        # Step 2: Apply then_pick (combinations)
        result = []
        for s in range(then_from, then_to + 1):
            if s > len(primary_items):
                continue
            for combo in combinations(primary_items, s):
                result.append(list(combo))

        return result

    def _count_pick_then_pick(
        self,
        n: int,
        primary_spec: SizeSpec,
        then_spec: SizeSpec
    ) -> int:
        """Count pick-then-pick combinations."""
        primary_from, primary_to = self._normalize_spec(primary_spec)
        then_from, then_to = self._normalize_spec(then_spec)

        # Count primary combinations
        total_primary = sum(
            comb(n, s) for s in range(primary_from, primary_to + 1) if s <= n
        )

        # Count then_pick combinations
        return sum(
            comb(total_primary, s)
            for s in range(then_from, then_to + 1) if s <= total_primary
        )

    def _handle_pick_then_arrange(
        self,
        choices: List[Any],
        primary_spec: SizeSpec,
        then_spec: SizeSpec
    ) -> ExpandedResult:
        """Pick from choices, then arrange results."""
        primary_from, primary_to = self._normalize_spec(primary_spec)
        then_from, then_to = self._normalize_spec(then_spec)

        # Step 1: Generate primary combinations
        primary_items = []
        for s in range(primary_from, primary_to + 1):
            if s > len(choices):
                continue
            for combo in combinations(choices, s):
                if len(combo) == 1:
                    primary_items.append(combo[0])
                else:
                    primary_items.append(list(combo))

        # Step 2: Apply then_arrange (permutations)
        result = []
        for s in range(then_from, then_to + 1):
            if s > len(primary_items):
                continue
            for perm in permutations(primary_items, s):
                result.append(list(perm))

        return result

    def _count_pick_then_arrange(
        self,
        n: int,
        primary_spec: SizeSpec,
        then_spec: SizeSpec
    ) -> int:
        """Count pick-then-arrange permutations."""
        primary_from, primary_to = self._normalize_spec(primary_spec)
        then_from, then_to = self._normalize_spec(then_spec)

        # Count primary combinations
        total_primary = sum(
            comb(n, s) for s in range(primary_from, primary_to + 1) if s <= n
        )

        # Count then_arrange permutations
        return sum(
            factorial(total_primary) // factorial(total_primary - s)
            for s in range(then_from, then_to + 1) if s <= total_primary
        )

    def _handle_arrange_then_pick(
        self,
        choices: List[Any],
        primary_spec: SizeSpec,
        then_spec: SizeSpec
    ) -> ExpandedResult:
        """Arrange from choices, then pick from results."""
        primary_from, primary_to = self._normalize_spec(primary_spec)
        then_from, then_to = self._normalize_spec(then_spec)

        # Step 1: Generate primary permutations
        primary_items = []
        for s in range(primary_from, primary_to + 1):
            if s > len(choices):
                continue
            for perm in permutations(choices, s):
                if len(perm) == 1:
                    primary_items.append(perm[0])
                else:
                    primary_items.append(list(perm))

        # Step 2: Apply then_pick (combinations)
        result = []
        for s in range(then_from, then_to + 1):
            if s > len(primary_items):
                continue
            for combo in combinations(primary_items, s):
                result.append(list(combo))

        return result

    def _count_arrange_then_pick(
        self,
        n: int,
        primary_spec: SizeSpec,
        then_spec: SizeSpec
    ) -> int:
        """Count arrange-then-pick combinations."""
        primary_from, primary_to = self._normalize_spec(primary_spec)
        then_from, then_to = self._normalize_spec(then_spec)

        # Count primary permutations
        total_primary = sum(
            factorial(n) // factorial(n - s)
            for s in range(primary_from, primary_to + 1) if s <= n
        )

        # Count then_pick combinations
        return sum(
            comb(total_primary, s)
            for s in range(then_from, then_to + 1) if s <= total_primary
        )

    def _handle_arrange_then_arrange(
        self,
        choices: List[Any],
        primary_spec: SizeSpec,
        then_spec: SizeSpec
    ) -> ExpandedResult:
        """Arrange from choices, then arrange results."""
        primary_from, primary_to = self._normalize_spec(primary_spec)
        then_from, then_to = self._normalize_spec(then_spec)

        # Step 1: Generate primary permutations
        primary_items = []
        for s in range(primary_from, primary_to + 1):
            if s > len(choices):
                continue
            for perm in permutations(choices, s):
                if len(perm) == 1:
                    primary_items.append(perm[0])
                else:
                    primary_items.append(list(perm))

        # Step 2: Apply then_arrange (permutations)
        result = []
        for s in range(then_from, then_to + 1):
            if s > len(primary_items):
                continue
            for perm in permutations(primary_items, s):
                result.append(list(perm))

        return result

    def _count_arrange_then_arrange(
        self,
        n: int,
        primary_spec: SizeSpec,
        then_spec: SizeSpec
    ) -> int:
        """Count arrange-then-arrange permutations."""
        primary_from, primary_to = self._normalize_spec(primary_spec)
        then_from, then_to = self._normalize_spec(then_spec)

        # Count primary permutations
        total_primary = sum(
            factorial(n) // factorial(n - s)
            for s in range(primary_from, primary_to + 1) if s <= n
        )

        # Count then_arrange permutations
        return sum(
            factorial(total_primary) // factorial(total_primary - s)
            for s in range(then_from, then_to + 1) if s <= total_primary
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _normalize_spec(self, spec: SizeSpec) -> Tuple[int, int]:
        """Normalize size specification to (from, to) tuple.

        Args:
            spec: Size specification (int, tuple, or list).

        Returns:
            Tuple of (from_size, to_size).

        Raises:
            ValueError: If specification is invalid.
        """
        if isinstance(spec, int):
            return (spec, spec)
        elif isinstance(spec, (tuple, list)) and len(spec) == 2:
            return (spec[0], spec[1])
        else:
            raise ValueError(f"Invalid size spec: {spec}. Must be int or (from, to).")

    def _is_valid_size_spec(self, spec: Any) -> bool:
        """Check if a size specification is valid."""
        if isinstance(spec, int):
            return True
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            return all(isinstance(x, int) for x in spec)
        return False

    def _expand_combination(
        self,
        combo: tuple,
        expand_nested: Optional[callable]
    ) -> List[List[Any]]:
        """Expand a combination by taking Cartesian product of expanded elements.

        Args:
            combo: Tuple of items from combinations/permutations.
            expand_nested: Callback to expand nested nodes.

        Returns:
            List of expanded combinations.
        """
        if expand_nested:
            expanded_elements = [expand_nested(item) for item in combo]
        else:
            expanded_elements = [[item] for item in combo]

        # Take Cartesian product
        results = []
        for expanded_combo in product(*expanded_elements):
            results.append(list(expanded_combo))

        return results

    # -------------------------------------------------------------------------
    # Constraint Handling (Phase 4)
    # -------------------------------------------------------------------------

    def _apply_constraints(
        self,
        results: ExpandedResult,
        mutex_groups: List[List[Any]],
        requires_groups: List[List[Any]],
        exclude_combos: List[List[Any]]
    ) -> ExpandedResult:
        """Apply constraint filters to expanded results.

        Args:
            results: List of expanded combinations.
            mutex_groups: Mutual exclusion groups - items that can't appear together.
            requires_groups: Dependency pairs - if A is present, B must be too.
            exclude_combos: Specific combinations to exclude.

        Returns:
            Filtered results satisfying all constraints.

        Examples:
            >>> # Mutex: A and B can't be together
            >>> self._apply_constraints(
            ...     [["A","B"], ["A","C"], ["B","C"]],
            ...     mutex_groups=[["A","B"]], requires_groups=[], exclude_combos=[]
            ... )
            [["A","C"], ["B","C"]]
        """
        from ..constraints import apply_all_constraints

        # Only filter list results (combinations/permutations)
        # Basic single-choice results don't need constraint filtering
        if not results or not isinstance(results[0], list):
            return results

        return apply_all_constraints(
            results,
            mutex_groups=mutex_groups,
            requires_groups=requires_groups,
            exclude_combos=exclude_combos
        )
