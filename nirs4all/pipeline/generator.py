# Fix: Add list handling to expand_spec function
from itertools import product, combinations, permutations
from collections.abc import Mapping
from math import comb, factorial
import random

def expand_spec(node):
    # NEW: Handle lists by expanding each element and taking the product
    if isinstance(node, list):
        if not node:
            return [[]]  # Empty list -> single empty result

        # Special case: if there's only one element in the list, check if we should unwrap
        if len(node) == 1:
            element_result = expand_spec(node[0])

            # If the result contains lists (combinations), return directly to prevent extra wrapping
            # If the result contains scalars, we still need to wrap them properly
            if element_result and isinstance(element_result[0], list):
                return element_result
            # Otherwise, fall through to normal processing

        # Expand each element in the list
        expanded_elements = [expand_spec(element) for element in node]

        # Take Cartesian product of all expansions
        results = []
        for combo in product(*expanded_elements):
            results.append(list(combo))  # Convert tuple to list
        return results

    # Rest of the logic remains the same for dictionaries
    if not isinstance(node, Mapping):
        return [node]

    # NEW: Handle _range_ node for numeric sequences
    if set(node.keys()).issubset({"_range_", "count"}):
        range_spec = node["_range_"]
        count = node.get("count", None)

        # Generate numeric range based on specification
        range_values = _generate_range(range_spec)

        # Apply count limit if specified
        if count is not None and len(range_values) > count:
            range_values = random.sample(range_values, count)

        return range_values

    # Case 1: pure OR node (with optional size, count)
    if set(node.keys()).issubset({"_or_", "size", "count"}):
        choices = node["_or_"]
        size = node.get("size", None)
        count = node.get("count", None)

        if size is not None:
            # NEW: Check for second-order array notation [outer, inner]
            if isinstance(size, list) and len(size) == 2:
                results = _handle_nested_combinations(choices, size)
                # Apply count limit if specified
                if count is not None and len(results) > count:
                    results = random.sample(results, count)
                return results

            # Apply size constraints first, then expand
            out = []

            # Handle tuple size (from, to) or single size
            if isinstance(size, tuple) and len(size) == 2:
                from_size, to_size = size
                # Generate combinations for all sizes from from_size to to_size (inclusive)
                for s in range(from_size, to_size + 1):
                    if s > len(choices):
                        continue
                    for combo in combinations(choices, s):
                        # Expand this specific combination and return as individual elements
                        combo_results = _expand_combination(combo)
                        out.extend(combo_results)
            else:
                # Single size value
                if size <= len(choices):
                    for combo in combinations(choices, size):
                        # Expand this specific combination and return as individual elements
                        combo_results = _expand_combination(combo)
                        out.extend(combo_results)

            # Apply count limit if specified
            if count is not None and len(out) > count:
                out = random.sample(out, count)

            return out
        else:
            # Original behavior: expand all choices
            out = []
            for choice in choices:
                out.extend(expand_spec(choice))

            # Apply count limit if specified
            if count is not None and len(out) > count:
                out = random.sample(out, count)

            return out

    # Case 2: dict that also contains "_or_" -> branch and merge
    if "_or_" in node:
        # Extract size and count if present
        size = node.get("size", None)
        count = node.get("count", None)
        base = {k: v for k, v in node.items() if k not in ["_or_", "size", "count"]}
        base_expanded = expand_spec(base)            # list[dict]

        # Create a temporary or node with size and count
        or_node = {"_or_": node["_or_"]}
        if size is not None:
            or_node["size"] = size
        if count is not None:
            or_node["count"] = count

        choice_expanded = expand_spec(or_node)  # list[dict or scalar]
        results = []
        for b in base_expanded:
            for c in choice_expanded:
                if isinstance(c, Mapping):
                    merged = {**b, **c}
                    results.append(merged)
                else:
                    # Scalar choices only make sense as values under a key, not top-level merges
                    raise ValueError("Top-level 'or' choices must be dicts.")
        return results

    # Case 3: normal dict -> product over keys
    keys, options = zip(*[(k, _expand_value(v)) for k, v in node.items()]) if node else ([], [])
    if not keys:
        return [{}]
    out = []
    for combo in product(*options):
        d = {}
        for k, v in zip(keys, combo):
            d[k] = v
        out.append(d)
    return out

def count_combinations(node):
    """Calculate total number of combinations without generating them.
    Returns the count of results that expand_spec would produce.
    """
    # Handle lists by taking product of counts
    if isinstance(node, list):
        if not node:
            return 1  # Empty list -> single empty result

        total_count = 1
        for element in node:
            total_count *= count_combinations(element)
        return total_count

    # Scalars return 1
    if not isinstance(node, Mapping):
        return 1

    # NEW: Handle _range_ node for counting numeric sequences
    if set(node.keys()).issubset({"_range_", "count"}):
        range_spec = node["_range_"]
        count = node.get("count", None)

        # Calculate range size
        range_size = _count_range(range_spec)

        # Apply count limit if specified
        if count is not None:
            return min(count, range_size)
        return range_size

    # Case 1: pure OR node (with optional size, count)
    if set(node.keys()).issubset({"_or_", "size", "count"}):
        choices = node["_or_"]
        size = node.get("size", None)
        count = node.get("count", None)

        if size is not None:
            # Check for second-order array notation [outer, inner]
            if isinstance(size, list) and len(size) == 2:
                total_count = _count_nested_combinations(choices, size)
                # Apply count limit if specified
                if count is not None:
                    return min(count, total_count)
                return total_count

            # Handle tuple size (from, to) or single size
            total_count = 0
            if isinstance(size, tuple) and len(size) == 2:
                from_size, to_size = size
                for s in range(from_size, to_size + 1):
                    if s <= len(choices):
                        total_count += comb(len(choices), s)
            else:
                # Single size value
                if size <= len(choices):
                    total_count = comb(len(choices), size)

            # Apply count limit if specified
            if count is not None:
                return min(count, total_count)
            return total_count
        else:
            # All choices - count each choice's expansions
            total_count = 0
            for choice in choices:
                total_count += count_combinations(choice)

            # Apply count limit if specified
            if count is not None:
                return min(count, total_count)
            return total_count

    # Case 2: dict that also contains "_or_" -> branch and merge
    if "_or_" in node:
        size = node.get("size", None)
        count = node.get("count", None)
        base = {k: v for k, v in node.items() if k not in ["_or_", "size", "count"]}

        base_count = count_combinations(base)

        # Create temporary or node
        or_node = {"_or_": node["_or_"]}
        if size is not None:
            or_node["size"] = size
        if count is not None:
            or_node["count"] = count

        choice_count = count_combinations(or_node)

        return base_count * choice_count

    # Case 3: normal dict -> product over keys
    if not node:
        return 1

    total_count = 1
    for k, v in node.items():
        total_count *= _count_value(v)

    return total_count

def _handle_nested_combinations(choices, nested_size):
    """Handle second-order combinations with array syntax [outer, inner]
    outer and inner can be int or tuple (from, to)
    For second-order: inner elements use permutations (order matters within sub-arrays)
    For outer selection: uses combinations (order doesn't matter for selecting which sub-arrays)
    """
    outer_size, inner_size = nested_size

    # Handle inner size - can be int or tuple
    if isinstance(inner_size, int):
        inner_from, inner_to = inner_size, inner_size
    else:  # tuple (from, to)
        inner_from, inner_to = inner_size

    # Handle outer size - can be int or tuple
    if isinstance(outer_size, int):
        outer_from, outer_to = outer_size, outer_size
    else:  # tuple (from, to)
        outer_from, outer_to = outer_size

    # Step 1: Generate all possible inner arrangements (permutations - order matters!)
    inner_arrangements = []
    for inner_s in range(inner_from, inner_to + 1):
        if inner_s > len(choices):
            continue
        # Use permutations for inner elements (order matters within sub-arrays)
        for perm in permutations(choices, inner_s):
            # Each inner arrangement is a simple list
            if len(perm) == 1:
                # Single element - don't wrap in extra list
                inner_arrangements.append(perm[0])
            else:
                # Multiple elements - keep as list
                inner_arrangements.append(list(perm))

    # Step 2: Select outer combinations (combinations - order doesn't matter for selection)
    out = []
    for outer_s in range(outer_from, outer_to + 1):
        if outer_s > len(inner_arrangements):
            continue
        # Use combinations for outer selection (order doesn't matter for which sub-arrays to pick)
        for outer_combo in combinations(inner_arrangements, outer_s):
            # Convert tuple to list and add to results
            out.append(list(outer_combo))

    return out

def _count_nested_combinations(choices, nested_size):
    """Count second-order combinations without generating them."""
    outer_size, inner_size = nested_size

    # Handle inner size - can be int or tuple
    if isinstance(inner_size, int):
        inner_from, inner_to = inner_size, inner_size
    else:
        inner_from, inner_to = inner_size

    # Handle outer size - can be int or tuple
    if isinstance(outer_size, int):
        outer_from, outer_to = outer_size, outer_size
    else:
        outer_from, outer_to = outer_size

    # Count inner arrangements (permutations)
    total_inner_arrangements = 0
    n_choices = len(choices)

    for inner_s in range(inner_from, inner_to + 1):
        if inner_s <= n_choices:
            # Permutations: P(n,k) = n!/(n-k)!
            perms = factorial(n_choices) // factorial(n_choices - inner_s) if inner_s <= n_choices else 0
            total_inner_arrangements += perms

    # Count outer combinations from inner arrangements
    total_count = 0
    for outer_s in range(outer_from, outer_to + 1):
        if outer_s <= total_inner_arrangements:
            total_count += comb(total_inner_arrangements, outer_s)

    return total_count

def _expand_combination(combo):
    """Expand a specific combination of choices by taking their cartesian product."""
    expanded_choices = []
    for choice in combo:
        expanded_choices.append(expand_spec(choice))

    # Take cartesian product
    results = []
    for expanded_combo in product(*expanded_choices):
        # Return the combination as a single list (flatten one level)
        results.append(list(expanded_combo))

    return results

def _expand_value(v):
    # Value position returns a list of *values* (scalars or dicts)
    if isinstance(v, Mapping) and ("_or_" in v.keys()):
        # Value-level OR can yield scalars or dicts as values (with optional size, count)
        choices = v["_or_"]
        size = v.get("size", None)
        count = v.get("count", None)

        if size is not None:
            # NEW: Check for second-order array notation [outer, inner]
            if isinstance(size, list) and len(size) == 2:
                results = _handle_nested_combinations(choices, size)
                # Apply count limit if specified
                if count is not None and len(results) > count:
                    results = random.sample(results, count)
                return results

            # Apply size constraints first, then expand
            vals = []

            # Handle tuple size (from, to) or single size
            if isinstance(size, tuple) and len(size) == 2:
                from_size, to_size = size
                # Generate combinations for all sizes from from_size to to_size (inclusive)
                for s in range(from_size, to_size + 1):
                    if s > len(choices):
                        continue
                    for combo in combinations(choices, s):
                        # Expand this specific combination
                        combo_results = _expand_combination(combo)
                        vals.extend(combo_results)
            else:
                # Single size value
                if size <= len(choices):
                    for combo in combinations(choices, size):
                        # Expand this specific combination
                        combo_results = _expand_combination(combo)
                        vals.extend(combo_results)

            # Apply count limit if specified
            if count is not None and len(vals) > count:
                vals = random.sample(vals, count)

            return vals
        else:
            # Original behavior: expand all choices
            vals = []
            for choice in choices:
                ex = expand_spec(choice)
                # expand_spec returns list; extend with each item (scalar or dict value)
                vals.extend(ex)

            # Apply count limit if specified
            if count is not None and len(vals) > count:
                vals = random.sample(vals, count)

            return vals
    elif isinstance(v, Mapping) and ("_range_" in v.keys()):
        # Handle _range_ in value position
        range_spec = v["_range_"]
        count = v.get("count", None)

        # Generate numeric range based on specification
        range_values = _generate_range(range_spec)

        # Apply count limit if specified
        if count is not None and len(range_values) > count:
            range_values = random.sample(range_values, count)

        return range_values
    elif isinstance(v, Mapping):
        # Nested object: expand to list of dict values
        return expand_spec(v)
    elif isinstance(v, list):
        # Handle lists in value positions
        return expand_spec(v)
    else:
        return [v]

def _count_value(v):
    """Count combinations for a value position."""
    if isinstance(v, Mapping) and ("_or_" in v.keys()):
        return count_combinations(v)
    elif isinstance(v, Mapping) and ("_range_" in v.keys()):
        return count_combinations(v)
    elif isinstance(v, Mapping):
        return count_combinations(v)
    elif isinstance(v, list):
        return count_combinations(v)
    else:
        return 1


def _generate_range(range_spec):
    """Generate numeric range from specification.

    Supports two syntaxes:
    - Array: [from, to, step] where step defaults to 1
    - Dict: {"from": start, "to": end, "step": step}
    """
    if isinstance(range_spec, list):
        if len(range_spec) == 2:
            start, end = range_spec
            step = 1
        elif len(range_spec) == 3:
            start, end, step = range_spec
        else:
            raise ValueError("Range array must be [from, to] or [from, to, step]")
    elif isinstance(range_spec, dict):
        start = range_spec["from"]
        end = range_spec["to"]
        step = range_spec.get("step", 1)
    else:
        raise ValueError("Range specification must be array [from, to, step] or dict {'from': start, 'to': end, 'step': step}")

    # Generate range - end is inclusive
    if step > 0:
        return list(range(start, end + 1, step))
    else:
        # For negative steps, we need to ensure end is included
        return list(range(start, end - 1, step))


def _count_range(range_spec):
    """Count elements in a numeric range without generating them."""
    if isinstance(range_spec, list):
        if len(range_spec) == 2:
            start, end = range_spec
            step = 1
        elif len(range_spec) == 3:
            start, end, step = range_spec
        else:
            raise ValueError("Range array must be [from, to] or [from, to, step]")
    elif isinstance(range_spec, dict):
        start = range_spec["from"]
        end = range_spec["to"]
        step = range_spec.get("step", 1)
    else:
        raise ValueError("Range specification must be array [from, to, step] or dict {'from': start, 'to': end, 'step': step}")

    # Calculate count: (end - start) // step + 1 (end is inclusive)
    if step > 0 and end >= start:
        return (end - start) // step + 1
    elif step < 0 and end <= start:
        return (start - end) // abs(step) + 1
    else:
        return 0
