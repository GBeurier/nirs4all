"""Export and visualization utilities for expanded configurations.

This module provides tools for inspecting, exporting, and visualizing
configuration spaces and expanded configurations.

Main Functions:
    to_dataframe(configs): Convert configs to pandas DataFrame
    diff_configs(config1, config2): Show differences between configs
    get_expansion_tree(spec): Get tree representation of config space
    print_expansion_tree(spec): Print tree to console
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union


def to_dataframe(
    configs: List[Any],
    flatten: bool = True,
    prefix_sep: str = ".",
    include_index: bool = True
) -> Any:
    """Convert expanded configurations to a pandas DataFrame.

    Args:
        configs: List of expanded configurations.
        flatten: If True, flatten nested dicts with dot notation.
        prefix_sep: Separator for flattened keys (default ".").
        include_index: If True, include a config index column.

    Returns:
        pandas DataFrame with one row per configuration.

    Raises:
        ImportError: If pandas is not installed.

    Examples:
        >>> configs = [
        ...     {"model": "PLS", "n_components": 5},
        ...     {"model": "PLS", "n_components": 10},
        ...     {"model": "RF", "n_estimators": 100}
        ... ]
        >>> df = to_dataframe(configs)
        >>> df.columns.tolist()
        ['config_index', 'model', 'n_components', 'n_estimators']
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for to_dataframe(). "
            "Install with: pip install pandas"
        )

    if not configs:
        return pd.DataFrame()

    rows = []
    for i, config in enumerate(configs):
        if flatten and isinstance(config, dict):
            row = _flatten_dict(config, prefix_sep=prefix_sep)
        elif isinstance(config, dict):
            row = dict(config)
        else:
            row = {"value": config}

        if include_index:
            row = {"config_index": i, **row}

        rows.append(row)

    return pd.DataFrame(rows)


def _flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    prefix_sep: str = "."
) -> Dict[str, Any]:
    """Flatten a nested dictionary.

    Args:
        d: Dict to flatten.
        parent_key: Prefix for keys.
        prefix_sep: Separator between parent and child keys.

    Returns:
        Flattened dict.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{prefix_sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, prefix_sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def diff_configs(
    config1: Any,
    config2: Any,
    path: str = ""
) -> Dict[str, Tuple[Any, Any]]:
    """Find differences between two configurations.

    Args:
        config1: First configuration.
        config2: Second configuration.
        path: Current path (for nested diff reporting).

    Returns:
        Dict mapping paths to (value1, value2) tuples where values differ.

    Examples:
        >>> config1 = {"model": "PLS", "n_components": 5}
        >>> config2 = {"model": "PLS", "n_components": 10}
        >>> diff_configs(config1, config2)
        {'n_components': (5, 10)}

        >>> config1 = {"a": {"b": 1}}
        >>> config2 = {"a": {"b": 2}}
        >>> diff_configs(config1, config2)
        {'a.b': (1, 2)}
    """
    diffs = {}

    if not isinstance(config1, type(config2)) and not isinstance(config2, type(config1)):
        diffs[path or "root"] = (config1, config2)
        return diffs

    if isinstance(config1, dict) and isinstance(config2, dict):
        all_keys = set(config1.keys()) | set(config2.keys())
        for key in all_keys:
            sub_path = f"{path}.{key}" if path else key
            if key not in config1:
                diffs[sub_path] = (None, config2[key])
            elif key not in config2:
                diffs[sub_path] = (config1[key], None)
            else:
                sub_diffs = diff_configs(config1[key], config2[key], sub_path)
                diffs.update(sub_diffs)
    elif isinstance(config1, list) and isinstance(config2, list):
        if config1 != config2:
            diffs[path or "root"] = (config1, config2)
    elif config1 != config2:
        diffs[path or "root"] = (config1, config2)

    return diffs


def summarize_configs(
    configs: List[Any],
    max_unique: int = 10
) -> Dict[str, Any]:
    """Summarize a list of configurations.

    Args:
        configs: List of configurations to summarize.
        max_unique: Maximum unique values to show per key.

    Returns:
        Summary dict with statistics for each key.

    Examples:
        >>> configs = [
        ...     {"model": "PLS", "n": 5},
        ...     {"model": "PLS", "n": 10},
        ...     {"model": "RF", "n": 5}
        ... ]
        >>> summary = summarize_configs(configs)
        >>> summary["model"]["unique_values"]
        ['PLS', 'RF']
    """
    if not configs:
        return {"count": 0, "keys": {}}

    summary = {
        "count": len(configs),
        "keys": {}
    }

    # Collect values for each key
    key_values: Dict[str, List[Any]] = {}
    for config in configs:
        if not isinstance(config, dict):
            continue
        for key, value in config.items():
            if key not in key_values:
                key_values[key] = []
            key_values[key].append(value)

    # Summarize each key
    for key, values in key_values.items():
        unique = list(set(str(v) for v in values))
        summary["keys"][key] = {
            "count": len(values),
            "unique_count": len(unique),
            "unique_values": unique[:max_unique],
            "truncated": len(unique) > max_unique
        }

    return summary


# =============================================================================
# Tree Visualization
# =============================================================================

class ExpansionTreeNode:
    """Node in an expansion tree visualization."""

    def __init__(
        self,
        key: str,
        node_type: str,
        count: int,
        children: Optional[List["ExpansionTreeNode"]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize tree node.

        Args:
            key: Node key/name.
            node_type: Type of node (e.g., "_or_", "_range_", "dict").
            count: Number of variants this node generates.
            children: Child nodes.
            details: Additional node details.
        """
        self.key = key
        self.node_type = node_type
        self.count = count
        self.children = children or []
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dict representation."""
        result = {
            "key": self.key,
            "type": self.node_type,
            "count": self.count,
        }
        if self.details:
            result["details"] = self.details
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        return result


def get_expansion_tree(
    spec: Any,
    key: str = "root"
) -> ExpansionTreeNode:
    """Build an expansion tree for a specification.

    Args:
        spec: Specification to analyze.
        key: Key name for this node.

    Returns:
        ExpansionTreeNode representing the configuration space.

    Examples:
        >>> spec = {"x": {"_or_": [1, 2]}, "y": {"_range_": [1, 3]}}
        >>> tree = get_expansion_tree(spec)
        >>> tree.count
        6  # 2 x 3
    """
    from ..keywords import (
        OR_KEYWORD, RANGE_KEYWORD, LOG_RANGE_KEYWORD,
        GRID_KEYWORD, ZIP_KEYWORD, CHAIN_KEYWORD, SAMPLE_KEYWORD,
        SIZE_KEYWORD, PICK_KEYWORD, ARRANGE_KEYWORD
    )
    # Late import to avoid circular dependency
    from ..core import count_combinations

    if not isinstance(spec, dict):
        if isinstance(spec, list):
            return ExpansionTreeNode(
                key=key,
                node_type="list",
                count=len(spec) if spec else 1,
                details={"length": len(spec)}
            )
        return ExpansionTreeNode(
            key=key,
            node_type="scalar",
            count=1,
            details={"value": str(spec)[:50]}
        )

    # Check for generator keywords
    if OR_KEYWORD in spec:
        choices = spec[OR_KEYWORD]
        size = spec.get(SIZE_KEYWORD) or spec.get(PICK_KEYWORD) or spec.get(ARRANGE_KEYWORD)

        children = []
        for i, choice in enumerate(choices):
            child = get_expansion_tree(choice, f"[{i}]")
            children.append(child)

        details = {"choices": len(choices)}
        if size:
            details["size"] = size

        return ExpansionTreeNode(
            key=key,
            node_type="_or_",
            count=count_combinations(spec),
            children=children,
            details=details
        )

    if RANGE_KEYWORD in spec:
        range_spec = spec[RANGE_KEYWORD]
        return ExpansionTreeNode(
            key=key,
            node_type="_range_",
            count=count_combinations(spec),
            details={"range": str(range_spec)}
        )

    if LOG_RANGE_KEYWORD in spec:
        range_spec = spec[LOG_RANGE_KEYWORD]
        return ExpansionTreeNode(
            key=key,
            node_type="_log_range_",
            count=count_combinations(spec),
            details={"range": str(range_spec)}
        )

    if GRID_KEYWORD in spec:
        grid_spec = spec[GRID_KEYWORD]
        children = []
        for k, v in grid_spec.items():
            child = get_expansion_tree(v, k)
            children.append(child)
        return ExpansionTreeNode(
            key=key,
            node_type="_grid_",
            count=count_combinations(spec),
            children=children
        )

    if ZIP_KEYWORD in spec:
        zip_spec = spec[ZIP_KEYWORD]
        children = []
        for k, v in zip_spec.items():
            child = get_expansion_tree(v, k)
            children.append(child)
        return ExpansionTreeNode(
            key=key,
            node_type="_zip_",
            count=count_combinations(spec),
            children=children
        )

    if CHAIN_KEYWORD in spec:
        chain_spec = spec[CHAIN_KEYWORD]
        children = []
        for i, item in enumerate(chain_spec):
            child = get_expansion_tree(item, f"[{i}]")
            children.append(child)
        return ExpansionTreeNode(
            key=key,
            node_type="_chain_",
            count=count_combinations(spec),
            children=children
        )

    if SAMPLE_KEYWORD in spec:
        sample_spec = spec[SAMPLE_KEYWORD]
        return ExpansionTreeNode(
            key=key,
            node_type="_sample_",
            count=count_combinations(spec),
            details=sample_spec
        )

    # Regular dict
    children = []
    total_count = 1
    for k, v in spec.items():
        child = get_expansion_tree(v, k)
        children.append(child)
        total_count *= child.count

    return ExpansionTreeNode(
        key=key,
        node_type="dict",
        count=total_count,
        children=children,
        details={"keys": len(spec)}
    )


def print_expansion_tree(
    spec: Any,
    indent: str = "  ",
    show_counts: bool = True,
    max_depth: Optional[int] = None
) -> str:
    """Format expansion tree as a printable string.

    Args:
        spec: Specification to visualize.
        indent: Indentation string.
        show_counts: Whether to show counts in output.
        max_depth: Maximum depth to display.

    Returns:
        Formatted tree string.

    Examples:
        >>> spec = {"x": {"_or_": [1, 2]}, "y": {"_range_": [1, 3]}}
        >>> print(print_expansion_tree(spec))
        root (6 variants)
        ├── x: _or_ (2 variants)
        │   ├── [0]: scalar
        │   └── [1]: scalar
        └── y: _range_ (3 variants)
    """
    tree = get_expansion_tree(spec)
    lines = []
    _format_tree_node(tree, lines, "", True, show_counts, max_depth, 0)
    return "\n".join(lines)


def _format_tree_node(
    node: ExpansionTreeNode,
    lines: List[str],
    prefix: str,
    is_last: bool,
    show_counts: bool,
    max_depth: Optional[int],
    current_depth: int
) -> None:
    """Recursively format tree node.

    Args:
        node: Node to format.
        lines: Output lines list (modified in-place).
        prefix: Current line prefix.
        is_last: Whether this is the last sibling.
        show_counts: Whether to show counts.
        max_depth: Maximum depth.
        current_depth: Current depth.
    """
    # Build node line
    if prefix:
        connector = "└── " if is_last else "├── "
        line = f"{prefix}{connector}{node.key}: {node.node_type}"
    else:
        line = f"{node.key}: {node.node_type}"

    if show_counts:
        line += f" ({node.count} variants)"

    # Add details for terminal nodes
    if not node.children and node.details:
        if "value" in node.details:
            line += f" = {node.details['value']}"
        elif "range" in node.details:
            line += f" = {node.details['range']}"

    lines.append(line)

    # Check depth limit
    if max_depth is not None and current_depth >= max_depth:
        if node.children:
            child_prefix = prefix + ("    " if is_last else "│   ")
            lines.append(f"{child_prefix}... ({len(node.children)} children)")
        return

    # Format children
    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(node.children):
        child_is_last = (i == len(node.children) - 1)
        _format_tree_node(
            child, lines, child_prefix, child_is_last,
            show_counts, max_depth, current_depth + 1
        )


def format_config_table(
    configs: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    max_rows: int = 20
) -> str:
    """Format configurations as an ASCII table.

    Args:
        configs: List of configuration dicts.
        columns: Specific columns to show (None for auto-detect).
        max_rows: Maximum rows to display.

    Returns:
        Formatted ASCII table string.
    """
    if not configs:
        return "(no configurations)"

    # Determine columns
    if columns is None:
        columns = []
        for config in configs:
            if isinstance(config, dict):
                for key in config.keys():
                    if key not in columns:
                        columns.append(key)

    if not columns:
        return "(no columns)"

    # Calculate column widths
    col_widths = {col: len(col) for col in columns}
    for config in configs[:max_rows]:
        if isinstance(config, dict):
            for col in columns:
                val = str(config.get(col, ""))[:30]
                col_widths[col] = max(col_widths[col], len(val))

    # Build header
    header = " | ".join(col.ljust(col_widths[col]) for col in columns)
    separator = "-+-".join("-" * col_widths[col] for col in columns)

    lines = [header, separator]

    # Build rows
    for i, config in enumerate(configs):
        if i >= max_rows:
            lines.append(f"... ({len(configs) - max_rows} more rows)")
            break
        if isinstance(config, dict):
            row = " | ".join(
                str(config.get(col, ""))[:30].ljust(col_widths[col])
                for col in columns
            )
        else:
            row = str(config)[:80]
        lines.append(row)

    return "\n".join(lines)
