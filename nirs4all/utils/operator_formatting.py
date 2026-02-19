"""Utility functions for formatting operators with their parameters."""

from typing import Any

from nirs4all.pipeline.config.component_serialization import _changed_kwargs


def format_operator_with_params(operator: Any, short_name: str | None = None) -> str:
    """Format an operator name with its non-default parameters.

    Args:
        operator: The operator instance to format
        short_name: Optional short name to use instead of class name

    Returns:
        Formatted string like "SNV(alpha=0.5)" or just "SNV" if no params differ

    Examples:
        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = StandardScaler(with_mean=False)
        >>> format_operator_with_params(scaler)
        'StandardScaler(with_mean=False)'
        >>> format_operator_with_params(scaler, "Std")
        'Std(with_mean=False)'
    """
    # Get the base name
    if short_name is None:
        short_name = operator.__class__.__name__

    # Get non-default parameters
    changed_params = _changed_kwargs(operator)

    # If no parameters differ from defaults, return just the name
    if not changed_params:
        return short_name

    # Format parameters as "key=value" strings
    # Limit string representations to avoid overly long names
    param_strs = []
    for key, value in changed_params.items():
        if isinstance(value, str):
            # Quote strings
            value_str = f"'{value}'" if len(value) < 20 else f"'{value[:17]}...'"
        elif isinstance(value, (int, float, bool, type(None))):
            value_str = str(value)
        elif isinstance(value, (list, tuple)):
            # Show actual values for small collections (≤5 items), otherwise show length
            if len(value) <= 5:
                # Format each item
                formatted_items = []
                for item in value:
                    if isinstance(item, str):
                        formatted_items.append(f"'{item}'" if len(item) < 10 else f"'{item[:7]}...'")
                    else:
                        formatted_items.append(str(item))
                value_str = f"({', '.join(formatted_items)})" if isinstance(value, tuple) else f"[{', '.join(formatted_items)}]"
            else:
                value_str = f"[{len(value)} items]"
        elif isinstance(value, dict):
            value_str = f"{{{len(value)} keys}}"
        else:
            # For objects, just show the type name
            value_str = f"<{type(value).__name__}>"

        param_strs.append(f"{key}={value_str}")

    # Combine into final string
    params_str = ", ".join(param_strs)
    return f"{short_name}({params_str})"
