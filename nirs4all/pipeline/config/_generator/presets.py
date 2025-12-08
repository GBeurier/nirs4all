"""Preset system for named configuration templates.

This module provides a preset registry for storing and retrieving
named configuration templates. Presets allow users to define reusable
configuration patterns and reference them by name.

Usage:
    # Register a preset
    register_preset("spectral_transforms", {
        "_or_": ["SNV", "MSC", "Detrend"],
        "pick": 2
    })

    # Use preset in configuration
    config = {"transforms": {"_preset_": "spectral_transforms"}}

    # Expand with preset resolution
    expanded = expand_spec(config)  # Automatically resolves presets

Keywords:
    _preset_: Reference to a named preset configuration

Examples:
    # Define preprocessing options
    register_preset("standard_preprocessing", {
        "_or_": [
            {"class": "StandardScaler"},
            {"class": "MinMaxScaler"},
            {"class": "RobustScaler"}
        ]
    })

    # Define model options
    register_preset("regression_models", {
        "_or_": [
            {"class": "PLSRegression", "n_components": {"_range_": [5, 20]}},
            {"class": "RandomForestRegressor", "n_estimators": 100}
        ]
    })

    # Use in pipeline
    pipeline_spec = {
        "preprocessing": {"_preset_": "standard_preprocessing"},
        "model": {"_preset_": "regression_models"}
    }
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Set

# Global preset registry
_PRESET_REGISTRY: Dict[str, Any] = {}

# Keyword for preset reference
PRESET_KEYWORD: str = "_preset_"


def register_preset(
    name: str,
    spec: Any,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    overwrite: bool = False
) -> None:
    """Register a named preset configuration.

    Args:
        name: Unique name for the preset.
        spec: Configuration specification (dict, list, or scalar).
        description: Optional human-readable description.
        tags: Optional list of tags for categorization.
        overwrite: If True, overwrite existing preset with same name.

    Raises:
        ValueError: If preset name already exists and overwrite=False.

    Examples:
        >>> register_preset("my_models", {"_or_": ["PLS", "RF"]})
        >>> register_preset("my_models", {"_or_": ["SVM"]}, overwrite=True)
    """
    if name in _PRESET_REGISTRY and not overwrite:
        raise ValueError(
            f"Preset '{name}' already exists. Use overwrite=True to replace."
        )

    _PRESET_REGISTRY[name] = {
        'spec': deepcopy(spec),
        'description': description,
        'tags': tags or [],
    }


def unregister_preset(name: str) -> bool:
    """Remove a preset from the registry.

    Args:
        name: Name of preset to remove.

    Returns:
        True if preset was removed, False if it didn't exist.
    """
    if name in _PRESET_REGISTRY:
        del _PRESET_REGISTRY[name]
        return True
    return False


def get_preset(name: str) -> Any:
    """Retrieve a preset specification by name.

    Args:
        name: Name of the preset.

    Returns:
        Deep copy of the preset specification.

    Raises:
        KeyError: If preset doesn't exist.
    """
    if name not in _PRESET_REGISTRY:
        raise KeyError(f"Preset '{name}' not found. Available: {list_presets()}")
    return deepcopy(_PRESET_REGISTRY[name]['spec'])


def get_preset_info(name: str) -> Dict[str, Any]:
    """Get full preset info including metadata.

    Args:
        name: Name of the preset.

    Returns:
        Dict with spec, description, tags.

    Raises:
        KeyError: If preset doesn't exist.
    """
    if name not in _PRESET_REGISTRY:
        raise KeyError(f"Preset '{name}' not found.")
    return deepcopy(_PRESET_REGISTRY[name])


def list_presets(tags: Optional[List[str]] = None) -> List[str]:
    """List all registered preset names.

    Args:
        tags: If provided, filter to presets with any of these tags.

    Returns:
        List of preset names.
    """
    if tags is None:
        return list(_PRESET_REGISTRY.keys())

    result = []
    tag_set = set(tags)
    for name, info in _PRESET_REGISTRY.items():
        if tag_set & set(info.get('tags', [])):
            result.append(name)
    return result


def clear_presets() -> int:
    """Clear all registered presets.

    Returns:
        Number of presets cleared.
    """
    count = len(_PRESET_REGISTRY)
    _PRESET_REGISTRY.clear()
    return count


def has_preset(name: str) -> bool:
    """Check if a preset exists.

    Args:
        name: Name to check.

    Returns:
        True if preset exists.
    """
    return name in _PRESET_REGISTRY


def is_preset_reference(node: Any) -> bool:
    """Check if a node is a preset reference.

    Args:
        node: Node to check.

    Returns:
        True if node is a dict with _preset_ key.
    """
    return isinstance(node, dict) and PRESET_KEYWORD in node


def resolve_preset(node: Dict[str, Any]) -> Any:
    """Resolve a single preset reference.

    Args:
        node: Dict containing _preset_ key.

    Returns:
        Resolved preset specification.

    Raises:
        KeyError: If referenced preset doesn't exist.
        ValueError: If _preset_ value is not a string.
    """
    preset_name = node.get(PRESET_KEYWORD)
    if not isinstance(preset_name, str):
        raise ValueError(
            f"_preset_ must be a string, got {type(preset_name).__name__}"
        )
    return get_preset(preset_name)


def resolve_presets_recursive(node: Any, resolved: Optional[Set[str]] = None) -> Any:
    """Recursively resolve all preset references in a configuration.

    Handles circular reference detection.

    Args:
        node: Configuration node (dict, list, or scalar).
        resolved: Set of already-resolved presets (for cycle detection).

    Returns:
        Node with all preset references resolved.

    Raises:
        ValueError: If circular preset reference detected.
    """
    if resolved is None:
        resolved = set()

    # Handle preset reference
    if is_preset_reference(node):
        preset_name = node[PRESET_KEYWORD]
        if preset_name in resolved:
            raise ValueError(
                f"Circular preset reference detected: {preset_name}"
            )

        resolved.add(preset_name)
        preset_spec = resolve_preset(node)

        # Recursively resolve nested presets
        return resolve_presets_recursive(preset_spec, resolved.copy())

    # Handle dict
    if isinstance(node, dict):
        return {
            k: resolve_presets_recursive(v, resolved.copy())
            for k, v in node.items()
        }

    # Handle list
    if isinstance(node, list):
        return [resolve_presets_recursive(item, resolved.copy()) for item in node]

    # Scalar - return as-is
    return node


def export_presets() -> Dict[str, Any]:
    """Export all presets for serialization.

    Returns:
        Dict of all presets with metadata.
    """
    return deepcopy(_PRESET_REGISTRY)


def import_presets(
    presets: Dict[str, Any],
    overwrite: bool = False
) -> int:
    """Import presets from a dict.

    Args:
        presets: Dict mapping preset names to info dicts or specs.
        overwrite: If True, overwrite existing presets.

    Returns:
        Number of presets imported.
    """
    count = 0
    for name, value in presets.items():
        if isinstance(value, dict) and 'spec' in value:
            # Full info dict
            register_preset(
                name,
                value['spec'],
                description=value.get('description'),
                tags=value.get('tags'),
                overwrite=overwrite
            )
        else:
            # Direct spec
            register_preset(name, value, overwrite=overwrite)
        count += 1
    return count


# =============================================================================
# Built-in Presets
# =============================================================================

def register_builtin_presets() -> None:
    """Register built-in preset configurations.

    These are common patterns that users might want to use.
    """
    # Standard scaler options
    register_preset(
        "standard_scalers",
        {
            "_or_": [
                {"class": "sklearn.preprocessing.StandardScaler"},
                {"class": "sklearn.preprocessing.MinMaxScaler"},
                {"class": "sklearn.preprocessing.RobustScaler"},
                None  # No scaling option
            ]
        },
        description="Common sklearn scalers including no-scaling option",
        tags=["preprocessing", "sklearn"],
        overwrite=True
    )

    # Common PLS component ranges
    register_preset(
        "pls_components",
        {"_range_": [2, 20]},
        description="Common range for PLS n_components",
        tags=["hyperparameter", "pls"],
        overwrite=True
    )

    # Learning rate schedules
    register_preset(
        "learning_rates",
        {"_log_range_": [0.0001, 0.1, 10]},
        description="Logarithmic range of learning rates",
        tags=["hyperparameter", "deep_learning"],
        overwrite=True
    )
