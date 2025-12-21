"""
Configuration serializer for dataset configurations.

This module provides serialization/deserialization of dataset configurations
to/from YAML and JSON formats, with normalization and diffing support.

Phase 8 Implementation - Dataset Configuration Roadmap
Section 8.3: Configuration Serialization
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from copy import deepcopy

import yaml
import numpy as np

from nirs4all.core.logging import get_logger
from nirs4all.data.schema import DatasetConfigSchema

logger = get_logger(__name__)


class SerializationFormat(str, Enum):
    """Supported serialization formats."""
    YAML = "yaml"
    JSON = "json"


@dataclass
class ConfigDiff:
    """Result of comparing two configurations.

    Attributes:
        added: Keys added in the new config.
        removed: Keys removed from the old config.
        changed: Keys with different values, with (old, new) tuples.
        unchanged: Keys with identical values.
    """

    added: Dict[str, Any]
    removed: Dict[str, Any]
    changed: Dict[str, Tuple[Any, Any]]
    unchanged: Set[str]

    def is_identical(self) -> bool:
        """Check if configs are identical."""
        return not self.added and not self.removed and not self.changed

    def summary(self) -> str:
        """Get a summary of changes."""
        lines = []
        if self.added:
            lines.append(f"Added: {list(self.added.keys())}")
        if self.removed:
            lines.append(f"Removed: {list(self.removed.keys())}")
        if self.changed:
            lines.append(f"Changed: {list(self.changed.keys())}")
        if not lines:
            return "No changes"
        return "; ".join(lines)

    def __str__(self) -> str:
        lines = []
        lines.append("Configuration Diff:")

        if self.added:
            lines.append("\n  Added:")
            for key, value in self.added.items():
                lines.append(f"    + {key}: {_format_value(value)}")

        if self.removed:
            lines.append("\n  Removed:")
            for key, value in self.removed.items():
                lines.append(f"    - {key}: {_format_value(value)}")

        if self.changed:
            lines.append("\n  Changed:")
            for key, (old_val, new_val) in self.changed.items():
                lines.append(f"    ~ {key}:")
                lines.append(f"        old: {_format_value(old_val)}")
                lines.append(f"        new: {_format_value(new_val)}")

        if self.is_identical():
            lines.append("  (no differences)")

        return "\n".join(lines)


def _format_value(value: Any, max_length: int = 100) -> str:
    """Format a value for display, truncating if necessary."""
    formatted = repr(value)
    if len(formatted) > max_length:
        return formatted[:max_length - 3] + "..."
    return formatted


class ConfigSerializer:
    """Serializer for dataset configurations.

    Handles serialization to YAML/JSON with:
    - Normalization of configs before serialization
    - Conversion of numpy arrays to lists
    - Conversion of Path objects to strings
    - Enum value serialization
    - Removal of internal/private keys

    Example:
        ```python
        serializer = ConfigSerializer()

        # Serialize to YAML
        yaml_str = serializer.to_yaml(config_dict)

        # Serialize to JSON
        json_str = serializer.to_json(config_dict)

        # Save to file
        serializer.save(config_dict, "config.yaml")

        # Load from file
        config = serializer.load("config.yaml")

        # Compare configs
        diff = serializer.diff(old_config, new_config)
        ```
    """

    # Keys to exclude from serialization (internal use only)
    INTERNAL_KEYS = {
        "_sources",
        "_variations",
        "_variation_mode",
        "_parsed",
        "_normalized",
        "_original",
    }

    def __init__(
        self,
        include_defaults: bool = False,
        normalize: bool = True,
        sort_keys: bool = True,
    ):
        """Initialize serializer.

        Args:
            include_defaults: Whether to include default values.
            normalize: Whether to normalize configs before serialization.
            sort_keys: Whether to sort keys in output.
        """
        self.include_defaults = include_defaults
        self.normalize = normalize
        self.sort_keys = sort_keys

    def to_yaml(
        self,
        config: Union[Dict[str, Any], DatasetConfigSchema],
        **kwargs,
    ) -> str:
        """Serialize config to YAML string.

        Args:
            config: Configuration dict or schema object.
            **kwargs: Additional arguments for yaml.dump.

        Returns:
            YAML string.
        """
        prepared = self._prepare_for_serialization(config)

        yaml_kwargs = {
            "default_flow_style": False,
            "sort_keys": self.sort_keys,
            "allow_unicode": True,
        }
        yaml_kwargs.update(kwargs)

        return yaml.dump(prepared, **yaml_kwargs)

    def to_json(
        self,
        config: Union[Dict[str, Any], DatasetConfigSchema],
        indent: int = 2,
        **kwargs,
    ) -> str:
        """Serialize config to JSON string.

        Args:
            config: Configuration dict or schema object.
            indent: Indentation level.
            **kwargs: Additional arguments for json.dumps.

        Returns:
            JSON string.
        """
        prepared = self._prepare_for_serialization(config)

        json_kwargs = {
            "indent": indent,
            "sort_keys": self.sort_keys,
            "ensure_ascii": False,
        }
        json_kwargs.update(kwargs)

        return json.dumps(prepared, **json_kwargs)

    def save(
        self,
        config: Union[Dict[str, Any], DatasetConfigSchema],
        path: Union[str, Path],
        format: Optional[SerializationFormat] = None,
    ) -> None:
        """Save config to file.

        Args:
            config: Configuration to save.
            path: Output file path.
            format: Output format (auto-detected from extension if None).
        """
        path = Path(path)

        if format is None:
            format = self._detect_format(path)

        if format == SerializationFormat.YAML:
            content = self.to_yaml(config)
        else:
            content = self.to_json(config)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.debug(f"Saved config to {path}")

    def load(
        self,
        path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Load config from file.

        Args:
            path: Path to config file.

        Returns:
            Configuration dictionary.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is invalid.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        format = self._detect_format(path)

        if format == SerializationFormat.YAML:
            config = yaml.safe_load(content)
        else:
            config = json.loads(content)

        if config is None:
            raise ValueError(f"Empty config file: {path}")

        if not isinstance(config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(config).__name__}")

        return config

    def diff(
        self,
        old_config: Union[Dict[str, Any], DatasetConfigSchema],
        new_config: Union[Dict[str, Any], DatasetConfigSchema],
    ) -> ConfigDiff:
        """Compare two configurations.

        Args:
            old_config: Original configuration.
            new_config: New configuration.

        Returns:
            ConfigDiff with differences.
        """
        old_dict = self._to_comparable_dict(old_config)
        new_dict = self._to_comparable_dict(new_config)

        old_keys = set(old_dict.keys())
        new_keys = set(new_dict.keys())

        added = {k: new_dict[k] for k in new_keys - old_keys}
        removed = {k: old_dict[k] for k in old_keys - new_keys}

        common_keys = old_keys & new_keys
        changed = {}
        unchanged = set()

        for key in common_keys:
            old_val = old_dict[key]
            new_val = new_dict[key]

            if self._values_equal(old_val, new_val):
                unchanged.add(key)
            else:
                changed[key] = (old_val, new_val)

        return ConfigDiff(
            added=added,
            removed=removed,
            changed=changed,
            unchanged=unchanged,
        )

    def _prepare_for_serialization(
        self,
        config: Union[Dict[str, Any], DatasetConfigSchema]
    ) -> Dict[str, Any]:
        """Prepare config for serialization.

        Args:
            config: Configuration to prepare.

        Returns:
            Serializable dictionary.
        """
        if isinstance(config, DatasetConfigSchema):
            config = config.model_dump(exclude_none=True)

        # Deep copy to avoid modifying original
        result = deepcopy(config)

        # Remove internal keys
        for key in self.INTERNAL_KEYS:
            result.pop(key, None)

        # Convert special types
        result = self._convert_types(result)

        # Normalize if requested
        if self.normalize:
            result = self._normalize_config(result)

        return result

    def _convert_types(self, obj: Any) -> Any:
        """Recursively convert special types for serialization.

        Args:
            obj: Object to convert.

        Returns:
            Converted object.
        """
        if obj is None:
            return None

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, Enum):
            return obj.value

        if isinstance(obj, dict):
            return {k: self._convert_types(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self._convert_types(item) for item in obj]

        return obj

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize configuration for consistent serialization.

        Args:
            config: Configuration dictionary.

        Returns:
            Normalized configuration.
        """
        result = {}

        # Ordered key groups for consistent output
        key_order = [
            # Identification
            "name", "description",
            # Task configuration
            "task_type", "signal_type",
            # Data paths
            "train_x", "train_y", "train_group",
            "test_x", "test_y", "test_group",
            # Parameters
            "global_params", "train_params", "test_params",
            "train_x_params", "train_y_params", "train_group_params",
            "test_x_params", "test_y_params", "test_group_params",
            # Filters
            "train_x_filter", "train_y_filter", "train_group_filter",
            "test_x_filter", "test_y_filter", "test_group_filter",
            # Aggregation
            "aggregate", "aggregate_method", "aggregate_exclude_outliers",
            # New formats
            "files", "sources", "variations", "variation_mode", "variation_select",
            # Folds
            "folds",
        ]

        # Add keys in order
        for key in key_order:
            if key in config:
                result[key] = config[key]

        # Add remaining keys
        for key in config:
            if key not in result:
                result[key] = config[key]

        return result

    def _to_comparable_dict(
        self,
        config: Union[Dict[str, Any], DatasetConfigSchema]
    ) -> Dict[str, Any]:
        """Convert config to comparable dictionary.

        Args:
            config: Configuration.

        Returns:
            Dictionary suitable for comparison.
        """
        if isinstance(config, DatasetConfigSchema):
            return config.model_dump(exclude_none=True)
        return deepcopy(config)

    def _values_equal(self, val1: Any, val2: Any) -> bool:
        """Compare two values for equality.

        Handles numpy arrays, lists, and nested structures.

        Args:
            val1: First value.
            val2: Second value.

        Returns:
            True if values are equal.
        """
        # Handle numpy arrays
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            return np.array_equal(val1, val2)

        if isinstance(val1, np.ndarray):
            val1 = val1.tolist()
        if isinstance(val2, np.ndarray):
            val2 = val2.tolist()

        # Handle dicts
        if isinstance(val1, dict) and isinstance(val2, dict):
            if set(val1.keys()) != set(val2.keys()):
                return False
            return all(
                self._values_equal(val1[k], val2[k])
                for k in val1
            )

        # Handle lists
        if isinstance(val1, list) and isinstance(val2, list):
            if len(val1) != len(val2):
                return False
            return all(
                self._values_equal(v1, v2)
                for v1, v2 in zip(val1, val2)
            )

        # Handle enums
        if isinstance(val1, Enum):
            val1 = val1.value
        if isinstance(val2, Enum):
            val2 = val2.value

        return val1 == val2

    def _detect_format(self, path: Path) -> SerializationFormat:
        """Detect format from file extension.

        Args:
            path: File path.

        Returns:
            Serialization format.
        """
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return SerializationFormat.YAML
        return SerializationFormat.JSON


def serialize_config(
    config: Union[Dict[str, Any], DatasetConfigSchema],
    format: SerializationFormat = SerializationFormat.YAML,
    **kwargs,
) -> str:
    """Convenience function to serialize config.

    Args:
        config: Configuration to serialize.
        format: Output format.
        **kwargs: Additional serializer options.

    Returns:
        Serialized string.
    """
    serializer = ConfigSerializer(**kwargs)
    if format == SerializationFormat.YAML:
        return serializer.to_yaml(config)
    return serializer.to_json(config)


def deserialize_config(
    content: str,
    format: SerializationFormat = SerializationFormat.YAML,
) -> Dict[str, Any]:
    """Convenience function to deserialize config.

    Args:
        content: Serialized content.
        format: Content format.

    Returns:
        Configuration dictionary.
    """
    if format == SerializationFormat.YAML:
        config = yaml.safe_load(content)
    else:
        config = json.loads(content)

    if config is None:
        return {}
    return config


def diff_configs(
    old_config: Union[Dict[str, Any], DatasetConfigSchema],
    new_config: Union[Dict[str, Any], DatasetConfigSchema],
) -> ConfigDiff:
    """Convenience function to diff configs.

    Args:
        old_config: Original configuration.
        new_config: New configuration.

    Returns:
        ConfigDiff with differences.
    """
    serializer = ConfigSerializer()
    return serializer.diff(old_config, new_config)
