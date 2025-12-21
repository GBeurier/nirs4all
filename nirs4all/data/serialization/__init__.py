"""
Serialization module for dataset configuration.

This module provides functionality for serializing and deserializing
dataset configurations to/from YAML and JSON formats.
"""

from .serializer import (
    ConfigSerializer,
    serialize_config,
    deserialize_config,
    diff_configs,
)

__all__ = [
    "ConfigSerializer",
    "serialize_config",
    "deserialize_config",
    "diff_configs",
]
