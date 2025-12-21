"""
Dataset configuration parser.

This module provides functions for parsing dataset configurations from various formats.
It serves as the main entry point for configuration parsing, delegating to specialized
parsers in the nirs4all.data.parsers module.

The parser supports:
- Folder paths with auto-scanning for data files
- JSON/YAML configuration files
- Dictionary configurations (legacy train_x/test_x format)
- In-memory numpy arrays

For the new schema-based validation, see nirs4all.data.schema.
For specialized parsers, see nirs4all.data.parsers.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import yaml

from nirs4all.core.logging import get_logger

# Import from new parser module for internal use
from nirs4all.data.parsers.legacy_parser import normalize_config_keys as _normalize_keys
from nirs4all.data.parsers.normalizer import ConfigNormalizer

logger = get_logger(__name__)

# Create shared normalizer instance
_normalizer = ConfigNormalizer()


def _load_config_from_file(file_path: str) -> Tuple[Dict[str, Any], str]:
    """Load dataset config from JSON/YAML file.

    Args:
        file_path: Path to a JSON or YAML configuration file.

    Returns:
        Tuple of (config_dict, dataset_name).

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file contains invalid JSON/YAML or is empty.
    """
    return _normalizer._load_config_file(file_path)


def normalize_config_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize dataset configuration keys to standard format.

    Maps variations like 'x_train', 'X_train', 'Xtrain' to 'train_x'
    Maps metadata variations like 'metadata_train', 'train_metadata', 'm_train' to 'train_group'

    Args:
        config: Original configuration dictionary

    Returns:
        Normalized configuration with standardized keys
    """
    return _normalize_keys(config)

def _s_(path):
    """Convert path(s) to POSIX format. Handles both single paths and lists of paths."""
    if path is None:
        return None
    if isinstance(path, list):
        return [Path(p).as_posix() for p in path]
    return Path(path).as_posix()


def browse_folder(folder_path, global_params=None):
    """Scan a folder for data files matching standard naming conventions.

    This function delegates to FolderParser for the actual scanning.

    Args:
        folder_path: Path to folder to scan.
        global_params: Optional global loading parameters.

    Returns:
        Configuration dictionary with detected file paths.
    """
    from nirs4all.data.parsers.folder_parser import FolderParser

    parser = FolderParser()

    # Create input for parser
    if global_params is not None:
        input_data = {"folder": folder_path, "global_params": global_params}
    else:
        input_data = folder_path

    result = parser.parse(input_data)

    if result.success:
        return result.config

    # Return empty config on failure (backward compatible behavior)
    logger.error(f"Failed to browse folder {folder_path}: {result.errors}")
    return {
        "train_x": None, "train_x_filter": None, "train_x_params": None,
        "train_y": None, "train_y_filter": None, "train_y_params": None,
        "train_group": None, "train_group_filter": None, "train_group_params": None,
        "train_params": None,
        "test_x": None, "test_x_filter": None, "test_x_params": None,
        "test_y": None, "test_y_filter": None, "test_y_params": None,
        "test_group": None, "test_group_filter": None, "test_group_params": None,
        "test_params": None,
        "global_params": global_params
    }


def folder_to_name(folder_path):
    """Extract a dataset name from a folder path.

    Args:
        folder_path: Path to folder.

    Returns:
        Cleaned dataset name.
    """
    path = Path(folder_path)
    for part in reversed(path.parts):
        clean_part = ''.join(c if c.isalnum() else '_' for c in part)
        if clean_part:
            return clean_part.lower()
    return "Unknown_dataset"


def parse_config(data_config):
    """Parse a dataset configuration.

    Handles multiple input formats:
    - String path to a folder: auto-browse for data files
    - String path to JSON/YAML file (.json, .yaml, .yml): load config from file
    - Dict with 'folder' key: browse folder with optional params
    - Dict with data keys (train_x, test_x, etc.): use directly

    Args:
        data_config: Dataset configuration in any supported format.

    Returns:
        Tuple of (parsed_config_dict, dataset_name).
        Returns (None, 'Unknown_dataset') if parsing fails.
    """
    # Use the new normalizer for unified handling
    return _normalizer.normalize(data_config)




