"""
Configuration normalizer for dataset configuration.

This module provides the ConfigNormalizer class that combines all parsers
and produces a canonical representation of dataset configurations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from .base import BaseParser, ParserResult
from .legacy_parser import LegacyParser, normalize_config_keys
from .files_parser import FilesParser, SourcesParser, VariationsParser
from .folder_parser import FolderParser
from ..schema import DatasetConfigSchema


class ConfigNormalizer:
    """Normalizes dataset configurations from various input formats.

    This class combines multiple parsers to handle:
    - Folder paths (auto-scanning)
    - JSON/YAML config files
    - Dictionary configurations (legacy format)
    - Sources configurations (multi-source format)
    - Variations configurations (preprocessed data / feature variations)
    - In-memory numpy arrays

    All inputs are normalized to a canonical dictionary format that can be
    validated and processed by the loader.

    Example:
        ```python
        normalizer = ConfigNormalizer()

        # From folder path
        config, name = normalizer.normalize("/path/to/data/")

        # From config file
        config, name = normalizer.normalize("config.yaml")

        # From dictionary
        config, name = normalizer.normalize({"train_x": "data/X.csv"})

        # From sources format
        config, name = normalizer.normalize({
            "sources": [
                {"name": "NIR", "train_x": "NIR_train.csv"},
                {"name": "MIR", "train_x": "MIR_train.csv"}
            ]
        })

        # From variations format
        config, name = normalizer.normalize({
            "variations": [
                {"name": "raw", "train_x": "X_raw.csv"},
                {"name": "snv", "train_x": "X_snv.csv"}
            ],
            "variation_mode": "separate"
        })
        ```
    """

    def __init__(self, parsers: Optional[List[BaseParser]] = None):
        """Initialize the normalizer with parsers.

        Args:
            parsers: Optional list of parsers. If None, uses default parsers.
        """
        if parsers is None:
            # Default parser order - more specific first
            self.parsers = [
                VariationsParser(), # New variations syntax (Phase 7)
                SourcesParser(),    # New sources syntax (Phase 6)
                FilesParser(),      # New files syntax
                FolderParser(),     # Folder auto-scanning
                LegacyParser(),     # Legacy train_x/test_x format
            ]
        else:
            self.parsers = parsers

    def normalize(
        self,
        input_data: Any
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Normalize a configuration to canonical format.

        Args:
            input_data: Configuration in any supported format.

        Returns:
            Tuple of (normalized_config, dataset_name).
            Returns (None, 'Unknown_dataset') if parsing fails.
        """
        # Handle None input
        if input_data is None:
            return None, 'Unknown_dataset'

        # Handle string inputs (file paths)
        if isinstance(input_data, str):
            return self._normalize_string(input_data)

        # Handle Path objects
        if isinstance(input_data, Path):
            return self._normalize_string(str(input_data))

        # Handle dictionary inputs
        if isinstance(input_data, dict):
            return self._normalize_dict(input_data)

        # Unsupported type
        return None, 'Unknown_dataset'

    def _normalize_string(
        self,
        path_str: str
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Normalize a string path input.

        Args:
            path_str: Path to folder or config file.

        Returns:
            Tuple of (config, name).
        """
        lower_path = path_str.lower()

        # Check if it's a JSON/YAML config file
        if lower_path.endswith(('.json', '.yaml', '.yml')):
            return self._load_config_file(path_str)

        # Otherwise, treat as folder path
        parser = FolderParser()
        if parser.can_parse(path_str):
            result = parser.parse(path_str)
            if result.success:
                return result.config, result.dataset_name
            else:
                # Log errors
                for error in result.errors:
                    pass  # Errors are in result, caller handles them
                return None, 'Unknown_dataset'

        return None, 'Unknown_dataset'

    def _normalize_dict(
        self,
        config: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Normalize a dictionary configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            Tuple of (normalized_config, name).
        """
        # Check for 'folder' key first
        if 'folder' in config:
            parser = FolderParser()
            result = parser.parse(config)
            if result.success:
                return result.config, result.dataset_name
            return None, 'Unknown_dataset'

        # Try each parser
        for parser in self.parsers:
            if parser.can_parse(config):
                result = parser.parse(config)
                if result.success:
                    # Handle schema objects - convert to dict
                    parsed_config = result.config
                    dataset_name = result.dataset_name

                    if isinstance(parsed_config, DatasetConfigSchema):
                        # Check if it's a variations format - convert to legacy
                        if parsed_config.is_variations_format():
                            legacy_config = parsed_config.variations_to_legacy_format()
                            return legacy_config, dataset_name
                        # Check if it's a sources format - convert to legacy
                        elif parsed_config.is_sources_format():
                            legacy_config = parsed_config.to_legacy_format()
                            return legacy_config, dataset_name
                        else:
                            # Convert to dict
                            return parsed_config.to_dict(), dataset_name
                    elif isinstance(parsed_config, dict):
                        return parsed_config, dataset_name
                    else:
                        return result.config, dataset_name
                # If parser matched but failed, don't try other parsers
                return None, 'Unknown_dataset'

        # No parser matched - normalize keys and return
        normalized = normalize_config_keys(config)
        name = self._extract_name(normalized)
        return normalized, name

    def _load_config_file(
        self,
        file_path: str
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Load configuration from JSON/YAML file.

        Args:
            file_path: Path to config file.

        Returns:
            Tuple of (config, name).

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the file contains invalid JSON/YAML or is empty.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset configuration file not found: {file_path}\n"
                f"Please check the file path and try again."
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {file_path}\n"
                f"Expected a JSON (.json) or YAML (.yaml, .yml) configuration file."
            )

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                raise ValueError(f"Configuration file is empty: {file_path}")

            # Parse based on extension
            if path.suffix.lower() == '.json':
                config = self._parse_json(content, file_path)
            else:
                config = self._parse_yaml(content, file_path)

            if config is None:
                raise ValueError(
                    f"Configuration file is empty or contains only null: {file_path}"
                )

            if not isinstance(config, dict):
                raise ValueError(
                    f"Configuration file must contain a dictionary/object at the root level.\n"
                    f"Got: {type(config).__name__}\n"
                    f"File: {file_path}"
                )

        except (IOError, OSError) as exc:
            raise ValueError(f"Error reading configuration file {file_path}: {exc}") from exc

        # Normalize keys
        config = normalize_config_keys(config)

        # Extract dataset name
        dataset_name = config.get('name', path.stem)

        return config, dataset_name

    def _parse_json(self, content: str, file_path: str) -> Any:
        """Parse JSON content.

        Args:
            content: JSON string.
            file_path: Path for error messages.

        Returns:
            Parsed JSON data.

        Raises:
            ValueError: If JSON is invalid.
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {file_path}\n"
                f"Error at line {exc.lineno}, column {exc.colno}:\n"
                f"  {exc.msg}\n\n"
                f"Please check your JSON syntax."
            ) from exc

    def _parse_yaml(self, content: str, file_path: str) -> Any:
        """Parse YAML content.

        Args:
            content: YAML string.
            file_path: Path for error messages.

        Returns:
            Parsed YAML data.

        Raises:
            ValueError: If YAML is invalid.
        """
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError as exc:
            if hasattr(exc, 'problem_mark') and exc.problem_mark:
                mark = exc.problem_mark
                line_num = mark.line + 1
                col_num = mark.column + 1
                raise ValueError(
                    f"Invalid YAML in {file_path}\n"
                    f"Error at line {line_num}, column {col_num}:\n"
                    f"  {getattr(exc, 'problem', 'Unknown error')}\n\n"
                    f"Please check your YAML syntax."
                ) from exc
            else:
                raise ValueError(
                    f"Invalid YAML in {file_path}:\n"
                    f"  {exc}\n\n"
                    f"Please check your YAML syntax."
                ) from exc

    def _extract_name(self, config: Dict[str, Any]) -> str:
        """Extract dataset name from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            Dataset name.
        """
        # Check for explicit name
        if 'name' in config:
            return config['name']

        # Try to extract from train_x or test_x path
        for key in ['train_x', 'test_x']:
            path_value = config.get(key)
            if path_value is None:
                continue

            # Handle list (multi-source)
            if isinstance(path_value, list) and len(path_value) > 0:
                path_value = path_value[0]

            # Handle string/Path
            if isinstance(path_value, (str, Path)):
                path = Path(path_value)
                return f"{path.parent.name}_{path.stem}"

        return "array_dataset"


def normalize_config(input_data: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    """Convenience function to normalize a configuration.

    Args:
        input_data: Configuration in any supported format.

    Returns:
        Tuple of (normalized_config, dataset_name).
    """
    normalizer = ConfigNormalizer()
    return normalizer.normalize(input_data)
