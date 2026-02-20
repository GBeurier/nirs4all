"""
Configuration normalizer for dataset configuration.

This module provides the ConfigNormalizer class that combines all parsers
and produces a canonical representation of dataset configurations.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from ..schema import DatasetConfigSchema
from .base import BaseParser, ParserResult
from .files_parser import FilesParser, SourcesParser, VariationsParser
from .folder_parser import FolderParser


class ConfigNormalizer:
    """Normalizes dataset configurations from various input formats.

    This class combines multiple parsers to handle:
    - Folder paths (auto-scanning)
    - JSON/YAML config files
    - Dictionary configurations (canonical keys + aliases)
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

    def __init__(self, parsers: list[BaseParser] | None = None):
        """Initialize the normalizer with parsers.

        Args:
            parsers: Optional list of parsers. If None, uses default parsers.
        """
        if parsers is None:
            # Default parser order - more specific first
            self.parsers = [
                VariationsParser(), # Variations syntax (Phase 7)
                SourcesParser(),    # Sources syntax (Phase 6)
                FilesParser(),      # Files syntax
                FolderParser(),     # Folder auto-scanning
            ]
        else:
            self.parsers = parsers

    @staticmethod
    def _normalize_key(key: Any) -> str:
        """Normalize a config key for case/separator-insensitive matching."""
        return "".join(ch.lower() for ch in str(key) if ch.isalnum())

    @staticmethod
    @lru_cache(maxsize=1)
    def _key_alias_map() -> dict[str, str]:
        """Build mapping of accepted key aliases to canonical config keys."""

        def _combine_partition_role(partitions: list[str], roles: list[str]) -> list[str]:
            aliases: list[str] = []
            for partition in partitions:
                for role in roles:
                    aliases.extend([
                        f"{partition}_{role}",
                        f"{role}_{partition}",
                        f"{partition}{role}",
                        f"{role}{partition}",
                    ])
            return aliases

        train_partitions = ["train", "trn", "cal", "calibration", "fit"]
        test_partitions = ["test", "tst", "val", "validation", "eval", "holdout", "predict", "inference"]

        feature_roles = ["x", "feature", "features", "spectrum", "spectra", "signal", "signals"]
        target_roles = ["y", "target", "targets", "label", "labels", "response", "responses"]
        metadata_roles = ["group", "groups", "meta", "metadata", "m", "samplemeta", "samplemetadata"]

        # Base aliases for core train/test data keys.
        aliases_by_key: dict[str, list[str]] = {
            "train_x": _combine_partition_role(train_partitions, feature_roles),
            "test_x": _combine_partition_role(test_partitions, feature_roles),
            "train_y": _combine_partition_role(train_partitions, target_roles),
            "test_y": _combine_partition_role(test_partitions, target_roles),
            "train_group": _combine_partition_role(train_partitions, metadata_roles),
            "test_group": _combine_partition_role(test_partitions, metadata_roles),
            "name": ["dataset_name", "data_name"],
            "description": ["desc", "dataset_description", "data_description"],
            "task_type": ["task", "tasktype", "problem_type", "problemtype", "ml_task", "target_type"],
            "folder": ["dir", "directory", "data_folder", "data_dir", "folder_path", "path"],
            "global_params": [
                "global", "global_config", "global_settings", "global_options",
                "loading_params", "loader_params", "read_params", "io_params",
                "global_loading_params", "global_loading_options", "shared_params",
            ],
            "train_params": _combine_partition_role(train_partitions, ["params", "param", "config", "settings", "options"]),
            "test_params": _combine_partition_role(test_partitions, ["params", "param", "config", "settings", "options"]),
            "aggregate": [
                "aggregation", "aggregate_by", "aggregation_by",
                "group_by", "groupby", "aggregate_column", "aggregation_column",
            ],
            "aggregate_method": [
                "aggregation_method", "aggregate_mode", "aggregation_mode",
                "aggregate_strategy", "aggregation_strategy",
            ],
            "aggregate_exclude_outliers": [
                "exclude_outliers", "remove_outliers", "drop_outliers",
                "exclude_outliers_before_aggregation",
                "remove_outliers_before_aggregation",
                "aggregation_exclude_outliers",
            ],
            "repetition": [
                "repetitions", "repeat", "repeat_by", "repetition_column", "repeat_column",
                "sample_id_column", "sampleidcolumn", "group_column",
            ],
            "folds": ["fold", "cv", "cv_folds", "cross_validation", "cross_validation_folds", "splits", "cv_splits"],
            "files": ["file_list", "data_files", "input_files"],
            "sources": ["source_list", "sensor_sources", "input_sources", "feature_sources", "modalities"],
            # Keep parser-facing keys as canonical (targets/metadata) so SourcesParser can consume them.
            "targets": ["target_spec", "targets_spec", "shared_targets", "shared_target", "common_targets"],
            "metadata": ["metadata_spec", "meta_spec", "shared_metadata", "shared_meta", "common_metadata"],
            "variations": ["feature_variations", "data_variations", "preprocessing_variations"],
            "variation_mode": ["variation_strategy", "variation_type", "variation_method"],
            "variation_select": ["selected_variations", "variation_selection", "choose_variations", "chosen_variations"],
            "variation_prefix": ["prefix_variations", "add_variation_prefix", "variation_name_prefix"],
        }

        # Generate aliases for *_params and *_filter variants of train/test x/y/group keys.
        suffix_aliases = {
            "_params": ["params", "param", "config", "configs", "settings", "options"],
            "_filter": ["filter", "filters", "columns", "cols", "indices", "idx", "select", "selection"],
        }
        partitioned_bases = ["train_x", "test_x", "train_y", "test_y", "train_group", "test_group"]

        for base_key in partitioned_bases:
            base_aliases = list(aliases_by_key.get(base_key, []))
            for suffix, terms in suffix_aliases.items():
                canonical_key = f"{base_key}{suffix}"
                derived_aliases: list[str] = []
                for base_alias in base_aliases:
                    for term in terms:
                        derived_aliases.extend([
                            f"{base_alias}_{term}",
                            f"{base_alias}{term}",
                            f"{term}_{base_alias}",
                            f"{term}{base_alias}",
                        ])
                aliases_by_key[canonical_key] = derived_aliases

        alias_map: dict[str, str] = {}
        for canonical_key, aliases in aliases_by_key.items():
            for alias in [canonical_key, *aliases]:
                normalized_alias = ConfigNormalizer._normalize_key(alias)
                # First mapping wins to avoid accidental override by later broad aliases.
                if normalized_alias not in alias_map:
                    alias_map[normalized_alias] = canonical_key
        return alias_map

    @classmethod
    def _apply_key_aliases(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Normalize accepted key aliases to canonical config keys."""
        if not isinstance(config, dict):
            return config

        alias_map = cls._key_alias_map()
        normalized = dict(config)

        for key, value in list(config.items()):
            mapped_key = alias_map.get(cls._normalize_key(key))
            if mapped_key is None or key == mapped_key:
                continue

            # Canonical key wins when both are present.
            if mapped_key not in normalized:
                normalized[mapped_key] = value
            normalized.pop(key, None)

        return normalized

    def normalize(
        self,
        input_data: Any
    ) -> tuple[dict[str, Any] | None, str]:
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
    ) -> tuple[dict[str, Any] | None, str]:
        """Normalize a string path input.

        Args:
            path_str: Path to folder or config file.

        Returns:
            Tuple of (config, name).
        """
        lower_path = path_str.lower()

        # Check if it's a JSON/YAML config file
        if lower_path.endswith(('.json', '.yaml', '.yml')):
            config, name = self._load_config_file(path_str)
            if config is None:
                return None, name
            return self._apply_key_aliases(config), name

        # Otherwise, treat as folder path
        parser = FolderParser()
        if parser.can_parse(path_str):
            result = parser.parse(path_str)
            if result.success:
                return result.config, result.dataset_name or 'Unknown_dataset'
            else:
                # Log errors
                for _ in result.errors:
                    pass  # Errors are in result, caller handles them
                return None, 'Unknown_dataset'

        return None, 'Unknown_dataset'

    def _normalize_dict(
        self,
        config: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str]:
        """Normalize a dictionary configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            Tuple of (normalized_config, name).
        """
        config = self._apply_key_aliases(config)

        # Check for 'folder' key first
        if 'folder' in config:
            folder_parser = FolderParser()
            result = folder_parser.parse(config)
            if result.success:
                return result.config, result.dataset_name or 'Unknown_dataset'
            return None, 'Unknown_dataset'

        # Try each parser
        for parser in self.parsers:
            if parser.can_parse(config):
                result = parser.parse(config)
                if result.success:
                    # Handle schema objects - convert to dict
                    parsed_config = result.config
                    dataset_name = result.dataset_name or 'Unknown_dataset'

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
                        # Parsers may return model_dump() dicts; reconstruct
                        # schema to convert sources/variations to legacy format.
                        if "sources" in parsed_config or "variations" in parsed_config:
                            schema = DatasetConfigSchema(**parsed_config)
                            if schema.is_variations_format():
                                return schema.variations_to_legacy_format(), dataset_name
                            elif schema.is_sources_format():
                                return schema.to_legacy_format(), dataset_name
                        return parsed_config, dataset_name
                    else:
                        return result.config, dataset_name
                # If parser matched but failed, don't try other parsers
                return None, 'Unknown_dataset'

        # No parser matched - return dict as-is with name extracted
        name = self._extract_name(config)
        return config, name

    def _load_config_file(
        self,
        file_path: str
    ) -> tuple[dict[str, Any] | None, str]:
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
            with open(path, encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                raise ValueError(f"Configuration file is empty: {file_path}")

            # Parse based on extension
            config = self._parse_json(content, file_path) if path.suffix.lower() == '.json' else self._parse_yaml(content, file_path)

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

        except OSError as exc:
            raise ValueError(f"Error reading configuration file {file_path}: {exc}") from exc

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

    def _extract_name(self, config: dict[str, Any]) -> str:
        """Extract dataset name from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            Dataset name.
        """
        # Check for explicit name
        if 'name' in config:
            return str(config['name'])

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

def normalize_config(input_data: Any) -> tuple[dict[str, Any] | None, str]:
    """Convenience function to normalize a configuration.

    Args:
        input_data: Configuration in any supported format.

    Returns:
        Tuple of (normalized_config, dataset_name).
    """
    normalizer = ConfigNormalizer()
    return normalizer.normalize(input_data)
