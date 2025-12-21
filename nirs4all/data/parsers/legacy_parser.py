"""
Legacy parser for dataset configuration.

This parser handles the current train_x/test_x format that is fully implemented
and widely used. It provides backward compatibility with existing configurations.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseParser, ParserResult


# Key mapping for normalization
# Maps various naming conventions to standard keys
KEY_MAPPINGS = {
    # train_x variations
    'train_x': 'train_x',
    'x_train': 'train_x',
    'xtrain': 'train_x',
    'trainx': 'train_x',

    # train_y variations
    'train_y': 'train_y',
    'y_train': 'train_y',
    'ytrain': 'train_y',
    'trainy': 'train_y',

    # test_x variations (including val)
    'test_x': 'test_x',
    'x_test': 'test_x',
    'xtest': 'test_x',
    'testx': 'test_x',
    'val_x': 'test_x',
    'x_val': 'test_x',
    'xval': 'test_x',
    'valx': 'test_x',

    # test_y variations (including val)
    'test_y': 'test_y',
    'y_test': 'test_y',
    'ytest': 'test_y',
    'testy': 'test_y',
    'val_y': 'test_y',
    'y_val': 'test_y',
    'yval': 'test_y',
    'valy': 'test_y',

    # train_group (metadata) variations
    'train_group': 'train_group',
    'group_train': 'train_group',
    'grouptrain': 'train_group',
    'traingroup': 'train_group',
    'train_metadata': 'train_group',
    'metadata_train': 'train_group',
    'metadatatrain': 'train_group',
    'trainmetadata': 'train_group',
    'train_meta': 'train_group',
    'meta_train': 'train_group',
    'metatrain': 'train_group',
    'trainmeta': 'train_group',
    'train_m': 'train_group',
    'm_train': 'train_group',
    'mtrain': 'train_group',
    'trainm': 'train_group',

    # test_group (metadata) variations
    'test_group': 'test_group',
    'group_test': 'test_group',
    'grouptest': 'test_group',
    'testgroup': 'test_group',
    'test_metadata': 'test_group',
    'metadata_test': 'test_group',
    'metadatatest': 'test_group',
    'testmetadata': 'test_group',
    'test_meta': 'test_group',
    'meta_test': 'test_group',
    'metatest': 'test_group',
    'testmeta': 'test_group',
    'test_m': 'test_group',
    'm_test': 'test_group',
    'mtest': 'test_group',
    'testm': 'test_group',
    'val_group': 'test_group',
    'group_val': 'test_group',
    'groupval': 'test_group',
    'valgroup': 'test_group',
    'val_metadata': 'test_group',
    'metadata_val': 'test_group',
    'metadataval': 'test_group',
    'valmetadata': 'test_group',
    'val_meta': 'test_group',
    'meta_val': 'test_group',
    'metaval': 'test_group',
    'valmeta': 'test_group',
    'val_m': 'test_group',
    'm_val': 'test_group',
    'mval': 'test_group',
    'valm': 'test_group',
}


def normalize_config_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize dataset configuration keys to standard format.

    Maps variations like 'x_train', 'X_train', 'Xtrain' to 'train_x'.
    Maps metadata variations like 'metadata_train', 'train_metadata', 'm_train' to 'train_group'.

    Args:
        config: Original configuration dictionary.

    Returns:
        Normalized configuration with standardized keys.
    """
    normalized = {}

    for key, value in config.items():
        # Try case-insensitive lookup
        normalized_key = KEY_MAPPINGS.get(key.lower(), key)
        normalized[normalized_key] = value

    return normalized


class LegacyParser(BaseParser):
    """Parser for legacy train_x/test_x configuration format.

    This parser handles dictionary configurations using the established
    key format: train_x, train_y, test_x, test_y, train_group, test_group.

    It also handles flexible key naming (X_train, Xtrain, etc.) by normalizing
    to the standard format.
    """

    def can_parse(self, input_data: Any) -> bool:
        """Check if this is a legacy format configuration.

        Args:
            input_data: The input to check.

        Returns:
            True if input is a dict with legacy keys or data arrays.
        """
        if not isinstance(input_data, dict):
            return False

        # Normalize keys for checking
        normalized = normalize_config_keys(input_data)

        # Check for legacy keys
        legacy_keys = ['train_x', 'train_y', 'test_x', 'test_y', 'train_group', 'test_group']
        has_legacy_keys = any(key in normalized for key in legacy_keys)

        # Also handle folder dict format
        has_folder = 'folder' in input_data

        return has_legacy_keys or has_folder

    def parse(self, input_data: Dict[str, Any]) -> ParserResult:
        """Parse a legacy format configuration.

        Args:
            input_data: Dictionary configuration to parse.

        Returns:
            ParserResult with normalized configuration.
        """
        if not isinstance(input_data, dict):
            return ParserResult(
                success=False,
                errors=[f"Expected dict, got {type(input_data).__name__}"],
                source_type="unknown"
            )

        # Normalize keys
        config = normalize_config_keys(input_data)
        warnings = []

        # Extract dataset name
        dataset_name = self._infer_dataset_name(config)

        # Validate required data
        has_train = config.get('train_x') is not None
        has_test = config.get('test_x') is not None

        if not has_train and not has_test:
            return ParserResult(
                success=False,
                errors=["No data source found. Provide train_x or test_x."],
                source_type="dict"
            )

        return ParserResult(
            success=True,
            config=config,
            dataset_name=dataset_name,
            warnings=warnings,
            source_type="dict"
        )

    def _infer_dataset_name(self, config: Dict[str, Any]) -> str:
        """Infer dataset name from configuration.

        Priority:
        1. 'name' key in config
        2. Path from train_x or test_x
        3. Default 'unnamed_dataset'

        Args:
            config: Normalized configuration dictionary.

        Returns:
            Inferred dataset name.
        """
        # Check for explicit name
        if 'name' in config:
            return config['name']

        # Try to extract from file path
        for key in ['train_x', 'test_x']:
            path_value = config.get(key)
            if path_value is None:
                continue

            # Handle list (multi-source) - use first path
            if isinstance(path_value, list):
                if len(path_value) > 0:
                    path_value = path_value[0]
                else:
                    continue

            # Handle string paths
            if isinstance(path_value, str):
                path = Path(path_value)
                return f"{path.parent.name}_{path.stem}"

            # Handle Path objects
            if isinstance(path_value, Path):
                return f"{path_value.parent.name}_{path_value.stem}"

        return "array_dataset"
