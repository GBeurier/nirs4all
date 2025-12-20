"""
Configuration validation with JSON Schema for nirs4all.

Provides validation functions for pipeline and dataset configuration files,
with detailed error messages including line numbers and suggestions.
"""

import json
from pathlib import Path
from typing import Tuple, List, Any, Dict, Optional, Union
import yaml

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails.

    Attributes:
        errors: List of validation error messages.
        config_path: Path to the configuration file (if any).
    """

    def __init__(self, errors: List[str], config_path: Optional[str] = None):
        self.errors = errors
        self.config_path = config_path
        message = f"Configuration validation failed"
        if config_path:
            message += f" for {config_path}"
        message += f": {'; '.join(errors)}"
        super().__init__(message)


# =============================================================================
# JSON Schemas for configuration validation
# =============================================================================

# Schema for component specification (sklearn, TF, PyTorch, JAX objects)
COMPONENT_SCHEMA = {
    "type": "object",
    "properties": {
        "class": {"type": "string", "description": "Full module path to the class"},
        "params": {"type": "object", "description": "Constructor parameters"},
        "function": {"type": "string", "description": "Full module path to a function"},
    },
    "anyOf": [
        {"required": ["class"]},
        {"required": ["function"]},
    ]
}

# Schema for a pipeline step
STEP_SCHEMA = {
    "oneOf": [
        # Simple component: {"class": "...", "params": {...}}
        COMPONENT_SCHEMA,
        # Model step: {"model": {...}}
        {
            "type": "object",
            "properties": {
                "model": {"$ref": "#/$defs/component"},
                "name": {"type": "string"},
                "finetune_params": {"type": "object"},
            },
            "required": ["model"]
        },
        # Preprocessing step: {"preprocessing": {...}}
        {
            "type": "object",
            "properties": {
                "preprocessing": {"$ref": "#/$defs/component"},
            },
            "required": ["preprocessing"]
        },
        # Y processing step: {"y_processing": {...}}
        {
            "type": "object",
            "properties": {
                "y_processing": {"$ref": "#/$defs/component"},
            },
            "required": ["y_processing"]
        },
        # Feature augmentation: {"feature_augmentation": {...}}
        {
            "type": "object",
            "properties": {
                "feature_augmentation": {"type": "object"},
            },
            "required": ["feature_augmentation"]
        },
        # Generator syntax: {"_or_": [...], ...}
        {
            "type": "object",
            "properties": {
                "_or_": {"type": "array"},
                "count": {"type": "integer", "minimum": 1},
            },
            "required": ["_or_"]
        },
        # Range generator: {"_range_": [...], ...}
        {
            "type": "object",
            "properties": {
                "_range_": {"type": "array", "minItems": 2, "maxItems": 3},
                "param": {"type": "string"},
            },
            "required": ["_range_"]
        },
        # Branch step: {"branch": {...}}
        {
            "type": "object",
            "properties": {
                "branch": {"type": "object"},
            },
            "required": ["branch"]
        },
    ]
}

# Schema for pipeline configuration file
PIPELINE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "nirs4all Pipeline Configuration",
    "description": "Schema for nirs4all pipeline configuration files",
    "type": "object",
    "properties": {
        "pipeline": {
            "type": "array",
            "description": "List of pipeline steps",
            "items": {
                "type": "object",
                "description": "A pipeline step (preprocessing, model, splitter, etc.)"
            },
            "minItems": 1
        },
        "name": {
            "type": "string",
            "description": "Pipeline name"
        },
        "description": {
            "type": "string",
            "description": "Pipeline description"
        }
    },
    "required": ["pipeline"]
}

# Schema for dataset configuration file
DATASET_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "nirs4all Dataset Configuration",
    "description": "Schema for nirs4all dataset configuration files",
    "type": "object",
    "properties": {
        # Data file paths
        "train_x": {
            "oneOf": [
                {"type": "string", "description": "Path to training features file"},
                {"type": "array", "items": {"type": "string"}, "description": "Paths to multiple training feature files (multi-source)"}
            ]
        },
        "train_y": {
            "oneOf": [
                {"type": "string", "description": "Path to training targets file"},
                {"type": "array", "items": {"type": "string"}}
            ]
        },
        "test_x": {
            "oneOf": [
                {"type": "string", "description": "Path to test/validation features file"},
                {"type": "array", "items": {"type": "string"}}
            ]
        },
        "test_y": {
            "oneOf": [
                {"type": "string", "description": "Path to test/validation targets file"},
                {"type": "array", "items": {"type": "string"}}
            ]
        },
        "train_group": {
            "oneOf": [
                {"type": "string", "description": "Path to training metadata file"},
                {"type": "array", "items": {"type": "string"}}
            ]
        },
        "test_group": {
            "oneOf": [
                {"type": "string", "description": "Path to test/validation metadata file"},
                {"type": "array", "items": {"type": "string"}}
            ]
        },

        # Dataset metadata
        "name": {
            "type": "string",
            "description": "Dataset name"
        },
        "task_type": {
            "type": "string",
            "enum": ["regression", "binary_classification", "multiclass_classification", "auto"],
            "description": "Type of ML task"
        },
        "signal_type": {
            "type": "string",
            "enum": ["absorbance", "reflectance", "reflectance%", "transmittance", "transmittance%", "auto"],
            "description": "Type of spectral signal"
        },

        # Aggregation settings
        "aggregate": {
            "oneOf": [
                {"type": "string", "description": "Metadata column name for aggregation"},
                {"type": "boolean", "description": "True to aggregate by y values"}
            ]
        },
        "aggregate_method": {
            "type": "string",
            "enum": ["mean", "median", "vote"],
            "description": "Method for aggregating predictions"
        },
        "aggregate_exclude_outliers": {
            "type": "boolean",
            "description": "Whether to exclude outliers before aggregation"
        },

        # Loading parameters
        "global_params": {
            "type": "object",
            "properties": {
                "header_unit": {
                    "type": "string",
                    "enum": ["cm-1", "nm", "none", "text", "index"],
                    "description": "Unit of wavelength headers"
                },
                "signal_type": {"type": "string"},
                "delimiter": {"type": "string"},
                "decimal_separator": {"type": "string"},
                "has_header": {"type": "boolean"},
                "na_policy": {"type": "string", "enum": ["drop", "fill", "error"]}
            }
        },
        "train_x_params": {"type": "object", "description": "Override parameters for train_x loading"},
        "train_y_params": {"type": "object", "description": "Override parameters for train_y loading"},
        "test_x_params": {"type": "object", "description": "Override parameters for test_x loading"},
        "test_y_params": {"type": "object", "description": "Override parameters for test_y loading"},
        "train_group_params": {"type": "object", "description": "Override parameters for train_group loading"},
        "test_group_params": {"type": "object", "description": "Override parameters for test_group loading"},

        # Filters
        "train_x_filter": {"type": "string", "description": "Filter expression for train_x"},
        "train_y_filter": {"type": "string", "description": "Filter expression for train_y"},
        "test_x_filter": {"type": "string", "description": "Filter expression for test_x"},
        "test_y_filter": {"type": "string", "description": "Filter expression for test_y"},
    },
    "anyOf": [
        {"required": ["train_x"]},
        {"required": ["test_x"]},
        {"required": ["folder"]}
    ]
}


# =============================================================================
# Validation Functions
# =============================================================================

def _load_config_content(config_path: str) -> Tuple[Dict[str, Any], str]:
    """Load and parse a configuration file.

    Args:
        config_path: Path to JSON or YAML configuration file.

    Returns:
        Tuple of (parsed_config, file_format).

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file is invalid JSON/YAML.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {config_path}")

    suffix = path.suffix.lower()
    if suffix not in ('.json', '.yaml', '.yml'):
        raise ValueError(
            f"Unsupported file format: {suffix}\n"
            f"Expected .json, .yaml, or .yml"
        )

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            raise ValueError(f"Configuration file is empty: {config_path}")

        if suffix == '.json':
            try:
                config = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {config_path}\n"
                    f"Error at line {exc.lineno}, column {exc.colno}:\n"
                    f"  {exc.msg}"
                ) from exc
        else:
            try:
                config = yaml.safe_load(content)
            except yaml.YAMLError as exc:
                if hasattr(exc, 'problem_mark') and exc.problem_mark:
                    mark = exc.problem_mark
                    raise ValueError(
                        f"Invalid YAML in {config_path}\n"
                        f"Error at line {mark.line + 1}, column {mark.column + 1}:\n"
                        f"  {getattr(exc, 'problem', 'Unknown error')}"
                    ) from exc
                else:
                    raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc

        if config is None:
            raise ValueError(f"Configuration file is empty or null: {config_path}")

        if not isinstance(config, dict):
            raise ValueError(
                f"Configuration must be a dictionary/object.\n"
                f"Got: {type(config).__name__}"
            )

        return config, suffix

    except (IOError, OSError) as exc:
        raise ValueError(f"Error reading file {config_path}: {exc}") from exc


def _validate_against_schema(
    config: Dict[str, Any],
    schema: Dict[str, Any],
    config_type: str
) -> List[str]:
    """Validate config against JSON schema.

    Args:
        config: Configuration dictionary.
        schema: JSON schema to validate against.
        config_type: Type name for error messages ('pipeline' or 'dataset').

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    try:
        import jsonschema
        from jsonschema import Draft7Validator, ValidationError

        validator = Draft7Validator(schema)
        for error in sorted(validator.iter_errors(config), key=lambda e: str(e.path)):
            path = ".".join(str(p) for p in error.path) if error.path else "(root)"
            errors.append(f"At '{path}': {error.message}")

    except ImportError:
        # jsonschema not installed - do basic validation
        logger.warning("jsonschema not installed. Using basic validation only.")
        errors.extend(_basic_validate(config, config_type))

    return errors


def _basic_validate(config: Dict[str, Any], config_type: str) -> List[str]:
    """Basic validation without jsonschema dependency.

    Args:
        config: Configuration dictionary.
        config_type: 'pipeline' or 'dataset'.

    Returns:
        List of validation error messages.
    """
    errors = []

    if config_type == 'pipeline':
        if 'pipeline' not in config:
            errors.append("Missing required key: 'pipeline'")
        elif not isinstance(config['pipeline'], list):
            errors.append("'pipeline' must be a list of steps")
        elif len(config['pipeline']) == 0:
            errors.append("'pipeline' list cannot be empty")
    else:  # dataset
        has_data = any(key in config for key in ['train_x', 'test_x', 'folder'])
        if not has_data:
            errors.append("Missing data source: need 'train_x', 'test_x', or 'folder'")

        # Validate task_type if present
        if 'task_type' in config:
            valid_task_types = ['regression', 'binary_classification', 'multiclass_classification', 'auto']
            if config['task_type'] not in valid_task_types:
                errors.append(
                    f"Invalid task_type: '{config['task_type']}'. "
                    f"Valid values: {valid_task_types}"
                )

        # Validate signal_type if present
        if 'signal_type' in config:
            valid_signal_types = ['absorbance', 'reflectance', 'reflectance%', 'transmittance', 'transmittance%', 'auto']
            if config['signal_type'] not in valid_signal_types:
                errors.append(
                    f"Invalid signal_type: '{config['signal_type']}'. "
                    f"Valid values: {valid_signal_types}"
                )

    return errors


def _check_file_paths(config: Dict[str, Any], base_path: Optional[Path] = None) -> List[str]:
    """Check that referenced data files exist.

    Args:
        config: Dataset configuration dictionary.
        base_path: Base path for resolving relative paths.

    Returns:
        List of warning messages for missing files.
    """
    warnings = []
    file_keys = ['train_x', 'train_y', 'test_x', 'test_y', 'train_group', 'test_group']

    for key in file_keys:
        if key not in config:
            continue

        value = config[key]
        paths = value if isinstance(value, list) else [value]

        for file_path in paths:
            if not isinstance(file_path, str):
                continue

            path = Path(file_path)
            if not path.is_absolute() and base_path:
                path = base_path / path

            if not path.exists():
                warnings.append(f"File not found for '{key}': {file_path}")

    return warnings


def validate_pipeline_config(
    config_source: Union[str, Dict[str, Any]],
    check_class_paths: bool = False
) -> Tuple[bool, List[str], List[str]]:
    """Validate a pipeline configuration.

    Args:
        config_source: Path to config file, or config dictionary.
        check_class_paths: If True, verify that class paths are importable.

    Returns:
        Tuple of (is_valid, errors, warnings).
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Load config if path
    if isinstance(config_source, str):
        try:
            config, _ = _load_config_content(config_source)
        except (FileNotFoundError, ValueError) as exc:
            return False, [str(exc)], []
    else:
        config = config_source

    # Validate against schema
    schema_errors = _validate_against_schema(config, PIPELINE_SCHEMA, 'pipeline')
    errors.extend(schema_errors)

    # Additional validations
    if 'pipeline' in config and isinstance(config['pipeline'], list):
        for i, step in enumerate(config['pipeline']):
            if step is None:
                warnings.append(f"Step {i+1} is null (will be skipped)")

            if isinstance(step, dict) and 'class' in step:
                class_path = step['class']

                if check_class_paths:
                    # Try to import the class
                    try:
                        module_path, class_name = class_path.rsplit('.', 1)
                        import importlib
                        module = importlib.import_module(module_path)
                        if not hasattr(module, class_name):
                            errors.append(
                                f"Step {i+1}: Class '{class_name}' not found in module '{module_path}'"
                            )
                    except (ValueError, ImportError) as exc:
                        errors.append(
                            f"Step {i+1}: Cannot import class '{class_path}': {exc}"
                        )

    return len(errors) == 0, errors, warnings


def validate_dataset_config(
    config_source: Union[str, Dict[str, Any]],
    check_files: bool = True
) -> Tuple[bool, List[str], List[str]]:
    """Validate a dataset configuration.

    Args:
        config_source: Path to config file, or config dictionary.
        check_files: If True, check that referenced data files exist.

    Returns:
        Tuple of (is_valid, errors, warnings).
    """
    errors: List[str] = []
    warnings: List[str] = []
    base_path: Optional[Path] = None

    # Load config if path
    if isinstance(config_source, str):
        try:
            config, _ = _load_config_content(config_source)
            base_path = Path(config_source).parent
        except (FileNotFoundError, ValueError) as exc:
            return False, [str(exc)], []
    else:
        config = config_source

    # Validate against schema
    schema_errors = _validate_against_schema(config, DATASET_SCHEMA, 'dataset')
    errors.extend(schema_errors)

    # Check file paths
    if check_files and len(errors) == 0:
        file_warnings = _check_file_paths(config, base_path)
        warnings.extend(file_warnings)

    return len(errors) == 0, errors, warnings


def validate_config_file(
    config_path: str,
    config_type: Optional[str] = None,
    check_files: bool = True,
    check_class_paths: bool = False
) -> Tuple[bool, List[str], List[str]]:
    """Validate a configuration file, auto-detecting type if not specified.

    Args:
        config_path: Path to the configuration file.
        config_type: 'pipeline', 'dataset', or None for auto-detection.
        check_files: For dataset configs, check if data files exist.
        check_class_paths: For pipeline configs, verify class imports.

    Returns:
        Tuple of (is_valid, errors, warnings).
    """
    # Load config first
    try:
        config, _ = _load_config_content(config_path)
    except (FileNotFoundError, ValueError) as exc:
        return False, [str(exc)], []

    # Auto-detect type if not specified
    if config_type is None:
        if 'pipeline' in config:
            config_type = 'pipeline'
        elif any(k in config for k in ['train_x', 'test_x', 'folder']):
            config_type = 'dataset'
        else:
            return False, [
                "Cannot determine configuration type.\n"
                "Pipeline configs should have a 'pipeline' key.\n"
                "Dataset configs should have 'train_x', 'test_x', or 'folder'."
            ], []

    # Validate based on type
    if config_type == 'pipeline':
        return validate_pipeline_config(config, check_class_paths=check_class_paths)
    else:
        return validate_dataset_config(
            config,
            check_files=check_files
        )


def get_validation_summary(
    is_valid: bool,
    errors: List[str],
    warnings: List[str],
    config_path: Optional[str] = None
) -> str:
    """Generate a human-readable validation summary.

    Args:
        is_valid: Whether validation passed.
        errors: List of error messages.
        warnings: List of warning messages.
        config_path: Optional path for context.

    Returns:
        Formatted summary string.
    """
    lines = []

    if config_path:
        lines.append(f"Validation results for: {config_path}")
        lines.append("=" * 60)

    if is_valid:
        lines.append("✓ Configuration is valid")
    else:
        lines.append("✗ Configuration has errors")

    if errors:
        lines.append("\nErrors:")
        for error in errors:
            lines.append(f"  - {error}")

    if warnings:
        lines.append("\nWarnings:")
        for warning in warnings:
            lines.append(f"  - {warning}")

    return "\n".join(lines)
