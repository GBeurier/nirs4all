"""
Configuration module for nirs4all.

Provides validation and schema utilities for pipeline and dataset configurations.
"""

from nirs4all.config.validator import (
    validate_pipeline_config,
    validate_dataset_config,
    validate_config_file,
    ConfigValidationError,
    PIPELINE_SCHEMA,
    DATASET_SCHEMA,
)

__all__ = [
    'validate_pipeline_config',
    'validate_dataset_config',
    'validate_config_file',
    'ConfigValidationError',
    'PIPELINE_SCHEMA',
    'DATASET_SCHEMA',
]
