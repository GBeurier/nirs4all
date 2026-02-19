"""
Configuration module for nirs4all.

Provides validation and schema utilities for pipeline and dataset configurations,
and the CacheConfig dataclass for runtime caching settings.
"""

from nirs4all.config.cache_config import CacheConfig
from nirs4all.config.validator import (
    DATASET_SCHEMA,
    PIPELINE_SCHEMA,
    ConfigValidationError,
    validate_config_file,
    validate_dataset_config,
    validate_pipeline_config,
)

__all__ = [
    'CacheConfig',
    'validate_pipeline_config',
    'validate_dataset_config',
    'validate_config_file',
    'ConfigValidationError',
    'PIPELINE_SCHEMA',
    'DATASET_SCHEMA',
]
