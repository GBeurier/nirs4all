"""Validation utilities for generated configurations.

This module provides validation functionality for generator specifications
and expanded configurations, including schema validation and semantic checks.

Main Classes:
    ValidationError: Exception for validation failures
    ValidationResult: Dataclass containing validation outcome

Main Functions:
    validate_spec: Validate a generator specification before expansion
    validate_config: Validate an expanded configuration
"""

from .schema import (
    ValidationError,
    ValidationResult,
    ValidationSeverity,
    validate_spec,
    validate_config,
    validate_expanded_configs,
)

__all__ = [
    "ValidationError",
    "ValidationResult",
    "ValidationSeverity",
    "validate_spec",
    "validate_config",
    "validate_expanded_configs",
]
