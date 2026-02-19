"""
Validation module for dataset configuration.

This module provides validators for dataset configuration schemas,
offering detailed error messages and validation results.
"""

from .error_codes import (
    DiagnosticBuilder,
    DiagnosticMessage,
    DiagnosticReport,
    ErrorCategory,
    ErrorCode,
    ErrorRegistry,
    ErrorSeverity,
)
from .validators import (
    ConfigValidator,
    ValidationError,
    ValidationResult,
    ValidationWarning,
    validate_config,
)

__all__ = [
    # Validators
    "ConfigValidator",
    "ValidationError",
    "ValidationWarning",
    "ValidationResult",
    "validate_config",
    # Error codes
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorCode",
    "ErrorRegistry",
    "DiagnosticMessage",
    "DiagnosticBuilder",
    "DiagnosticReport",
]
