"""
Validation module for dataset configuration.

This module provides validators for dataset configuration schemas,
offering detailed error messages and validation results.
"""

from .validators import (
    ConfigValidator,
    ValidationError,
    ValidationWarning,
    ValidationResult,
    validate_config,
)

from .error_codes import (
    ErrorCategory,
    ErrorSeverity,
    ErrorCode,
    ErrorRegistry,
    DiagnosticMessage,
    DiagnosticBuilder,
    DiagnosticReport,
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
