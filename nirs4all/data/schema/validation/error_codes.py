"""
Error codes and diagnostics for dataset configuration.

This module provides comprehensive error codes, diagnostic messages,
and suggestion systems for configuration issues.

Phase 8 Implementation - Dataset Configuration Roadmap
Section 8.4: Error Handling & Diagnostics
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type


class ErrorCategory(str, Enum):
    """Categories of configuration errors."""
    SCHEMA = "schema"           # Schema validation errors
    FILE = "file"               # File-related errors
    DATA = "data"               # Data content errors
    LOADING = "loading"         # Loading/parsing errors
    PARTITION = "partition"     # Partition-related errors
    AGGREGATION = "aggregation" # Aggregation errors
    VARIATION = "variation"     # Variation/source errors
    FOLD = "fold"               # Cross-validation fold errors
    RUNTIME = "runtime"         # Runtime errors


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    ERROR = "error"         # Fatal error, cannot proceed
    WARNING = "warning"     # Non-fatal issue
    INFO = "info"           # Informational message


@dataclass
class ErrorCode:
    """Error code definition.

    Attributes:
        code: Unique error code (e.g., "E001").
        category: Error category.
        severity: Error severity.
        message_template: Template for error message with {placeholders}.
        suggestion_template: Template for fix suggestion.
        documentation_url: Link to relevant documentation.
    """
    code: str
    category: ErrorCategory
    severity: ErrorSeverity
    message_template: str
    suggestion_template: Optional[str] = None
    documentation_url: Optional[str] = None


# =============================================================================
# Error Code Registry
# =============================================================================

class ErrorRegistry:
    """Registry of all error codes."""

    _codes: Dict[str, ErrorCode] = {}

    # Schema Errors (E1xx)
    E100 = ErrorCode(
        code="E100",
        category=ErrorCategory.SCHEMA,
        severity=ErrorSeverity.ERROR,
        message_template="Invalid configuration structure: {details}",
        suggestion_template="Check that the configuration is a valid dictionary.",
    )

    E101 = ErrorCode(
        code="E101",
        category=ErrorCategory.SCHEMA,
        severity=ErrorSeverity.ERROR,
        message_template="Missing required field: {field}",
        suggestion_template="Add the '{field}' field to your configuration.",
    )

    E102 = ErrorCode(
        code="E102",
        category=ErrorCategory.SCHEMA,
        severity=ErrorSeverity.ERROR,
        message_template="Invalid type for '{field}': expected {expected}, got {actual}",
        suggestion_template="Change '{field}' to be of type {expected}.",
    )

    E103 = ErrorCode(
        code="E103",
        category=ErrorCategory.SCHEMA,
        severity=ErrorSeverity.ERROR,
        message_template="Invalid value for '{field}': {value}. Valid values: {valid_values}",
        suggestion_template="Use one of the valid values: {valid_values}",
    )

    E104 = ErrorCode(
        code="E104",
        category=ErrorCategory.SCHEMA,
        severity=ErrorSeverity.ERROR,
        message_template="No data source specified",
        suggestion_template="Add 'train_x', 'test_x', 'folder', 'sources', or 'variations' to your configuration.",
    )

    # File Errors (E2xx)
    E200 = ErrorCode(
        code="E200",
        category=ErrorCategory.FILE,
        severity=ErrorSeverity.ERROR,
        message_template="File not found: {path}",
        suggestion_template="Check that the file path is correct and the file exists.",
    )

    E201 = ErrorCode(
        code="E201",
        category=ErrorCategory.FILE,
        severity=ErrorSeverity.ERROR,
        message_template="Cannot read file: {path}. Error: {error}",
        suggestion_template="Check file permissions and encoding.",
    )

    E202 = ErrorCode(
        code="E202",
        category=ErrorCategory.FILE,
        severity=ErrorSeverity.ERROR,
        message_template="Unsupported file format: {format}",
        suggestion_template="Supported formats: CSV, NPY, NPZ, Parquet, Excel, MATLAB",
    )

    E203 = ErrorCode(
        code="E203",
        category=ErrorCategory.FILE,
        severity=ErrorSeverity.ERROR,
        message_template="Empty file: {path}",
        suggestion_template="Ensure the file contains data.",
    )

    E204 = ErrorCode(
        code="E204",
        category=ErrorCategory.FILE,
        severity=ErrorSeverity.WARNING,
        message_template="File encoding issue: {path}. Using fallback encoding: {encoding}",
        suggestion_template="Specify the encoding explicitly in loading parameters.",
    )

    # Data Errors (E3xx)
    E300 = ErrorCode(
        code="E300",
        category=ErrorCategory.DATA,
        severity=ErrorSeverity.ERROR,
        message_template="Data shape mismatch: {details}",
        suggestion_template="Ensure all data arrays have consistent sample counts.",
    )

    E301 = ErrorCode(
        code="E301",
        category=ErrorCategory.DATA,
        severity=ErrorSeverity.ERROR,
        message_template="NA values found in data and na_policy='abort': {details}",
        suggestion_template="Set na_policy='remove' or clean your data before loading.",
    )

    E302 = ErrorCode(
        code="E302",
        category=ErrorCategory.DATA,
        severity=ErrorSeverity.WARNING,
        message_template="NA values removed: {count} rows affected",
        suggestion_template="Review your data for missing values.",
    )

    E303 = ErrorCode(
        code="E303",
        category=ErrorCategory.DATA,
        severity=ErrorSeverity.ERROR,
        message_template="Column not found: '{column}' in {file}",
        suggestion_template="Available columns: {available}",
    )

    E304 = ErrorCode(
        code="E304",
        category=ErrorCategory.DATA,
        severity=ErrorSeverity.WARNING,
        message_template="Non-numeric values in feature data at column(s): {columns}",
        suggestion_template="Features should be numeric. Non-numeric values will be converted to NaN.",
    )

    # Loading Errors (E4xx)
    E400 = ErrorCode(
        code="E400",
        category=ErrorCategory.LOADING,
        severity=ErrorSeverity.ERROR,
        message_template="Failed to parse CSV: {error}",
        suggestion_template="Check delimiter and encoding settings.",
    )

    E401 = ErrorCode(
        code="E401",
        category=ErrorCategory.LOADING,
        severity=ErrorSeverity.ERROR,
        message_template="Invalid JSON configuration at line {line}: {error}",
        suggestion_template="Check JSON syntax around line {line}.",
    )

    E402 = ErrorCode(
        code="E402",
        category=ErrorCategory.LOADING,
        severity=ErrorSeverity.ERROR,
        message_template="Invalid YAML configuration at line {line}: {error}",
        suggestion_template="Check YAML indentation and syntax around line {line}.",
    )

    E403 = ErrorCode(
        code="E403",
        category=ErrorCategory.LOADING,
        severity=ErrorSeverity.ERROR,
        message_template="Archive error: {error}",
        suggestion_template="Ensure the archive is not corrupted and contains the expected files.",
    )

    # Partition Errors (E5xx)
    E500 = ErrorCode(
        code="E500",
        category=ErrorCategory.PARTITION,
        severity=ErrorSeverity.ERROR,
        message_template="Invalid partition specification: {details}",
        suggestion_template="Use 'train', 'test', column-based, or percentage-based partition.",
    )

    E501 = ErrorCode(
        code="E501",
        category=ErrorCategory.PARTITION,
        severity=ErrorSeverity.ERROR,
        message_template="Partition column not found: '{column}'",
        suggestion_template="Available columns: {available}",
    )

    E502 = ErrorCode(
        code="E502",
        category=ErrorCategory.PARTITION,
        severity=ErrorSeverity.ERROR,
        message_template="Partition indices out of range: max index {max_index}, data has {n_samples} samples",
        suggestion_template="Ensure partition indices are within valid range.",
    )

    E503 = ErrorCode(
        code="E503",
        category=ErrorCategory.PARTITION,
        severity=ErrorSeverity.WARNING,
        message_template="Overlapping partition indices detected",
        suggestion_template="Train and test indices should not overlap.",
    )

    # Aggregation Errors (E6xx)
    E600 = ErrorCode(
        code="E600",
        category=ErrorCategory.AGGREGATION,
        severity=ErrorSeverity.ERROR,
        message_template="Aggregation column not found: '{column}'",
        suggestion_template="Available columns in metadata: {available}",
    )

    E601 = ErrorCode(
        code="E601",
        category=ErrorCategory.AGGREGATION,
        severity=ErrorSeverity.WARNING,
        message_template="Group '{group}' has only {count} sample(s), below minimum {min_samples}",
        suggestion_template="Consider lowering aggregate_min_samples or reviewing your data.",
    )

    E602 = ErrorCode(
        code="E602",
        category=ErrorCategory.AGGREGATION,
        severity=ErrorSeverity.ERROR,
        message_template="Invalid aggregation method: '{method}'",
        suggestion_template="Valid methods: mean, median, vote, min, max, sum, std, first, last",
    )

    # Variation/Source Errors (E7xx)
    E700 = ErrorCode(
        code="E700",
        category=ErrorCategory.VARIATION,
        severity=ErrorSeverity.ERROR,
        message_template="Duplicate source name: '{name}'",
        suggestion_template="Each source must have a unique name.",
    )

    E701 = ErrorCode(
        code="E701",
        category=ErrorCategory.VARIATION,
        severity=ErrorSeverity.ERROR,
        message_template="Duplicate variation name: '{name}'",
        suggestion_template="Each variation must have a unique name.",
    )

    E702 = ErrorCode(
        code="E702",
        category=ErrorCategory.VARIATION,
        severity=ErrorSeverity.ERROR,
        message_template="Unknown variation(s) in variation_select: {names}",
        suggestion_template="Available variations: {available}",
    )

    E703 = ErrorCode(
        code="E703",
        category=ErrorCategory.VARIATION,
        severity=ErrorSeverity.ERROR,
        message_template="variation_mode='select' requires 'variation_select' to be specified",
        suggestion_template="Add 'variation_select: [\"var1\", \"var2\"]' to your configuration.",
    )

    E704 = ErrorCode(
        code="E704",
        category=ErrorCategory.VARIATION,
        severity=ErrorSeverity.ERROR,
        message_template="Sample count mismatch across sources: {details}",
        suggestion_template="All sources must have the same number of samples.",
    )

    # Fold Errors (E8xx)
    E800 = ErrorCode(
        code="E800",
        category=ErrorCategory.FOLD,
        severity=ErrorSeverity.ERROR,
        message_template="Invalid fold file format: {error}",
        suggestion_template="Fold files should be CSV with fold columns or JSON/YAML with fold definitions.",
    )

    E801 = ErrorCode(
        code="E801",
        category=ErrorCategory.FOLD,
        severity=ErrorSeverity.ERROR,
        message_template="Fold sample IDs do not match dataset: {details}",
        suggestion_template="Ensure fold file was generated for this dataset.",
    )

    E802 = ErrorCode(
        code="E802",
        category=ErrorCategory.FOLD,
        severity=ErrorSeverity.WARNING,
        message_template="Fold file has {fold_samples} samples, dataset has {data_samples} samples",
        suggestion_template="Folds will be adjusted to match current dataset size.",
    )

    # Runtime Errors (E9xx)
    E900 = ErrorCode(
        code="E900",
        category=ErrorCategory.RUNTIME,
        severity=ErrorSeverity.ERROR,
        message_template="Unexpected error during loading: {error}",
        suggestion_template="Please report this issue with the full error traceback.",
    )

    @classmethod
    def get(cls, code: str) -> Optional[ErrorCode]:
        """Get error code by code string."""
        return getattr(cls, code, None)

    @classmethod
    def all_codes(cls) -> Dict[str, ErrorCode]:
        """Get all error codes."""
        return {
            name: value
            for name, value in cls.__dict__.items()
            if isinstance(value, ErrorCode)
        }


@dataclass
class DiagnosticMessage:
    """A diagnostic message with formatted content.

    Attributes:
        error_code: The ErrorCode definition.
        message: Formatted error message.
        suggestion: Formatted suggestion (if any).
        context: Additional context information.
        location: File/line location (if applicable).
    """
    error_code: ErrorCode
    message: str
    suggestion: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    location: Optional[str] = None

    @property
    def code(self) -> str:
        """Get the error code string."""
        return self.error_code.code

    @property
    def severity(self) -> ErrorSeverity:
        """Get the error severity."""
        return self.error_code.severity

    @property
    def category(self) -> ErrorCategory:
        """Get the error category."""
        return self.error_code.category

    def __str__(self) -> str:
        parts = [f"[{self.code}] {self.message}"]
        if self.location:
            parts.insert(0, f"At {self.location}:")
        if self.suggestion:
            parts.append(f"\n  Suggestion: {self.suggestion}")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code": self.code,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "suggestion": self.suggestion,
            "location": self.location,
            "context": self.context,
        }


class DiagnosticBuilder:
    """Builder for diagnostic messages.

    Example:
        ```python
        builder = DiagnosticBuilder()

        # Create error message
        error = builder.create(
            ErrorRegistry.E200,
            path="/path/to/file.csv"
        )

        # Create with location
        error = builder.create(
            ErrorRegistry.E401,
            line=10,
            error="Unexpected token",
            location="config.json:10"
        )
        ```
    """

    def create(
        self,
        error_code: ErrorCode,
        location: Optional[str] = None,
        **kwargs,
    ) -> DiagnosticMessage:
        """Create a diagnostic message.

        Args:
            error_code: The error code definition.
            location: Optional file/line location.
            **kwargs: Parameters for message template.

        Returns:
            DiagnosticMessage instance.
        """
        # Format message
        try:
            message = error_code.message_template.format(**kwargs)
        except KeyError as e:
            message = f"{error_code.message_template} (missing: {e})"

        # Format suggestion
        suggestion = None
        if error_code.suggestion_template:
            try:
                suggestion = error_code.suggestion_template.format(**kwargs)
            except KeyError:
                suggestion = error_code.suggestion_template

        return DiagnosticMessage(
            error_code=error_code,
            message=message,
            suggestion=suggestion,
            context=kwargs,
            location=location,
        )

    def file_not_found(self, path: str) -> DiagnosticMessage:
        """Create file not found error."""
        return self.create(ErrorRegistry.E200, path=path)

    def invalid_value(
        self,
        field: str,
        value: Any,
        valid_values: List[Any],
    ) -> DiagnosticMessage:
        """Create invalid value error."""
        return self.create(
            ErrorRegistry.E103,
            field=field,
            value=value,
            valid_values=valid_values,
        )

    def missing_field(self, field: str) -> DiagnosticMessage:
        """Create missing field error."""
        return self.create(ErrorRegistry.E101, field=field)

    def type_error(
        self,
        field: str,
        expected: str,
        actual: str,
    ) -> DiagnosticMessage:
        """Create type error."""
        return self.create(
            ErrorRegistry.E102,
            field=field,
            expected=expected,
            actual=actual,
        )


@dataclass
class DiagnosticReport:
    """Collection of diagnostic messages.

    Attributes:
        messages: List of diagnostic messages.
        config_path: Path to the configuration file (if any).
    """
    messages: List[DiagnosticMessage] = field(default_factory=list)
    config_path: Optional[str] = None

    def add(self, message: DiagnosticMessage) -> None:
        """Add a diagnostic message."""
        self.messages.append(message)

    def add_error(
        self,
        error_code: ErrorCode,
        location: Optional[str] = None,
        **kwargs,
    ) -> DiagnosticMessage:
        """Create and add an error message."""
        builder = DiagnosticBuilder()
        message = builder.create(error_code, location, **kwargs)
        self.add(message)
        return message

    @property
    def errors(self) -> List[DiagnosticMessage]:
        """Get all error messages."""
        return [m for m in self.messages if m.severity == ErrorSeverity.ERROR]

    @property
    def warnings(self) -> List[DiagnosticMessage]:
        """Get all warning messages."""
        return [m for m in self.messages if m.severity == ErrorSeverity.WARNING]

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(m.severity == ErrorSeverity.ERROR for m in self.messages)

    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid (no errors)."""
        return not self.has_errors

    def __str__(self) -> str:
        lines = []
        if self.config_path:
            lines.append(f"Diagnostics for: {self.config_path}")
            lines.append("=" * 60)

        if not self.messages:
            lines.append("✓ No issues found")
            return "\n".join(lines)

        errors = self.errors
        warnings = self.warnings

        if errors:
            lines.append(f"\nErrors ({len(errors)}):")
            for msg in errors:
                lines.append(f"  ✗ {msg}")

        if warnings:
            lines.append(f"\nWarnings ({len(warnings)}):")
            for msg in warnings:
                lines.append(f"  ⚠ {msg}")

        if not errors:
            lines.append("\n✓ Configuration is valid")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config_path": self.config_path,
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "messages": [m.to_dict() for m in self.messages],
        }
