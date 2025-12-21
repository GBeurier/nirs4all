"""
Validators for dataset configuration.

This module provides validation logic for dataset configurations,
checking for consistency, required fields, file existence, and other rules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np


@dataclass
class ValidationError:
    """Represents a validation error.

    Attributes:
        code: Error code for programmatic handling.
        message: Human-readable error message.
        field: The configuration field that caused the error.
        value: The value that caused the error.
        suggestion: Optional suggestion for fixing the error.
    """

    code: str
    message: str
    field: Optional[str] = None
    value: Any = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        parts = [self.message]
        if self.field:
            parts.insert(0, f"[{self.field}]")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " ".join(parts)


@dataclass
class ValidationWarning:
    """Represents a validation warning (non-fatal issue).

    Attributes:
        code: Warning code for programmatic handling.
        message: Human-readable warning message.
        field: The configuration field that caused the warning.
    """

    code: str
    message: str
    field: Optional[str] = None

    def __str__(self) -> str:
        if self.field:
            return f"[{self.field}] {self.message}"
        return self.message


@dataclass
class ValidationResult:
    """Result of configuration validation.

    Attributes:
        is_valid: Whether the configuration is valid (no errors).
        errors: List of validation errors.
        warnings: List of validation warnings.
        normalized_config: The validated and normalized configuration.
    """

    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)
    normalized_config: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        if self.is_valid:
            msg = "Configuration is valid"
            if self.warnings:
                msg += f" with {len(self.warnings)} warning(s)"
            return msg
        return f"Configuration is invalid: {len(self.errors)} error(s)"

    def raise_if_invalid(self) -> None:
        """Raise ValueError if configuration is invalid."""
        if not self.is_valid:
            error_messages = [str(e) for e in self.errors]
            raise ValueError(
                f"Invalid configuration:\n" +
                "\n".join(f"  - {msg}" for msg in error_messages)
            )


class ConfigValidator:
    """Validator for dataset configurations.

    Provides validation rules and methods for checking dataset configurations.
    Supports both legacy and new format configurations.

    Example:
        ```python
        validator = ConfigValidator()
        result = validator.validate(config_dict)
        if not result.is_valid:
            for error in result.errors:
                print(f"Error: {error}")
        ```
    """

    def __init__(
        self,
        check_file_existence: bool = False,
        custom_validators: Optional[List[Callable]] = None
    ):
        """Initialize the validator.

        Args:
            check_file_existence: Whether to check if referenced files exist.
                Default is False to allow validation before files are available.
            custom_validators: Optional list of custom validation functions.
                Each function should accept (config, errors, warnings) and
                add any issues to the lists.
        """
        self.check_file_existence = check_file_existence
        self.custom_validators = custom_validators or []

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate a configuration dictionary.

        Args:
            config: Configuration dictionary to validate.

        Returns:
            ValidationResult with errors, warnings, and normalized config.
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationWarning] = []

        # Create a copy for normalization
        normalized = dict(config)

        # Run validation rules
        self._validate_data_sources(normalized, errors, warnings)
        self._validate_task_type(normalized, errors, warnings)
        self._validate_loading_params(normalized, errors, warnings)
        self._validate_aggregation(normalized, errors, warnings)

        if self.check_file_existence:
            self._validate_file_existence(normalized, errors, warnings)

        # Run custom validators
        for validator in self.custom_validators:
            validator(normalized, errors, warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            normalized_config=normalized if len(errors) == 0 else None
        )

    def _validate_data_sources(
        self,
        config: Dict[str, Any],
        errors: List[ValidationError],
        warnings: List[ValidationWarning]
    ) -> None:
        """Validate that data sources are properly specified."""
        has_train_x = config.get("train_x") is not None
        has_test_x = config.get("test_x") is not None
        has_files = config.get("files") is not None
        has_sources = config.get("sources") is not None

        # Check for any data source
        if not has_train_x and not has_test_x and not has_files and not has_sources:
            errors.append(ValidationError(
                code="NO_DATA_SOURCE",
                message="No data source specified. "
                        "Provide train_x, test_x, files, or sources.",
                suggestion="Add train_x with path to training features CSV."
            ))

        # Check for mixed formats (warning only)
        if (has_train_x or has_test_x) and (has_files or has_sources):
            warnings.append(ValidationWarning(
                code="MIXED_FORMAT",
                message="Both legacy (train_x/test_x) and new format (files/sources) detected. "
                        "Legacy format will take precedence."
            ))

        # Validate multi-source consistency
        train_x = config.get("train_x")
        if isinstance(train_x, list):
            # Check all paths are same type
            if not all(isinstance(p, (str, Path)) for p in train_x):
                if not all(isinstance(p, np.ndarray) for p in train_x):
                    errors.append(ValidationError(
                        code="MIXED_SOURCE_TYPES",
                        message="Multi-source train_x contains mixed types.",
                        field="train_x",
                        suggestion="Use either all file paths or all numpy arrays."
                    ))

    def _validate_task_type(
        self,
        config: Dict[str, Any],
        errors: List[ValidationError],
        warnings: List[ValidationWarning]
    ) -> None:
        """Validate task_type configuration."""
        task_type = config.get("task_type")
        if task_type is not None:
            valid_types = ["auto", "regression", "binary_classification", "multiclass_classification"]
            if isinstance(task_type, str) and task_type.lower() not in valid_types:
                errors.append(ValidationError(
                    code="INVALID_TASK_TYPE",
                    message=f"Invalid task_type: '{task_type}'",
                    field="task_type",
                    value=task_type,
                    suggestion=f"Valid values: {valid_types}"
                ))

    def _validate_loading_params(
        self,
        config: Dict[str, Any],
        errors: List[ValidationError],
        warnings: List[ValidationWarning]
    ) -> None:
        """Validate loading parameters."""
        # Check global_params
        global_params = config.get("global_params")
        if global_params is not None:
            self._validate_params_dict(global_params, "global_params", errors, warnings)

        # Check partition-level params
        for partition in ["train", "test"]:
            params_key = f"{partition}_params"
            params = config.get(params_key)
            if params is not None:
                self._validate_params_dict(params, params_key, errors, warnings)

            # Check file-level params
            for data_type in ["x", "y", "group"]:
                params_key = f"{partition}_{data_type}_params"
                params = config.get(params_key)
                if params is not None:
                    self._validate_params_dict(params, params_key, errors, warnings)

    def _validate_params_dict(
        self,
        params: Any,
        field_name: str,
        errors: List[ValidationError],
        warnings: List[ValidationWarning]
    ) -> None:
        """Validate a params dictionary."""
        if not isinstance(params, dict):
            errors.append(ValidationError(
                code="INVALID_PARAMS_TYPE",
                message=f"Expected dict for {field_name}, got {type(params).__name__}",
                field=field_name,
                value=params
            ))
            return

        # Validate header_unit
        header_unit = params.get("header_unit")
        if header_unit is not None:
            valid_units = ["cm-1", "nm", "none", "text", "index"]
            if isinstance(header_unit, str) and header_unit.lower() not in valid_units:
                errors.append(ValidationError(
                    code="INVALID_HEADER_UNIT",
                    message=f"Invalid header_unit: '{header_unit}'",
                    field=f"{field_name}.header_unit",
                    value=header_unit,
                    suggestion=f"Valid values: {valid_units}"
                ))

        # Validate signal_type
        signal_type = params.get("signal_type")
        if signal_type is not None:
            valid_types = [
                "auto", "absorbance", "reflectance", "reflectance%",
                "transmittance", "transmittance%", "log(1/R)", "kubelka-munk"
            ]
            if isinstance(signal_type, str) and signal_type.lower() not in valid_types:
                warnings.append(ValidationWarning(
                    code="UNKNOWN_SIGNAL_TYPE",
                    message=f"Unknown signal_type: '{signal_type}'. May be auto-detected.",
                    field=f"{field_name}.signal_type"
                ))

        # Validate na_policy
        na_policy = params.get("na_policy")
        if na_policy is not None:
            valid_policies = ["auto", "remove", "abort"]
            if isinstance(na_policy, str) and na_policy.lower() not in valid_policies:
                errors.append(ValidationError(
                    code="INVALID_NA_POLICY",
                    message=f"Invalid na_policy: '{na_policy}'",
                    field=f"{field_name}.na_policy",
                    value=na_policy,
                    suggestion=f"Valid values: {valid_policies}"
                ))

    def _validate_aggregation(
        self,
        config: Dict[str, Any],
        errors: List[ValidationError],
        warnings: List[ValidationWarning]
    ) -> None:
        """Validate aggregation settings."""
        aggregate = config.get("aggregate")
        aggregate_method = config.get("aggregate_method")

        # aggregate_method without aggregate
        if aggregate_method is not None and aggregate is None:
            warnings.append(ValidationWarning(
                code="UNUSED_AGGREGATE_METHOD",
                message="aggregate_method specified without aggregate. It will be ignored.",
                field="aggregate_method"
            ))

        # Validate aggregate_method value
        if aggregate_method is not None:
            valid_methods = ["mean", "median", "vote"]
            if isinstance(aggregate_method, str) and aggregate_method.lower() not in valid_methods:
                errors.append(ValidationError(
                    code="INVALID_AGGREGATE_METHOD",
                    message=f"Invalid aggregate_method: '{aggregate_method}'",
                    field="aggregate_method",
                    value=aggregate_method,
                    suggestion=f"Valid values: {valid_methods}"
                ))

    def _validate_file_existence(
        self,
        config: Dict[str, Any],
        errors: List[ValidationError],
        warnings: List[ValidationWarning]
    ) -> None:
        """Validate that referenced files exist."""
        file_fields = [
            "train_x", "train_y", "train_group",
            "test_x", "test_y", "test_group"
        ]

        for field_name in file_fields:
            value = config.get(field_name)
            if value is None:
                continue

            # Skip numpy arrays
            if isinstance(value, np.ndarray):
                continue

            # Handle lists (multi-source)
            if isinstance(value, list):
                for i, path in enumerate(value):
                    if isinstance(path, (str, Path)) and not Path(path).exists():
                        warnings.append(ValidationWarning(
                            code="FILE_NOT_FOUND",
                            message=f"File not found: {path}",
                            field=f"{field_name}[{i}]"
                        ))
            elif isinstance(value, (str, Path)):
                if not Path(value).exists():
                    warnings.append(ValidationWarning(
                        code="FILE_NOT_FOUND",
                        message=f"File not found: {value}",
                        field=field_name
                    ))


def validate_config(
    config: Dict[str, Any],
    check_file_existence: bool = False
) -> ValidationResult:
    """Convenience function to validate a configuration.

    Args:
        config: Configuration dictionary to validate.
        check_file_existence: Whether to check if referenced files exist.

    Returns:
        ValidationResult with errors, warnings, and normalized config.
    """
    validator = ConfigValidator(check_file_existence=check_file_existence)
    return validator.validate(config)
