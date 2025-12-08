"""Schema validation for generator specifications and expanded configurations.

This module provides comprehensive validation for:
- Generator specification syntax (before expansion)
- Expanded configuration structure (after expansion)
- Semantic validation of keyword usage

Classes:
    ValidationError: Exception containing validation failure details
    ValidationResult: Dataclass with validation outcome
    ValidationSeverity: Enum for error severity levels

Functions:
    validate_spec: Validate a generator specification
    validate_config: Validate an expanded configuration
    validate_expanded_configs: Validate a list of expanded configs
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from ..keywords import (
    OR_KEYWORD,
    RANGE_KEYWORD,
    SIZE_KEYWORD,
    COUNT_KEYWORD,
    PICK_KEYWORD,
    ARRANGE_KEYWORD,
    THEN_PICK_KEYWORD,
    THEN_ARRANGE_KEYWORD,
    ALL_KEYWORDS,
    PURE_OR_KEYS,
    PURE_RANGE_KEYS,
)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"  # Critical issue that will cause failure
    WARNING = "warning"  # Potential issue that may cause unexpected behavior
    INFO = "info"  # Informational, non-blocking suggestion


@dataclass
class ValidationError(Exception):
    """Exception for validation failures with detailed context.

    Attributes:
        message: Human-readable error description
        path: JSONPath-like location of the error (e.g., "root._or_[0]")
        severity: Error severity level
        code: Machine-readable error code
        suggestion: Optional suggestion for fixing the error
    """

    message: str
    path: str = ""
    severity: ValidationSeverity = ValidationSeverity.ERROR
    code: str = ""
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        """Format error message with path."""
        location = f" at {self.path}" if self.path else ""
        return f"[{self.severity.value.upper()}] {self.message}{location}"

    def __repr__(self) -> str:
        return (
            f"ValidationError(message={self.message!r}, path={self.path!r}, "
            f"severity={self.severity}, code={self.code!r})"
        )


@dataclass
class ValidationResult:
    """Result of configuration validation.

    Attributes:
        is_valid: True if no errors (warnings allowed)
        errors: List of validation errors
        warnings: List of validation warnings
        info: List of informational messages
        node_count: Number of nodes validated
        generator_count: Number of generator nodes found
    """

    is_valid: bool = True
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info: List[ValidationError] = field(default_factory=list)
    node_count: int = 0
    generator_count: int = 0

    def add_error(self, error: ValidationError) -> None:
        """Add a validation error."""
        if error.severity == ValidationSeverity.ERROR:
            self.errors.append(error)
            self.is_valid = False
        elif error.severity == ValidationSeverity.WARNING:
            self.warnings.append(error)
        else:
            self.info.append(error)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another validation result into this one."""
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        self.node_count += other.node_count
        self.generator_count += other.generator_count
        return self

    def __str__(self) -> str:
        """Format validation result summary."""
        if self.is_valid:
            status = "VALID"
        else:
            status = f"INVALID ({len(self.errors)} errors)"

        parts = [f"ValidationResult: {status}"]
        if self.warnings:
            parts.append(f"{len(self.warnings)} warnings")
        parts.append(f"{self.node_count} nodes, {self.generator_count} generators")

        return " | ".join(parts)


# =============================================================================
# Specification Validation (before expansion)
# =============================================================================

def validate_spec(
    spec: Any,
    path: str = "root",
    strict: bool = False,
    custom_validators: Optional[List[Callable]] = None
) -> ValidationResult:
    """Validate a generator specification before expansion.

    Recursively validates the structure of a generator specification,
    checking for valid syntax, consistent keyword usage, and semantic
    correctness.

    Args:
        spec: The specification to validate (can be any type).
        path: JSONPath-like location for error reporting.
        strict: If True, also report warnings as errors.
        custom_validators: Optional list of custom validation functions.
            Each function should accept (node, path) and return ValidationResult.

    Returns:
        ValidationResult containing validation outcome.

    Examples:
        >>> result = validate_spec({"_or_": ["A", "B"]})
        >>> result.is_valid
        True

        >>> result = validate_spec({"_or_": "not a list"})
        >>> result.is_valid
        False
        >>> result.errors[0].message
        "_or_ must be a list, got str"
    """
    result = ValidationResult()
    result.node_count = 1

    # Handle non-dict types
    if isinstance(spec, list):
        for i, item in enumerate(spec):
            item_result = validate_spec(
                item, f"{path}[{i}]", strict, custom_validators
            )
            result.merge(item_result)
        return result

    if not isinstance(spec, dict):
        # Scalars are always valid
        return result

    # Validate dict node
    result = _validate_dict_spec(spec, path, strict)

    # Recursively validate nested values
    for key, value in spec.items():
        if key not in ALL_KEYWORDS and isinstance(value, (dict, list)):
            nested_result = validate_spec(
                value, f"{path}.{key}", strict, custom_validators
            )
            result.merge(nested_result)

    # Run custom validators
    if custom_validators:
        for validator in custom_validators:
            custom_result = validator(spec, path)
            if custom_result:
                result.merge(custom_result)

    return result


def _validate_dict_spec(spec: Dict[str, Any], path: str, strict: bool) -> ValidationResult:
    """Validate a dictionary specification node.

    Args:
        spec: Dictionary node to validate.
        path: Current path for error reporting.
        strict: Whether to treat warnings as errors.

    Returns:
        ValidationResult for this node.
    """
    result = ValidationResult()
    result.node_count = 1

    # Check if this is a generator node
    has_or = OR_KEYWORD in spec
    has_range = RANGE_KEYWORD in spec

    if has_or and has_range:
        result.add_error(ValidationError(
            message="Cannot have both _or_ and _range_ in the same node",
            path=path,
            code="CONFLICTING_KEYWORDS",
            suggestion="Use separate nodes for _or_ and _range_"
        ))
        return result

    # Validate OR node
    if has_or:
        result.generator_count = 1
        or_result = _validate_or_spec(spec, path, strict)
        result.merge(or_result)
        return result

    # Validate RANGE node
    if has_range:
        result.generator_count = 1
        range_result = _validate_range_spec(spec, path, strict)
        result.merge(range_result)
        return result

    # Check for orphaned modifier keywords
    orphaned = set(spec.keys()) & {
        SIZE_KEYWORD, PICK_KEYWORD, ARRANGE_KEYWORD,
        THEN_PICK_KEYWORD, THEN_ARRANGE_KEYWORD
    }
    if orphaned:
        result.add_error(ValidationError(
            message=f"Modifier keywords {orphaned} without _or_",
            path=path,
            severity=ValidationSeverity.WARNING if not strict else ValidationSeverity.ERROR,
            code="ORPHANED_MODIFIERS",
            suggestion="Add _or_ keyword or remove orphaned modifiers"
        ))

    return result


def _validate_or_spec(spec: Dict[str, Any], path: str, strict: bool) -> ValidationResult:
    """Validate an _or_ specification node.

    Args:
        spec: Dictionary node containing _or_.
        path: Current path for error reporting.
        strict: Whether to treat warnings as errors.

    Returns:
        ValidationResult for this OR node.
    """
    result = ValidationResult()
    or_value = spec[OR_KEYWORD]

    # _or_ must be a list
    if not isinstance(or_value, list):
        result.add_error(ValidationError(
            message=f"_or_ must be a list, got {type(or_value).__name__}",
            path=f"{path}.{OR_KEYWORD}",
            code="INVALID_OR_TYPE"
        ))
        return result

    # Check for empty _or_
    if len(or_value) == 0:
        result.add_error(ValidationError(
            message="Empty _or_ list will generate no configurations",
            path=f"{path}.{OR_KEYWORD}",
            severity=ValidationSeverity.WARNING if not strict else ValidationSeverity.ERROR,
            code="EMPTY_OR"
        ))

    # Validate size/pick/arrange specifications
    for key in (SIZE_KEYWORD, PICK_KEYWORD, ARRANGE_KEYWORD):
        if key in spec:
            size_result = _validate_size_spec(spec[key], key, len(or_value), f"{path}.{key}")
            result.merge(size_result)

    # Validate then_pick/then_arrange
    for key in (THEN_PICK_KEYWORD, THEN_ARRANGE_KEYWORD):
        if key in spec:
            if PICK_KEYWORD not in spec and ARRANGE_KEYWORD not in spec and SIZE_KEYWORD not in spec:
                result.add_error(ValidationError(
                    message=f"{key} requires pick, arrange, or size to be specified",
                    path=f"{path}.{key}",
                    code="ORPHANED_THEN_KEYWORD"
                ))

    # Validate count
    if COUNT_KEYWORD in spec:
        count = spec[COUNT_KEYWORD]
        if not isinstance(count, int):
            result.add_error(ValidationError(
                message=f"count must be an integer, got {type(count).__name__}",
                path=f"{path}.{COUNT_KEYWORD}",
                code="INVALID_COUNT_TYPE"
            ))
        elif count < 0:
            result.add_error(ValidationError(
                message=f"count must be non-negative, got {count}",
                path=f"{path}.{COUNT_KEYWORD}",
                code="NEGATIVE_COUNT"
            ))

    # Check for conflicting selection modes
    selection_modes = sum(1 for k in (SIZE_KEYWORD, PICK_KEYWORD, ARRANGE_KEYWORD) if k in spec)
    if selection_modes > 1:
        result.add_error(ValidationError(
            message="Cannot use size, pick, and arrange together",
            path=path,
            severity=ValidationSeverity.WARNING if not strict else ValidationSeverity.ERROR,
            code="CONFLICTING_SELECTION",
            suggestion="Use only one of: size (legacy), pick (combinations), or arrange (permutations)"
        ))

    # Check for unknown keys in pure OR node
    if set(spec.keys()).issubset(PURE_OR_KEYS):
        extra_keys = set(spec.keys()) - PURE_OR_KEYS
        if extra_keys:
            result.add_error(ValidationError(
                message=f"Unknown keys in OR node: {extra_keys}",
                path=path,
                severity=ValidationSeverity.WARNING if not strict else ValidationSeverity.ERROR,
                code="UNKNOWN_OR_KEYS"
            ))

    # Recursively validate choices
    for i, choice in enumerate(or_value):
        if isinstance(choice, (dict, list)):
            choice_result = validate_spec(choice, f"{path}.{OR_KEYWORD}[{i}]", strict)
            result.merge(choice_result)

    return result


def _validate_range_spec(spec: Dict[str, Any], path: str, strict: bool) -> ValidationResult:
    """Validate a _range_ specification node.

    Args:
        spec: Dictionary node containing _range_.
        path: Current path for error reporting.
        strict: Whether to treat warnings as errors.

    Returns:
        ValidationResult for this range node.
    """
    result = ValidationResult()
    range_value = spec[RANGE_KEYWORD]

    # Validate array syntax
    if isinstance(range_value, list):
        if len(range_value) not in (2, 3):
            result.add_error(ValidationError(
                message=f"Range array must have 2 or 3 elements, got {len(range_value)}",
                path=f"{path}.{RANGE_KEYWORD}",
                code="INVALID_RANGE_LENGTH"
            ))
        elif not all(isinstance(x, (int, float)) for x in range_value):
            result.add_error(ValidationError(
                message="Range array elements must be numeric",
                path=f"{path}.{RANGE_KEYWORD}",
                code="INVALID_RANGE_ELEMENTS"
            ))
        elif len(range_value) >= 2:
            start, end = range_value[0], range_value[1]
            step = range_value[2] if len(range_value) == 3 else 1

            if step == 0:
                result.add_error(ValidationError(
                    message="Range step cannot be zero",
                    path=f"{path}.{RANGE_KEYWORD}",
                    code="ZERO_STEP"
                ))
            elif (end < start and step > 0) or (end > start and step < 0):
                result.add_error(ValidationError(
                    message="Range will produce no values (step direction mismatch)",
                    path=f"{path}.{RANGE_KEYWORD}",
                    severity=ValidationSeverity.WARNING,
                    code="EMPTY_RANGE"
                ))

    # Validate dict syntax
    elif isinstance(range_value, dict):
        required = {"from", "to"}
        missing = required - set(range_value.keys())
        if missing:
            result.add_error(ValidationError(
                message=f"Range dict missing required keys: {missing}",
                path=f"{path}.{RANGE_KEYWORD}",
                code="MISSING_RANGE_KEYS"
            ))

        for key in ("from", "to", "step"):
            if key in range_value and not isinstance(range_value[key], (int, float)):
                result.add_error(ValidationError(
                    message=f"Range '{key}' must be numeric",
                    path=f"{path}.{RANGE_KEYWORD}.{key}",
                    code="INVALID_RANGE_VALUE"
                ))

    else:
        result.add_error(ValidationError(
            message=f"Range spec must be array or dict, got {type(range_value).__name__}",
            path=f"{path}.{RANGE_KEYWORD}",
            code="INVALID_RANGE_TYPE"
        ))

    # Validate count
    if COUNT_KEYWORD in spec:
        count = spec[COUNT_KEYWORD]
        if not isinstance(count, int):
            result.add_error(ValidationError(
                message=f"count must be an integer, got {type(count).__name__}",
                path=f"{path}.{COUNT_KEYWORD}",
                code="INVALID_COUNT_TYPE"
            ))
        elif count < 0:
            result.add_error(ValidationError(
                message=f"count must be non-negative, got {count}",
                path=f"{path}.{COUNT_KEYWORD}",
                code="NEGATIVE_COUNT"
            ))

    # Check for invalid keys in range node
    valid_range_keys = {RANGE_KEYWORD, COUNT_KEYWORD}
    extra_keys = set(spec.keys()) - valid_range_keys
    if extra_keys:
        # If pure range node has extra keys, it's an error
        if set(spec.keys()) <= {RANGE_KEYWORD, COUNT_KEYWORD}:
            pass  # Pure range, no extra keys
        else:
            # Mixed node - check if extra keys are valid
            for key in extra_keys:
                if key in PURE_OR_KEYS and key != OR_KEYWORD:
                    result.add_error(ValidationError(
                        message=f"OR modifier '{key}' not valid with _range_",
                        path=f"{path}.{key}",
                        code="INVALID_RANGE_MODIFIER"
                    ))

    return result


def _validate_size_spec(
    spec: Any,
    key_name: str,
    max_size: int,
    path: str
) -> ValidationResult:
    """Validate a size/pick/arrange specification.

    Args:
        spec: The size specification value.
        key_name: Name of the key (size/pick/arrange).
        max_size: Maximum valid size (length of _or_ list).
        path: Current path for error reporting.

    Returns:
        ValidationResult for this size spec.
    """
    result = ValidationResult()

    # Single integer
    if isinstance(spec, int):
        if spec < 0:
            result.add_error(ValidationError(
                message=f"{key_name} must be non-negative, got {spec}",
                path=path,
                code="NEGATIVE_SIZE"
            ))
        elif spec > max_size:
            result.add_error(ValidationError(
                message=f"{key_name}={spec} exceeds available choices ({max_size})",
                path=path,
                severity=ValidationSeverity.WARNING,
                code="SIZE_EXCEEDS_CHOICES"
            ))
        return result

    # Tuple or list (range or nested)
    if isinstance(spec, (tuple, list)):
        if len(spec) != 2:
            result.add_error(ValidationError(
                message=f"{key_name} tuple/list must have 2 elements, got {len(spec)}",
                path=path,
                code="INVALID_SIZE_LENGTH"
            ))
            return result

        # Check if it's a range (tuple) or nested [outer, inner] (list)
        if isinstance(spec, tuple):
            # Range specification (from, to)
            from_val, to_val = spec
            if not isinstance(from_val, int) or not isinstance(to_val, int):
                result.add_error(ValidationError(
                    message=f"{key_name} range must contain integers",
                    path=path,
                    code="INVALID_SIZE_RANGE_TYPE"
                ))
            elif from_val < 0 or to_val < 0:
                result.add_error(ValidationError(
                    message=f"{key_name} range values must be non-negative",
                    path=path,
                    code="NEGATIVE_SIZE_RANGE"
                ))
            elif from_val > to_val:
                result.add_error(ValidationError(
                    message=f"{key_name} range start ({from_val}) > end ({to_val})",
                    path=path,
                    code="INVERTED_SIZE_RANGE"
                ))
        else:
            # List could be nested [outer, inner] or range
            # Nested syntax validation
            for i, val in enumerate(spec):
                if not isinstance(val, int):
                    result.add_error(ValidationError(
                        message=f"{key_name}[{i}] must be an integer, got {type(val).__name__}",
                        path=f"{path}[{i}]",
                        code="INVALID_NESTED_SIZE"
                    ))
        return result

    # Invalid type
    result.add_error(ValidationError(
        message=f"{key_name} must be int, tuple, or list, got {type(spec).__name__}",
        path=path,
        code="INVALID_SIZE_TYPE"
    ))

    return result


# =============================================================================
# Configuration Validation (after expansion)
# =============================================================================

def validate_config(
    config: Any,
    schema: Optional[Dict[str, Any]] = None,
    required_keys: Optional[Set[str]] = None,
    forbidden_keys: Optional[Set[str]] = None,
    path: str = "root"
) -> ValidationResult:
    """Validate an expanded configuration.

    This validates configurations after expansion, checking for
    structural correctness and optionally against a schema.

    Args:
        config: The expanded configuration to validate.
        schema: Optional schema definition for validation.
        required_keys: Optional set of keys that must be present.
        forbidden_keys: Optional set of keys that must not be present.
        path: JSONPath-like location for error reporting.

    Returns:
        ValidationResult containing validation outcome.

    Examples:
        >>> config = {"class": "MyClass", "params": {"n": 5}}
        >>> result = validate_config(config, required_keys={"class"})
        >>> result.is_valid
        True
    """
    result = ValidationResult()
    result.node_count = 1

    if not isinstance(config, dict):
        # Non-dict configs are valid unless schema requires dict
        if schema and schema.get("type") == "object":
            result.add_error(ValidationError(
                message=f"Expected object, got {type(config).__name__}",
                path=path,
                code="TYPE_MISMATCH"
            ))
        return result

    # Check required keys
    if required_keys:
        missing = required_keys - set(config.keys())
        if missing:
            result.add_error(ValidationError(
                message=f"Missing required keys: {missing}",
                path=path,
                code="MISSING_REQUIRED_KEYS"
            ))

    # Check forbidden keys
    if forbidden_keys:
        present = forbidden_keys & set(config.keys())
        if present:
            result.add_error(ValidationError(
                message=f"Forbidden keys present: {present}",
                path=path,
                code="FORBIDDEN_KEYS_PRESENT"
            ))

    # Check for unexpanded generator keywords (should not be present after expansion)
    generator_keywords = {OR_KEYWORD, RANGE_KEYWORD}
    unexpanded = generator_keywords & set(config.keys())
    if unexpanded:
        result.add_error(ValidationError(
            message=f"Unexpanded generator keywords found: {unexpanded}",
            path=path,
            severity=ValidationSeverity.WARNING,
            code="UNEXPANDED_KEYWORDS",
            suggestion="Ensure expand_spec() was called on this configuration"
        ))

    # Schema validation if provided
    if schema:
        schema_result = _validate_against_schema(config, schema, path)
        result.merge(schema_result)

    return result


def validate_expanded_configs(
    configs: List[Any],
    schema: Optional[Dict[str, Any]] = None,
    min_count: int = 0,
    max_count: Optional[int] = None
) -> ValidationResult:
    """Validate a list of expanded configurations.

    Args:
        configs: List of expanded configurations.
        schema: Optional schema for each configuration.
        min_count: Minimum number of configurations required.
        max_count: Maximum number of configurations allowed.

    Returns:
        ValidationResult for the entire list.
    """
    result = ValidationResult()

    if not isinstance(configs, list):
        result.add_error(ValidationError(
            message=f"Expected list of configs, got {type(configs).__name__}",
            path="root",
            code="NOT_A_LIST"
        ))
        return result

    # Check count constraints
    if len(configs) < min_count:
        result.add_error(ValidationError(
            message=f"Too few configurations: {len(configs)} < {min_count}",
            path="root",
            code="TOO_FEW_CONFIGS"
        ))

    if max_count is not None and len(configs) > max_count:
        result.add_error(ValidationError(
            message=f"Too many configurations: {len(configs)} > {max_count}",
            path="root",
            code="TOO_MANY_CONFIGS"
        ))

    # Validate each configuration
    for i, config in enumerate(configs):
        config_result = validate_config(config, schema=schema, path=f"configs[{i}]")
        result.merge(config_result)

    return result


def _validate_against_schema(
    config: Dict[str, Any],
    schema: Dict[str, Any],
    path: str
) -> ValidationResult:
    """Validate config against schema definition.

    Simple schema validation supporting:
    - type: Expected type ("string", "number", "integer", "boolean", "array", "object")
    - required: List of required keys
    - properties: Dict of property schemas
    - items: Schema for array items

    Args:
        config: Configuration to validate.
        schema: Schema definition.
        path: Current path for error reporting.

    Returns:
        ValidationResult for schema validation.
    """
    result = ValidationResult()

    # Type check
    expected_type = schema.get("type")
    if expected_type:
        if not _check_type(config, expected_type):
            result.add_error(ValidationError(
                message=f"Type mismatch: expected {expected_type}, got {type(config).__name__}",
                path=path,
                code="SCHEMA_TYPE_MISMATCH"
            ))
            return result  # Don't continue if type is wrong

    # Required keys
    required = schema.get("required", [])
    if required and isinstance(config, dict):
        missing = set(required) - set(config.keys())
        if missing:
            result.add_error(ValidationError(
                message=f"Missing required properties: {missing}",
                path=path,
                code="SCHEMA_MISSING_REQUIRED"
            ))

    # Property validation
    properties = schema.get("properties", {})
    if properties and isinstance(config, dict):
        for key, prop_schema in properties.items():
            if key in config:
                prop_result = _validate_against_schema(
                    config[key], prop_schema, f"{path}.{key}"
                )
                result.merge(prop_result)

    # Array items validation
    items_schema = schema.get("items")
    if items_schema and isinstance(config, list):
        for i, item in enumerate(config):
            item_result = _validate_against_schema(
                item, items_schema, f"{path}[{i}]"
            )
            result.merge(item_result)

    return result


def _check_type(value: Any, expected: str) -> bool:
    """Check if value matches expected type string.

    Args:
        value: Value to check.
        expected: Type string ("string", "number", "integer", etc.)

    Returns:
        True if type matches, False otherwise.
    """
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    expected_types = type_map.get(expected)
    if expected_types is None:
        return True  # Unknown type, assume valid

    return isinstance(value, expected_types)
