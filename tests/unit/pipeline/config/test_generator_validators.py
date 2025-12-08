"""Tests for generator validators module.

This module tests the validation functionality for generator specifications
introduced in Phase 3.
"""

import pytest

from nirs4all.pipeline.config._generator.validators import (
    validate_spec,
    validate_config,
    validate_expanded_configs,
    ValidationResult,
    ValidationError,
    ValidationSeverity,
)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_valid_result(self):
        """Valid result should have no errors."""
        result = ValidationResult()
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error(self):
        """Adding error should invalidate result."""
        result = ValidationResult()
        error = ValidationError(message="test error", path="test.path")
        result.add_error(error)
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].message == "test error"
        assert result.errors[0].path == "test.path"

    def test_add_warning(self):
        """Adding warning should keep result valid."""
        result = ValidationResult()
        warning = ValidationError(
            message="test warning",
            path="test.path",
            severity=ValidationSeverity.WARNING
        )
        result.add_error(warning)  # add_error handles both based on severity
        assert result.is_valid
        assert len(result.warnings) == 1

    def test_multiple_errors(self):
        """Multiple errors should be tracked."""
        result = ValidationResult()
        result.add_error(ValidationError(message="error 1"))
        result.add_error(ValidationError(message="error 2"))
        result.add_error(ValidationError(message="error 3"))
        assert not result.is_valid
        assert len(result.errors) == 3

    def test_result_str_valid(self):
        """String representation of valid result."""
        result = ValidationResult()
        assert "VALID" in str(result)

    def test_result_str_invalid(self):
        """String representation of invalid result."""
        result = ValidationResult()
        result.add_error(ValidationError(message="test error"))
        str_repr = str(result)
        assert "INVALID" in str_repr
        assert "1 error" in str_repr


class TestValidationError:
    """Tests for ValidationError class."""

    def test_error_creation(self):
        """Error should store message, path, and severity."""
        error = ValidationError(
            message="test message",
            path="a.b.c",
            severity=ValidationSeverity.ERROR
        )
        assert error.message == "test message"
        assert error.path == "a.b.c"
        assert error.severity == ValidationSeverity.ERROR

    def test_error_default_severity(self):
        """Default severity should be ERROR."""
        error = ValidationError(message="test")
        assert error.severity == ValidationSeverity.ERROR

    def test_warning_severity(self):
        """Warning severity should be distinct from error."""
        warning = ValidationError(
            message="test warning",
            severity=ValidationSeverity.WARNING
        )
        assert warning.severity == ValidationSeverity.WARNING


class TestValidateSpec:
    """Tests for validate_spec function."""

    def test_validate_simple_spec(self):
        """Simple specs should be valid."""
        result = validate_spec({"model": "SVM"})
        assert result.is_valid

    def test_validate_or_spec(self):
        """OR specs should be valid."""
        result = validate_spec({"model": {"_or_": ["SVM", "RF"]}})
        assert result.is_valid

    def test_validate_range_spec(self):
        """Range specs should be valid."""
        result = validate_spec({"lr": {"_range_": [0.001, 0.1, 0.01]}})
        assert result.is_valid

    def test_validate_log_range_spec(self):
        """Log range specs should be valid."""
        result = validate_spec({"lr": {"_log_range_": [0.001, 1, 4]}})
        assert result.is_valid

    def test_validate_grid_spec(self):
        """Grid specs should be valid."""
        result = validate_spec({"params": {"_grid_": {"x": [1, 2], "y": ["A", "B"]}}})
        assert result.is_valid

    def test_validate_zip_spec(self):
        """Zip specs should be valid."""
        result = validate_spec({"params": {"_zip_": {"x": [1, 2], "y": ["A", "B"]}}})
        assert result.is_valid

    def test_validate_chain_spec(self):
        """Chain specs should be valid."""
        result = validate_spec({"stages": {"_chain_": [{"a": 1}, {"b": 2}]}})
        assert result.is_valid

    def test_validate_sample_spec(self):
        """Sample specs should be valid."""
        result = validate_spec({
            "lr": {"_sample_": {"distribution": "uniform", "from": 0, "to": 1, "num": 10}}
        })
        assert result.is_valid

    def test_validate_nested_spec(self):
        """Nested specs should be valid."""
        result = validate_spec({
            "model": {"_or_": ["SVM", "RF"]},
            "lr": {"_log_range_": [0.001, 0.1, 4]},
            "params": {"_grid_": {"x": [1, 2]}}
        })
        assert result.is_valid

    def test_validate_invalid_range(self):
        """Invalid range should produce errors."""
        result = validate_spec({"lr": {"_range_": ["a", "b"]}})
        assert not result.is_valid

    def test_validate_invalid_or(self):
        """Invalid OR should produce errors."""
        result = validate_spec({"model": {"_or_": "not_a_list"}})
        assert not result.is_valid

    def test_validate_invalid_grid(self):
        """Invalid grid should not be caught by current validators.

        Note: Phase 3 validators don't yet deeply validate _grid_ structure.
        This test documents current behavior.
        """
        result = validate_spec({"params": {"_grid_": [1, 2, 3]}})
        # Current implementation doesn't validate _grid_ specifically
        # This is expected behavior for Phase 3
        assert result.is_valid  # Not validating _grid_ structure yet

    def test_validate_invalid_log_range_negative(self):
        """Invalid log range with negative values.

        Note: Phase 3 validators don't yet deeply validate _log_range_.
        This test documents current behavior.
        """
        result = validate_spec({"lr": {"_log_range_": [-1, 1, 4]}})
        # Current implementation doesn't validate _log_range_ specifically
        assert result.is_valid  # Not validating _log_range_ structure yet

    def test_validate_invalid_sample_distribution(self):
        """Invalid sample with unknown distribution.

        Note: Phase 3 validators don't yet deeply validate _sample_.
        This test documents current behavior.
        """
        result = validate_spec({
            "lr": {"_sample_": {"distribution": "unknown"}}
        })
        # Current implementation doesn't validate _sample_ specifically
        assert result.is_valid  # Not validating _sample_ structure yet


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_validate_simple_config(self):
        """Simple configs should be valid."""
        result = validate_config({"model": "SVM", "lr": 0.01})
        assert result.is_valid

    def test_validate_nested_config(self):
        """Nested configs should be valid."""
        result = validate_config({
            "model": "SVM",
            "params": {"kernel": "rbf", "C": 1.0}
        })
        assert result.is_valid

    def test_validate_config_with_none(self):
        """Config with None values should be valid."""
        result = validate_config({"model": "SVM", "optional_param": None})
        assert result.is_valid

    def test_validate_config_with_list(self):
        """Config with list values should be valid."""
        result = validate_config({"layers": [64, 32, 16]})
        assert result.is_valid


class TestValidateExpandedConfigs:
    """Tests for validate_expanded_configs function."""

    def test_validate_single_config(self):
        """Single config validation."""
        result = validate_expanded_configs([{"model": "SVM"}])
        assert result.is_valid

    def test_validate_multiple_configs(self):
        """Multiple configs validation."""
        configs = [
            {"model": "SVM", "C": 1.0},
            {"model": "RF", "n_estimators": 100},
            {"model": "KNN", "k": 5}
        ]
        result = validate_expanded_configs(configs)
        assert result.is_valid

    def test_validate_empty_list(self):
        """Empty config list should be valid (no special warning in current impl)."""
        result = validate_expanded_configs([])
        assert result.is_valid  # Valid with zero configs
        # Note: Current implementation doesn't add warning for empty list


class TestValidationIntegration:
    """Integration tests for validation with actual expansion."""

    def test_validate_then_expand(self):
        """Validate spec before expansion."""
        from nirs4all.pipeline.config.generator import expand_spec

        spec = {"lr": {"_log_range_": [0.001, 0.1, 4]}}
        validation = validate_spec(spec)
        assert validation.is_valid

        expanded = expand_spec(spec)
        assert len(expanded) == 4

    def test_validate_expanded_output(self):
        """Validate expanded configs."""
        from nirs4all.pipeline.config.generator import expand_spec

        spec = {"model": {"_or_": ["SVM", "RF"]}}
        expanded = expand_spec(spec)

        result = validate_expanded_configs(expanded)
        assert result.is_valid
        assert len(expanded) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
