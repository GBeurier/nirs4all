"""
Tests for the schema module (Phase 1).

Tests the Pydantic-based schema models, validation, and normalization.
"""

import pytest
import numpy as np
from pathlib import Path

from nirs4all.data.schema import (
    DatasetConfigSchema,
    FileConfig,
    ColumnConfig,
    PartitionConfig,
    LoadingParams,
    TaskType,
    HeaderUnit,
    SignalTypeEnum,
    NAPolicy,
    AggregateMethod,
    ConfigValidator,
    ValidationResult,
    ValidationError,
    ValidationWarning,
)


class TestLoadingParams:
    """Test suite for LoadingParams model."""

    def test_default_values(self):
        """Test that LoadingParams has None defaults."""
        params = LoadingParams()

        assert params.delimiter is None
        assert params.decimal_separator is None
        assert params.has_header is None
        assert params.header_unit is None
        assert params.signal_type is None

    def test_explicit_values(self):
        """Test setting explicit values."""
        params = LoadingParams(
            delimiter=";",
            decimal_separator=".",
            has_header=True,
            header_unit="nm",
            signal_type="absorbance"
        )

        assert params.delimiter == ";"
        assert params.decimal_separator == "."
        assert params.has_header is True
        assert params.header_unit == HeaderUnit.WAVELENGTH
        assert params.signal_type == SignalTypeEnum.ABSORBANCE

    def test_header_unit_normalization(self):
        """Test that header_unit is normalized to enum."""
        params = LoadingParams(header_unit="cm-1")
        assert params.header_unit == HeaderUnit.WAVENUMBER

        params = LoadingParams(header_unit="NM")  # uppercase
        assert params.header_unit == HeaderUnit.WAVELENGTH

    def test_signal_type_normalization(self):
        """Test that signal_type is normalized to enum."""
        params = LoadingParams(signal_type="reflectance%")
        assert params.signal_type == SignalTypeEnum.REFLECTANCE_PERCENT

    def test_merge_with_none(self):
        """Test merging with None returns self."""
        params = LoadingParams(delimiter=";")
        merged = params.merge_with(None)

        assert merged.delimiter == ";"

    def test_merge_with_other(self):
        """Test merging two LoadingParams."""
        base = LoadingParams(delimiter=",", has_header=False)
        override = LoadingParams(delimiter=";", signal_type="absorbance")

        merged = override.merge_with(base)

        # Override takes precedence
        assert merged.delimiter == ";"
        assert merged.has_header is False  # From base
        assert merged.signal_type == SignalTypeEnum.ABSORBANCE  # From override

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed for forward compatibility."""
        params = LoadingParams(delimiter=";", custom_field="value")
        assert params.delimiter == ";"
        # Extra field should be accessible via model_dump
        data = params.model_dump()
        assert data.get("custom_field") == "value"


class TestDatasetConfigSchema:
    """Test suite for DatasetConfigSchema model."""

    def test_minimal_config(self):
        """Test minimal configuration with train_x."""
        config = DatasetConfigSchema(train_x="path/to/X.csv")

        assert config.train_x == "path/to/X.csv"
        assert config.is_legacy_format() is True
        assert config.is_files_format() is False

    def test_full_legacy_config(self):
        """Test full legacy format configuration."""
        config = DatasetConfigSchema(
            name="wheat_protein",
            train_x="data/Xcal.csv",
            train_y="data/Ycal.csv",
            train_group="data/Mcal.csv",
            test_x="data/Xval.csv",
            test_y="data/Yval.csv",
            test_group="data/Mval.csv",
            task_type="regression",
            global_params={"delimiter": ";", "has_header": True}
        )

        assert config.name == "wheat_protein"
        assert config.train_x == "data/Xcal.csv"
        assert config.task_type == TaskType.REGRESSION
        assert config.global_params.delimiter == ";"
        assert config.global_params.has_header is True

    def test_task_type_normalization(self):
        """Test that task_type strings are normalized to enum."""
        config = DatasetConfigSchema(
            train_x="X.csv",
            task_type="REGRESSION"  # uppercase
        )
        assert config.task_type == TaskType.REGRESSION

    def test_invalid_task_type_raises(self):
        """Test that invalid task_type raises error."""
        with pytest.raises(ValueError, match="Invalid task_type"):
            DatasetConfigSchema(train_x="X.csv", task_type="invalid_type")

    def test_aggregate_method_normalization(self):
        """Test that aggregate_method is normalized."""
        config = DatasetConfigSchema(
            train_x="X.csv",
            aggregate="sample_id",
            aggregate_method="MEDIAN"
        )
        assert config.aggregate_method == AggregateMethod.MEDIAN

    def test_multi_source_detection(self):
        """Test multi-source detection."""
        # Single source
        config = DatasetConfigSchema(train_x="X.csv")
        assert config.is_multi_source() is False

        # Multi-source
        config = DatasetConfigSchema(train_x=["X1.csv", "X2.csv"])
        assert config.is_multi_source() is True

    def test_get_effective_params(self):
        """Test parameter merging for specific files."""
        config = DatasetConfigSchema(
            train_x="X.csv",
            global_params={"delimiter": ",", "has_header": True},
            train_params={"decimal_separator": "."},
            train_x_params={"header_unit": "nm"}
        )

        params = config.get_effective_params("train", "x")

        assert params.delimiter == ","  # From global
        assert params.decimal_separator == "."  # From train
        assert params.header_unit == HeaderUnit.WAVELENGTH  # From train_x

    def test_to_dict_excludes_none(self):
        """Test that to_dict excludes None values."""
        config = DatasetConfigSchema(
            train_x="X.csv",
            task_type="regression"
        )

        data = config.to_dict()

        assert "train_x" in data
        assert "task_type" in data
        assert "train_y" not in data  # Was None
        assert "test_x" not in data  # Was None

    def test_from_dict(self):
        """Test creating schema from dict."""
        data = {
            "train_x": "X.csv",
            "train_y": "Y.csv",
            "task_type": "regression"
        }

        config = DatasetConfigSchema.from_dict(data)

        assert config.train_x == "X.csv"
        assert config.train_y == "Y.csv"
        assert config.task_type == TaskType.REGRESSION

    def test_loading_params_as_dict(self):
        """Test that dict values are converted to LoadingParams."""
        config = DatasetConfigSchema(
            train_x="X.csv",
            global_params={"delimiter": ";"}
        )

        assert isinstance(config.global_params, LoadingParams)
        assert config.global_params.delimiter == ";"


class TestFileConfig:
    """Test suite for FileConfig model (stub)."""

    def test_basic_file_config(self):
        """Test basic FileConfig creation."""
        config = FileConfig(path="data/file.csv")
        assert config.path == "data/file.csv"
        assert config.partition is None

    def test_file_config_with_partition(self):
        """Test FileConfig with partition."""
        from nirs4all.data.schema import PartitionType

        config = FileConfig(
            path="data/train.csv",
            partition=PartitionType.TRAIN
        )
        assert config.partition == PartitionType.TRAIN


class TestConfigValidator:
    """Test suite for ConfigValidator."""

    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = {
            "train_x": "path/to/X.csv",
            "train_y": "path/to/Y.csv"
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_missing_data_source(self):
        """Test that missing data source is detected."""
        config = {
            "task_type": "regression"
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.is_valid is False
        assert any(e.code == "NO_DATA_SOURCE" for e in result.errors)

    def test_invalid_task_type(self):
        """Test that invalid task_type is detected."""
        config = {
            "train_x": "X.csv",
            "task_type": "invalid_type"
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.is_valid is False
        assert any(e.code == "INVALID_TASK_TYPE" for e in result.errors)

    def test_invalid_header_unit(self):
        """Test that invalid header_unit is detected."""
        config = {
            "train_x": "X.csv",
            "global_params": {"header_unit": "invalid_unit"}
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.is_valid is False
        assert any(e.code == "INVALID_HEADER_UNIT" for e in result.errors)

    def test_invalid_na_policy(self):
        """Test that invalid na_policy is detected."""
        config = {
            "train_x": "X.csv",
            "global_params": {"na_policy": "invalid_policy"}
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.is_valid is False
        assert any(e.code == "INVALID_NA_POLICY" for e in result.errors)

    def test_invalid_aggregate_method(self):
        """Test that invalid aggregate_method is detected."""
        config = {
            "train_x": "X.csv",
            "aggregate": "sample_id",
            "aggregate_method": "invalid_method"
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.is_valid is False
        assert any(e.code == "INVALID_AGGREGATE_METHOD" for e in result.errors)

    def test_unused_aggregate_method_warning(self):
        """Test warning when aggregate_method without aggregate."""
        config = {
            "train_x": "X.csv",
            "aggregate_method": "mean"
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        # Should still be valid
        assert result.is_valid is True
        # But should have warning
        assert any(w.code == "UNUSED_AGGREGATE_METHOD" for w in result.warnings)

    def test_file_existence_check_disabled_by_default(self):
        """Test that file existence is not checked by default."""
        config = {
            "train_x": "nonexistent/path/X.csv"
        }

        validator = ConfigValidator(check_file_existence=False)
        result = validator.validate(config)

        assert result.is_valid is True
        assert not any(w.code == "FILE_NOT_FOUND" for w in result.warnings)

    def test_file_existence_check_enabled(self):
        """Test that file existence is checked when enabled."""
        config = {
            "train_x": "nonexistent/path/X.csv"
        }

        validator = ConfigValidator(check_file_existence=True)
        result = validator.validate(config)

        # Still valid (file not found is a warning, not error)
        assert result.is_valid is True
        assert any(w.code == "FILE_NOT_FOUND" for w in result.warnings)

    def test_mixed_format_warning(self):
        """Test warning when mixing train_x/test_x and files/sources formats."""
        config = {
            "train_x": "X.csv",
            "files": [{"path": "other.csv"}]
        }

        validator = ConfigValidator()
        result = validator.validate(config)

        # Should have warning about mixed format
        mixed_warnings = [w for w in result.warnings if w.code == "MIXED_FORMAT"]
        assert len(mixed_warnings) == 1
        assert "Use one format consistently" in mixed_warnings[0].message

    def test_validation_result_raise_if_invalid(self):
        """Test that raise_if_invalid raises for invalid config."""
        result = ValidationResult(
            is_valid=False,
            errors=[ValidationError(code="TEST", message="Test error")]
        )

        with pytest.raises(ValueError, match="Invalid configuration"):
            result.raise_if_invalid()

    def test_validation_result_raise_if_valid(self):
        """Test that raise_if_invalid doesn't raise for valid config."""
        result = ValidationResult(is_valid=True)
        result.raise_if_invalid()  # Should not raise


class TestValidationError:
    """Test suite for ValidationError."""

    def test_error_str(self):
        """Test string representation of error."""
        error = ValidationError(
            code="TEST_ERROR",
            message="Something went wrong",
            field="train_x",
            suggestion="Fix it"
        )

        error_str = str(error)
        assert "[train_x]" in error_str
        assert "Something went wrong" in error_str
        assert "Fix it" in error_str

    def test_error_str_no_field(self):
        """Test string representation without field."""
        error = ValidationError(
            code="TEST_ERROR",
            message="Something went wrong"
        )

        error_str = str(error)
        assert "Something went wrong" in error_str


class TestValidationWarning:
    """Test suite for ValidationWarning."""

    def test_warning_str(self):
        """Test string representation of warning."""
        warning = ValidationWarning(
            code="TEST_WARNING",
            message="This might be an issue",
            field="global_params"
        )

        warning_str = str(warning)
        assert "[global_params]" in warning_str
        assert "This might be an issue" in warning_str


class TestSchemaIntegration:
    """Integration tests for schema with real data structures."""

    def test_numpy_array_in_config(self):
        """Test that numpy arrays work in config."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        config = DatasetConfigSchema(
            train_x=X,
            train_y=y,
            task_type="regression"
        )

        assert isinstance(config.train_x, np.ndarray)
        assert config.train_x.shape == (100, 50)

    def test_list_of_paths(self):
        """Test multi-source paths in config."""
        config = DatasetConfigSchema(
            train_x=["X1.csv", "X2.csv", "X3.csv"],
            train_y="Y.csv"
        )

        assert isinstance(config.train_x, list)
        assert len(config.train_x) == 3
        assert config.is_multi_source() is True

    def test_path_objects(self):
        """Test that Path objects work in config."""
        config = DatasetConfigSchema(
            train_x=Path("data/X.csv"),
            train_y=Path("data/Y.csv")
        )

        # Pydantic might convert to string, but should work
        assert config.train_x is not None
        assert config.train_y is not None


class TestNAPolicyEnum:
    """Test suite for NAPolicy enum — canonical values and structure."""

    def test_exactly_six_members(self):
        """NAPolicy enum has exactly 6 members."""
        assert len(NAPolicy) == 6

    def test_canonical_values(self):
        """NAPolicy enum contains all canonical values."""
        expected_values = {"auto", "abort", "remove_sample", "remove_feature", "replace", "ignore"}
        actual_values = {member.value for member in NAPolicy}
        assert actual_values == expected_values

    def test_no_old_remove_member(self):
        """NAPolicy does NOT contain the old 'remove' member."""
        values = {member.value for member in NAPolicy}
        assert "remove" not in values

    def test_no_old_drop_member(self):
        """NAPolicy does NOT contain the old 'drop' member."""
        values = {member.value for member in NAPolicy}
        assert "drop" not in values

    def test_no_old_keep_member(self):
        """NAPolicy does NOT contain the old 'keep' member."""
        values = {member.value for member in NAPolicy}
        assert "keep" not in values

    def test_no_old_fill_members(self):
        """NAPolicy does NOT contain old 'fill_mean', 'fill_median', 'fill_zero'."""
        values = {member.value for member in NAPolicy}
        assert "fill_mean" not in values
        assert "fill_median" not in values
        assert "fill_zero" not in values

    def test_string_comparison(self):
        """NAPolicy members work as strings."""
        assert NAPolicy.AUTO == "auto"
        assert NAPolicy.ABORT == "abort"
        assert NAPolicy.REMOVE_SAMPLE == "remove_sample"
        assert NAPolicy.REMOVE_FEATURE == "remove_feature"
        assert NAPolicy.REPLACE == "replace"
        assert NAPolicy.IGNORE == "ignore"


class TestNAFillConfig:
    """Test suite for NAFillConfig model."""

    def test_default_values(self):
        """NAFillConfig defaults to method='value', fill_value=0.0, per_column=True."""
        from nirs4all.data.schema.config import NAFillConfig, NAFillMethod

        config = NAFillConfig()
        assert config.method == NAFillMethod.VALUE
        assert config.fill_value == 0.0
        assert config.per_column is True

    def test_custom_values(self):
        """NAFillConfig accepts custom values."""
        from nirs4all.data.schema.config import NAFillConfig, NAFillMethod

        config = NAFillConfig(method=NAFillMethod.MEAN, fill_value=99.0, per_column=False)
        assert config.method == NAFillMethod.MEAN
        assert config.fill_value == 99.0
        assert config.per_column is False


class TestNAError:
    """Test suite for NAError exception."""

    def test_importable_from_exceptions(self):
        """NAError is importable from nirs4all.core.exceptions."""
        from nirs4all.core.exceptions import NAError as ImportedNAError
        assert ImportedNAError is not None

    def test_is_exception_subclass(self):
        """NAError is a subclass of Exception."""
        from nirs4all.core.exceptions import NAError as ImportedNAError
        assert issubclass(ImportedNAError, Exception)

    def test_can_be_raised_and_caught(self):
        """NAError can be raised and caught."""
        from nirs4all.core.exceptions import NAError as ImportedNAError
        with pytest.raises(ImportedNAError, match="test message"):
            raise ImportedNAError("test message")
