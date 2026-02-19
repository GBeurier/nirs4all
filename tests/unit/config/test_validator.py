"""Tests for configuration validation module.

Tests the JSON Schema validation for pipeline and dataset configurations.
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from nirs4all.config.validator import (
    DATASET_SCHEMA,
    PIPELINE_SCHEMA,
    ConfigValidationError,
    get_validation_summary,
    validate_config_file,
    validate_dataset_config,
    validate_pipeline_config,
)


class TestValidatePipelineConfig:
    """Test suite for pipeline configuration validation."""

    def test_valid_pipeline_dict(self):
        """Test validation of a valid pipeline config dict."""
        config = {
            "pipeline": [
                {"class": "sklearn.preprocessing.MinMaxScaler"},
                {"model": {"class": "sklearn.linear_model.LinearRegression"}}
            ]
        }

        is_valid, errors, warnings = validate_pipeline_config(config)

        assert is_valid is True
        assert len(errors) == 0

    def test_valid_pipeline_json_file(self, tmp_path):
        """Test validation of a valid pipeline JSON file."""
        config = {
            "pipeline": [
                {"class": "sklearn.preprocessing.StandardScaler"},
                {
                    "class": "sklearn.model_selection.ShuffleSplit",
                    "params": {"n_splits": 5, "test_size": 0.25}
                },
                {"model": {"class": "sklearn.cross_decomposition.PLSRegression"}}
            ]
        }

        config_file = tmp_path / "pipeline.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)

        is_valid, errors, warnings = validate_pipeline_config(str(config_file))

        assert is_valid is True
        assert len(errors) == 0

    def test_valid_pipeline_yaml_file(self, tmp_path):
        """Test validation of a valid pipeline YAML file."""
        yaml_content = """
pipeline:
  - class: sklearn.preprocessing.MinMaxScaler
    params:
      feature_range: [0, 1]

  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 10
"""
        config_file = tmp_path / "pipeline.yaml"
        config_file.write_text(yaml_content)

        is_valid, errors, warnings = validate_pipeline_config(str(config_file))

        assert is_valid is True
        assert len(errors) == 0

    def test_missing_pipeline_key(self):
        """Test that missing 'pipeline' key is an error."""
        config = {
            "steps": [{"class": "sklearn.preprocessing.MinMaxScaler"}]
        }

        is_valid, errors, warnings = validate_pipeline_config(config)

        assert is_valid is False
        assert any("pipeline" in e.lower() for e in errors)

    def test_pipeline_not_a_list(self):
        """Test that pipeline must be a list."""
        config = {
            "pipeline": {"step1": {"class": "sklearn.preprocessing.MinMaxScaler"}}
        }

        is_valid, errors, warnings = validate_pipeline_config(config)

        assert is_valid is False
        assert any("list" in e.lower() or "array" in e.lower() for e in errors)

    def test_empty_pipeline_list(self):
        """Test that empty pipeline list is an error."""
        config = {"pipeline": []}

        is_valid, errors, warnings = validate_pipeline_config(config)

        assert is_valid is False

    def test_file_not_found(self, tmp_path):
        """Test error when config file doesn't exist."""
        is_valid, errors, warnings = validate_pipeline_config(
            str(tmp_path / "nonexistent.json")
        )

        assert is_valid is False
        assert any("not found" in e.lower() for e in errors)

    def test_invalid_json_syntax(self, tmp_path):
        """Test error for invalid JSON syntax."""
        config_file = tmp_path / "bad.json"
        config_file.write_text('{"pipeline": [}')

        is_valid, errors, warnings = validate_pipeline_config(str(config_file))

        assert is_valid is False
        assert any("json" in e.lower() or "line" in e.lower() for e in errors)

    def test_null_step_warning(self):
        """Test that null steps generate warnings or are caught by validation."""
        config = {
            "pipeline": [
                {"class": "sklearn.preprocessing.MinMaxScaler"},
                None,
                {"model": {"class": "sklearn.linear_model.LinearRegression"}}
            ]
        }

        is_valid, errors, warnings = validate_pipeline_config(config)

        # Null steps may be caught as errors by schema validation
        # or as warnings - either is acceptable
        has_feedback = len(warnings) > 0 or len(errors) > 0
        assert has_feedback, "Null step should generate a warning or error"

class TestValidateDatasetConfig:
    """Test suite for dataset configuration validation."""

    def test_valid_dataset_dict(self):
        """Test validation of a valid dataset config dict."""
        config = {
            "train_x": "path/to/X.csv",
            "train_y": "path/to/Y.csv",
            "task_type": "regression"
        }

        is_valid, errors, warnings = validate_dataset_config(config, check_files=False)

        assert is_valid is True
        assert len(errors) == 0

    def test_valid_test_only_config(self):
        """Test that test_x only is valid (for prediction scenarios)."""
        config = {
            "test_x": "path/to/Xtest.csv"
        }

        is_valid, errors, warnings = validate_dataset_config(config, check_files=False)

        assert is_valid is True

    def test_valid_folder_config(self):
        """Test that folder config is valid."""
        config = {
            "folder": "path/to/data"
        }

        is_valid, errors, warnings = validate_dataset_config(config, check_files=False)

        assert is_valid is True

    def test_valid_dataset_json_file(self, tmp_path):
        """Test validation of a valid dataset JSON file."""
        config = {
            "train_x": "path/to/Xcal.csv",
            "train_y": "path/to/Ycal.csv",
            "test_x": "path/to/Xval.csv",
            "test_y": "path/to/Yval.csv",
            "task_type": "regression",
            "signal_type": "absorbance"
        }

        config_file = tmp_path / "dataset.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)

        is_valid, errors, warnings = validate_dataset_config(
            str(config_file), check_files=False
        )

        assert is_valid is True

    def test_valid_dataset_yaml_file(self, tmp_path):
        """Test validation of a valid dataset YAML file."""
        yaml_content = """
train_x: path/to/Xcal.csv
train_y: path/to/Ycal.csv
test_x: path/to/Xval.csv
test_y: path/to/Yval.csv

task_type: regression
signal_type: absorbance
aggregate: sample_id

global_params:
  header_unit: nm
  delimiter: ","
"""
        config_file = tmp_path / "dataset.yaml"
        config_file.write_text(yaml_content)

        is_valid, errors, warnings = validate_dataset_config(
            str(config_file), check_files=False
        )

        assert is_valid is True

    def test_missing_data_source(self):
        """Test that missing data source is an error."""
        config = {
            "task_type": "regression"
        }

        is_valid, errors, warnings = validate_dataset_config(config, check_files=False)

        assert is_valid is False
        assert len(errors) > 0  # Should have at least one error about missing data

    def test_invalid_task_type(self):
        """Test that invalid task_type is an error."""
        config = {
            "train_x": "path/to/X.csv",
            "task_type": "invalid_task"
        }

        is_valid, errors, warnings = validate_dataset_config(config, check_files=False)

        assert is_valid is False
        assert any("task_type" in e.lower() for e in errors)

    def test_invalid_signal_type(self):
        """Test that invalid signal_type is an error."""
        config = {
            "train_x": "path/to/X.csv",
            "signal_type": "invalid_signal"
        }

        is_valid, errors, warnings = validate_dataset_config(config, check_files=False)

        assert is_valid is False
        assert any("signal_type" in e.lower() for e in errors)

    def test_file_check_warnings(self, tmp_path):
        """Test that missing data files generate warnings."""
        config = {
            "train_x": str(tmp_path / "nonexistent.csv"),
            "train_y": str(tmp_path / "also_missing.csv")
        }

        is_valid, errors, warnings = validate_dataset_config(config, check_files=True)

        # Config is structurally valid, but files are missing
        assert is_valid is True
        assert len(warnings) >= 2
        assert any("not found" in w.lower() for w in warnings)

class TestValidateConfigFile:
    """Test suite for auto-detecting config type validation."""

    def test_auto_detect_pipeline(self, tmp_path):
        """Test auto-detection of pipeline config."""
        config = {"pipeline": [{"class": "sklearn.preprocessing.MinMaxScaler"}]}

        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)

        is_valid, errors, warnings = validate_config_file(str(config_file))

        assert is_valid is True

    def test_auto_detect_dataset(self, tmp_path):
        """Test auto-detection of dataset config."""
        config = {"train_x": "path/to/X.csv"}

        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        is_valid, errors, warnings = validate_config_file(
            str(config_file), check_files=False
        )

        assert is_valid is True

    def test_explicit_pipeline_type(self, tmp_path):
        """Test explicit pipeline type specification."""
        config = {"pipeline": [{"class": "sklearn.preprocessing.MinMaxScaler"}]}

        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)

        is_valid, errors, warnings = validate_config_file(
            str(config_file), config_type='pipeline'
        )

        assert is_valid is True

    def test_explicit_dataset_type(self, tmp_path):
        """Test explicit dataset type specification."""
        config = {"train_x": "path/to/X.csv"}

        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)

        is_valid, errors, warnings = validate_config_file(
            str(config_file), config_type='dataset', check_files=False
        )

        assert is_valid is True

    def test_unknown_config_type(self, tmp_path):
        """Test error when config type cannot be determined."""
        config = {"some_unknown_key": "value"}

        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)

        is_valid, errors, warnings = validate_config_file(str(config_file))

        assert is_valid is False
        assert any("cannot determine" in e.lower() for e in errors)

class TestGetValidationSummary:
    """Test suite for validation summary formatting."""

    def test_valid_summary(self):
        """Test summary for valid config."""
        summary = get_validation_summary(True, [], [], "config.yaml")

        assert "valid" in summary.lower()
        assert "config.yaml" in summary

    def test_invalid_summary_with_errors(self):
        """Test summary with errors."""
        errors = ["Missing 'pipeline' key", "Invalid step format"]
        summary = get_validation_summary(False, errors, [], "pipeline.json")

        assert "errors" in summary.lower()
        assert "Missing 'pipeline' key" in summary
        assert "Invalid step format" in summary

    def test_summary_with_warnings(self):
        """Test summary with warnings."""
        warnings = ["File not found: data.csv"]
        summary = get_validation_summary(True, [], warnings)

        assert "valid" in summary.lower()
        assert "warnings" in summary.lower()
        assert "File not found" in summary

class TestConfigValidationError:
    """Test suite for ConfigValidationError exception."""

    def test_error_with_single_message(self):
        """Test error with single error message."""
        error = ConfigValidationError(["Missing required field"])

        assert "Missing required field" in str(error)

    def test_error_with_multiple_messages(self):
        """Test error with multiple error messages."""
        error = ConfigValidationError(
            ["Error 1", "Error 2"],
            config_path="config.yaml"
        )

        assert "Error 1" in str(error)
        assert "Error 2" in str(error)
        assert "config.yaml" in str(error)

    def test_error_attributes(self):
        """Test error attributes."""
        error = ConfigValidationError(
            ["Error message"],
            config_path="/path/to/config.json"
        )

        assert error.errors == ["Error message"]
        assert error.config_path == "/path/to/config.json"
