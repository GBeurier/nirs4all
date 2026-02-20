"""
Tests for the parsers module (Phase 1).

Tests the parser classes and configuration normalization.
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from nirs4all.data.parsers import (
    BaseParser,
    ConfigNormalizer,
    FilesParser,
    FolderParser,
    ParserResult,
    normalize_config,
)


class TestParserResult:
    """Test suite for ParserResult."""

    def test_successful_result(self):
        """Test successful parser result."""
        result = ParserResult(
            success=True,
            config={"train_x": "X.csv"},
            dataset_name="test_dataset"
        )

        assert result.success is True
        assert result.config["train_x"] == "X.csv"
        assert result.dataset_name == "test_dataset"
        assert "success=True" in str(result)

    def test_failed_result(self):
        """Test failed parser result."""
        result = ParserResult(
            success=False,
            errors=["Missing train_x"]
        )

        assert result.success is False
        assert len(result.errors) == 1
        assert "success=False" in str(result)

class TestFilesParser:
    """Test suite for FilesParser (stub)."""

    def test_can_parse_files_syntax(self):
        """Test that parser recognizes files syntax."""
        parser = FilesParser()
        config = {
            "files": [{"path": "data.csv"}]
        }

        assert parser.can_parse(config) is True

    def test_cannot_parse_empty_files(self):
        """Test that parser rejects empty files list."""
        parser = FilesParser()
        config = {"files": []}

        assert parser.can_parse(config) is False

    def test_cannot_parse_no_files(self):
        """Test that parser rejects config without files."""
        parser = FilesParser()
        config = {"train_x": "X.csv"}

        assert parser.can_parse(config) is False

    def test_parse_returns_not_implemented(self):
        """Test that parsing files syntax now works."""
        parser = FilesParser()
        config = {"files": [{"path": "data.csv"}]}

        result = parser.parse(config)

        # Now implemented in Phase 4
        assert result.success is True
        assert result.source_type == "files"
        assert result.config is not None
        assert result.config["train_x"] == "data.csv"

    def test_parse_with_partition_assignment(self):
        """Test parsing files with partition assignment."""
        parser = FilesParser()
        config = {
            "name": "test_dataset",
            "files": [
                {"path": "train_data.csv", "partition": "train"},
                {"path": "test_data.csv", "partition": "test"},
            ]
        }

        result = parser.parse(config)

        assert result.success is True
        assert result.config["train_x"] == "train_data.csv"
        assert result.config["test_x"] == "test_data.csv"

    def test_parse_multiple_train_files(self):
        """Test parsing multiple files for same partition."""
        parser = FilesParser()
        config = {
            "files": [
                {"path": "train1.csv", "partition": "train"},
                {"path": "train2.csv", "partition": "train"},
            ]
        }

        result = parser.parse(config)

        assert result.success is True
        assert result.config["train_x"] == ["train1.csv", "train2.csv"]

    def test_parse_infers_partition_from_path(self):
        """Test that partition is inferred from filename."""
        parser = FilesParser()
        config = {
            "files": [
                {"path": "Xcal_data.csv"},  # Should infer train
                {"path": "Xval_data.csv"},  # Should infer test
            ]
        }

        result = parser.parse(config)

        assert result.success is True
        assert result.config["train_x"] == "Xcal_data.csv"
        assert result.config["test_x"] == "Xval_data.csv"

class TestFolderParser:
    """Test suite for FolderParser."""

    @pytest.fixture
    def sample_folder(self, tmp_path):
        """Create a sample folder with data files."""
        # Create sample files
        (tmp_path / "Xcal.csv").write_text("1,2,3\n4,5,6")
        (tmp_path / "Ycal.csv").write_text("1\n2")
        (tmp_path / "Xval.csv").write_text("7,8,9")
        (tmp_path / "Yval.csv").write_text("3")

        return tmp_path

    def test_can_parse_directory(self, sample_folder):
        """Test that parser recognizes directory path."""
        parser = FolderParser()

        assert parser.can_parse(str(sample_folder)) is True
        assert parser.can_parse(sample_folder) is True

    def test_can_parse_folder_dict(self, sample_folder):
        """Test that parser recognizes folder dict."""
        parser = FolderParser()
        config = {"folder": str(sample_folder)}

        assert parser.can_parse(config) is True

    def test_cannot_parse_file(self, tmp_path):
        """Test that parser rejects file path."""
        parser = FolderParser()
        file_path = tmp_path / "test.json"
        file_path.write_text("{}")

        assert parser.can_parse(str(file_path)) is False

    def test_cannot_parse_nonexistent(self):
        """Test that parser rejects nonexistent path."""
        parser = FolderParser()

        # Note: can_parse checks if directory exists
        assert parser.can_parse("/nonexistent/path") is False

    def test_parse_folder(self, sample_folder):
        """Test parsing a folder."""
        parser = FolderParser()

        result = parser.parse(str(sample_folder))

        assert result.success is True
        assert result.config["train_x"] is not None
        assert result.config["train_y"] is not None
        assert result.config["test_x"] is not None
        assert result.config["test_y"] is not None

    def test_parse_folder_dict(self, sample_folder):
        """Test parsing folder dict with params."""
        parser = FolderParser()
        config = {
            "folder": str(sample_folder),
            "global_params": {"delimiter": ";"}
        }

        result = parser.parse(config)

        assert result.success is True
        assert result.config["global_params"]["delimiter"] == ";"

    def test_parse_empty_folder(self, tmp_path):
        """Test parsing empty folder fails."""
        parser = FolderParser()

        result = parser.parse(str(tmp_path))

        assert result.success is False
        assert any("no data files" in e.lower() for e in result.errors)

    def test_parse_multi_source_detection(self, tmp_path):
        """Test that multiple matching files are detected."""
        parser = FolderParser()

        # Create multiple train_x files
        (tmp_path / "Xcal_NIR.csv").write_text("1,2,3")
        (tmp_path / "Xcal_MIR.csv").write_text("4,5,6")
        (tmp_path / "Ycal.csv").write_text("1")

        result = parser.parse(str(tmp_path))

        assert result.success is True
        # train_x should be a list (multi-source)
        assert isinstance(result.config["train_x"], list)
        assert len(result.config["train_x"]) == 2

class TestConfigNormalizer:
    """Test suite for ConfigNormalizer."""

    @pytest.fixture
    def sample_folder(self, tmp_path):
        """Create a sample folder with data files."""
        (tmp_path / "Xcal.csv").write_text("1,2,3")
        (tmp_path / "Ycal.csv").write_text("1")
        return tmp_path

    def test_normalize_folder_path(self, sample_folder):
        """Test normalizing folder path."""
        normalizer = ConfigNormalizer()

        config, name = normalizer.normalize(str(sample_folder))

        assert config is not None
        assert config["train_x"] is not None

    def test_normalize_json_file(self, tmp_path):
        """Test normalizing JSON file path."""
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps({
            "train_x": "X.csv",
            "train_y": "Y.csv"
        }))

        normalizer = ConfigNormalizer()
        config, name = normalizer.normalize(str(json_file))

        assert config["train_x"] == "X.csv"
        assert name == "config"

    def test_normalize_yaml_file(self, tmp_path):
        """Test normalizing YAML file path."""
        yaml_file = tmp_path / "dataset.yaml"
        yaml_file.write_text(yaml.dump({
            "train_x": "X.csv",
            "task_type": "regression"
        }))

        normalizer = ConfigNormalizer()
        config, name = normalizer.normalize(str(yaml_file))

        assert config["train_x"] == "X.csv"
        assert config["task_type"] == "regression"
        assert name == "dataset"

    def test_normalize_dict(self):
        """Test normalizing dictionary with canonical keys."""
        normalizer = ConfigNormalizer()
        input_dict = {
            "train_x": "X.csv",
            "train_y": "Y.csv"
        }

        config, name = normalizer.normalize(input_dict)

        assert config["train_x"] == "X.csv"
        assert config["train_y"] == "Y.csv"

    def test_normalize_dict_with_key_aliases(self):
        """Test normalization of accepted key aliases to canonical keys."""
        normalizer = ConfigNormalizer()
        input_dict = {
            "X_test": "Xval.csv",
            "train_m": "Mcal.csv",
            "test_m_params": {"has_header": True},
        }

        config, _ = normalizer.normalize(input_dict)

        assert config["test_x"] == "Xval.csv"
        assert config["train_group"] == "Mcal.csv"
        assert config["test_group_params"] == {"has_header": True}
        assert "X_test" not in config
        assert "train_m" not in config
        assert "test_m_params" not in config

    def test_normalize_dict_with_broad_aliases_for_core_keys(self):
        """Test broad alias coverage for all core dataset keys."""
        normalizer = ConfigNormalizer()
        input_dict = {
            "data_name": "alias_dataset",
            "desc": "Alias coverage",
            "ml_task": "regression",
            "cal_features": "Xcal.csv",
            "fit_targets": "Ycal.csv",
            "calibration_metadata": "Mcal.csv",
            "eval_spectra": "Xval.csv",
            "holdout_labels": "Yval.csv",
            "predict_meta": "Mval.csv",
            "cal_features_config": {"has_header": False},
            "holdout_labels_cols": [0],
            "global_loading_options": {"delimiter": ";"},
            "fit_settings": {"has_header": True},
            "eval_settings": {"has_header": True},
            "aggregation_column": "sample_id",
            "aggregation_strategy": "mean",
            "remove_outliers_before_aggregation": True,
            "repeat_column": "sample_id",
            "cv_splits": "folds.csv",
        }

        config, name = normalizer.normalize(input_dict)

        assert name == "alias_dataset"
        assert config["name"] == "alias_dataset"
        assert config["description"] == "Alias coverage"
        assert config["task_type"] == "regression"
        assert config["train_x"] == "Xcal.csv"
        assert config["train_y"] == "Ycal.csv"
        assert config["train_group"] == "Mcal.csv"
        assert config["test_x"] == "Xval.csv"
        assert config["test_y"] == "Yval.csv"
        assert config["test_group"] == "Mval.csv"
        assert config["train_x_params"] == {"has_header": False}
        assert config["test_y_filter"] == [0]
        assert config["global_params"] == {"delimiter": ";"}
        assert config["train_params"] == {"has_header": True}
        assert config["test_params"] == {"has_header": True}
        assert config["aggregate"] == "sample_id"
        assert config["aggregate_method"] == "mean"
        assert config["aggregate_exclude_outliers"] is True
        assert config["repetition"] == "sample_id"
        assert config["folds"] == "folds.csv"

    def test_normalize_dict_with_sources_aliases(self):
        """Test source syntax aliases map to parser-facing keys."""
        normalizer = ConfigNormalizer()
        input_dict = {
            "sensor_sources": [
                {"name": "NIR", "train_x": "nir_train.csv", "test_x": "nir_test.csv"}
            ],
            "target_spec": {"path": "targets.csv"},
            "metadata_spec": {"path": "metadata.csv"},
        }

        config, _ = normalizer.normalize(input_dict)

        assert config["train_x"] == "nir_train.csv"
        assert config["test_x"] == "nir_test.csv"
        assert config["train_y"] == "targets.csv"
        assert config["train_group"] == "metadata.csv"

    def test_normalize_dict_prefers_canonical_on_alias_conflict(self):
        """Test canonical keys are not overridden by aliases."""
        normalizer = ConfigNormalizer()
        input_dict = {
            "test_x": "canonical.csv",
            "X_test": "legacy.csv",
        }

        config, _ = normalizer.normalize(input_dict)
        assert config["test_x"] == "canonical.csv"

    def test_normalize_dict_with_folder(self, sample_folder):
        """Test normalizing dict with folder key."""
        normalizer = ConfigNormalizer()
        input_dict = {"folder": str(sample_folder)}

        config, name = normalizer.normalize(input_dict)

        assert config is not None
        assert config["train_x"] is not None

    def test_normalize_none(self):
        """Test normalizing None returns None."""
        normalizer = ConfigNormalizer()

        config, name = normalizer.normalize(None)

        assert config is None
        assert name == "Unknown_dataset"

    def test_normalize_path_object(self, sample_folder):
        """Test normalizing Path object."""
        normalizer = ConfigNormalizer()

        config, name = normalizer.normalize(sample_folder)

        assert config is not None

    def test_normalize_extracts_name_from_config(self):
        """Test that name is extracted from config."""
        normalizer = ConfigNormalizer()
        input_dict = {
            "name": "my_dataset",
            "train_x": "X.csv"
        }

        config, name = normalizer.normalize(input_dict)

        assert name == "my_dataset"

class TestNormalizeConfigFunction:
    """Test suite for normalize_config convenience function."""

    def test_normalize_dict(self):
        """Test normalizing a dict."""
        config, name = normalize_config({
            "train_x": "X.csv",
            "train_y": "Y.csv"
        })

        assert config["train_x"] == "X.csv"

    def test_normalize_empty_dict(self):
        """Test normalizing empty dict."""
        config, name = normalize_config({})

        # Empty config is normalized but won't have data
        assert config is not None or name == "Unknown_dataset"
