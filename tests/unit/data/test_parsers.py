"""
Tests for the parsers module (Phase 1).

Tests the parser classes and configuration normalization.
"""

import json
import pytest
import tempfile
from pathlib import Path

import yaml

from nirs4all.data.parsers import (
    BaseParser,
    ParserResult,
    LegacyParser,
    FilesParser,
    FolderParser,
    ConfigNormalizer,
    normalize_config,
)
from nirs4all.data.parsers.legacy_parser import normalize_config_keys


class TestNormalizeConfigKeys:
    """Test suite for key normalization."""

    def test_standard_keys_unchanged(self):
        """Test that standard keys pass through unchanged."""
        config = {
            "train_x": "X.csv",
            "train_y": "Y.csv",
            "test_x": "Xtest.csv"
        }
        result = normalize_config_keys(config)

        assert result["train_x"] == "X.csv"
        assert result["train_y"] == "Y.csv"
        assert result["test_x"] == "Xtest.csv"

    def test_train_x_variations(self):
        """Test various train_x naming conventions."""
        test_cases = [
            ("X_train", "train_x"),
            ("Xtrain", "train_x"),
            ("trainX", "train_x"),
            ("x_train", "train_x"),
            ("TRAIN_X", "train_x"),
        ]

        for input_key, expected_key in test_cases:
            config = {input_key: "file.csv"}
            result = normalize_config_keys(config)
            assert expected_key in result, f"Failed for {input_key}"

    def test_test_x_variations(self):
        """Test test/val X naming conventions."""
        test_cases = [
            ("X_test", "test_x"),
            ("Xtest", "test_x"),
            ("X_val", "test_x"),
            ("Xval", "test_x"),
            ("valX", "test_x"),
        ]

        for input_key, expected_key in test_cases:
            config = {input_key: "file.csv"}
            result = normalize_config_keys(config)
            assert expected_key in result, f"Failed for {input_key}"

    def test_metadata_variations(self):
        """Test metadata/group key normalization."""
        test_cases = [
            ("train_metadata", "train_group"),
            ("metadata_train", "train_group"),
            ("M_train", "train_group"),
            ("train_m", "train_group"),
            ("test_metadata", "test_group"),
            ("val_meta", "test_group"),
        ]

        for input_key, expected_key in test_cases:
            config = {input_key: "file.csv"}
            result = normalize_config_keys(config)
            assert expected_key in result, f"Failed for {input_key}"

    def test_unknown_keys_preserved(self):
        """Test that unknown keys are preserved."""
        config = {
            "train_x": "X.csv",
            "custom_key": "value",
            "signal_type": "absorbance"
        }
        result = normalize_config_keys(config)

        assert result["custom_key"] == "value"
        assert result["signal_type"] == "absorbance"


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


class TestLegacyParser:
    """Test suite for LegacyParser."""

    def test_can_parse_dict_with_train_x(self):
        """Test that parser recognizes train_x config."""
        parser = LegacyParser()
        config = {"train_x": "X.csv", "train_y": "Y.csv"}

        assert parser.can_parse(config) is True

    def test_can_parse_dict_with_test_x(self):
        """Test that parser recognizes test_x config."""
        parser = LegacyParser()
        config = {"test_x": "X.csv"}

        assert parser.can_parse(config) is True

    def test_can_parse_non_legacy_keys(self):
        """Test that parser normalizes and recognizes variant keys."""
        parser = LegacyParser()
        config = {"X_train": "X.csv"}  # Variant format

        assert parser.can_parse(config) is True

    def test_cannot_parse_non_dict(self):
        """Test that parser rejects non-dict input."""
        parser = LegacyParser()

        assert parser.can_parse("string") is False
        assert parser.can_parse(123) is False
        assert parser.can_parse(None) is False

    def test_cannot_parse_empty_dict(self):
        """Test that parser rejects dict without data keys."""
        parser = LegacyParser()
        config = {"task_type": "regression"}

        assert parser.can_parse(config) is False

    def test_parse_basic_config(self):
        """Test parsing basic configuration."""
        parser = LegacyParser()
        config = {
            "train_x": "data/X.csv",
            "train_y": "data/Y.csv"
        }

        result = parser.parse(config)

        assert result.success is True
        assert result.config["train_x"] == "data/X.csv"
        assert result.config["train_y"] == "data/Y.csv"
        assert "data" in result.dataset_name.lower()

    def test_parse_normalizes_keys(self):
        """Test that parsing normalizes keys."""
        parser = LegacyParser()
        config = {
            "X_train": "X.csv",
            "Y_train": "Y.csv"
        }

        result = parser.parse(config)

        assert result.success is True
        assert "train_x" in result.config
        assert "train_y" in result.config

    def test_parse_with_explicit_name(self):
        """Test that explicit name is used."""
        parser = LegacyParser()
        config = {
            "name": "my_dataset",
            "train_x": "X.csv"
        }

        result = parser.parse(config)

        assert result.dataset_name == "my_dataset"

    def test_parse_error_no_data(self):
        """Test error when no data source found."""
        parser = LegacyParser()
        config = {"task_type": "regression"}

        result = parser.parse(config)

        assert result.success is False
        assert any("train_x" in e or "test_x" in e for e in result.errors)


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
        assert result.config.train_x == "data.csv"

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
        assert result.config.train_x == "train_data.csv"
        assert result.config.test_x == "test_data.csv"

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
        assert result.config.train_x == ["train1.csv", "train2.csv"]

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
        assert result.config.train_x == "Xcal_data.csv"
        assert result.config.test_x == "Xval_data.csv"


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
        """Test normalizing dictionary."""
        normalizer = ConfigNormalizer()
        input_dict = {
            "X_train": "X.csv",  # Variant key
            "train_y": "Y.csv"
        }

        config, name = normalizer.normalize(input_dict)

        assert "train_x" in config  # Normalized key
        assert config["train_y"] == "Y.csv"

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
