"""Tests for dataset configuration file loading (JSON/YAML support).

Tests the ability to load DatasetConfigs from JSON and YAML files,
as well as error handling for invalid files.
"""

import json
import pytest
import tempfile
from pathlib import Path
import yaml

from nirs4all.data.config import DatasetConfigs
from nirs4all.data.config_parser import parse_config, _load_config_from_file


class TestLoadConfigFromFile:
    """Test suite for _load_config_from_file function."""

    def test_load_json_file(self, tmp_path):
        """Test loading a valid JSON config file."""
        config_data = {
            "train_x": "path/to/X.csv",
            "train_y": "path/to/Y.csv",
            "task_type": "regression"
        }

        json_file = tmp_path / "dataset.json"
        with open(json_file, 'w') as f:
            json.dump(config_data, f)

        config, name = _load_config_from_file(str(json_file))

        assert config["train_x"] == "path/to/X.csv"
        assert config["train_y"] == "path/to/Y.csv"
        assert config["task_type"] == "regression"
        assert name == "dataset"  # Uses file stem

    def test_load_yaml_file(self, tmp_path):
        """Test loading a valid YAML config file."""
        config_data = {
            "train_x": "path/to/X.csv",
            "train_y": "path/to/Y.csv",
            "signal_type": "absorbance"
        }

        yaml_file = tmp_path / "my_dataset.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(config_data, f)

        config, name = _load_config_from_file(str(yaml_file))

        assert config["train_x"] == "path/to/X.csv"
        assert config["train_y"] == "path/to/Y.csv"
        assert config["signal_type"] == "absorbance"
        assert name == "my_dataset"

    def test_load_yml_file(self, tmp_path):
        """Test loading a valid .yml config file."""
        config_data = {"test_x": "data/test.csv"}

        yml_file = tmp_path / "config.yml"
        with open(yml_file, 'w') as f:
            yaml.dump(config_data, f)

        config, name = _load_config_from_file(str(yml_file))
        assert config["test_x"] == "data/test.csv"
        assert name == "config"

    def test_name_from_config(self, tmp_path):
        """Test that 'name' key in config overrides file stem."""
        config_data = {
            "name": "wheat_protein",
            "train_x": "path/to/X.csv"
        }

        json_file = tmp_path / "dataset.json"
        with open(json_file, 'w') as f:
            json.dump(config_data, f)

        config, name = _load_config_from_file(str(json_file))

        assert name == "wheat_protein"
        assert config["name"] == "wheat_protein"

    def test_keys_preserved(self, tmp_path):
        """Test that config keys are preserved as-is."""
        config_data = {
            "train_x": "path/to/X.csv",
            "train_y": "path/to/Y.csv",
            "test_x": "path/to/Xtest.csv"
        }

        yaml_file = tmp_path / "dataset.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(config_data, f)

        config, _ = _load_config_from_file(str(yaml_file))

        assert config["train_x"] == "path/to/X.csv"
        assert config["train_y"] == "path/to/Y.csv"
        assert config["test_x"] == "path/to/Xtest.csv"

    def test_file_not_found(self, tmp_path):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="not found"):
            _load_config_from_file(str(tmp_path / "nonexistent.json"))

    def test_path_is_directory(self, tmp_path):
        """Test error when path is a directory, not a file."""
        subdir = tmp_path / "configs.json"  # Looks like a file but is a dir
        subdir.mkdir()

        with pytest.raises(ValueError, match="not a file"):
            _load_config_from_file(str(subdir))

    def test_empty_json_file(self, tmp_path):
        """Test error for empty JSON file."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("")

        with pytest.raises(ValueError, match="empty"):
            _load_config_from_file(str(json_file))

    def test_empty_yaml_file(self, tmp_path):
        """Test error for empty YAML file."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        with pytest.raises(ValueError, match="empty"):
            _load_config_from_file(str(yaml_file))

    def test_invalid_json_syntax(self, tmp_path):
        """Test error for invalid JSON with line number."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text('{\n  "key": "value"\n  "missing_comma": true\n}')

        with pytest.raises(ValueError) as exc_info:
            _load_config_from_file(str(json_file))

        error_msg = str(exc_info.value)
        assert "Invalid JSON" in error_msg
        assert "line" in error_msg.lower()

    def test_invalid_yaml_syntax(self, tmp_path):
        """Test error for invalid YAML with line number."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("key: value\n  bad_indent: true")  # Invalid indentation

        with pytest.raises(ValueError) as exc_info:
            _load_config_from_file(str(yaml_file))

        error_msg = str(exc_info.value)
        assert "Invalid YAML" in error_msg

    def test_json_null_content(self, tmp_path):
        """Test error for JSON file with only null."""
        json_file = tmp_path / "null.json"
        json_file.write_text("null")

        with pytest.raises(ValueError, match="(?i)empty.*null|null.*empty"):
            _load_config_from_file(str(json_file))

    def test_yaml_null_content(self, tmp_path):
        """Test error for YAML file with only null/~."""
        yaml_file = tmp_path / "null.yaml"
        yaml_file.write_text("~")

        with pytest.raises(ValueError, match="(?i)empty.*null|null.*empty"):
            _load_config_from_file(str(yaml_file))

    def test_json_array_not_dict(self, tmp_path):
        """Test error when JSON contains array instead of object."""
        json_file = tmp_path / "array.json"
        json_file.write_text('["item1", "item2"]')

        with pytest.raises(ValueError, match="dictionary"):
            _load_config_from_file(str(json_file))


class TestParseConfigWithFiles:
    """Test suite for parse_config function with file paths."""

    def test_parse_json_file_path(self, tmp_path):
        """Test parse_config with JSON file path."""
        config_data = {
            "train_x": "path/to/X.csv",
            "train_y": "path/to/Y.csv"
        }

        json_file = tmp_path / "dataset.json"
        with open(json_file, 'w') as f:
            json.dump(config_data, f)

        config, name = parse_config(str(json_file))

        assert config["train_x"] == "path/to/X.csv"
        assert name == "dataset"

    def test_parse_yaml_file_path(self, tmp_path):
        """Test parse_config with YAML file path."""
        config_data = {
            "test_x": "path/to/Xtest.csv",
            "task_type": "regression"
        }

        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(config_data, f)

        config, name = parse_config(str(yaml_file))

        assert config["test_x"] == "path/to/Xtest.csv"
        assert config["task_type"] == "regression"
        assert name == "config"

    def test_parse_yaml_file_path_with_aliases(self, tmp_path):
        """Test parse_config maps accepted key aliases in YAML files."""
        config_data = {
            "X_test": "path/to/Xtest.csv",
            "train_m": "path/to/Mcal.csv",
        }

        yaml_file = tmp_path / "aliases.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(config_data, f)

        config, name = parse_config(str(yaml_file))

        assert config["test_x"] == "path/to/Xtest.csv"
        assert config["train_group"] == "path/to/Mcal.csv"
        assert "X_test" not in config
        assert "train_m" not in config
        assert name == "aliases"

    def test_parse_folder_path_still_works(self, tmp_path):
        """Test that folder path parsing still works."""
        # Create a mock data folder with recognizable files
        data_folder = tmp_path / "my_data"
        data_folder.mkdir()
        (data_folder / "Xcal.csv").write_text("1,2,3")
        (data_folder / "Ycal.csv").write_text("1")

        config, name = parse_config(str(data_folder))

        assert config["train_x"] is not None
        assert "my_data" in name.lower()

    def test_parse_dict_config_still_works(self):
        """Test that dict config parsing still works."""
        config_dict = {
            "train_x": "path/to/X.csv",
            "train_y": "path/to/Y.csv"
        }

        config, name = parse_config(config_dict)

        assert config["train_x"] == "path/to/X.csv"
        assert config["train_y"] == "path/to/Y.csv"


class TestDatasetConfigsWithFiles:
    """Test suite for DatasetConfigs with file-based configs."""

    @pytest.fixture
    def sample_data_files(self, tmp_path):
        """Create temporary CSV files for testing."""
        # Create train_x
        x_file = tmp_path / "Xcal.csv"
        x_file.write_text("1000;2000;3000\n0.1;0.2;0.3\n0.4;0.5;0.6")

        # Create train_y
        y_file = tmp_path / "Ycal.csv"
        y_file.write_text("target\n10.5\n20.3")

        return {"x": str(x_file), "y": str(y_file)}

    def test_dataset_configs_from_json_file(self, tmp_path, sample_data_files):
        """Test DatasetConfigs initialization from JSON file."""
        config_data = {
            "train_x": sample_data_files["x"],
            "train_y": sample_data_files["y"],
            "task_type": "regression",
            "global_params": {"delimiter": ";", "has_header": True}
        }

        config_file = tmp_path / "dataset.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        configs = DatasetConfigs(str(config_file))

        assert len(configs.configs) == 1
        dataset = configs.get_dataset_at(0)
        assert dataset is not None
        assert dataset.name == "dataset"

    def test_dataset_configs_from_yaml_file(self, tmp_path, sample_data_files):
        """Test DatasetConfigs initialization from YAML file."""
        yaml_content = f"""
train_x: {sample_data_files['x']}
train_y: {sample_data_files['y']}
task_type: regression
global_params:
  delimiter: ";"
  has_header: true
"""
        config_file = tmp_path / "wheat_protein.yaml"
        config_file.write_text(yaml_content)

        configs = DatasetConfigs(str(config_file))

        assert len(configs.configs) == 1
        dataset = configs.get_dataset_at(0)
        assert dataset is not None
        assert dataset.name == "wheat_protein"

    def test_dataset_configs_with_custom_name_in_file(self, tmp_path, sample_data_files):
        """Test that 'name' in config file is used."""
        config_data = {
            "name": "my_custom_dataset",
            "train_x": sample_data_files["x"],
            "train_y": sample_data_files["y"],
            "global_params": {"delimiter": ";", "has_header": True}
        }

        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        configs = DatasetConfigs(str(config_file))
        dataset = configs.get_dataset_at(0)

        assert dataset.name == "my_custom_dataset"

    def test_dataset_configs_with_task_type_in_file(self, tmp_path, sample_data_files):
        """Test task_type specified in config file."""
        config_data = {
            "train_x": sample_data_files["x"],
            "train_y": sample_data_files["y"],
            "task_type": "regression",
            "global_params": {"delimiter": ";", "has_header": True}
        }

        config_file = tmp_path / "dataset.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        # task_type from file should be used
        configs = DatasetConfigs(str(config_file))
        dataset = configs.get_dataset_at(0)

        # The dataset should have the task type from file
        assert configs._task_types[0] == "regression"

    def test_dataset_configs_aggregate_from_file(self, tmp_path, sample_data_files):
        """Test that aggregate is read from config file."""
        config_data = {
            "train_x": sample_data_files["x"],
            "train_y": sample_data_files["y"],
            "task_type": "regression",
            "aggregate": "sample_id",
            "global_params": {"delimiter": ";", "has_header": True}
        }

        config_file = tmp_path / "dataset.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        configs = DatasetConfigs(str(config_file))

        assert configs._aggregates[0] == "sample_id"

    def test_dataset_configs_multiple_json_files(self, tmp_path, sample_data_files):
        """Test loading multiple JSON config files."""
        config1 = {
            "train_x": sample_data_files["x"],
            "train_y": sample_data_files["y"],
            "global_params": {"delimiter": ";", "has_header": True}
        }
        config2 = {
            "train_x": sample_data_files["x"],
            "train_y": sample_data_files["y"],
            "global_params": {"delimiter": ";", "has_header": True}
        }

        file1 = tmp_path / "dataset1.json"
        file2 = tmp_path / "dataset2.json"
        with open(file1, 'w') as f:
            json.dump(config1, f)
        with open(file2, 'w') as f:
            json.dump(config2, f)

        configs = DatasetConfigs([str(file1), str(file2)])

        assert len(configs.configs) == 2
        assert configs.configs[0][1] == "dataset1"
        assert configs.configs[1][1] == "dataset2"

    def test_dataset_configs_invalid_json_file_error(self, tmp_path):
        """Test that invalid JSON file raises helpful error."""
        config_file = tmp_path / "bad.json"
        config_file.write_text('{\n  "train_x": "path"\n  "missing_comma": true\n}')

        with pytest.raises(ValueError) as exc_info:
            DatasetConfigs(str(config_file))

        error_msg = str(exc_info.value)
        assert "Invalid JSON" in error_msg or "line" in error_msg.lower()
