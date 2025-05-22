import os
import json
import shutil
import tempfile
from pathlib import Path
from click.testing import CliRunner
from nirs4all.cli.main import nirs4all_cli
import pytest
import nirs4all.cli.presets as presets_module

# Define paths to sample configs and presets relative to this test file
# Assuming the test is run from the root of the project or in a way that resolves these paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent # Project root (nirs4all)
SAMPLE_CONFIG_DIR = BASE_DIR / "tests" / "cli" / "sample_configs"
PRESETS_DIR = BASE_DIR / "nirs4all" / "presets"


def _create_dummy_data_dir(path_str: str, content_file_name: str = "data.txt", content: str = "dummy_data"):
    """Helper to create a directory and a dummy file inside it."""
    data_dir = Path(path_str)
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / content_file_name, "w") as f:
        f.write(content)
    return str(data_dir.resolve())

def _create_dummy_json_file(path_str: str, data: dict):
    """Helper to create a dummy JSON file."""
    file_path = Path(path_str)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f)
    return str(file_path.resolve())

class TestNirs4allCli:
    runner = CliRunner()

    def setup_method(self):
        """Create necessary dummy preset and data dirs for tests."""
        # Ensure global presets directory exists
        PRESETS_DIR.mkdir(parents=True, exist_ok=True)

        # Create dummy preset files in global presets dir
        _create_dummy_json_file(
            PRESETS_DIR / "slow_train.json",
            {"dataset": "data/from_slow_train_preset"}
        )
        _create_dummy_json_file(
            PRESETS_DIR / "fast_predict.json",
            {"dataset": "data/from_fast_predict_preset"}
        )
        _create_dummy_json_file(
            PRESETS_DIR / "quick_finetune.json",
            {"dataset": "data/from_quick_finetune_preset"}
        )
        _create_dummy_json_file(
            PRESETS_DIR / "generic.json",
            {"seed": 123, "dataset": "data/from_generic_preset"}
        )

        # Create dummy data directories that these presets/configs might point to
        _create_dummy_data_dir(str(BASE_DIR / "data" / "from_slow_train_preset"))
        _create_dummy_data_dir(str(BASE_DIR / "data" / "from_fast_predict_preset"))
        _create_dummy_data_dir(str(BASE_DIR / "data" / "from_quick_finetune_preset"))
        _create_dummy_data_dir(str(BASE_DIR / "data" / "from_generic_preset"))
        _create_dummy_data_dir(str(SAMPLE_CONFIG_DIR / "data" / "train_data_config")) # from sample_train_config.json
        _create_dummy_data_dir(str(SAMPLE_CONFIG_DIR / "data" / "predict_data_config"))# from sample_predict_config.json


    def teardown_method(self):
        """Clean up dummy data directories created by tests if necessary."""
        # Presets are part of the source, don't delete.
        # Sample configs are part of tests, don't delete.
        # Dummy data dirs created by _create_dummy_data_dir might need cleanup if they are not under tempfile.
        # For simplicity, we are creating them in fixed locations for now.
        # If tests create unique temp dirs, they should handle their own cleanup.
        # The dummy data dirs created in setup_method are inside the project,
        # which is fine for testing as they are small and gitignored if needed.
        pass


    def test_cli_help(self):
        result = self.runner.invoke(nirs4all_cli, ['--help'])
        assert result.exit_code == 0
        assert "Usage: nirs4all-cli [OPTIONS] COMMAND [ARGS]..." in result.output
        assert "train" in result.output
        assert "predict" in result.output
        assert "finetune" in result.output

    def test_load_global_config_train(self, tmp_path):
        sample_conf = SAMPLE_CONFIG_DIR / "sample_train_config.json"
        # Create a dummy data directory and use it via --data-path
        data_dir = tmp_path / "dummy_data"
        _create_dummy_data_dir(str(data_dir))

        result = self.runner.invoke(nirs4all_cli, [
            "--config", str(sample_conf),
            "--data-path", str(data_dir),
            "train"
        ])
        assert result.exit_code == 0, result.output
        assert f"Loaded global configuration from: {str(sample_conf)}" in result.output
        assert f"Applied global --data-path override: {str(data_dir)}" in result.output
        assert f"Final effective dataset path: {str(data_dir)}" in result.output
        assert "Executing CLI command: train" in result.output

    def test_global_data_path_override(self, tmp_path):
        sample_conf = SAMPLE_CONFIG_DIR / "sample_train_config.json"
        override_data_dir = _create_dummy_data_dir(str(tmp_path / "override_data"))

        result = self.runner.invoke(nirs4all_cli, [
            "--config", str(sample_conf),
            "--data-path", override_data_dir,
            "train"
        ])
        assert result.exit_code == 0, result.output
        assert f"Applied global --data-path override: {override_data_dir}" in result.output
        assert f"Final effective dataset path: {override_data_dir}" in result.output

    def test_train_with_preset(self, tmp_path):
        # Preset defines its own dataset path: data/from_slow_train_preset
        preset_data_path = str((BASE_DIR / "data" / "from_slow_train_preset").resolve())
        _create_dummy_data_dir(preset_data_path) # Ensure it exists

        result = self.runner.invoke(nirs4all_cli, ["train", "--preset", "slow_train"])
        assert result.exit_code == 0, result.output
        assert "Applied preset 'slow_train'" in result.output
        assert f"Final effective dataset path: {preset_data_path}" in result.output
        assert "experiment" in result.output # Check if preset values (like epochs) are implicitly loaded

    def test_train_config_and_preset_preset_overrides_config_dataset(self, tmp_path):
        sample_conf_path = SAMPLE_CONFIG_DIR / "sample_train_config.json" # Defines dataset: data/train_data_config
        _create_dummy_data_dir(str(SAMPLE_CONFIG_DIR / "data" / "train_data_config"))

        preset_data_path = str((BASE_DIR / "data" / "from_slow_train_preset").resolve()) # From slow_train.json
        _create_dummy_data_dir(preset_data_path)

        result = self.runner.invoke(nirs4all_cli, [
            "--config", str(sample_conf_path),
            "train",
            "--preset", "slow_train"
        ])
        assert result.exit_code == 0, result.output
        assert f"Loaded global configuration from: {str(sample_conf_path)}" in result.output
        assert "Applied preset 'slow_train'" in result.output
        # Preset's dataset path should take precedence over the one in the global config
        assert f"Final effective dataset path: {preset_data_path}" in result.output

    def test_predict_command_specific_data_path_override(self, tmp_path):
        sample_conf = SAMPLE_CONFIG_DIR / "sample_predict_config.json" # Defines data/predict_data_config
        _create_dummy_data_dir(str(SAMPLE_CONFIG_DIR / "data" / "predict_data_config"))

        global_override_data_dir = _create_dummy_data_dir(str(tmp_path / "global_predict_data"))
        cmd_specific_data_dir = _create_dummy_data_dir(str(tmp_path / "cmd_specific_predict_data"))

        result = self.runner.invoke(nirs4all_cli, [
            "--config", str(sample_conf),
            "--data-path", global_override_data_dir, # Global override
            "predict",
            "--preset", "fast_predict", # Preset might have its own data path
            "--data-path", cmd_specific_data_dir # Command-specific override
        ])
        assert result.exit_code == 0, result.output
        assert f"Applied command-specific --data-path: {cmd_specific_data_dir}" in result.output
        assert f"Final effective dataset path: {cmd_specific_data_dir}" in result.output
        assert "Applied preset 'fast_predict'" in result.output # Ensure preset was still loaded

    def test_action_inference_from_config(self, tmp_path):
        config_content = {
            "experiment": {"action": "train"},
            "dataset": str(tmp_path / "action_infer_data")
        }
        _create_dummy_data_dir(str(tmp_path / "action_infer_data"))
        config_file = _create_dummy_json_file(str(tmp_path / "config_with_action.json"), config_content)

        result = self.runner.invoke(nirs4all_cli, ["--config", config_file])
        assert result.exit_code == 0, result.output
        assert f"Action 'train' inferred from global config." in result.output
        assert "Executing CLI command: train" in result.output # Check that train command was actually called
        assert f"Final effective dataset path: {str(Path(config_content['dataset']).resolve())}" in result.output

    def test_action_inference_unknown_action(self, tmp_path):
        config_content = {
            "experiment": {"action": "unknown_action"},
            "dataset": str(tmp_path / "action_infer_data_unknown")
        }
        _create_dummy_data_dir(str(tmp_path / "action_infer_data_unknown"))
        config_file = _create_dummy_json_file(str(tmp_path / "config_with_unknown_action.json"), config_content)

        result = self.runner.invoke(nirs4all_cli, ["--config", config_file])
        assert result.exit_code == 0 # Main CLI group doesn't exit with error, but prints error message
        assert "Unknown action 'unknown_action' in config." in result.output

    def test_precedence_all_data_paths(self, tmp_path):
        # 1. Config data path
        config_data_content = {"dataset": str(tmp_path / "config_data")}
        _create_dummy_data_dir(config_data_content["dataset"])
        config_file = _create_dummy_json_file(str(tmp_path / "precedence_config.json"), config_data_content)

        # 2. Preset data path (using generic preset for this test)
        # generic.json defines "dataset": "data/from_generic_preset"
        # This path is relative to BASE_DIR for the preset loader.
        preset_defined_data_path = str((BASE_DIR / "data" / "from_generic_preset").resolve())
        _create_dummy_data_dir(preset_defined_data_path)


        # 3. Global --data-path
        global_override_path = _create_dummy_data_dir(str(tmp_path / "global_override_data"))

        # 4. Command-specific --data-path (highest precedence)
        cmd_specific_path = _create_dummy_data_dir(str(tmp_path / "cmd_finetune_data"))

        result = self.runner.invoke(nirs4all_cli, [
            "--config", config_file,
            "--data-path", global_override_path,
            "finetune", # Using finetune command for this test
            "--preset", "generic", # Preset that defines a dataset path
            "--data-path", cmd_specific_path
        ])
        assert result.exit_code == 0, result.output
        assert f"Loaded global configuration from: {config_file}" in result.output
        assert f"Applied preset 'generic'" in result.output
        assert f"Applied global --data-path override: {global_override_path}" in result.output
        assert f"Applied command-specific --data-path: {cmd_specific_path}" in result.output
        assert f"Final effective dataset path: {cmd_specific_path}" in result.output # Command specific wins

    def test_error_missing_config_file(self):
        result = self.runner.invoke(nirs4all_cli, ["--config", "non_existent_config.json", "train"])
        assert result.exit_code != 0
        assert "Invalid value for '--config': Path 'non_existent_config.json' does not exist." in result.output

    def test_error_invalid_json_config(self, tmp_path):
        invalid_json_file = tmp_path / "invalid.json"
        with open(invalid_json_file, "w") as f:
            f.write("{\"malformed_json\": ") # Corrected invalid JSON
        
        _create_dummy_data_dir(str(tmp_path / "dummy_data_for_invalid_json_test"))


        result = self.runner.invoke(nirs4all_cli, ["--config", str(invalid_json_file), "train", "--data-path", str(tmp_path / "dummy_data_for_invalid_json_test")])
        assert result.exit_code != 0 # Should be 1 due to ClickException
        assert f"Error loading global configuration file {str(invalid_json_file)}" in result.output

    def test_error_missing_preset_file(self, tmp_path):
        _create_dummy_data_dir(str(tmp_path / "dummy_data_for_missing_preset"))
        result = self.runner.invoke(nirs4all_cli, ["train", "--preset", "non_existent_preset", "--data-path", str(tmp_path / "dummy_data_for_missing_preset")])
        assert result.exit_code != 0 # ClickException
        assert "Preset 'non_existent_preset' (as 'non_existent_preset.json') not found" in result.output
        
    def test_error_missing_data_path_if_not_provided_anywhere(self, tmp_path):
        # Config without dataset, no global data-path, no preset, no command data-path
        config_no_dataset_content = {"model": {"type": "some_model"}}
        config_file = _create_dummy_json_file(str(tmp_path / "config_no_dataset.json"), config_no_dataset_content)

        result = self.runner.invoke(nirs4all_cli, ["--config", config_file, "train"])
        assert result.exit_code != 0 # ClickException
        assert "Dataset path is required but not found" in result.output

    def test_no_command_no_action_in_config(self, tmp_path):
        config_no_action_content = {"dataset": str(tmp_path / "no_action_data")} # Has dataset, but no action
        _create_dummy_data_dir(config_no_action_content["dataset"])
        config_file = _create_dummy_json_file(str(tmp_path / "config_no_action.json"), config_no_action_content)

        result = self.runner.invoke(nirs4all_cli, ["--config", config_file])
        assert result.exit_code == 0 # Prints error, doesn't raise exception from main group
        assert "No command specified and no 'action' found in global config." in result.output

    def test_no_command_no_config_shows_help(self):
        result = self.runner.invoke(nirs4all_cli, [])
        assert result.exit_code == 0
        assert "Usage: nirs4all-cli [OPTIONS] COMMAND [ARGS]..." in result.output # Shows help

    def test_finetune_command_with_preset_and_data_path(self, tmp_path):
        cmd_specific_data_dir = _create_dummy_data_dir(str(tmp_path / "finetune_specific_data"))
        preset_data_path = str((BASE_DIR / "data" / "from_quick_finetune_preset").resolve())
        _create_dummy_data_dir(preset_data_path) # Ensure preset data path exists

        result = self.runner.invoke(nirs4all_cli, [
            "finetune",
            "--preset", "quick_finetune",
            "--data-path", cmd_specific_data_dir
        ])
        assert result.exit_code == 0, result.output
        assert "Applied preset 'quick_finetune'" in result.output
        assert f"Applied command-specific --data-path: {cmd_specific_data_dir}" in result.output
        assert f"Final effective dataset path: {cmd_specific_data_dir}" in result.output
        assert "Executing CLI command: finetune" in result.output
        assert "experiment" in result.output # Check if preset values (like lr) are implicitly loaded

    def test_predict_with_config_model_and_preset_model(self, tmp_path):
        # Config has a model, preset also has a model. Preset should override.
        sample_conf_path = SAMPLE_CONFIG_DIR / "sample_predict_config.json" # Has model.SimpleCNN
        # sample_predict_config.json also defines dataset: data/predict_data_config
        _create_dummy_data_dir(str(SAMPLE_CONFIG_DIR / "data" / "predict_data_config"))


        # fast_predict.json defines model.type = FastModel and dataset = data/from_fast_predict_preset
        preset_data_path = str((BASE_DIR / "data" / "from_fast_predict_preset").resolve())
        _create_dummy_data_dir(preset_data_path)


        result = self.runner.invoke(nirs4all_cli, [
            "--config", str(sample_conf_path),
            "predict",
            "--preset", "fast_predict"
            # No command-specific data path, so preset's data path should be used
        ])
        assert result.exit_code == 0, result.output
        assert "Applied preset 'fast_predict'" in result.output
        # Check that the model from the preset is mentioned (or would be used)
        # The current CLI output for predict doesn't explicitly show the model type from config object.
        # We rely on the fact that apply_preset_to_config works correctly.
        # A more robust test would involve inspecting the Config object if possible,
        # or having the CLI command output more details about the final config.

# Helper to create dummy data dirs
def create_dummy_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "dummy_file.txt").write_text("dummy content")

@pytest.fixture
def cli_test_env(tmp_path, monkeypatch):
    # Data directories
    data_paths = {
        "config": tmp_path / "data_from_config",
        "global_override": tmp_path / "data_global_override",
        "cmd_override": tmp_path / "data_cmd_override",
        "preset_train": tmp_path / "data_from_train_preset",
        # "preset_predict": tmp_path / "data_from_predict_preset",  # fast_predict preset doesn't define a dataset path
    }
    for p_name, p_val in data_paths.items():
        create_dummy_dir(p_val)

    # Config files
    test_configs_dir = tmp_path / "test_configs"
    test_configs_dir.mkdir()

    train_config_content = {
        "dataset": str(data_paths["config"]),
        "model": {"class": "models.SimpleCNN"},
        "experiment": {"action": "train", "name": "test_train_run"}
    }
    train_config_path = test_configs_dir / "train_config.json"
    train_config_path.write_text(json.dumps(train_config_content))

    predict_config_content = {
        "dataset": str(data_paths["config"]),
        "model": {"class": "models.SimpleCNN", "params": {"weights_path": str(tmp_path / "dummy_weights.pth")}},
        "experiment": {"action": "predict", "name": "test_predict_run"}
    }
    predict_config_path = test_configs_dir / "predict_config.json"
    predict_config_path.write_text(json.dumps(predict_config_content))
    (tmp_path / "dummy_weights.pth").touch()

    no_action_config_content = {"dataset": str(data_paths["config"])}
    no_action_config_path = test_configs_dir / "no_action_config.json"
    no_action_config_path.write_text(json.dumps(no_action_config_content))

    no_dataset_config_content = {"model": {"class": "models.SimpleCNN"}}
    no_dataset_config_path = test_configs_dir / "no_dataset_config.json"
    no_dataset_config_path.write_text(json.dumps(no_dataset_config_content))
    
    invalid_config_path = test_configs_dir / "invalid_config.json"
    invalid_config_path.write_text("{\"malformed_json\": ")  # Corrected invalid JSON

    # Presets
    test_presets_root_dir = tmp_path / "test_cli_presets_root"  # Renamed to avoid conflict with "presets" in path
    test_presets_root_dir.mkdir(exist_ok=True)
    
    # Create action-specific subdirectories train/ and predict/
    (test_presets_root_dir / "train").mkdir(exist_ok=True)
    (test_presets_root_dir / "predict").mkdir(exist_ok=True)

    slow_train_preset_content = {
        "dataset": str(data_paths["preset_train"]),
        "optimizer": {"params": {"lr": 0.00001}},
        "experiment": {"epochs": 250}
    }
    slow_train_preset_path = test_presets_root_dir / "train" / "slow_train.json"
    slow_train_preset_path.write_text(json.dumps(slow_train_preset_content))

    fast_predict_preset_content = {
        "model": {"params": {"weights_path": str(tmp_path / "dummy_preset_weights.pth")}},
        "experiment": {"batch_size": 128}
    }
    fast_predict_preset_path = test_presets_root_dir / "predict" / "fast_predict.json"
    fast_predict_preset_path.write_text(json.dumps(fast_predict_preset_content))
    (tmp_path / "dummy_preset_weights.pth").touch()

    monkeypatch.setattr(presets_module, 'PRESET_DIR', str(test_presets_root_dir))

    return {
        "runner": CliRunner(),
        "prog_name": "nirs4all",  # As defined in pyproject.toml entry point
        "configs": {
            "train": str(train_config_path),
            "predict": str(predict_config_path),
            "no_action": str(no_action_config_path),
            "no_dataset": str(no_dataset_config_path),
            "invalid": str(invalid_config_path),
        },
        "data_paths": {k: str(v) for k, v in data_paths.items()},
    }

def test_cli_no_command_shows_help(cli_test_env):
    runner = cli_test_env["runner"]
    result = runner.invoke(nirs4all_cli, prog_name=cli_test_env["prog_name"])
    assert result.exit_code == 0
    assert f"Usage: {cli_test_env['prog_name']} [OPTIONS] COMMAND [ARGS]" in result.output

def test_global_config_train_command(cli_test_env):
    runner = cli_test_env["runner"]
    config_path = cli_test_env["configs"]["train"]
    data_path_expected = cli_test_env["data_paths"]["config"]

    result = runner.invoke(nirs4all_cli, ["--config", config_path, "train"], prog_name=cli_test_env["prog_name"])
    assert result.exit_code == 0
    assert f"Loaded global configuration from: {config_path}" in result.output
    assert "Executing CLI command: train" in result.output
    assert f"Final effective dataset path: {data_path_expected}" in result.output

def test_global_data_path_override(cli_test_env):
    runner = cli_test_env["runner"]
    config_path = cli_test_env["configs"]["train"]
    override_data_path = cli_test_env["data_paths"]["global_override"]

    result = runner.invoke(nirs4all_cli, ["--config", config_path, "--data-path", override_data_path, "train"], prog_name=cli_test_env["prog_name"])
    assert result.exit_code == 0
    assert f"Loaded global configuration from: {config_path}" in result.output
    assert f"Applied global --data-path override: {override_data_path}" in result.output
    assert "Executing CLI command: train" in result.output
    assert f"Final effective dataset path: {override_data_path}" in result.output

def test_train_with_preset_only(cli_test_env):
    runner = cli_test_env["runner"]
    preset_data_path_expected = cli_test_env["data_paths"]["preset_train"]

    result = runner.invoke(nirs4all_cli, ["train", "--preset", "slow_train"], prog_name=cli_test_env["prog_name"])
    assert result.exit_code == 0
    assert "Executing CLI command: train" in result.output
    assert "Applied preset 'slow_train'." in result.output
    assert f"Final effective dataset path: {preset_data_path_expected}" in result.output

def test_predict_cmd_specific_data_path_override(cli_test_env):
    runner = cli_test_env["runner"]
    config_path = cli_test_env["configs"]["predict"]
    cmd_override_data_path = cli_test_env["data_paths"]["cmd_override"]

    result = runner.invoke(nirs4all_cli, [
        "--config", config_path,
        "predict",
        "--preset", "fast_predict",
        "--data-path", cmd_override_data_path
    ], prog_name=cli_test_env["prog_name"])
    assert result.exit_code == 0
    assert f"Loaded global configuration from: {config_path}" in result.output
    assert "Applied preset 'fast_predict'." in result.output
    assert f"Applied command-specific --data-path: {cmd_override_data_path}" in result.output
    assert "Executing CLI command: predict" in result.output
    assert f"Final effective dataset path: {cmd_override_data_path}" in result.output

def test_action_inference_from_config_train(cli_test_env):
    runner = cli_test_env["runner"]
    config_path = cli_test_env["configs"]["train"]  # Has action: "train"
    data_path_expected = cli_test_env["data_paths"]["config"]

    result = runner.invoke(nirs4all_cli, ["--config", config_path], prog_name=cli_test_env["prog_name"])
    assert result.exit_code == 0
    assert f"Loaded global configuration from: {config_path}" in result.output
    assert "Action 'train' inferred from global config." in result.output
    assert "Executing CLI command: train" in result.output
    assert f"Final effective dataset path: {data_path_expected}" in result.output

def test_data_path_precedence_train(cli_test_env):  # Global > Preset > Config
    runner = cli_test_env["runner"]
    config_path = cli_test_env["configs"]["train"]  # Dataset: data_paths["config"]
    global_override_path = cli_test_env["data_paths"]["global_override"]
    # slow_train preset has dataset: data_paths["preset_train"]

    result = runner.invoke(nirs4all_cli, [
        "--config", config_path,
        "--data-path", global_override_path,
        "train",
        "--preset", "slow_train"
    ], prog_name=cli_test_env["prog_name"])
    assert result.exit_code == 0
    assert "Applied preset 'slow_train'." in result.output
    assert f"Applied global --data-path override: {global_override_path}" in result.output
    assert f"Final effective dataset path: {global_override_path}" in result.output

def test_data_path_precedence_predict(cli_test_env):  # Cmd > Global > Preset > Config
    runner = cli_test_env["runner"]
    config_path = cli_test_env["configs"]["predict"]  # Dataset: data_paths["config"]
    global_override_path = cli_test_env["data_paths"]["global_override"]
    cmd_override_path = cli_test_env["data_paths"]["cmd_override"]
    # fast_predict preset has no dataset path

    result = runner.invoke(nirs4all_cli, [
        "--config", config_path,
        "--data-path", global_override_path,
        "predict",
        "--preset", "fast_predict",
        "--data-path", cmd_override_path
    ], prog_name=cli_test_env["prog_name"])
    assert result.exit_code == 0
    assert f"Applied global --data-path override: {global_override_path}" in result.output
    assert f"Applied command-specific --data-path: {cmd_override_path}" in result.output
    assert f"Final effective dataset path: {cmd_override_path}" in result.output

def test_error_missing_config_file(cli_test_env):
    runner = cli_test_env["runner"]
    result = runner.invoke(nirs4all_cli, ["--config", "non_existent_config.json", "train"], prog_name=cli_test_env["prog_name"])
    assert result.exit_code != 0
    assert "Error: Invalid value for '--config': Path 'non_existent_config.json' does not exist." in result.output

def test_error_invalid_json_config(cli_test_env):
    runner = cli_test_env["runner"]
    invalid_config_f = cli_test_env["configs"]["invalid"]
    result = runner.invoke(nirs4all_cli, ["--config", invalid_config_f, "train"], prog_name=cli_test_env["prog_name"])
    assert result.exit_code != 0
    assert f"Error loading global configuration file {invalid_config_f}" in result.output  # From main.py ClickException
    assert "Error decoding JSON from file" in str(result.exception)  # More specific check on exception

def test_error_missing_preset_file(cli_test_env):
    runner = cli_test_env["runner"]
    data_path = cli_test_env["data_paths"]["config"]  # Need a valid data path to avoid other errors
    result = runner.invoke(nirs4all_cli, ["train", "--preset", "non_existent_preset", "--data-path", data_path], prog_name=cli_test_env["prog_name"])
    assert result.exit_code != 0
    assert "Preset 'non_existent_preset' (as 'non_existent_preset.json') not found." in result.output

def test_error_missing_data_path_train(cli_test_env):
    runner = cli_test_env["runner"]
    result = runner.invoke(nirs4all_cli, ["train"], prog_name=cli_test_env["prog_name"])
    assert result.exit_code != 0
    assert "Dataset path is required but not found." in result.output

def test_error_missing_data_path_predict_with_config_no_dataset(cli_test_env):
    runner = cli_test_env["runner"]
    # Config that doesn't define a dataset
    config_no_dataset_path = cli_test_env["configs"]["no_dataset"]
    result = runner.invoke(nirs4all_cli, ["--config", config_no_dataset_path, "predict"], prog_name=cli_test_env["prog_name"])
    assert result.exit_code != 0
    assert "Dataset path is required but not found." in result.output

def test_config_no_action_no_command_prints_help_message(cli_test_env):
    runner = cli_test_env["runner"]
    config_path = cli_test_env["configs"]["no_action"]
    result = runner.invoke(nirs4all_cli, ["--config", config_path], prog_name=cli_test_env["prog_name"])
    # As per current main.py, this prints a message to stderr and exits 0.
    assert result.exit_code == 0 
    assert "No command specified and no 'action' found in global config. Use --help for options." in result.output

def test_error_global_data_path_not_exists(cli_test_env):
    runner = cli_test_env["runner"]
    config_path = cli_test_env["configs"]["train"]
    result = runner.invoke(nirs4all_cli, ["--config", config_path, "--data-path", "non_existent_dir_global", "train"], prog_name=cli_test_env["prog_name"])
    assert result.exit_code != 0
    assert "Error: Invalid value for '--data-path': Path 'non_existent_dir_global' does not exist." in result.output

def test_error_cmd_data_path_not_exists_for_predict(cli_test_env):
    runner = cli_test_env["runner"]
    config_path = cli_test_env["configs"]["predict"]
    result = runner.invoke(nirs4all_cli, ["--config", config_path, "predict", "--data-path", "non_existent_dir_cmd"], prog_name=cli_test_env["prog_name"])
    assert result.exit_code != 0
    assert "Error: Invalid value for '--data-path': Path 'non_existent_dir_cmd' does not exist." in result.output