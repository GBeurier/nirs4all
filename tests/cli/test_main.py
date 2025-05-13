import os
import tempfile
from click.testing import CliRunner
from nirs4all.cli.main import nirs4all_cli


def test_nirs4all_cli_entry_point():
    runner = CliRunner()
    result = runner.invoke(nirs4all_cli, ['--help'])  # Explicitly ask for help
    assert result.exit_code == 0
    assert "nirs4all: A comprehensive command-line interface for NIRS data analysis." in result.output
    assert "Usage: nirs4all-cli [OPTIONS] COMMAND [ARGS]..." in result.output


def test_nirs4all_cli_train_command_no_args():
    runner = CliRunner()
    result = runner.invoke(nirs4all_cli, ["train"])
    assert result.exit_code == 0
    assert "CLI command: train" in result.output
    assert "No arguments provided. (not yet implemented)" in result.output


def test_nirs4all_cli_train_command_with_args():
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(delete=False) as tmp_config, tempfile.TemporaryDirectory() as tmp_data_dir:
        tmp_config_path = tmp_config.name
        tmp_data_path = os.path.join(tmp_data_dir, "data.csv")
        # Create a dummy data file
        with open(tmp_data_path, 'w') as f:
            f.write("dummy data")

        result = runner.invoke(nirs4all_cli, [
            "train",
            "--config-file", tmp_config_path,
            "--model-preset", "test_preset",
            "--data-path", tmp_data_path
        ])
        assert result.exit_code == 0, result.output
        assert "CLI command: train" in result.output
        assert f"Config file: {tmp_config_path}" in result.output
        assert "Model preset: test_preset" in result.output
        assert f"Data path: {tmp_data_path}" in result.output
    os.unlink(tmp_config_path)  # Clean up the temporary file


def test_nirs4all_cli_predict_command_missing_data_path():
    runner = CliRunner()
    result = runner.invoke(nirs4all_cli, ["predict"])
    assert result.exit_code != 0  # Should fail because data-path is required
    assert "Missing option '-d' / '--data-path'" in result.output


def test_nirs4all_cli_predict_command_with_args():
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(delete=False) as tmp_config, tempfile.TemporaryDirectory() as tmp_data_dir:
        tmp_config_path = tmp_config.name
        tmp_data_path = os.path.join(tmp_data_dir, "predict_data.csv")
        with open(tmp_data_path, 'w') as f:
            f.write("predict data")

        result = runner.invoke(nirs4all_cli, [
            "predict",
            "--config-file", tmp_config_path,
            "--model-preset", "predict_preset",
            "--data-path", tmp_data_path
        ])
        assert result.exit_code == 0, result.output
        assert "CLI command: predict" in result.output
        assert f"Config/Model file: {tmp_config_path}" in result.output
        assert "Model preset: predict_preset" in result.output
        assert f"Data path: {tmp_data_path}" in result.output
    os.unlink(tmp_config_path)


def test_nirs4all_cli_finetune_command_missing_args():
    runner = CliRunner()
    result_no_args = runner.invoke(nirs4all_cli, ["finetune"])
    assert result_no_args.exit_code != 0
    assert "Missing option '-c' / '--config-file'" in result_no_args.output

    with tempfile.NamedTemporaryFile(delete=False) as tmp_config:
        tmp_config_path = tmp_config.name
        result_missing_data = runner.invoke(nirs4all_cli, ["finetune", "--config-file", tmp_config_path])
        assert result_missing_data.exit_code != 0
        assert "Missing option '-d' / '--data-path'" in result_missing_data.output
    os.unlink(tmp_config_path)


def test_nirs4all_cli_finetune_command_with_args():
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(delete=False) as tmp_config, tempfile.TemporaryDirectory() as tmp_data_dir:
        tmp_config_path = tmp_config.name
        tmp_data_path = os.path.join(tmp_data_dir, "finetune_data.csv")
        with open(tmp_data_path, 'w') as f:
            f.write("finetune data")

        result = runner.invoke(nirs4all_cli, [
            "finetune",
            "--config-file", tmp_config_path,
            "--data-path", tmp_data_path
        ])
        assert result.exit_code == 0, result.output
        assert "CLI command: finetune" in result.output
        assert f"Config file: {tmp_config_path}" in result.output
        assert f"Data path: {tmp_data_path}" in result.output
    os.unlink(tmp_config_path)
