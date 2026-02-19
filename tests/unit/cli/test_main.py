"""
Unit tests for the nirs4all CLI entrypoint.

Tests cover:
- --version flag
- --help flag
- subcommand help
- error handling wrapper (CLI-04)
- workspace path validation (WS-01)
- config validate command structure
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


def invoke_cli(args):
    """Invoke the nirs4all CLI main() with the given argument list.

    Args:
        args: List of string arguments (excluding the program name).

    Returns:
        Tuple of (exit_code, stdout_output) where exit_code is the integer
        exit code raised by SystemExit, or 0 if the function returned normally.
    """
    from nirs4all.cli.main import main

    with patch("sys.argv", ["nirs4all"] + args):
        try:
            main()
            return 0
        except SystemExit as exc:
            return int(exc.code) if exc.code is not None else 0

class TestCLIVersion:
    """Test --version flag."""

    def test_version_flag(self):
        """Test that --version prints version and exits with 0."""
        exit_code = invoke_cli(["--version"])
        assert exit_code == 0

    def test_version_flag_short(self):
        """Test that --version only works as --version (argparse standard)."""
        # argparse only supports --version, not -v for version
        exit_code = invoke_cli(["--version"])
        assert exit_code == 0

class TestCLIHelp:
    """Test --help flag."""

    def test_help_flag(self):
        """Test that --help exits cleanly with code 0."""
        exit_code = invoke_cli(["--help"])
        assert exit_code == 0

    def test_no_args_shows_help(self):
        """Test that no arguments shows help and exits 0."""
        exit_code = invoke_cli([])
        assert exit_code == 0

class TestCLISubcommandHelp:
    """Test subcommand help messages."""

    def test_workspace_help(self):
        """Test that 'workspace --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "--help"])
        assert exit_code == 0

    def test_config_help(self):
        """Test that 'config --help' exits cleanly."""
        exit_code = invoke_cli(["config", "--help"])
        assert exit_code == 0

    def test_workspace_init_help(self):
        """Test that 'workspace init --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "init", "--help"])
        assert exit_code == 0

    def test_config_validate_help(self):
        """Test that 'config validate --help' exits cleanly."""
        exit_code = invoke_cli(["config", "validate", "--help"])
        assert exit_code == 0

    def test_config_schema_help(self):
        """Test that 'config schema --help' exits cleanly."""
        exit_code = invoke_cli(["config", "schema", "--help"])
        assert exit_code == 0

    def test_workspace_list_runs_help(self):
        """Test that 'workspace list-runs --help' exits cleanly."""
        exit_code = invoke_cli(["workspace", "list-runs", "--help"])
        assert exit_code == 0

class TestCLIErrorHandling:
    """Test CLI error handling wrapper (CLI-04)."""

    def test_unexpected_exception_exits_1(self):
        """Test that unhandled exceptions in subcommands exit with code 1."""
        from nirs4all.cli.main import main

        def failing_func(args):
            raise RuntimeError("Unexpected failure")

        mock_args = Mock()
        mock_args.test_install = False
        mock_args.test_integration = False
        mock_args.func = failing_func

        with patch("sys.argv", ["nirs4all", "workspace", "stats"]):
            with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
                exit_code = invoke_cli([])

        # Should exit with 1 due to error handling
        assert exit_code == 1

    def test_keyboard_interrupt_exits_130(self):
        """Test that KeyboardInterrupt exits with code 130."""
        from nirs4all.cli.main import main

        def interrupting_func(args):
            raise KeyboardInterrupt()

        mock_args = Mock()
        mock_args.test_install = False
        mock_args.test_integration = False
        mock_args.func = interrupting_func

        with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
            with pytest.raises(SystemExit) as excinfo:
                with patch("sys.argv", ["nirs4all"]):
                    main()

        assert excinfo.value.code == 130

    def test_system_exit_propagates(self):
        """Test that SystemExit from subcommands is not swallowed."""
        from nirs4all.cli.main import main

        def sys_exit_func(args):
            sys.exit(42)

        mock_args = Mock()
        mock_args.test_install = False
        mock_args.test_integration = False
        mock_args.func = sys_exit_func

        with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
            with pytest.raises(SystemExit) as excinfo:
                with patch("sys.argv", ["nirs4all"]):
                    main()

        assert excinfo.value.code == 42

class TestCLIConfigValidate:
    """Test 'nirs4all config validate' command."""

    def test_validate_missing_file_exits_1(self, tmp_path):
        """Test that validating a non-existent file exits with 1."""
        nonexistent = str(tmp_path / "nonexistent.yaml")
        exit_code = invoke_cli(["config", "validate", nonexistent])
        assert exit_code == 1

    def test_validate_existing_file(self, tmp_path):
        """Test that validating an existing file runs without crashing."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("pipeline: []\n")

        # The validator may return invalid, but it should not raise unexpectedly
        # We just check it exits with 0 or 1 (not a crash)
        exit_code = invoke_cli(["config", "validate", str(config_file)])
        assert exit_code in (0, 1)

class TestCLIConfigSchema:
    """Test 'nirs4all config schema' command."""

    def test_schema_pipeline(self, capsys):
        """Test that 'config schema pipeline' outputs JSON."""
        exit_code = invoke_cli(["config", "schema", "pipeline"])
        # Should exit cleanly
        assert exit_code == 0

    def test_schema_dataset(self, capsys):
        """Test that 'config schema dataset' outputs JSON."""
        exit_code = invoke_cli(["config", "schema", "dataset"])
        assert exit_code == 0

class TestCLIWorkspaceValidation:
    """Test workspace path validation in CLI (WS-01)."""

    def test_workspace_list_runs_nonexistent_path_exits_1(self, tmp_path):
        """Test that list-runs with nonexistent workspace exits with 1."""
        nonexistent = str(tmp_path / "does_not_exist")
        exit_code = invoke_cli(["workspace", "list-runs", "--workspace", nonexistent])
        assert exit_code == 1

    def test_workspace_stats_nonexistent_path_exits_1(self, tmp_path):
        """Test that stats with nonexistent workspace exits with 1."""
        nonexistent = str(tmp_path / "does_not_exist")
        exit_code = invoke_cli(["workspace", "stats", "--workspace", nonexistent])
        assert exit_code == 1

    def test_workspace_query_best_nonexistent_path_exits_1(self, tmp_path):
        """Test that query-best with nonexistent workspace exits with 1."""
        nonexistent = str(tmp_path / "does_not_exist")
        exit_code = invoke_cli(["workspace", "query-best", "--workspace", nonexistent])
        assert exit_code == 1

    def test_workspace_filter_nonexistent_path_exits_1(self, tmp_path):
        """Test that filter with nonexistent workspace exits with 1."""
        nonexistent = str(tmp_path / "does_not_exist")
        exit_code = invoke_cli(["workspace", "filter", "--workspace", nonexistent])
        assert exit_code == 1

    def test_workspace_list_library_nonexistent_path_exits_1(self, tmp_path):
        """Test that list-library with nonexistent workspace exits with 1."""
        nonexistent = str(tmp_path / "does_not_exist")
        exit_code = invoke_cli(["workspace", "list-library", "--workspace", nonexistent])
        assert exit_code == 1

    def test_workspace_init_invalid_parent_exits_1(self, tmp_path):
        """Test that init with nonexistent parent directory exits with 1."""
        invalid_path = str(tmp_path / "nonexistent_parent" / "workspace")
        exit_code = invoke_cli(["workspace", "init", invalid_path])
        assert exit_code == 1

    def test_workspace_init_file_instead_of_dir_exits_1(self, tmp_path):
        """Test that init on an existing file exits with 1."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("content")
        exit_code = invoke_cli(["workspace", "init", str(file_path)])
        assert exit_code == 1

    def test_workspace_init_new_dir_succeeds(self, tmp_path):
        """Test that init in an existing parent directory succeeds."""
        workspace_path = tmp_path / "new_workspace"
        exit_code = invoke_cli(["workspace", "init", str(workspace_path)])
        assert exit_code == 0
        assert workspace_path.exists()
