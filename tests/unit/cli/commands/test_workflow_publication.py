"""Reject unusable CLI outputs before any workflow fitting or publication."""

from argparse import Namespace
from unittest.mock import Mock

import pytest

from nirs4all.cli.commands import workflow


def command(tmp_path, output):
    return Namespace(workflow_command="run", output=str(output), archive=str(tmp_path / "model.n4a"),
                     results_directory=None)


@pytest.mark.parametrize("destination", ["existing", "archive", "results", "missing_parent", "dangling"])
def test_output_refused_before_workflow(tmp_path, monkeypatch, destination):
    output = tmp_path / "output.json"
    if destination == "existing":
        output.write_text("preserve")
    elif destination == "archive":
        output = tmp_path / "model.n4a"
    elif destination == "results":
        output = tmp_path / "model.n4a.results"
    elif destination == "missing_parent":
        output = tmp_path / "missing" / "output.json"
    else:
        output.symlink_to(tmp_path / "absent.json")
    execute = Mock(side_effect=AssertionError("FIT must not be reached"))
    monkeypatch.setattr(workflow, "_execute_value", execute)
    with pytest.raises((ValueError, OSError)):
        workflow._execute(command(tmp_path, output))
    execute.assert_not_called()
    if destination == "existing":
        assert output.read_text() == "preserve"
    if destination == "dangling":
        assert output.is_symlink()
    assert not (tmp_path / "model.n4a").exists()


def test_failed_workflow_releases_own_output_reservation(tmp_path, monkeypatch):
    output = tmp_path / "output.json"

    def fail(args):
        assert output.is_file()
        with pytest.raises(FileExistsError):
            output.open("x")
        raise RuntimeError("native validation refused before FIT")

    monkeypatch.setattr(workflow, "_execute_value", fail)
    with pytest.raises(RuntimeError, match="native validation refused"):
        workflow._execute(command(tmp_path, output))
    assert not output.exists()


def test_successful_output_keeps_serialized_native_value(tmp_path, monkeypatch):
    output = tmp_path / "output.json"
    monkeypatch.setattr(workflow, "_execute_value", lambda args: {"run_id": "native", "metric": 0.0})
    workflow._execute(command(tmp_path, output))
    assert output.read_text() == '{"run_id": "native", "metric": 0.0}'
