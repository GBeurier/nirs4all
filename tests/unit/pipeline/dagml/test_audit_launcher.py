"""The subprocess launcher supports spaces in both executable paths."""

import subprocess
import sys

from nirs4all.pipeline.dagml import process_adapter
from nirs4all.pipeline.dagml.cli_runner import write_launcher_shim


def test_launcher_quotes_interpreter_and_adapter_paths(tmp_path, monkeypatch):
    directory = tmp_path / "my environment"
    directory.mkdir()
    interpreter = directory / "python executable"
    interpreter.symlink_to(sys.executable)
    adapter = directory / "adapter file.py"
    adapter.write_text('import sys\nprint(sys.argv[1:])\n')
    monkeypatch.setattr(process_adapter, "__file__", str(adapter))
    launcher = write_launcher_shim(directory / "launcher", str(interpreter))
    run = subprocess.run([str(launcher), "--describe", "argument with spaces"], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr
    assert "argument with spaces" in run.stdout
