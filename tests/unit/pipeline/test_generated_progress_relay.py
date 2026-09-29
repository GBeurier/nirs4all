"""The worker progress mailbox tolerates an event retired after its ack."""

from __future__ import annotations

import os
from pathlib import Path

import cloudpickle
import pytest

from nirs4all.pipeline.dagml.generated_subprocess import _relay_worker_progress
from nirs4all.pipeline.dagml.worker_environment import scientific_worker_environment


def test_retired_progress_event_is_not_read_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    event = tmp_path / "progress.event"
    event.write_bytes(cloudpickle.dumps({"operation": "checkpoint"}))
    original_open = Path.open

    def retire_before_open(path: Path, *args: object, **kwargs: object) -> object:
        if path == event:
            event.unlink()
            raise FileNotFoundError(event)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", retire_before_open)
    _relay_worker_progress(tmp_path, lambda _event: pytest.fail("retired event reached callback"))
    assert not (tmp_path / "progress.answer").exists()


def test_distinct_worker_interpreter_does_not_inherit_parent_native_wheels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A separately selected Python keeps its own scientific ABI closure."""
    custom_modules = tmp_path / "custom-modules"
    custom_modules.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(custom_modules))
    environment = scientific_worker_environment(str(tmp_path / "other-python"))
    paths = environment["PYTHONPATH"].split(os.pathsep)
    assert paths == [str(Path.cwd()), str(custom_modules)]
    assert environment["PYTHONNOUSERSITE"] == "1"
