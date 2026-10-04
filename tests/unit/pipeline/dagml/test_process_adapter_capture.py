"""A response must not become visible before its complete JSONL capture."""

import io
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nirs4all.pipeline.dagml import process_adapter

_WORKER = r'''
import importlib.util
import os
import sys

source, capture, worker, size, frames, pause = sys.argv[1:]
spec = importlib.util.spec_from_file_location("adapter_under_test", source)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

class Primary:
    def write(self, text):
        count = sys.stdout.write(text)
        sys.stdout.flush()
        if pause == "1" and text.endswith("\n"):
            # Hold this precise IO interleaving until the parent reads/kills us.
            # No sleep, model callback, or product scheduler is modified.
            sys.stdin.buffer.read(1)
        return count

    def flush(self):
        sys.stdout.flush()

with open(capture, "a", encoding="utf-8") as stream:
    for sequence in range(int(frames)):
        adapter._emit(adapter._Tee(Primary(), stream), {
            "worker": int(worker), "sequence": sequence, "values": "λ" * int(size),
        })
'''


def _command(capture: Path, *, worker: int = 0, size: int = 64, frames: int = 1, pause: bool = False) -> list[str]:
    return [sys.executable, "-I", "-B", "-u", "-c", _WORKER, process_adapter.__file__, str(capture),
            str(worker), str(size), str(frames), "1" if pause else "0"]


def test_emit_serializes_one_complete_frame_before_writing() -> None:
    writes = []

    class Writer:
        def write(self, text: str) -> int:
            writes.append(text)
            return len(text)

        def flush(self) -> None:
            pass

    payload = {"type": "result", "values": [1, 2], "label": "λ"}
    process_adapter._emit(Writer(), payload)
    assert writes == [json.dumps(payload, sort_keys=True) + "\n"]


@pytest.mark.parametrize("size", [64, 100_000])
def test_capture_is_readable_before_primary_receives_frame(tmp_path: Path, size: int) -> None:
    capture = tmp_path / "results.jsonl"
    payload = {"type": "result", "values": "λ" * size}
    snapshots = []

    class Primary:
        def write(self, text: str) -> int:
            snapshots.append([json.loads(line) for line in capture.read_text().splitlines()])
            assert snapshots[-1] == [payload]
            return len(text)

        def flush(self) -> None:
            pass

    with capture.open("a", encoding="utf-8") as stream:
        process_adapter._emit(process_adapter._Tee(Primary(), stream), payload)
    assert snapshots == [[payload]]


@pytest.mark.skipif(os.name != "posix", reason="Abrupt POSIX termination witness")
@pytest.mark.parametrize("size", [64, 100_000])
def test_capture_survives_kill_at_visible_response(tmp_path: Path, size: int) -> None:
    capture = tmp_path / "results.jsonl"
    process = subprocess.Popen(_command(capture, size=size, pause=True), stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout is not None
    # communicate cannot be used: the worker intentionally waits after making
    # the response visible, before _Tee.write returns to _emit's final flush.
    with ThreadPoolExecutor(max_workers=1) as reader:
        try:
            response = reader.submit(process.stdout.readline).result(timeout=10)
            expected = {"worker": 0, "sequence": 0, "values": "λ" * size}
            assert json.loads(response) == expected
            visible_capture = capture.read_bytes()
        finally:
            process.kill()
            process.communicate(timeout=10)
    assert visible_capture == response.encode("utf-8")
    assert capture.read_bytes() == visible_capture
    assert [json.loads(line) for line in capture.read_text().splitlines()] == [expected]


def test_concurrent_workers_append_complete_large_frames(tmp_path: Path) -> None:
    capture = tmp_path / "results.jsonl"
    processes = [subprocess.Popen(_command(capture, worker=worker, size=100_000, frames=3),
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                 for worker in range(4)]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda process: process.communicate(timeout=20), processes))
        assert all(process.returncode == 0 for process in processes), results
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
    frames = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(frames) == 12
    assert {(frame["worker"], frame["sequence"]) for frame in frames} == {(worker, sequence) for worker in range(4) for sequence in range(3)}
    assert all(frame["values"] == "λ" * 100_000 for frame in frames)


def test_serialization_failure_publishes_no_partial_frame(tmp_path: Path) -> None:
    capture = tmp_path / "results.jsonl"
    primary = io.StringIO()
    with capture.open("a", encoding="utf-8") as stream:
        with pytest.raises(TypeError):
            process_adapter._emit(process_adapter._Tee(primary, stream), {"result": object()})
    assert primary.getvalue() == ""
    assert capture.read_bytes() == b""
