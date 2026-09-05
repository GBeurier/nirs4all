"""Only a fully replay-validated native archive may reach the requested path."""

import importlib
from types import SimpleNamespace

import pytest

from nirs4all.api.native_archive_training import NativeArchiveTrainingError, NativeMethodsArchiveRunResult


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["validation", "reference"])
def test_refused_archive_never_replaces_destination(tmp_path, monkeypatch, existing, failure):
    target = tmp_path / "result.n4a"
    if existing:
        target.write_bytes(b"previous good archive")
    existing_entries = set(tmp_path.iterdir())
    result = NativeMethodsArchiveRunResult.__new__(NativeMethodsArchiveRunResult)
    result._native_dag_ml = SimpleNamespace(build_archive_v2_native_portable_payloads=lambda *_args: ({}, {}))
    result._native_archive_id = "archive:test"
    result._native_outcome_contract = {}
    result._native_package_contract = {}
    result._methods_library_path = "/runtime/libn4m.so"

    def write(path, *_args):
        path.write_bytes(b"new invalid archive")
        return None if failure == "reference" else {"archive_id": "archive:test", "archive_sha256": "0" * 64}

    def refuse(*_args, **_kwargs):
        raise ValueError("provenance rejected")

    result._native_core = SimpleNamespace(write_archive_v2_from_native_payloads=write)
    replay = importlib.import_module("nirs4all.pipeline.dagml.core_archive_replay")
    monkeypatch.setattr(replay, "validate_core_methods_archive_v2", refuse)
    with pytest.raises(NativeArchiveTrainingError, match="publication"):
        result.export(target)
    if existing:
        assert target.read_bytes() == b"previous good archive"
    else:
        assert not target.exists()
    assert set(tmp_path.iterdir()) == existing_entries
