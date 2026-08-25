"""Resolution coverage for the wheel-bundled Methods runtime."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nirs4all.pipeline.dagml.methods_runtime import resolve_methods_library_path
from nirs4all.pipeline.dagml.native_client import DagMLNativeCoverageError


def test_explicit_methods_runtime_path_must_be_absolute_regular_file(tmp_path) -> None:
    library = tmp_path / "libn4m.so"
    library.write_bytes(b"native")

    assert resolve_methods_library_path(library) == str(library.resolve())

    with pytest.raises(ValueError, match="absolute libn4m file"):
        resolve_methods_library_path("libn4m.so")


def test_bundled_methods_runtime_is_resolved_from_official_python_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    library = tmp_path / "libn4m.so"
    library.write_bytes(b"native")
    module = SimpleNamespace(library_path=lambda: str(library), abi_version=lambda: (2, 2, 0))
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.methods_runtime.importlib.import_module",
        lambda name: module if name == "n4m" else None,
    )

    assert resolve_methods_library_path() == str(library.resolve())


def test_bundled_methods_runtime_refuses_the_pre_optimizer_abi(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    library = tmp_path / "libn4m.so"
    library.write_bytes(b"native")
    module = SimpleNamespace(library_path=lambda: str(library), abi_version=lambda: (2, 0, 0))
    monkeypatch.setattr(
        "nirs4all.pipeline.dagml.methods_runtime.importlib.import_module",
        lambda name: module if name == "n4m" else None,
    )

    with pytest.raises(DagMLNativeCoverageError, match="nirs4all-methods>=1.0.10"):
        resolve_methods_library_path()


def test_missing_bundled_methods_runtime_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str):
        raise ModuleNotFoundError("n4m")

    monkeypatch.setattr("nirs4all.pipeline.dagml.methods_runtime.importlib.import_module", missing)

    with pytest.raises(DagMLNativeCoverageError, match=r"install nirs4all\[native\]"):
        resolve_methods_library_path()
