"""Unit tests for the backend-engine selector (default is legacy (interim, pre-refactoring))."""

from __future__ import annotations

import pytest

from nirs4all.pipeline.engine import DEFAULT_ENGINE, ENGINE_ENV_VAR, resolve_engine


def test_defaults_to_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    # default is legacy (interim, pre-refactoring): the public-maintained nirs4all stays pure-Python
    # by default; dag-ml stays fully selectable via engine="dag-ml" / $N4A_ENGINE=dag-ml.
    monkeypatch.delenv(ENGINE_ENV_VAR, raising=False)
    assert DEFAULT_ENGINE == "legacy"
    assert resolve_engine() == "legacy"


def test_explicit_legacy_case_insensitive() -> None:
    assert resolve_engine("legacy") == "legacy"
    assert resolve_engine("  LEGACY  ") == "legacy"


def test_env_var_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENGINE_ENV_VAR, "dag-ml")
    assert resolve_engine() == "dag-ml"


def test_explicit_arg_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENGINE_ENV_VAR, "dag-ml")
    assert resolve_engine("legacy") == "legacy"


def test_dagml_engine_resolves() -> None:
    # The dag-ml backend is wired (run dispatches to the dag-ml-cli runner); it resolves cleanly.
    assert resolve_engine("dag-ml") == "dag-ml"
    assert resolve_engine("  DAG-ML  ") == "dag-ml"


def test_dual_engine_refused() -> None:
    # Side-by-side comparison mode is still unimplemented.
    with pytest.raises(NotImplementedError):
        resolve_engine("dual")


def test_unknown_engine_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_engine("rust")
