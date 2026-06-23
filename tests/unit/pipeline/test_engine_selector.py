"""Unit tests for the inert backend-engine selector (ADR-17 migration skeleton)."""

from __future__ import annotations

import pytest

from nirs4all.pipeline.engine import DEFAULT_ENGINE, ENGINE_ENV_VAR, resolve_engine


def test_defaults_to_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENGINE_ENV_VAR, raising=False)
    assert DEFAULT_ENGINE == "legacy"
    assert resolve_engine() == "legacy"


def test_explicit_legacy_case_insensitive() -> None:
    assert resolve_engine("legacy") == "legacy"
    assert resolve_engine("  LEGACY  ") == "legacy"


def test_env_var_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENGINE_ENV_VAR, "dag-ml")
    with pytest.raises(NotImplementedError):
        resolve_engine()


def test_explicit_arg_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENGINE_ENV_VAR, "dag-ml")
    assert resolve_engine("legacy") == "legacy"


@pytest.mark.parametrize("name", ["dag-ml", "dual"])
def test_unimplemented_engines_refused(name: str) -> None:
    with pytest.raises(NotImplementedError):
        resolve_engine(name)


def test_unknown_engine_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_engine("rust")
