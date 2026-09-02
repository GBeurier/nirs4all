"""Unit tests for the backend-engine selector (default is dag-ml, legacy is explicit compatibility)."""

from __future__ import annotations

import pytest

from nirs4all.pipeline.engine import (
    DEFAULT_ENGINE,
    DEFAULT_EXECUTION_PROFILE,
    ENGINE_ENV_VAR,
    ExecutionProfileError,
    resolve_engine,
)


def test_defaults_to_dagml(monkeypatch: pytest.MonkeyPatch) -> None:
    # V1 default is dag-ml; legacy stays available only through an explicit selector/env override.
    monkeypatch.delenv(ENGINE_ENV_VAR, raising=False)
    assert DEFAULT_ENGINE == "dag-ml"
    assert DEFAULT_EXECUTION_PROFILE == "rollback-capable"
    assert resolve_engine() == "dag-ml"


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


def test_dual_engine_resolves_for_strict_run_dispatch() -> None:
    assert resolve_engine("dual") == "dual"
    assert resolve_engine("  DUAL  ") == "dual"


def test_unknown_engine_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_engine("rust")


@pytest.mark.parametrize("engine", ["dag-ml", "native"])
def test_strict_profile_allows_only_product_engines(engine: str) -> None:
    assert resolve_engine(engine, execution_profile="strict") == engine


@pytest.mark.parametrize("engine", ["legacy", "dual"])
def test_strict_profile_refuses_every_direct_legacy_path(engine: str) -> None:
    with pytest.raises(ExecutionProfileError) as caught:
        resolve_engine(engine, execution_profile="strict")
    assert caught.value.code == "legacy_execution_forbidden"


def test_strict_profile_refuses_environment_selected_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENGINE_ENV_VAR, "legacy")
    with pytest.raises(ExecutionProfileError) as caught:
        resolve_engine(execution_profile="strict")
    assert caught.value.code == "legacy_execution_forbidden"


def test_strict_profile_refuses_legacy_fallback() -> None:
    with pytest.raises(ExecutionProfileError) as caught:
        resolve_engine("dag-ml", execution_profile="strict", allow_fallback=True)
    assert caught.value.code == "legacy_fallback_forbidden"


def test_rollback_profile_retains_explicit_legacy_and_fallback() -> None:
    assert resolve_engine("legacy") == "legacy"
    assert resolve_engine("dag-ml", allow_fallback=True) == "dag-ml"


def test_unknown_execution_profile_is_typed_and_fail_closed() -> None:
    with pytest.raises(ExecutionProfileError) as caught:
        resolve_engine("dag-ml", execution_profile="studio-ish")
    assert caught.value.code == "profile_unknown"
