"""Unit tests for the R2 native-default backend selector."""

from __future__ import annotations

import pytest

from nirs4all.pipeline.engine import DEFAULT_ENGINE, ENGINE_ENV_VAR, DualRunUnsupported, require_legacy_engine, resolve_engine


def test_defaults_to_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENGINE_ENV_VAR, raising=False)
    assert DEFAULT_ENGINE == "native"
    assert resolve_engine() == "native"


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


def test_dual_engine_resolves_for_run_oracle() -> None:
    assert resolve_engine("dual") == "dual"


def test_native_engine_resolves_for_the_explicit_archive_predict_subset() -> None:
    assert resolve_engine("native") == "native"
    with pytest.raises(NotImplementedError, match="engine='native'"):
        require_legacy_engine("retrain", "native")


def test_non_run_helpers_refuse_dual_engine() -> None:
    with pytest.raises(DualRunUnsupported, match="only for nirs4all.run"):
        require_legacy_engine("predict", "dual")


def test_unknown_engine_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_engine("rust")
