"""Focused cutover tests: native default, fail-closed refusal, explicit legacy."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from nirs4all.api.session import Session
from nirs4all.pipeline.dagml.rt import RtError
from nirs4all.pipeline.engine import (
    DEFAULT_ENGINE,
    DEFAULT_EXECUTION_PROFILE,
    ENGINE_ENV_VAR,
    ExecutionProfileError,
    resolve_engine,
)


def test_defaults_to_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENGINE_ENV_VAR, raising=False)
    assert DEFAULT_ENGINE == "native"
    assert DEFAULT_EXECUTION_PROFILE == "rollback-capable"
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


def test_rollback_profile_refuses_environment_selected_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENGINE_ENV_VAR, "legacy")
    with pytest.raises(ExecutionProfileError) as caught:
        resolve_engine()
    assert caught.value.code == "ambient_legacy_execution_forbidden"


def test_strict_profile_refuses_legacy_fallback() -> None:
    with pytest.raises(ExecutionProfileError) as caught:
        resolve_engine("dag-ml", execution_profile="strict", allow_fallback=True)
    assert caught.value.code == "legacy_fallback_forbidden"


def test_rollback_profile_retains_only_explicit_legacy() -> None:
    assert resolve_engine("legacy") == "legacy"
    with pytest.raises(ExecutionProfileError) as caught:
        resolve_engine("native", allow_fallback=True)
    assert caught.value.code == "legacy_fallback_forbidden"


def test_unknown_execution_profile_is_typed_and_fail_closed() -> None:
    with pytest.raises(ExecutionProfileError) as caught:
        resolve_engine("dag-ml", execution_profile="studio-ish")
    assert caught.value.code == "profile_unknown"


def test_portable_run_and_session_select_native_without_pipeline_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_module = importlib.import_module("nirs4all.api.run")
    native_module = importlib.import_module("nirs4all.api.native_archive_training")
    sentinel = object()
    calls: list[tuple[object, object, object]] = []

    class LegacyPathReached:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("default native run constructed PipelineRunner")

    def native_run(pipeline: object, dataset: object, **kwargs: object) -> object:
        calls.append((pipeline, dataset, kwargs.get("session")))
        return sentinel

    monkeypatch.setattr(run_module, "PipelineRunner", LegacyPathReached)
    monkeypatch.setattr(native_module, "run_native_methods_archive", native_run)

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold

    # Automatic selection requires an actual portable declaration. Empty
    # placeholders are not evidence that the portable profile covers a request.
    pipeline: list[object] = [KFold(n_splits=2), {"model": PLSRegression(n_components=1)}]
    dataset: dict[str, object] = {"X": np.ones((8, 3)), "y": np.arange(8.0), "sample_ids": list(range(8))}
    assert run_module.run(pipeline, dataset) is sentinel

    session = Session(pipeline=pipeline)
    assert session.run(dataset) is sentinel
    assert session._runner is None
    assert calls == [(pipeline, dataset, None), (pipeline, dataset, session)]


def test_explicit_native_refuses_unsupported_shape_without_pipeline_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    class LegacyPathReached:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("unsupported native run constructed PipelineRunner")

    monkeypatch.setattr(run_module, "PipelineRunner", LegacyPathReached)
    with pytest.raises(TypeError, match="requires a list pipeline"):
        run_module.run({"legacy": "shape"}, object(), engine="native")


def test_predict_defaults_to_core_v2_and_legacy_archive_refuses_without_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predict_module = importlib.import_module("nirs4all.api.predict")
    replay_module = importlib.import_module("nirs4all.pipeline.dagml.core_archive_replay")

    class LegacyPathReached:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("default native prediction constructed PipelineRunner")

    monkeypatch.setattr(predict_module, "PipelineRunner", LegacyPathReached)
    monkeypatch.setattr(replay_module, "detect_core_archive_version", lambda _model: 2)
    monkeypatch.setattr(
        replay_module,
        "predict_core_methods_archive_v2",
        lambda *_args, **_kwargs: (np.asarray([1.5]), {"engine": "core"}),
    )
    result = predict_module.predict(model=Path("portable.n4a"), data=np.asarray([[1.0]]))
    np.testing.assert_array_equal(result.y_pred, np.asarray([1.5]))

    monkeypatch.setattr(replay_module, "detect_core_archive_version", lambda _model: None)
    with pytest.raises(RtError) as caught:
        predict_module.predict(model=Path("legacy.n4a"), data=np.asarray([[1.0]]))
    assert caught.value.cause == "unsupported_capability"
    assert caught.value.unsupported_capability == "legacy_archive_conversion_required"


def test_predict_explicit_legacy_remains_available(monkeypatch: pytest.MonkeyPatch) -> None:
    predict_module = importlib.import_module("nirs4all.api.predict")
    sentinel = object()
    monkeypatch.setattr(predict_module, "_predict_from_model", lambda **_kwargs: sentinel)

    assert predict_module.predict(model={"model": "legacy"}, data=object(), engine="legacy") is sentinel


def test_legacy_session_prediction_requires_explicit_selector_before_runner() -> None:
    session = Session()
    session._status = "trained"
    session._bundle_path = Path("legacy.n4a")

    with pytest.raises(RtError) as caught:
        session.predict(object())
    assert caught.value.unsupported_capability == "legacy_archive_conversion_required"
    assert session._runner is None
