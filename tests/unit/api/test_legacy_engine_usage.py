"""CUT-002 visibility contract for explicitly selected legacy-bearing engines."""

from __future__ import annotations

import importlib
import json
import warnings
from typing import Any

import pytest

from nirs4all.api.session import Session
from nirs4all.pipeline.engine import (
    LEGACY_USAGE_COUNTER_ENV_VAR,
    DualRunUnsupported,
    ExecutionProfileError,
    LegacyEngineUsageWarning,
    get_legacy_engine_usage_counts,
)

run_api = importlib.import_module("nirs4all.api.run")
predict_api = importlib.import_module("nirs4all.api.predict")
native_training = importlib.import_module("nirs4all.api.native_archive_training")


class _StoppedBeforeExecution(RuntimeError):
    """Sentinel proving a dispatch boundary was reached without real work."""


def _legacy_warnings(caught: list[warnings.WarningMessage]) -> list[LegacyEngineUsageWarning]:
    return [warning.message for warning in caught if isinstance(warning.message, LegacyEngineUsageWarning)]


def test_explicit_legacy_and_dual_requests_warn_and_increment_opt_in_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(LEGACY_USAGE_COUNTER_ENV_VAR, "1")
    baseline = get_legacy_engine_usage_counts()
    monkeypatch.setattr(
        run_api,
        "PipelineRunner",
        lambda **_kwargs: (_ for _ in ()).throw(_StoppedBeforeExecution()),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(_StoppedBeforeExecution):
            run_api.run(object(), object(), engine="legacy")
        with pytest.raises(DualRunUnsupported):
            run_api.run(object(), object(), engine="dual")
        with pytest.raises(ValueError, match="No pipeline defined"):
            Session().run(object(), engine="legacy")
        with pytest.raises(ValueError, match="must be trained"):
            Session().predict(object(), engine="legacy")
        with pytest.raises(ValueError, match="either 'model' or 'chain_id'"):
            predict_api.predict(engine="legacy")

    diagnostics = [warning.diagnostic for warning in _legacy_warnings(caught)]
    assert diagnostics == [
        {"schema_version": 1, "code": "nirs4all.explicit_legacy_engine", "engine": "legacy", "operation": "run"},
        {"schema_version": 1, "code": "nirs4all.explicit_legacy_engine", "engine": "dual", "operation": "run"},
        {"schema_version": 1, "code": "nirs4all.explicit_legacy_engine", "engine": "legacy", "operation": "Session.run"},
        {"schema_version": 1, "code": "nirs4all.explicit_legacy_engine", "engine": "legacy", "operation": "Session.predict"},
        {"schema_version": 1, "code": "nirs4all.explicit_legacy_engine", "engine": "legacy", "operation": "predict"},
    ]
    assert [json.loads(str(warning)) for warning in _legacy_warnings(caught)] == diagnostics

    counts = get_legacy_engine_usage_counts()
    assert counts["legacy"] - baseline["legacy"] == 4
    assert counts["dual"] - baseline["dual"] == 1
    assert counts["total"] - baseline["total"] == 5


def test_counter_is_disabled_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LEGACY_USAGE_COUNTER_ENV_VAR, raising=False)
    baseline = get_legacy_engine_usage_counts()
    monkeypatch.setattr(
        run_api,
        "PipelineRunner",
        lambda **_kwargs: (_ for _ in ()).throw(_StoppedBeforeExecution()),
    )

    with pytest.warns(LegacyEngineUsageWarning):
        with pytest.raises(_StoppedBeforeExecution):
            run_api.run(object(), object(), engine="legacy")

    assert get_legacy_engine_usage_counts() == baseline


def test_native_defaults_and_strict_preflight_are_silent_and_never_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Untouchable:
        def __getattribute__(self, name: str) -> Any:
            raise AssertionError(f"input was inspected through {name}")

    monkeypatch.setenv(LEGACY_USAGE_COUNTER_ENV_VAR, "1")
    baseline = get_legacy_engine_usage_counts()

    def fail_native(*_args: Any, **_kwargs: Any) -> None:
        raise _StoppedBeforeExecution("native failure")

    def legacy_runner_forbidden(**_kwargs: Any) -> None:
        raise AssertionError("native exception reached PipelineRunner")

    monkeypatch.setattr(native_training, "run_native_methods_archive", fail_native)
    monkeypatch.setattr(run_api, "PipelineRunner", legacy_runner_forbidden)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(_StoppedBeforeExecution, match="native failure"):
            # Arbitrary objects are not a portable request. Keep this test on
            # the explicit strict profile; automatic capability selection is
            # covered by test_run_selection.py with valid declarations.
            run_api.run(object(), object(), engine="native")
        with pytest.raises(ValueError, match="either 'model' or 'chain_id'"):
            predict_api.predict()
        with pytest.raises(ValueError, match="No pipeline defined"):
            Session().run(object())
        with pytest.raises(ValueError, match="must be trained"):
            Session().predict(object())
        with pytest.raises(ExecutionProfileError) as error:
            run_api._run_strict_product(
                Untouchable(),
                Untouchable(),
                engine="legacy",
                allow_fallback=False,
            )

    assert error.value.code == "legacy_execution_forbidden"
    assert _legacy_warnings(caught) == []
    assert get_legacy_engine_usage_counts() == baseline
