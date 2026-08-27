"""Transition-release backend selector coverage for public helper APIs."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from nirs4all.api.explain import explain
from nirs4all.api.predict import predict
from nirs4all.api.retrain import retrain
from nirs4all.pipeline.dagml.errors import DagMlUnsupported
from nirs4all.pipeline.engine import require_legacy_engine


def test_require_legacy_engine_accepts_legacy() -> None:
    assert require_legacy_engine("predict", "legacy") == "legacy"


@pytest.mark.parametrize(
    ("operation", "call"),
    [
        (
            "predict",
            lambda: predict(model={"model_name": "dummy"}, data=np.zeros((2, 3)), engine="dag-ml"),
        ),
        (
            "predict",
            lambda: predict(chain_id="chain-1", data=np.zeros((2, 3)), engine="dag-ml"),
        ),
        (
            "explain",
            lambda: explain({"model_name": "dummy"}, np.zeros((2, 3)), engine="dag-ml"),
        ),
        (
            "retrain",
            lambda: retrain({"model_name": "dummy"}, (np.zeros((2, 3)), np.zeros(2)), engine="dag-ml"),
        ),
    ],
)
def test_public_helpers_reject_dagml_until_native_paths_exist(operation: str, call) -> None:
    with pytest.raises(NotImplementedError, match=rf"nirs4all\.{operation}.*dag-ml"):
        call()


@pytest.mark.parametrize(
    "operation",
    [
        "predict",
        "explain",
        "retrain",
    ],
)
def test_public_helpers_ignore_dagml_env_with_warning(monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    monkeypatch.setenv("N4A_ENGINE", "dag-ml")

    with pytest.warns(RuntimeWarning, match=rf"N4A_ENGINE=dag-ml.*nirs4all\.{operation}.*legacy"):
        assert require_legacy_engine(operation) == "legacy"


def test_dagml_strict_mode_refuses_without_starting_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The migration qualifier can ask for native-only execution explicitly."""
    run_module = importlib.import_module("nirs4all.api.run")
    import nirs4all.pipeline.dagml.run_backend as run_backend

    def _native_refusal(*_args: object, **_kwargs: object) -> object:
        raise DagMlUnsupported("unsupported test shape")

    def _legacy_must_not_start(*_args: object, **_kwargs: object) -> object:
        pytest.fail("strict dag-ml mode must not create a legacy runner")

    monkeypatch.setattr(run_backend, "run_via_dagml", _native_refusal)
    monkeypatch.setattr(run_module, "PipelineRunner", _legacy_must_not_start)

    with pytest.raises(DagMlUnsupported, match="unsupported test shape"):
        run_module.run(
            pipeline=[],
            dataset={"X": np.zeros((2, 1)), "y": np.zeros(2)},
            engine="dag-ml",
            allow_legacy_fallback=False,
            verbose=0,
        )


def test_dagml_fallback_flag_must_be_boolean() -> None:
    run_module = importlib.import_module("nirs4all.api.run")

    with pytest.raises(TypeError, match="allow_legacy_fallback must be a bool"):
        run_module.run(
            pipeline=[],
            dataset={"X": np.zeros((2, 1)), "y": np.zeros(2)},
            engine="dag-ml",
            allow_legacy_fallback="never",  # type: ignore[arg-type]
            verbose=0,
        )
