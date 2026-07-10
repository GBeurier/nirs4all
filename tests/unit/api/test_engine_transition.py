"""Transition-release backend selector coverage for public helper APIs."""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.api.explain import explain
from nirs4all.api.predict import predict
from nirs4all.api.retrain import retrain
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
    ("operation", "call"),
    [
        (
            "predict",
            lambda: predict(model={"model_name": "dummy"}, data=np.zeros((2, 3))),
        ),
        (
            "explain",
            lambda: explain({"model_name": "dummy"}, np.zeros((2, 3))),
        ),
        (
            "retrain",
            lambda: retrain({"model_name": "dummy"}, (np.zeros((2, 3)), np.zeros(2))),
        ),
    ],
)
def test_public_helpers_honor_dagml_env_boundary(monkeypatch: pytest.MonkeyPatch, operation: str, call) -> None:
    monkeypatch.setenv("N4A_ENGINE", "dag-ml")

    with pytest.raises(NotImplementedError, match=rf"nirs4all\.{operation}.*dag-ml"):
        call()
