"""Unit tests for the callback-free strict Methods terminal lowerer."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from nirs4all.pipeline.dagml import terminal_predict_lowerer
from nirs4all.pipeline.dagml.terminal_predict_lowerer import (
    lower_strict_methods_terminal_prediction,
)


def _require_terminal_facade() -> None:
    dag_ml = pytest.importorskip("dag_ml")
    required = {
        "attach_predict_cohort_to_envelope",
        "sign_training_request",
        "sample_relation_set_fingerprint_json",
    }
    if not all(callable(getattr(dag_ml, name, None)) for name in required):
        pytest.skip("installed dag_ml lacks the strict terminal contract helpers")


def _pipeline() -> list[object]:
    return [KFold(n_splits=3, shuffle=False), {"model": PLSRegression(n_components=1)}]


def _training() -> tuple[np.ndarray, np.ndarray, list[str]]:
    return (
        np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]]),
        np.arange(6.0),
        ["fit-z", "fit-a", "fit-y", "fit-b", "fit-x", "fit-c"],
    )


def test_terminal_lowerer_canonicalizes_rows_and_builds_the_closed_v2_contract() -> None:
    _require_terminal_facade()
    X, y, sample_ids = _training()

    execution = lower_strict_methods_terminal_prediction(
        _pipeline(),
        X,
        y,
        sample_ids=sample_ids,
        terminal_predict={
            "X": np.asarray([[7.0, 1.0], [6.0, 0.0]]),
            "sample_ids": ["predict-z", "predict-a"],
        },
        seed=17,
    )

    assert not hasattr(execution, "op_callback")
    assert execution.fit_identity_frame.sample_ids == ("fit-a", "fit-b", "fit-c", "fit-x", "fit-y", "fit-z")
    assert execution.predict_identity_frame.sample_ids == ("predict-a", "predict-z")
    assert execution.methods_inputs["model:terminal.x"]["sample_ids"] == list(execution.fit_identity_frame.sample_ids)
    assert execution.methods_inputs["model:terminal.x"]["x"] == [
        [1.0, 0.0],
        [3.0, 0.0],
        [5.0, 0.0],
        [4.0, 1.0],
        [2.0, 1.0],
        [0.0, 1.0],
    ]
    assert execution.predict_input["sample_ids"] == list(execution.predict_identity_frame.sample_ids)
    assert execution.predict_input["x"] == [[6.0, 0.0], [7.0, 1.0]]
    assert set(execution.predict_input) == {"sample_ids", "x", "target_names"}
    assert execution.request["graph"]["nodes"] == [
        {
            "id": "model:terminal",
            "kind": "model",
            "operator": {"type": "PLSRegression"},
            "params": {"n_components": 1},
            "ports": {
                "inputs": [
                    {
                        "name": "x",
                        "kind": "data",
                        "representation": "tabular_numeric",
                        "cardinality": "one",
                        "description": "",
                    }
                ],
                "outputs": [
                    {
                        "name": "oof",
                        "kind": "prediction",
                        "representation": None,
                        "cardinality": "one",
                        "description": "",
                    }
                ],
            },
            "metadata": {},
            "seed_label": None,
        }
    ]
    options = execution.request["options"]
    assert options["refit_strategy"] == "refit_one"
    assert options["selection"]["evaluation_scope"] == "oof"
    assert options["artifacts"] == {
        "cv_artifacts": "discard",
        "prediction_caches": "discard",
        "fitted_artifacts": "portable_required",
    }
    assert execution.request["graph"]["edges"] == []
    cohort = execution.predict_envelope["predict_cohort"]
    assert execution.predict_envelope["schema_version"] == 2
    assert cohort["role"] == "inference"
    assert cohort["physical_sample_ids"] == ["predict-a", "predict-z"]
    assert cohort["origin_sample_ids"] == ["predict-a", "predict-z"]
    assert "target_content_fingerprint" not in cohort
    assert "target_content_fingerprint" not in execution.predict_envelope


def test_terminal_lowerer_rejects_width_before_importing_or_accessing_native_dagml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y, sample_ids = _training()

    def reached(_module_name: str) -> object:
        raise AssertionError("DAG-ML was imported before terminal X-width preflight")

    monkeypatch.setattr(terminal_predict_lowerer, "_import_dagml", reached)
    with pytest.raises(ValueError, match="feature widths"):
        lower_strict_methods_terminal_prediction(
            _pipeline(),
            X,
            y,
            sample_ids=sample_ids,
            terminal_predict={"X": np.ones((2, 3)), "sample_ids": ["predict-a", "predict-b"]},
            seed=1,
        )


@pytest.mark.parametrize(
    ("X", "y", "terminal_predict"),
    [
        (
            np.asarray(
                [[0.0 + 1.0j, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]]
            ),
            np.arange(6.0),
            {"X": np.ones((2, 2)), "sample_ids": ["predict-a", "predict-b"]},
        ),
        (
            np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]]),
            np.arange(6.0) + 1.0j,
            {"X": np.ones((2, 2)), "sample_ids": ["predict-a", "predict-b"]},
        ),
        (
            np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]]),
            np.arange(6.0),
            {
                "X": np.asarray([[1.0 + 1.0j, 0.0], [2.0, 1.0]]),
                "sample_ids": ["predict-a", "predict-b"],
            },
        ),
    ],
)
def test_terminal_lowerer_refuses_complex_arrays_before_importing_dagml(
    monkeypatch: pytest.MonkeyPatch,
    X: np.ndarray,
    y: np.ndarray,
    terminal_predict: dict[str, object],
) -> None:
    sample_ids = ["fit-z", "fit-a", "fit-y", "fit-b", "fit-x", "fit-c"]

    def reached(_module_name: str) -> object:
        raise AssertionError("DAG-ML was imported before complex-array preflight")

    monkeypatch.setattr(terminal_predict_lowerer, "_import_dagml", reached)
    with pytest.raises(TypeError, match="complex values are unsupported"):
        lower_strict_methods_terminal_prediction(
            _pipeline(),
            X,
            y,
            sample_ids=sample_ids,
            terminal_predict=terminal_predict,
            seed=1,
        )


@pytest.mark.parametrize(
    ("pipeline", "terminal_predict", "message"),
    [
        (
            [KFold(n_splits=3, shuffle=False), object()],
            {"X": np.ones((2, 2)), "sample_ids": ["predict-a", "predict-b"]},
            "PLSRegression",
        ),
        (
            [KFold(n_splits=3, shuffle=True), {"model": PLSRegression(n_components=1)}],
            {"X": np.ones((2, 2)), "sample_ids": ["predict-a", "predict-b"]},
            "shuffle=False",
        ),
        (
            [KFold(n_splits=7, shuffle=False), {"model": PLSRegression(n_components=1)}],
            {"X": np.ones((2, 2)), "sample_ids": ["predict-a", "predict-b"]},
            "cannot split",
        ),
        (
            [KFold(n_splits=3, shuffle=False), {"transform": "snv"}],
            {"X": np.ones((2, 2)), "sample_ids": ["predict-a", "predict-b"]},
            "KFold splitter",
        ),
        (
            [KFold(n_splits=3, shuffle=False), {"model": PLSRegression(n_components=1), "_grid_": [1, 2]}],
            {"X": np.ones((2, 2)), "sample_ids": ["predict-a", "predict-b"]},
            "KFold splitter",
        ),
        (
            _pipeline(),
            {"X": np.ones((2, 2)), "y": [0.0, 1.0], "sample_ids": ["predict-a", "predict-b"]},
            "target-free",
        ),
        (
            _pipeline(),
            {
                "X": np.ones((2, 2)),
                "sample_ids": ["predict-a", "predict-b"],
                "calibration": {"y": [0.0, 1.0]},
                "groups": ["g-a", "g-b"],
                "metadata": {"batch": [1, 2]},
                "external_oof": {"cache": "forbidden"},
            },
            "target-free",
        ),
    ],
)
def test_terminal_lowerer_refuses_unsupported_shapes_before_importing_dagml(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: list[object],
    terminal_predict: dict[str, object],
    message: str,
) -> None:
    X, y, sample_ids = _training()
    monkeypatch.setattr(
        terminal_predict_lowerer,
        "_import_dagml",
        lambda _module_name: (_ for _ in ()).throw(AssertionError("DAG-ML import reached unsupported terminal shape")),
    )

    with pytest.raises((TypeError, ValueError), match=message):
        lower_strict_methods_terminal_prediction(
            pipeline,
            X,
            y,
            sample_ids=sample_ids,
            terminal_predict=terminal_predict,
            seed=1,
        )
