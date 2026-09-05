"""Composite sklearn models retain their actual constructor graph through DAG."""

import json

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all.pipeline.dagml.operator_parameters import decode_constructor_value, encode_constructor_value
from nirs4all.pipeline.dagml.operator_routing import route_operator
from nirs4all.pipeline.dagml_bridge import _json_safe_params, _qualname


@pytest.mark.parametrize("model", [
    StackingRegressor([("first", Ridge(2)), ("second", Ridge(3))], final_estimator=Ridge(0.2), cv=3),
    VotingRegressor([("first", Ridge(2)), ("second", Ridge(3))], weights=[1, 2]),
    Pipeline([("scale", StandardScaler()), ("model", Ridge(2))]),
    MultiOutputRegressor(Ridge(2)),
])
def test_nested_estimators_roundtrip_without_constructor_aliases(model):
    params = _json_safe_params(model)
    assert not any("__" in key for key in params)
    restored = route_operator("model", _qualname(model), params)
    assert type(restored) is type(model)
    assert restored.get_params(deep=False).keys() == model.get_params(deep=False).keys()
    X = np.random.default_rng(1).normal(size=(24, 5))
    y = np.column_stack([X[:, 0], X[:, 1]]) if isinstance(model, MultiOutputRegressor) else X[:, 0]
    np.testing.assert_allclose(restored.fit(X, y).predict(X), clone(model).fit(X, y).predict(X), rtol=0, atol=0)


def test_nested_sweep_uses_set_params_and_keeps_input_immutable():
    model = StackingRegressor([("base", Ridge())], final_estimator=Ridge(), cv=3)
    params = _json_safe_params(model)
    before = json.dumps(params, sort_keys=True)
    restored = route_operator("model", _qualname(model), params, variant_overrides={"final_estimator__alpha": 0.25})
    assert restored.final_estimator.alpha == 0.25
    assert model.final_estimator.alpha == 1
    assert json.dumps(params, sort_keys=True) == before


def test_plain_parameters_and_marker_like_user_mappings_are_preserved():
    plain = Ridge(alpha=2).get_params()
    assert _json_safe_params(Ridge(alpha=2)) == json.loads(json.dumps(plain))
    value = {"__nirs4all_constructor_component_v1__": {"kind": "component", "value": "sklearn.linear_model.Ridge"}}
    assert decode_constructor_value(json.loads(json.dumps(encode_constructor_value(value)))) == value
    assert decode_constructor_value("sklearn.linear_model.Ridge") == "sklearn.linear_model.Ridge"


@pytest.mark.parametrize("classification", [False, True])
def test_public_stacking_executes_real_dag_and_captures_fitted_model(classification, monkeypatch, tmp_path):
    import nirs4all
    from nirs4all.pipeline.runner import PipelineRunner

    def forbidden(*args, **kwargs):
        raise AssertionError("implicit legacy execution")

    monkeypatch.setattr(PipelineRunner, "run", forbidden)
    X = np.random.default_rng(314159).normal(size=(48, 30))
    y = X[:, 0] * 2 - X[:, 1]
    model = StackingRegressor([("first", Ridge(1)), ("second", Ridge(2))], final_estimator=Ridge(0.2), cv=3)
    if classification:
        y = (y > np.median(y)).astype(int)
        model = StackingClassifier([("first", LogisticRegression(C=1)), ("second", LogisticRegression(C=2))], final_estimator=LogisticRegression(), cv=3)
    result = nirs4all.run([StandardScaler(), KFold(3, shuffle=True, random_state=42), model], (X, y),
                          workspace_path=tmp_path, verbose=0, save_charts=False, random_state=42)
    try:
        assert result.execution_engine == "dag-ml"
        assert result._dagml_score_set is not None
        assert result._dagml_refit_artifacts
        assert np.isfinite(result.cv_best_score)
        import joblib

        result.export_model(tmp_path / "stacking.joblib")
        fitted = joblib.load(tmp_path / "stacking.joblib")
        assert np.isfinite(fitted.predict(X)).all()
    finally:
        result.close()
