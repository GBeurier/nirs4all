"""Callbacks remain executable references through JSON and real model pipelines."""
import json
from functools import partial

import numpy as np
import pytest
from sklearn.ensemble import StackingRegressor
from sklearn.feature_selection import SelectFdr, SequentialFeatureSelector, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component


def roundtrip(component):
    return deserialize_component(json.loads(json.dumps(serialize_component(component))))


@pytest.mark.parametrize("reference", [f_regression, mean_absolute_error])
def test_callback_roundtrip_preserves_function_identity(reference):
    assert roundtrip(reference) is reference
    assert deserialize_component(f"{reference.__module__}.{reference.__name__}") is reference


def test_partial_callback_roundtrip_preserves_bound_arguments():
    callback = partial(f_regression, center=False)
    restored = roundtrip(callback)
    assert restored.func is f_regression and restored.keywords == {"center": False}
    X = np.random.default_rng(31).normal(size=(32, 4))
    y = 2 * X[:, 0] + X[:, 1]
    np.testing.assert_allclose(restored(X, y), callback(X, y))


def test_scorer_roundtrip_preserves_sign_function_and_options():
    scorer = make_scorer(mean_absolute_error, greater_is_better=False, multioutput="uniform_average")
    restored = roundtrip(scorer)
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10) ** 2
    estimator = Ridge().fit(X, y)
    assert restored(estimator, X, y) == pytest.approx(scorer(estimator, X, y))
    assert restored(estimator, X, y) < 0


def test_callback_and_scorer_survive_nested_multiple_estimators():
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    estimator = StackingRegressor(estimators=[
        ("selected", Pipeline([("select", SelectFdr(f_regression)), ("ridge", Ridge())])),
        ("sequential", Pipeline([("select", SequentialFeatureSelector(
            Ridge(), n_features_to_select=2, scoring=scorer, cv=2)), ("linear", LinearRegression())])),
    ], final_estimator=Ridge(), cv=2)
    restored = roundtrip(estimator)
    X = np.random.default_rng(43).normal(size=(48, 4))
    y = 3 * X[:, 0] - X[:, 1]
    expected = estimator.fit(X, y).predict(X)
    actual = restored.fit(X, y).predict(X)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_select_fdr_regression_callback_executes_and_replays_two_models(tmp_path):
    import nirs4all
    from nirs4all.pipeline.storage import WorkspaceStore

    X = np.random.default_rng(117).normal(size=(48, 6)).astype("float32")
    y = 4 * X[:, 0] - 2 * X[:, 1] + .01 * X[:, 2]
    workspace = tmp_path / "workspace"
    nirs4all.run([SelectFdr(score_func=f_regression, alpha=.2), KFold(2),
                  Ridge(), LinearRegression()], (X, y), workspace_path=workspace,
                 engine="legacy", verbose=0)
    with WorkspaceStore(workspace) as store:
        rows = store.query_predictions().to_dicts()
        assert {row["model_name"] for row in rows} == {"Ridge", "LinearRegression"}
        for chain_id in {row["chain_id"] for row in rows}:
            direct = np.asarray(store.replay_chain(chain_id, X[:8])).ravel()
            archive = store.export_chain(chain_id, tmp_path / f"{chain_id}.n4a")
            replay = nirs4all.predict(archive, X[:8], engine="legacy", verbose=0)
            np.testing.assert_allclose(np.asarray(replay.y_pred).ravel(), direct, atol=1e-5)
