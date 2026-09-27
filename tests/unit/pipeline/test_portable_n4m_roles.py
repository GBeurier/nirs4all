"""Version 8 trained envelopes: n4m role recipes with their N4ME states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nirs4all.pipeline.portable_n4m_roles import SCHEMA, PortableN4MRolePipeline

roles = pytest.importorskip("n4m.roles")
N4MError = pytest.importorskip("n4m").N4MError

pytestmark = pytest.mark.methods

RECIPE = {
    "pipeline": [
        {"class": "n4m:filters.y_outlier", "params": {"threshold": 2.0}},
        "n4m:preprocessing.scatter.snv",
        {"class": "n4m:filters.variance", "params": {"top_k": 30}},
        {"class": "n4m:models.pls.kernel", "params": {"n_components": 3, "kernel": "linear"}},
    ]
}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(70, 2))
    X = scores @ rng.normal(size=(2, 40)) + 1.0 + 0.02 * rng.normal(size=(70, 40))
    y = scores[:, 0] - 0.4 * scores[:, 1]
    return X[:50], y[:50], X[50:]


def test_round_trip_predicts_identically(data, tmp_path):
    X, y, X_test = data
    fitted = PortableN4MRolePipeline.fit_recipe(RECIPE, X, y)
    path = tmp_path / "trained.json"
    # Kernel PLS keeps its training rows: sharing them is an explicit choice (F10).
    with pytest.raises(N4MError, match="retains training rows"):
        fitted.to_json(path)
    text = fitted.to_json(path, allow_training_rows=True)
    document = json.loads(text)
    assert document["schema"] == SCHEMA
    assert [(s["method_id"], s["contains_training_rows"]) for s in document["states"]] == [
        ("preprocessing.scatter.snv", False), ("filters.variance", False), ("models.pls.kernel", True),
    ]
    replayed = PortableN4MRolePipeline.from_json(path)
    np.testing.assert_array_equal(replayed.predict(X_test), fitted.predict(X_test))
    assert json.loads(replayed.to_json(allow_training_rows=True)) == document


def test_matches_the_steps_fitted_by_hand(data):
    X, y, X_test = data
    keep = roles.YOutlierFilter(threshold=2.0).fit(X, y).get_mask(X, y)
    snv = roles.SNV().fit(X[keep])
    select = roles.VarianceFilter(top_k=30).fit(snv.transform(X[keep]))
    model = roles.KernelPLS(n_components=3, kernel="linear").fit(select.transform(snv.transform(X[keep])), y[keep])
    expected = model.predict(select.transform(snv.transform(X_test)))
    fitted = PortableN4MRolePipeline.fit_recipe(RECIPE, X, y)
    np.testing.assert_array_equal(fitted.predict(X_test), expected)
    np.testing.assert_array_equal(fitted.retrain(X, y).predict(X_test), expected)


def test_classifier_keeps_label_names(data):
    X, y, X_test = data
    labels = np.where(y > np.median(y), "high", "low")
    recipe = {"pipeline": ["n4m:preprocessing.scatter.snv", "n4m:models.classification.pls_qda"]}
    fitted = PortableN4MRolePipeline.fit_recipe(recipe, X, labels)
    replayed = PortableN4MRolePipeline.from_json(fitted.to_json())
    assert set(replayed.predict(X_test)) <= {"high", "low"}
    np.testing.assert_array_equal(replayed.predict(X_test), fitted.predict(X_test))


def test_dataframe_columns_are_checked_after_replay(data):
    """F03: a DataFrame with the same columns in another order is refused, not silently re-read."""
    X, y, X_test = data
    columns = [f"nm{1000 + 2 * i}" for i in range(X.shape[1])]
    recipe = {"pipeline": ["n4m:preprocessing.scatter.snv", {"class": "n4m:models.pls.cppls", "params": {"n_components": 2}}]}
    fitted = PortableN4MRolePipeline.fit_recipe(recipe, pd.DataFrame(X, columns=columns), y)
    document = json.loads(fitted.to_json())
    assert document["feature_names"] == columns
    replayed = PortableN4MRolePipeline.from_json(json.dumps(document))
    frame = pd.DataFrame(X_test, columns=columns)
    np.testing.assert_array_equal(replayed.predict(frame), fitted.predict(frame))
    with pytest.raises(N4MError, match="reordered"):
        replayed.predict(frame[columns[::-1]])
    with pytest.raises(N4MError, match="columns"):
        replayed.predict(frame[columns[:-1]])


def test_recipe_contradicting_its_states_is_refused(data):
    """F05: the recipe of an envelope must describe the model it carries."""
    X, y, _ = data
    recipe = {"pipeline": [{"class": "n4m:models.pls.cppls", "params": {"n_components": 2}}]}
    document = json.loads(PortableN4MRolePipeline.fit_recipe(recipe, X, y).to_json())
    document["recipe"]["pipeline"][0]["params"]["n_components"] = 1
    with pytest.raises(N4MError, match="'n_components' is 2 in the state but 1 in the recipe"):
        PortableN4MRolePipeline.from_json(json.dumps(document))
    document["recipe"]["pipeline"], document["states"] = [], []
    with pytest.raises(N4MError, match="at least one step"):
        PortableN4MRolePipeline.from_json(json.dumps(document))


def test_multi_target_reaches_supervised_transformers(data):
    """F06: every target column reaches the PLS transformer before the final PLS."""
    X, y, X_test = data
    Y = np.column_stack([y, 0.5 * y + X[:, 0]])
    pls = {"class": "n4m:models.pls.pls_regression", "params": {"n_components": 2}}
    fitted = PortableN4MRolePipeline.fit_recipe({"pipeline": [pls, pls]}, X, Y)
    first = roles.PLSRegression(n_components=2).fit(X, Y)
    second = roles.PLSRegression(n_components=2).fit(first.transform(X), Y)
    expected = second.predict(first.transform(X_test))
    predicted = fitted.predict(X_test)
    assert predicted.shape == (len(X_test), 2)
    np.testing.assert_array_equal(predicted, expected)
    np.testing.assert_array_equal(PortableN4MRolePipeline.from_json(fitted.to_json()).predict(X_test), predicted)


def test_rejects_tampered_or_foreign_envelopes(data):
    X, y, _ = data
    document = json.loads(PortableN4MRolePipeline.fit_recipe(RECIPE, X, y).to_json(allow_training_rows=True))
    document["states"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        PortableN4MRolePipeline.from_json(json.dumps(document))
    with pytest.raises(ValueError, match="n4m:<method id>"):
        PortableN4MRolePipeline.fit_recipe({"pipeline": ["sklearn.cross_decomposition.PLSRegression"]}, X, y)
    with pytest.raises(N4MError, match="ends with one regressor"):
        PortableN4MRolePipeline.fit_recipe({"pipeline": ["n4m:preprocessing.scatter.snv"]}, X, y)
    with pytest.raises(N4MError, match="at least one step"):
        PortableN4MRolePipeline.fit_recipe({"pipeline": []}, X, y)


def test_r_trained_envelope_predicts_identically_in_python():
    """Fixture from nirs4all-r tests/helpers/trained_roles_v8_fixture.R (written before the additive v8 fields)."""
    fixtures = Path(__file__).resolve().parents[2] / "fixtures"
    oracle = json.loads((fixtures / "portable_roles_v8_r_oracle.json").read_text(encoding="utf-8"))
    replayed = PortableN4MRolePipeline.from_json(fixtures / "portable_roles_v8_r_envelope.json")
    np.testing.assert_allclose(replayed.predict(oracle["x_test"]), oracle["predict"], rtol=1e-12, atol=1e-12)
    refit = replayed.retrain(np.asarray(oracle["x_train"]), np.asarray(oracle["y_train"]))
    np.testing.assert_allclose(refit.predict(oracle["x_test"]), oracle["predict"], rtol=1e-9, atol=1e-9)
