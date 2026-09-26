"""Version 8 trained envelopes: n4m role recipes with their N4ME states."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nirs4all.pipeline.portable_n4m_roles import SCHEMA, PortableN4MRolePipeline

roles = pytest.importorskip("n4m.roles")

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
    text = fitted.to_json(path)
    document = json.loads(text)
    assert document["schema"] == SCHEMA
    assert [s["method_id"] for s in document["states"]] == [
        "preprocessing.scatter.snv", "filters.variance", "models.pls.kernel",
    ]
    replayed = PortableN4MRolePipeline.from_json(path)
    np.testing.assert_array_equal(replayed.predict(X_test), fitted.predict(X_test))
    assert json.loads(replayed.to_json()) == document


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


def test_rejects_tampered_or_foreign_envelopes(data):
    X, y, _ = data
    document = json.loads(PortableN4MRolePipeline.fit_recipe(RECIPE, X, y).to_json())
    document["states"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        PortableN4MRolePipeline.from_json(json.dumps(document))
    with pytest.raises(ValueError, match="n4m:<method id>"):
        PortableN4MRolePipeline.fit_recipe({"pipeline": ["sklearn.cross_decomposition.PLSRegression"]}, X, y)
    with pytest.raises(ValueError, match="ends with one regressor"):
        PortableN4MRolePipeline.fit_recipe({"pipeline": ["n4m:preprocessing.scatter.snv"]}, X, y)


def test_r_trained_envelope_predicts_identically_in_python():
    """Fixture from nirs4all-r tests/helpers/trained_roles_v8_fixture.R."""
    from pathlib import Path

    fixtures = Path(__file__).resolve().parents[2] / "fixtures"
    oracle = json.loads((fixtures / "portable_roles_v8_r_oracle.json").read_text(encoding="utf-8"))
    replayed = PortableN4MRolePipeline.from_json(fixtures / "portable_roles_v8_r_envelope.json")
    np.testing.assert_allclose(replayed.predict(oracle["x_test"]), oracle["predict"], rtol=1e-12, atol=1e-12)
    refit = replayed.retrain(np.asarray(oracle["x_train"]), np.asarray(oracle["y_train"]))
    np.testing.assert_allclose(refit.predict(oracle["x_test"]), oracle["predict"], rtol=1e-9, atol=1e-9)
