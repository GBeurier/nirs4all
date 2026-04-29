"""Tests for AOMPLSRegressor and POPPLSRegressor."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from aompls.banks import compact_bank
from aompls.estimators import AOMPLSRegressor, POPPLSRegressor
from aompls.operators import IdentityOperator
from aompls.synthetic import make_regression


@pytest.fixture
def regression_data():
    return make_regression(n_train=70, n_test=30, p=80, random_state=2)


def test_aom_fit_predict_shapes(regression_data):
    ds = regression_data
    est = AOMPLSRegressor(max_components=5, criterion="covariance")
    est.fit(ds.X_train, ds.y_train)
    pred = est.predict(ds.X_test)
    assert pred.shape == (ds.X_test.shape[0],)


def test_aom_default_runs_cv(regression_data):
    ds = regression_data
    est = AOMPLSRegressor(max_components=4, criterion="cv", cv=3)
    est.fit(ds.X_train, ds.y_train)
    pred = est.predict(ds.X_test)
    assert pred.shape == (ds.X_test.shape[0],)
    diag = est.get_diagnostics()
    assert "engine" in diag
    assert diag["engine"] == "simpls_covariance"


def test_pop_default_runs_cv(regression_data):
    ds = regression_data
    est = POPPLSRegressor(max_components=4, criterion="cv", cv=3)
    est.fit(ds.X_train, ds.y_train)
    pred = est.predict(ds.X_test)
    assert pred.shape == (ds.X_test.shape[0],)
    seq = est.get_selected_operators()
    assert len(seq) >= 1


def test_identity_only_aom_close_to_pls(regression_data):
    """AOM with identity-only bank reduces to standard PLS at the same K."""
    ds = regression_data
    bank = [IdentityOperator(p=ds.X_train.shape[1])]
    est = AOMPLSRegressor(
        n_components=5,
        max_components=5,
        engine="simpls_covariance",
        operator_bank=bank,
        criterion="covariance",
    )
    est.fit(ds.X_train, ds.y_train)
    ref = PLSRegression(n_components=5, scale=False)
    ref.fit(ds.X_train, ds.y_train)
    pred_aom = est.predict(ds.X_test)
    pred_ref = ref.predict(ds.X_test).ravel()
    assert np.allclose(pred_aom, pred_ref, atol=1e-3)


def test_max_components_respected(regression_data):
    ds = regression_data
    est = AOMPLSRegressor(max_components=3, criterion="covariance", n_components="auto")
    est.fit(ds.X_train, ds.y_train)
    assert est.n_components_ <= 3


def test_get_params_set_params():
    est = AOMPLSRegressor()
    params = est.get_params()
    assert "selection" in params
    assert "operator_bank" in params
    est.set_params(max_components=7)
    assert est.get_params()["max_components"] == 7


def test_score_returns_r2(regression_data):
    ds = regression_data
    est = AOMPLSRegressor(max_components=4, criterion="covariance")
    est.fit(ds.X_train, ds.y_train)
    score = est.score(ds.X_test, ds.y_test)
    assert isinstance(score, float)


def test_diagnostics_contain_operator_names(regression_data):
    ds = regression_data
    est = AOMPLSRegressor(max_components=3, criterion="covariance")
    est.fit(ds.X_train, ds.y_train)
    diag = est.get_diagnostics()
    assert isinstance(diag["selected_operator_names"], list)
    assert len(diag["selected_operator_names"]) == est.n_components_


def test_pop_per_component_sequence(regression_data):
    ds = regression_data
    est = POPPLSRegressor(max_components=4, criterion="covariance", n_components=4)
    est.fit(ds.X_train, ds.y_train)
    assert len(est.selected_operator_indices_) == est.n_components_
