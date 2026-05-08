"""Tests for FCKResidualRegressor.

Cover the do-no-harm contract (s=0 produces teacher-only predictions),
the held-out shrinkage CV selection, OOF teacher fold composition, and
the diagnostics exposed via ``last_diagnostics_``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

from nirs4all.operators.transforms import FCKStaticTransformer

# fck_residual.py lives in `bench/fck_pls/` (sibling of the tests dir).
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from fck_residual import FCKResidualRegressor  # noqa: E402


def _toy_data(n_train=64, n_test=24, p=120, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_train + n_test, p).astype(np.float64)
    beta = rng.randn(p) * 0.05
    y = X @ beta + 0.1 * rng.randn(n_train + n_test)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


class TestFCKResidualRegressor:
    def test_basic_fit_predict(self):
        X_train, y_train, X_test, y_test = _toy_data(seed=0)
        model = FCKResidualRegressor(
            teacher=PLSRegression(n_components=5),
            fck=FCKStaticTransformer(alphas=(1.0,), scales=(1,), kernel_sizes=(15,)),
            residual_head=Ridge(alpha=1.0),
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == y_test.shape

    def test_diagnostics_present(self):
        X_train, y_train, _, _ = _toy_data(seed=0)
        model = FCKResidualRegressor(
            teacher=PLSRegression(n_components=5),
            fck=FCKStaticTransformer(alphas=(1.0,), scales=(1,), kernel_sizes=(15,)),
        )
        model.fit(X_train, y_train)
        diag = model.last_diagnostics_
        for key in (
            "shrinkage_s_star",
            "shrinkage_inner_rmse_per_s",
            "teacher_calibration_rmse",
            "calibration_rmse_at_s_star",
            "catastrophic",
        ):
            assert key in diag
        assert model.shrinkage_s_star_ in (0.0, 0.25, 0.5, 0.75, 1.0)
        for s in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert s in model.shrinkage_inner_rmse_per_s_

    def test_zero_in_grid_required(self):
        X_train, y_train, _, _ = _toy_data(seed=0)
        model = FCKResidualRegressor(
            teacher=PLSRegression(n_components=5),
            shrinkage_grid=(0.5, 1.0),
        )
        with pytest.raises(ValueError, match="0.0"):
            model.fit(X_train, y_train)

    def test_do_no_harm_under_random_target(self):
        """When residual signal is noise, the calibration RMSE at s=0 should
        not be obviously beaten — many seeds will pick s* = 0. Test the
        contract: prediction at s* = 0 equals teacher-only prediction."""
        X_train, y_train, X_test, _ = _toy_data(seed=42)
        # Replace y with pure noise to break any residual signal.
        y_train = np.random.RandomState(42).randn(len(y_train))
        model = FCKResidualRegressor(
            teacher=PLSRegression(n_components=5),
            fck=FCKStaticTransformer(alphas=(1.0,), scales=(1,), kernel_sizes=(15,)),
            residual_head=Ridge(alpha=1.0),
            random_state=42,
        )
        model.fit(X_train, y_train)
        # If s* = 0, predictions equal the teacher-only predictions.
        if model.shrinkage_s_star_ == 0.0:
            teacher_only = model.teacher_.predict(X_test).ravel()
            np.testing.assert_allclose(model.predict(X_test), teacher_only)

    def test_oof_teacher_uses_n_folds(self):
        X_train, y_train, _, _ = _toy_data(seed=0)
        model = FCKResidualRegressor(
            teacher=PLSRegression(n_components=5),
            oof_n_folds=4,
        )
        z_oof = model._oof_teacher_predictions(X_train, y_train)
        # Each row must have a prediction (no missing).
        assert z_oof.shape == y_train.shape
        assert not np.any(np.isnan(z_oof))

    def test_predict_before_fit_raises(self):
        model = FCKResidualRegressor(teacher=PLSRegression(n_components=5))
        with pytest.raises(RuntimeError, match="fit must be called"):
            model.predict(np.zeros((1, 10)))

    def test_random_state_determinism(self):
        X_train, y_train, X_test, _ = _toy_data(seed=0)
        m1 = FCKResidualRegressor(
            teacher=PLSRegression(n_components=5),
            fck=FCKStaticTransformer(alphas=(1.0,), scales=(1,), kernel_sizes=(15,)),
            random_state=7,
        ).fit(X_train, y_train)
        m2 = FCKResidualRegressor(
            teacher=PLSRegression(n_components=5),
            fck=FCKStaticTransformer(alphas=(1.0,), scales=(1,), kernel_sizes=(15,)),
            random_state=7,
        ).fit(X_train, y_train)
        np.testing.assert_array_equal(m1.predict(X_test), m2.predict(X_test))
        assert m1.shrinkage_s_star_ == m2.shrinkage_s_star_

    def test_oof_n_folds_below_two_rejected(self):
        X_train, y_train, _, _ = _toy_data(seed=0)
        model = FCKResidualRegressor(
            teacher=PLSRegression(n_components=5),
            oof_n_folds=1,
        )
        with pytest.raises(ValueError, match="oof_n_folds"):
            model.fit(X_train, y_train)

    def test_default_residual_head_is_ridge(self):
        X_train, y_train, X_test, _ = _toy_data(seed=0)
        model = FCKResidualRegressor(
            teacher=PLSRegression(n_components=5),
            fck=FCKStaticTransformer(alphas=(1.0,), scales=(1,), kernel_sizes=(15,)),
        )
        model.fit(X_train, y_train)
        # Smoke: prediction returns the right shape
        assert model.predict(X_test).shape == (len(X_test),)
