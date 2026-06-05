"""Unit tests for :class:`TabPFNNIRSRegressor`.

These tests require the ``tabpfn`` package to be importable. The whole
module is skipped when ``tabpfn`` is unavailable so the rest of the test
suite keeps running on environments without the optional dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tabpfn")

from nirs4all.operators.models import TabPFNNIRSRegressor  # noqa: E402


def _fit_or_skip(est: TabPFNNIRSRegressor, X: np.ndarray, y: np.ndarray) -> TabPFNNIRSRegressor:
    """Fit, skipping when TabPFN cannot obtain its model weights here.

    Recent ``tabpfn`` releases gate the weight download behind a one-time license
    acceptance (``TABPFN_TOKEN``); fresh CI runners have neither the token nor a
    weights cache, so ``fit`` raises ``TabPFNLicenseError``. Matched by name so the
    module keeps working with older tabpfn versions that lack the error class.
    """
    try:
        return est.fit(X, y)
    except Exception as exc:
        if type(exc).__name__ == "TabPFNLicenseError":
            pytest.skip("TabPFN license not accepted in this environment (no TABPFN_TOKEN or cached weights)")
        raise


def _make_nirs_like_data(
    n_train: int = 80,
    n_test: int = 40,
    p: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return synthetic spectra (smooth baseline + Gaussian peak) and a
    nonlinear y, sized for quick CPU tests."""
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(0.0, 1.0, p)
    baseline = np.exp(-((wavelengths - 0.5) ** 2) / 0.01)

    def _spectra(n: int) -> np.ndarray:
        return rng.normal(0.5, 0.05, (n, p)) + baseline + 0.1 * rng.standard_normal((n, p))

    peak_idx = int(np.argmax(wavelengths > 0.5))
    X_train = _spectra(n_train)
    X_test = _spectra(n_test)
    y_train = 2.0 * X_train[:, peak_idx] + 0.5 * X_train[:, peak_idx] ** 2 + 0.1 * rng.standard_normal(n_train)
    y_test = 2.0 * X_test[:, peak_idx] + 0.5 * X_test[:, peak_idx] ** 2 + 0.1 * rng.standard_normal(n_test)
    return X_train.astype(np.float64), y_train.astype(np.float64), X_test.astype(np.float64), y_test.astype(np.float64)


def test_default_params() -> None:
    """Default hyperparameters match the empirically-selected recipe."""
    est = TabPFNNIRSRegressor()
    params = est.get_params()
    assert params["n_estimators"] == 16
    assert params["max_features"] == 2000
    assert params["sg_window_length"] == 11
    assert params["sg_polyorder"] == 2
    assert params["sg_deriv"] == 1
    assert params["osc_n_components"] == 1


def test_fit_predict_shape() -> None:
    """fit/predict return a 1D float array of the right length."""
    X_train, y_train, X_test, _y_test = _make_nirs_like_data()
    est = TabPFNNIRSRegressor(n_estimators=4, device="auto")
    fitted = _fit_or_skip(est, X_train, y_train)
    assert fitted is est

    y_pred = est.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],)
    assert np.issubdtype(y_pred.dtype, np.floating)
    assert np.isfinite(y_pred).all()


def test_fit_populates_attributes() -> None:
    """After fit, expected attributes are populated and consistent."""
    X_train, y_train, _X_test, _y_test = _make_nirs_like_data()
    est = _fit_or_skip(TabPFNNIRSRegressor(n_estimators=4, device="auto"), X_train, y_train)

    assert est.n_features_in_ == X_train.shape[1]
    assert est.n_features_used_ <= est.max_features
    assert est.sg_ is not None
    assert est.osc_ is not None
    assert est.scaler_ is not None
    assert est.model_ is not None
    # p=200 is well below max_features=2000, no subsample expected
    assert est.subsample_idx_ is None
    assert est.n_features_used_ == est.n_features_in_


def test_subsample_when_p_exceeds_max_features() -> None:
    """When p > max_features, the cap is applied and predict reuses indices."""
    X_train, y_train, X_test, _y_test = _make_nirs_like_data(p=2200)
    est = TabPFNNIRSRegressor(n_estimators=4, max_features=1500, device="auto")
    _fit_or_skip(est, X_train, y_train)

    assert est.n_features_used_ == 1500
    assert est.subsample_idx_ is not None
    assert len(est.subsample_idx_) == 1500
    assert est.subsample_idx_.min() >= 0
    assert est.subsample_idx_.max() < X_train.shape[1]

    # Predict must not raise and must reuse the cached indices
    y_pred = est.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],)
    assert np.isfinite(y_pred).all()


def test_get_set_params_roundtrip() -> None:
    """``get_params`` / ``set_params`` follow the sklearn contract."""
    est = TabPFNNIRSRegressor()
    expected_keys = {
        "n_estimators",
        "max_features",
        "sg_window_length",
        "sg_polyorder",
        "sg_deriv",
        "osc_n_components",
        "random_state",
        "device",
        "model_path",
    }
    assert set(est.get_params()) == expected_keys

    est.set_params(n_estimators=8, sg_window_length=15, osc_n_components=2)
    assert est.n_estimators == 8
    assert est.sg_window_length == 15
    assert est.osc_n_components == 2

    with pytest.raises(ValueError):
        est.set_params(does_not_exist=42)


def test_clone_compatibility() -> None:
    """``sklearn.base.clone`` reconstructs an estimator with identical params."""
    from sklearn.base import clone

    est = TabPFNNIRSRegressor(n_estimators=8, sg_window_length=15)
    cloned = clone(est)
    assert cloned is not est
    assert cloned.get_params() == est.get_params()
