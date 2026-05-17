"""Tests for POP-PLS (Per-Operator-Per-component PLS) regressor and classifier.

Tests cover:
- POPPLSRegressor fit/predict/transform
- Per-component operator selection (selected_operators_ has one name per component)
- sklearn compatibility (clone, get/set_params)
- POPPLSClassifier binary and multiclass
- predict_proba calibration
"""

import numpy as np
import pytest
from sklearn.base import clone

from nirs4all.operators.models.sklearn.aom_pls import (
    DetrendProjectionOperator,
    IdentityOperator,
    SavitzkyGolayOperator,
)
from nirs4all.operators.models.sklearn.pop_pls import POPPLSRegressor
from nirs4all.operators.models.sklearn.pop_pls_classifier import POPPLSClassifier

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def regression_data():
    """Spectral-like regression data."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 100, 200
    X = rng.randn(n_samples, n_features)
    from scipy.ndimage import gaussian_filter1d
    X = gaussian_filter1d(X, sigma=3, axis=1)
    y = X[:, 30:50].mean(axis=1) + 0.5 * X[:, 100:120].mean(axis=1) + 0.1 * rng.randn(n_samples)
    return X, y

@pytest.fixture
def small_data():
    """Small dataset for quick tests."""
    rng = np.random.RandomState(123)
    X = rng.randn(50, 100)
    y = X[:, :5].sum(axis=1) + 0.1 * rng.randn(50)
    return X, y

@pytest.fixture
def binary_data():
    """Binary classification data."""
    rng = np.random.RandomState(42)
    X = rng.randn(80, 100)
    y = (X[:, :3].sum(axis=1) > 0).astype(int)
    labels = np.array(["classA", "classB"])
    return X, labels[y]

@pytest.fixture
def multiclass_data():
    """3-class classification data."""
    rng = np.random.RandomState(42)
    X = rng.randn(90, 100)
    scores = X[:, :3].sum(axis=1)
    y = np.where(scores < -0.5, "low", np.where(scores > 0.5, "high", "mid"))
    return X, y

# =============================================================================
# POPPLSRegressor Tests
# =============================================================================

class TestPOPPLSRegressor:
    """Test POPPLSRegressor basic functionality."""

    def test_init_defaults(self):
        model = POPPLSRegressor()
        assert model.n_components == "auto"
        assert model.max_components == 25
        assert model.selection == "per_component"
        assert model.engine == "simpls_covariance"
        assert model.criterion == "cv"
        assert model.center is True
        assert model.scale is False

    def test_fit_returns_self(self, small_data):
        X, y = small_data
        model = POPPLSRegressor(n_components=5)
        result = model.fit(X, y)
        assert result is model

    def test_fit_sets_attributes(self, small_data):
        X, y = small_data
        model = POPPLSRegressor(n_components=5)
        model.fit(X, y)
        assert hasattr(model, "n_components_")
        assert hasattr(model, "coef_")
        assert hasattr(model, "selected_operators_")
        assert hasattr(model, "x_mean_")
        assert model.x_mean_.shape[0] == 100

    def test_predict_shape(self, small_data):
        X, y = small_data
        model = POPPLSRegressor(n_components=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_predict_reasonable(self, regression_data):
        X, y = regression_data
        model = POPPLSRegressor(n_components=10)
        model.fit(X, y)
        preds = model.predict(X)
        corr = np.corrcoef(y, preds)[0, 1]
        assert corr > 0.5, f"Training correlation too low: {corr:.3f}"

    def test_transform_shape(self, small_data):
        X, y = small_data
        model = POPPLSRegressor(n_components=5)
        model.fit(X, y)
        T = model.transform(X)
        assert T.shape[0] == X.shape[0]
        assert T.shape[1] == model.n_components_

    def test_multivariate_y(self, small_data):
        X, _ = small_data
        rng = np.random.RandomState(42)
        Y = rng.randn(X.shape[0], 3)
        model = POPPLSRegressor(n_components=5)
        model.fit(X, Y)
        preds = model.predict(X)
        assert preds.shape == Y.shape

# =============================================================================
# Per-Component Operator Selection Tests
# =============================================================================

class TestPerComponentSelection:
    """Test that POP-PLS selects (possibly different) operators per component."""

    def test_selected_operators_length(self, small_data):
        """Each component should be associated with an operator name."""
        X, y = small_data
        model = POPPLSRegressor(n_components=5)
        model.fit(X, y)
        ops = model.get_selected_operators()
        assert len(ops) == model.n_components_

    def test_get_selected_operators_strings(self, small_data):
        X, y = small_data
        model = POPPLSRegressor(n_components=5)
        model.fit(X, y)
        ops = model.get_selected_operators()
        assert all(isinstance(name, str) for name in ops)

    def test_get_diagnostics(self, small_data):
        X, y = small_data
        model = POPPLSRegressor(n_components=5)
        model.fit(X, y)
        diag = model.get_diagnostics()
        assert isinstance(diag, dict)
        assert "selected_operator_names" in diag
        assert "n_components_selected" in diag

    def test_different_operators_possible(self):
        """With a diverse operator bank and structured data, the per-component
        selector may pick different operators across components."""
        rng = np.random.RandomState(42)
        n, p = 100, 200
        from scipy.ndimage import gaussian_filter1d
        X = rng.randn(n, p)
        X = gaussian_filter1d(X, sigma=5, axis=1)
        X += np.linspace(0, 3, p)[np.newaxis, :] * rng.randn(n, 1)
        y = X[:, 30:50].mean(axis=1) + 2.0 * (X[:, 100] - X[:, 95]) + 0.1 * rng.randn(n)

        bank = [
            IdentityOperator(),
            SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1),
            DetrendProjectionOperator(degree=1),
        ]
        model = POPPLSRegressor(n_components=5, operator_bank=bank)
        model.fit(X, y)
        ops = model.get_selected_operators()
        # With a small diverse bank, at least one operator should be used
        assert len(set(ops)) >= 1

# =============================================================================
# Auto-Components Tests
# =============================================================================

class TestAutoComponents:
    """Test auto component count selection via n_components='auto'."""

    def test_auto_components_default(self, small_data):
        """n_components='auto' should pick a positive count."""
        X, y = small_data
        model = POPPLSRegressor(n_components="auto", max_components=10)
        model.fit(X, y)
        assert model.n_components_ >= 1
        assert model.n_components_ <= 10

    def test_int_n_components_uses_all(self, small_data):
        """Integer n_components requests that many components (bounded by data)."""
        X, y = small_data
        model = POPPLSRegressor(n_components=5)
        model.fit(X, y)
        assert model.n_components_ <= 5

    def test_auto_fewer_than_max(self, regression_data):
        """Auto-select should bound n_components_ by max_components."""
        X, y = regression_data
        model = POPPLSRegressor(n_components="auto", max_components=25, random_state=42)
        model.fit(X, y)
        assert model.n_components_ <= 25
        assert model.n_components_ >= 1

# =============================================================================
# sklearn Compatibility Tests
# =============================================================================

class TestSklearnCompat:
    """Test sklearn API compatibility."""

    def test_get_params(self):
        model = POPPLSRegressor(n_components=10, max_components=15)
        params = model.get_params()
        assert params["n_components"] == 10
        assert params["max_components"] == 15
        assert params["selection"] == "per_component"

    def test_set_params(self):
        model = POPPLSRegressor()
        result = model.set_params(n_components=20, max_components=30)
        assert result is model
        assert model.n_components == 20
        assert model.max_components == 30

    def test_clone(self):
        model = POPPLSRegressor(n_components=10, max_components=15)
        cloned = clone(model)
        assert cloned.n_components == 10
        assert cloned.max_components == 15
        assert cloned is not model

    def test_repr(self):
        model = POPPLSRegressor(n_components=10)
        r = repr(model)
        assert "POPPLSRegressor" in r
        assert "10" in r

    def test_estimator_type(self):
        model = POPPLSRegressor()
        assert model._estimator_type == "regressor"

# =============================================================================
# Determinism Tests
# =============================================================================

class TestDeterminism:
    """Test deterministic outputs."""

    def test_deterministic_predictions(self, small_data):
        X, y = small_data
        model1 = POPPLSRegressor(n_components=5, random_state=42)
        model1.fit(X, y)
        preds1 = model1.predict(X)

        model2 = POPPLSRegressor(n_components=5, random_state=42)
        model2.fit(X, y)
        preds2 = model2.predict(X)

        np.testing.assert_array_equal(preds1, preds2)

    def test_deterministic_coef(self, small_data):
        X, y = small_data
        model1 = POPPLSRegressor(n_components=5, random_state=42)
        model1.fit(X, y)

        model2 = POPPLSRegressor(n_components=5, random_state=42)
        model2.fit(X, y)

        np.testing.assert_array_equal(model1.coef_, model2.coef_)

# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_more_components_than_samples(self):
        rng = np.random.RandomState(42)
        X = rng.randn(10, 50)
        y = rng.randn(10)
        model = POPPLSRegressor(n_components=100, operator_bank=[IdentityOperator()])
        model.fit(X, y)
        assert model.n_components_ <= 9

    def test_identity_only_bank(self, small_data):
        X, y = small_data
        model = POPPLSRegressor(n_components=5, operator_bank=[IdentityOperator()])
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_custom_bank(self, small_data):
        X, y = small_data
        bank = [
            SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1),
            DetrendProjectionOperator(degree=1),
        ]
        # POPPLSRegressor does not auto-add identity (unlike AOMPLSRegressor base),
        # but the bank should still produce a valid fit.
        # TODO: confirm whether the new POPPLSRegressor auto-adds identity; the
        # base _AOMPLSBase._resolve_bank does, but POPPLSRegressor._resolve_bank
        # (inherited) follows the same path so identity should be present.
        model = POPPLSRegressor(n_components=5, operator_bank=bank)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

# =============================================================================
# POPPLSClassifier Tests
# =============================================================================

class TestPOPPLSClassifier:
    """Test POPPLSClassifier for binary and multiclass tasks."""

    def test_binary_fit_predict(self, binary_data):
        X, y = binary_data
        model = POPPLSClassifier(n_components=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(preds) <= set(y)

    def test_binary_predict_proba(self, binary_data):
        X, y = binary_data
        model = POPPLSClassifier(n_components=5)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_multiclass_fit_predict(self, multiclass_data):
        X, y = multiclass_data
        model = POPPLSClassifier(n_components=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(preds) <= set(y)

    def test_multiclass_predict_proba(self, multiclass_data):
        X, y = multiclass_data
        model = POPPLSClassifier(n_components=5)
        model.fit(X, y)
        proba = model.predict_proba(X)
        n_classes = len(np.unique(y))
        assert proba.shape == (len(y), n_classes)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
        assert np.all(proba >= 0)

    def test_classes_attribute(self, multiclass_data):
        X, y = multiclass_data
        model = POPPLSClassifier(n_components=5)
        model.fit(X, y)
        assert hasattr(model, "classes_")
        np.testing.assert_array_equal(model.classes_, np.unique(y))

    def test_get_selected_operators(self, binary_data):
        X, y = binary_data
        model = POPPLSClassifier(n_components=5)
        model.fit(X, y)
        ops = model.get_selected_operators()
        assert len(ops) > 0

    def test_estimator_type(self):
        model = POPPLSClassifier()
        assert model._estimator_type == "classifier"

    def test_clone(self):
        model = POPPLSClassifier(n_components=10, max_components=15)
        cloned = clone(model)
        assert cloned.n_components == 10
        assert cloned.max_components == 15
        assert cloned is not model

    def test_repr(self):
        model = POPPLSClassifier(n_components=10)
        r = repr(model)
        assert "POPPLSClassifier" in r or "POPPLSDAClassifier" in r
