"""Tests for AOM-PLS (Adaptive Operator-Mixture PLS) regressor.

Test plan:

Unit tests:
- Operator adjoint identity: |<Ax, y> - <x, A^T y>| < tol
- Identity-only bank recovers SIMPLS behavior
- sklearn compatibility (clone, get_params)

Regression tests:
- Deterministic output under fixed random_state
"""

import numpy as np
import pytest
from sklearn.base import clone, is_regressor

from nirs4all.operators.models.sklearn.aom_pls import (
    AOMPLSRegressor,
    ComposedOperator,
    DetrendProjectionOperator,
    IdentityOperator,
    LinearSpectralOperator,
    SavitzkyGolayOperator,
    default_operator_bank,
)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def regression_data():
    """Generate regression data with spectral-like structure."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 120, 200
    X = rng.randn(n_samples, n_features)
    # Smooth X to mimic spectral data
    from scipy.ndimage import gaussian_filter1d
    X = gaussian_filter1d(X, sigma=3, axis=1)
    # Target depends on a few wavelength regions
    y = X[:, 30:50].mean(axis=1) + 0.5 * X[:, 100:120].mean(axis=1) + 0.1 * rng.randn(n_samples)
    return X, y

@pytest.fixture
def small_data():
    """Small dataset for quick tests."""
    rng = np.random.RandomState(123)
    X = rng.randn(50, 100)
    y = X[:, :5].sum(axis=1) + 0.1 * rng.randn(50)
    return X, y

# =============================================================================
# Operator Adjoint Tests
# =============================================================================

class TestOperatorAdjoint:
    """Test that <A x, y> == <x, A^T y> for all operators."""

    P = 200  # Signal length for tests
    TOL = 1e-8  # Numerical tolerance

    def _check_adjoint(self, op: LinearSpectralOperator, p: int = None):
        """Verify adjoint identity for an operator."""
        if p is None:
            p = self.P
        op.fit(np.zeros((1, p)))
        rng = np.random.RandomState(42)

        for _ in range(5):
            x = rng.randn(1, p)
            y_vec = rng.randn(p)

            ax = op.transform(x).ravel()
            aty = op.adjoint_vec(y_vec)

            lhs = np.dot(ax, y_vec)
            rhs = np.dot(x.ravel(), aty)

            assert abs(lhs - rhs) < self.TOL, (
                f"Adjoint identity failed for {op.name}: "
                f"|<Ax,y> - <x,A^T y>| = {abs(lhs - rhs):.2e}"
            )

    def test_identity_adjoint(self):
        op = IdentityOperator()
        self._check_adjoint(op)

    def test_sg_smoothing_adjoint(self):
        op = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0)
        self._check_adjoint(op)

    def test_sg_first_deriv_adjoint(self):
        op = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1)
        self._check_adjoint(op)

    def test_sg_second_deriv_adjoint(self):
        op = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=2)
        self._check_adjoint(op)

    def test_sg_large_window_adjoint(self):
        op = SavitzkyGolayOperator(window_length=21, polyorder=3, deriv=1)
        self._check_adjoint(op)

    def test_detrend_linear_adjoint(self):
        op = DetrendProjectionOperator(degree=1)
        self._check_adjoint(op)

    def test_detrend_quadratic_adjoint(self):
        op = DetrendProjectionOperator(degree=2)
        self._check_adjoint(op)

    def test_composed_detrend_sg_adjoint(self):
        op = ComposedOperator([
            DetrendProjectionOperator(degree=1),
            SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1),
        ])
        self._check_adjoint(op)

    def test_adjoint_short_signal(self):
        """Adjoint should hold even for short signals where boundary effects dominate."""
        op = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1)
        self._check_adjoint(op, p=30)

    def test_adjoint_all_default_bank(self):
        """Test adjoint for every operator in the default bank."""
        bank = default_operator_bank(p=self.P)
        for op in bank:
            self._check_adjoint(op)

# =============================================================================
# Operator Property Tests
# =============================================================================

class TestOperatorProperties:
    """Test operator-specific properties."""

    def test_identity_is_identity(self):
        op = IdentityOperator()
        op.fit(np.zeros((1, 100)))
        x = np.random.randn(3, 100)
        np.testing.assert_array_equal(op.transform(x), x)

    def test_detrend_removes_linear_trend(self):
        op = DetrendProjectionOperator(degree=1)
        p = 200
        op.fit(np.zeros((1, p)))
        # Linear signal
        x = np.linspace(0, 10, p).reshape(1, -1)
        result = op.transform(x)
        # Should be near zero (linear trend removed)
        assert np.max(np.abs(result)) < 1e-10

    def test_detrend_preserves_residual(self):
        """Detrend is idempotent: applying twice gives same result."""
        op = DetrendProjectionOperator(degree=2)
        p = 200
        op.fit(np.zeros((1, p)))
        x = np.random.randn(5, p)
        once = op.transform(x)
        twice = op.transform(once)
        np.testing.assert_allclose(once, twice, atol=1e-10)

    def test_sg_smoothing_reduces_noise(self):
        op = SavitzkyGolayOperator(window_length=21, polyorder=2, deriv=0)
        op.fit(np.zeros((1, 200)))
        rng = np.random.RandomState(42)
        signal = np.sin(np.linspace(0, 4 * np.pi, 200))
        noisy = signal + 0.3 * rng.randn(200)
        smoothed = op.transform(noisy.reshape(1, -1)).ravel()
        # Smoothed should be closer to true signal than noisy
        assert np.std(smoothed - signal) < np.std(noisy - signal)

    def test_composed_operator_applies_both(self):
        detrend = DetrendProjectionOperator(degree=1)
        sg = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1)
        composed = ComposedOperator([detrend, sg])
        composed.fit(np.zeros((1, 200)))

        x = np.random.randn(3, 200)
        detrend.fit(np.zeros((1, 200)))
        sg.fit(np.zeros((1, 200)))

        expected = sg.transform(detrend.transform(x))
        result = composed.transform(x)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_default_bank_has_identity(self):
        bank = default_operator_bank(p=200)
        assert any(isinstance(op, IdentityOperator) for op in bank)

    def test_default_bank_reasonable_size(self):
        bank = default_operator_bank(p=200)
        assert 8 <= len(bank) <= 120

# =============================================================================
# AOMPLSRegressor Tests
# =============================================================================

class TestAOMPLSRegressor:
    """Test AOMPLSRegressor sklearn compatibility and behavior."""

    def test_init_default(self):
        model = AOMPLSRegressor()
        assert model.n_components == "auto"
        assert model.max_components == 25
        assert model.engine == "simpls_covariance"
        assert model.selection == "global"
        assert model.criterion == "cv"
        assert model.operator_bank == "default"
        assert model.center is True
        assert model.scale is False
        assert model.backend == "numpy"

    def test_init_custom(self):
        model = AOMPLSRegressor(n_components=10, max_components=15, backend="numpy")
        assert model.n_components == 10
        assert model.max_components == 15
        assert model.backend == "numpy"

    def test_fit_returns_self(self, small_data):
        X, y = small_data
        model = AOMPLSRegressor(n_components=5)
        result = model.fit(X, y)
        assert result is model

    def test_fit_sets_attributes(self, small_data):
        X, y = small_data
        model = AOMPLSRegressor(n_components=5)
        model.fit(X, y)

        assert hasattr(model, "n_features_in_") or hasattr(model, "x_mean_")
        assert hasattr(model, "n_components_")
        assert hasattr(model, "coef_")
        assert hasattr(model, "selected_operators_")
        assert model.x_mean_.shape[0] == 100

    def test_predict_shape(self, small_data):
        X, y = small_data
        model = AOMPLSRegressor(n_components=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_predict_reasonable_values(self, regression_data):
        X, y = regression_data
        model = AOMPLSRegressor(n_components=10)
        model.fit(X, y)
        preds = model.predict(X)
        # Training predictions should correlate with targets
        corr = np.corrcoef(y, preds)[0, 1]
        assert corr > 0.5, f"Training correlation too low: {corr:.3f}"

    def test_transform_shape(self, small_data):
        X, y = small_data
        model = AOMPLSRegressor(n_components=5)
        model.fit(X, y)
        T = model.transform(X)
        assert T.shape[0] == X.shape[0]
        assert T.shape[1] == model.n_components_

    def test_multivariate_y(self, small_data):
        X, _ = small_data
        rng = np.random.RandomState(42)
        Y = rng.randn(X.shape[0], 3)
        model = AOMPLSRegressor(n_components=5)
        model.fit(X, Y)
        preds = model.predict(X)
        assert preds.shape == Y.shape

    def test_get_selected_operators(self, small_data):
        X, y = small_data
        model = AOMPLSRegressor(n_components=5)
        model.fit(X, y)
        ops = model.get_selected_operators()
        assert isinstance(ops, list)
        assert len(ops) > 0
        assert all(isinstance(name, str) for name in ops)

    def test_get_diagnostics(self, small_data):
        X, y = small_data
        model = AOMPLSRegressor(n_components=5)
        model.fit(X, y)
        diag = model.get_diagnostics()
        assert isinstance(diag, dict)
        assert "selected_operator_names" in diag
        assert "n_components_selected" in diag

# =============================================================================
# Identity-Only Bank Recovery Test
# =============================================================================

class TestIdentityBankRecovery:
    """Test that identity-only bank recovers standard PLS predictions."""

    def test_identity_bank_matches_pls(self):
        """With only identity operator, AOM-PLS should match SIMPLS."""
        rng = np.random.RandomState(42)
        X = rng.randn(80, 50)
        y = X[:, :5].sum(axis=1) + 0.1 * rng.randn(80)

        # AOM-PLS with identity-only bank
        identity_bank = [IdentityOperator()]
        aom = AOMPLSRegressor(
            n_components=10,
            operator_bank=identity_bank,
            center=True,
            scale=True,
            backend="numpy",
        )
        aom.fit(X, y)
        preds_aom = aom.predict(X)

        # SIMPLS (same predictions as NIPALS for univariate y)
        from nirs4all.operators.models.sklearn.simpls import SIMPLS
        simpls = SIMPLS(n_components=10, scale=True, center=True, backend="numpy")
        simpls.fit(X, y)
        preds_simpls = simpls.predict(X)

        # Predictions should be very close
        np.testing.assert_allclose(preds_aom, preds_simpls, rtol=0.05, atol=0.1)

# =============================================================================
# sklearn Compatibility Tests
# =============================================================================

class TestSklearnCompat:
    """Test sklearn API compatibility."""

    def test_get_params(self):
        model = AOMPLSRegressor(n_components=15, max_components=20)
        params = model.get_params()
        assert params["n_components"] == 15
        assert params["max_components"] == 20
        assert params["backend"] == "numpy"
        assert params["center"] is True
        assert params["scale"] is False

    def test_set_params(self):
        model = AOMPLSRegressor(n_components=10)
        result = model.set_params(n_components=20, max_components=30)
        assert result is model
        assert model.n_components == 20
        assert model.max_components == 30

    def test_clone(self):
        model = AOMPLSRegressor(n_components=15, max_components=20)
        cloned = clone(model)
        assert cloned.n_components == 15
        assert cloned.max_components == 20
        assert cloned is not model

    def test_repr(self):
        model = AOMPLSRegressor(n_components=10)
        r = repr(model)
        assert "AOMPLSRegressor" in r
        assert "10" in r

    def test_estimator_type(self):
        model = AOMPLSRegressor()
        assert is_regressor(model)

# =============================================================================
# Deterministic Output Tests
# =============================================================================

class TestDeterminism:
    """Test that outputs are deterministic."""

    def test_deterministic_predictions(self, small_data):
        X, y = small_data

        model1 = AOMPLSRegressor(n_components=5, random_state=42)
        model1.fit(X, y)
        preds1 = model1.predict(X)

        model2 = AOMPLSRegressor(n_components=5, random_state=42)
        model2.fit(X, y)
        preds2 = model2.predict(X)

        np.testing.assert_array_equal(preds1, preds2)

    def test_deterministic_coef(self, small_data):
        X, y = small_data

        model1 = AOMPLSRegressor(n_components=5, random_state=42)
        model1.fit(X, y)

        model2 = AOMPLSRegressor(n_components=5, random_state=42)
        model2.fit(X, y)

        np.testing.assert_array_equal(model1.coef_, model2.coef_)

# =============================================================================
# Custom Operator Bank Tests
# =============================================================================

class TestCustomBank:
    """Test with custom operator banks."""

    def test_minimal_bank(self, small_data):
        """Single identity operator should work."""
        X, y = small_data
        model = AOMPLSRegressor(n_components=5, operator_bank=[IdentityOperator()])
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_sg_only_bank(self, small_data):
        X, y = small_data
        bank = [
            SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1),
            SavitzkyGolayOperator(window_length=21, polyorder=2, deriv=2),
        ]
        # Identity will be auto-added by AOMPLSRegressor._resolve_bank
        model = AOMPLSRegressor(n_components=5, operator_bank=bank)
        model.fit(X, y)
        # Bank should have identity prepended (2 SG + auto-added identity = 3)
        assert len(model._bank) == 3

    def test_identity_auto_added(self, small_data):
        """If no identity in bank, it should be auto-added."""
        X, y = small_data
        bank = [SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1)]
        model = AOMPLSRegressor(n_components=5, operator_bank=bank)
        model.fit(X, y)
        assert any(isinstance(op, IdentityOperator) for op in model._bank)

# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_more_components_than_samples(self):
        """n_components should be limited by data dimensions."""
        rng = np.random.RandomState(42)
        X = rng.randn(10, 50)
        y = rng.randn(10)
        model = AOMPLSRegressor(n_components=100, operator_bank=[IdentityOperator()])
        model.fit(X, y)
        assert model.n_components_ <= 9  # n_samples - 1

    def test_constant_feature(self):
        """Constant features should not cause errors."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 20)
        X[:, 0] = 5.0  # Constant feature
        y = rng.randn(50)
        model = AOMPLSRegressor(n_components=3, operator_bank=[IdentityOperator()])
        model.fit(X, y)
        preds = model.predict(X)
        assert not np.any(np.isnan(preds))

    def test_center_only_no_scale(self, small_data):
        """center=True, scale=False should center."""
        X, y = small_data
        model = AOMPLSRegressor(n_components=3, center=True, scale=False)
        model.fit(X, y)
        # x_mean_ should be the column means
        np.testing.assert_allclose(model.x_mean_, X.mean(axis=0))

    def test_no_center_no_scale(self, small_data):
        """center=False, scale=False should set x_mean_ to zeros."""
        X, y = small_data
        model = AOMPLSRegressor(n_components=3, center=False, scale=False)
        model.fit(X, y)
        np.testing.assert_array_equal(model.x_mean_, np.zeros(X.shape[1]))
