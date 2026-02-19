"""Tests for AOM-PLS (Adaptive Operator-Mixture PLS) regressor.

Test plan from the design document:

Unit tests:
- Operator adjoint identity: |<Ax, y> - <x, A^T y>| < tol
- Identity-only bank recovers NIPALS PLS behavior
- Sparsemax properties: simplex, sparsity
- Normalized scoring reduces scale bias under synthetic rescaling
- sklearn compatibility (clone, cross_val_score)

Regression tests:
- Deterministic output under fixed random_state
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_regression

from nirs4all.operators.models.sklearn.aom_pls import (
    AOMPLSRegressor,
    ComposedOperator,
    DetrendProjectionOperator,
    IdentityOperator,
    LinearOperator,
    SavitzkyGolayOperator,
    _sparsemax,
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

@pytest.fixture
def val_data():
    """Validation split for prefix selection tests."""
    rng = np.random.RandomState(99)
    X = rng.randn(30, 100)
    y = X[:, :5].sum(axis=1) + 0.1 * rng.randn(30)
    return X, y

# =============================================================================
# Operator Adjoint Tests
# =============================================================================

class TestOperatorAdjoint:
    """Test that <A x, y> == <x, A^T y> for all operators."""

    P = 200  # Signal length for tests
    TOL = 1e-8  # Numerical tolerance

    def _check_adjoint(self, op, p=None):
        """Verify adjoint identity for an operator."""
        if p is None:
            p = self.P
        op.initialize(p)
        rng = np.random.RandomState(42)

        for _ in range(5):
            x = rng.randn(1, p)
            y_vec = rng.randn(p)

            ax = op.apply(x).ravel()
            aty = op.apply_adjoint(y_vec)

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
        op = SavitzkyGolayOperator(window=11, polyorder=2, deriv=0)
        self._check_adjoint(op)

    def test_sg_first_deriv_adjoint(self):
        op = SavitzkyGolayOperator(window=11, polyorder=2, deriv=1)
        self._check_adjoint(op)

    def test_sg_second_deriv_adjoint(self):
        op = SavitzkyGolayOperator(window=11, polyorder=2, deriv=2)
        self._check_adjoint(op)

    def test_sg_large_window_adjoint(self):
        op = SavitzkyGolayOperator(window=21, polyorder=3, deriv=1)
        self._check_adjoint(op)

    def test_detrend_linear_adjoint(self):
        op = DetrendProjectionOperator(degree=1)
        self._check_adjoint(op)

    def test_detrend_quadratic_adjoint(self):
        op = DetrendProjectionOperator(degree=2)
        self._check_adjoint(op)

    def test_composed_detrend_sg_adjoint(self):
        op = ComposedOperator(
            DetrendProjectionOperator(degree=1),
            SavitzkyGolayOperator(window=11, polyorder=2, deriv=1),
        )
        self._check_adjoint(op)

    def test_adjoint_short_signal(self):
        """Adjoint should hold even for short signals where boundary effects dominate."""
        op = SavitzkyGolayOperator(window=11, polyorder=2, deriv=1)
        self._check_adjoint(op, p=30)

    def test_adjoint_all_default_bank(self):
        """Test adjoint for every operator in the default bank."""
        bank = default_operator_bank()
        for op in bank:
            self._check_adjoint(op)

# =============================================================================
# Operator Property Tests
# =============================================================================

class TestOperatorProperties:
    """Test operator-specific properties."""

    def test_identity_is_identity(self):
        op = IdentityOperator()
        op.initialize(100)
        x = np.random.randn(3, 100)
        np.testing.assert_array_equal(op.apply(x), x)

    def test_identity_frobenius_norm(self):
        op = IdentityOperator()
        op.initialize(100)
        assert op.frobenius_norm_sq() == 100.0

    def test_detrend_removes_linear_trend(self):
        op = DetrendProjectionOperator(degree=1)
        p = 200
        op.initialize(p)
        # Linear signal
        x = np.linspace(0, 10, p).reshape(1, -1)
        result = op.apply(x)
        # Should be near zero (linear trend removed)
        assert np.max(np.abs(result)) < 1e-10

    def test_detrend_preserves_residual(self):
        """Detrend is idempotent: applying twice gives same result."""
        op = DetrendProjectionOperator(degree=2)
        p = 200
        op.initialize(p)
        x = np.random.randn(5, p)
        once = op.apply(x)
        twice = op.apply(once)
        np.testing.assert_allclose(once, twice, atol=1e-10)

    def test_detrend_frobenius_norm(self):
        op = DetrendProjectionOperator(degree=1)
        op.initialize(200)
        # For degree 1: nu = p - 2 = 198
        assert op.frobenius_norm_sq() == 198.0

    def test_sg_smoothing_reduces_noise(self):
        op = SavitzkyGolayOperator(window=21, polyorder=2, deriv=0)
        op.initialize(200)
        rng = np.random.RandomState(42)
        signal = np.sin(np.linspace(0, 4 * np.pi, 200))
        noisy = signal + 0.3 * rng.randn(200)
        smoothed = op.apply(noisy.reshape(1, -1)).ravel()
        # Smoothed should be closer to true signal than noisy
        assert np.std(smoothed - signal) < np.std(noisy - signal)

    def test_sg_frobenius_norm_positive(self):
        op = SavitzkyGolayOperator(window=11, polyorder=2, deriv=1)
        op.initialize(200)
        assert op.frobenius_norm_sq() > 0

    def test_composed_operator_applies_both(self):
        detrend = DetrendProjectionOperator(degree=1)
        sg = SavitzkyGolayOperator(window=11, polyorder=2, deriv=1)
        composed = ComposedOperator(detrend, sg)
        composed.initialize(200)

        x = np.random.randn(3, 200)
        detrend.initialize(200)
        sg.initialize(200)

        expected = sg.apply(detrend.apply(x))
        result = composed.apply(x)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_default_bank_has_identity(self):
        bank = default_operator_bank()
        assert any(isinstance(op, IdentityOperator) for op in bank)

    def test_default_bank_reasonable_size(self):
        bank = default_operator_bank()
        assert 8 <= len(bank) <= 120

# =============================================================================
# Sparsemax Tests
# =============================================================================

class TestSparsemax:
    """Test sparsemax activation properties."""

    def test_simplex_sum(self):
        """Output sums to 1."""
        z = np.array([2.0, 1.0, 0.5, -1.0])
        p = _sparsemax(z)
        assert abs(np.sum(p) - 1.0) < 1e-10

    def test_non_negative(self):
        """All outputs are non-negative."""
        z = np.array([3.0, 1.0, -2.0, -5.0])
        p = _sparsemax(z)
        assert np.all(p >= 0)

    def test_sparsity(self):
        """Weak entries are exactly zero."""
        z = np.array([10.0, 1.0, -5.0, -10.0])
        p = _sparsemax(z)
        assert p[2] == 0.0
        assert p[3] == 0.0

    def test_uniform_input(self):
        """Equal inputs → uniform output."""
        z = np.array([1.0, 1.0, 1.0, 1.0])
        p = _sparsemax(z)
        np.testing.assert_allclose(p, 0.25 * np.ones(4), atol=1e-10)

    def test_single_dominant(self):
        """One very large input → all mass on that entry."""
        z = np.array([100.0, 0.0, 0.0])
        p = _sparsemax(z)
        assert p[0] > 0.99
        assert p[1] == 0.0
        assert p[2] == 0.0

    def test_two_close_values(self):
        """Two close values share the mass."""
        z = np.array([5.0, 4.9, 0.0])
        p = _sparsemax(z)
        assert p[0] > 0
        assert p[1] > 0
        assert p[2] == 0.0
        assert abs(np.sum(p) - 1.0) < 1e-10

# =============================================================================
# Normalized Block Scoring Tests
# =============================================================================

class TestNormalizedScoring:
    """Test that normalized scoring reduces scale bias."""

    def test_rescaled_operator_same_gating(self):
        """Scaling an operator should not change its gating weight significantly.

        If we scale an operator A → α*A, the raw score scales as α^2 but
        the normalized score (||g||^2 / ν) should be approximately invariant.
        """
        p = 200
        rng = np.random.RandomState(42)
        c = rng.randn(p)
        eps = 1e-12

        # Original SG operator
        op = SavitzkyGolayOperator(window=11, polyorder=2, deriv=1)
        op.initialize(p)

        g = op.apply_adjoint(c)
        raw_score = np.sum(g ** 2)
        nu = op.frobenius_norm_sq()
        normalized_score = raw_score / (nu + eps)

        # Simulate scaling by a factor of 10:
        # A scaled operator would have g_scaled = 10 * g, nu_scaled = 100 * nu
        g_scaled = 10.0 * g
        nu_scaled = 100.0 * nu
        raw_score_scaled = np.sum(g_scaled ** 2)
        normalized_score_scaled = raw_score_scaled / (nu_scaled + eps)

        # Normalized scores should be equal
        np.testing.assert_allclose(normalized_score, normalized_score_scaled, rtol=1e-10)

        # Raw scores differ by factor of 100
        assert abs(raw_score_scaled / raw_score - 100.0) < 1e-6

# =============================================================================
# AOMPLSRegressor Tests
# =============================================================================

class TestAOMPLSRegressor:
    """Test AOMPLSRegressor sklearn compatibility and behavior."""

    def test_init_default(self):
        model = AOMPLSRegressor()
        assert model.n_components == 15
        assert model.gate == "hard"
        assert model.tau == 0.5
        assert model.n_orth == 0
        assert model.center is True
        assert model.scale is False
        assert model.backend == "numpy"

    def test_init_custom(self):
        model = AOMPLSRegressor(n_components=10, tau=0.3, n_orth=2, backend="numpy")
        assert model.n_components == 10
        assert model.tau == 0.3
        assert model.n_orth == 2

    def test_fit_returns_self(self, small_data):
        X, y = small_data
        model = AOMPLSRegressor(n_components=5)
        result = model.fit(X, y)
        assert result is model

    def test_fit_sets_attributes(self, small_data):
        X, y = small_data
        model = AOMPLSRegressor(n_components=5)
        model.fit(X, y)

        assert hasattr(model, "n_features_in_")
        assert hasattr(model, "n_components_")
        assert hasattr(model, "k_selected_")
        assert hasattr(model, "gamma_")
        assert hasattr(model, "coef_")
        assert hasattr(model, "block_names_")
        assert model.n_features_in_ == 100

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
        assert T.shape[1] == model.k_selected_

    def test_predict_with_n_components(self, small_data):
        X, y = small_data
        # Use operator_index to skip internal holdout (which may limit prefix)
        model = AOMPLSRegressor(n_components=10, operator_index=0)
        model.fit(X, y)
        # Predict with fewer components
        preds_2 = model.predict(X, n_components=2)
        preds_all = model.predict(X)
        assert preds_2.shape == preds_all.shape
        # Different n_components should give different predictions
        assert not np.allclose(preds_2, preds_all)

    def test_get_block_weights(self, small_data):
        X, y = small_data
        model = AOMPLSRegressor(n_components=5)
        model.fit(X, y)
        gamma = model.get_block_weights()
        assert gamma.shape[0] == model.n_components_
        assert gamma.shape[1] == len(model.block_names_)
        # Each row should sum to ~1
        for k in range(gamma.shape[0]):
            assert abs(np.sum(gamma[k]) - 1.0) < 0.1, f"Row {k} sums to {np.sum(gamma[k])}"
        # All values non-negative
        assert np.all(gamma >= 0)

    def test_get_preprocessing_report(self, small_data):
        X, y = small_data
        model = AOMPLSRegressor(n_components=5)
        model.fit(X, y)
        report = model.get_preprocessing_report()
        assert len(report) == model.n_components_
        for entry in report:
            assert "component" in entry
            assert "blocks" in entry
            assert len(entry["blocks"]) > 0
            for block in entry["blocks"]:
                assert "name" in block
                assert "weight" in block
                assert block["weight"] > 0

    def test_multivariate_y(self, small_data):
        X, _ = small_data
        rng = np.random.RandomState(42)
        Y = rng.randn(X.shape[0], 3)
        model = AOMPLSRegressor(n_components=5)
        model.fit(X, Y)
        preds = model.predict(X)
        assert preds.shape == Y.shape

# =============================================================================
# Identity-Only Bank Recovery Test
# =============================================================================

class TestIdentityBankRecovery:
    """Test that identity-only bank recovers standard PLS predictions."""

    def test_identity_bank_matches_pls(self):
        """With only identity operator, AOM-PLS should match NIPALS PLS."""
        rng = np.random.RandomState(42)
        X = rng.randn(80, 50)
        y = X[:, :5].sum(axis=1) + 0.1 * rng.randn(80)

        # AOM-PLS with identity-only bank
        identity_bank = [IdentityOperator()]
        aom = AOMPLSRegressor(
            n_components=10,
            operator_bank=identity_bank,
            tau=1.0,
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
        model = AOMPLSRegressor(n_components=15, tau=0.3)
        params = model.get_params()
        assert params["n_components"] == 15
        assert params["tau"] == 0.3
        assert params["backend"] == "numpy"
        assert params["center"] is True
        assert params["scale"] is False

    def test_set_params(self):
        model = AOMPLSRegressor(n_components=10)
        result = model.set_params(n_components=20, tau=2.0)
        assert result is model
        assert model.n_components == 20
        assert model.tau == 2.0

    def test_clone(self):
        model = AOMPLSRegressor(n_components=15, tau=0.3, n_orth=2)
        cloned = clone(model)
        assert cloned.n_components == 15
        assert cloned.tau == 0.3
        assert cloned.n_orth == 2
        assert cloned is not model

    def test_repr(self):
        model = AOMPLSRegressor(n_components=10, tau=0.5)
        r = repr(model)
        assert "AOMPLSRegressor" in r
        assert "10" in r

    def test_estimator_type(self):
        model = AOMPLSRegressor()
        assert model._estimator_type == "regressor"

# =============================================================================
# Validation Prefix Selection Tests
# =============================================================================

class TestPrefixSelection:
    """Test validation-based component count selection."""

    def test_with_validation_data(self, small_data, val_data):
        X, y = small_data
        X_val, y_val = val_data
        model = AOMPLSRegressor(n_components=10)
        model.fit(X, y, X_val=X_val, y_val=y_val)
        # k_selected_ should be between 1 and n_components_
        assert 1 <= model.k_selected_ <= model.n_components_

    def test_without_validation_uses_all(self, small_data):
        X, y = small_data
        model = AOMPLSRegressor(n_components=5)
        model.fit(X, y)
        assert model.k_selected_ == model.n_components_

# =============================================================================
# OPLS Pre-filter Tests
# =============================================================================

class TestOPLSPrefilter:
    """Test OPLS orthogonal pre-filter integration."""

    def test_opls_runs(self, small_data):
        X, y = small_data
        model = AOMPLSRegressor(n_components=5, n_orth=2)
        model.fit(X, y)
        assert model.n_components_ > 0
        assert model._P_orth is not None

    def test_opls_improves_or_maintains(self, regression_data):
        """OPLS pre-filter should not drastically worsen predictions."""
        X, y = regression_data
        model_no_opls = AOMPLSRegressor(n_components=10, n_orth=0)
        model_no_opls.fit(X, y)
        preds_no = model_no_opls.predict(X)
        rmse_no = np.sqrt(np.mean((y - preds_no) ** 2))

        model_opls = AOMPLSRegressor(n_components=10, n_orth=2)
        model_opls.fit(X, y)
        preds_opls = model_opls.predict(X)
        rmse_opls = np.sqrt(np.mean((y - preds_opls) ** 2))

        # OPLS version should not be dramatically worse
        assert rmse_opls < rmse_no * 3.0

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

    def test_deterministic_gating(self, small_data):
        X, y = small_data

        model1 = AOMPLSRegressor(n_components=5, random_state=42)
        model1.fit(X, y)

        model2 = AOMPLSRegressor(n_components=5, random_state=42)
        model2.fit(X, y)

        np.testing.assert_array_equal(model1.gamma_, model2.gamma_)

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
            SavitzkyGolayOperator(window=11, polyorder=2, deriv=1),
            SavitzkyGolayOperator(window=21, polyorder=2, deriv=2),
        ]
        # Identity will be auto-added
        model = AOMPLSRegressor(n_components=5, operator_bank=bank)
        model.fit(X, y)
        assert len(model.block_names_) == 3  # 2 SG + auto-added identity

    def test_identity_auto_added(self, small_data):
        """If no identity in bank, it should be auto-added."""
        X, y = small_data
        bank = [SavitzkyGolayOperator(window=11, polyorder=2, deriv=1)]
        model = AOMPLSRegressor(n_components=5, operator_bank=bank)
        model.fit(X, y)
        assert any("identity" in name for name in model.block_names_)

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
        """Constant features should not cause errors (handled by std=1 fallback)."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 20)
        X[:, 0] = 5.0  # Constant feature
        y = rng.randn(50)
        model = AOMPLSRegressor(n_components=3, operator_bank=[IdentityOperator()])
        model.fit(X, y)
        preds = model.predict(X)
        assert not np.any(np.isnan(preds))

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="backend must be"):
            model = AOMPLSRegressor(backend="invalid")
            model.fit(np.random.randn(10, 5), np.random.randn(10))

    def test_invalid_gate(self):
        with pytest.raises(ValueError, match="gate must be"):
            model = AOMPLSRegressor(gate="softmax")
            model.fit(np.random.randn(10, 5), np.random.randn(10))

    def test_tau_effect_on_sparsity(self, small_data):
        """Lower tau should produce sparser gating (sparsemax mode)."""
        X, y = small_data

        model_sparse = AOMPLSRegressor(n_components=5, gate="sparsemax", tau=0.01)
        model_sparse.fit(X, y)
        gamma_sparse = model_sparse.get_block_weights()

        model_dense = AOMPLSRegressor(n_components=5, gate="sparsemax", tau=10.0)
        model_dense.fit(X, y)
        gamma_dense = model_dense.get_block_weights()

        # Sparse model should have more zeros on average
        sparsity_sparse = np.mean(gamma_sparse == 0)
        sparsity_dense = np.mean(gamma_dense == 0)
        assert sparsity_sparse >= sparsity_dense

    def test_center_only_no_scale(self, small_data):
        """center=True, scale=False should center but not scale X per-column."""
        X, y = small_data
        model = AOMPLSRegressor(n_components=3, center=True, scale=False)
        model.fit(X, y)
        # x_std_ should be all ones (no per-column scaling)
        np.testing.assert_array_equal(model.x_std_, np.ones(X.shape[1]))
        # x_mean_ should be the column means
        np.testing.assert_allclose(model.x_mean_, X.mean(axis=0))

    def test_no_center_no_scale(self, small_data):
        """center=False, scale=False should do nothing to X."""
        X, y = small_data
        model = AOMPLSRegressor(n_components=3, center=False, scale=False)
        model.fit(X, y)
        np.testing.assert_array_equal(model.x_mean_, np.zeros(X.shape[1]))
        np.testing.assert_array_equal(model.x_std_, np.ones(X.shape[1]))

# =============================================================================
# Torch Backend Tests (only run if torch available)
# =============================================================================

def _torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False

@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
class TestTorchBackend:
    """Test Torch backend produces results consistent with NumPy backend."""

    def test_torch_fit_predict(self, small_data):
        X, y = small_data
        model = AOMPLSRegressor(n_components=5, backend="torch", operator_bank=[IdentityOperator()])
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert not np.any(np.isnan(preds))

    def test_torch_numpy_parity(self, small_data):
        """Torch and NumPy backends should produce similar predictions with identity bank."""
        X, y = small_data
        # Use identity-only bank to test NIPALS parity without operator selection differences
        bank = [IdentityOperator()]

        model_np = AOMPLSRegressor(n_components=5, operator_bank=bank, backend="numpy")
        model_np.fit(X, y)
        preds_np = model_np.predict(X)

        model_torch = AOMPLSRegressor(n_components=5, operator_bank=bank, backend="torch")
        model_torch.fit(X, y)
        preds_torch = model_torch.predict(X)

        # Float32 vs float64 tolerance
        np.testing.assert_allclose(preds_np, preds_torch, rtol=0.05, atol=0.5)

    def test_torch_gating_weights(self, small_data):
        X, y = small_data
        model = AOMPLSRegressor(n_components=5, backend="torch")
        model.fit(X, y)
        gamma = model.get_block_weights()
        # Check simplex constraints
        for k in range(gamma.shape[0]):
            assert np.all(gamma[k] >= 0)
            assert abs(np.sum(gamma[k]) - 1.0) < 0.1
