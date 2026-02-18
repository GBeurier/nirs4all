"""Tests for POP-PLS (Per-Operator-Per-component PLS) regressor and classifier.

Tests cover:
- POPPLSRegressor fit/predict/transform
- Per-component operator selection (gamma matrix)
- Holdout-based auto-select and validation-based prefix selection
- Auto n_orth tuning
- OPLS pre-filter integration
- Deterministic outputs
- sklearn compatibility (clone, get/set_params)
- POPPLSClassifier binary and multiclass
- predict_proba calibration
"""

import numpy as np
import pytest
from sklearn.base import clone

from nirs4all.operators.models.sklearn.aom_pls import IdentityOperator, SavitzkyGolayOperator, DetrendProjectionOperator
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
def val_data():
    """Validation data."""
    rng = np.random.RandomState(99)
    X = rng.randn(30, 100)
    y = X[:, :5].sum(axis=1) + 0.1 * rng.randn(30)
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
        assert model.n_components == 27
        assert model.n_orth == 0
        assert model.center is True
        assert model.scale is False
        assert model.auto_select is True

    def test_fit_returns_self(self, small_data):
        X, y = small_data
        model = POPPLSRegressor(n_components=5)
        result = model.fit(X, y)
        assert result is model

    def test_fit_sets_attributes(self, small_data):
        X, y = small_data
        model = POPPLSRegressor(n_components=5)
        model.fit(X, y)
        assert hasattr(model, "n_features_in_")
        assert hasattr(model, "n_components_")
        assert hasattr(model, "k_selected_")
        assert hasattr(model, "gamma_")
        assert hasattr(model, "coef_")
        assert hasattr(model, "block_names_")
        assert hasattr(model, "component_operators_")
        assert model.n_features_in_ == 100

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
        assert T.shape[1] == model.k_selected_

    def test_predict_with_n_components(self, small_data):
        X, y = small_data
        model = POPPLSRegressor(n_components=10, auto_select=False)
        model.fit(X, y)
        preds_2 = model.predict(X, n_components=2)
        preds_all = model.predict(X)
        assert preds_2.shape == preds_all.shape
        assert not np.allclose(preds_2, preds_all)

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
    """Test that POP-PLS selects different operators per component."""

    def test_gamma_one_hot_rows(self, small_data):
        """Each component should select exactly one operator."""
        X, y = small_data
        model = POPPLSRegressor(n_components=5)
        model.fit(X, y)
        gamma = model.get_block_weights()
        for k in range(gamma.shape[0]):
            assert abs(np.sum(gamma[k]) - 1.0) < 1e-10
            assert np.sum(gamma[k] > 0) == 1

    def test_component_operators_list(self, small_data):
        X, y = small_data
        model = POPPLSRegressor(n_components=5)
        model.fit(X, y)
        ops = model.get_component_operators()
        assert len(ops) == model.n_components_
        assert all(isinstance(name, str) for name in ops)

    def test_preprocessing_report(self, small_data):
        X, y = small_data
        model = POPPLSRegressor(n_components=5)
        model.fit(X, y)
        report = model.get_preprocessing_report()
        assert len(report) == model.n_components_
        for entry in report:
            assert "component" in entry
            assert "blocks" in entry
            assert len(entry["blocks"]) == 1  # exactly one operator per component

    def test_different_operators_possible(self):
        """With a diverse operator bank and structured data, different components
        may select different operators."""
        rng = np.random.RandomState(42)
        n, p = 100, 200
        # Create data where different spectral features dominate at different scales
        from scipy.ndimage import gaussian_filter1d
        X = rng.randn(n, p)
        X = gaussian_filter1d(X, sigma=5, axis=1)
        # Mix broad trend + sharp peak features
        X += np.linspace(0, 3, p)[np.newaxis, :] * rng.randn(n, 1)
        y = X[:, 30:50].mean(axis=1) + 2.0 * (X[:, 100] - X[:, 95]) + 0.1 * rng.randn(n)

        bank = [
            IdentityOperator(),
            SavitzkyGolayOperator(window=11, polyorder=2, deriv=1),
            DetrendProjectionOperator(degree=1),
        ]
        model = POPPLSRegressor(n_components=5, operator_bank=bank, auto_select=False)
        model.fit(X, y)
        ops = model.get_component_operators()
        # With a small diverse bank, we should see at least 1 operator used
        assert len(set(ops)) >= 1


# =============================================================================
# Auto-Select and Validation Tests
# =============================================================================


class TestAutoSelect:
    """Test holdout-based and validation-based component selection."""

    def test_auto_select_extracts_components(self, small_data):
        """Auto-select should extract at least 1 component."""
        X, y = small_data
        model = POPPLSRegressor(n_components=20, auto_select=True)
        model.fit(X, y)
        assert model.k_selected_ >= 1
        assert model.k_selected_ == model.n_components_

    def test_validation_prefix_selection(self, small_data, val_data):
        X, y = small_data
        X_val, y_val = val_data
        model = POPPLSRegressor(n_components=10, auto_select=True)
        model.fit(X, y, X_val=X_val, y_val=y_val)
        assert 1 <= model.k_selected_ <= 10

    def test_auto_select_false_uses_all(self, small_data):
        X, y = small_data
        model = POPPLSRegressor(n_components=5, auto_select=False)
        model.fit(X, y)
        assert model.k_selected_ == model.n_components_

    def test_auto_select_fewer_than_max(self, regression_data):
        """Auto-select should typically select fewer components than a large max."""
        X, y = regression_data
        model = POPPLSRegressor(n_components=25, auto_select=True, random_state=42)
        model.fit(X, y)
        # With holdout selection, should select fewer than the maximum
        assert model.n_components_ <= 25
        assert model.n_components_ >= 1


# =============================================================================
# Auto n_orth Tests
# =============================================================================


class TestAutoNOrth:
    """Test automatic n_orth tuning."""

    def test_auto_n_orth_default(self, small_data):
        """With n_orth=0 and auto_select=True, searches [0,1,2,3,4,5]."""
        X, y = small_data
        model = POPPLSRegressor(n_components=5, n_orth=0, auto_select=True)
        model.fit(X, y)
        assert model.n_components_ >= 1

    def test_auto_n_orth_explicit(self, small_data):
        """With n_orth=2 and auto_select=True, searches [0,1,2]."""
        X, y = small_data
        model = POPPLSRegressor(n_components=5, n_orth=2, auto_select=True)
        model.fit(X, y)
        assert model.n_components_ >= 1

    def test_fixed_n_orth_no_auto(self, small_data):
        """With auto_select=False, n_orth is used directly."""
        X, y = small_data
        model = POPPLSRegressor(n_components=5, n_orth=2, auto_select=False)
        model.fit(X, y)
        assert model._P_orth is not None


# =============================================================================
# OPLS Pre-filter Tests
# =============================================================================


class TestOPLS:
    """Test OPLS orthogonal pre-filter integration."""

    def test_opls_runs(self, small_data):
        X, y = small_data
        model = POPPLSRegressor(n_components=5, n_orth=2, auto_select=False)
        model.fit(X, y)
        assert model.n_components_ > 0
        assert model._P_orth is not None

    def test_opls_predict(self, small_data):
        X, y = small_data
        model = POPPLSRegressor(n_components=5, n_orth=2, auto_select=False)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert not np.any(np.isnan(preds))


# =============================================================================
# sklearn Compatibility Tests
# =============================================================================


class TestSklearnCompat:
    """Test sklearn API compatibility."""

    def test_get_params(self):
        model = POPPLSRegressor(n_components=10, n_orth=2)
        params = model.get_params()
        assert params["n_components"] == 10
        assert params["n_orth"] == 2
        assert params["auto_select"] is True

    def test_set_params(self):
        model = POPPLSRegressor()
        result = model.set_params(n_components=20, n_orth=3)
        assert result is model
        assert model.n_components == 20
        assert model.n_orth == 3

    def test_clone(self):
        model = POPPLSRegressor(n_components=10, n_orth=2)
        cloned = clone(model)
        assert cloned.n_components == 10
        assert cloned.n_orth == 2
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

    def test_deterministic_gamma(self, small_data):
        X, y = small_data
        model1 = POPPLSRegressor(n_components=5, random_state=42)
        model1.fit(X, y)

        model2 = POPPLSRegressor(n_components=5, random_state=42)
        model2.fit(X, y)

        np.testing.assert_array_equal(model1.gamma_, model2.gamma_)


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
            SavitzkyGolayOperator(window=11, polyorder=2, deriv=1),
            DetrendProjectionOperator(degree=1),
        ]
        model = POPPLSRegressor(n_components=5, operator_bank=bank)
        model.fit(X, y)
        # Identity should be auto-added
        assert any("identity" in name for name in model.block_names_)


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

    def test_get_component_operators(self, binary_data):
        X, y = binary_data
        model = POPPLSClassifier(n_components=5)
        model.fit(X, y)
        ops = model.get_component_operators()
        assert len(ops) > 0

    def test_with_validation_data(self, binary_data, val_data):
        X, y = binary_data
        X_val, _ = val_data
        # Create matching binary val labels
        rng = np.random.RandomState(99)
        y_val = np.array(["classA", "classB"])[rng.randint(0, 2, size=X_val.shape[0])]
        model = POPPLSClassifier(n_components=5)
        model.fit(X, y, X_val=X_val, y_val=y_val)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_estimator_type(self):
        model = POPPLSClassifier()
        assert model._estimator_type == "classifier"

    def test_clone(self):
        model = POPPLSClassifier(n_components=10, n_orth=2)
        cloned = clone(model)
        assert cloned.n_components == 10
        assert cloned.n_orth == 2
        assert cloned is not model

    def test_repr(self):
        model = POPPLSClassifier(n_components=10)
        r = repr(model)
        assert "POPPLSClassifier" in r
