"""Tests for AOM-PLS Discriminant Analysis classifier.

Tests cover:
- Binary classification: fit, predict, predict_proba
- Multiclass classification: fit, predict, predict_proba
- Probability calibration (logistic, softmax fallback)
- sklearn compatibility (clone, get/set_params)
"""

import numpy as np
import pytest
from sklearn.base import clone

from nirs4all.operators.models.sklearn.aom_pls import IdentityOperator
from nirs4all.operators.models.sklearn.aom_pls_classifier import AOMPLSClassifier

# =============================================================================
# Fixtures
# =============================================================================

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
# Binary Classification Tests
# =============================================================================

class TestBinaryClassification:
    """Test binary classification."""

    def test_fit_predict(self, binary_data):
        X, y = binary_data
        model = AOMPLSClassifier(n_components=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(preds) <= set(y)

    def test_predict_proba_shape(self, binary_data):
        X, y = binary_data
        model = AOMPLSClassifier(n_components=5)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)

    def test_predict_proba_sums_to_one(self, binary_data):
        X, y = binary_data
        model = AOMPLSClassifier(n_components=5)
        model.fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_proba_bounded(self, binary_data):
        X, y = binary_data
        model = AOMPLSClassifier(n_components=5)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_classes_attribute(self, binary_data):
        X, y = binary_data
        model = AOMPLSClassifier(n_components=5)
        model.fit(X, y)
        assert hasattr(model, "classes_")
        np.testing.assert_array_equal(model.classes_, np.unique(y))

# =============================================================================
# Multiclass Classification Tests
# =============================================================================

class TestMulticlassClassification:
    """Test multiclass classification."""

    def test_fit_predict(self, multiclass_data):
        X, y = multiclass_data
        model = AOMPLSClassifier(n_components=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(preds) <= set(y)

    def test_predict_proba_shape(self, multiclass_data):
        X, y = multiclass_data
        model = AOMPLSClassifier(n_components=5)
        model.fit(X, y)
        proba = model.predict_proba(X)
        n_classes = len(np.unique(y))
        assert proba.shape == (len(y), n_classes)

    def test_predict_proba_sums_to_one(self, multiclass_data):
        X, y = multiclass_data
        model = AOMPLSClassifier(n_components=5)
        model.fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_proba_non_negative(self, multiclass_data):
        X, y = multiclass_data
        model = AOMPLSClassifier(n_components=5)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert np.all(proba >= 0)

# =============================================================================
# Delegated Method Tests
# =============================================================================

class TestDelegatedMethods:
    """Test diagnostic methods on the classifier."""

    def test_get_selected_operators(self, binary_data):
        X, y = binary_data
        model = AOMPLSClassifier(n_components=5)
        model.fit(X, y)
        ops = model.get_selected_operators()
        assert isinstance(ops, list)
        assert len(ops) > 0

    def test_get_diagnostics(self, binary_data):
        X, y = binary_data
        model = AOMPLSClassifier(n_components=5)
        model.fit(X, y)
        diag = model.get_diagnostics()
        assert isinstance(diag, dict)
        assert "selected_operator_names" in diag

# =============================================================================
# sklearn Compatibility Tests
# =============================================================================

class TestSklearnCompat:
    """Test sklearn API compatibility."""

    def test_get_params(self):
        model = AOMPLSClassifier(n_components=10, max_components=15)
        params = model.get_params()
        assert params["n_components"] == 10
        assert params["max_components"] == 15

    def test_set_params(self):
        model = AOMPLSClassifier()
        result = model.set_params(n_components=20)
        assert result is model
        assert model.n_components == 20

    def test_clone(self):
        model = AOMPLSClassifier(n_components=10, max_components=15)
        cloned = clone(model)
        assert cloned.n_components == 10
        assert cloned.max_components == 15
        assert cloned is not model

    def test_repr(self):
        model = AOMPLSClassifier(n_components=10)
        r = repr(model)
        assert "AOMPLSClassifier" in r or "AOMPLSDAClassifier" in r

    def test_estimator_type(self):
        model = AOMPLSClassifier()
        assert model._estimator_type == "classifier"
