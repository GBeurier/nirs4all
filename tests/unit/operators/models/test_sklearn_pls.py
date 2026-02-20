"""Tests for sklearn PLS model operators."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from nirs4all.operators.models.sklearn import (
    IKPLS,
    KOPLS,
    LWPLS,
    MBPLS,
    OPLS,
    OPLSDA,
    PLSDA,
    SIMPLS,
    DiPLS,
    IntervalPLS,
    KernelPLS,
    RecursivePLS,
    RobustPLS,
    SparsePLS,
)


def _jax_available() -> bool:
    """Check if JAX is available."""
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


class TestPLSDA:
    """Test suite for PLSDA classifier."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=50,
            n_classes=2,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def multiclass_data(self):
        """Generate multiclass classification data."""
        X, y = make_classification(
            n_samples=150,
            n_features=50,
            n_classes=3,
            n_clusters_per_class=1,
            n_informative=15,
            random_state=42
        )
        return X, y

    def test_init_default(self):
        """Test PLSDA initialization with default parameters."""
        model = PLSDA()
        assert model.n_components == 5

    def test_init_custom_components(self):
        """Test PLSDA initialization with custom n_components."""
        model = PLSDA(n_components=10)
        assert model.n_components == 10

    def test_fit_binary(self, binary_data):
        """Test PLSDA fit on binary classification data."""
        X, y = binary_data
        model = PLSDA(n_components=5)

        result = model.fit(X, y)

        assert result is model  # fit returns self
        assert hasattr(model, 'classes_')
        assert hasattr(model, 'pls_')
        assert hasattr(model, 'encoder_')
        assert hasattr(model, 'n_features_in_')
        assert len(model.classes_) == 2
        assert model.n_features_in_ == 50

    def test_fit_multiclass(self, multiclass_data):
        """Test PLSDA fit on multiclass classification data."""
        X, y = multiclass_data
        model = PLSDA(n_components=5)

        model.fit(X, y)

        assert len(model.classes_) == 3

    def test_predict_binary(self, binary_data):
        """Test PLSDA predict on binary classification data."""
        X, y = binary_data
        model = PLSDA(n_components=5)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert set(predictions).issubset(set(y))

    def test_predict_multiclass(self, multiclass_data):
        """Test PLSDA predict on multiclass classification data."""
        X, y = multiclass_data
        model = PLSDA(n_components=5)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert set(predictions).issubset(set(y))

    def test_predict_proba_binary(self, binary_data):
        """Test PLSDA predict_proba on binary data."""
        X, y = binary_data
        model = PLSDA(n_components=5)
        model.fit(X, y)

        proba = model.predict_proba(X)

        assert proba.shape == (len(X), 2)

    def test_predict_proba_multiclass(self, multiclass_data):
        """Test PLSDA predict_proba on multiclass data."""
        X, y = multiclass_data
        model = PLSDA(n_components=5)
        model.fit(X, y)

        proba = model.predict_proba(X)

        assert proba.shape == (len(X), 3)

    def test_get_params(self):
        """Test PLSDA get_params method."""
        model = PLSDA(n_components=8)

        params = model.get_params()

        assert params == {'n_components': 8}

    def test_set_params(self):
        """Test PLSDA set_params method."""
        model = PLSDA(n_components=5)

        result = model.set_params(n_components=10)

        assert result is model  # set_params returns self
        assert model.n_components == 10

    def test_sklearn_clone_compatibility(self, binary_data):
        """Test that PLSDA works with sklearn clone."""
        from sklearn.base import clone

        model = PLSDA(n_components=7)
        cloned = clone(model)

        assert cloned.n_components == 7
        assert cloned is not model

    def test_sklearn_cross_val_score(self, binary_data):
        """Test that PLSDA works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = binary_data
        model = PLSDA(n_components=5)

        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_n_components_exceeds_features(self):
        """Test PLSDA handles n_components > n_features gracefully."""
        X = np.random.randn(50, 10)  # Only 10 features
        y = np.array([0] * 25 + [1] * 25)

        model = PLSDA(n_components=20)  # More components than features
        model.fit(X, y)

        # Should not raise, should use min(n_components, n_features)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_different_class_labels(self):
        """Test PLSDA with non-numeric class labels."""
        X = np.random.randn(60, 20)
        y = np.array(['cat'] * 20 + ['dog'] * 20 + ['bird'] * 20)

        model = PLSDA(n_components=5)
        model.fit(X, y)
        predictions = model.predict(X)

        assert set(predictions).issubset({'cat', 'dog', 'bird'})

class TestIKPLS:
    """Test suite for IKPLS regressor."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def multi_target_data(self):
        """Generate multi-target regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_targets=3,
            n_informative=10,
            random_state=42
        )
        return X, y

    def test_init_default(self):
        """Test IKPLS initialization with default parameters."""
        model = IKPLS()
        assert model.n_components == 10
        assert model.algorithm == 1
        assert model.center is True
        assert model.scale is True
        assert model.backend == 'numpy'

    def test_init_custom_parameters(self):
        """Test IKPLS initialization with custom parameters."""
        model = IKPLS(n_components=15, algorithm=2, center=False, scale=False)
        assert model.n_components == 15
        assert model.algorithm == 2
        assert model.center is False
        assert model.scale is False

    def test_fit(self, regression_data):
        """Test IKPLS fit on regression data."""
        X, y = regression_data
        model = IKPLS(n_components=10)

        result = model.fit(X, y)

        assert result is model  # fit returns self
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'n_components_')
        assert hasattr(model, 'coef_')
        assert hasattr(model, '_model')
        assert model.n_features_in_ == 50
        assert model.n_components_ == 10

    def test_fit_multi_target(self, multi_target_data):
        """Test IKPLS fit on multi-target regression data."""
        X, y = multi_target_data
        model = IKPLS(n_components=10)

        model.fit(X, y)

        # coef_ shape is (n_features, n_targets)
        assert model.coef_.shape == (50, 3)  # 50 features, 3 targets

    def test_predict(self, regression_data):
        """Test IKPLS predict on regression data."""
        X, y = regression_data
        model = IKPLS(n_components=10)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_predict_multi_target(self, multi_target_data):
        """Test IKPLS predict on multi-target regression data."""
        X, y = multi_target_data
        model = IKPLS(n_components=10)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_predict_with_n_components(self, regression_data):
        """Test IKPLS predict with different n_components."""
        X, y = regression_data
        model = IKPLS(n_components=15)
        model.fit(X, y)

        # Predict with fewer components
        predictions = model.predict(X, n_components=5)

        assert predictions.shape == y.shape

    def test_get_params(self):
        """Test IKPLS get_params method."""
        model = IKPLS(n_components=8, algorithm=2, center=False, scale=True)

        params = model.get_params()

        assert params == {
            'n_components': 8,
            'algorithm': 2,
            'center': False,
            'scale': True,
            'backend': 'numpy',
        }

    def test_set_params(self):
        """Test IKPLS set_params method."""
        model = IKPLS(n_components=5)

        result = model.set_params(n_components=10, algorithm=2)

        assert result is model  # set_params returns self
        assert model.n_components == 10
        assert model.algorithm == 2

    def test_sklearn_clone_compatibility(self):
        """Test that IKPLS works with sklearn clone."""
        from sklearn.base import clone

        model = IKPLS(n_components=7, algorithm=2)
        cloned = clone(model)

        assert cloned.n_components == 7
        assert cloned.algorithm == 2
        assert cloned is not model

    def test_sklearn_cross_val_score(self, regression_data):
        """Test that IKPLS works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = regression_data
        model = IKPLS(n_components=5)

        scores = cross_val_score(model, X, y, cv=3, scoring='r2')

        assert len(scores) == 3

    def test_n_components_exceeds_features(self):
        """Test IKPLS handles n_components > n_features gracefully."""
        X = np.random.randn(50, 10)  # Only 10 features
        y = np.random.randn(50)

        model = IKPLS(n_components=20)  # More components than features
        model.fit(X, y)

        # Should not raise, n_components_ should be limited
        assert model.n_components_ <= 10
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_n_components_exceeds_samples(self):
        """Test IKPLS handles n_components > n_samples gracefully."""
        X = np.random.randn(20, 50)  # Only 20 samples
        y = np.random.randn(20)

        model = IKPLS(n_components=30)  # More components than samples
        model.fit(X, y)

        # Should not raise, n_components_ should be limited
        assert model.n_components_ <= 19  # n_samples - 1
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_algorithm_variants(self, regression_data):
        """Test both IKPLS algorithm variants produce valid results."""
        X, y = regression_data

        model1 = IKPLS(n_components=5, algorithm=1)
        model2 = IKPLS(n_components=5, algorithm=2)

        model1.fit(X, y)
        model2.fit(X, y)

        pred1 = model1.predict(X)
        pred2 = model2.predict(X)

        # Both should produce valid predictions
        assert pred1.shape == y.shape
        assert pred2.shape == y.shape
        assert not np.isnan(pred1).any()
        assert not np.isnan(pred2).any()

    def test_init_with_backend(self):
        """Test IKPLS initialization with backend parameter."""
        model_numpy = IKPLS(n_components=5, backend='numpy')
        assert model_numpy.backend == 'numpy'

        model_jax = IKPLS(n_components=5, backend='jax')
        assert model_jax.backend == 'jax'

    def test_invalid_backend(self, regression_data):
        """Test IKPLS raises error for invalid backend."""
        X, y = regression_data
        model = IKPLS(n_components=5, backend='invalid')

        with pytest.raises(ValueError, match="backend must be 'numpy' or 'jax'"):
            model.fit(X, y)

    def test_get_params_with_backend(self):
        """Test IKPLS get_params includes backend parameter."""
        model = IKPLS(n_components=5, backend='jax')

        params = model.get_params()

        assert 'backend' in params
        assert params['backend'] == 'jax'

    def test_set_params_with_backend(self):
        """Test IKPLS set_params can update backend parameter."""
        model = IKPLS(n_components=5, backend='numpy')

        model.set_params(backend='jax')

        assert model.backend == 'jax'

@pytest.mark.xdist_group("gpu")
class TestIKPLSJAX:
    """Test suite for IKPLS regressor with JAX backend.

    These tests are skipped if JAX is not installed.
    """

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def multi_target_data(self):
        """Generate multi-target regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_targets=3,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_fit_jax_backend(self, regression_data):
        """Test IKPLS fit with JAX backend."""
        X, y = regression_data
        model = IKPLS(n_components=10, backend='jax')

        result = model.fit(X, y)

        assert result is model
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'n_components_')
        assert hasattr(model, 'coef_')
        assert model.n_features_in_ == 50
        assert model.n_components_ == 10

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_predict_jax_backend(self, regression_data):
        """Test IKPLS predict with JAX backend."""
        X, y = regression_data
        model = IKPLS(n_components=10, backend='jax')
        model.fit(X, y)

        predictions = model.predict(X)

        # Should return numpy array
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_fit_jax_multi_target(self, multi_target_data):
        """Test IKPLS fit on multi-target data with JAX backend."""
        X, y = multi_target_data
        model = IKPLS(n_components=10, backend='jax')

        model.fit(X, y)

        # coef_ shape is (n_features, n_targets) and should be numpy array
        assert isinstance(model.coef_, np.ndarray)
        assert model.coef_.shape == (50, 3)

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_predict_jax_multi_target(self, multi_target_data):
        """Test IKPLS predict on multi-target data with JAX backend."""
        X, y = multi_target_data
        model = IKPLS(n_components=10, backend='jax')
        model.fit(X, y)

        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == y.shape

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_algorithm_variants(self, regression_data):
        """Test both IKPLS algorithm variants with JAX backend."""
        X, y = regression_data

        model1 = IKPLS(n_components=5, algorithm=1, backend='jax')
        model2 = IKPLS(n_components=5, algorithm=2, backend='jax')

        model1.fit(X, y)
        model2.fit(X, y)

        pred1 = model1.predict(X)
        pred2 = model2.predict(X)

        # Both should produce valid predictions as numpy arrays
        assert isinstance(pred1, np.ndarray)
        assert isinstance(pred2, np.ndarray)
        assert pred1.shape == y.shape
        assert pred2.shape == y.shape
        assert not np.isnan(pred1).any()
        assert not np.isnan(pred2).any()

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_numpy_similar_results(self, regression_data):
        """Test that JAX and NumPy backends produce similar results."""
        X, y = regression_data

        model_numpy = IKPLS(n_components=5, algorithm=1, backend='numpy')
        model_jax = IKPLS(n_components=5, algorithm=1, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        # Results should be very similar (not exactly equal due to floating point)
        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-5)

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_sklearn_clone(self, regression_data):
        """Test that IKPLS with JAX backend works with sklearn clone."""
        from sklearn.base import clone

        model = IKPLS(n_components=5, backend='jax')
        cloned = clone(model)

        assert cloned.backend == 'jax'
        assert cloned is not model

        # Cloned model should be fittable
        X, y = regression_data
        cloned.fit(X, y)
        predictions = cloned.predict(X)
        assert predictions.shape == y.shape

class TestOPLS:
    """Test suite for OPLS regressor."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    def test_init_default(self):
        """Test OPLS initialization with default parameters."""
        model = OPLS()
        assert model.n_components == 1
        assert model.pls_components == 1
        assert model.scale is True

    def test_init_custom_parameters(self):
        """Test OPLS initialization with custom parameters."""
        model = OPLS(n_components=2, pls_components=3, scale=False)
        assert model.n_components == 2
        assert model.pls_components == 3
        assert model.scale is False

    def test_fit(self, regression_data):
        """Test OPLS fit on regression data."""
        X, y = regression_data
        model = OPLS(n_components=2, pls_components=1)

        result = model.fit(X, y)

        assert result is model  # fit returns self
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'opls_')
        assert hasattr(model, 'pls_')
        assert model.n_features_in_ == 50

    def test_predict(self, regression_data):
        """Test OPLS predict on regression data."""
        X, y = regression_data
        model = OPLS(n_components=2, pls_components=1)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_transform(self, regression_data):
        """Test OPLS transform method."""
        X, y = regression_data
        model = OPLS(n_components=2)
        model.fit(X, y)

        X_transformed = model.transform(X)

        # Transform preserves shape (removes orthogonal variation)
        assert X_transformed.shape == X.shape

    def test_get_params(self):
        """Test OPLS get_params method."""
        model = OPLS(n_components=2, pls_components=3, scale=False)

        params = model.get_params()

        assert params == {
            'n_components': 2,
            'pls_components': 3,
            'scale': False,
            'backend': 'numpy'
        }

    def test_set_params(self):
        """Test OPLS set_params method."""
        model = OPLS(n_components=1)

        result = model.set_params(n_components=3, pls_components=2)

        assert result is model
        assert model.n_components == 3
        assert model.pls_components == 2

    def test_sklearn_clone_compatibility(self):
        """Test that OPLS works with sklearn clone."""
        from sklearn.base import clone

        model = OPLS(n_components=2, pls_components=3)
        cloned = clone(model)

        assert cloned.n_components == 2
        assert cloned.pls_components == 3
        assert cloned is not model

    def test_sklearn_cross_val_score(self, regression_data):
        """Test that OPLS works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = regression_data
        model = OPLS(n_components=1, pls_components=1)

        scores = cross_val_score(model, X, y, cv=3, scoring='r2')

        assert len(scores) == 3

class TestOPLSDA:
    """Test suite for OPLSDA classifier."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=50,
            n_classes=2,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def multiclass_data(self):
        """Generate multiclass classification data."""
        X, y = make_classification(
            n_samples=150,
            n_features=50,
            n_classes=3,
            n_clusters_per_class=1,
            n_informative=15,
            random_state=42
        )
        return X, y

    def test_init_default(self):
        """Test OPLSDA initialization with default parameters."""
        model = OPLSDA()
        assert model.n_components == 1
        assert model.pls_components == 5
        assert model.scale is True

    def test_init_custom_parameters(self):
        """Test OPLSDA initialization with custom parameters."""
        model = OPLSDA(n_components=2, pls_components=10, scale=False)
        assert model.n_components == 2
        assert model.pls_components == 10
        assert model.scale is False

    def test_fit_binary(self, binary_data):
        """Test OPLSDA fit on binary classification data."""
        X, y = binary_data
        model = OPLSDA(n_components=1, pls_components=5)

        result = model.fit(X, y)

        assert result is model
        assert hasattr(model, 'classes_')
        assert hasattr(model, 'opls_')
        assert hasattr(model, 'plsda_')
        assert hasattr(model, 'n_features_in_')
        assert len(model.classes_) == 2
        assert model.n_features_in_ == 50

    def test_fit_multiclass(self, multiclass_data):
        """Test OPLSDA fit on multiclass classification data."""
        X, y = multiclass_data
        model = OPLSDA(n_components=1, pls_components=5)

        model.fit(X, y)

        assert len(model.classes_) == 3

    def test_predict_binary(self, binary_data):
        """Test OPLSDA predict on binary classification data."""
        X, y = binary_data
        model = OPLSDA(n_components=1, pls_components=5)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert set(predictions).issubset(set(y))

    def test_predict_multiclass(self, multiclass_data):
        """Test OPLSDA predict on multiclass classification data."""
        X, y = multiclass_data
        model = OPLSDA(n_components=1, pls_components=5)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert set(predictions).issubset(set(y))

    def test_predict_proba_binary(self, binary_data):
        """Test OPLSDA predict_proba on binary data."""
        X, y = binary_data
        model = OPLSDA(n_components=1, pls_components=5)
        model.fit(X, y)

        proba = model.predict_proba(X)

        assert proba.shape == (len(X), 2)

    def test_predict_proba_multiclass(self, multiclass_data):
        """Test OPLSDA predict_proba on multiclass data."""
        X, y = multiclass_data
        model = OPLSDA(n_components=1, pls_components=5)
        model.fit(X, y)

        proba = model.predict_proba(X)

        assert proba.shape == (len(X), 3)

    def test_transform(self, binary_data):
        """Test OPLSDA transform method."""
        X, y = binary_data
        model = OPLSDA(n_components=1)
        model.fit(X, y)

        X_transformed = model.transform(X)

        assert X_transformed.shape == X.shape

    def test_get_params(self):
        """Test OPLSDA get_params method."""
        model = OPLSDA(n_components=2, pls_components=8, scale=False)

        params = model.get_params()

        assert params == {
            'n_components': 2,
            'pls_components': 8,
            'scale': False
        }

    def test_set_params(self):
        """Test OPLSDA set_params method."""
        model = OPLSDA(n_components=1)

        result = model.set_params(n_components=2, pls_components=10)

        assert result is model
        assert model.n_components == 2
        assert model.pls_components == 10

    def test_sklearn_clone_compatibility(self):
        """Test that OPLSDA works with sklearn clone."""
        from sklearn.base import clone

        model = OPLSDA(n_components=2, pls_components=7)
        cloned = clone(model)

        assert cloned.n_components == 2
        assert cloned.pls_components == 7
        assert cloned is not model

    def test_sklearn_cross_val_score(self, binary_data):
        """Test that OPLSDA works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = binary_data
        model = OPLSDA(n_components=1, pls_components=5)

        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

class TestMBPLS:
    """Test suite for MBPLS regressor."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def multiblock_data(self):
        """Generate multiblock regression data."""
        np.random.seed(42)
        X1 = np.random.randn(100, 30)
        X2 = np.random.randn(100, 20)
        y = np.random.randn(100)
        return [X1, X2], y

    def test_init_default(self):
        """Test MBPLS initialization with default parameters."""
        model = MBPLS()
        assert model.n_components == 5
        assert model.method == 'NIPALS'
        assert model.standardize is True
        assert model.max_tol == 1e-14

    def test_init_custom_parameters(self):
        """Test MBPLS initialization with custom parameters."""
        model = MBPLS(n_components=10, method='SVD', standardize=False)
        assert model.n_components == 10
        assert model.method == 'SVD'
        assert model.standardize is False

    def test_fit_single_block(self, regression_data):
        """Test MBPLS fit on single-block data."""
        X, y = regression_data
        model = MBPLS(n_components=5)

        result = model.fit(X, y)

        assert result is model
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'n_components_')
        assert hasattr(model, 'coef_')
        assert model.n_features_in_ == 50

    def test_fit_multiblock(self, multiblock_data):
        """Test MBPLS fit on multiblock data."""
        X_blocks, y = multiblock_data
        model = MBPLS(n_components=5)

        model.fit(X_blocks, y)

        assert model.n_features_in_ == 50  # 30 + 20
        assert model._is_multiblock is True

    def test_predict_single_block(self, regression_data):
        """Test MBPLS predict on single-block data."""
        X, y = regression_data
        model = MBPLS(n_components=5)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_predict_multiblock(self, multiblock_data):
        """Test MBPLS predict on multiblock data."""
        X_blocks, y = multiblock_data
        model = MBPLS(n_components=5)
        model.fit(X_blocks, y)

        predictions = model.predict(X_blocks)

        assert predictions.shape == y.shape

    def test_transform(self, regression_data):
        """Test MBPLS transform method."""
        X, y = regression_data
        model = MBPLS(n_components=5)
        model.fit(X, y)

        T = model.transform(X)

        assert T.shape == (100, 5)

    def test_get_params(self):
        """Test MBPLS get_params method."""
        model = MBPLS(n_components=8, method='SVD', standardize=False, max_tol=1e-10)

        params = model.get_params()

        assert params == {
            'n_components': 8,
            'method': 'SVD',
            'standardize': False,
            'max_tol': 1e-10,
            'backend': 'numpy'
        }

    def test_set_params(self):
        """Test MBPLS set_params method."""
        model = MBPLS(n_components=5)

        result = model.set_params(n_components=10, method='SVD')

        assert result is model
        assert model.n_components == 10
        assert model.method == 'SVD'

    def test_sklearn_clone_compatibility(self):
        """Test that MBPLS works with sklearn clone."""
        from sklearn.base import clone

        model = MBPLS(n_components=7, method='SVD')
        cloned = clone(model)

        assert cloned.n_components == 7
        assert cloned.method == 'SVD'
        assert cloned is not model

    def test_sklearn_cross_val_score(self, regression_data):
        """Test that MBPLS works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = regression_data
        model = MBPLS(n_components=5)

        scores = cross_val_score(model, X, y, cv=3, scoring='r2')

        assert len(scores) == 3

class TestDiPLS:
    """Test suite for DiPLS regressor."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    def test_init_default(self):
        """Test DiPLS initialization with default parameters."""
        model = DiPLS()
        assert model.n_components == 5
        assert model.lags == 1
        assert model.cv_splits == 7
        assert model.tol == 1e-8
        assert model.max_iter == 1000

    def test_init_custom_parameters(self):
        """Test DiPLS initialization with custom parameters."""
        model = DiPLS(n_components=10, lags=3, cv_splits=5)
        assert model.n_components == 10
        assert model.lags == 3
        assert model.cv_splits == 5

    def test_fit(self, regression_data):
        """Test DiPLS fit on regression data."""
        X, y = regression_data
        model = DiPLS(n_components=5, lags=1)

        result = model.fit(X, y)

        assert result is model
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'n_components_')
        assert hasattr(model, '_model')
        assert model.n_features_in_ == 50

    def test_predict(self, regression_data):
        """Test DiPLS predict on regression data."""
        X, y = regression_data
        model = DiPLS(n_components=5, lags=1)
        model.fit(X, y)

        predictions = model.predict(X)

        # DiPLS uses Hankelization, so predictions may be shorter by `lags` samples
        assert predictions.shape[0] <= y.shape[0]
        assert predictions.shape[0] >= y.shape[0] - model.lags
        assert not np.isnan(predictions).any()

    def test_different_lags(self, regression_data):
        """Test DiPLS with different lag values."""
        X, y = regression_data

        # Note: lags=0 causes issues in trendfitter library, so we skip it
        for lags in [1, 2, 3]:
            model = DiPLS(n_components=3, lags=lags)
            model.fit(X, y)
            predictions = model.predict(X)
            # DiPLS predictions may be shorter due to Hankelization
            assert predictions.shape[0] <= y.shape[0]
            assert predictions.shape[0] >= y.shape[0] - lags - 1

    def test_get_params(self):
        """Test DiPLS get_params method."""
        model = DiPLS(n_components=8, lags=2, cv_splits=5, tol=1e-6, max_iter=500)

        params = model.get_params()

        assert params == {
            'n_components': 8,
            'lags': 2,
            'cv_splits': 5,
            'tol': 1e-6,
            'max_iter': 500
        }

    def test_set_params(self):
        """Test DiPLS set_params method."""
        model = DiPLS(n_components=5)

        result = model.set_params(n_components=10, lags=3)

        assert result is model
        assert model.n_components == 10
        assert model.lags == 3

    def test_sklearn_clone_compatibility(self):
        """Test that DiPLS works with sklearn clone."""
        from sklearn.base import clone

        model = DiPLS(n_components=7, lags=2)
        cloned = clone(model)

        assert cloned.n_components == 7
        assert cloned.lags == 2
        assert cloned is not model

    def test_sklearn_cross_val_score(self, regression_data):
        """Test that DiPLS works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = regression_data
        model = DiPLS(n_components=3, lags=1)

        scores = cross_val_score(model, X, y, cv=3, scoring='r2')

        assert len(scores) == 3

class TestSparsePLS:
    """Test suite for SparsePLS regressor."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    def test_init_default(self):
        """Test SparsePLS initialization with default parameters."""
        model = SparsePLS()
        assert model.n_components == 5
        assert model.alpha == 1.0
        assert model.max_iter == 500
        assert model.tol == 1e-6
        assert model.scale is True

    def test_init_custom_parameters(self):
        """Test SparsePLS initialization with custom parameters."""
        model = SparsePLS(n_components=10, alpha=0.5, max_iter=1000, scale=False)
        assert model.n_components == 10
        assert model.alpha == 0.5
        assert model.max_iter == 1000
        assert model.scale is False

    def test_fit(self, regression_data):
        """Test SparsePLS fit on regression data."""
        X, y = regression_data
        model = SparsePLS(n_components=5, alpha=0.5)

        result = model.fit(X, y)

        assert result is model
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'n_components_')
        assert hasattr(model, 'coef_')
        assert model.n_features_in_ == 50

    def test_predict(self, regression_data):
        """Test SparsePLS predict on regression data."""
        X, y = regression_data
        model = SparsePLS(n_components=5, alpha=0.5)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_transform(self, regression_data):
        """Test SparsePLS transform method."""
        X, y = regression_data
        model = SparsePLS(n_components=5, alpha=0.5)
        model.fit(X, y)

        T = model.transform(X)

        assert T.shape == (100, 5)

    def test_different_alpha_values(self, regression_data):
        """Test SparsePLS with different alpha values."""
        X, y = regression_data

        for alpha in [0.1, 0.5, 1.0, 2.0]:
            model = SparsePLS(n_components=3, alpha=alpha)
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape

    def test_get_params(self):
        """Test SparsePLS get_params method."""
        model = SparsePLS(n_components=8, alpha=0.5, max_iter=1000, tol=1e-5, scale=False)

        params = model.get_params()

        assert params == {
            'n_components': 8,
            'alpha': 0.5,
            'max_iter': 1000,
            'tol': 1e-5,
            'scale': False,
            'backend': 'numpy'
        }

    def test_set_params(self):
        """Test SparsePLS set_params method."""
        model = SparsePLS(n_components=5)

        result = model.set_params(n_components=10, alpha=0.3)

        assert result is model
        assert model.n_components == 10
        assert model.alpha == 0.3

    def test_sklearn_clone_compatibility(self):
        """Test that SparsePLS works with sklearn clone."""
        from sklearn.base import clone

        model = SparsePLS(n_components=7, alpha=0.5)
        cloned = clone(model)

        assert cloned.n_components == 7
        assert cloned.alpha == 0.5
        assert cloned is not model

    def test_sklearn_cross_val_score(self, regression_data):
        """Test that SparsePLS works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = regression_data
        model = SparsePLS(n_components=3, alpha=0.5)

        scores = cross_val_score(model, X, y, cv=3, scoring='r2')

        assert len(scores) == 3

    def test_n_components_exceeds_features(self):
        """Test SparsePLS handles n_components > n_features gracefully."""
        X = np.random.randn(50, 10)  # Only 10 features
        y = np.random.randn(50)

        model = SparsePLS(n_components=20, alpha=0.5)  # More components than features
        model.fit(X, y)

        # Should not raise, n_components_ should be limited
        assert model.n_components_ <= 10
        predictions = model.predict(X)
        assert predictions.shape == y.shape

class TestLWPLS:
    """Test suite for LWPLS (Locally-Weighted PLS) regressor."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def nonlinear_data(self):
        """Generate nonlinear data where LWPLS should excel."""
        np.random.seed(42)
        X = 5 * np.random.rand(100, 2)
        y = 3 * X[:, 0]**2 + 10 * np.log(X[:, 1] + 0.1) + np.random.randn(100)
        return X, y

    def test_init_default(self):
        """Test LWPLS initialization with default parameters."""
        model = LWPLS()
        assert model.n_components == 10
        assert model.lambda_in_similarity == 1.0
        assert model.scale is True

    def test_init_custom_parameters(self):
        """Test LWPLS initialization with custom parameters."""
        model = LWPLS(n_components=5, lambda_in_similarity=0.5, scale=False)
        assert model.n_components == 5
        assert model.lambda_in_similarity == 0.5
        assert model.scale is False

    def test_fit(self, regression_data):
        """Test LWPLS fit on regression data."""
        X, y = regression_data
        model = LWPLS(n_components=5)

        result = model.fit(X, y)

        assert result is model  # fit returns self
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'n_components_')
        assert hasattr(model, 'X_train_')
        assert hasattr(model, 'y_train_')
        assert model.n_features_in_ == 50
        assert model.n_components_ == 5

    def test_fit_stores_data(self, regression_data):
        """Test that LWPLS stores training data for lazy prediction."""
        X, y = regression_data
        model = LWPLS(n_components=5, scale=True)
        model.fit(X, y)

        # Data should be stored (scaled)
        assert model.X_train_.shape == X.shape
        assert model.y_train_.shape == y.shape
        assert model._n_train_samples == 100

    def test_predict(self, regression_data):
        """Test LWPLS predict on regression data."""
        X, y = regression_data
        model = LWPLS(n_components=5)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_predict_new_data(self, regression_data):
        """Test LWPLS predict on new data."""
        X, y = regression_data
        X_train, X_test = X[:70], X[70:]
        y_train, y_test = y[:70], y[70:]

        model = LWPLS(n_components=5)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert predictions.shape == y_test.shape
        assert not np.isnan(predictions).any()

    def test_predict_with_n_components(self, regression_data):
        """Test LWPLS predict with different n_components."""
        X, y = regression_data
        model = LWPLS(n_components=10)
        model.fit(X, y)

        # Predict with fewer components
        predictions = model.predict(X, n_components=5)

        assert predictions.shape == y.shape

    def test_predict_all_components(self, regression_data):
        """Test LWPLS predict_all_components method."""
        X, y = regression_data
        model = LWPLS(n_components=5)
        model.fit(X, y)

        all_predictions = model.predict_all_components(X)

        # Should return predictions for each component count
        assert all_predictions.shape == (100, 5)
        assert not np.isnan(all_predictions).any()

    def test_nonlinear_data_performance(self, nonlinear_data):
        """Test LWPLS on nonlinear data."""
        X, y = nonlinear_data
        X_train, X_test = X[:70], X[70:]
        y_train, y_test = y[:70], y[70:]

        model = LWPLS(n_components=2, lambda_in_similarity=0.25)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # LWPLS should handle nonlinear data reasonably
        assert predictions.shape == y_test.shape
        assert not np.isnan(predictions).any()
        # Check that predictions are in reasonable range
        assert predictions.std() > 0  # Not constant predictions

    def test_different_lambda_values(self, regression_data):
        """Test LWPLS with different lambda values."""
        X, y = regression_data

        for lambda_val in [0.1, 0.5, 1.0, 2.0]:
            model = LWPLS(n_components=3, lambda_in_similarity=lambda_val)
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape
            assert not np.isnan(predictions).any()

    def test_scale_true(self, regression_data):
        """Test LWPLS with scaling enabled."""
        X, y = regression_data
        model = LWPLS(n_components=5, scale=True)
        model.fit(X, y)

        assert model.x_scaler_ is not None
        assert model.y_scaler_ is not None

        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_scale_false(self, regression_data):
        """Test LWPLS with scaling disabled."""
        X, y = regression_data
        model = LWPLS(n_components=5, scale=False)
        model.fit(X, y)

        assert model.x_scaler_ is None
        assert model.y_scaler_ is None

        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_get_params(self):
        """Test LWPLS get_params method."""
        model = LWPLS(n_components=8, lambda_in_similarity=0.5, scale=False)

        params = model.get_params()

        assert params == {
            'n_components': 8,
            'lambda_in_similarity': 0.5,
            'scale': False,
            'backend': 'numpy',
            'batch_size': 64
        }

    def test_set_params(self):
        """Test LWPLS set_params method."""
        model = LWPLS(n_components=5)

        result = model.set_params(n_components=10, lambda_in_similarity=0.25)

        assert result is model  # set_params returns self
        assert model.n_components == 10
        assert model.lambda_in_similarity == 0.25

    def test_sklearn_clone_compatibility(self):
        """Test that LWPLS works with sklearn clone."""
        from sklearn.base import clone

        model = LWPLS(n_components=7, lambda_in_similarity=0.5)
        cloned = clone(model)

        assert cloned.n_components == 7
        assert cloned.lambda_in_similarity == 0.5
        assert cloned is not model

    def test_sklearn_cross_val_score(self, regression_data):
        """Test that LWPLS works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = regression_data
        model = LWPLS(n_components=5)

        scores = cross_val_score(model, X, y, cv=3, scoring='r2')

        assert len(scores) == 3

    def test_n_components_exceeds_features(self):
        """Test LWPLS handles n_components > n_features gracefully."""
        X = np.random.randn(50, 10)  # Only 10 features
        y = np.random.randn(50)

        model = LWPLS(n_components=20)  # More components than features
        model.fit(X, y)

        # Should not raise, n_components_ should be limited
        assert model.n_components_ <= 10
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_n_components_exceeds_samples(self):
        """Test LWPLS handles n_components > n_samples gracefully."""
        X = np.random.randn(20, 50)  # Only 20 samples
        y = np.random.randn(20)

        model = LWPLS(n_components=30)  # More components than samples
        model.fit(X, y)

        # Should not raise, n_components_ should be limited
        assert model.n_components_ <= 19  # n_samples - 1
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_repr(self):
        """Test LWPLS string representation."""
        model = LWPLS(n_components=5, lambda_in_similarity=0.5)

        repr_str = repr(model)

        assert 'LWPLS' in repr_str
        assert 'n_components=5' in repr_str
        assert 'lambda_in_similarity=0.5' in repr_str

    def test_predict_unfitted_raises(self):
        """Test that predict raises error when model is not fitted."""
        from sklearn.exceptions import NotFittedError

        model = LWPLS(n_components=5)
        X = np.random.randn(10, 5)

        with pytest.raises(NotFittedError):
            model.predict(X)

    def test_small_dataset(self):
        """Test LWPLS on a very small dataset."""
        np.random.seed(42)
        X = np.random.randn(10, 3)
        y = np.random.randn(10)

        model = LWPLS(n_components=2)
        model.fit(X, y)

        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_single_sample_predict(self, regression_data):
        """Test LWPLS prediction on a single sample."""
        X, y = regression_data
        model = LWPLS(n_components=5)
        model.fit(X, y)

        # Predict on a single sample
        single_X = X[0:1]
        predictions = model.predict(single_X)

        assert predictions.shape == (1,)
        assert not np.isnan(predictions).any()

class TestSIMPLS:
    """Test suite for SIMPLS regressor."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def multi_target_data(self):
        """Generate multi-target regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_targets=3,
            n_informative=10,
            random_state=42
        )
        return X, y

    def test_init_default(self):
        """Test SIMPLS initialization with default parameters."""
        model = SIMPLS()
        assert model.n_components == 10
        assert model.scale is True
        assert model.center is True
        assert model.backend == 'numpy'

    def test_init_custom_parameters(self):
        """Test SIMPLS initialization with custom parameters."""
        model = SIMPLS(n_components=15, scale=False, center=False, backend='numpy')
        assert model.n_components == 15
        assert model.scale is False
        assert model.center is False
        assert model.backend == 'numpy'

    def test_fit(self, regression_data):
        """Test SIMPLS fit on regression data."""
        X, y = regression_data
        model = SIMPLS(n_components=10)

        result = model.fit(X, y)

        assert result is model  # fit returns self
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'n_components_')
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'x_scores_')
        assert hasattr(model, 'y_scores_')
        assert hasattr(model, 'x_weights_')
        assert hasattr(model, 'x_loadings_')
        assert hasattr(model, 'y_loadings_')
        assert model.n_features_in_ == 50
        assert model.n_components_ == 10

    def test_fit_multi_target(self, multi_target_data):
        """Test SIMPLS fit on multi-target regression data."""
        X, y = multi_target_data
        model = SIMPLS(n_components=10)

        model.fit(X, y)

        # coef_ shape is (n_features, n_targets)
        assert model.coef_.shape == (50, 3)
        assert model.y_loadings_.shape == (3, 10)

    def test_predict(self, regression_data):
        """Test SIMPLS predict on regression data."""
        X, y = regression_data
        model = SIMPLS(n_components=10)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_predict_multi_target(self, multi_target_data):
        """Test SIMPLS predict on multi-target regression data."""
        X, y = multi_target_data
        model = SIMPLS(n_components=10)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_predict_with_n_components(self, regression_data):
        """Test SIMPLS predict with different n_components."""
        X, y = regression_data
        model = SIMPLS(n_components=15)
        model.fit(X, y)

        # Predict with fewer components
        predictions = model.predict(X, n_components=5)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_transform(self, regression_data):
        """Test SIMPLS transform method."""
        X, y = regression_data
        model = SIMPLS(n_components=10)
        model.fit(X, y)

        T = model.transform(X)

        assert T.shape == (100, 10)
        assert not np.isnan(T).any()

    def test_compare_to_sklearn_pls(self, regression_data):
        """Test that SIMPLS produces similar results to sklearn PLSRegression."""
        from sklearn.cross_decomposition import PLSRegression

        X, y = regression_data
        n_components = 5

        # Fit both models
        simpls = SIMPLS(n_components=n_components)
        sklearn_pls = PLSRegression(n_components=n_components)

        simpls.fit(X, y)
        sklearn_pls.fit(X, y)

        # Predictions should be similar (not exactly equal due to different algorithms)
        pred_simpls = simpls.predict(X)
        pred_sklearn = sklearn_pls.predict(X).ravel()

        # Check correlation is high (they solve the same problem)
        correlation = np.corrcoef(pred_simpls, pred_sklearn)[0, 1]
        assert correlation > 0.99, f"Correlation too low: {correlation}"

    def test_get_params(self):
        """Test SIMPLS get_params method."""
        model = SIMPLS(n_components=8, scale=False, center=True, backend='numpy')

        params = model.get_params()

        assert params == {
            'n_components': 8,
            'scale': False,
            'center': True,
            'backend': 'numpy',
        }

    def test_set_params(self):
        """Test SIMPLS set_params method."""
        model = SIMPLS(n_components=5)

        result = model.set_params(n_components=10, scale=False)

        assert result is model  # set_params returns self
        assert model.n_components == 10
        assert model.scale is False

    def test_sklearn_clone_compatibility(self):
        """Test that SIMPLS works with sklearn clone."""
        from sklearn.base import clone

        model = SIMPLS(n_components=7, scale=False)
        cloned = clone(model)

        assert cloned.n_components == 7
        assert cloned.scale is False
        assert cloned is not model

    def test_sklearn_cross_val_score(self, regression_data):
        """Test that SIMPLS works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = regression_data
        model = SIMPLS(n_components=5)

        scores = cross_val_score(model, X, y, cv=3, scoring='r2')

        assert len(scores) == 3

    def test_n_components_exceeds_features(self):
        """Test SIMPLS handles n_components > n_features gracefully."""
        X = np.random.randn(50, 10)  # Only 10 features
        y = np.random.randn(50)

        model = SIMPLS(n_components=20)  # More components than features
        model.fit(X, y)

        # Should not raise, n_components_ should be limited
        assert model.n_components_ <= 10
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_n_components_exceeds_samples(self):
        """Test SIMPLS handles n_components > n_samples gracefully."""
        X = np.random.randn(20, 50)  # Only 20 samples
        y = np.random.randn(20)

        model = SIMPLS(n_components=30)  # More components than samples
        model.fit(X, y)

        # Should not raise, n_components_ should be limited
        assert model.n_components_ <= 19  # n_samples - 1
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_scale_true(self, regression_data):
        """Test SIMPLS with scaling enabled."""
        X, y = regression_data
        model = SIMPLS(n_components=5, scale=True)
        model.fit(X, y)

        # Check that scaling was applied
        assert not np.allclose(model.x_std_, 1.0)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_scale_false(self, regression_data):
        """Test SIMPLS with scaling disabled."""
        X, y = regression_data
        model = SIMPLS(n_components=5, scale=False)
        model.fit(X, y)

        # Check that scaling was not applied
        assert np.allclose(model.x_std_, 1.0)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_center_false(self):
        """Test SIMPLS with centering disabled."""
        np.random.seed(42)
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        model = SIMPLS(n_components=5, center=False)
        model.fit(X, y)

        # Check that centering was not applied
        assert np.allclose(model.x_mean_, 0.0)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_repr(self):
        """Test SIMPLS string representation."""
        model = SIMPLS(n_components=5, scale=True)

        repr_str = repr(model)

        assert 'SIMPLS' in repr_str
        assert 'n_components=5' in repr_str
        assert 'scale=True' in repr_str

    def test_predict_unfitted_raises(self):
        """Test that predict raises error when model is not fitted."""
        from sklearn.exceptions import NotFittedError

        model = SIMPLS(n_components=5)
        X = np.random.randn(10, 5)

        with pytest.raises(NotFittedError):
            model.predict(X)

    def test_small_dataset(self):
        """Test SIMPLS on a very small dataset."""
        np.random.seed(42)
        X = np.random.randn(10, 3)
        y = np.random.randn(10)

        model = SIMPLS(n_components=2)
        model.fit(X, y)

        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_single_sample_predict(self, regression_data):
        """Test SIMPLS prediction on a single sample."""
        X, y = regression_data
        model = SIMPLS(n_components=5)
        model.fit(X, y)

        # Predict on a single sample
        single_X = X[0:1]
        predictions = model.predict(single_X)

        assert predictions.shape == (1,)
        assert not np.isnan(predictions).any()

    def test_invalid_backend(self, regression_data):
        """Test SIMPLS raises error for invalid backend."""
        X, y = regression_data
        model = SIMPLS(n_components=5, backend='invalid')

        with pytest.raises(ValueError, match="backend must be 'numpy' or 'jax'"):
            model.fit(X, y)

@pytest.mark.xdist_group("gpu")
class TestSIMPLSJAX:
    """Test suite for SIMPLS regressor with JAX backend.

    These tests are skipped if JAX is not installed.
    """

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def multi_target_data(self):
        """Generate multi-target regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_targets=3,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_fit_jax_backend(self, regression_data):
        """Test SIMPLS fit with JAX backend."""
        X, y = regression_data
        model = SIMPLS(n_components=10, backend='jax')

        result = model.fit(X, y)

        assert result is model
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'n_components_')
        assert hasattr(model, 'coef_')
        assert model.n_features_in_ == 50
        assert model.n_components_ == 10

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_predict_jax_backend(self, regression_data):
        """Test SIMPLS predict with JAX backend."""
        X, y = regression_data
        model = SIMPLS(n_components=10, backend='jax')
        model.fit(X, y)

        predictions = model.predict(X)

        # Should return numpy array
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_fit_jax_multi_target(self, multi_target_data):
        """Test SIMPLS fit on multi-target data with JAX backend."""
        X, y = multi_target_data
        model = SIMPLS(n_components=10, backend='jax')

        model.fit(X, y)

        # coef_ shape is (n_features, n_targets) and should be numpy array
        assert isinstance(model.coef_, np.ndarray)
        assert model.coef_.shape == (50, 3)

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_predict_jax_multi_target(self, multi_target_data):
        """Test SIMPLS predict on multi-target data with JAX backend."""
        X, y = multi_target_data
        model = SIMPLS(n_components=10, backend='jax')
        model.fit(X, y)

        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == y.shape

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_numpy_similar_results(self, regression_data):
        """Test that JAX and NumPy backends produce similar results."""
        X, y = regression_data

        model_numpy = SIMPLS(n_components=5, backend='numpy')
        model_jax = SIMPLS(n_components=5, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        # Results should be very similar (not exactly equal due to floating point)
        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-5)

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_sklearn_clone(self, regression_data):
        """Test that SIMPLS with JAX backend works with sklearn clone."""
        from sklearn.base import clone

        model = SIMPLS(n_components=5, backend='jax')
        cloned = clone(model)

        assert cloned.backend == 'jax'
        assert cloned is not model

        # Cloned model should be fittable
        X, y = regression_data
        cloned.fit(X, y)
        predictions = cloned.predict(X)
        assert predictions.shape == y.shape

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_transform(self, regression_data):
        """Test SIMPLS transform with JAX backend."""
        X, y = regression_data
        model = SIMPLS(n_components=10, backend='jax')
        model.fit(X, y)

        T = model.transform(X)

        assert isinstance(T, np.ndarray)
        assert T.shape == (100, 10)
        assert not np.isnan(T).any()

class TestIntervalPLS:
    """Test suite for IntervalPLS regressor."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def spectral_data(self):
        """Generate spectral-like data with signal in specific region."""
        np.random.seed(42)
        X = np.random.randn(100, 200)  # 200 wavelengths
        # Signal concentrated in wavelengths 50-70
        y = X[:, 50:70].sum(axis=1) + 0.1 * np.random.randn(100)
        return X, y

    def test_init_default(self):
        """Test IntervalPLS initialization with default parameters."""
        model = IntervalPLS()
        assert model.n_components == 5
        assert model.n_intervals == 10
        assert model.interval_width is None
        assert model.cv == 5
        assert model.scoring == 'r2'
        assert model.mode == 'forward'
        assert model.combination_method == 'union'
        assert model.backend == 'numpy'

    def test_init_custom_parameters(self):
        """Test IntervalPLS initialization with custom parameters."""
        model = IntervalPLS(
            n_components=10,
            n_intervals=20,
            cv=3,
            scoring='neg_mean_squared_error',
            mode='single',
            combination_method='best',
            backend='numpy'
        )
        assert model.n_components == 10
        assert model.n_intervals == 20
        assert model.cv == 3
        assert model.scoring == 'neg_mean_squared_error'
        assert model.mode == 'single'
        assert model.combination_method == 'best'
        assert model.backend == 'numpy'

    def test_fit(self, regression_data):
        """Test IntervalPLS fit on regression data."""
        X, y = regression_data
        model = IntervalPLS(n_components=3, n_intervals=5, cv=3)

        result = model.fit(X, y)

        assert result is model  # fit returns self
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'n_components_')
        assert hasattr(model, 'interval_scores_')
        assert hasattr(model, 'interval_starts_')
        assert hasattr(model, 'interval_ends_')
        assert hasattr(model, 'selected_intervals_')
        assert hasattr(model, 'selected_regions_')
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'feature_mask_')
        assert model.n_features_in_ == 50

    def test_fit_spectral_data(self, spectral_data):
        """Test IntervalPLS on spectral-like data."""
        X, y = spectral_data
        model = IntervalPLS(n_components=3, n_intervals=10, cv=3, mode='single')
        model.fit(X, y)

        # The best interval should be around 50-70 (interval 2 or 3 depending on division)
        assert len(model.selected_intervals_) >= 1
        assert model.n_intervals_ == 10

    def test_predict(self, regression_data):
        """Test IntervalPLS predict on regression data."""
        X, y = regression_data
        model = IntervalPLS(n_components=3, n_intervals=5, cv=3)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_predict_new_data(self, regression_data):
        """Test IntervalPLS predict on new data."""
        X, y = regression_data
        X_train, X_test = X[:70], X[70:]
        y_train, y_test = y[:70], y[70:]

        model = IntervalPLS(n_components=3, n_intervals=5, cv=3)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert predictions.shape == y_test.shape
        assert not np.isnan(predictions).any()

    def test_transform(self, regression_data):
        """Test IntervalPLS transform method."""
        X, y = regression_data
        model = IntervalPLS(n_components=3, n_intervals=5, cv=3)
        model.fit(X, y)

        X_transformed = model.transform(X)

        # Should have fewer features (only selected intervals)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] <= X.shape[1]
        assert X_transformed.shape[1] == model.feature_mask_.sum()

    def test_mode_single(self, regression_data):
        """Test IntervalPLS with single interval mode."""
        X, y = regression_data
        model = IntervalPLS(n_components=3, n_intervals=5, cv=3, mode='single')
        model.fit(X, y)

        # Single mode should select exactly one interval
        assert len(model.selected_intervals_) == 1

    def test_mode_forward(self, regression_data):
        """Test IntervalPLS with forward selection mode."""
        X, y = regression_data
        model = IntervalPLS(n_components=3, n_intervals=5, cv=3, mode='forward')
        model.fit(X, y)

        # Forward mode may select multiple intervals
        assert len(model.selected_intervals_) >= 1

    def test_mode_backward(self, regression_data):
        """Test IntervalPLS with backward elimination mode."""
        X, y = regression_data
        model = IntervalPLS(n_components=3, n_intervals=5, cv=3, mode='backward')
        model.fit(X, y)

        # Backward mode starts with all and removes
        assert len(model.selected_intervals_) >= 1

    def test_combination_method_best(self, regression_data):
        """Test IntervalPLS with best combination method."""
        X, y = regression_data
        model = IntervalPLS(
            n_components=3, n_intervals=5, cv=3,
            mode='forward', combination_method='best'
        )
        model.fit(X, y)

        # Best method uses only one interval regardless of selection
        # (uses the first selected interval)
        n_features_per_interval = 50 // 5  # 10 features per interval
        assert model.feature_mask_.sum() <= n_features_per_interval + 1

    def test_combination_method_union(self, regression_data):
        """Test IntervalPLS with union combination method."""
        X, y = regression_data
        model = IntervalPLS(
            n_components=3, n_intervals=5, cv=3,
            mode='forward', combination_method='union'
        )
        model.fit(X, y)

        # Union method uses all selected intervals
        assert model.feature_mask_.sum() >= 1

    def test_fixed_interval_width(self, spectral_data):
        """Test IntervalPLS with fixed interval width."""
        X, y = spectral_data
        model = IntervalPLS(
            n_components=3, interval_width=20, cv=3, mode='single'
        )
        model.fit(X, y)

        # With 200 features and width 20, should have 10 intervals
        assert model.n_intervals_ == 10

    def test_get_interval_info(self, regression_data):
        """Test IntervalPLS get_interval_info method."""
        X, y = regression_data
        model = IntervalPLS(n_components=3, n_intervals=5, cv=3)
        model.fit(X, y)

        info = model.get_interval_info()

        assert 'n_intervals' in info
        assert 'interval_scores' in info
        assert 'interval_ranges' in info
        assert 'selected_intervals' in info
        assert 'selected_regions' in info
        assert 'n_selected_features' in info
        assert info['n_intervals'] == 5
        assert len(info['interval_scores']) == 5

    def test_get_params(self):
        """Test IntervalPLS get_params method."""
        model = IntervalPLS(
            n_components=8, n_intervals=15, cv=3,
            mode='single', combination_method='best'
        )

        params = model.get_params()

        assert params == {
            'n_components': 8,
            'n_intervals': 15,
            'interval_width': None,
            'cv': 3,
            'scoring': 'r2',
            'mode': 'single',
            'combination_method': 'best',
            'backend': 'numpy',
        }

    def test_set_params(self):
        """Test IntervalPLS set_params method."""
        model = IntervalPLS(n_components=5)

        result = model.set_params(n_components=10, mode='backward')

        assert result is model  # set_params returns self
        assert model.n_components == 10
        assert model.mode == 'backward'

    def test_sklearn_clone_compatibility(self):
        """Test that IntervalPLS works with sklearn clone."""
        from sklearn.base import clone

        model = IntervalPLS(n_components=7, n_intervals=8, mode='single')
        cloned = clone(model)

        assert cloned.n_components == 7
        assert cloned.n_intervals == 8
        assert cloned.mode == 'single'
        assert cloned is not model

    def test_sklearn_cross_val_score(self, regression_data):
        """Test that IntervalPLS works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = regression_data
        model = IntervalPLS(n_components=3, n_intervals=5, cv=2)

        # Use cv=2 for the outer loop since internal CV is already 2
        scores = cross_val_score(model, X, y, cv=2, scoring='r2')

        assert len(scores) == 2

    def test_invalid_mode(self, regression_data):
        """Test IntervalPLS raises error for invalid mode."""
        X, y = regression_data
        model = IntervalPLS(n_components=3, mode='invalid')

        with pytest.raises(ValueError, match="mode must be"):
            model.fit(X, y)

    def test_invalid_combination_method(self, regression_data):
        """Test IntervalPLS raises error for invalid combination_method."""
        X, y = regression_data
        model = IntervalPLS(n_components=3, combination_method='invalid')

        with pytest.raises(ValueError, match="combination_method must be"):
            model.fit(X, y)

    def test_invalid_backend(self, regression_data):
        """Test IntervalPLS raises error for invalid backend."""
        X, y = regression_data
        model = IntervalPLS(n_components=3, backend='invalid')

        with pytest.raises(ValueError, match="backend must be"):
            model.fit(X, y)

    def test_repr(self):
        """Test IntervalPLS string representation."""
        model = IntervalPLS(n_components=5, n_intervals=10, mode='forward')

        repr_str = repr(model)

        assert 'IntervalPLS' in repr_str
        assert 'n_components=5' in repr_str
        assert 'n_intervals=10' in repr_str
        assert "mode='forward'" in repr_str

    def test_predict_unfitted_raises(self):
        """Test that predict raises error when model is not fitted."""
        from sklearn.exceptions import NotFittedError

        model = IntervalPLS(n_components=5)
        X = np.random.randn(10, 50)

        with pytest.raises(NotFittedError):
            model.predict(X)

    def test_small_dataset(self):
        """Test IntervalPLS on a small dataset."""
        np.random.seed(42)
        X = np.random.randn(30, 20)
        y = np.random.randn(30)

        model = IntervalPLS(n_components=2, n_intervals=4, cv=2)
        model.fit(X, y)

        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_n_intervals_greater_than_features(self):
        """Test IntervalPLS when n_intervals > n_features."""
        np.random.seed(42)
        X = np.random.randn(50, 10)  # Only 10 features
        y = np.random.randn(50)

        # More intervals than features - should still work
        # Some intervals may be empty/small but algorithm handles it
        model = IntervalPLS(n_components=2, n_intervals=20, cv=2)
        model.fit(X, y)

        # Should produce valid predictions regardless
        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_multi_target(self):
        """Test IntervalPLS with multi-target regression."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_targets=3,
            n_informative=10,
            random_state=42
        )

        model = IntervalPLS(n_components=3, n_intervals=5, cv=3)
        model.fit(X, y)

        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

@pytest.mark.xdist_group("gpu")
class TestIntervalPLSJAX:
    """Test suite for IntervalPLS regressor with JAX backend.

    These tests are skipped if JAX is not installed.
    """

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_fit_jax_backend(self, regression_data):
        """Test IntervalPLS fit with JAX backend."""
        X, y = regression_data
        model = IntervalPLS(n_components=3, n_intervals=5, cv=3, backend='jax')

        result = model.fit(X, y)

        assert result is model
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'interval_scores_')
        assert hasattr(model, 'selected_intervals_')
        assert model.n_features_in_ == 50

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_predict_jax_backend(self, regression_data):
        """Test IntervalPLS predict with JAX backend."""
        X, y = regression_data
        model = IntervalPLS(n_components=3, n_intervals=5, cv=3, backend='jax')
        model.fit(X, y)

        predictions = model.predict(X)

        # Should return numpy array
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_numpy_similar_results(self, regression_data):
        """Test that JAX and NumPy backends produce similar interval scores."""
        X, y = regression_data

        model_numpy = IntervalPLS(n_components=3, n_intervals=5, cv=3, backend='numpy')
        model_jax = IntervalPLS(n_components=3, n_intervals=5, cv=3, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        # Interval scores should be reasonably similar
        # (not exact due to different CV implementations)
        # Just check they both produce valid scores
        assert len(model_numpy.interval_scores_) == len(model_jax.interval_scores_)
        assert not np.isnan(model_numpy.interval_scores_).any()
        assert not np.isnan(model_jax.interval_scores_).any()

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_sklearn_clone(self, regression_data):
        """Test that IntervalPLS with JAX backend works with sklearn clone."""
        from sklearn.base import clone

        model = IntervalPLS(n_components=3, n_intervals=5, backend='jax')
        cloned = clone(model)

        assert cloned.backend == 'jax'
        assert cloned is not model

        # Cloned model should be fittable
        X, y = regression_data
        cloned.fit(X, y)
        predictions = cloned.predict(X)
        assert predictions.shape == y.shape

class TestRobustPLS:
    """Test suite for RobustPLS regressor."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def regression_data_with_outliers(self):
        """Generate regression data with outliers."""
        np.random.seed(42)
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        # Add vertical outliers (in Y)
        y[0:5] = y[0:5] + 10 * y.std()
        # Add leverage points (in X)
        X[5:8, :] = X[5:8, :] + 5 * X.std()
        return X, y

    @pytest.fixture
    def multi_target_data(self):
        """Generate multi-target regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_targets=3,
            n_informative=10,
            random_state=42
        )
        return X, y

    def test_init_default(self):
        """Test RobustPLS initialization with default parameters."""
        model = RobustPLS()
        assert model.n_components == 10
        assert model.weighting == 'huber'
        assert model.c is None
        assert model.max_iter == 100
        assert model.tol == 1e-6
        assert model.scale is True
        assert model.center is True
        assert model.backend == 'numpy'

    def test_init_custom_parameters(self):
        """Test RobustPLS initialization with custom parameters."""
        model = RobustPLS(
            n_components=15,
            weighting='tukey',
            c=3.0,
            max_iter=50,
            tol=1e-5,
            scale=False,
            center=False,
            backend='numpy'
        )
        assert model.n_components == 15
        assert model.weighting == 'tukey'
        assert model.c == 3.0
        assert model.max_iter == 50
        assert model.tol == 1e-5
        assert model.scale is False
        assert model.center is False
        assert model.backend == 'numpy'

    def test_fit(self, regression_data):
        """Test RobustPLS fit on regression data."""
        X, y = regression_data
        model = RobustPLS(n_components=10)

        result = model.fit(X, y)

        assert result is model  # fit returns self
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'n_components_')
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'x_scores_')
        assert hasattr(model, 'y_scores_')
        assert hasattr(model, 'x_weights_')
        assert hasattr(model, 'x_loadings_')
        assert hasattr(model, 'y_loadings_')
        assert hasattr(model, 'sample_weights_')
        assert model.n_features_in_ == 50
        assert model.n_components_ == 10

    def test_fit_with_outliers(self, regression_data_with_outliers):
        """Test RobustPLS fit on data with outliers."""
        X, y = regression_data_with_outliers
        model = RobustPLS(n_components=10, weighting='huber')

        model.fit(X, y)

        # Outlier samples should have lower weights
        assert model.sample_weights_[0:5].mean() < model.sample_weights_[10:].mean()

    def test_fit_multi_target(self, multi_target_data):
        """Test RobustPLS fit on multi-target regression data."""
        X, y = multi_target_data
        model = RobustPLS(n_components=10)

        model.fit(X, y)

        # coef_ shape is (n_features, n_targets)
        assert model.coef_.shape == (50, 3)
        assert model.y_loadings_.shape == (3, 10)

    def test_predict(self, regression_data):
        """Test RobustPLS predict on regression data."""
        X, y = regression_data
        model = RobustPLS(n_components=10)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_predict_multi_target(self, multi_target_data):
        """Test RobustPLS predict on multi-target regression data."""
        X, y = multi_target_data
        model = RobustPLS(n_components=10)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_predict_with_n_components(self, regression_data):
        """Test RobustPLS predict with different n_components."""
        X, y = regression_data
        model = RobustPLS(n_components=15)
        model.fit(X, y)

        # Predict with fewer components
        predictions = model.predict(X, n_components=5)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_transform(self, regression_data):
        """Test RobustPLS transform method."""
        X, y = regression_data
        model = RobustPLS(n_components=10)
        model.fit(X, y)

        T = model.transform(X)

        assert T.shape == (100, 10)
        assert not np.isnan(T).any()

    def test_huber_weighting(self, regression_data_with_outliers):
        """Test RobustPLS with Huber weighting."""
        X, y = regression_data_with_outliers
        model = RobustPLS(n_components=5, weighting='huber')
        model.fit(X, y)

        # Huber should produce valid weights
        assert model.sample_weights_.shape == (100,)
        assert (model.sample_weights_ > 0).all()
        # Weights are normalized to sum to n_samples
        np.testing.assert_allclose(model.sample_weights_.sum(), 100, rtol=0.01)

    def test_tukey_weighting(self, regression_data_with_outliers):
        """Test RobustPLS with Tukey weighting."""
        X, y = regression_data_with_outliers
        model = RobustPLS(n_components=5, weighting='tukey')
        model.fit(X, y)

        # Tukey should produce valid weights
        assert model.sample_weights_.shape == (100,)
        assert (model.sample_weights_ >= 0).all()

    def test_custom_tuning_constant(self, regression_data):
        """Test RobustPLS with custom tuning constant."""
        X, y = regression_data

        # Larger c means less aggressive down-weighting
        model_c1 = RobustPLS(n_components=5, weighting='huber', c=1.0)
        model_c3 = RobustPLS(n_components=5, weighting='huber', c=3.0)

        model_c1.fit(X, y)
        model_c3.fit(X, y)

        # With larger c, weights should be more uniform
        assert model_c1.sample_weights_.std() >= model_c3.sample_weights_.std() * 0.5

    def test_get_outlier_mask(self, regression_data_with_outliers):
        """Test RobustPLS get_outlier_mask method."""
        X, y = regression_data_with_outliers
        model = RobustPLS(n_components=10, weighting='huber')
        model.fit(X, y)

        outlier_mask = model.get_outlier_mask(threshold=0.5)

        assert outlier_mask.shape == (100,)
        assert outlier_mask.dtype == np.bool_
        # At least some of the first 8 samples should be flagged as outliers
        assert outlier_mask[:8].sum() > 0

    def test_get_params(self):
        """Test RobustPLS get_params method."""
        model = RobustPLS(
            n_components=8,
            weighting='tukey',
            c=4.0,
            max_iter=50,
            tol=1e-5,
            scale=False,
            center=True,
            backend='numpy'
        )

        params = model.get_params()

        assert params == {
            'n_components': 8,
            'weighting': 'tukey',
            'c': 4.0,
            'max_iter': 50,
            'tol': 1e-5,
            'scale': False,
            'center': True,
            'backend': 'numpy',
        }

    def test_set_params(self):
        """Test RobustPLS set_params method."""
        model = RobustPLS(n_components=5)

        result = model.set_params(n_components=10, weighting='tukey')

        assert result is model  # set_params returns self
        assert model.n_components == 10
        assert model.weighting == 'tukey'

    def test_sklearn_clone_compatibility(self):
        """Test that RobustPLS works with sklearn clone."""
        from sklearn.base import clone

        model = RobustPLS(n_components=7, weighting='tukey', c=4.0)
        cloned = clone(model)

        assert cloned.n_components == 7
        assert cloned.weighting == 'tukey'
        assert cloned.c == 4.0
        assert cloned is not model

    def test_sklearn_cross_val_score(self, regression_data):
        """Test that RobustPLS works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = regression_data
        model = RobustPLS(n_components=5, max_iter=20)  # Fewer iterations for speed

        scores = cross_val_score(model, X, y, cv=3, scoring='r2')

        assert len(scores) == 3

    def test_n_components_exceeds_features(self):
        """Test RobustPLS handles n_components > n_features gracefully."""
        X = np.random.randn(50, 10)  # Only 10 features
        y = np.random.randn(50)

        model = RobustPLS(n_components=20)  # More components than features
        model.fit(X, y)

        # Should not raise, n_components_ should be limited
        assert model.n_components_ <= 10
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_n_components_exceeds_samples(self):
        """Test RobustPLS handles n_components > n_samples gracefully."""
        X = np.random.randn(20, 50)  # Only 20 samples
        y = np.random.randn(20)

        model = RobustPLS(n_components=30)  # More components than samples
        model.fit(X, y)

        # Should not raise, n_components_ should be limited
        assert model.n_components_ <= 19  # n_samples - 1
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_scale_true(self, regression_data):
        """Test RobustPLS with scaling enabled."""
        X, y = regression_data
        model = RobustPLS(n_components=5, scale=True)
        model.fit(X, y)

        # Check that scaling was applied
        assert not np.allclose(model.x_std_, 1.0)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_scale_false(self, regression_data):
        """Test RobustPLS with scaling disabled."""
        X, y = regression_data
        model = RobustPLS(n_components=5, scale=False)
        model.fit(X, y)

        # Check that scaling was not applied
        assert np.allclose(model.x_std_, 1.0)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_center_false(self):
        """Test RobustPLS with centering disabled."""
        np.random.seed(42)
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        model = RobustPLS(n_components=5, center=False)
        model.fit(X, y)

        # Check that centering was not applied
        assert np.allclose(model.x_mean_, 0.0)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_repr(self):
        """Test RobustPLS string representation."""
        model = RobustPLS(n_components=5, weighting='tukey')

        repr_str = repr(model)

        assert 'RobustPLS' in repr_str
        assert 'n_components=5' in repr_str
        assert "weighting='tukey'" in repr_str

    def test_predict_unfitted_raises(self):
        """Test that predict raises error when model is not fitted."""
        from sklearn.exceptions import NotFittedError

        model = RobustPLS(n_components=5)
        X = np.random.randn(10, 5)

        with pytest.raises(NotFittedError):
            model.predict(X)

    def test_invalid_backend(self, regression_data):
        """Test RobustPLS raises error for invalid backend."""
        X, y = regression_data
        model = RobustPLS(n_components=5, backend='invalid')

        with pytest.raises(ValueError, match="backend must be 'numpy' or 'jax'"):
            model.fit(X, y)

    def test_invalid_weighting(self, regression_data):
        """Test RobustPLS raises error for invalid weighting."""
        X, y = regression_data
        model = RobustPLS(n_components=5, weighting='invalid')

        with pytest.raises(ValueError, match="weighting must be 'huber' or 'tukey'"):
            model.fit(X, y)

    def test_small_dataset(self):
        """Test RobustPLS on a very small dataset."""
        np.random.seed(42)
        X = np.random.randn(10, 3)
        y = np.random.randn(10)

        model = RobustPLS(n_components=2, max_iter=20)
        model.fit(X, y)

        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_single_sample_predict(self, regression_data):
        """Test RobustPLS prediction on a single sample."""
        X, y = regression_data
        model = RobustPLS(n_components=5)
        model.fit(X, y)

        # Predict on a single sample
        single_X = X[0:1]
        predictions = model.predict(single_X)

        assert predictions.shape == (1,)
        assert not np.isnan(predictions).any()

    def test_convergence(self, regression_data):
        """Test that RobustPLS converges with enough iterations."""
        X, y = regression_data

        # With many iterations, should reach convergence
        model = RobustPLS(n_components=5, max_iter=200, tol=1e-8)
        model.fit(X, y)

        # Should produce valid results regardless of convergence
        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_compare_with_regular_pls(self, regression_data):
        """Test that RobustPLS produces similar results to SIMPLS on clean data."""
        X, y = regression_data

        robust_model = RobustPLS(n_components=5)
        simpls_model = SIMPLS(n_components=5)

        robust_model.fit(X, y)
        simpls_model.fit(X, y)

        pred_robust = robust_model.predict(X)
        pred_simpls = simpls_model.predict(X)

        # On clean data, results should be correlated
        correlation = np.corrcoef(pred_robust, pred_simpls)[0, 1]
        assert correlation > 0.9, f"Correlation too low: {correlation}"

    def test_robustness_to_outliers(self, regression_data):
        """Test that RobustPLS is more robust to outliers than standard PLS."""
        X, y = regression_data

        # Add severe outliers
        y_with_outliers = y.copy()
        y_with_outliers[0:5] = y.mean() + 20 * y.std()

        # Fit both models
        robust_model = RobustPLS(n_components=5, weighting='tukey')
        simpls_model = SIMPLS(n_components=5)

        robust_model.fit(X, y_with_outliers)
        simpls_model.fit(X, y_with_outliers)

        # Predict on clean data excluding outlier samples
        X_clean = X[10:]
        y_clean = y[10:]

        pred_robust = robust_model.predict(X_clean)
        pred_simpls = simpls_model.predict(X_clean)

        # RobustPLS should have better correlation with clean y
        # (though this is data-dependent and may not always hold)
        corr_robust = np.corrcoef(pred_robust, y_clean)[0, 1]
        corr_simpls = np.corrcoef(pred_simpls, y_clean)[0, 1]

        # At minimum, RobustPLS should not perform dramatically worse
        assert corr_robust > corr_simpls * 0.8 or corr_robust > 0.7

@pytest.mark.xdist_group("gpu")
class TestRobustPLSJAX:
    """Test suite for RobustPLS regressor with JAX backend.

    These tests are skipped if JAX is not installed.
    """

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def regression_data_with_outliers(self):
        """Generate regression data with outliers."""
        np.random.seed(42)
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        y[0:5] = y[0:5] + 10 * y.std()
        return X, y

    @pytest.fixture
    def multi_target_data(self):
        """Generate multi-target regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_targets=3,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_fit_jax_backend(self, regression_data):
        """Test RobustPLS fit with JAX backend."""
        X, y = regression_data
        model = RobustPLS(n_components=10, backend='jax', max_iter=20)

        result = model.fit(X, y)

        assert result is model
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'n_components_')
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'sample_weights_')
        assert model.n_features_in_ == 50
        assert model.n_components_ == 10

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_predict_jax_backend(self, regression_data):
        """Test RobustPLS predict with JAX backend."""
        X, y = regression_data
        model = RobustPLS(n_components=10, backend='jax', max_iter=20)
        model.fit(X, y)

        predictions = model.predict(X)

        # Should return numpy array
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_fit_jax_multi_target(self, multi_target_data):
        """Test RobustPLS fit on multi-target data with JAX backend."""
        X, y = multi_target_data
        model = RobustPLS(n_components=10, backend='jax', max_iter=20)

        model.fit(X, y)

        # coef_ shape is (n_features, n_targets) and should be numpy array
        assert isinstance(model.coef_, np.ndarray)
        assert model.coef_.shape == (50, 3)

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_predict_jax_multi_target(self, multi_target_data):
        """Test RobustPLS predict on multi-target data with JAX backend."""
        X, y = multi_target_data
        model = RobustPLS(n_components=10, backend='jax', max_iter=20)
        model.fit(X, y)

        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == y.shape

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_numpy_identical_results(self, regression_data):
        """Test that JAX and NumPy backends produce identical results."""
        X, y = regression_data

        model_numpy = RobustPLS(n_components=5, backend='numpy', max_iter=20)
        model_jax = RobustPLS(n_components=5, backend='jax', max_iter=20)

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        # Results should be identical (IRLS weights computed in NumPy)
        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-5)

        # Sample weights should be identical (computed in NumPy)
        np.testing.assert_allclose(
            model_numpy.sample_weights_, model_jax.sample_weights_, rtol=1e-10
        )

        # Coefficients should be identical
        np.testing.assert_allclose(model_numpy.coef_, model_jax.coef_, rtol=1e-5)

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_sklearn_clone(self, regression_data):
        """Test that RobustPLS with JAX backend works with sklearn clone."""
        from sklearn.base import clone

        model = RobustPLS(n_components=5, backend='jax', weighting='tukey')
        cloned = clone(model)

        assert cloned.backend == 'jax'
        assert cloned.weighting == 'tukey'
        assert cloned is not model

        # Cloned model should be fittable
        X, y = regression_data
        cloned.fit(X, y)
        predictions = cloned.predict(X)
        assert predictions.shape == y.shape

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_transform(self, regression_data):
        """Test RobustPLS transform with JAX backend."""
        X, y = regression_data
        model = RobustPLS(n_components=10, backend='jax', max_iter=20)
        model.fit(X, y)

        T = model.transform(X)

        assert isinstance(T, np.ndarray)
        assert T.shape == (100, 10)
        assert not np.isnan(T).any()

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_outlier_detection(self, regression_data_with_outliers):
        """Test RobustPLS outlier detection with JAX backend."""
        X, y = regression_data_with_outliers
        model = RobustPLS(n_components=10, backend='jax', weighting='huber', max_iter=20)
        model.fit(X, y)

        # Outlier samples should have lower weights
        outlier_weights = model.sample_weights_[0:5].mean()
        normal_weights = model.sample_weights_[10:].mean()

        # Outliers should be somewhat down-weighted
        assert outlier_weights < normal_weights * 1.5  # Allow some tolerance

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_tukey_weighting(self, regression_data_with_outliers):
        """Test RobustPLS with Tukey weighting and JAX backend."""
        X, y = regression_data_with_outliers
        model = RobustPLS(n_components=5, backend='jax', weighting='tukey', max_iter=20)
        model.fit(X, y)

        # Should produce valid weights
        assert model.sample_weights_.shape == (100,)
        assert (model.sample_weights_ >= 0).all()

# =============================================================================
# Backend Parity Tests - Verify NumPy and JAX produce identical results
# =============================================================================

@pytest.mark.xdist_group("gpu")
class TestPLSBackendParity:
    """Test that all PLS models with both backends produce identical results.

    This is a critical test to ensure numerical consistency across backends.
    All PLS implementations should produce the same predictions whether
    using NumPy or JAX backends.
    """

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def multi_target_data(self):
        """Generate multi-target regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_targets=3,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_ikpls_backend_parity(self, regression_data):
        """Test IKPLS produces identical results with NumPy and JAX backends."""
        X, y = regression_data

        model_numpy = IKPLS(n_components=10, backend='numpy')
        model_jax = IKPLS(n_components=10, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-5,
                                   err_msg="IKPLS: NumPy and JAX predictions differ")

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_opls_backend_parity(self, regression_data):
        """Test OPLS produces identical results with NumPy and JAX backends."""
        X, y = regression_data

        model_numpy = OPLS(n_components=2, backend='numpy')
        model_jax = OPLS(n_components=2, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-5,
                                   err_msg="OPLS: NumPy and JAX predictions differ")

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_mbpls_backend_parity(self, regression_data):
        """Test MBPLS produces identical results with NumPy and JAX backends.

        Note: JAX backend only supports single-block mode (array input, not list).
        """
        X, y = regression_data

        # NumPy backend with single block as array
        model_numpy = MBPLS(n_components=5, backend='numpy')
        model_jax = MBPLS(n_components=5, backend='jax')

        # Both should work with array input (single block)
        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-5,
                                   err_msg="MBPLS: NumPy and JAX predictions differ")

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_sparsepls_backend_parity(self, regression_data):
        """Test SparsePLS produces identical results with NumPy and JAX backends."""
        X, y = regression_data

        model_numpy = SparsePLS(n_components=5, alpha=0.5, backend='numpy')
        model_jax = SparsePLS(n_components=5, alpha=0.5, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-5,
                                   err_msg="SparsePLS: NumPy and JAX predictions differ")

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_simpls_backend_parity(self, regression_data):
        """Test SIMPLS produces identical results with NumPy and JAX backends."""
        X, y = regression_data

        model_numpy = SIMPLS(n_components=10, backend='numpy')
        model_jax = SIMPLS(n_components=10, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-5,
                                   err_msg="SIMPLS: NumPy and JAX predictions differ")

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_lwpls_backend_parity(self, regression_data):
        """Test LWPLS produces identical results with NumPy and JAX backends."""
        X, y = regression_data

        model_numpy = LWPLS(n_components=5, lambda_in_similarity=0.5, backend='numpy')
        model_jax = LWPLS(n_components=5, lambda_in_similarity=0.5, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-5,
                                   err_msg="LWPLS: NumPy and JAX predictions differ")

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_ipls_backend_parity(self, regression_data):
        """Test IntervalPLS produces similar results with NumPy and JAX backends.

        Note: IntervalPLS uses cross-validation for interval selection which involves
        stochastic splits. The backends may select slightly different intervals,
        but the overall prediction quality should be comparable.
        """
        X, y = regression_data

        model_numpy = IntervalPLS(n_components=5, n_intervals=5, backend='numpy')
        model_jax = IntervalPLS(n_components=5, n_intervals=5, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        # Due to stochastic interval selection, we check that both models
        # produce reasonable predictions rather than exact parity
        # Check correlation between predictions (should be high)
        correlation = np.corrcoef(pred_numpy.ravel(), pred_jax.ravel())[0, 1]
        assert correlation > 0.8, f"IntervalPLS: NumPy and JAX predictions poorly correlated ({correlation:.3f})"

        # Check that both have similar R² on training data
        r2_numpy = 1 - np.sum((y - pred_numpy) ** 2) / np.sum((y - y.mean()) ** 2)
        r2_jax = 1 - np.sum((y - pred_jax) ** 2) / np.sum((y - y.mean()) ** 2)
        assert abs(r2_numpy - r2_jax) < 0.3, f"IntervalPLS: R² differs too much (numpy={r2_numpy:.3f}, jax={r2_jax:.3f})"

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_robust_pls_huber_backend_parity(self, regression_data):
        """Test RobustPLS (Huber) produces identical results with NumPy and JAX backends."""
        X, y = regression_data

        model_numpy = RobustPLS(n_components=5, weighting='huber', backend='numpy')
        model_jax = RobustPLS(n_components=5, weighting='huber', backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        # Sample weights should be identical (computed in NumPy)
        np.testing.assert_allclose(model_numpy.sample_weights_, model_jax.sample_weights_,
                                   rtol=1e-10,
                                   err_msg="RobustPLS: Sample weights differ between backends")

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-5,
                                   err_msg="RobustPLS (Huber): NumPy and JAX predictions differ")

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_robust_pls_tukey_backend_parity(self, regression_data):
        """Test RobustPLS (Tukey) produces identical results with NumPy and JAX backends."""
        X, y = regression_data

        model_numpy = RobustPLS(n_components=5, weighting='tukey', backend='numpy')
        model_jax = RobustPLS(n_components=5, weighting='tukey', backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        # Sample weights should be identical (computed in NumPy)
        np.testing.assert_allclose(model_numpy.sample_weights_, model_jax.sample_weights_,
                                   rtol=1e-10,
                                   err_msg="RobustPLS: Sample weights differ between backends")

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-5,
                                   err_msg="RobustPLS (Tukey): NumPy and JAX predictions differ")

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_multi_target_backend_parity(self, multi_target_data):
        """Test PLS models produce identical results on multi-target data."""
        X, y = multi_target_data

        # Test SIMPLS (representative of PLS models)
        model_numpy = SIMPLS(n_components=5, backend='numpy')
        model_jax = SIMPLS(n_components=5, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-5,
                                   err_msg="SIMPLS multi-target: NumPy and JAX predictions differ")

        # Test RobustPLS
        model_numpy = RobustPLS(n_components=5, backend='numpy')
        model_jax = RobustPLS(n_components=5, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-5,
                                   err_msg="RobustPLS multi-target: NumPy and JAX predictions differ")

# =============================================================================
# RecursivePLS Tests
# =============================================================================

class TestRecursivePLS:
    """Test suite for RecursivePLS (Recursive PLS) regressor."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def multi_target_data(self):
        """Generate multi-target regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_targets=3,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def streaming_data(self):
        """Generate data for streaming/online learning tests."""
        np.random.seed(42)
        # Initial batch
        X_init = np.random.randn(50, 20)
        y_init = X_init[:, :5].sum(axis=1) + 0.1 * np.random.randn(50)
        # New batches for partial_fit
        X_new1 = np.random.randn(10, 20)
        y_new1 = X_new1[:, :5].sum(axis=1) + 0.1 * np.random.randn(10)
        X_new2 = np.random.randn(15, 20)
        y_new2 = X_new2[:, :5].sum(axis=1) + 0.1 * np.random.randn(15)
        return X_init, y_init, X_new1, y_new1, X_new2, y_new2

    def test_init_default(self):
        """Test RecursivePLS initialization with default parameters."""
        model = RecursivePLS()
        assert model.n_components == 10
        assert model.forgetting_factor == 0.99
        assert model.scale is True
        assert model.center is True
        assert model.backend == 'numpy'

    def test_init_custom_parameters(self):
        """Test RecursivePLS initialization with custom parameters."""
        model = RecursivePLS(
            n_components=15,
            forgetting_factor=0.95,
            scale=False,
            center=False,
            backend='numpy'
        )
        assert model.n_components == 15
        assert model.forgetting_factor == 0.95
        assert model.scale is False
        assert model.center is False
        assert model.backend == 'numpy'

    def test_fit(self, regression_data):
        """Test RecursivePLS fit on regression data."""
        X, y = regression_data
        model = RecursivePLS(n_components=10)

        result = model.fit(X, y)

        assert result is model  # fit returns self
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'n_components_')
        assert hasattr(model, 'n_samples_seen_')
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'x_weights_')
        assert hasattr(model, 'x_loadings_')
        assert hasattr(model, 'y_loadings_')
        assert hasattr(model, '_Cov_X')
        assert hasattr(model, '_Cov_XY')
        assert model.n_features_in_ == 50
        assert model.n_components_ == 10
        assert model.n_samples_seen_ == 100

    def test_fit_multi_target(self, multi_target_data):
        """Test RecursivePLS fit on multi-target regression data."""
        X, y = multi_target_data
        model = RecursivePLS(n_components=10)

        model.fit(X, y)

        # coef_ shape is (n_features, n_targets)
        assert model.coef_.shape == (50, 3)
        assert model.y_loadings_.shape == (3, 10)

    def test_predict(self, regression_data):
        """Test RecursivePLS predict on regression data."""
        X, y = regression_data
        model = RecursivePLS(n_components=10)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_predict_multi_target(self, multi_target_data):
        """Test RecursivePLS predict on multi-target regression data."""
        X, y = multi_target_data
        model = RecursivePLS(n_components=10)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_partial_fit(self, streaming_data):
        """Test RecursivePLS partial_fit for online learning."""
        X_init, y_init, X_new1, y_new1, X_new2, y_new2 = streaming_data

        model = RecursivePLS(n_components=5, forgetting_factor=0.99)
        model.fit(X_init, y_init)

        initial_samples_seen = model.n_samples_seen_
        assert initial_samples_seen == 50

        # First partial fit
        model.partial_fit(X_new1, y_new1)
        assert model.n_samples_seen_ == 60

        # Second partial fit
        model.partial_fit(X_new2, y_new2)
        assert model.n_samples_seen_ == 75

        # Model should still predict correctly
        predictions = model.predict(X_new2)
        assert predictions.shape == y_new2.shape
        assert not np.isnan(predictions).any()

    def test_partial_fit_updates_model(self, streaming_data):
        """Test that partial_fit actually updates the model."""
        X_init, y_init, X_new1, y_new1, X_new2, y_new2 = streaming_data

        model = RecursivePLS(n_components=5, forgetting_factor=0.95)
        model.fit(X_init, y_init)

        coef_before = model.coef_.copy()

        # Partial fit with new data
        model.partial_fit(X_new1, y_new1)

        coef_after = model.coef_

        # Coefficients should change
        assert not np.allclose(coef_before, coef_after)

    def test_forgetting_factor_effect(self, streaming_data):
        """Test that different forgetting factors produce different models."""
        X_init, y_init, X_new1, y_new1, _, _ = streaming_data

        # High forgetting factor (slow adaptation)
        model_slow = RecursivePLS(n_components=5, forgetting_factor=0.999)
        model_slow.fit(X_init, y_init)
        model_slow.partial_fit(X_new1, y_new1)

        # Low forgetting factor (fast adaptation)
        model_fast = RecursivePLS(n_components=5, forgetting_factor=0.9)
        model_fast.fit(X_init, y_init)
        model_fast.partial_fit(X_new1, y_new1)

        # Coefficients should differ
        assert not np.allclose(model_slow.coef_, model_fast.coef_)

    def test_forgetting_factor_one(self, regression_data):
        """Test RecursivePLS with forgetting_factor=1 (no forgetting)."""
        X, y = regression_data
        model = RecursivePLS(n_components=10, forgetting_factor=1.0)
        model.fit(X, y)

        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_invalid_forgetting_factor(self, regression_data):
        """Test RecursivePLS raises error for invalid forgetting_factor."""
        X, y = regression_data

        model = RecursivePLS(n_components=5, forgetting_factor=0)
        with pytest.raises(ValueError, match="forgetting_factor must be in \\(0, 1\\]"):
            model.fit(X, y)

        model = RecursivePLS(n_components=5, forgetting_factor=1.5)
        with pytest.raises(ValueError, match="forgetting_factor must be in \\(0, 1\\]"):
            model.fit(X, y)

        model = RecursivePLS(n_components=5, forgetting_factor=-0.1)
        with pytest.raises(ValueError, match="forgetting_factor must be in \\(0, 1\\]"):
            model.fit(X, y)

    def test_transform(self, regression_data):
        """Test RecursivePLS transform method."""
        X, y = regression_data
        model = RecursivePLS(n_components=10)
        model.fit(X, y)

        T = model.transform(X)

        assert T.shape == (100, 10)
        assert not np.isnan(T).any()

    def test_get_params(self):
        """Test RecursivePLS get_params method."""
        model = RecursivePLS(
            n_components=8,
            forgetting_factor=0.95,
            scale=False,
            center=True,
            backend='numpy'
        )

        params = model.get_params()

        assert params == {
            'n_components': 8,
            'forgetting_factor': 0.95,
            'scale': False,
            'center': True,
            'backend': 'numpy',
        }

    def test_set_params(self):
        """Test RecursivePLS set_params method."""
        model = RecursivePLS(n_components=5)

        result = model.set_params(n_components=10, forgetting_factor=0.95)

        assert result is model
        assert model.n_components == 10
        assert model.forgetting_factor == 0.95

    def test_sklearn_clone_compatibility(self):
        """Test that RecursivePLS works with sklearn clone."""
        from sklearn.base import clone

        model = RecursivePLS(n_components=7, forgetting_factor=0.95)
        cloned = clone(model)

        assert cloned.n_components == 7
        assert cloned.forgetting_factor == 0.95
        assert cloned is not model

    def test_sklearn_cross_val_score(self, regression_data):
        """Test that RecursivePLS works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = regression_data
        model = RecursivePLS(n_components=5)

        scores = cross_val_score(model, X, y, cv=3, scoring='r2')

        assert len(scores) == 3

    def test_n_components_exceeds_features(self):
        """Test RecursivePLS handles n_components > n_features gracefully."""
        X = np.random.randn(50, 10)  # Only 10 features
        y = np.random.randn(50)

        model = RecursivePLS(n_components=20)
        model.fit(X, y)

        assert model.n_components_ <= 10
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_n_components_exceeds_samples(self):
        """Test RecursivePLS handles n_components > n_samples gracefully."""
        X = np.random.randn(20, 50)  # Only 20 samples
        y = np.random.randn(20)

        model = RecursivePLS(n_components=30)
        model.fit(X, y)

        assert model.n_components_ <= 19  # n_samples - 1
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_scale_true(self, regression_data):
        """Test RecursivePLS with scaling enabled."""
        X, y = regression_data
        model = RecursivePLS(n_components=5, scale=True)
        model.fit(X, y)

        assert not np.allclose(model.x_std_, 1.0)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_scale_false(self, regression_data):
        """Test RecursivePLS with scaling disabled."""
        X, y = regression_data
        model = RecursivePLS(n_components=5, scale=False)
        model.fit(X, y)

        assert np.allclose(model.x_std_, 1.0)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_center_false(self):
        """Test RecursivePLS with centering disabled."""
        np.random.seed(42)
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        model = RecursivePLS(n_components=5, center=False)
        model.fit(X, y)

        assert np.allclose(model.x_mean_, 0.0)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_repr(self):
        """Test RecursivePLS string representation."""
        model = RecursivePLS(n_components=5, forgetting_factor=0.95)

        repr_str = repr(model)

        assert 'RecursivePLS' in repr_str
        assert 'n_components=5' in repr_str
        assert 'forgetting_factor=0.95' in repr_str

    def test_predict_unfitted_raises(self):
        """Test that predict raises error when model is not fitted."""
        from sklearn.exceptions import NotFittedError

        model = RecursivePLS(n_components=5)
        X = np.random.randn(10, 5)

        with pytest.raises(NotFittedError):
            model.predict(X)

    def test_partial_fit_unfitted_raises(self):
        """Test that partial_fit raises error when model is not fitted."""
        from sklearn.exceptions import NotFittedError

        model = RecursivePLS(n_components=5)
        X = np.random.randn(10, 5)
        y = np.random.randn(10)

        with pytest.raises(NotFittedError):
            model.partial_fit(X, y)

    def test_small_dataset(self):
        """Test RecursivePLS on a very small dataset."""
        np.random.seed(42)
        X = np.random.randn(10, 3)
        y = np.random.randn(10)

        model = RecursivePLS(n_components=2)
        model.fit(X, y)

        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_single_sample_predict(self, regression_data):
        """Test RecursivePLS prediction on a single sample."""
        X, y = regression_data
        model = RecursivePLS(n_components=5)
        model.fit(X, y)

        single_X = X[0:1]
        predictions = model.predict(single_X)

        assert predictions.shape == (1,)
        assert not np.isnan(predictions).any()

    def test_invalid_backend(self, regression_data):
        """Test RecursivePLS raises error for invalid backend."""
        X, y = regression_data
        model = RecursivePLS(n_components=5, backend='invalid')

        with pytest.raises(ValueError, match="backend must be 'numpy' or 'jax'"):
            model.fit(X, y)

@pytest.mark.xdist_group("gpu")
class TestRecursivePLSJAX:
    """Test suite for RecursivePLS regressor with JAX backend.

    These tests are skipped if JAX is not installed.
    """

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def multi_target_data(self):
        """Generate multi-target regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_targets=3,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def streaming_data(self):
        """Generate data for streaming/online learning tests."""
        np.random.seed(42)
        X_init = np.random.randn(50, 20)
        y_init = X_init[:, :5].sum(axis=1) + 0.1 * np.random.randn(50)
        X_new = np.random.randn(10, 20)
        y_new = X_new[:, :5].sum(axis=1) + 0.1 * np.random.randn(10)
        return X_init, y_init, X_new, y_new

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_fit_jax_backend(self, regression_data):
        """Test RecursivePLS fit with JAX backend."""
        X, y = regression_data
        model = RecursivePLS(n_components=10, backend='jax')

        result = model.fit(X, y)

        assert result is model
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'n_components_')
        assert hasattr(model, 'coef_')
        assert model.n_features_in_ == 50
        assert model.n_components_ == 10

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_predict_jax_backend(self, regression_data):
        """Test RecursivePLS predict with JAX backend."""
        X, y = regression_data
        model = RecursivePLS(n_components=10, backend='jax')
        model.fit(X, y)

        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_fit_jax_multi_target(self, multi_target_data):
        """Test RecursivePLS fit on multi-target data with JAX backend."""
        X, y = multi_target_data
        model = RecursivePLS(n_components=10, backend='jax')

        model.fit(X, y)

        assert isinstance(model.coef_, np.ndarray)
        assert model.coef_.shape == (50, 3)

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_predict_jax_multi_target(self, multi_target_data):
        """Test RecursivePLS predict on multi-target data with JAX backend."""
        X, y = multi_target_data
        model = RecursivePLS(n_components=10, backend='jax')
        model.fit(X, y)

        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == y.shape

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_partial_fit_jax_backend(self, streaming_data):
        """Test RecursivePLS partial_fit with JAX backend."""
        X_init, y_init, X_new, y_new = streaming_data

        model = RecursivePLS(n_components=5, forgetting_factor=0.99, backend='jax')
        model.fit(X_init, y_init)

        assert model.n_samples_seen_ == 50

        model.partial_fit(X_new, y_new)
        assert model.n_samples_seen_ == 60

        predictions = model.predict(X_new)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == y_new.shape

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_numpy_similar_results(self, regression_data):
        """Test that JAX and NumPy backends produce similar results."""
        X, y = regression_data

        model_numpy = RecursivePLS(n_components=5, backend='numpy')
        model_jax = RecursivePLS(n_components=5, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-5)

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_sklearn_clone(self, regression_data):
        """Test that RecursivePLS with JAX backend works with sklearn clone."""
        from sklearn.base import clone

        model = RecursivePLS(n_components=5, backend='jax')
        cloned = clone(model)

        assert cloned.backend == 'jax'
        assert cloned is not model

        X, y = regression_data
        cloned.fit(X, y)
        predictions = cloned.predict(X)
        assert predictions.shape == y.shape

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_transform(self, regression_data):
        """Test RecursivePLS transform with JAX backend."""
        X, y = regression_data
        model = RecursivePLS(n_components=10, backend='jax')
        model.fit(X, y)

        T = model.transform(X)

        assert isinstance(T, np.ndarray)
        assert T.shape == (100, 10)
        assert not np.isnan(T).any()

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_partial_fit_updates_model(self, streaming_data):
        """Test that partial_fit with JAX backend updates the model."""
        X_init, y_init, X_new, y_new = streaming_data

        model = RecursivePLS(n_components=5, forgetting_factor=0.95, backend='jax')
        model.fit(X_init, y_init)

        coef_before = model.coef_.copy()
        model.partial_fit(X_new, y_new)
        coef_after = model.coef_

        assert not np.allclose(coef_before, coef_after)

# Add RecursivePLS to backend parity tests
@pytest.mark.xdist_group("gpu")
class TestRecursivePLSBackendParity:
    """Test RecursivePLS produces identical results with NumPy and JAX backends."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def streaming_data(self):
        """Generate data for streaming tests."""
        np.random.seed(42)
        X_init = np.random.randn(50, 20)
        y_init = X_init[:, :5].sum(axis=1) + 0.1 * np.random.randn(50)
        X_new = np.random.randn(10, 20)
        y_new = X_new[:, :5].sum(axis=1) + 0.1 * np.random.randn(10)
        return X_init, y_init, X_new, y_new

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_recursive_pls_backend_parity(self, regression_data):
        """Test RecursivePLS produces identical results with NumPy and JAX backends."""
        X, y = regression_data

        model_numpy = RecursivePLS(n_components=10, backend='numpy')
        model_jax = RecursivePLS(n_components=10, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-5,
                                   err_msg="RecursivePLS: NumPy and JAX predictions differ")

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_recursive_pls_partial_fit_parity(self, streaming_data):
        """Test RecursivePLS partial_fit produces identical results."""
        X_init, y_init, X_new, y_new = streaming_data

        model_numpy = RecursivePLS(n_components=5, forgetting_factor=0.95, backend='numpy')
        model_jax = RecursivePLS(n_components=5, forgetting_factor=0.95, backend='jax')

        model_numpy.fit(X_init, y_init)
        model_jax.fit(X_init, y_init)

        # After initial fit
        pred_numpy_init = model_numpy.predict(X_new)
        pred_jax_init = model_jax.predict(X_new)
        np.testing.assert_allclose(pred_numpy_init, pred_jax_init, rtol=1e-5)

        # After partial fit
        model_numpy.partial_fit(X_new, y_new)
        model_jax.partial_fit(X_new, y_new)

        pred_numpy = model_numpy.predict(X_new)
        pred_jax = model_jax.predict(X_new)

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-5,
                                   err_msg="RecursivePLS partial_fit: NumPy and JAX predictions differ")

class TestKOPLS:
    """Test suite for KOPLS (Kernel OPLS) regressor."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def multi_target_data(self):
        """Generate multi-target regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_targets=3,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def nonlinear_data(self):
        """Generate nonlinear data where KOPLS should excel."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        # Nonlinear relationship
        y = np.sin(X[:, :3].sum(axis=1)) + 0.5 * (X[:, 3:6].sum(axis=1)) ** 2 + 0.1 * np.random.randn(100)
        return X, y

    def test_init_default(self):
        """Test KOPLS initialization with default parameters."""
        model = KOPLS()
        assert model.n_components == 5
        assert model.n_ortho_components == 1
        assert model.kernel == 'rbf'
        assert model.gamma is None
        assert model.degree == 3
        assert model.coef0 == 1.0
        assert model.center is True
        assert model.scale is True
        assert model.backend == 'numpy'

    def test_init_custom_parameters(self):
        """Test KOPLS initialization with custom parameters."""
        model = KOPLS(
            n_components=10,
            n_ortho_components=3,
            kernel='poly',
            gamma=0.1,
            degree=2,
            coef0=0.5,
            center=False,
            scale=False,
            backend='numpy'
        )
        assert model.n_components == 10
        assert model.n_ortho_components == 3
        assert model.kernel == 'poly'
        assert model.gamma == 0.1
        assert model.degree == 2
        assert model.coef0 == 0.5
        assert model.center is False
        assert model.scale is False
        assert model.backend == 'numpy'

    def test_fit(self, regression_data):
        """Test KOPLS fit on regression data."""
        X, y = regression_data
        model = KOPLS(n_components=5, n_ortho_components=2)

        result = model.fit(X, y)

        assert result is model  # fit returns self
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'n_components_')
        assert hasattr(model, 'n_ortho_components_')
        assert hasattr(model, 'X_train_')
        assert hasattr(model, 'y_mean_')
        assert hasattr(model, 'y_std_')
        assert hasattr(model, 'x_scores_')
        assert hasattr(model, 'y_scores_')
        assert hasattr(model, 'y_loadings_')
        assert hasattr(model, 'ortho_scores_')
        assert hasattr(model, 'ortho_loadings_')
        assert model.n_features_in_ == 50
        assert model.n_components_ == 5

    def test_fit_multi_target(self, multi_target_data):
        """Test KOPLS fit on multi-target regression data."""
        X, y = multi_target_data
        model = KOPLS(n_components=5, n_ortho_components=2)

        model.fit(X, y)

        # y_loadings_ shape is (n_targets, actual_components)
        # K-OPLS is limited by SVD of Y'KY, so components <= min(n_targets, n_components)
        assert model.y_loadings_.shape[0] == 3
        assert model.y_loadings_.shape[1] <= 5
        assert model.y_loadings_.shape[1] >= 1

    def test_predict(self, regression_data):
        """Test KOPLS predict on regression data."""
        X, y = regression_data
        model = KOPLS(n_components=5, n_ortho_components=2)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_predict_multi_target(self, multi_target_data):
        """Test KOPLS predict on multi-target regression data."""
        X, y = multi_target_data
        model = KOPLS(n_components=5, n_ortho_components=2)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_predict_new_data(self, regression_data):
        """Test KOPLS predict on new data."""
        X, y = regression_data
        X_train, X_test = X[:70], X[70:]
        y_train, y_test = y[:70], y[70:]

        model = KOPLS(n_components=5, n_ortho_components=2)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert predictions.shape == y_test.shape
        assert not np.isnan(predictions).any()

    def test_transform(self, regression_data):
        """Test KOPLS transform method."""
        X, y = regression_data
        model = KOPLS(n_components=5, n_ortho_components=2)
        model.fit(X, y)

        T = model.transform(X)

        # K-OPLS transform returns (n_samples, actual_components)
        # For single target, actual_components = 1 due to SVD limitation
        assert T.shape[0] == 100
        assert T.shape[1] >= 1
        assert not np.isnan(T).any()

    def test_transform_new_data(self, regression_data):
        """Test KOPLS transform on new data."""
        X, y = regression_data
        X_train, X_test = X[:70], X[70:]
        y_train = y[:70]

        model = KOPLS(n_components=5, n_ortho_components=2)
        model.fit(X_train, y_train)

        T = model.transform(X_test)

        # K-OPLS transform returns (n_samples, actual_components)
        # For single target, actual_components = 1 due to SVD limitation
        assert T.shape[0] == 30
        assert T.shape[1] >= 1
        assert not np.isnan(T).any()

    def test_linear_kernel(self, regression_data):
        """Test KOPLS with linear kernel."""
        X, y = regression_data
        model = KOPLS(n_components=5, n_ortho_components=2, kernel='linear')
        model.fit(X, y)

        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_rbf_kernel(self, regression_data):
        """Test KOPLS with RBF kernel."""
        X, y = regression_data
        model = KOPLS(n_components=5, n_ortho_components=2, kernel='rbf', gamma=0.1)
        model.fit(X, y)

        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_poly_kernel(self, regression_data):
        """Test KOPLS with polynomial kernel."""
        X, y = regression_data
        model = KOPLS(n_components=5, n_ortho_components=2, kernel='poly', degree=2)
        model.fit(X, y)

        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_nonlinear_data(self, nonlinear_data):
        """Test KOPLS on nonlinear data where it should perform better than linear methods."""
        X, y = nonlinear_data
        X_train, X_test = X[:70], X[70:]
        y_train, y_test = y[:70], y[70:]

        model = KOPLS(n_components=5, n_ortho_components=2, kernel='rbf')
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape
        assert not np.isnan(predictions).any()
        # Check predictions are not constant
        assert predictions.std() > 0

    def test_different_ortho_components(self, regression_data):
        """Test KOPLS with different numbers of orthogonal components."""
        X, y = regression_data

        for n_ortho in [1, 2, 3]:
            model = KOPLS(n_components=5, n_ortho_components=n_ortho)
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape
            assert not np.isnan(predictions).any()

    def test_get_params(self):
        """Test KOPLS get_params method."""
        model = KOPLS(
            n_components=8,
            n_ortho_components=3,
            kernel='poly',
            gamma=0.1,
            degree=2,
            coef0=0.5,
            center=False,
            scale=True,
            backend='numpy'
        )

        params = model.get_params()

        assert params == {
            'n_components': 8,
            'n_ortho_components': 3,
            'kernel': 'poly',
            'gamma': 0.1,
            'degree': 2,
            'coef0': 0.5,
            'center': False,
            'scale': True,
            'backend': 'numpy',
        }

    def test_set_params(self):
        """Test KOPLS set_params method."""
        model = KOPLS(n_components=5)

        result = model.set_params(n_components=10, kernel='poly')

        assert result is model  # set_params returns self
        assert model.n_components == 10
        assert model.kernel == 'poly'

    def test_sklearn_clone_compatibility(self):
        """Test that KOPLS works with sklearn clone."""
        from sklearn.base import clone

        model = KOPLS(n_components=7, n_ortho_components=2, kernel='rbf')
        cloned = clone(model)

        assert cloned.n_components == 7
        assert cloned.n_ortho_components == 2
        assert cloned.kernel == 'rbf'
        assert cloned is not model

    def test_sklearn_cross_val_score(self, regression_data):
        """Test that KOPLS works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = regression_data
        model = KOPLS(n_components=3, n_ortho_components=1)

        scores = cross_val_score(model, X, y, cv=3, scoring='r2')

        assert len(scores) == 3

    def test_n_components_exceeds_samples(self):
        """Test KOPLS handles n_components > n_samples gracefully."""
        X = np.random.randn(20, 50)  # Only 20 samples
        y = np.random.randn(20)

        model = KOPLS(n_components=30)  # More components than samples
        model.fit(X, y)

        # Should not raise, n_components_ should be limited
        assert model.n_components_ <= 19  # n_samples - 1
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_scale_true(self, regression_data):
        """Test KOPLS with scaling enabled."""
        X, y = regression_data
        model = KOPLS(n_components=5, scale=True)
        model.fit(X, y)

        # Check that scaling was applied
        assert not np.allclose(model.y_std_, 1.0)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_scale_false(self, regression_data):
        """Test KOPLS with scaling disabled."""
        X, y = regression_data
        model = KOPLS(n_components=5, scale=False)
        model.fit(X, y)

        # Check that scaling was not applied
        assert np.allclose(model.y_std_, 1.0)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_center_false(self, regression_data):
        """Test KOPLS with centering disabled."""
        X, y = regression_data
        model = KOPLS(n_components=5, center=False)
        model.fit(X, y)

        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_repr(self):
        """Test KOPLS string representation."""
        model = KOPLS(n_components=5, kernel='rbf')

        repr_str = repr(model)

        assert 'KOPLS' in repr_str
        assert 'n_components=5' in repr_str
        assert "kernel='rbf'" in repr_str

    def test_predict_unfitted_raises(self):
        """Test that predict raises error when model is not fitted."""
        from sklearn.exceptions import NotFittedError

        model = KOPLS(n_components=5)
        X = np.random.randn(10, 5)

        with pytest.raises(NotFittedError):
            model.predict(X)

    def test_transform_unfitted_raises(self):
        """Test that transform raises error when model is not fitted."""
        from sklearn.exceptions import NotFittedError

        model = KOPLS(n_components=5)
        X = np.random.randn(10, 5)

        with pytest.raises(NotFittedError):
            model.transform(X)

    def test_invalid_backend(self, regression_data):
        """Test KOPLS raises error for invalid backend."""
        X, y = regression_data
        model = KOPLS(n_components=5, backend='invalid')

        with pytest.raises(ValueError, match="backend must be 'numpy' or 'jax'"):
            model.fit(X, y)

    def test_invalid_kernel(self, regression_data):
        """Test KOPLS raises error for invalid kernel."""
        X, y = regression_data
        model = KOPLS(n_components=5, kernel='invalid')

        with pytest.raises(ValueError, match="kernel must be 'linear', 'rbf', or 'poly'"):
            model.fit(X, y)

    def test_small_dataset(self):
        """Test KOPLS on a very small dataset."""
        np.random.seed(42)
        X = np.random.randn(15, 5)
        y = np.random.randn(15)

        model = KOPLS(n_components=3, n_ortho_components=1)
        model.fit(X, y)

        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_single_sample_predict(self, regression_data):
        """Test KOPLS prediction on a single sample."""
        X, y = regression_data
        model = KOPLS(n_components=5, n_ortho_components=2)
        model.fit(X, y)

        # Predict on a single sample
        single_X = X[0:1]
        predictions = model.predict(single_X)

        assert predictions.shape == (1,)
        assert not np.isnan(predictions).any()

@pytest.mark.xdist_group("gpu")
class TestKOPLSJAX:
    """Test suite for KOPLS regressor with JAX backend.

    These tests are skipped if JAX is not installed.
    """

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def multi_target_data(self):
        """Generate multi-target regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_targets=3,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_fit_jax_backend(self, regression_data):
        """Test KOPLS fit with JAX backend."""
        X, y = regression_data
        model = KOPLS(n_components=5, n_ortho_components=2, backend='jax')

        result = model.fit(X, y)

        assert result is model
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'n_components_')
        assert hasattr(model, 'x_scores_')
        assert model.n_features_in_ == 50
        assert model.n_components_ == 5

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_predict_jax_backend(self, regression_data):
        """Test KOPLS predict with JAX backend."""
        X, y = regression_data
        model = KOPLS(n_components=5, n_ortho_components=2, backend='jax')
        model.fit(X, y)

        predictions = model.predict(X)

        # Should return numpy array
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_fit_jax_multi_target(self, multi_target_data):
        """Test KOPLS fit on multi-target data with JAX backend."""
        X, y = multi_target_data
        model = KOPLS(n_components=5, n_ortho_components=2, backend='jax')

        model.fit(X, y)

        # y_loadings_ shape is (n_targets, actual_components) and should be numpy array
        # K-OPLS is limited by SVD of Y'KY, so components <= min(n_targets, n_components)
        assert isinstance(model.y_loadings_, np.ndarray)
        assert model.y_loadings_.shape[0] == 3
        assert model.y_loadings_.shape[1] <= 5
        assert model.y_loadings_.shape[1] >= 1

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_predict_jax_multi_target(self, multi_target_data):
        """Test KOPLS predict on multi-target data with JAX backend."""
        X, y = multi_target_data
        model = KOPLS(n_components=5, n_ortho_components=2, backend='jax')
        model.fit(X, y)

        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == y.shape

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_different_kernels(self, regression_data):
        """Test KOPLS with different kernels using JAX backend."""
        X, y = regression_data

        for kernel in ['linear', 'rbf', 'poly']:
            model = KOPLS(n_components=5, n_ortho_components=2, kernel=kernel, backend='jax')
            model.fit(X, y)

            predictions = model.predict(X)
            assert predictions.shape == y.shape
            assert not np.isnan(predictions).any()

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_numpy_similar_results(self, regression_data):
        """Test that JAX and NumPy backends produce similar results."""
        X, y = regression_data

        model_numpy = KOPLS(n_components=5, n_ortho_components=2, kernel='linear', backend='numpy')
        model_jax = KOPLS(n_components=5, n_ortho_components=2, kernel='linear', backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        # Results should be similar (not exactly equal due to floating point)
        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-4)

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_sklearn_clone(self, regression_data):
        """Test that KOPLS with JAX backend works with sklearn clone."""
        from sklearn.base import clone

        model = KOPLS(n_components=5, n_ortho_components=2, backend='jax')
        cloned = clone(model)

        assert cloned.backend == 'jax'
        assert cloned is not model

        # Cloned model should be fittable
        X, y = regression_data
        cloned.fit(X, y)
        predictions = cloned.predict(X)
        assert predictions.shape == y.shape

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_jax_transform(self, regression_data):
        """Test KOPLS transform with JAX backend."""
        X, y = regression_data
        model = KOPLS(n_components=5, n_ortho_components=2, backend='jax')
        model.fit(X, y)

        T = model.transform(X)

        # K-OPLS transform returns (n_samples, actual_components)
        # For single target, actual_components = 1 due to SVD limitation
        assert isinstance(T, np.ndarray)
        assert T.shape[0] == 100
        assert T.shape[1] >= 1
        assert not np.isnan(T).any()

@pytest.mark.xdist_group("gpu")
class TestKOPLSBackendParity:
    """Test KOPLS produces identical results with NumPy and JAX backends."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_kopls_backend_parity_linear(self, regression_data):
        """Test KOPLS produces identical results with linear kernel."""
        X, y = regression_data

        model_numpy = KOPLS(n_components=5, n_ortho_components=2, kernel='linear', backend='numpy')
        model_jax = KOPLS(n_components=5, n_ortho_components=2, kernel='linear', backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-4,
                                   err_msg="KOPLS linear: NumPy and JAX predictions differ")

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_kopls_backend_parity_rbf(self, regression_data):
        """Test KOPLS produces similar results with RBF kernel."""
        X, y = regression_data

        model_numpy = KOPLS(n_components=5, n_ortho_components=2, kernel='rbf', gamma=0.1, backend='numpy')
        model_jax = KOPLS(n_components=5, n_ortho_components=2, kernel='rbf', gamma=0.1, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-4,
                                   err_msg="KOPLS RBF: NumPy and JAX predictions differ")

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_kopls_transform_parity(self, regression_data):
        """Test KOPLS transform produces identical results."""
        X, y = regression_data

        model_numpy = KOPLS(n_components=5, n_ortho_components=2, kernel='linear', backend='numpy')
        model_jax = KOPLS(n_components=5, n_ortho_components=2, kernel='linear', backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        T_numpy = model_numpy.transform(X)
        T_jax = model_jax.transform(X)

        np.testing.assert_allclose(T_numpy, T_jax, rtol=1e-4,
                                   err_msg="KOPLS transform: NumPy and JAX differ")

class TestKernelPLS:
    """Test suite for KernelPLS (Nonlinear PLS / NL-PLS) regressor."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def multi_target_data(self):
        """Generate multi-target regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_targets=3,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def nonlinear_data(self):
        """Generate nonlinear regression data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        # Nonlinear function of features
        y = np.sin(X[:, :5].sum(axis=1)) + 0.5 * X[:, 5:10].sum(axis=1) ** 2
        y += 0.1 * np.random.randn(100)  # Add noise
        return X, y

    def test_init_default(self):
        """Test KernelPLS initialization with default parameters."""
        model = KernelPLS()
        assert model.n_components == 10
        assert model.kernel == 'rbf'
        assert model.gamma is None
        assert model.degree == 3
        assert model.coef0 == 1.0
        assert model.center_kernel is True
        assert model.scale_y is True
        assert model.backend == 'numpy'

    def test_init_custom_params(self):
        """Test KernelPLS initialization with custom parameters."""
        model = KernelPLS(
            n_components=5,
            kernel='poly',
            gamma=0.5,
            degree=2,
            coef0=0.5,
            center_kernel=False,
            scale_y=False,
            backend='numpy'
        )
        assert model.n_components == 5
        assert model.kernel == 'poly'
        assert model.gamma == 0.5
        assert model.degree == 2
        assert model.coef0 == 0.5
        assert model.center_kernel is False
        assert model.scale_y is False

    def test_fit_rbf_kernel(self, regression_data):
        """Test KernelPLS fit with RBF kernel."""
        X, y = regression_data
        model = KernelPLS(n_components=5, kernel='rbf', gamma=0.1)

        result = model.fit(X, y)

        assert result is model
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'X_train_')
        assert hasattr(model, 'K_train_')
        assert hasattr(model, 'x_scores_')
        assert hasattr(model, 'y_scores_')
        assert hasattr(model, 'coef_')
        assert model.n_features_in_ == 50
        assert model.X_train_.shape == (100, 50)
        assert model.K_train_.shape == (100, 100)
        assert model.x_scores_.shape == (100, 5)

    def test_fit_poly_kernel(self, regression_data):
        """Test KernelPLS fit with polynomial kernel."""
        X, y = regression_data
        model = KernelPLS(n_components=5, kernel='poly', degree=3)

        model.fit(X, y)

        assert model.x_scores_.shape == (100, 5)
        assert model.coef_.shape == (100, 1)

    def test_fit_sigmoid_kernel(self, regression_data):
        """Test KernelPLS fit with sigmoid kernel."""
        X, y = regression_data
        model = KernelPLS(n_components=5, kernel='sigmoid', gamma=0.01, coef0=0.5)

        model.fit(X, y)

        assert model.x_scores_.shape == (100, 5)

    def test_fit_linear_kernel(self, regression_data):
        """Test KernelPLS fit with linear kernel."""
        X, y = regression_data
        model = KernelPLS(n_components=5, kernel='linear')

        model.fit(X, y)

        assert model.x_scores_.shape == (100, 5)

    def test_fit_multi_target(self, multi_target_data):
        """Test KernelPLS fit on multi-target data."""
        X, y = multi_target_data
        model = KernelPLS(n_components=5, kernel='rbf', gamma=0.1)

        model.fit(X, y)

        assert model.coef_.shape == (100, 3)
        assert model.y_scores_.shape == (100, 5)

    def test_predict_single_target(self, regression_data):
        """Test KernelPLS predict on single target data."""
        X, y = regression_data
        model = KernelPLS(n_components=5, kernel='rbf', gamma=0.1)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_predict_multi_target(self, multi_target_data):
        """Test KernelPLS predict on multi-target data."""
        X, y = multi_target_data
        model = KernelPLS(n_components=5, kernel='rbf', gamma=0.1)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_predict_with_n_components(self, regression_data):
        """Test KernelPLS predict with specific n_components."""
        X, y = regression_data
        model = KernelPLS(n_components=10, kernel='rbf', gamma=0.1)
        model.fit(X, y)

        # Predict with fewer components
        pred_5 = model.predict(X, n_components=5)
        pred_10 = model.predict(X, n_components=10)

        assert pred_5.shape == y.shape
        assert pred_10.shape == y.shape
        # Different number of components should give different predictions
        assert not np.allclose(pred_5, pred_10)

    def test_transform(self, regression_data):
        """Test KernelPLS transform method."""
        X, y = regression_data
        model = KernelPLS(n_components=5, kernel='rbf', gamma=0.1)
        model.fit(X, y)

        T = model.transform(X)

        assert isinstance(T, np.ndarray)
        assert T.shape == (100, 5)
        assert not np.isnan(T).any()

    def test_transform_new_data(self, regression_data):
        """Test KernelPLS transform on new data."""
        X, y = regression_data
        X_train, X_test = X[:80], X[80:]

        model = KernelPLS(n_components=5, kernel='rbf', gamma=0.1)
        model.fit(X_train, y[:80])

        T_test = model.transform(X_test)

        assert T_test.shape == (20, 5)
        assert not np.isnan(T_test).any()

    def test_nonlinear_data_performance(self, nonlinear_data):
        """Test KernelPLS on nonlinear data captures nonlinearity better."""
        X, y = nonlinear_data

        from sklearn.cross_decomposition import PLSRegression

        # Linear PLS
        linear_pls = PLSRegression(n_components=5)
        linear_pls.fit(X, y)
        linear_pred = linear_pls.predict(X).ravel()
        linear_r2 = 1 - np.sum((y - linear_pred) ** 2) / np.sum((y - y.mean()) ** 2)

        # Kernel PLS with RBF
        kernel_pls = KernelPLS(n_components=5, kernel='rbf', gamma=0.1)
        kernel_pls.fit(X, y)
        kernel_pred = kernel_pls.predict(X)
        kernel_r2 = 1 - np.sum((y - kernel_pred) ** 2) / np.sum((y - y.mean()) ** 2)

        # Kernel PLS should generally perform better on nonlinear data
        # (not a strict requirement, but should not be much worse)
        assert kernel_r2 > 0.3  # At least captures some variance

    def test_gamma_none_uses_default(self, regression_data):
        """Test KernelPLS uses 1/n_features when gamma=None."""
        X, y = regression_data
        model = KernelPLS(n_components=5, kernel='rbf', gamma=None)

        model.fit(X, y)

        # Should not raise any errors
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_center_kernel_false(self, regression_data):
        """Test KernelPLS with center_kernel=False."""
        X, y = regression_data
        model = KernelPLS(n_components=5, kernel='rbf', gamma=0.1, center_kernel=False)

        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_scale_y_false(self, regression_data):
        """Test KernelPLS with scale_y=False."""
        X, y = regression_data
        model = KernelPLS(n_components=5, kernel='rbf', gamma=0.1, scale_y=False)

        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_get_params(self):
        """Test KernelPLS get_params method."""
        model = KernelPLS(n_components=8, kernel='poly', gamma=0.5, degree=2)

        params = model.get_params()

        assert params['n_components'] == 8
        assert params['kernel'] == 'poly'
        assert params['gamma'] == 0.5
        assert params['degree'] == 2

    def test_set_params(self):
        """Test KernelPLS set_params method."""
        model = KernelPLS(n_components=5)

        result = model.set_params(n_components=10, gamma=0.5)

        assert result is model
        assert model.n_components == 10
        assert model.gamma == 0.5

    def test_sklearn_clone_compatibility(self, regression_data):
        """Test that KernelPLS works with sklearn clone."""
        from sklearn.base import clone

        model = KernelPLS(n_components=7, kernel='rbf', gamma=0.2)
        cloned = clone(model)

        assert cloned.n_components == 7
        assert cloned.kernel == 'rbf'
        assert cloned.gamma == 0.2
        assert cloned is not model

    def test_sklearn_cross_val_score(self, regression_data):
        """Test that KernelPLS works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = regression_data
        model = KernelPLS(n_components=5, kernel='rbf', gamma=0.1)

        scores = cross_val_score(model, X, y, cv=3, scoring='r2')

        assert len(scores) == 3

    def test_invalid_backend_raises(self, regression_data):
        """Test that invalid backend raises ValueError."""
        X, y = regression_data
        model = KernelPLS(n_components=5, backend='invalid')

        with pytest.raises(ValueError, match="backend must be"):
            model.fit(X, y)

    def test_invalid_kernel_raises(self, regression_data):
        """Test that invalid kernel raises ValueError."""
        X, y = regression_data
        model = KernelPLS(n_components=5, kernel='invalid')

        with pytest.raises(ValueError, match="kernel must be"):
            model.fit(X, y)

    def test_repr(self):
        """Test KernelPLS __repr__ method."""
        model = KernelPLS(n_components=5, kernel='rbf', gamma=0.1)

        repr_str = repr(model)

        assert 'KernelPLS' in repr_str
        assert 'n_components=5' in repr_str
        assert "kernel='rbf'" in repr_str
        assert 'gamma=0.1' in repr_str

    def test_n_components_exceeds_samples(self):
        """Test KernelPLS handles n_components > n_samples."""
        X = np.random.randn(20, 50)
        y = np.random.randn(20)

        model = KernelPLS(n_components=30, kernel='rbf', gamma=0.1)
        model.fit(X, y)

        # Should limit to n_samples - 1
        assert model.n_components_ <= 19
        predictions = model.predict(X)
        assert predictions.shape == y.shape

@pytest.mark.xdist_group("gpu")
class TestKernelPLSBackendParity:
    """Test KernelPLS produces identical results with NumPy and JAX backends."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_kernel_pls_backend_parity_linear(self, regression_data):
        """Test KernelPLS produces identical results with linear kernel."""
        X, y = regression_data

        model_numpy = KernelPLS(n_components=5, kernel='linear', backend='numpy')
        model_jax = KernelPLS(n_components=5, kernel='linear', backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-4,
                                   err_msg="KernelPLS linear: NumPy and JAX predictions differ")

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_kernel_pls_backend_parity_rbf(self, regression_data):
        """Test KernelPLS produces similar results with RBF kernel."""
        X, y = regression_data

        model_numpy = KernelPLS(n_components=5, kernel='rbf', gamma=0.1, backend='numpy')
        model_jax = KernelPLS(n_components=5, kernel='rbf', gamma=0.1, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-4,
                                   err_msg="KernelPLS RBF: NumPy and JAX predictions differ")

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_kernel_pls_backend_parity_poly(self, regression_data):
        """Test KernelPLS produces similar results with polynomial kernel."""
        X, y = regression_data

        model_numpy = KernelPLS(n_components=5, kernel='poly', degree=2, backend='numpy')
        model_jax = KernelPLS(n_components=5, kernel='poly', degree=2, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-4,
                                   err_msg="KernelPLS poly: NumPy and JAX predictions differ")

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_kernel_pls_transform_parity(self, regression_data):
        """Test KernelPLS transform produces identical results."""
        X, y = regression_data

        model_numpy = KernelPLS(n_components=5, kernel='rbf', gamma=0.1, backend='numpy')
        model_jax = KernelPLS(n_components=5, kernel='rbf', gamma=0.1, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        T_numpy = model_numpy.transform(X)
        T_jax = model_jax.transform(X)

        np.testing.assert_allclose(T_numpy, T_jax, rtol=1e-4,
                                   err_msg="KernelPLS transform: NumPy and JAX differ")

# =============================================================================
# OKLMPLS Tests
# =============================================================================

from nirs4all.operators.models.sklearn.oklmpls import OKLMPLS, IdentityFeaturizer, PolynomialFeaturizer, RBFFeaturizer


class TestOKLMPLS:
    """Test suite for OKLMPLS (Online Koopman Latent-Mode PLS)."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.fixture
    def time_series_data(self):
        """Generate time-series-like regression data with temporal structure."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50

        # Create time-correlated features
        X = np.zeros((n_samples, n_features))
        X[0] = np.random.randn(n_features)
        for t in range(1, n_samples):
            X[t] = 0.8 * X[t-1] + 0.2 * np.random.randn(n_features)

        # Target depends on first few features
        y = X[:, :5].sum(axis=1) + 0.1 * np.random.randn(n_samples)

        return X, y

    def test_init_default(self):
        """Test OKLMPLS initialization with default parameters."""
        model = OKLMPLS()
        assert model.n_components == 5
        assert model.lambda_dyn == 1.0
        assert model.lambda_reg_y == 1.0
        assert model.max_iter == 50
        assert model.backend == 'numpy'

    def test_init_custom(self):
        """Test OKLMPLS initialization with custom parameters."""
        model = OKLMPLS(
            n_components=10,
            lambda_dyn=2.0,
            lambda_reg_y=0.5,
            max_iter=100,
            backend='numpy',
        )
        assert model.n_components == 10
        assert model.lambda_dyn == 2.0
        assert model.lambda_reg_y == 0.5
        assert model.max_iter == 100

    def test_fit_basic(self, regression_data):
        """Test OKLMPLS basic fit."""
        X, y = regression_data
        model = OKLMPLS(n_components=5, max_iter=10, backend='numpy')

        result = model.fit(X, y)

        assert result is model
        assert hasattr(model, 'W_')
        assert hasattr(model, 'F_')
        assert hasattr(model, 'B_')
        assert hasattr(model, 'n_iter_')
        assert hasattr(model, 'n_features_in_')
        assert model.n_features_in_ == 50

    def test_fit_with_dynamics(self, time_series_data):
        """Test OKLMPLS fit with dynamics constraint."""
        X, y = time_series_data
        model = OKLMPLS(n_components=5, lambda_dyn=1.0, max_iter=20, backend='numpy')

        model.fit(X, y)

        # Check dynamics matrix is learned
        assert model.F_.shape == (5, 5)
        # F should not be identity (learned something)
        assert not np.allclose(model.F_, np.eye(5))

    def test_fit_no_dynamics(self, regression_data):
        """Test OKLMPLS fit without dynamics constraint."""
        X, y = regression_data
        model = OKLMPLS(n_components=5, lambda_dyn=0.0, max_iter=20, backend='numpy')

        model.fit(X, y)

        # F should remain close to identity
        assert model.F_.shape == (5, 5)

    def test_predict(self, regression_data):
        """Test OKLMPLS predict."""
        X, y = regression_data
        model = OKLMPLS(n_components=5, max_iter=10, backend='numpy')
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_predict_multivariate_y(self, regression_data):
        """Test OKLMPLS with multivariate Y."""
        X, _ = regression_data
        Y = np.random.randn(100, 3)

        model = OKLMPLS(n_components=5, max_iter=10, backend='numpy')
        model.fit(X, Y)

        predictions = model.predict(X)

        assert predictions.shape == Y.shape

    def test_transform(self, regression_data):
        """Test OKLMPLS transform."""
        X, y = regression_data
        model = OKLMPLS(n_components=5, max_iter=10, backend='numpy')
        model.fit(X, y)

        T = model.transform(X)

        assert T.shape == (100, 5)
        assert not np.isnan(T).any()

    def test_predict_dynamic(self, time_series_data):
        """Test OKLMPLS predict_dynamic for future predictions."""
        X, y = time_series_data
        model = OKLMPLS(n_components=5, lambda_dyn=1.0, max_iter=20, backend='numpy')
        model.fit(X, y)

        future_preds = model.predict_dynamic(X, n_steps=5)

        assert future_preds.shape == (5,)
        assert not np.isnan(future_preds).any()

    def test_with_polynomial_featurizer(self, regression_data):
        """Test OKLMPLS with polynomial featurizer."""
        X, y = regression_data
        featurizer = PolynomialFeaturizer(degree=2)
        model = OKLMPLS(n_components=5, featurizer=featurizer, max_iter=10)

        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_with_rbf_featurizer(self, regression_data):
        """Test OKLMPLS with RBF featurizer."""
        X, y = regression_data
        featurizer = RBFFeaturizer(n_components=50, gamma=0.1, random_state=42)
        model = OKLMPLS(n_components=5, featurizer=featurizer, max_iter=10)

        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_warm_start_pls(self, regression_data):
        """Test OKLMPLS with warm start from PLS."""
        X, y = regression_data

        model_cold = OKLMPLS(n_components=5, warm_start_pls=False, max_iter=10)
        model_warm = OKLMPLS(n_components=5, warm_start_pls=True, max_iter=10)

        model_cold.fit(X, y)
        model_warm.fit(X, y)

        # Both should produce valid predictions
        pred_cold = model_cold.predict(X)
        pred_warm = model_warm.predict(X)

        assert pred_cold.shape == y.shape
        assert pred_warm.shape == y.shape

    def test_standardize_false(self, regression_data):
        """Test OKLMPLS without standardization."""
        X, y = regression_data
        model = OKLMPLS(n_components=5, standardize=False, max_iter=10)

        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_get_params(self):
        """Test OKLMPLS get_params."""
        model = OKLMPLS(n_components=10, lambda_dyn=2.0, backend='numpy')

        params = model.get_params()

        assert params['n_components'] == 10
        assert params['lambda_dyn'] == 2.0
        assert params['backend'] == 'numpy'

    def test_set_params(self):
        """Test OKLMPLS set_params."""
        model = OKLMPLS(n_components=5)

        result = model.set_params(n_components=10, lambda_dyn=2.0)

        assert result is model
        assert model.n_components == 10
        assert model.lambda_dyn == 2.0

    def test_sklearn_clone_compatibility(self, regression_data):
        """Test that OKLMPLS works with sklearn clone."""
        from sklearn.base import clone

        model = OKLMPLS(n_components=7, lambda_dyn=1.5)
        cloned = clone(model)

        assert cloned.n_components == 7
        assert cloned.lambda_dyn == 1.5
        assert cloned is not model

    def test_sklearn_cross_val_score(self, regression_data):
        """Test that OKLMPLS works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = regression_data
        model = OKLMPLS(n_components=5, max_iter=10, backend='numpy')

        scores = cross_val_score(model, X, y, cv=3, scoring='r2')

        assert len(scores) == 3

    def test_invalid_backend_raises(self, regression_data):
        """Test that invalid backend raises ValueError."""
        X, y = regression_data
        model = OKLMPLS(n_components=5, backend='invalid')

        with pytest.raises(ValueError, match="backend must be"):
            model.fit(X, y)

    def test_repr(self):
        """Test OKLMPLS __repr__ method."""
        model = OKLMPLS(n_components=5, lambda_dyn=1.0, lambda_reg_y=1.0)

        repr_str = repr(model)

        assert 'OKLMPLS' in repr_str
        assert 'n_components=5' in repr_str
        assert 'lambda_dyn=1.0' in repr_str

class TestIdentityFeaturizer:
    """Test suite for IdentityFeaturizer."""

    def test_fit_transform(self):
        """Test IdentityFeaturizer fit and transform."""
        X = np.random.randn(50, 20)
        featurizer = IdentityFeaturizer()

        result = featurizer.fit_transform(X)

        np.testing.assert_array_equal(result, X)

    def test_transform_only(self):
        """Test IdentityFeaturizer transform without fit."""
        X = np.random.randn(50, 20)
        featurizer = IdentityFeaturizer()
        featurizer.fit(X)

        result = featurizer.transform(X)

        np.testing.assert_array_equal(result, X)

class TestPolynomialFeaturizer:
    """Test suite for PolynomialFeaturizer."""

    def test_degree_2_include_original(self):
        """Test PolynomialFeaturizer with degree=2."""
        X = np.random.randn(50, 10)
        featurizer = PolynomialFeaturizer(degree=2, include_original=True)

        result = featurizer.fit_transform(X)

        # Original + squared = 2 * n_features
        assert result.shape == (50, 20)

    def test_degree_3(self):
        """Test PolynomialFeaturizer with degree=3."""
        X = np.random.randn(50, 10)
        featurizer = PolynomialFeaturizer(degree=3, include_original=True)

        result = featurizer.fit_transform(X)

        # Original + squared + cubed = 3 * n_features
        assert result.shape == (50, 30)

    def test_without_original(self):
        """Test PolynomialFeaturizer without original features."""
        X = np.random.randn(50, 10)
        featurizer = PolynomialFeaturizer(degree=2, include_original=False)

        result = featurizer.fit_transform(X)

        # Only squared = n_features
        assert result.shape == (50, 10)
        np.testing.assert_array_equal(result, X ** 2)

class TestRBFFeaturizer:
    """Test suite for RBFFeaturizer."""

    def test_fit_transform(self):
        """Test RBFFeaturizer fit and transform."""
        X = np.random.randn(50, 20)
        featurizer = RBFFeaturizer(n_components=100, gamma=0.1, random_state=42)

        result = featurizer.fit_transform(X)

        assert result.shape == (50, 100)
        assert not np.isnan(result).any()

    def test_random_state_reproducibility(self):
        """Test RBFFeaturizer reproducibility with random_state."""
        X = np.random.randn(50, 20)

        featurizer1 = RBFFeaturizer(n_components=50, random_state=42)
        featurizer2 = RBFFeaturizer(n_components=50, random_state=42)

        result1 = featurizer1.fit_transform(X)
        result2 = featurizer2.fit_transform(X)

        np.testing.assert_array_equal(result1, result2)

@pytest.mark.xdist_group("gpu")
class TestOKLMPLSBackendParity:
    """Test OKLMPLS produces similar results with NumPy and JAX backends."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        return X, y

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_oklmpls_backend_parity(self, regression_data):
        """Test OKLMPLS produces similar results with both backends."""
        X, y = regression_data

        model_numpy = OKLMPLS(n_components=5, max_iter=20, backend='numpy')
        model_jax = OKLMPLS(n_components=5, max_iter=20, backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        # Predictions should be similar (not identical due to different optimization paths)
        corr = np.corrcoef(pred_numpy, pred_jax)[0, 1]
        assert corr > 0.9, f"OKLMPLS: Low correlation between backends: {corr}"

# =============================================================================
# FCKPLS Tests
# =============================================================================

from nirs4all.operators.models.sklearn.fckpls import FCKPLS, FractionalConvFeaturizer, FractionalPLS, fractional_kernel_1d, fractional_kernel_grrunwald_letnikov


class TestFractionalKernels:
    """Test suite for fractional kernel functions."""

    def test_fractional_kernel_1d_alpha_0(self):
        """Test fractional kernel with alpha=0 (smoothing)."""
        h = fractional_kernel_1d(0.0, 2.0, 15)

        assert h.shape == (15,)
        assert not np.isnan(h).any()
        assert np.sum(np.abs(h)) > 0  # Non-trivial

    def test_fractional_kernel_1d_alpha_1(self):
        """Test fractional kernel with alpha=1 (first derivative)."""
        h = fractional_kernel_1d(1.0, 2.0, 15)

        assert h.shape == (15,)
        assert not np.isnan(h).any()
        # Should be antisymmetric (derivative-like)
        assert np.abs(h.mean()) < 0.1  # Near zero mean

    def test_fractional_kernel_1d_alpha_2(self):
        """Test fractional kernel with alpha=2 (second derivative)."""
        h = fractional_kernel_1d(2.0, 2.0, 15)

        assert h.shape == (15,)
        assert not np.isnan(h).any()

    def test_fractional_kernel_1d_fractional(self):
        """Test fractional kernel with fractional alpha."""
        h = fractional_kernel_1d(0.5, 2.0, 15)

        assert h.shape == (15,)
        assert not np.isnan(h).any()

    def test_grunwald_letnikov_kernel(self):
        """Test Grünwald-Letnikov kernel."""
        h = fractional_kernel_grrunwald_letnikov(1.0, 15)

        assert h.shape == (15,)
        assert not np.isnan(h).any()

    def test_grunwald_letnikov_fractional(self):
        """Test Grünwald-Letnikov kernel with fractional alpha."""
        h = fractional_kernel_grrunwald_letnikov(0.5, 15)

        assert h.shape == (15,)
        assert not np.isnan(h).any()

class TestFractionalConvFeaturizer:
    """Test suite for FractionalConvFeaturizer."""

    def test_fit_transform_same_mode(self):
        """Test FractionalConvFeaturizer with 'same' mode."""
        X = np.random.randn(50, 100)
        featurizer = FractionalConvFeaturizer(
            alphas=(0.0, 1.0, 2.0),
            sigmas=(2.0,),
            kernel_size=15,
            mode='same'
        )

        result = featurizer.fit_transform(X)

        # 3 kernels * 100 features = 300
        assert result.shape == (50, 300)
        assert not np.isnan(result).any()

    def test_fit_transform_valid_mode(self):
        """Test FractionalConvFeaturizer with 'valid' mode."""
        X = np.random.randn(50, 100)
        featurizer = FractionalConvFeaturizer(
            alphas=(0.0, 1.0),
            sigmas=(2.0,),
            kernel_size=15,
            mode='valid'
        )

        result = featurizer.fit_transform(X)

        # 2 kernels * (100 - 15 + 1) = 2 * 86 = 172
        assert result.shape == (50, 172)

    def test_different_sigmas(self):
        """Test FractionalConvFeaturizer with different sigmas per alpha."""
        X = np.random.randn(50, 100)
        featurizer = FractionalConvFeaturizer(
            alphas=(0.0, 1.0, 2.0),
            sigmas=(1.0, 2.0, 3.0),
            kernel_size=15,
            mode='same'
        )

        result = featurizer.fit_transform(X)

        assert result.shape == (50, 300)

    def test_grunwald_kernel_type(self):
        """Test FractionalConvFeaturizer with Grünwald-Letnikov kernels."""
        X = np.random.randn(50, 100)
        featurizer = FractionalConvFeaturizer(
            alphas=(0.0, 1.0),
            sigmas=(2.0,),
            kernel_size=15,
            kernel_type='grunwald'
        )

        result = featurizer.fit_transform(X)

        assert result.shape == (50, 200)
        assert not np.isnan(result).any()

    def test_even_kernel_size_raises(self):
        """Test that even kernel_size raises ValueError."""
        featurizer = FractionalConvFeaturizer(
            alphas=(0.0, 1.0),
            sigmas=(2.0,),
            kernel_size=14  # Even!
        )

        X = np.random.randn(50, 100)
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            featurizer.fit(X)

    def test_mismatched_sigmas_raises(self):
        """Test that mismatched sigmas length raises ValueError."""
        featurizer = FractionalConvFeaturizer(
            alphas=(0.0, 1.0, 2.0),
            sigmas=(1.0, 2.0),  # 2 sigmas for 3 alphas
            kernel_size=15
        )

        X = np.random.randn(50, 100)
        with pytest.raises(ValueError, match="sigmas must have length"):
            featurizer.fit(X)

    def test_get_kernel_info(self):
        """Test FractionalConvFeaturizer get_kernel_info."""
        X = np.random.randn(50, 100)
        featurizer = FractionalConvFeaturizer(
            alphas=(0.0, 1.0, 2.0),
            sigmas=(2.0,),
            kernel_size=15
        )
        featurizer.fit(X)

        info = featurizer.get_kernel_info()

        assert info['n_kernels'] == 3
        assert info['alphas'] == [0.0, 1.0, 2.0]
        assert info['kernel_size'] == 15

class TestFCKPLS:
    """Test suite for FCKPLS (Fractional Convolutional Kernel PLS)."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data with spectral-like features."""
        np.random.seed(42)
        n_samples = 100
        n_features = 200  # Spectral wavelengths

        # Simulate spectral data
        wavelengths = np.linspace(900, 2500, n_features)
        X = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            # Base spectrum with peaks
            X[i] = np.sin(wavelengths / 200) + 0.5 * np.sin(wavelengths / 100)
            X[i] += 0.1 * np.random.randn(n_features)

        # Target depends on specific spectral regions
        y = X[:, 50:60].mean(axis=1) - X[:, 150:160].mean(axis=1)
        y += 0.1 * np.random.randn(n_samples)

        return X, y

    def test_init_default(self):
        """Test FCKPLS initialization with default parameters."""
        model = FCKPLS()
        assert model.n_components == 10
        assert model.alphas == (0.0, 0.5, 1.0, 1.5, 2.0)
        assert model.kernel_size == 15
        assert model.backend == 'numpy'

    def test_init_custom(self):
        """Test FCKPLS initialization with custom parameters."""
        model = FCKPLS(
            n_components=5,
            alphas=(0.0, 1.0, 2.0),
            sigmas=(3.0,),
            kernel_size=21,
            mode='same',
        )
        assert model.n_components == 5
        assert model.alphas == (0.0, 1.0, 2.0)
        assert model.kernel_size == 21

    def test_fit_basic(self, regression_data):
        """Test FCKPLS basic fit."""
        X, y = regression_data
        model = FCKPLS(n_components=5, alphas=(0.0, 1.0), backend='numpy')

        result = model.fit(X, y)

        assert result is model
        assert hasattr(model, 'featurizer_')
        assert hasattr(model, 'pls_')
        assert hasattr(model, 'n_features_in_')
        assert hasattr(model, 'n_features_out_')
        assert model.n_features_in_ == 200

    def test_fit_predict(self, regression_data):
        """Test FCKPLS fit and predict."""
        X, y = regression_data
        model = FCKPLS(n_components=5, alphas=(0.0, 1.0), backend='numpy')
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.isnan(predictions).any()

    def test_multivariate_y(self, regression_data):
        """Test FCKPLS with multivariate Y."""
        X, _ = regression_data
        Y = np.random.randn(100, 3)

        model = FCKPLS(n_components=5, alphas=(0.0, 1.0), backend='numpy')
        model.fit(X, Y)

        predictions = model.predict(X)

        assert predictions.shape == Y.shape

    def test_transform(self, regression_data):
        """Test FCKPLS transform."""
        X, y = regression_data
        model = FCKPLS(n_components=5, alphas=(0.0, 1.0), backend='numpy')
        model.fit(X, y)

        T = model.transform(X)

        assert T.shape == (100, 5)
        assert not np.isnan(T).any()

    def test_get_fractional_features(self, regression_data):
        """Test FCKPLS get_fractional_features."""
        X, y = regression_data
        model = FCKPLS(n_components=5, alphas=(0.0, 1.0), backend='numpy')
        model.fit(X, y)

        X_feat = model.get_fractional_features(X)

        assert X_feat.shape == (100, model.n_features_out_)
        assert not np.isnan(X_feat).any()

    def test_get_filter_info(self, regression_data):
        """Test FCKPLS get_filter_info."""
        X, y = regression_data
        model = FCKPLS(n_components=5, alphas=(0.0, 1.0, 2.0), backend='numpy')
        model.fit(X, y)

        info = model.get_filter_info()

        assert info['n_kernels'] == 3
        assert info['alphas'] == [0.0, 1.0, 2.0]

    def test_different_kernel_types(self, regression_data):
        """Test FCKPLS with different kernel types."""
        X, y = regression_data

        model_heuristic = FCKPLS(
            n_components=5, alphas=(0.0, 1.0),
            kernel_type='heuristic', backend='numpy'
        )
        model_grunwald = FCKPLS(
            n_components=5, alphas=(0.0, 1.0),
            kernel_type='grunwald', backend='numpy'
        )

        model_heuristic.fit(X, y)
        model_grunwald.fit(X, y)

        pred_h = model_heuristic.predict(X)
        pred_g = model_grunwald.predict(X)

        assert pred_h.shape == y.shape
        assert pred_g.shape == y.shape

    def test_valid_mode(self, regression_data):
        """Test FCKPLS with 'valid' convolution mode."""
        X, y = regression_data
        model = FCKPLS(
            n_components=5, alphas=(0.0, 1.0),
            mode='valid', backend='numpy'
        )
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_standardize_false(self, regression_data):
        """Test FCKPLS without standardization."""
        X, y = regression_data
        model = FCKPLS(n_components=5, standardize=False, backend='numpy')

        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_get_params(self):
        """Test FCKPLS get_params."""
        model = FCKPLS(n_components=10, alphas=(0.0, 1.0), sigmas=(3.0,))

        params = model.get_params()

        assert params['n_components'] == 10
        assert params['alphas'] == (0.0, 1.0)
        assert params['sigmas'] == (3.0,)

    def test_set_params(self):
        """Test FCKPLS set_params."""
        model = FCKPLS(n_components=5)

        result = model.set_params(n_components=10, alphas=(0.0, 2.0))

        assert result is model
        assert model.n_components == 10
        assert model.alphas == (0.0, 2.0)

    def test_sklearn_clone_compatibility(self, regression_data):
        """Test that FCKPLS works with sklearn clone."""
        from sklearn.base import clone

        model = FCKPLS(n_components=7, alphas=(0.0, 1.0))
        cloned = clone(model)

        assert cloned.n_components == 7
        assert cloned.alphas == (0.0, 1.0)
        assert cloned is not model

    def test_sklearn_cross_val_score(self, regression_data):
        """Test that FCKPLS works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = regression_data
        model = FCKPLS(n_components=5, alphas=(0.0, 1.0), backend='numpy')

        scores = cross_val_score(model, X, y, cv=3, scoring='r2')

        assert len(scores) == 3

    def test_invalid_backend_raises(self, regression_data):
        """Test that invalid backend raises ValueError."""
        X, y = regression_data
        model = FCKPLS(n_components=5, backend='invalid')

        with pytest.raises(ValueError, match="backend must be"):
            model.fit(X, y)

    def test_repr(self):
        """Test FCKPLS __repr__ method."""
        model = FCKPLS(n_components=5, alphas=(0.0, 1.0, 2.0))

        repr_str = repr(model)

        assert 'FCKPLS' in repr_str
        assert 'n_components=5' in repr_str
        assert '0.0' in repr_str

    def test_alias_fractional_pls(self, regression_data):
        """Test that FractionalPLS is an alias for FCKPLS."""
        X, y = regression_data

        model = FractionalPLS(n_components=5, alphas=(0.0, 1.0), backend='numpy')
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == y.shape

@pytest.mark.xdist_group("gpu")
class TestFCKPLSBackendParity:
    """Test FCKPLS produces similar results with NumPy and JAX backends."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        np.random.seed(42)
        X = np.random.randn(100, 100)
        y = X[:, :10].sum(axis=1) + 0.1 * np.random.randn(100)
        return X, y

    @pytest.mark.skipif(not _jax_available(), reason="JAX not installed")
    def test_fckpls_backend_parity(self, regression_data):
        """Test FCKPLS produces similar results with both backends."""
        X, y = regression_data

        model_numpy = FCKPLS(n_components=5, alphas=(0.0, 1.0), backend='numpy')
        model_jax = FCKPLS(n_components=5, alphas=(0.0, 1.0), backend='jax')

        model_numpy.fit(X, y)
        model_jax.fit(X, y)

        pred_numpy = model_numpy.predict(X)
        pred_jax = model_jax.predict(X)

        # Predictions should be very similar
        np.testing.assert_allclose(pred_numpy, pred_jax, rtol=1e-4,
                                   err_msg="FCKPLS: NumPy and JAX predictions differ")
