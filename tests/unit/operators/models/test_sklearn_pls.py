"""Tests for sklearn PLS model operators."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from nirs4all.operators.models.sklearn.pls import (
    PLSDA, IKPLS, OPLS, OPLSDA, MBPLS, DiPLS, SparsePLS
)
from nirs4all.operators.models.sklearn.lwpls import LWPLS


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

    @staticmethod
    def _jax_available():
        """Check if JAX is available."""
        try:
            import jax
            return True
        except ImportError:
            return False

    @pytest.mark.skipif(not _jax_available.__func__(), reason="JAX not installed")
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

    @pytest.mark.skipif(not _jax_available.__func__(), reason="JAX not installed")
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

    @pytest.mark.skipif(not _jax_available.__func__(), reason="JAX not installed")
    def test_fit_jax_multi_target(self, multi_target_data):
        """Test IKPLS fit on multi-target data with JAX backend."""
        X, y = multi_target_data
        model = IKPLS(n_components=10, backend='jax')

        model.fit(X, y)

        # coef_ shape is (n_features, n_targets) and should be numpy array
        assert isinstance(model.coef_, np.ndarray)
        assert model.coef_.shape == (50, 3)

    @pytest.mark.skipif(not _jax_available.__func__(), reason="JAX not installed")
    def test_predict_jax_multi_target(self, multi_target_data):
        """Test IKPLS predict on multi-target data with JAX backend."""
        X, y = multi_target_data
        model = IKPLS(n_components=10, backend='jax')
        model.fit(X, y)

        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == y.shape

    @pytest.mark.skipif(not _jax_available.__func__(), reason="JAX not installed")
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

    @pytest.mark.skipif(not _jax_available.__func__(), reason="JAX not installed")
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

    @pytest.mark.skipif(not _jax_available.__func__(), reason="JAX not installed")
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
            'scale': False
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
        assert hasattr(model, '_model')
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
            'max_tol': 1e-10
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
        assert hasattr(model, '_model')
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
            'scale': False
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
            'scale': False
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
