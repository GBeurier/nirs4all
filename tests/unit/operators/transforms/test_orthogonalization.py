"""Unit tests for OSC and EPO operators."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all.operators.transforms import EPO, OSC


class TestOSC:
    """Test suite for Orthogonal Signal Correction (OSC)."""

    def test_osc_basic_fit_transform(self):
        """Test basic OSC fit and transform with random data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        osc = OSC(n_components=2)
        X_filtered = osc.fit_transform(X, y)

        assert X_filtered.shape == X.shape
        assert hasattr(osc, "W_ortho_")
        assert hasattr(osc, "P_ortho_")
        assert osc.W_ortho_.shape == (50, 2)
        assert osc.P_ortho_.shape == (50, 2)
        assert osc.n_components_ == 2

    def test_osc_requires_y(self):
        """Test that OSC raises error without y."""
        X = np.random.randn(100, 50)
        osc = OSC()

        with pytest.raises(TypeError):
            osc.fit(X)  # Missing y

    def test_osc_fitted_attributes(self):
        """Test that all expected attributes are set after fit."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        osc = OSC(n_components=3, scale=True)
        osc.fit(X, y)

        # Check all fitted attributes exist
        assert hasattr(osc, "W_ortho_")
        assert hasattr(osc, "P_ortho_")
        assert hasattr(osc, "X_mean_")
        assert hasattr(osc, "X_std_")
        assert hasattr(osc, "y_mean_")
        assert hasattr(osc, "y_std_")
        assert hasattr(osc, "n_features_in_")
        assert hasattr(osc, "n_components_")

        # Check shapes
        assert osc.X_mean_.shape == (50,)
        assert osc.X_std_.shape == (50,)
        assert osc.n_features_in_ == 50

    def test_osc_transform_shape(self):
        """Test that transform preserves input shape."""
        np.random.seed(42)
        X_train = np.random.randn(100, 50)
        y_train = np.random.randn(100)
        X_test = np.random.randn(30, 50)

        osc = OSC(n_components=2)
        osc.fit(X_train, y_train)
        X_test_filtered = osc.transform(X_test)

        assert X_test_filtered.shape == X_test.shape

    def test_osc_preserves_y_correlation(self):
        """Test that OSC preserves Y-predictive information."""
        np.random.seed(42)
        n_samples, n_features = 200, 100
        y = np.random.randn(n_samples)

        # Create X with Y-related and orthogonal components
        # Y-related: strong signal
        X_related = np.outer(y, np.random.randn(n_features)) * 2.0

        # Orthogonal noise: moderate
        X_ortho = np.random.randn(n_samples, n_features) * 0.5

        X = X_related + X_ortho

        osc = OSC(n_components=3)
        X_filtered = osc.fit_transform(X, y)

        # Correlation with Y should remain high
        corr_before = np.abs(np.corrcoef(X.sum(axis=1), y)[0, 1])
        corr_after = np.abs(np.corrcoef(X_filtered.sum(axis=1), y)[0, 1])

        # OSC should preserve most Y-correlation (allow some loss due to noise)
        assert corr_after >= 0.7 * corr_before, f"Correlation dropped too much: {corr_before:.3f} -> {corr_after:.3f}"

    def test_osc_sklearn_compatibility(self):
        """Test sklearn Pipeline compatibility."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        pipe = Pipeline([("osc", OSC(n_components=2)), ("scale", StandardScaler()), ("pls", PLSRegression(n_components=5))])

        pipe.fit(X, y)
        y_pred = pipe.predict(X)

        # PLS predict can return (n,) or (n, 1) depending on y shape
        assert y_pred.shape in [(100,), (100, 1)]

    def test_osc_component_limit(self):
        """Test that n_components is limited by data dimensions."""
        np.random.seed(42)
        X = np.random.randn(20, 10)  # Fewer features than requested components
        y = np.random.randn(20)

        osc = OSC(n_components=50)  # Request too many
        osc.fit(X, y)

        # Should be limited to min(n_features - 1, n_samples - 2) = min(9, 18) = 9
        assert osc.n_components_ <= 9
        assert osc.W_ortho_.shape[1] == osc.n_components_

    def test_osc_orthogonality(self):
        """Test mathematical orthogonality property."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        osc = OSC(n_components=2, scale=True)
        osc.fit(X, y)

        # Center and scale X and y as OSC does
        X_centered = (X - osc.X_mean_) / osc.X_std_
        y_centered = (y - osc.y_mean_) / osc.y_std_

        # Compute Y-predictive direction
        w_pls = X_centered.T @ y_centered
        w_pls = w_pls / np.linalg.norm(w_pls)

        # Check that orthogonal weights are orthogonal to w_pls
        for i in range(osc.n_components_):
            w_ortho = osc.W_ortho_[:, i]
            dot_product = np.dot(w_ortho, w_pls)
            assert np.abs(dot_product) < 0.1, f"Component {i} not orthogonal to Y-direction: dot={dot_product:.6f}"

    def test_osc_copy_parameter(self):
        """Test that copy parameter works correctly."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # With copy=True (default)
        osc_copy = OSC(n_components=2, copy=True)
        X_copy = X.copy()
        osc_copy.fit_transform(X_copy, y)
        # Original should be unchanged (but note: fit_transform returns new array)

        # With copy=False
        osc_no_copy = OSC(n_components=2, copy=False)
        X_no_copy = X.copy()
        osc_no_copy.fit_transform(X_no_copy, y)
        # May modify in-place (implementation detail)

    def test_osc_scale_parameter(self):
        """Test that scale parameter affects fitted attributes."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        # With scale=True
        osc_scaled = OSC(n_components=2, scale=True)
        osc_scaled.fit(X, y)
        assert not np.allclose(osc_scaled.X_mean_, 0.0)
        assert not np.allclose(osc_scaled.X_std_, 1.0)

        # With scale=False
        osc_unscaled = OSC(n_components=2, scale=False)
        osc_unscaled.fit(X, y)
        assert np.allclose(osc_unscaled.X_mean_, 0.0)
        assert np.allclose(osc_unscaled.X_std_, 1.0)

    def test_osc_multidimensional_y(self):
        """Test OSC with multi-dimensional Y."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100, 3)  # 3 targets

        osc = OSC(n_components=2)
        X_filtered = osc.fit_transform(X, y)

        assert X_filtered.shape == X.shape
        # OSC uses only first target
        assert osc.y_mean_.shape == (3,)

    def test_osc_incompatible_shapes(self):
        """Test error handling for incompatible X and y shapes."""
        X = np.random.randn(100, 50)
        y = np.random.randn(80)  # Wrong number of samples

        osc = OSC()
        with pytest.raises(ValueError, match="incompatible shapes"):
            osc.fit(X, y)

    def test_osc_sparse_input_error(self):
        """Test that sparse matrices raise TypeError."""
        from scipy.sparse import csr_matrix

        X = csr_matrix(np.random.randn(100, 50))
        y = np.random.randn(100)

        osc = OSC()
        with pytest.raises(TypeError, match="does not support scipy.sparse"):
            osc.fit(X, y)

    def test_osc_transform_before_fit_error(self):
        """Test that transform before fit raises NotFittedError."""
        from sklearn.exceptions import NotFittedError

        X = np.random.randn(100, 50)
        osc = OSC()

        with pytest.raises(NotFittedError):
            osc.transform(X)

    def test_osc_invalid_n_components(self):
        """Test error handling for invalid n_components."""
        X = np.random.randn(10, 5)
        y = np.random.randn(10)

        osc = OSC(n_components=0)
        with pytest.raises(ValueError, match="n_components must be >= 1"):
            osc.fit(X, y)

class TestEPO:
    """Test suite for External Parameter Orthogonalization (EPO)."""

    def test_epo_basic_fit_transform(self):
        """Test basic EPO fit and transform."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        temperature = np.random.randn(100)  # External parameter

        epo = EPO()
        X_filtered = epo.fit_transform(X, temperature)

        assert X_filtered.shape == X.shape
        assert hasattr(epo, "projection_coefs_")
        assert hasattr(epo, "X_mean_")
        assert hasattr(epo, "d_mean_")

    def test_epo_fitted_attributes(self):
        """Test that all expected attributes are set after fit."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        d = np.random.randn(100)

        epo = EPO(scale=True)
        epo.fit(X, d)

        assert hasattr(epo, "projection_coefs_")
        assert hasattr(epo, "X_mean_")
        assert hasattr(epo, "d_mean_")
        assert hasattr(epo, "n_features_in_")

        # Check shapes
        assert epo.X_mean_.shape == (50,)
        assert epo.projection_coefs_.shape == (1, 50)  # 1 external param, 50 features
        assert epo.n_features_in_ == 50

    def test_epo_removes_external_correlation(self):
        """Test that EPO removes correlation with external parameter."""
        np.random.seed(42)
        n_samples, n_features = 200, 50
        temperature = np.random.randn(n_samples)

        # Create X with temperature-dependent component
        X_temp = np.outer(temperature, np.random.randn(n_features)) * 2.0
        X_indep = np.random.randn(n_samples, n_features) * 0.5
        X = X_temp + X_indep

        epo = EPO()
        X_filtered = epo.fit_transform(X, temperature)

        # Correlation with temperature should decrease significantly
        corr_before = np.abs(np.corrcoef(X[:, 0], temperature)[0, 1])
        corr_after = np.abs(np.corrcoef(X_filtered[:, 0], temperature)[0, 1])

        assert corr_after < corr_before * 0.7, f"EPO did not reduce correlation: {corr_before:.3f} -> {corr_after:.3f}"

    def test_epo_multiple_parameters(self):
        """Test EPO with multiple external parameters."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        external = np.random.randn(100, 3)  # 3 external parameters

        epo = EPO()
        X_filtered = epo.fit_transform(X, external)

        assert X_filtered.shape == X.shape
        assert epo.projection_coefs_.shape == (3, 50)

    def test_epo_sklearn_compatibility(self):
        """Test sklearn Pipeline compatibility."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        temperature = np.random.randn(100)

        # Note: EPO needs external parameter during fit, which is tricky with Pipeline
        # For now, just test that the class is compatible
        epo = EPO()
        epo.fit(X, temperature)

        # Transform without external parameter
        X_test = np.random.randn(30, 50)
        X_test_filtered = epo.transform(X_test)

        assert X_test_filtered.shape == X_test.shape

    def test_epo_transform_shape(self):
        """Test that transform preserves input shape."""
        np.random.seed(42)
        X_train = np.random.randn(100, 50)
        d_train = np.random.randn(100)
        X_test = np.random.randn(30, 50)

        epo = EPO()
        epo.fit(X_train, d_train)
        X_test_filtered = epo.transform(X_test)

        assert X_test_filtered.shape == X_test.shape

    def test_epo_copy_parameter(self):
        """Test that copy parameter works correctly."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        d = np.random.randn(100)

        # With copy=True (default)
        epo_copy = EPO(copy=True)
        X_copy = X.copy()
        epo_copy.fit_transform(X_copy, d)
        # Original should be unchanged

        # With copy=False
        epo_no_copy = EPO(copy=False)
        X_no_copy = X.copy()
        epo_no_copy.fit_transform(X_no_copy, d)
        # May modify in-place

    def test_epo_scale_parameter(self):
        """Test that scale parameter affects fitted attributes."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        d = np.random.randn(100)

        # With scale=True
        epo_scaled = EPO(scale=True)
        epo_scaled.fit(X, d)
        assert not np.allclose(epo_scaled.X_mean_, 0.0)

        # With scale=False
        epo_unscaled = EPO(scale=False)
        epo_unscaled.fit(X, d)
        assert np.allclose(epo_unscaled.X_mean_, 0.0)

    def test_epo_incompatible_shapes(self):
        """Test error handling for incompatible X and d shapes."""
        X = np.random.randn(100, 50)
        d = np.random.randn(80)  # Wrong number of samples

        epo = EPO()
        with pytest.raises(ValueError, match="incompatible shapes"):
            epo.fit(X, d)

    def test_epo_sparse_input_error(self):
        """Test that sparse matrices raise TypeError."""
        from scipy.sparse import csr_matrix

        X = csr_matrix(np.random.randn(100, 50))
        d = np.random.randn(100)

        epo = EPO()
        with pytest.raises(TypeError, match="does not support scipy.sparse"):
            epo.fit(X, d)

    def test_epo_transform_before_fit_error(self):
        """Test that transform before fit raises NotFittedError."""
        from sklearn.exceptions import NotFittedError

        X = np.random.randn(100, 50)
        epo = EPO()

        with pytest.raises(NotFittedError):
            epo.transform(X)

class TestOSCDataLeakage:
    """Tests verifying OSC does not leak test data into fitting."""

    def test_osc_fit_uses_only_train_y(self):
        """Verify OSC fitted on train partition does not see test Y values.

        Constructs train/test with very different Y distributions. If test Y
        leaked into fit, the fitted y_mean_ would differ from train-only mean.
        """
        np.random.seed(42)
        n_features = 30

        # Train Y centred around 10, test Y centred around 1000
        X_train = np.random.randn(80, n_features)
        y_train = np.random.randn(80) + 10.0
        X_test = np.random.randn(20, n_features)
        y_test = np.random.randn(20) + 1000.0

        osc = OSC(n_components=1, scale=True)
        osc.fit(X_train, y_train)

        # y_mean_ should match train Y, not combined Y
        assert abs(osc.y_mean_[0] - np.mean(y_train)) < 1e-10
        assert abs(osc.y_mean_[0] - np.mean(np.concatenate([y_train, y_test]))) > 100

    def test_osc_transform_uses_train_statistics_only(self):
        """Verify transform uses only training statistics, not test data."""
        np.random.seed(42)
        n_features = 30
        X_train = np.random.randn(80, n_features) + 5.0
        y_train = np.random.randn(80)
        X_test = np.random.randn(20, n_features) + 500.0  # Very different scale

        osc = OSC(n_components=1, scale=True)
        osc.fit(X_train, y_train)

        # X_mean_ should match training data, not test
        assert np.allclose(osc.X_mean_, np.mean(X_train, axis=0), atol=1e-10)

        # Transform test data — should use training mean/std
        X_test_filtered = osc.transform(X_test)
        assert X_test_filtered.shape == X_test.shape

    def test_osc_separate_fit_transform_matches_different_data(self):
        """Verify fit and transform can use different-sized datasets safely."""
        np.random.seed(42)
        X_train = np.random.randn(100, 50)
        y_train = np.random.randn(100)
        X_all = np.random.randn(130, 50)  # All data (train + test)

        osc = OSC(n_components=2, scale=True)
        osc.fit(X_train, y_train)
        X_all_filtered = osc.transform(X_all)

        # Shapes must be preserved
        assert X_all_filtered.shape == X_all.shape
        # The fitted model should only know about 100 training samples
        assert osc.n_features_in_ == 50

    def test_osc_no_leakage_in_pipeline(self):
        """End-to-end test: OSC in nirs4all pipeline doesn't leak test data.

        Constructs a dataset where train Y ~ 0 and test Y = 999.
        If test Y leaked into OSC fitting, the model's training-fold
        predictions would be pulled toward 999. Without leakage, the
        model learns from training Y only, and test RMSE is ~999
        (model predicts ~0 on test set whose true Y = 999).
        """
        from sklearn.model_selection import KFold

        import nirs4all
        from nirs4all.data.dataset import SpectroDataset

        np.random.seed(42)
        n_features = 40
        X_train = np.random.randn(80, n_features)
        y_train = X_train[:, :5].mean(axis=1) + np.random.randn(80) * 0.1
        X_test = np.random.randn(20, n_features)
        y_test = np.full(20, 999.0)  # Extreme test Y — any leakage is obvious

        dataset = SpectroDataset(name="osc_leak_test")
        dataset.add_samples(X_train, indexes={"partition": "train"})
        dataset.add_samples(X_test, indexes={"partition": "test"})
        dataset.add_targets(np.concatenate([y_train, y_test]))

        pipeline = [OSC(n_components=1), KFold(n_splits=3), PLSRegression(n_components=5)]
        result = nirs4all.run(pipeline=pipeline, dataset=dataset, verbose=0, refit=False)

        assert result is not None
        # cv_best_score is the validation-fold score (val_score), not the
        # test-partition score.  Since all training Y ~ 0, a reasonable
        # val RMSE confirms that test Y=999 never leaked into OSC fitting.
        val_score = abs(result.cv_best_score)
        assert val_score < 5, (
            f"CV val score {val_score:.2f} is too high, suggesting "
            f"test Y values (999) may have leaked into OSC fitting"
        )

class TestOSCEPOIntegration:
    """Integration tests for OSC and EPO."""

    def test_osc_in_nirs4all_pipeline(self):
        """Test OSC integration with nirs4all.run() controller."""
        import nirs4all

        # Generate synthetic dataset
        dataset = nirs4all.generate.regression(n_samples=100)

        # Pipeline with OSC - should work with TransformerMixinController
        pipeline = [OSC(n_components=2), PLSRegression(n_components=5)]

        result = nirs4all.run(pipeline=pipeline, dataset=dataset, verbose=0)

        # Verify the pipeline completed successfully
        assert result is not None
        assert hasattr(result, "best_rmse")
        assert result.best_rmse > 0

    def test_osc_and_epo_in_sequence(self):
        """Test using OSC and EPO in sequence."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        temperature = np.random.randn(100)

        # Apply EPO first (remove temperature), then OSC (remove Y-orthogonal)
        epo = EPO()
        X_epo = epo.fit_transform(X, temperature)

        osc = OSC(n_components=2)
        X_final = osc.fit_transform(X_epo, y)

        assert X_final.shape == X.shape

    def test_comparison_with_and_without_osc(self):
        """Test that OSC affects the data meaningfully."""
        np.random.seed(42)
        n_samples, n_features = 100, 50

        # Create data with clear structure:
        # - First part: Y-related (should be preserved)
        # - Second part: Y-orthogonal (should be removed)
        y = np.random.randn(n_samples)

        # Y-related features (first 25 features)
        X_related = np.outer(y, np.ones(25)) + np.random.randn(n_samples, 25) * 0.1

        # Create a vector orthogonal to y for the remaining features
        orthog_vector = np.random.randn(n_samples)
        orthog_vector = orthog_vector - np.dot(orthog_vector, y) / np.dot(y, y) * y  # Gram-Schmidt
        orthog_vector = orthog_vector / np.linalg.norm(orthog_vector)

        # Y-orthogonal features (last 25 features)
        X_orthogonal = np.outer(orthog_vector, np.ones(25)) * 2.0 + np.random.randn(n_samples, 25) * 0.1

        X = np.hstack([X_related, X_orthogonal])

        osc = OSC(n_components=2)
        X_filtered = osc.fit_transform(X, y)

        # OSC should change the data (removes orthogonal components)
        difference = np.linalg.norm(X - X_filtered)
        assert difference > 0.1, f"OSC did not change data significantly: difference={difference}"

        # Y-correlation should be preserved or improved
        corr_before = np.abs(np.corrcoef(X.sum(axis=1), y)[0, 1])
        corr_after = np.abs(np.corrcoef(X_filtered.sum(axis=1), y)[0, 1])
        assert corr_after >= 0.8 * corr_before
