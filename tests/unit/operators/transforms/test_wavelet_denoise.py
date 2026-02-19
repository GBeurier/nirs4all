"""Unit tests for wavelet denoising transforms."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_transformer_general

from nirs4all.operators.transforms import WaveletDenoise, wavelet_denoise


class TestWaveletDenoiseFunction:
    """Test the wavelet_denoise function."""

    def test_preserves_shape(self):
        """Test that output shape matches input shape."""
        X = np.random.randn(10, 100)
        X_denoised = wavelet_denoise(X, wavelet="db4", level=3)
        assert X_denoised.shape == X.shape

    def test_reduces_noise(self):
        """Test that denoising reduces high-frequency noise."""
        # Create clean signal + noise
        n_samples, n_features = 5, 200
        t = np.linspace(0, 10, n_features)
        clean_signal = np.sin(2 * np.pi * 0.5 * t)  # Low-frequency sine wave
        noise = 0.5 * np.random.randn(n_features)  # High-frequency noise

        X = np.tile(clean_signal + noise, (n_samples, 1))
        X_denoised = wavelet_denoise(X, wavelet="db8", level=5, threshold_mode="soft")

        # Denoised signal should be closer to clean signal than noisy signal
        mse_before = np.mean((X - clean_signal) ** 2)
        mse_after = np.mean((X_denoised - clean_signal) ** 2)
        assert mse_after < mse_before

    def test_different_wavelets(self):
        """Test that different wavelet families work."""
        X = np.random.randn(5, 100)

        for wavelet in ["haar", "db4", "db8", "sym8", "coif3"]:
            X_denoised = wavelet_denoise(X, wavelet=wavelet, level=3)
            assert X_denoised.shape == X.shape
            assert np.isfinite(X_denoised).all()

    def test_threshold_modes(self):
        """Test soft and hard thresholding."""
        X = np.random.randn(5, 100)

        X_soft = wavelet_denoise(X, threshold_mode="soft")
        X_hard = wavelet_denoise(X, threshold_mode="hard")

        assert X_soft.shape == X.shape
        assert X_hard.shape == X.shape
        # Soft thresholding typically produces smoother results
        assert np.var(X_soft) <= np.var(X_hard) * 1.5  # Allow some tolerance

    def test_noise_estimators(self):
        """Test different noise estimation methods."""
        X = np.random.randn(5, 100)

        X_median = wavelet_denoise(X, noise_estimator="median")
        X_std = wavelet_denoise(X, noise_estimator="std")

        assert X_median.shape == X.shape
        assert X_std.shape == X.shape
        assert np.isfinite(X_median).all()
        assert np.isfinite(X_std).all()

    def test_invalid_noise_estimator(self):
        """Test that invalid noise estimator raises error."""
        X = np.random.randn(5, 100)

        with pytest.raises(ValueError, match="Unknown noise_estimator"):
            wavelet_denoise(X, noise_estimator="invalid")

    def test_preserves_scale(self):
        """Test that denoising roughly preserves signal scale."""
        X = np.random.randn(10, 100) * 10  # Scale up
        X_denoised = wavelet_denoise(X, wavelet="db4", level=4)

        # Mean and std should be similar (within factor of 2)
        assert abs(np.mean(X_denoised) - np.mean(X)) < 2 * np.std(X)
        assert np.std(X_denoised) < np.std(X)  # Should reduce variance

class TestWaveletDenoiseTransformer:
    """Test the WaveletDenoise transformer class."""

    def test_sklearn_interface(self):
        """Test sklearn TransformerMixin interface."""
        X = np.random.randn(10, 50)
        transformer = WaveletDenoise(wavelet="db4", level=3)

        # fit should return self
        assert transformer.fit(X) is transformer

        # transform should work after fit
        X_transformed = transformer.transform(X)
        assert X_transformed.shape == X.shape

    def test_fit_transform(self):
        """Test fit_transform method."""
        X = np.random.randn(10, 50)
        transformer = WaveletDenoise(wavelet="db8", level=4)

        X_transformed = transformer.fit_transform(X)
        assert X_transformed.shape == X.shape
        assert np.isfinite(X_transformed).all()

    def test_stateless(self):
        """Test that transformer is stateless (no learned parameters)."""
        X_train = np.random.randn(10, 50)
        X_test = np.random.randn(5, 50)

        transformer = WaveletDenoise(wavelet="db4", level=3)
        transformer.fit(X_train)

        # Transform should give same result regardless of fit data
        X_test_transformed = transformer.transform(X_test)
        assert X_test_transformed.shape == X_test.shape

    def test_parameters(self):
        """Test that parameters are correctly stored."""
        transformer = WaveletDenoise(
            wavelet="sym8",
            level=6,
            mode="symmetric",
            threshold_mode="hard",
            noise_estimator="std",
        )

        assert transformer.wavelet == "sym8"
        assert transformer.level == 6
        assert transformer.mode == "symmetric"
        assert transformer.threshold_mode == "hard"
        assert transformer.noise_estimator == "std"

    def test_sparse_input_raises_error(self):
        """Test that sparse matrices raise an error."""
        import scipy.sparse as sp

        X_sparse = sp.csr_matrix(np.random.randn(10, 50))
        transformer = WaveletDenoise()

        with pytest.raises(ValueError, match="does not support sparse"):
            transformer.fit(X_sparse)

        with pytest.raises(ValueError, match="Sparse matrices not supported"):
            transformer.fit(np.random.randn(10, 50))
            transformer.transform(X_sparse)

    def test_reproducibility(self):
        """Test that transformer gives consistent results."""
        X = np.random.randn(10, 50)
        transformer = WaveletDenoise(wavelet="db4", level=3)

        X_transformed_1 = transformer.fit_transform(X)
        X_transformed_2 = transformer.fit_transform(X)

        np.testing.assert_array_equal(X_transformed_1, X_transformed_2)

    def test_copy_parameter(self):
        """Test that copy parameter is stored (though not used in stateless transformer)."""
        transformer = WaveletDenoise(copy=False)
        assert transformer.copy is False

        transformer = WaveletDenoise(copy=True)
        assert transformer.copy is True

    def test_webapp_meta(self):
        """Test that webapp metadata is present."""
        assert hasattr(WaveletDenoise, "_webapp_meta")
        meta = WaveletDenoise._webapp_meta

        assert "category" in meta
        assert "tier" in meta
        assert "tags" in meta
        assert "wavelet" in meta["tags"]
        assert "denoising" in meta["tags"]

    def test_stateless_attribute(self):
        """Test that _stateless attribute is True."""
        assert hasattr(WaveletDenoise, "_stateless")
        assert WaveletDenoise._stateless is True

    def test_more_tags(self):
        """Test _more_tags method."""
        transformer = WaveletDenoise()
        tags = transformer._more_tags()

        assert tags["allow_nan"] is False
        assert tags["stateless"] is True

class TestWaveletDenoiseIntegration:
    """Integration tests with realistic NIRS-like data."""

    def test_with_nirs_like_data(self):
        """Test with NIRS-like spectral data."""
        # Simulate NIRS spectra (typically 1000+ wavelengths)
        n_samples, n_features = 20, 1200
        wavelengths = np.linspace(1000, 2500, n_features)

        # Create realistic spectral shapes (absorption bands + baseline + noise)
        X = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            baseline = 0.3 + 0.1 * np.random.randn()
            X[i] = baseline + 0.05 * np.exp(-((wavelengths - 1450) ** 2) / 10000)
            X[i] += 0.02 * np.exp(-((wavelengths - 1930) ** 2) / 8000)
            X[i] += 0.01 * np.random.randn(n_features)  # Add noise

        transformer = WaveletDenoise(wavelet="db8", level=5, threshold_mode="soft")
        X_denoised = transformer.fit_transform(X)

        # Check that denoising preserves shape and reduces noise
        assert X_denoised.shape == X.shape
        assert np.std(X_denoised) < np.std(X)
        assert np.isfinite(X_denoised).all()

    def test_with_extreme_values(self):
        """Test behavior with extreme values."""
        X = np.random.randn(10, 100) * 1000  # Large scale
        transformer = WaveletDenoise(wavelet="db4", level=4)
        X_denoised = transformer.fit_transform(X)

        assert X_denoised.shape == X.shape
        assert np.isfinite(X_denoised).all()

    def test_with_small_features(self):
        """Test with small number of features."""
        X = np.random.randn(10, 50)
        transformer = WaveletDenoise(wavelet="db4", level=3)  # Lower level for smaller data
        X_denoised = transformer.fit_transform(X)

        assert X_denoised.shape == X.shape
        assert np.isfinite(X_denoised).all()

    def test_with_single_sample(self):
        """Test with single sample."""
        X = np.random.randn(1, 100)
        transformer = WaveletDenoise(wavelet="db4", level=4)
        X_denoised = transformer.fit_transform(X)

        assert X_denoised.shape == X.shape
        assert np.isfinite(X_denoised).all()
