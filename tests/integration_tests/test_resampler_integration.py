"""
Unit tests for the Resampler operator and ResamplerController.
"""

import pytest
import numpy as np
from nirs4all.operators.transforms import Resampler
from nirs4all.controllers.data.op_resampler import ResamplerController


class TestResampler:
    """Test the Resampler operator."""

    def test_basic_resampling(self):
        """Test basic linear resampling."""
        # Create synthetic data
        original_wl = np.linspace(1000, 2500, 200)
        target_wl = np.linspace(1000, 2500, 100)

        X = np.random.randn(50, 200)

        resampler = Resampler(target_wavelengths=target_wl, method='linear')
        resampler.fit(X, wavelengths=original_wl)
        X_resampled = resampler.transform(X)

        assert X_resampled.shape == (50, 100)
        assert hasattr(resampler, 'original_wavelengths_')
        assert resampler.n_features_out_ == 100

    def test_cubic_interpolation(self):
        """Test cubic spline interpolation."""
        original_wl = np.linspace(1000, 2500, 200)
        target_wl = np.linspace(1000, 2500, 150)

        X = np.random.randn(30, 200)

        resampler = Resampler(target_wavelengths=target_wl, method='cubic')
        resampler.fit(X, wavelengths=original_wl)
        X_resampled = resampler.transform(X)

        assert X_resampled.shape == (30, 150)

    def test_upsampling(self):
        """Test upsampling (interpolating to more points)."""
        original_wl = np.linspace(1000, 2500, 100)
        target_wl = np.linspace(1000, 2500, 300)

        X = np.random.randn(20, 100)

        resampler = Resampler(target_wavelengths=target_wl, method='linear')
        resampler.fit(X, wavelengths=original_wl)
        X_resampled = resampler.transform(X)

        assert X_resampled.shape == (20, 300)

    def test_wavelength_validation(self):
        """Test wavelength validation."""
        target_wl = np.linspace(1000, 2500, 100)

        # Non-monotonic wavelengths should raise error
        bad_wl = np.array([1000, 1500, 1200, 2000])
        X = np.random.randn(10, 4)

        resampler = Resampler(target_wavelengths=target_wl)
        with pytest.raises(ValueError, match="strictly monotonic"):
            resampler.fit(X, wavelengths=bad_wl)

    def test_no_overlap_error(self):
        """Test error when wavelengths don't overlap."""
        original_wl = np.linspace(1000, 2000, 100)
        target_wl = np.linspace(3000, 4000, 100)  # No overlap

        X = np.random.randn(10, 100)

        resampler = Resampler(target_wavelengths=target_wl)
        with pytest.raises(ValueError, match="No overlap"):
            resampler.fit(X, wavelengths=original_wl)

    def test_extrapolation_warning(self):
        """Test warning when target extends beyond original range."""
        original_wl = np.linspace(1000, 2000, 100)
        target_wl = np.linspace(900, 2100, 100)  # Extends beyond

        X = np.random.randn(10, 100)

        resampler = Resampler(target_wavelengths=target_wl, fill_value=0.0)

        with pytest.warns(UserWarning, match="extend"):
            resampler.fit(X, wavelengths=original_wl)

    def test_crop_range(self):
        """Test cropping before resampling."""
        original_wl = np.linspace(1000, 2500, 200)
        target_wl = np.linspace(1200, 2200, 100)

        X = np.random.randn(20, 200)

        resampler = Resampler(
            target_wavelengths=target_wl,
            crop_range=(1100, 2300)
        )
        resampler.fit(X, wavelengths=original_wl)
        X_resampled = resampler.transform(X)

        assert X_resampled.shape == (20, 100)

    def test_missing_wavelengths_error(self):
        """Test error when wavelengths not provided."""
        target_wl = np.linspace(1000, 2500, 100)
        X = np.random.randn(10, 200)

        resampler = Resampler(target_wavelengths=target_wl)

        with pytest.raises(ValueError, match="Wavelengths must be provided"):
            resampler.fit(X)  # No wavelengths provided

    def test_shape_mismatch_error(self):
        """Test error when wavelengths count doesn't match features."""
        original_wl = np.linspace(1000, 2500, 100)  # 100 wavelengths
        target_wl = np.linspace(1000, 2500, 50)
        X = np.random.randn(10, 200)  # 200 features - mismatch!

        resampler = Resampler(target_wavelengths=target_wl)

        with pytest.raises(ValueError, match="must match number of features"):
            resampler.fit(X, wavelengths=original_wl)


class TestResamplerController:
    """Test the ResamplerController."""

    def test_controller_matches_resampler(self):
        """Test that controller matches Resampler objects."""
        target_wl = np.linspace(1000, 2500, 100)
        resampler = Resampler(target_wavelengths=target_wl)

        assert ResamplerController.matches(
            step=resampler,
            operator=resampler,
            keyword=""
        )

    def test_controller_matches_dict_step(self):
        """Test that controller matches Resampler in dict step."""
        target_wl = np.linspace(1000, 2500, 100)
        resampler = Resampler(target_wavelengths=target_wl)

        step = {'model': resampler}

        assert ResamplerController.matches(
            step=step,
            operator=resampler,
            keyword=""
        )

    def test_controller_multi_source_support(self):
        """Test that controller supports multi-source."""
        assert ResamplerController.use_multi_source() is True

    def test_controller_prediction_mode_support(self):
        """Test that controller supports prediction mode."""
        assert ResamplerController.supports_prediction_mode() is True


class TestResamplerIntegration:
    """Integration tests with synthetic datasets."""

    def test_resampler_preserves_spectral_features(self):
        """Test that resampling preserves main spectral features."""
        # Create synthetic spectrum with known peaks
        original_wl = np.linspace(1000, 2500, 200)

        # Create spectrum with Gaussian peaks at specific wavelengths
        spectrum = np.zeros(200)
        peak_positions = [1200, 1500, 1900, 2200]
        for peak_wl in peak_positions:
            idx = np.argmin(np.abs(original_wl - peak_wl))
            spectrum += np.exp(-0.5 * ((np.arange(200) - idx) / 5) ** 2)

        X = np.tile(spectrum, (10, 1))  # 10 identical spectra

        # Resample to different resolution
        target_wl = np.linspace(1000, 2500, 100)
        resampler = Resampler(target_wavelengths=target_wl, method='cubic')
        resampler.fit(X, wavelengths=original_wl)
        X_resampled = resampler.transform(X)

        # Check that peak positions are approximately preserved
        # Find peak in resampled data
        peak_idx = np.argmax(X_resampled[0])
        peak_wl_resampled = target_wl[peak_idx]

        # Should be close to one of the original peaks
        assert any(abs(peak_wl_resampled - p) < 50 for p in peak_positions)

    def test_resampler_consistency(self):
        """Test that resampling is consistent across samples."""
        original_wl = np.linspace(1000, 2500, 200)
        target_wl = np.linspace(1000, 2500, 100)

        # Create data where all samples are identical
        base_spectrum = np.random.randn(200)
        X = np.tile(base_spectrum, (20, 1))

        resampler = Resampler(target_wavelengths=target_wl, method='linear')
        resampler.fit(X, wavelengths=original_wl)
        X_resampled = resampler.transform(X)

        # All resampled spectra should be identical
        for i in range(1, 20):
            np.testing.assert_allclose(X_resampled[0], X_resampled[i], rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
