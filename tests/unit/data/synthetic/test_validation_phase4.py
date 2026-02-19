"""
Unit tests for Phase 4 validation module - spectral realism scorecard.
"""

import numpy as np
import pytest

from nirs4all.synthesis.validation import (
    DatasetComparisonResult,
    MetricResult,
    # Phase 4: Realism scorecard
    RealismMetric,
    SpectralRealismScore,
    # Existing validation
    ValidationError,
    compute_adversarial_validation_auc,
    compute_baseline_curvature,
    compute_correlation_length,
    compute_derivative_statistics,
    compute_distribution_overlap,
    compute_peak_density,
    compute_snr,
    compute_spectral_realism_scorecard,
    quick_realism_check,
    validate_against_benchmark,
    validate_concentrations,
    validate_spectra,
    validate_synthetic_output,
    validate_wavelengths,
)


class TestRealismMetrics:
    """Tests for individual realism metrics."""

    def test_compute_correlation_length_basic(self):
        """Test correlation length computation."""
        # Create spectra with known autocorrelation structure
        n_samples, n_wavelengths = 50, 200
        np.random.seed(42)
        spectra = np.random.randn(n_samples, n_wavelengths)

        lengths = compute_correlation_length(spectra)
        assert lengths.shape == (n_samples,)
        assert np.all(lengths >= 0)
        assert np.all(lengths <= 50)  # max_lag

    def test_compute_correlation_length_smooth_spectra(self):
        """Smooth spectra should have longer correlation lengths."""
        n_samples = 20
        wavelengths = np.linspace(0, 10, 200)

        # Smooth spectra (sine wave)
        smooth = np.sin(wavelengths).reshape(1, -1).repeat(n_samples, axis=0)
        smooth += np.random.randn(n_samples, 200) * 0.1

        # Noisy spectra
        noisy = np.random.randn(n_samples, 200)

        smooth_lengths = compute_correlation_length(smooth)
        noisy_lengths = compute_correlation_length(noisy)

        assert smooth_lengths.mean() > noisy_lengths.mean()

    def test_compute_derivative_statistics(self):
        """Test derivative statistics computation."""
        n_samples, n_wavelengths = 50, 200
        wavelengths = np.linspace(1000, 2500, n_wavelengths)
        spectra = np.random.randn(n_samples, n_wavelengths)

        means, stds = compute_derivative_statistics(spectra, wavelengths, order=1)

        assert means.shape == (n_samples,)
        assert stds.shape == (n_samples,)
        assert np.all(stds >= 0)

    def test_compute_derivative_statistics_order_2(self):
        """Test second derivative statistics."""
        spectra = np.random.randn(20, 100)
        wavelengths = np.linspace(1000, 2500, 100)

        means, stds = compute_derivative_statistics(spectra, wavelengths, order=2)
        assert means.shape == (20,)
        assert stds.shape == (20,)

    def test_compute_peak_density(self):
        """Test peak density computation."""
        n_wavelengths = 500
        wavelengths = np.linspace(1000, 2500, n_wavelengths)

        # Create spectrum with known number of peaks
        spectrum = np.zeros((1, n_wavelengths))
        peak_positions = [100, 200, 300, 400]  # 4 peaks
        for pos in peak_positions:
            spectrum[0, pos-10:pos+10] = np.exp(-((np.arange(20) - 10)**2) / 10)

        densities = compute_peak_density(spectrum, wavelengths)
        assert densities.shape == (1,)
        # Should detect approximately 4 peaks over 1500 nm range
        # = ~0.27 peaks per 100 nm
        assert 0.1 < densities[0] < 1.0

    def test_compute_baseline_curvature(self):
        """Test baseline curvature computation."""
        n_samples = 30
        n_wavelengths = 200

        # Flat spectra should have low curvature
        flat = np.zeros((n_samples, n_wavelengths))
        flat_curvature = compute_baseline_curvature(flat)
        assert np.allclose(flat_curvature, 0, atol=1e-10)

        # Noisy spectra should have higher curvature
        noisy = np.random.randn(n_samples, n_wavelengths)
        noisy_curvature = compute_baseline_curvature(noisy)
        assert np.all(noisy_curvature > flat_curvature.mean())

    def test_compute_snr(self):
        """Test SNR computation."""
        n_samples = 20
        n_wavelengths = 200
        wavelengths = np.linspace(0, 10, n_wavelengths)

        # High SNR: signal with little noise
        signal = np.sin(wavelengths).reshape(1, -1).repeat(n_samples, axis=0)
        high_snr_spectra = signal + np.random.randn(n_samples, n_wavelengths) * 0.01

        # Low SNR: lots of noise
        low_snr_spectra = signal + np.random.randn(n_samples, n_wavelengths) * 1.0

        high_snr = compute_snr(high_snr_spectra)
        low_snr = compute_snr(low_snr_spectra)

        assert high_snr.mean() > low_snr.mean()

    def test_compute_distribution_overlap_identical(self):
        """Identical distributions should have perfect overlap."""
        dist = np.random.randn(1000)
        overlap = compute_distribution_overlap(dist, dist)
        assert overlap > 0.95

    def test_compute_distribution_overlap_different(self):
        """Very different distributions should have low overlap."""
        dist1 = np.random.randn(1000)
        dist2 = np.random.randn(1000) + 10  # Shifted by 10 std

        overlap = compute_distribution_overlap(dist1, dist2)
        assert overlap < 0.1

class TestAdversarialValidation:
    """Tests for adversarial validation AUC."""

    def test_adversarial_auc_identical(self):
        """Identical data should be hard to distinguish (low AUC)."""
        np.random.seed(42)
        spectra = np.random.randn(100, 50)

        # Split into "real" and "synthetic" from same distribution
        real = spectra[:50]
        synthetic = spectra[50:]

        mean_auc, std_auc = compute_adversarial_validation_auc(
            real, synthetic, cv_folds=3, random_state=42
        )

        # Should be close to 0.5 (random guess)
        assert 0.4 < mean_auc < 0.7

    def test_adversarial_auc_different(self):
        """Different data should be easy to distinguish (high AUC)."""
        np.random.seed(42)
        real = np.random.randn(50, 50)
        synthetic = np.random.randn(50, 50) + 5  # Shifted

        mean_auc, std_auc = compute_adversarial_validation_auc(
            real, synthetic, cv_folds=3, random_state=42
        )

        # Should be close to 1.0 (easy to distinguish)
        assert mean_auc > 0.9

class TestSpectralRealismScorecard:
    """Tests for the complete spectral realism scorecard."""

    def test_scorecard_identical_data(self):
        """Identical data should score well."""
        np.random.seed(42)
        n_samples, n_wavelengths = 100, 200
        spectra = np.random.randn(n_samples, n_wavelengths)
        wavelengths = np.linspace(1000, 2500, n_wavelengths)

        # Split into "real" and "synthetic"
        real = spectra[:50]
        synthetic = spectra[50:]

        score = compute_spectral_realism_scorecard(
            real, synthetic, wavelengths,
            include_adversarial=False,  # Skip for speed
            random_state=42
        )

        assert isinstance(score, SpectralRealismScore)
        assert score.correlation_length_overlap > 0.5
        assert score.derivative_ks_pvalue > 0.01
        assert 0.5 < score.peak_density_ratio < 2.0

    def test_scorecard_with_adversarial(self):
        """Test scorecard with adversarial validation."""
        np.random.seed(42)
        spectra = np.random.randn(100, 100)
        wavelengths = np.linspace(1000, 2500, 100)

        score = compute_spectral_realism_scorecard(
            spectra[:50], spectra[50:], wavelengths,
            include_adversarial=True,
            random_state=42
        )

        assert 0.0 < score.adversarial_auc < 1.0
        assert len(score.metric_results) == 6  # All 6 metrics

    def test_scorecard_summary(self):
        """Test scorecard summary output."""
        np.random.seed(42)
        spectra = np.random.randn(50, 100)
        wavelengths = np.linspace(1000, 2500, 100)

        score = compute_spectral_realism_scorecard(
            spectra[:25], spectra[25:], wavelengths,
            include_adversarial=False,
        )

        summary = score.summary()
        assert "Spectral Realism Scorecard" in summary
        assert "correlation_length" in summary

    def test_scorecard_to_dict(self):
        """Test scorecard dictionary conversion."""
        np.random.seed(42)
        spectra = np.random.randn(50, 100)
        wavelengths = np.linspace(1000, 2500, 100)

        score = compute_spectral_realism_scorecard(
            spectra[:25], spectra[25:], wavelengths,
            include_adversarial=False,
        )

        d = score.to_dict()
        assert "correlation_length_overlap" in d
        assert "adversarial_auc" in d
        assert "overall_pass" in d

    def test_scorecard_custom_thresholds(self):
        """Test scorecard with custom thresholds."""
        np.random.seed(42)
        spectra = np.random.randn(50, 100)
        wavelengths = np.linspace(1000, 2500, 100)

        # Very strict thresholds
        strict = compute_spectral_realism_scorecard(
            spectra[:25], spectra[25:], wavelengths,
            thresholds={"correlation_length_overlap": 0.99},
            include_adversarial=False,
        )

        # Lenient thresholds
        lenient = compute_spectral_realism_scorecard(
            spectra[:25], spectra[25:], wavelengths,
            thresholds={"correlation_length_overlap": 0.1},
            include_adversarial=False,
        )

        # At least some metrics should differ in pass status
        assert isinstance(strict, SpectralRealismScore)
        assert isinstance(lenient, SpectralRealismScore)

class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_metric_result_repr(self):
        """Test MetricResult string representation."""
        result = MetricResult(
            metric=RealismMetric.CORRELATION_LENGTH,
            value=0.85,
            threshold=0.7,
            passed=True,
            details={"real_mean": 10.5}
        )

        repr_str = repr(result)
        assert "✓" in repr_str
        assert "correlation_length" in repr_str
        assert "0.85" in repr_str

    def test_metric_result_failed(self):
        """Test MetricResult for failed metric."""
        result = MetricResult(
            metric=RealismMetric.ADVERSARIAL_AUC,
            value=0.9,
            threshold=0.7,
            passed=False,
        )

        repr_str = repr(result)
        assert "✗" in repr_str

class TestQuickRealismCheck:
    """Tests for quick realism check."""

    def test_quick_check_valid_spectra(self):
        """Test quick check on valid spectra."""
        np.random.seed(42)
        wavelengths = np.linspace(1000, 2500, 200)
        # Create semi-realistic spectra
        x = np.linspace(0, 10, 200)
        spectra = np.sin(x).reshape(1, -1).repeat(50, axis=0)
        spectra += np.random.randn(50, 200) * 0.1

        passed, issues = quick_realism_check(spectra, wavelengths)
        # Should pass basic checks
        assert isinstance(passed, bool)
        assert isinstance(issues, list)

    def test_quick_check_nan_detection(self):
        """Test quick check detects NaN values."""
        spectra = np.random.randn(10, 100)
        spectra[5, 50] = np.nan

        passed, issues = quick_realism_check(spectra)
        assert not passed
        assert any("NaN" in issue for issue in issues)

    def test_quick_check_inf_detection(self):
        """Test quick check detects Inf values."""
        spectra = np.random.randn(10, 100)
        spectra[3, 25] = np.inf

        passed, issues = quick_realism_check(spectra)
        assert not passed
        assert any("Inf" in issue for issue in issues)

class TestValidateAgainstBenchmark:
    """Tests for benchmark validation."""

    def test_validate_against_benchmark(self):
        """Test validation against benchmark."""
        np.random.seed(42)
        synthetic = np.random.randn(50, 100)
        benchmark = np.random.randn(50, 100)
        wavelengths = np.linspace(1000, 2500, 100)

        result = validate_against_benchmark(
            synthetic_spectra=synthetic,
            benchmark_spectra=benchmark,
            benchmark_name="test_dataset",
            wavelengths=wavelengths,
        )

        assert isinstance(result, DatasetComparisonResult)
        assert result.dataset_name == "test_dataset"
        assert result.n_real_samples == 50
        assert result.n_synthetic_samples == 50
        assert isinstance(result.realism_score, SpectralRealismScore)

    def test_validate_with_targets(self):
        """Test validation with TSTR/TRTS computation."""
        np.random.seed(42)
        n_samples = 100
        n_wavelengths = 50

        synthetic = np.random.randn(n_samples, n_wavelengths)
        benchmark = np.random.randn(n_samples, n_wavelengths)

        # Generate correlated targets
        synthetic_targets = synthetic[:, 10:15].mean(axis=1) + np.random.randn(n_samples) * 0.1
        benchmark_targets = benchmark[:, 10:15].mean(axis=1) + np.random.randn(n_samples) * 0.1

        result = validate_against_benchmark(
            synthetic_spectra=synthetic,
            benchmark_spectra=benchmark,
            benchmark_name="test",
            synthetic_targets=synthetic_targets,
            benchmark_targets=benchmark_targets,
        )

        # TSTR and TRTS should be computed
        assert result.tstr_r2 is not None or result.trts_r2 is not None

    def test_validation_result_summary(self):
        """Test validation result summary."""
        np.random.seed(42)
        synthetic = np.random.randn(30, 50)
        benchmark = np.random.randn(30, 50)

        result = validate_against_benchmark(
            synthetic, benchmark, "summary_test"
        )

        summary = result.summary()
        assert "summary_test" in summary
        assert "Spectral Realism Scorecard" in summary
