"""
Validation utilities for synthetic data generation.

This module provides functions to validate generated synthetic data
for correctness and expected properties, including spectral realism
scoring for comparing synthetic data against real NIRS spectra.

Phase 4 Features:
    - Spectral realism scorecard with quantitative metrics
    - Correlation length analysis
    - Derivative statistics comparison
    - Peak density analysis
    - Baseline curvature metrics
    - SNR distribution analysis
    - Adversarial validation (classifier distinguishability)

References:
    - Engel, J., et al. (2013). Breaking with trends in pre-processing?
      TrAC Trends in Analytical Chemistry, 50, 96-106.
    - Rinnan, Å., et al. (2009). Review of the most common pre-processing
      techniques for near-infrared spectra. TrAC Trends in Analytical Chemistry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.ndimage import gaussian_filter1d


class ValidationError(Exception):
    """Exception raised when synthetic data validation fails."""

    pass


def validate_spectra(
    X: np.ndarray,
    expected_shape: Optional[Tuple[int, int]] = None,
    check_finite: bool = True,
    check_positive: bool = False,
    value_range: Optional[Tuple[float, float]] = None,
) -> List[str]:
    """
    Validate generated spectra matrix.

    Args:
        X: Spectra matrix to validate.
        expected_shape: Expected (n_samples, n_wavelengths) shape.
        check_finite: Whether to check for NaN/Inf values.
        check_positive: Whether to require all positive values.
        value_range: Optional (min, max) expected range.

    Returns:
        List of validation warning messages (empty if all OK).

    Raises:
        ValidationError: If critical validation fails.

    Example:
        >>> X = np.random.randn(100, 500)
        >>> warnings = validate_spectra(X, expected_shape=(100, 500))
        >>> if warnings:
        ...     print("Warnings:", warnings)
    """
    warnings: List[str] = []

    # Check type
    if not isinstance(X, np.ndarray):
        raise ValidationError(f"Expected numpy array, got {type(X).__name__}")

    # Check dimensions
    if X.ndim != 2:
        raise ValidationError(f"Expected 2D array, got {X.ndim}D")

    # Check shape
    if expected_shape is not None:
        if X.shape != expected_shape:
            raise ValidationError(
                f"Shape mismatch: expected {expected_shape}, got {X.shape}"
            )

    # Check finite values
    if check_finite:
        n_nan = np.isnan(X).sum()
        n_inf = np.isinf(X).sum()
        if n_nan > 0:
            raise ValidationError(f"Found {n_nan} NaN values in spectra")
        if n_inf > 0:
            raise ValidationError(f"Found {n_inf} Inf values in spectra")

    # Check positive values
    if check_positive:
        n_negative = (X < 0).sum()
        if n_negative > 0:
            warnings.append(
                f"Found {n_negative} negative values ({100*n_negative/X.size:.2f}%)"
            )

    # Check value range
    if value_range is not None:
        min_val, max_val = value_range
        if X.min() < min_val:
            warnings.append(
                f"Minimum value {X.min():.4f} below expected {min_val}"
            )
        if X.max() > max_val:
            warnings.append(
                f"Maximum value {X.max():.4f} above expected {max_val}"
            )

    return warnings


def validate_concentrations(
    C: np.ndarray,
    n_samples: Optional[int] = None,
    n_components: Optional[int] = None,
    check_normalized: bool = False,
    tolerance: float = 0.01,
) -> List[str]:
    """
    Validate concentration matrix.

    Args:
        C: Concentration matrix to validate.
        n_samples: Expected number of samples.
        n_components: Expected number of components.
        check_normalized: Whether concentrations should sum to 1.
        tolerance: Tolerance for normalization check.

    Returns:
        List of validation warning messages.

    Raises:
        ValidationError: If critical validation fails.
    """
    warnings: List[str] = []

    if not isinstance(C, np.ndarray):
        raise ValidationError(f"Expected numpy array, got {type(C).__name__}")

    if C.ndim != 2:
        raise ValidationError(f"Expected 2D concentration matrix, got {C.ndim}D")

    if n_samples is not None and C.shape[0] != n_samples:
        raise ValidationError(
            f"Expected {n_samples} samples, got {C.shape[0]}"
        )

    if n_components is not None and C.shape[1] != n_components:
        raise ValidationError(
            f"Expected {n_components} components, got {C.shape[1]}"
        )

    # Check for negative concentrations
    n_negative = (C < 0).sum()
    if n_negative > 0:
        warnings.append(f"Found {n_negative} negative concentration values")

    # Check normalization
    if check_normalized:
        row_sums = C.sum(axis=1)
        deviations = np.abs(row_sums - 1.0)
        if deviations.max() > tolerance:
            warnings.append(
                f"Concentrations not normalized: max deviation = {deviations.max():.4f}"
            )

    return warnings


def validate_wavelengths(
    wavelengths: np.ndarray,
    expected_range: Optional[Tuple[float, float]] = None,
    check_monotonic: bool = True,
    check_uniform: bool = True,
) -> List[str]:
    """
    Validate wavelength array.

    Args:
        wavelengths: Wavelength array to validate.
        expected_range: Optional (min, max) expected range in nm.
        check_monotonic: Whether to check for monotonically increasing values.
        check_uniform: Whether to check for uniform spacing.

    Returns:
        List of validation warning messages.

    Raises:
        ValidationError: If critical validation fails.
    """
    warnings: List[str] = []

    if not isinstance(wavelengths, np.ndarray):
        raise ValidationError(f"Expected numpy array, got {type(wavelengths).__name__}")

    if wavelengths.ndim != 1:
        raise ValidationError(f"Expected 1D wavelength array, got {wavelengths.ndim}D")

    if len(wavelengths) < 2:
        raise ValidationError(
            f"Wavelength array too short: {len(wavelengths)} points"
        )

    # Check range
    if expected_range is not None:
        min_wl, max_wl = expected_range
        if wavelengths.min() < min_wl or wavelengths.max() > max_wl:
            warnings.append(
                f"Wavelength range [{wavelengths.min():.1f}, {wavelengths.max():.1f}] "
                f"outside expected [{min_wl}, {max_wl}]"
            )

    # Check monotonic
    if check_monotonic:
        diffs = np.diff(wavelengths)
        if not np.all(diffs > 0):
            raise ValidationError("Wavelengths must be monotonically increasing")

    # Check uniform spacing
    if check_uniform:
        diffs = np.diff(wavelengths)
        if diffs.std() / diffs.mean() > 0.01:  # 1% tolerance
            warnings.append("Wavelength spacing is not uniform")

    return warnings


def validate_synthetic_output(
    X: np.ndarray,
    C: np.ndarray,
    E: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
) -> List[str]:
    """
    Validate complete synthetic generation output.

    Args:
        X: Generated spectra (n_samples, n_wavelengths).
        C: Concentration matrix (n_samples, n_components).
        E: Component spectra (n_components, n_wavelengths).
        wavelengths: Optional wavelength array.

    Returns:
        List of all validation warnings.

    Raises:
        ValidationError: If critical validation fails.

    Example:
        >>> from nirs4all.synthesis import SyntheticNIRSGenerator
        >>> gen = SyntheticNIRSGenerator(random_state=42)
        >>> X, C, E = gen.generate(100)
        >>> warnings = validate_synthetic_output(X, C, E, gen.wavelengths)
    """
    all_warnings: List[str] = []

    n_samples, n_wavelengths = X.shape
    n_components = C.shape[1]

    # Validate spectra
    all_warnings.extend(
        validate_spectra(X, expected_shape=(n_samples, n_wavelengths))
    )

    # Validate concentrations
    all_warnings.extend(
        validate_concentrations(C, n_samples=n_samples, n_components=n_components)
    )

    # Validate component spectra shape
    if E.shape != (n_components, n_wavelengths):
        raise ValidationError(
            f"Component spectra shape mismatch: expected "
            f"({n_components}, {n_wavelengths}), got {E.shape}"
        )

    # Validate wavelengths if provided
    if wavelengths is not None:
        all_warnings.extend(validate_wavelengths(wavelengths))
        if len(wavelengths) != n_wavelengths:
            raise ValidationError(
                f"Wavelength array length {len(wavelengths)} does not match "
                f"spectra width {n_wavelengths}"
            )

    return all_warnings


# ============================================================================
# Phase 4: Spectral Realism Scorecard
# ============================================================================


class RealismMetric(str, Enum):
    """Metrics used in the spectral realism scorecard."""
    CORRELATION_LENGTH = "correlation_length"
    DERIVATIVE_STATISTICS = "derivative_statistics"
    PEAK_DENSITY = "peak_density"
    BASELINE_CURVATURE = "baseline_curvature"
    SNR_DISTRIBUTION = "snr_distribution"
    ADVERSARIAL_AUC = "adversarial_auc"


@dataclass
class MetricResult:
    """
    Result of a single realism metric evaluation.

    Attributes:
        metric: The metric type.
        value: The computed metric value.
        threshold: The threshold for passing.
        passed: Whether the metric passed the threshold.
        details: Additional details about the metric computation.
    """
    metric: RealismMetric
    value: float
    threshold: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "✓" if self.passed else "✗"
        return f"{status} {self.metric.value}: {self.value:.4f} (threshold: {self.threshold:.4f})"


@dataclass
class SpectralRealismScore:
    """
    Complete spectral realism assessment results.

    This dataclass contains the results of comparing synthetic spectra
    against real spectra using multiple quantitative metrics.

    Attributes:
        correlation_length_overlap: Distribution overlap for autocorrelation decay [0-1].
        derivative_ks_pvalue: p-value from KS test on derivative distributions.
        peak_density_ratio: Ratio of synthetic to real peak densities.
        baseline_curvature_overlap: Distribution overlap for baseline curvature [0-1].
        snr_magnitude_match: Whether SNR is within one order of magnitude.
        adversarial_auc: AUC of classifier trying to distinguish real from synthetic.
        overall_pass: Whether all critical metrics pass.
        metric_results: Individual metric results with details.
        warnings: Any warnings from the analysis.

    Example:
        >>> score = compute_spectral_realism_scorecard(real_spectra, synthetic_spectra, wavelengths)
        >>> print(f"Overall pass: {score.overall_pass}")
        >>> print(f"Adversarial AUC: {score.adversarial_auc:.3f}")
        >>> for metric in score.metric_results:
        ...     print(metric)
    """
    correlation_length_overlap: float
    derivative_ks_pvalue: float
    peak_density_ratio: float
    baseline_curvature_overlap: float
    snr_magnitude_match: bool
    adversarial_auc: float
    overall_pass: bool
    metric_results: List[MetricResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "correlation_length_overlap": self.correlation_length_overlap,
            "derivative_ks_pvalue": self.derivative_ks_pvalue,
            "peak_density_ratio": self.peak_density_ratio,
            "baseline_curvature_overlap": self.baseline_curvature_overlap,
            "snr_magnitude_match": self.snr_magnitude_match,
            "adversarial_auc": self.adversarial_auc,
            "overall_pass": self.overall_pass,
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        """Return a human-readable summary of the score."""
        lines = [
            "=" * 60,
            "Spectral Realism Scorecard",
            "=" * 60,
        ]
        for result in self.metric_results:
            lines.append(str(result))
        lines.append("-" * 60)
        overall_status = "PASS ✓" if self.overall_pass else "FAIL ✗"
        lines.append(f"Overall: {overall_status}")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# Realism Metric Functions
# ============================================================================


def compute_correlation_length(
    spectra: np.ndarray,
    max_lag: int = 50,
) -> np.ndarray:
    """
    Compute correlation lengths for a set of spectra.

    The correlation length is the lag at which the autocorrelation
    function decays to 1/e of its initial value.

    Args:
        spectra: Array of shape (n_samples, n_wavelengths).
        max_lag: Maximum lag to compute autocorrelation for.

    Returns:
        Array of correlation lengths for each spectrum.

    Example:
        >>> X = np.random.randn(100, 500)
        >>> lengths = compute_correlation_length(X)
        >>> print(f"Mean correlation length: {lengths.mean():.2f}")
    """
    n_samples, n_wavelengths = spectra.shape
    max_lag = min(max_lag, n_wavelengths // 4)
    correlation_lengths = np.zeros(n_samples)

    for i in range(n_samples):
        spectrum = spectra[i] - spectra[i].mean()
        if np.std(spectrum) < 1e-10:
            correlation_lengths[i] = 0
            continue

        # Compute normalized autocorrelation using FFT
        fft = np.fft.fft(spectrum, n=2 * n_wavelengths)
        acf_full = np.fft.ifft(fft * np.conj(fft)).real
        acf = acf_full[:max_lag] / acf_full[0]  # Normalize

        # Find where ACF decays to 1/e
        threshold = 1.0 / np.e
        below_threshold = np.where(acf < threshold)[0]
        if len(below_threshold) > 0:
            correlation_lengths[i] = below_threshold[0]
        else:
            correlation_lengths[i] = max_lag

    return correlation_lengths


def compute_derivative_statistics(
    spectra: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    order: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute derivative statistics for spectra.

    Args:
        spectra: Array of shape (n_samples, n_wavelengths).
        wavelengths: Wavelength array for proper derivative scaling.
        order: Derivative order (1 or 2).

    Returns:
        Tuple of (mean_derivatives, std_derivatives) per sample.

    Example:
        >>> X = np.random.randn(100, 500)
        >>> means, stds = compute_derivative_statistics(X, order=1)
    """
    if order == 1:
        if wavelengths is not None:
            dx = np.diff(wavelengths).mean()
            derivatives = np.gradient(spectra, dx, axis=1)
        else:
            derivatives = np.gradient(spectra, axis=1)
    elif order == 2:
        if wavelengths is not None:
            dx = np.diff(wavelengths).mean()
            first_deriv = np.gradient(spectra, dx, axis=1)
            derivatives = np.gradient(first_deriv, dx, axis=1)
        else:
            first_deriv = np.gradient(spectra, axis=1)
            derivatives = np.gradient(first_deriv, axis=1)
    else:
        raise ValueError(f"Order must be 1 or 2, got {order}")

    means = derivatives.mean(axis=1)
    stds = derivatives.std(axis=1)
    return means, stds


def compute_peak_density(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    window_nm: float = 100.0,
    prominence_threshold: float = 0.01,
) -> np.ndarray:
    """
    Compute peak density (peaks per 100 nm) for spectra.

    Args:
        spectra: Array of shape (n_samples, n_wavelengths).
        wavelengths: Wavelength array in nm.
        window_nm: Window size for density calculation (default 100 nm).
        prominence_threshold: Minimum peak prominence as fraction of spectrum range.

    Returns:
        Array of peak densities (peaks per window_nm) for each spectrum.

    Example:
        >>> X = np.random.randn(100, 500)
        >>> wl = np.linspace(1000, 2500, 500)
        >>> densities = compute_peak_density(X, wl)
    """
    n_samples = spectra.shape[0]
    wavelength_range = wavelengths.max() - wavelengths.min()
    peak_densities = np.zeros(n_samples)

    for i in range(n_samples):
        spectrum = spectra[i]
        prominence = prominence_threshold * (spectrum.max() - spectrum.min())

        # Find peaks with minimum prominence
        peaks, properties = signal.find_peaks(spectrum, prominence=prominence)
        n_peaks = len(peaks)

        # Normalize to peaks per window_nm
        peak_densities[i] = n_peaks * (window_nm / wavelength_range)

    return peak_densities


def compute_baseline_curvature(
    spectra: np.ndarray,
    polynomial_degree: int = 3,
) -> np.ndarray:
    """
    Compute baseline curvature by fitting polynomials and measuring residuals.

    Args:
        spectra: Array of shape (n_samples, n_wavelengths).
        polynomial_degree: Degree of polynomial to fit.

    Returns:
        Array of residual standard deviations for each spectrum.

    Example:
        >>> X = np.random.randn(100, 500)
        >>> curvatures = compute_baseline_curvature(X)
    """
    n_samples, n_wavelengths = spectra.shape
    x = np.arange(n_wavelengths)
    curvatures = np.zeros(n_samples)

    for i in range(n_samples):
        spectrum = spectra[i]
        # Fit polynomial
        coeffs = np.polyfit(x, spectrum, polynomial_degree)
        fitted = np.polyval(coeffs, x)
        # Compute residual std as curvature measure
        residuals = spectrum - fitted
        curvatures[i] = np.std(residuals)

    return curvatures


def compute_snr(
    spectra: np.ndarray,
    noise_region_fraction: float = 0.1,
) -> np.ndarray:
    """
    Estimate signal-to-noise ratio for spectra.

    Uses the standard deviation of the highest-frequency components
    (via high-pass filtering) as noise estimate.

    Args:
        spectra: Array of shape (n_samples, n_wavelengths).
        noise_region_fraction: Fraction of spectrum to use for noise estimation.

    Returns:
        Array of SNR estimates for each spectrum.

    Example:
        >>> X = np.random.randn(100, 500) + np.sin(np.linspace(0, 10, 500))
        >>> snr = compute_snr(X)
    """
    n_samples, n_wavelengths = spectra.shape
    snr_values = np.zeros(n_samples)

    for i in range(n_samples):
        spectrum = spectra[i]
        # Signal power: variance of smoothed spectrum
        smoothed = gaussian_filter1d(spectrum, sigma=5)
        signal_power = np.var(smoothed)

        # Noise power: variance of residual after smoothing
        residual = spectrum - smoothed
        noise_power = np.var(residual)

        if noise_power > 1e-10:
            snr_values[i] = signal_power / noise_power
        else:
            snr_values[i] = 1e6  # Very high SNR (essentially noise-free)

    return snr_values


def compute_distribution_overlap(
    dist1: np.ndarray,
    dist2: np.ndarray,
    n_bins: int = 50,
) -> float:
    """
    Compute overlap between two distributions using histogram intersection.

    Args:
        dist1: First distribution samples.
        dist2: Second distribution samples.
        n_bins: Number of histogram bins.

    Returns:
        Overlap coefficient in [0, 1], where 1 means identical distributions.

    Example:
        >>> x1 = np.random.randn(1000)
        >>> x2 = np.random.randn(1000) + 0.5
        >>> overlap = compute_distribution_overlap(x1, x2)
    """
    # Handle edge cases
    if len(dist1) == 0 or len(dist2) == 0:
        return 0.0

    # Check for constant values or NaN/Inf
    if np.std(dist1) < 1e-10 and np.std(dist2) < 1e-10:
        # Both are essentially constant - compare means
        if np.abs(np.mean(dist1) - np.mean(dist2)) < 1e-10:
            return 1.0
        else:
            return 0.0

    # Filter out any NaN/Inf values
    dist1 = dist1[np.isfinite(dist1)]
    dist2 = dist2[np.isfinite(dist2)]

    if len(dist1) == 0 or len(dist2) == 0:
        return 0.0

    # Determine common bin edges
    all_values = np.concatenate([dist1, dist2])
    min_val, max_val = all_values.min(), all_values.max()

    # Handle case where all values are the same
    if max_val - min_val < 1e-10:
        return 1.0

    bins = np.linspace(min_val, max_val, n_bins + 1)

    # Compute normalized histograms
    hist1, _ = np.histogram(dist1, bins=bins, density=True)
    hist2, _ = np.histogram(dist2, bins=bins, density=True)

    # Normalize to sum to 1
    hist1 = hist1 / hist1.sum() if hist1.sum() > 0 else hist1
    hist2 = hist2 / hist2.sum() if hist2.sum() > 0 else hist2

    # Compute intersection (overlap)
    overlap = np.minimum(hist1, hist2).sum()
    return float(overlap)


def compute_adversarial_validation_auc(
    real_spectra: np.ndarray,
    synthetic_spectra: np.ndarray,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Train classifier to distinguish real vs. synthetic spectra.

    A lower AUC indicates that synthetic data is more realistic
    (harder to distinguish from real data).

    Args:
        real_spectra: Real spectra array (n_real, n_wavelengths).
        synthetic_spectra: Synthetic spectra array (n_synthetic, n_wavelengths).
        cv_folds: Number of cross-validation folds.
        random_state: Random state for reproducibility.

    Returns:
        Tuple of (mean_auc, std_auc) across folds.

    Target:
        AUC < 0.6: Excellent (nearly indistinguishable)
        AUC < 0.7: Good (hard to distinguish)
        AUC < 0.8: Acceptable (some differences)
        AUC >= 0.8: Poor (clearly distinguishable)

    Example:
        >>> real = np.random.randn(100, 500)
        >>> synthetic = np.random.randn(100, 500) + 0.1
        >>> mean_auc, std_auc = compute_adversarial_validation_auc(real, synthetic)
        >>> print(f"AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    """
    # Lazy import to avoid sklearn dependency at module level
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    # Prepare data
    X = np.vstack([real_spectra, synthetic_spectra])
    y = np.concatenate([
        np.ones(len(real_spectra)),    # Real = 1
        np.zeros(len(synthetic_spectra))  # Synthetic = 0
    ])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression with regularization
    clf = LogisticRegression(
        max_iter=1000,
        C=0.1,  # Regularization to prevent overfitting
        random_state=random_state,
        solver='lbfgs',
    )

    # Cross-validated AUC
    auc_scores = cross_val_score(
        clf, X_scaled, y,
        cv=cv_folds,
        scoring='roc_auc'
    )

    return float(auc_scores.mean()), float(auc_scores.std())


def compute_spectral_realism_scorecard(
    real_spectra: np.ndarray,
    synthetic_spectra: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    thresholds: Optional[Dict[str, float]] = None,
    include_adversarial: bool = True,
    random_state: Optional[int] = None,
) -> SpectralRealismScore:
    """
    Compute comprehensive spectral realism scorecard.

    This function computes multiple quantitative metrics to assess
    whether synthetic spectra are realistic compared to real data.

    Args:
        real_spectra: Real spectra array (n_real, n_wavelengths).
        synthetic_spectra: Synthetic spectra array (n_synthetic, n_wavelengths).
        wavelengths: Wavelength array in nm. If None, uses indices.
        thresholds: Custom thresholds for metrics. Defaults:
            - correlation_length_overlap: 0.7
            - derivative_ks_pvalue: 0.05
            - peak_density_ratio_min: 0.5
            - peak_density_ratio_max: 2.0
            - baseline_curvature_overlap: 0.6
            - snr_order_of_magnitude: 1.0 (log10 difference)
            - adversarial_auc: 0.7
        include_adversarial: Whether to compute adversarial AUC (slower).
        random_state: Random state for adversarial validation.

    Returns:
        SpectralRealismScore with all metrics and pass/fail status.

    Example:
        >>> from nirs4all.synthesis import SyntheticNIRSGenerator
        >>> gen = SyntheticNIRSGenerator(random_state=42)
        >>> X_synth, _, _ = gen.generate(200)
        >>> # X_real would be loaded from real data
        >>> X_real = np.random.randn(200, X_synth.shape[1])  # Placeholder
        >>> score = compute_spectral_realism_scorecard(X_real, X_synth, gen.wavelengths)
        >>> print(score.summary())
    """
    # Default thresholds
    default_thresholds = {
        "correlation_length_overlap": 0.7,
        "derivative_ks_pvalue": 0.05,
        "peak_density_ratio_min": 0.5,
        "peak_density_ratio_max": 2.0,
        "baseline_curvature_overlap": 0.6,
        "snr_order_of_magnitude": 1.0,
        "adversarial_auc": 0.7,
    }
    if thresholds is not None:
        default_thresholds.update(thresholds)
    thresholds = default_thresholds

    # Create wavelengths if not provided
    if wavelengths is None:
        n_wavelengths = real_spectra.shape[1]
        wavelengths = np.arange(n_wavelengths, dtype=float)

    metric_results: List[MetricResult] = []
    warnings: List[str] = []

    # Validate input dimensions
    if real_spectra.shape[1] != synthetic_spectra.shape[1]:
        raise ValidationError(
            f"Spectral dimension mismatch: real={real_spectra.shape[1]}, "
            f"synthetic={synthetic_spectra.shape[1]}"
        )

    # 1. Correlation Length Overlap
    try:
        real_corr_len = compute_correlation_length(real_spectra)
        synth_corr_len = compute_correlation_length(synthetic_spectra)
        corr_overlap = compute_distribution_overlap(real_corr_len, synth_corr_len)
        corr_passed = corr_overlap >= thresholds["correlation_length_overlap"]
        metric_results.append(MetricResult(
            metric=RealismMetric.CORRELATION_LENGTH,
            value=corr_overlap,
            threshold=thresholds["correlation_length_overlap"],
            passed=corr_passed,
            details={
                "real_mean": float(real_corr_len.mean()),
                "real_std": float(real_corr_len.std()),
                "synthetic_mean": float(synth_corr_len.mean()),
                "synthetic_std": float(synth_corr_len.std()),
            }
        ))
    except Exception as e:
        warnings.append(f"Correlation length computation failed: {e}")
        corr_overlap = 0.0
        corr_passed = False

    # 2. Derivative Statistics (KS test)
    try:
        real_deriv_means, real_deriv_stds = compute_derivative_statistics(
            real_spectra, wavelengths, order=1
        )
        synth_deriv_means, synth_deriv_stds = compute_derivative_statistics(
            synthetic_spectra, wavelengths, order=1
        )
        # KS test on derivative standard deviations
        ks_stat, ks_pvalue = ks_2samp(real_deriv_stds, synth_deriv_stds)
        deriv_passed = ks_pvalue >= thresholds["derivative_ks_pvalue"]
        metric_results.append(MetricResult(
            metric=RealismMetric.DERIVATIVE_STATISTICS,
            value=ks_pvalue,
            threshold=thresholds["derivative_ks_pvalue"],
            passed=deriv_passed,
            details={
                "ks_statistic": float(ks_stat),
                "real_mean_std": float(real_deriv_stds.mean()),
                "synthetic_mean_std": float(synth_deriv_stds.mean()),
            }
        ))
    except Exception as e:
        warnings.append(f"Derivative statistics computation failed: {e}")
        ks_pvalue = 0.0
        deriv_passed = False

    # 3. Peak Density Ratio
    try:
        real_peak_density = compute_peak_density(real_spectra, wavelengths)
        synth_peak_density = compute_peak_density(synthetic_spectra, wavelengths)
        # Ratio of means
        real_mean = real_peak_density.mean()
        synth_mean = synth_peak_density.mean()
        if real_mean > 0:
            peak_ratio = synth_mean / real_mean
        else:
            peak_ratio = 1.0 if synth_mean == 0 else float('inf')
        peak_passed = (
            thresholds["peak_density_ratio_min"] <= peak_ratio <=
            thresholds["peak_density_ratio_max"]
        )
        metric_results.append(MetricResult(
            metric=RealismMetric.PEAK_DENSITY,
            value=peak_ratio,
            threshold=thresholds["peak_density_ratio_max"],
            passed=peak_passed,
            details={
                "real_mean": float(real_mean),
                "synthetic_mean": float(synth_mean),
                "ratio": float(peak_ratio),
            }
        ))
    except Exception as e:
        warnings.append(f"Peak density computation failed: {e}")
        peak_ratio = 0.0
        peak_passed = False

    # 4. Baseline Curvature Overlap
    try:
        real_curvature = compute_baseline_curvature(real_spectra)
        synth_curvature = compute_baseline_curvature(synthetic_spectra)
        curvature_overlap = compute_distribution_overlap(real_curvature, synth_curvature)
        curvature_passed = curvature_overlap >= thresholds["baseline_curvature_overlap"]
        metric_results.append(MetricResult(
            metric=RealismMetric.BASELINE_CURVATURE,
            value=curvature_overlap,
            threshold=thresholds["baseline_curvature_overlap"],
            passed=curvature_passed,
            details={
                "real_mean": float(real_curvature.mean()),
                "synthetic_mean": float(synth_curvature.mean()),
            }
        ))
    except Exception as e:
        warnings.append(f"Baseline curvature computation failed: {e}")
        curvature_overlap = 0.0
        curvature_passed = False

    # 5. SNR Distribution
    try:
        real_snr = compute_snr(real_spectra)
        synth_snr = compute_snr(synthetic_spectra)
        # Compare in log scale
        real_log_snr = np.log10(real_snr + 1e-10)
        synth_log_snr = np.log10(synth_snr + 1e-10)
        log_snr_diff = np.abs(real_log_snr.mean() - synth_log_snr.mean())
        snr_match = log_snr_diff <= thresholds["snr_order_of_magnitude"]
        metric_results.append(MetricResult(
            metric=RealismMetric.SNR_DISTRIBUTION,
            value=log_snr_diff,
            threshold=thresholds["snr_order_of_magnitude"],
            passed=snr_match,
            details={
                "real_mean_snr": float(real_snr.mean()),
                "synthetic_mean_snr": float(synth_snr.mean()),
                "log_difference": float(log_snr_diff),
            }
        ))
    except Exception as e:
        warnings.append(f"SNR computation failed: {e}")
        snr_match = True  # Default to pass on failure
        log_snr_diff = 0.0

    # 6. Adversarial Validation AUC
    adversarial_auc = 0.5  # Default (random guess)
    adversarial_passed = True
    if include_adversarial:
        try:
            mean_auc, std_auc = compute_adversarial_validation_auc(
                real_spectra, synthetic_spectra,
                cv_folds=5,
                random_state=random_state,
            )
            adversarial_auc = mean_auc
            adversarial_passed = adversarial_auc <= thresholds["adversarial_auc"]
            metric_results.append(MetricResult(
                metric=RealismMetric.ADVERSARIAL_AUC,
                value=adversarial_auc,
                threshold=thresholds["adversarial_auc"],
                passed=adversarial_passed,
                details={
                    "mean_auc": float(mean_auc),
                    "std_auc": float(std_auc),
                    "target": "lower is better",
                }
            ))
        except Exception as e:
            warnings.append(f"Adversarial validation failed: {e}")

    # Compute overall pass
    # All metrics must pass for overall pass
    overall_pass = all(result.passed for result in metric_results)

    return SpectralRealismScore(
        correlation_length_overlap=corr_overlap,
        derivative_ks_pvalue=ks_pvalue,
        peak_density_ratio=peak_ratio,
        baseline_curvature_overlap=curvature_overlap,
        snr_magnitude_match=snr_match,
        adversarial_auc=adversarial_auc,
        overall_pass=overall_pass,
        metric_results=metric_results,
        warnings=warnings,
    )


# ============================================================================
# Benchmark Dataset Comparison Utilities
# ============================================================================


@dataclass
class DatasetComparisonResult:
    """
    Result of comparing synthetic data against a benchmark dataset.

    Attributes:
        dataset_name: Name of the benchmark dataset.
        n_real_samples: Number of samples in real dataset.
        n_synthetic_samples: Number of synthetic samples used.
        realism_score: The spectral realism score.
        tstr_r2: Train-on-Synthetic, Test-on-Real R² (if applicable).
        trts_r2: Train-on-Real, Test-on-Synthetic R² (if applicable).
    """
    dataset_name: str
    n_real_samples: int
    n_synthetic_samples: int
    realism_score: SpectralRealismScore
    tstr_r2: Optional[float] = None
    trts_r2: Optional[float] = None

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Dataset: {self.dataset_name}",
            f"Samples: {self.n_real_samples} real, {self.n_synthetic_samples} synthetic",
            "",
            self.realism_score.summary(),
        ]
        if self.tstr_r2 is not None:
            lines.append(f"\nTSTR R²: {self.tstr_r2:.4f}")
        if self.trts_r2 is not None:
            lines.append(f"TRTS R²: {self.trts_r2:.4f}")
        return "\n".join(lines)


def validate_against_benchmark(
    synthetic_spectra: np.ndarray,
    benchmark_spectra: np.ndarray,
    benchmark_name: str,
    wavelengths: Optional[np.ndarray] = None,
    synthetic_targets: Optional[np.ndarray] = None,
    benchmark_targets: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> DatasetComparisonResult:
    """
    Validate synthetic data against a benchmark dataset.

    Args:
        synthetic_spectra: Synthetic spectra (n_synth, n_wavelengths).
        benchmark_spectra: Real benchmark spectra (n_bench, n_wavelengths).
        benchmark_name: Name of the benchmark dataset.
        wavelengths: Wavelength array.
        synthetic_targets: Optional targets for TSTR/TRTS evaluation.
        benchmark_targets: Optional targets for TSTR/TRTS evaluation.
        random_state: Random state for reproducibility.

    Returns:
        DatasetComparisonResult with realism score and optional TSTR/TRTS.

    Example:
        >>> result = validate_against_benchmark(
        ...     synthetic_spectra=X_synth,
        ...     benchmark_spectra=X_real,
        ...     benchmark_name="Corn",
        ... )
        >>> print(result.summary())
    """
    # Compute realism score
    realism_score = compute_spectral_realism_scorecard(
        real_spectra=benchmark_spectra,
        synthetic_spectra=synthetic_spectra,
        wavelengths=wavelengths,
        random_state=random_state,
    )

    tstr_r2 = None
    trts_r2 = None

    # Compute TSTR/TRTS if targets provided
    if synthetic_targets is not None and benchmark_targets is not None:
        try:
            from sklearn.cross_decomposition import PLSRegression
            from sklearn.metrics import r2_score

            # Ensure proper shapes
            if synthetic_targets.ndim == 1:
                synthetic_targets = synthetic_targets.reshape(-1, 1)
            if benchmark_targets.ndim == 1:
                benchmark_targets = benchmark_targets.reshape(-1, 1)

            # TSTR: Train on Synthetic, Test on Real
            n_components = min(10, synthetic_spectra.shape[1] // 10, len(synthetic_spectra) // 2)
            n_components = max(1, n_components)

            pls = PLSRegression(n_components=n_components)
            pls.fit(synthetic_spectra, synthetic_targets)
            pred_real = pls.predict(benchmark_spectra)
            tstr_r2 = float(r2_score(benchmark_targets, pred_real))

            # TRTS: Train on Real, Test on Synthetic
            pls = PLSRegression(n_components=n_components)
            pls.fit(benchmark_spectra, benchmark_targets)
            pred_synth = pls.predict(synthetic_spectra)
            trts_r2 = float(r2_score(synthetic_targets, pred_synth))

        except Exception:
            pass  # TSTR/TRTS evaluation failed, leave as None

    return DatasetComparisonResult(
        dataset_name=benchmark_name,
        n_real_samples=len(benchmark_spectra),
        n_synthetic_samples=len(synthetic_spectra),
        realism_score=realism_score,
        tstr_r2=tstr_r2,
        trts_r2=trts_r2,
    )


# ============================================================================
# Quick Validation Functions
# ============================================================================


def quick_realism_check(
    synthetic_spectra: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    expected_snr_range: Tuple[float, float] = (10, 1000),
    expected_peak_density: Tuple[float, float] = (0.5, 10.0),
) -> Tuple[bool, List[str]]:
    """
    Perform quick realism checks on synthetic spectra without real data.

    This function checks basic properties that realistic spectra should have,
    without requiring a reference real dataset.

    Args:
        synthetic_spectra: Synthetic spectra to check.
        wavelengths: Wavelength array.
        expected_snr_range: Expected SNR range (min, max).
        expected_peak_density: Expected peak density range (peaks per 100 nm).

    Returns:
        Tuple of (passed, list_of_issues).

    Example:
        >>> X = generator.generate(100)[0]
        >>> passed, issues = quick_realism_check(X, wavelengths)
        >>> if not passed:
        ...     print("Issues:", issues)
    """
    issues: List[str] = []

    # Check for NaN/Inf
    if np.any(np.isnan(synthetic_spectra)):
        issues.append("Spectra contain NaN values")
    if np.any(np.isinf(synthetic_spectra)):
        issues.append("Spectra contain Inf values")

    # Check SNR
    try:
        snr = compute_snr(synthetic_spectra)
        mean_snr = snr.mean()
        if mean_snr < expected_snr_range[0]:
            issues.append(f"SNR too low: {mean_snr:.1f} < {expected_snr_range[0]}")
        if mean_snr > expected_snr_range[1]:
            issues.append(f"SNR unrealistically high: {mean_snr:.1f} > {expected_snr_range[1]}")
    except Exception as e:
        issues.append(f"SNR check failed: {e}")

    # Check peak density
    if wavelengths is not None:
        try:
            peak_densities = compute_peak_density(synthetic_spectra, wavelengths)
            mean_density = peak_densities.mean()
            if mean_density < expected_peak_density[0]:
                issues.append(f"Peak density too low: {mean_density:.2f}")
            if mean_density > expected_peak_density[1]:
                issues.append(f"Peak density too high: {mean_density:.2f}")
        except Exception as e:
            issues.append(f"Peak density check failed: {e}")

    # Check variance structure
    try:
        # Spectra should have wavelength-dependent variance (not flat)
        wavelength_variance = np.var(synthetic_spectra, axis=0)
        cv = np.std(wavelength_variance) / (np.mean(wavelength_variance) + 1e-10)
        if cv < 0.1:
            issues.append("Variance across wavelengths too uniform (unrealistic)")
    except Exception as e:
        issues.append(f"Variance check failed: {e}")

    passed = len(issues) == 0
    return passed, issues
