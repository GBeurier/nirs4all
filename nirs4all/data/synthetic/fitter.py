"""
Real data fitting utilities for synthetic NIRS spectra generation.

This module provides tools to analyze real NIRS datasets and fit generator
parameters to match their statistical and spectral properties.

Key Features:
    - Statistical property analysis (mean, std, skewness, kurtosis)
    - Spectral shape analysis (slope, curvature, noise)
    - PCA structure analysis
    - Parameter estimation for SyntheticNIRSGenerator
    - Comparison between synthetic and real data

Example:
    >>> from nirs4all.data.synthetic import RealDataFitter, SyntheticNIRSGenerator
    >>>
    >>> # Analyze real data
    >>> fitter = RealDataFitter()
    >>> params = fitter.fit(X_real, wavelengths=wavelengths)
    >>>
    >>> # Create generator with fitted parameters
    >>> generator = SyntheticNIRSGenerator(**params.to_generator_kwargs())
    >>> X_synthetic, _, _ = generator.generate(n_samples=1000)

References:
    - Based on comparator.py from bench/synthetic/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.signal import savgol_filter, find_peaks

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset


@dataclass
class SpectralProperties:
    """
    Container for computed spectral properties of a dataset.

    This dataclass holds various statistical and spectral properties
    computed from a NIRS dataset for comparison and fitting purposes.

    Attributes:
        name: Dataset identifier.
        n_samples: Number of samples.
        n_wavelengths: Number of wavelengths.
        wavelengths: Wavelength grid.

        # Basic statistics
        mean_spectrum: Mean spectrum across samples.
        std_spectrum: Standard deviation spectrum.
        global_mean: Overall mean absorbance.
        global_std: Overall standard deviation.
        global_range: (min, max) absorbance range.

        # Shape properties
        mean_slope: Average spectral slope (per 1000nm).
        slope_std: Standard deviation of slopes.
        mean_curvature: Average curvature (second derivative).

        # Distribution statistics
        skewness: Skewness of absorbance distribution.
        kurtosis: Kurtosis of absorbance distribution.

        # Noise characteristics
        noise_estimate: Estimated noise level.
        snr_estimate: Signal-to-noise ratio estimate.

        # PCA properties
        pca_explained_variance: Explained variance ratios.
        pca_n_components_95: Components for 95% variance.
    """

    name: str = "dataset"
    n_samples: int = 0
    n_wavelengths: int = 0
    wavelengths: Optional[np.ndarray] = None

    # Basic statistics
    mean_spectrum: Optional[np.ndarray] = None
    std_spectrum: Optional[np.ndarray] = None
    global_mean: float = 0.0
    global_std: float = 0.0
    global_range: Tuple[float, float] = (0.0, 0.0)

    # Shape properties
    mean_slope: float = 0.0
    slope_std: float = 0.0
    slopes: Optional[np.ndarray] = None
    mean_curvature: float = 0.0
    curvature_std: float = 0.0

    # Distribution statistics
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Noise characteristics
    noise_estimate: float = 0.0
    snr_estimate: float = 0.0

    # PCA properties
    pca_explained_variance: Optional[np.ndarray] = None
    pca_n_components_95: int = 0

    # Peak analysis
    n_peaks_mean: float = 0.0
    peak_positions: Optional[np.ndarray] = None


@dataclass
class FittedParameters:
    """
    Parameters fitted from real data for synthetic generation.

    This dataclass contains all parameters needed to configure
    a SyntheticNIRSGenerator to produce spectra similar to a
    real dataset.

    Attributes:
        wavelength_start: Start wavelength (nm).
        wavelength_end: End wavelength (nm).
        wavelength_step: Wavelength step (nm).
        global_slope_mean: Mean global slope.
        global_slope_std: Slope standard deviation.
        noise_base: Base noise level.
        noise_signal_dep: Signal-dependent noise factor.
        path_length_std: Path length variation.
        baseline_amplitude: Baseline drift amplitude.
        scatter_alpha_std: Multiplicative scatter std.
        scatter_beta_std: Additive scatter std.
        complexity: Suggested complexity level.
        source_name: Name of source dataset.
        source_properties: Full SpectralProperties of source.
    """

    # Wavelength grid
    wavelength_start: float = 1000.0
    wavelength_end: float = 2500.0
    wavelength_step: float = 2.0

    # Slope parameters
    global_slope_mean: float = 0.0
    global_slope_std: float = 0.02

    # Noise parameters
    noise_base: float = 0.001
    noise_signal_dep: float = 0.005

    # Variation parameters
    path_length_std: float = 0.05
    baseline_amplitude: float = 0.02
    scatter_alpha_std: float = 0.05
    scatter_beta_std: float = 0.01
    tilt_std: float = 0.01

    # Metadata
    complexity: str = "realistic"
    source_name: str = ""
    source_properties: Optional[SpectralProperties] = field(default=None, repr=False)

    def to_generator_kwargs(self) -> Dict[str, Any]:
        """
        Convert fitted parameters to kwargs for SyntheticNIRSGenerator.

        Returns:
            Dictionary of keyword arguments.

        Example:
            >>> params = fitter.fit(X_real)
            >>> generator = SyntheticNIRSGenerator(**params.to_generator_kwargs())
        """
        return {
            "wavelength_start": self.wavelength_start,
            "wavelength_end": self.wavelength_end,
            "wavelength_step": self.wavelength_step,
            "complexity": self.complexity,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all parameters to a dictionary.

        Returns:
            Dictionary with all parameter values.
        """
        return {
            "wavelength_start": self.wavelength_start,
            "wavelength_end": self.wavelength_end,
            "wavelength_step": self.wavelength_step,
            "global_slope_mean": self.global_slope_mean,
            "global_slope_std": self.global_slope_std,
            "noise_base": self.noise_base,
            "noise_signal_dep": self.noise_signal_dep,
            "path_length_std": self.path_length_std,
            "baseline_amplitude": self.baseline_amplitude,
            "scatter_alpha_std": self.scatter_alpha_std,
            "scatter_beta_std": self.scatter_beta_std,
            "tilt_std": self.tilt_std,
            "complexity": self.complexity,
            "source_name": self.source_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FittedParameters":
        """
        Create FittedParameters from a dictionary.

        Args:
            data: Dictionary with parameter values.

        Returns:
            FittedParameters instance.
        """
        return cls(
            wavelength_start=data.get("wavelength_start", 1000.0),
            wavelength_end=data.get("wavelength_end", 2500.0),
            wavelength_step=data.get("wavelength_step", 2.0),
            global_slope_mean=data.get("global_slope_mean", 0.0),
            global_slope_std=data.get("global_slope_std", 0.02),
            noise_base=data.get("noise_base", 0.001),
            noise_signal_dep=data.get("noise_signal_dep", 0.005),
            path_length_std=data.get("path_length_std", 0.05),
            baseline_amplitude=data.get("baseline_amplitude", 0.02),
            scatter_alpha_std=data.get("scatter_alpha_std", 0.05),
            scatter_beta_std=data.get("scatter_beta_std", 0.01),
            tilt_std=data.get("tilt_std", 0.01),
            complexity=data.get("complexity", "realistic"),
            source_name=data.get("source_name", ""),
        )

    def save(self, path: str) -> None:
        """
        Save parameters to JSON file.

        Args:
            path: Output file path.
        """
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FittedParameters":
        """
        Load parameters from JSON file.

        Args:
            path: Input file path.

        Returns:
            FittedParameters instance.
        """
        import json

        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def compute_spectral_properties(
    X: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    name: str = "dataset",
    n_pca_components: int = 20,
) -> SpectralProperties:
    """
    Compute comprehensive spectral properties of a dataset.

    Analyzes a matrix of spectra to extract statistical and spectral
    properties useful for fitting and comparison.

    Args:
        X: Spectra matrix (n_samples, n_wavelengths).
        wavelengths: Optional wavelength grid.
        name: Dataset identifier.
        n_pca_components: Maximum PCA components to compute.

    Returns:
        SpectralProperties with computed metrics.

    Example:
        >>> props = compute_spectral_properties(X_real, wavelengths)
        >>> print(f"Mean slope: {props.mean_slope:.4f}")
    """
    n_samples, n_wavelengths = X.shape

    if wavelengths is None:
        wavelengths = np.arange(n_wavelengths)

    props = SpectralProperties(
        name=name,
        n_samples=n_samples,
        n_wavelengths=n_wavelengths,
        wavelengths=wavelengths.copy(),
    )

    # Basic statistics
    props.mean_spectrum = X.mean(axis=0)
    props.std_spectrum = X.std(axis=0)
    props.global_mean = float(X.mean())
    props.global_std = float(X.std())
    props.global_range = (float(X.min()), float(X.max()))

    # Slope analysis
    wl_range = np.ptp(wavelengths)
    if wl_range > 0:
        x_norm = (wavelengths - wavelengths.min()) / wl_range
        slopes = []
        for i in range(n_samples):
            coeffs = np.polyfit(x_norm, X[i], 1)
            # Convert to slope per 1000nm
            slopes.append(coeffs[0] * 1000.0 / wl_range)
        props.slopes = np.array(slopes)
        props.mean_slope = float(np.mean(slopes))
        props.slope_std = float(np.std(slopes))

    # Curvature analysis
    window_size = min(21, n_wavelengths // 10 * 2 + 1)
    if window_size >= 5:
        curvatures = []
        for i in range(min(n_samples, 100)):  # Sample subset for speed
            try:
                smoothed = savgol_filter(X[i], window_size, 2)
                d2 = np.gradient(np.gradient(smoothed))
                curvatures.append(np.mean(np.abs(d2)))
            except Exception:
                pass
        if curvatures:
            props.mean_curvature = float(np.mean(curvatures))
            props.curvature_std = float(np.std(curvatures))

    # Distribution statistics
    flat_data = X.flatten()
    props.skewness = float(stats.skew(flat_data))
    props.kurtosis = float(stats.kurtosis(flat_data))

    # Noise estimation (from first difference)
    first_diff = np.diff(X, axis=1)
    props.noise_estimate = float(first_diff.std() / np.sqrt(2))

    # SNR estimation
    signal_power = props.std_spectrum.mean()
    if props.noise_estimate > 0:
        props.snr_estimate = float(signal_power / props.noise_estimate)
    else:
        props.snr_estimate = float("inf")

    # PCA analysis
    try:
        from sklearn.decomposition import PCA

        n_comp = min(n_pca_components, n_samples, n_wavelengths)
        pca = PCA(n_components=n_comp)
        pca.fit(X)
        props.pca_explained_variance = pca.explained_variance_ratio_

        # Components for 95% variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        props.pca_n_components_95 = int(np.searchsorted(cumsum, 0.95) + 1)
    except ImportError:
        pass

    # Peak analysis
    try:
        window_size = min(21, n_wavelengths // 10 * 2 + 1)
        if window_size >= 5 and props.std_spectrum is not None:
            smoothed_mean = savgol_filter(props.mean_spectrum, window_size, 2)
            prominence = props.std_spectrum.mean() * 0.5
            peaks, _ = find_peaks(smoothed_mean, prominence=prominence)
            props.peak_positions = wavelengths[peaks] if len(peaks) > 0 else np.array([])
            props.n_peaks_mean = float(len(peaks))
    except Exception:
        props.peak_positions = np.array([])
        props.n_peaks_mean = 0.0

    return props


class RealDataFitter:
    """
    Fit generator parameters to match real dataset properties.

    This class analyzes real NIRS data and estimates parameters for
    the SyntheticNIRSGenerator to produce similar spectra.

    Attributes:
        source_properties: SpectralProperties of the analyzed data.
        fitted_params: FittedParameters after fitting.

    Example:
        >>> fitter = RealDataFitter()
        >>> params = fitter.fit(X_real, wavelengths=wavelengths)
        >>>
        >>> # Use fitted parameters
        >>> generator = SyntheticNIRSGenerator(**params.to_generator_kwargs())
        >>> X_synth, _, _ = generator.generate(1000)
        >>>
        >>> # Compare
        >>> similarity = fitter.evaluate_similarity(X_synth)
    """

    def __init__(self) -> None:
        """Initialize the fitter."""
        self.source_properties: Optional[SpectralProperties] = None
        self.fitted_params: Optional[FittedParameters] = None

    def fit(
        self,
        X: Union[np.ndarray, "SpectroDataset"],
        *,
        wavelengths: Optional[np.ndarray] = None,
        name: str = "source",
    ) -> FittedParameters:
        """
        Fit generator parameters to real data.

        Analyzes the input data and estimates optimal parameters for
        generating synthetic spectra with similar properties.

        Args:
            X: Real spectra matrix (n_samples, n_wavelengths) or SpectroDataset.
            wavelengths: Wavelength grid (required if X is ndarray).
            name: Dataset name for reference.

        Returns:
            FittedParameters object with estimated parameters.

        Raises:
            ValueError: If X is empty or has wrong shape.

        Example:
            >>> fitter = RealDataFitter()
            >>> params = fitter.fit(X_real, wavelengths=wl, name="wheat")
        """
        # Handle SpectroDataset input
        if hasattr(X, "x") and callable(X.x):
            # It's a SpectroDataset
            X_array = X.x({}, layout="2d")
            if wavelengths is None:
                try:
                    wavelengths = X.wavelengths
                except (AttributeError, TypeError):
                    wavelengths = np.arange(X_array.shape[1])
            if hasattr(X, "name"):
                name = X.name or name
        else:
            X_array = np.asarray(X)

        # Validate input
        if X_array.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_array.shape}")
        if X_array.shape[0] < 5:
            raise ValueError(f"Need at least 5 samples, got {X_array.shape[0]}")

        n_samples, n_wavelengths = X_array.shape

        # Create default wavelengths if not provided
        if wavelengths is None:
            wavelengths = np.arange(n_wavelengths)
        wavelengths = np.asarray(wavelengths)

        # Compute spectral properties
        self.source_properties = compute_spectral_properties(
            X_array, wavelengths, name
        )

        # Estimate parameters
        params = FittedParameters(
            source_name=name,
            source_properties=self.source_properties,
        )

        # Wavelength grid
        params.wavelength_start = float(wavelengths.min())
        params.wavelength_end = float(wavelengths.max())
        if len(wavelengths) > 1:
            params.wavelength_step = float(np.median(np.diff(wavelengths)))

        # Slope parameters
        props = self.source_properties
        params.global_slope_mean = props.mean_slope
        params.global_slope_std = props.slope_std

        # Noise parameters
        # Base noise is estimated from first difference
        params.noise_base = props.noise_estimate * 0.5
        # Signal-dependent noise from variation
        params.noise_signal_dep = props.noise_estimate * 0.5 / max(props.global_std, 0.01)

        # Scatter parameters (estimated from sample variation)
        # Multiplicative scatter from overall std
        params.scatter_alpha_std = min(0.15, props.global_std / max(props.global_mean, 0.1) * 0.3)
        # Additive scatter
        params.scatter_beta_std = props.global_std * 0.1

        # Path length variation (from sample-to-sample intensity variation)
        intensity_variation = np.std(X_array.mean(axis=1)) / max(np.mean(X_array.mean(axis=1)), 0.1)
        params.path_length_std = min(0.2, intensity_variation * 0.5)

        # Baseline amplitude (from low-frequency variation)
        params.baseline_amplitude = props.global_std * 0.2

        # Tilt standard deviation
        params.tilt_std = abs(props.mean_slope) * 0.1

        # Determine complexity
        if props.snr_estimate > 50 and props.pca_n_components_95 <= 5:
            params.complexity = "simple"
        elif props.snr_estimate < 20 or props.pca_n_components_95 > 15:
            params.complexity = "complex"
        else:
            params.complexity = "realistic"

        self.fitted_params = params
        return params

    def fit_from_path(
        self,
        path: str,
        *,
        name: Optional[str] = None,
    ) -> FittedParameters:
        """
        Fit parameters from a dataset path.

        Loads data using DatasetConfigs and fits parameters.

        Args:
            path: Path to dataset folder.
            name: Optional name override.

        Returns:
            FittedParameters object.

        Example:
            >>> params = fitter.fit_from_path("sample_data/regression")
        """
        from nirs4all.data import DatasetConfigs

        dataset_config = DatasetConfigs(path)
        datasets = dataset_config.get_datasets()

        if not datasets:
            raise ValueError(f"No datasets found at {path}")

        dataset = datasets[0]
        X = dataset.x({}, layout="2d")

        # Try to get wavelengths
        wavelengths = None
        try:
            wavelengths = dataset.wavelengths
        except (AttributeError, TypeError):
            pass

        return self.fit(X, wavelengths=wavelengths, name=name or dataset.name)

    def evaluate_similarity(
        self,
        X_synthetic: np.ndarray,
        wavelengths: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate similarity between synthetic and source data.

        Computes various metrics comparing synthetic spectra to the
        original real data.

        Args:
            X_synthetic: Synthetic spectra matrix.
            wavelengths: Optional wavelength grid.

        Returns:
            Dictionary with similarity metrics.

        Raises:
            RuntimeError: If fit() hasn't been called.

        Example:
            >>> params = fitter.fit(X_real)
            >>> X_synth, _, _ = generator.generate(1000)
            >>> metrics = fitter.evaluate_similarity(X_synth)
            >>> print(f"Similarity: {metrics['overall_score']:.1f}/100")
        """
        if self.source_properties is None:
            raise RuntimeError("Must call fit() before evaluate_similarity()")

        # Use source wavelengths if not provided
        if wavelengths is None and self.source_properties.wavelengths is not None:
            # Assume same wavelength grid
            wavelengths = self.source_properties.wavelengths

        # Compute synthetic properties
        synth_props = compute_spectral_properties(
            X_synthetic, wavelengths, "synthetic"
        )

        real_props = self.source_properties
        metrics: Dict[str, Any] = {}

        # Mean comparison
        if real_props.global_mean != 0:
            metrics["mean_rel_diff"] = (
                (synth_props.global_mean - real_props.global_mean)
                / abs(real_props.global_mean)
            )
        else:
            metrics["mean_rel_diff"] = synth_props.global_mean

        # Std comparison
        if real_props.global_std != 0:
            metrics["std_rel_diff"] = (
                (synth_props.global_std - real_props.global_std)
                / real_props.global_std
            )
        else:
            metrics["std_rel_diff"] = synth_props.global_std

        # Slope comparison
        metrics["slope_diff"] = synth_props.mean_slope - real_props.mean_slope
        if real_props.mean_slope != 0:
            metrics["slope_ratio"] = synth_props.mean_slope / real_props.mean_slope
        else:
            metrics["slope_ratio"] = float("inf")

        # Noise comparison
        if real_props.noise_estimate != 0:
            metrics["noise_ratio"] = synth_props.noise_estimate / real_props.noise_estimate
        else:
            metrics["noise_ratio"] = float("inf")

        # SNR comparison
        if real_props.snr_estimate != 0 and real_props.snr_estimate != float("inf"):
            metrics["snr_ratio"] = synth_props.snr_estimate / real_props.snr_estimate
        else:
            metrics["snr_ratio"] = float("inf")

        # PCA complexity
        metrics["pca_complexity_diff"] = (
            synth_props.pca_n_components_95 - real_props.pca_n_components_95
        )

        # Mean spectrum correlation (if wavelengths match)
        if (real_props.n_wavelengths == synth_props.n_wavelengths and
            real_props.mean_spectrum is not None and
            synth_props.mean_spectrum is not None):
            corr = np.corrcoef(
                real_props.mean_spectrum, synth_props.mean_spectrum
            )[0, 1]
            metrics["mean_spectrum_correlation"] = float(corr)

        # Slope distribution comparison
        if real_props.slopes is not None and synth_props.slopes is not None:
            ks_stat, ks_pval = stats.ks_2samp(real_props.slopes, synth_props.slopes)
            metrics["slope_ks_statistic"] = float(ks_stat)
            metrics["slope_ks_pvalue"] = float(ks_pval)

        # Overall similarity score (0-100)
        scores = []
        if "mean_rel_diff" in metrics:
            scores.append(max(0, 100 - abs(metrics["mean_rel_diff"]) * 100))
        if "std_rel_diff" in metrics:
            scores.append(max(0, 100 - abs(metrics["std_rel_diff"]) * 100))
        if "noise_ratio" in metrics and metrics["noise_ratio"] != float("inf"):
            scores.append(max(0, 100 - abs(1 - metrics["noise_ratio"]) * 100))
        if "mean_spectrum_correlation" in metrics:
            scores.append(metrics["mean_spectrum_correlation"] * 100)

        metrics["overall_score"] = float(np.mean(scores)) if scores else 0.0

        return metrics

    def get_tuning_recommendations(self) -> List[str]:
        """
        Get recommendations for tuning generation parameters.

        Based on the fitted parameters and source data, provides
        suggestions for manual tuning.

        Returns:
            List of recommendation strings.

        Example:
            >>> params = fitter.fit(X_real)
            >>> for rec in fitter.get_tuning_recommendations():
            ...     print(f"- {rec}")
        """
        if self.source_properties is None or self.fitted_params is None:
            return ["Call fit() first to analyze data."]

        recs = []
        props = self.source_properties
        params = self.fitted_params

        # Noise recommendations
        if props.snr_estimate < 15:
            recs.append(
                f"High noise detected (SNR={props.snr_estimate:.1f}). "
                f"Using noise_base={params.noise_base:.4f}"
            )
        elif props.snr_estimate > 100:
            recs.append(
                f"Very low noise detected (SNR={props.snr_estimate:.1f}). "
                "Consider using 'simple' complexity for faster generation."
            )

        # Slope recommendations
        if abs(props.mean_slope) > 0.1:
            recs.append(
                f"Significant slope detected ({props.mean_slope:.3f}/1000nm). "
                "Ensure global_slope_mean is correctly set."
            )

        # Complexity recommendations
        if props.pca_n_components_95 > 10:
            recs.append(
                f"High complexity ({props.pca_n_components_95} PCA components for 95%). "
                "Consider using more spectral components."
            )
        elif props.pca_n_components_95 <= 3:
            recs.append(
                f"Low complexity ({props.pca_n_components_95} PCA components). "
                "Simple mode may be sufficient."
            )

        # Variation recommendations
        if params.path_length_std > 0.15:
            recs.append(
                f"High sample-to-sample variation detected. "
                f"path_length_std set to {params.path_length_std:.3f}"
            )

        return recs


def fit_to_real_data(
    X: Union[np.ndarray, "SpectroDataset"],
    wavelengths: Optional[np.ndarray] = None,
    name: str = "source",
) -> FittedParameters:
    """
    Quick function to fit parameters to real data.

    Convenience function for simple fitting use cases.

    Args:
        X: Real spectra or SpectroDataset.
        wavelengths: Wavelength grid.
        name: Dataset name.

    Returns:
        FittedParameters object.

    Example:
        >>> params = fit_to_real_data(X_real, wavelengths)
        >>> generator = SyntheticNIRSGenerator(**params.to_generator_kwargs())
    """
    fitter = RealDataFitter()
    return fitter.fit(X, wavelengths=wavelengths, name=name)


def compare_datasets(
    X_synthetic: np.ndarray,
    X_real: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Quick comparison between synthetic and real datasets.

    Args:
        X_synthetic: Synthetic spectra.
        X_real: Real spectra.
        wavelengths: Wavelength grid.

    Returns:
        Dictionary with comparison metrics.

    Example:
        >>> metrics = compare_datasets(X_synth, X_real)
        >>> print(f"Similarity: {metrics['overall_score']:.1f}/100")
    """
    fitter = RealDataFitter()
    fitter.fit(X_real, wavelengths=wavelengths, name="real")
    return fitter.evaluate_similarity(X_synthetic, wavelengths)
