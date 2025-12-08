"""
Comparison tools for synthetic vs real NIRS spectra.

This module provides tools to compare statistical and spectral properties
of synthetic spectra against real datasets to assess realism and tune
generation parameters.

Key Features:
- Statistical property comparison (mean, std, skewness, kurtosis)
- Spectral shape analysis (global slope, curvature, peak detection)
- PCA structure comparison (explained variance, loading similarity)
- Noise characterization comparison
- Comprehensive visual comparison plots
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm


@dataclass
class SpectralProperties:
    """
    Container for computed spectral properties of a dataset.

    Attributes:
        name: Dataset name
        n_samples: Number of samples
        n_wavelengths: Number of wavelengths
        wavelengths: Wavelength grid (if available)

        # Basic statistics
        mean_spectrum: Mean spectrum across samples
        std_spectrum: Std spectrum across samples
        min_spectrum: Minimum values per wavelength
        max_spectrum: Maximum values per wavelength

        # Global properties
        global_mean: Overall mean absorbance
        global_std: Overall std absorbance
        global_range: Overall min-max range

        # Slope analysis
        mean_slope: Average slope (absorbance change per 1000nm)
        slope_std: Std of slopes across samples
        slopes: Individual sample slopes

        # Curvature
        mean_curvature: Average second derivative (curvature)
        curvature_std: Std of curvatures

        # Distribution statistics
        skewness: Skewness of absorbance distribution
        kurtosis: Kurtosis of absorbance distribution

        # Noise characteristics
        noise_estimate: Estimated noise level (from first difference)
        snr_estimate: Signal-to-noise ratio estimate

        # PCA properties
        pca_explained_variance: Explained variance ratios
        pca_n_components_95: Components needed for 95% variance

        # Peak analysis
        n_peaks_mean: Average number of peaks per spectrum
        peak_positions: Common peak positions
    """
    name: str
    n_samples: int
    n_wavelengths: int
    wavelengths: Optional[np.ndarray] = None

    # Basic statistics
    mean_spectrum: Optional[np.ndarray] = None
    std_spectrum: Optional[np.ndarray] = None
    min_spectrum: Optional[np.ndarray] = None
    max_spectrum: Optional[np.ndarray] = None

    # Global properties
    global_mean: float = 0.0
    global_std: float = 0.0
    global_range: Tuple[float, float] = (0.0, 0.0)

    # Slope analysis
    mean_slope: float = 0.0
    slope_std: float = 0.0
    slopes: Optional[np.ndarray] = None

    # Curvature
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


def compute_spectral_properties(
    X: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    name: str = "dataset",
    n_pca_components: int = 20,
) -> SpectralProperties:
    """
    Compute comprehensive spectral properties of a dataset.

    Args:
        X: Spectra matrix (n_samples, n_wavelengths)
        wavelengths: Optional wavelength grid
        name: Dataset name
        n_pca_components: Number of PCA components to compute

    Returns:
        SpectralProperties object with computed metrics
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
    props.min_spectrum = X.min(axis=0)
    props.max_spectrum = X.max(axis=0)

    # Global properties
    props.global_mean = X.mean()
    props.global_std = X.std()
    props.global_range = (X.min(), X.max())

    # Slope analysis (linear fit per sample)
    wl_range = np.ptp(wavelengths)
    if wl_range > 0:
        x_norm = (wavelengths - wavelengths.min()) / wl_range
        slopes = []
        for i in range(n_samples):
            # Simple linear regression
            coeffs = np.polyfit(x_norm, X[i], 1)
            # Convert to slope per 1000nm
            slopes.append(coeffs[0] * 1000.0 / wl_range)
        props.slopes = np.array(slopes)
        props.mean_slope = np.mean(slopes)
        props.slope_std = np.std(slopes)

    # Curvature analysis (second derivative)
    curvatures = []
    for i in range(n_samples):
        # Smooth and compute second derivative
        smoothed = savgol_filter(X[i], min(21, n_wavelengths // 10 * 2 + 1), 2)
        d2 = np.gradient(np.gradient(smoothed))
        curvatures.append(np.mean(np.abs(d2)))
    props.mean_curvature = np.mean(curvatures)
    props.curvature_std = np.std(curvatures)

    # Distribution statistics
    flat_data = X.flatten()
    props.skewness = stats.skew(flat_data)
    props.kurtosis = stats.kurtosis(flat_data)

    # Noise estimation (from first difference)
    first_diff = np.diff(X, axis=1)
    props.noise_estimate = first_diff.std() / np.sqrt(2)

    # SNR estimation
    signal_power = props.std_spectrum.mean()
    if props.noise_estimate > 0:
        props.snr_estimate = signal_power / props.noise_estimate
    else:
        props.snr_estimate = np.inf

    # PCA analysis
    try:
        from sklearn.decomposition import PCA
        n_comp = min(n_pca_components, n_samples, n_wavelengths)
        pca = PCA(n_components=n_comp)
        pca.fit(X)
        props.pca_explained_variance = pca.explained_variance_ratio_

        # Find components for 95% variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        props.pca_n_components_95 = int(np.searchsorted(cumsum, 0.95) + 1)
    except ImportError:
        pass

    # Peak analysis on mean spectrum
    try:
        # Smooth the mean spectrum
        smoothed_mean = savgol_filter(props.mean_spectrum, min(21, n_wavelengths // 10 * 2 + 1), 2)
        peaks, _ = find_peaks(smoothed_mean, prominence=props.std_spectrum.mean() * 0.5)
        props.peak_positions = wavelengths[peaks] if len(peaks) > 0 else np.array([])

        # Count peaks per sample
        n_peaks_list = []
        for i in range(min(n_samples, 100)):  # Sample up to 100
            smoothed = savgol_filter(X[i], min(21, n_wavelengths // 10 * 2 + 1), 2)
            peaks_i, _ = find_peaks(smoothed, prominence=props.std_spectrum.mean() * 0.3)
            n_peaks_list.append(len(peaks_i))
        props.n_peaks_mean = np.mean(n_peaks_list)
    except Exception:
        props.n_peaks_mean = 0
        props.peak_positions = np.array([])

    return props


class SyntheticRealComparator:
    """
    Compare synthetic spectra with real datasets to assess realism.

    This class computes and compares various spectral properties between
    synthetic and real datasets, providing metrics and visualizations
    to help tune synthetic generation parameters.

    Example:
        >>> from examples.synthetic import SyntheticNIRSGenerator, SyntheticRealComparator
        >>> from nirs4all.data import DatasetConfigs
        >>>
        >>> # Load real data
        >>> dataset = DatasetConfigs('sample_data/regression').get_datasets()[0]
        >>> X_real = dataset.x({}, layout='2d')
        >>>
        >>> # Generate synthetic data
        >>> generator = SyntheticNIRSGenerator(random_state=42)
        >>> X_synth, Y_synth, E = generator.generate(n_samples=500)
        >>>
        >>> # Compare
        >>> comparator = SyntheticRealComparator()
        >>> comparator.add_real_dataset(X_real, name="real_wheat")
        >>> comparator.add_synthetic_dataset(X_synth, generator.wavelengths, name="synthetic")
        >>> comparator.compute_comparison()
        >>> comparator.plot_comparison()
    """

    def __init__(self):
        """Initialize the comparator."""
        self.real_datasets: Dict[str, SpectralProperties] = {}
        self.synthetic_datasets: Dict[str, SpectralProperties] = {}
        self.comparison_results: Optional[Dict[str, Any]] = None

    def add_real_dataset(
        self,
        X: np.ndarray,
        wavelengths: Optional[np.ndarray] = None,
        name: str = "real",
    ) -> 'SyntheticRealComparator':
        """
        Add a real dataset for comparison.

        Args:
            X: Spectra matrix (n_samples, n_wavelengths)
            wavelengths: Optional wavelength grid
            name: Dataset name

        Returns:
            Self for chaining
        """
        props = compute_spectral_properties(X, wavelengths, name)
        self.real_datasets[name] = props
        return self

    def add_synthetic_dataset(
        self,
        X: np.ndarray,
        wavelengths: Optional[np.ndarray] = None,
        name: str = "synthetic",
    ) -> 'SyntheticRealComparator':
        """
        Add a synthetic dataset for comparison.

        Args:
            X: Spectra matrix (n_samples, n_wavelengths)
            wavelengths: Optional wavelength grid
            name: Dataset name

        Returns:
            Self for chaining
        """
        props = compute_spectral_properties(X, wavelengths, name)
        self.synthetic_datasets[name] = props
        return self

    def compute_comparison(self) -> Dict[str, Any]:
        """
        Compute comparison metrics between all real and synthetic datasets.

        Returns:
            Dictionary with comparison results
        """
        results = {
            "real": {},
            "synthetic": {},
            "comparison": {},
        }

        # Store properties
        for name, props in self.real_datasets.items():
            results["real"][name] = self._props_to_dict(props)

        for name, props in self.synthetic_datasets.items():
            results["synthetic"][name] = self._props_to_dict(props)

        # Compute pairwise comparisons
        for real_name, real_props in self.real_datasets.items():
            for synth_name, synth_props in self.synthetic_datasets.items():
                key = f"{real_name}_vs_{synth_name}"
                results["comparison"][key] = self._compare_pair(real_props, synth_props)

        self.comparison_results = results
        return results

    def _props_to_dict(self, props: SpectralProperties) -> Dict[str, Any]:
        """Convert SpectralProperties to a summary dictionary."""
        return {
            "n_samples": props.n_samples,
            "n_wavelengths": props.n_wavelengths,
            "global_mean": props.global_mean,
            "global_std": props.global_std,
            "global_range": props.global_range,
            "mean_slope": props.mean_slope,
            "slope_std": props.slope_std,
            "mean_curvature": props.mean_curvature,
            "skewness": props.skewness,
            "kurtosis": props.kurtosis,
            "noise_estimate": props.noise_estimate,
            "snr_estimate": props.snr_estimate,
            "pca_n_components_95": props.pca_n_components_95,
            "n_peaks_mean": props.n_peaks_mean,
        }

    def _compare_pair(
        self,
        real: SpectralProperties,
        synth: SpectralProperties,
    ) -> Dict[str, Any]:
        """Compare a pair of real and synthetic datasets."""
        comparison = {}

        # Relative differences
        if real.global_mean != 0:
            comparison["mean_rel_diff"] = (synth.global_mean - real.global_mean) / abs(real.global_mean)
        else:
            comparison["mean_rel_diff"] = synth.global_mean

        if real.global_std != 0:
            comparison["std_rel_diff"] = (synth.global_std - real.global_std) / real.global_std
        else:
            comparison["std_rel_diff"] = synth.global_std

        # Slope comparison
        comparison["slope_diff"] = synth.mean_slope - real.mean_slope
        comparison["slope_ratio"] = synth.mean_slope / real.mean_slope if real.mean_slope != 0 else np.inf

        # Noise comparison
        if real.noise_estimate != 0:
            comparison["noise_ratio"] = synth.noise_estimate / real.noise_estimate
        else:
            comparison["noise_ratio"] = np.inf

        # SNR comparison
        if real.snr_estimate != 0 and real.snr_estimate != np.inf:
            comparison["snr_ratio"] = synth.snr_estimate / real.snr_estimate
        else:
            comparison["snr_ratio"] = np.inf

        # PCA complexity comparison
        comparison["pca_complexity_diff"] = synth.pca_n_components_95 - real.pca_n_components_95

        # Mean spectrum correlation (if wavelengths match)
        if (real.n_wavelengths == synth.n_wavelengths and
            real.mean_spectrum is not None and
            synth.mean_spectrum is not None):
            corr = np.corrcoef(real.mean_spectrum, synth.mean_spectrum)[0, 1]
            comparison["mean_spectrum_correlation"] = corr

        # Slope distribution overlap (Kolmogorov-Smirnov test)
        if real.slopes is not None and synth.slopes is not None:
            ks_stat, ks_pval = stats.ks_2samp(real.slopes, synth.slopes)
            comparison["slope_ks_statistic"] = ks_stat
            comparison["slope_ks_pvalue"] = ks_pval

        # Overall similarity score (0-100)
        scores = []
        if "mean_rel_diff" in comparison:
            scores.append(max(0, 100 - abs(comparison["mean_rel_diff"]) * 100))
        if "std_rel_diff" in comparison:
            scores.append(max(0, 100 - abs(comparison["std_rel_diff"]) * 100))
        if "noise_ratio" in comparison and comparison["noise_ratio"] != np.inf:
            scores.append(max(0, 100 - abs(1 - comparison["noise_ratio"]) * 100))
        if "mean_spectrum_correlation" in comparison:
            scores.append(comparison["mean_spectrum_correlation"] * 100)

        comparison["similarity_score"] = np.mean(scores) if scores else 0

        return comparison

    def print_summary(self) -> None:
        """Print a summary of the comparison results."""
        if self.comparison_results is None:
            self.compute_comparison()

        print("=" * 80)
        print("SYNTHETIC vs REAL SPECTRA COMPARISON")
        print("=" * 80)

        # Print real dataset properties
        print("\n📊 REAL DATASETS:")
        print("-" * 40)
        for name, props in self.comparison_results["real"].items():
            print(f"\n  {name}:")
            print(f"    Samples: {props['n_samples']}, Wavelengths: {props['n_wavelengths']}")
            print(f"    Mean: {props['global_mean']:.4f} ± {props['global_std']:.4f}")
            print(f"    Range: [{props['global_range'][0]:.3f}, {props['global_range'][1]:.3f}]")
            print(f"    Slope: {props['mean_slope']:.4f} ± {props['slope_std']:.4f} (per 1000nm)")
            print(f"    Noise: {props['noise_estimate']:.5f}, SNR: {props['snr_estimate']:.1f}")
            print(f"    PCA 95%: {props['pca_n_components_95']} components")

        # Print synthetic dataset properties
        print("\n🔬 SYNTHETIC DATASETS:")
        print("-" * 40)
        for name, props in self.comparison_results["synthetic"].items():
            print(f"\n  {name}:")
            print(f"    Samples: {props['n_samples']}, Wavelengths: {props['n_wavelengths']}")
            print(f"    Mean: {props['global_mean']:.4f} ± {props['global_std']:.4f}")
            print(f"    Range: [{props['global_range'][0]:.3f}, {props['global_range'][1]:.3f}]")
            print(f"    Slope: {props['mean_slope']:.4f} ± {props['slope_std']:.4f} (per 1000nm)")
            print(f"    Noise: {props['noise_estimate']:.5f}, SNR: {props['snr_estimate']:.1f}")
            print(f"    PCA 95%: {props['pca_n_components_95']} components")

        # Print comparisons
        print("\n📈 COMPARISON METRICS:")
        print("-" * 40)
        for key, comp in self.comparison_results["comparison"].items():
            print(f"\n  {key}:")
            print(f"    Similarity Score: {comp['similarity_score']:.1f}/100")
            print(f"    Mean Diff: {comp['mean_rel_diff']*100:+.1f}%")
            print(f"    Std Diff: {comp['std_rel_diff']*100:+.1f}%")
            print(f"    Slope Diff: {comp['slope_diff']:+.4f}")
            if "noise_ratio" in comp and comp["noise_ratio"] != np.inf:
                print(f"    Noise Ratio: {comp['noise_ratio']:.2f}x")
            if "mean_spectrum_correlation" in comp:
                print(f"    Mean Spectrum Corr: {comp['mean_spectrum_correlation']:.3f}")
            if "slope_ks_pvalue" in comp:
                print(f"    Slope Distribution Match (KS p-value): {comp['slope_ks_pvalue']:.4f}")

        print("\n" + "=" * 80)

    def get_tuning_recommendations(self) -> Dict[str, List[str]]:
        """
        Get recommendations for tuning synthetic generation parameters.

        Returns:
            Dictionary mapping parameter names to recommendations
        """
        if self.comparison_results is None:
            self.compute_comparison()

        recommendations = {}

        for key, comp in self.comparison_results["comparison"].items():
            recs = []

            # Slope recommendations
            if "slope_diff" in comp:
                if comp["slope_diff"] < -0.02:
                    recs.append(f"Increase global_slope_mean (current slope is {comp['slope_diff']:.3f} lower)")
                elif comp["slope_diff"] > 0.02:
                    recs.append(f"Decrease global_slope_mean (current slope is {comp['slope_diff']:.3f} higher)")

            # Noise recommendations
            if "noise_ratio" in comp and comp["noise_ratio"] != np.inf:
                if comp["noise_ratio"] < 0.7:
                    recs.append(f"Increase noise_base or noise_signal_dep (noise is {(1-comp['noise_ratio'])*100:.0f}% too low)")
                elif comp["noise_ratio"] > 1.3:
                    recs.append(f"Decrease noise_base or noise_signal_dep (noise is {(comp['noise_ratio']-1)*100:.0f}% too high)")

            # Amplitude recommendations
            if "std_rel_diff" in comp:
                if comp["std_rel_diff"] < -0.2:
                    recs.append("Increase path_length_std or scatter_alpha_std for more variation")
                elif comp["std_rel_diff"] > 0.2:
                    recs.append("Decrease path_length_std or scatter_alpha_std for less variation")

            # Complexity recommendations
            if "pca_complexity_diff" in comp:
                if comp["pca_complexity_diff"] < -2:
                    recs.append("Synthetic data is too simple - add more components or variation")
                elif comp["pca_complexity_diff"] > 2:
                    recs.append("Synthetic data is too complex - reduce components or variation")

            recommendations[key] = recs

        return recommendations

    def plot_comparison(
        self,
        figsize: Tuple[float, float] = (16, 12),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Create comprehensive comparison plots.

        Args:
            figsize: Figure size
            save_path: Optional path to save the figure
            show: Whether to show the plot

        Returns:
            Figure object
        """
        if self.comparison_results is None:
            self.compute_comparison()

        fig = plt.figure(figsize=figsize)

        # Determine grid size based on number of datasets
        n_real = len(self.real_datasets)
        n_synth = len(self.synthetic_datasets)

        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Plot 1: Mean spectra overlay
        ax1 = fig.add_subplot(gs[0, 0])
        cmap_real = cm.Blues
        cmap_synth = cm.Oranges

        for i, (name, props) in enumerate(self.real_datasets.items()):
            color = cmap_real(0.5 + 0.3 * i / max(n_real, 1))
            ax1.plot(props.wavelengths, props.mean_spectrum, color=color,
                     linewidth=2, label=f"{name} (real)")
            ax1.fill_between(props.wavelengths,
                             props.mean_spectrum - props.std_spectrum,
                             props.mean_spectrum + props.std_spectrum,
                             color=color, alpha=0.2)

        for i, (name, props) in enumerate(self.synthetic_datasets.items()):
            color = cmap_synth(0.5 + 0.3 * i / max(n_synth, 1))
            ax1.plot(props.wavelengths, props.mean_spectrum, color=color,
                     linewidth=2, linestyle='--', label=f"{name} (synth)")
            ax1.fill_between(props.wavelengths,
                             props.mean_spectrum - props.std_spectrum,
                             props.mean_spectrum + props.std_spectrum,
                             color=color, alpha=0.2)

        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Absorbance")
        ax1.set_title("Mean Spectra ± Std", fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Slope distributions
        ax2 = fig.add_subplot(gs[0, 1])
        for i, (name, props) in enumerate(self.real_datasets.items()):
            if props.slopes is not None:
                color = cmap_real(0.5 + 0.3 * i / max(n_real, 1))
                ax2.hist(props.slopes, bins=30, alpha=0.5, color=color, label=f"{name} (real)")

        for i, (name, props) in enumerate(self.synthetic_datasets.items()):
            if props.slopes is not None:
                color = cmap_synth(0.5 + 0.3 * i / max(n_synth, 1))
                ax2.hist(props.slopes, bins=30, alpha=0.5, color=color, label=f"{name} (synth)")

        ax2.set_xlabel("Slope (per 1000nm)")
        ax2.set_ylabel("Count")
        ax2.set_title("Global Slope Distribution", fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Plot 3: PCA explained variance
        ax3 = fig.add_subplot(gs[0, 2])
        for i, (name, props) in enumerate(self.real_datasets.items()):
            if props.pca_explained_variance is not None:
                color = cmap_real(0.5 + 0.3 * i / max(n_real, 1))
                cumsum = np.cumsum(props.pca_explained_variance) * 100
                ax3.plot(range(1, len(cumsum) + 1), cumsum, 'o-',
                         color=color, label=f"{name} (real)")

        for i, (name, props) in enumerate(self.synthetic_datasets.items()):
            if props.pca_explained_variance is not None:
                color = cmap_synth(0.5 + 0.3 * i / max(n_synth, 1))
                cumsum = np.cumsum(props.pca_explained_variance) * 100
                ax3.plot(range(1, len(cumsum) + 1), cumsum, 's--',
                         color=color, label=f"{name} (synth)")

        ax3.axhline(95, color='gray', linestyle=':', label='95%')
        ax3.set_xlabel("Number of Components")
        ax3.set_ylabel("Cumulative Variance (%)")
        ax3.set_title("PCA Cumulative Variance", fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Noise comparison
        ax4 = fig.add_subplot(gs[1, 0])
        labels = []
        noise_values = []
        colors = []

        for name, props in self.real_datasets.items():
            labels.append(f"{name}\n(real)")
            noise_values.append(props.noise_estimate)
            colors.append(cmap_real(0.7))

        for name, props in self.synthetic_datasets.items():
            labels.append(f"{name}\n(synth)")
            noise_values.append(props.noise_estimate)
            colors.append(cmap_synth(0.7))

        bars = ax4.bar(labels, noise_values, color=colors)
        ax4.set_ylabel("Noise Estimate (σ)")
        ax4.set_title("Noise Level Comparison", fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')

        # Plot 5: Property radar chart
        ax5 = fig.add_subplot(gs[1, 1], projection='polar')

        # Properties to compare (normalized)
        prop_names = ['Mean Abs', 'Std', 'Slope', 'Noise', 'SNR', 'PCA Comp']
        n_props = len(prop_names)
        angles = np.linspace(0, 2 * np.pi, n_props, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Get reference values from first real dataset
        if self.real_datasets:
            ref_props = list(self.real_datasets.values())[0]
            ref_values = [
                ref_props.global_mean,
                ref_props.global_std,
                abs(ref_props.mean_slope) + 0.001,
                ref_props.noise_estimate + 0.0001,
                min(ref_props.snr_estimate, 100),
                ref_props.pca_n_components_95,
            ]
        else:
            ref_values = [1] * n_props

        for i, (name, props) in enumerate(self.real_datasets.items()):
            values = [
                props.global_mean / (ref_values[0] + 1e-10),
                props.global_std / (ref_values[1] + 1e-10),
                abs(props.mean_slope) / (ref_values[2] + 1e-10),
                props.noise_estimate / (ref_values[3] + 1e-10),
                min(props.snr_estimate, 100) / (ref_values[4] + 1e-10),
                props.pca_n_components_95 / (ref_values[5] + 1e-10),
            ]
            values += values[:1]
            color = cmap_real(0.5 + 0.3 * i / max(n_real, 1))
            ax5.plot(angles, values, 'o-', color=color, label=f"{name} (real)")
            ax5.fill(angles, values, alpha=0.1, color=color)

        for i, (name, props) in enumerate(self.synthetic_datasets.items()):
            values = [
                props.global_mean / (ref_values[0] + 1e-10),
                props.global_std / (ref_values[1] + 1e-10),
                abs(props.mean_slope) / (ref_values[2] + 1e-10),
                props.noise_estimate / (ref_values[3] + 1e-10),
                min(props.snr_estimate, 100) / (ref_values[4] + 1e-10),
                props.pca_n_components_95 / (ref_values[5] + 1e-10),
            ]
            values += values[:1]
            color = cmap_synth(0.5 + 0.3 * i / max(n_synth, 1))
            ax5.plot(angles, values, 's--', color=color, label=f"{name} (synth)")
            ax5.fill(angles, values, alpha=0.1, color=color)

        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(prop_names, size=8)
        ax5.set_title("Property Comparison (normalized)", fontweight='bold', pad=20)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=7)

        # Plot 6: Similarity scores
        ax6 = fig.add_subplot(gs[1, 2])
        comp_names = list(self.comparison_results["comparison"].keys())
        scores = [self.comparison_results["comparison"][k]["similarity_score"] for k in comp_names]

        colors_score = [cm.RdYlGn(s / 100) for s in scores]
        bars = ax6.barh(comp_names, scores, color=colors_score)
        ax6.set_xlim(0, 100)
        ax6.set_xlabel("Similarity Score")
        ax6.set_title("Overall Similarity Scores", fontweight='bold')
        ax6.axvline(80, color='green', linestyle='--', alpha=0.5, label='Good (80)')
        ax6.axvline(60, color='orange', linestyle='--', alpha=0.5, label='Fair (60)')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3, axis='x')

        # Add score labels
        for bar, score in zip(bars, scores):
            ax6.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f'{score:.1f}', va='center', fontsize=9)

        # Plot 7-9: Sample spectra comparison (if we have matched wavelengths)
        ax7 = fig.add_subplot(gs[2, :])

        # Show sample spectra from each dataset
        n_show = 20
        for i, (name, props) in enumerate(list(self.real_datasets.items())[:1]):
            # Need to get actual spectra - for now just show envelope
            ax7.fill_between(props.wavelengths, props.min_spectrum, props.max_spectrum,
                             alpha=0.3, color=cmap_real(0.7), label=f"{name} range (real)")
            ax7.plot(props.wavelengths, props.mean_spectrum, color=cmap_real(0.9),
                     linewidth=2, label=f"{name} mean (real)")

        for i, (name, props) in enumerate(list(self.synthetic_datasets.items())[:1]):
            ax7.fill_between(props.wavelengths, props.min_spectrum, props.max_spectrum,
                             alpha=0.3, color=cmap_synth(0.7), label=f"{name} range (synth)")
            ax7.plot(props.wavelengths, props.mean_spectrum, color=cmap_synth(0.9),
                     linewidth=2, linestyle='--', label=f"{name} mean (synth)")

        ax7.set_xlabel("Wavelength (nm)")
        ax7.set_ylabel("Absorbance")
        ax7.set_title("Spectral Range Comparison", fontweight='bold')
        ax7.legend(loc='upper right', fontsize=8)
        ax7.grid(True, alpha=0.3)

        plt.suptitle("Synthetic vs Real NIRS Spectra Comparison",
                     fontsize=14, fontweight='bold', y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig


def compare_with_real_data(
    X_synthetic: np.ndarray,
    X_real: np.ndarray,
    wavelengths_synth: Optional[np.ndarray] = None,
    wavelengths_real: Optional[np.ndarray] = None,
    synth_name: str = "synthetic",
    real_name: str = "real",
    show_plot: bool = True,
    print_summary: bool = True,
) -> SyntheticRealComparator:
    """
    Quick comparison function for synthetic vs real spectra.

    Args:
        X_synthetic: Synthetic spectra (n_samples, n_wavelengths)
        X_real: Real spectra (n_samples, n_wavelengths)
        wavelengths_synth: Wavelength grid for synthetic
        wavelengths_real: Wavelength grid for real
        synth_name: Name for synthetic dataset
        real_name: Name for real dataset
        show_plot: Whether to show comparison plots
        print_summary: Whether to print summary

    Returns:
        SyntheticRealComparator with results
    """
    comparator = SyntheticRealComparator()
    comparator.add_real_dataset(X_real, wavelengths_real, real_name)
    comparator.add_synthetic_dataset(X_synthetic, wavelengths_synth, synth_name)
    comparator.compute_comparison()

    if print_summary:
        comparator.print_summary()

        # Print recommendations
        recs = comparator.get_tuning_recommendations()
        if any(recs.values()):
            print("\n💡 TUNING RECOMMENDATIONS:")
            print("-" * 40)
            for key, rec_list in recs.items():
                if rec_list:
                    print(f"\n  For {key}:")
                    for rec in rec_list:
                        print(f"    • {rec}")
            print()

    if show_plot:
        comparator.plot_comparison()

    return comparator
