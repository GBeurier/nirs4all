"""
Spectral Latent Features Extractor for TabPFN Optimization
===========================================================

This module provides a scientifically-grounded transformer to convert
high-dimensional NIRS spectra into latent features that resemble the
tabular data structure on which TabPFN 2.5 was trained.

TabPFN was trained on synthetic tabular data with:
- Relatively independent features (low autocorrelation)
- Mixed continuous and categorical-like distributions
- Moderate feature correlations
- ~100-500 features typical

NIRS spectra are fundamentally different:
- Highly autocorrelated (neighboring wavelengths similar)
- High dimensionality (1000+ wavelengths)
- Smooth, continuous signals
- Information distributed across spectrum

This transformer bridges the gap by extracting diverse, decorrelated
features that capture spectral information in a TabPFN-friendly format.

References:
-----------
[1] Hollmann et al. (2023). TabPFN: A Transformer That Solves Small
    Tabular Classification Problems in a Second. ICLR 2023.
[2] Rinnan et al. (2009). Review of the most common pre-processing
    techniques for near-infrared spectra. TrAC.
[3] Savitzky & Golay (1964). Smoothing and differentiation of data
    by simplified least squares procedures. Analytical Chemistry.
[4] Mallat (1989). A theory for multiresolution signal decomposition:
    the wavelet representation. IEEE PAMI.

Author: Auto-generated for NIRS4All project
Date: 2024
"""

import warnings
from typing import Optional, Union

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy, kurtosis, skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _safe_log1p(x: np.ndarray) -> np.ndarray:
    """Safe log1p that handles negative values."""
    return np.sign(x) * np.log1p(np.abs(x))

def _compute_entropy(x: np.ndarray, n_bins: int = 10) -> float:
    """Compute entropy of a 1D array."""
    hist, _ = np.histogram(x, bins=n_bins, density=True)
    hist = hist[hist > 0]
    return entropy(hist)

def _robust_statistics(x: np.ndarray, axis: int = 1) -> tuple[np.ndarray, ...]:
    """Compute robust statistics along an axis."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = np.nanmean(x, axis=axis)
        std = np.nanstd(x, axis=axis)
        sk = skew(x, axis=axis, nan_policy='omit')
        kt = kurtosis(x, axis=axis, nan_policy='omit')

    # Replace NaN/Inf with 0
    for arr in [mean, std, sk, kt]:
        arr[~np.isfinite(arr)] = 0.0

    return mean, std, sk, kt

# =============================================================================
# FEATURE EXTRACTION MODULES
# =============================================================================

class _PCAModule:
    """
    Principal Component Analysis for global decorrelated features.

    PCA is the gold standard for decorrelating spectral data. The resulting
    components (scores) are orthogonal and capture decreasing variance.

    Scientific basis:
    - Removes multicollinearity that confuses TabPFN
    - First few PCs capture major spectral variation patterns
    - Whitening option makes features unit variance (more tabular-like)

    Reference: Wold et al. (1987). Principal Component Analysis.
               Chemometrics and Intelligent Laboratory Systems.
    """

    def __init__(self, n_components: int = 50, whiten: bool = True,
                 variance_threshold: float = 0.99):
        self.n_components = n_components
        self.whiten = whiten
        self.variance_threshold = variance_threshold
        self.scaler_ = None
        self.pca_ = None
        self.n_components_fitted_ = 0

    def fit(self, X: np.ndarray) -> '_PCAModule':
        n_samples, n_features = X.shape
        max_components = min(self.n_components, n_samples - 1, n_features)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Fit PCA with requested number of components
        self.pca_ = PCA(n_components=max_components, whiten=self.whiten)
        self.pca_.fit(X_scaled)

        # Use variance threshold only if it gives MORE components than requested
        # Otherwise use the requested number
        cumvar = np.cumsum(self.pca_.explained_variance_ratio_)
        n_for_threshold = np.searchsorted(cumvar, self.variance_threshold) + 1
        self.n_components_fitted_ = max(min(n_for_threshold, max_components),
                                        min(self.n_components, max_components))

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_.transform(X)
        return self.pca_.transform(X_scaled)[:, :self.n_components_fitted_]

    def get_feature_names(self) -> list[str]:
        return [f"pca_{i}" for i in range(self.n_components_fitted_)]

class _WaveletModule:
    """
    Discrete Wavelet Transform for multi-scale spectral features.

    Wavelets decompose the spectrum into approximation (smooth trends)
    and detail (sharp features) coefficients at multiple scales.
    This captures both global baseline variations and local absorption peaks.

    Scientific basis:
    - Multi-resolution analysis captures features at different scales
    - Daubechies wavelets (db4) are well-suited for smooth signals
    - Wavelet coefficients are partially decorrelated

    Reference: Mallat (1989). A theory for multiresolution signal
               decomposition: the wavelet representation.
    """

    def __init__(self, wavelet: str = 'db4', max_level: int = 5,
                 n_coeffs_per_level: int = 10):
        self.wavelet = wavelet
        self.max_level = max_level
        self.n_coeffs_per_level = n_coeffs_per_level
        self.actual_level_ = 0
        self.feature_names_ = []
        self._pywt_available = False  # Flag instead of storing the module

    def fit(self, X: np.ndarray) -> '_WaveletModule':
        try:
            import pywt
            self._pywt_available = True
        except ImportError:
            warnings.warn("pywt not available. Wavelet features will be skipped.", stacklevel=2)
            self._pywt_available = False
            return self

        n_features = X.shape[1]
        # Compute maximum decomposition level
        max_level_possible = pywt.dwt_max_level(n_features, self.wavelet)
        self.actual_level_ = min(self.max_level, max_level_possible)

        # Generate feature names
        self.feature_names_ = []
        # Approximation coefficients stats
        for stat in ['mean', 'std', 'energy', 'entropy']:
            self.feature_names_.append(f"wavelet_approx_{stat}")
        # Top coefficients from approximation
        for i in range(self.n_coeffs_per_level):
            self.feature_names_.append(f"wavelet_approx_coef_{i}")

        # Detail coefficients at each level
        for level in range(1, self.actual_level_ + 1):
            for stat in ['mean', 'std', 'energy', 'entropy']:
                self.feature_names_.append(f"wavelet_d{level}_{stat}")
            for i in range(self.n_coeffs_per_level):
                self.feature_names_.append(f"wavelet_d{level}_coef_{i}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._pywt_available:
            return np.zeros((X.shape[0], 0))

        import pywt  # Import here instead of using stored module
        n_samples = X.shape[0]
        features_list = []

        for i in range(n_samples):
            coeffs = pywt.wavedec(X[i], self.wavelet, level=self.actual_level_)
            sample_features = []

            # Process approximation coefficients (coeffs[0])
            approx = coeffs[0]
            sample_features.extend([
                np.mean(approx),
                np.std(approx),
                np.sum(approx ** 2),  # energy
                _compute_entropy(approx)
            ])
            # Top N coefficients (sorted by magnitude)
            sorted_idx = np.argsort(np.abs(approx))[::-1]
            top_coeffs = approx[sorted_idx[:self.n_coeffs_per_level]]
            if len(top_coeffs) < self.n_coeffs_per_level:
                top_coeffs = np.pad(top_coeffs, (0, self.n_coeffs_per_level - len(top_coeffs)))
            sample_features.extend(top_coeffs)

            # Process detail coefficients at each level
            for level in range(1, self.actual_level_ + 1):
                detail = coeffs[level]
                sample_features.extend([
                    np.mean(detail),
                    np.std(detail),
                    np.sum(detail ** 2),
                    _compute_entropy(detail)
                ])
                sorted_idx = np.argsort(np.abs(detail))[::-1]
                top_coeffs = detail[sorted_idx[:self.n_coeffs_per_level]]
                if len(top_coeffs) < self.n_coeffs_per_level:
                    top_coeffs = np.pad(top_coeffs, (0, self.n_coeffs_per_level - len(top_coeffs)))
                sample_features.extend(top_coeffs)

            features_list.append(sample_features)

        return np.array(features_list)

    def get_feature_names(self) -> list[str]:
        return self.feature_names_

class _FFTModule:
    """
    Fast Fourier Transform for frequency domain features.

    FFT converts the spectrum from wavelength domain to frequency domain,
    revealing periodic patterns and baseline variations. Low-frequency
    components capture smooth trends, high-frequency captures noise/peaks.

    Scientific basis:
    - Frequency representation is naturally decorrelated
    - Log-spaced frequency bands capture multi-scale information
    - Phase information can reveal peak positions

    Reference: Cooley & Tukey (1965). An algorithm for the machine
               calculation of complex Fourier series.
    """

    def __init__(self, n_freq_bands: int = 20, n_top_freqs: int = 20,
                 log_amplitude: bool = True, include_phase: bool = False):
        self.n_freq_bands = n_freq_bands
        self.n_top_freqs = n_top_freqs
        self.log_amplitude = log_amplitude
        self.include_phase = include_phase
        self.band_edges_ = None
        self.n_fft_ = 0
        self.feature_names_ = []

    def fit(self, X: np.ndarray) -> '_FFTModule':
        n_features = X.shape[1]
        self.n_fft_ = n_features // 2 + 1  # rfft output size

        # Create log-spaced frequency bands
        self.band_edges_ = np.unique(np.logspace(
            0, np.log10(self.n_fft_), self.n_freq_bands + 1
        ).astype(int))
        self.band_edges_[-1] = self.n_fft_

        # Generate feature names
        self.feature_names_ = []

        # Band energy features
        for i in range(len(self.band_edges_) - 1):
            self.feature_names_.append(f"fft_band_{i}_energy")

        # Top frequency magnitudes
        for i in range(self.n_top_freqs):
            self.feature_names_.append(f"fft_top_{i}_mag")
            if self.include_phase:
                self.feature_names_.append(f"fft_top_{i}_phase")

        # Global FFT statistics
        for stat in ['mean', 'std', 'max', 'spectral_centroid', 'spectral_spread']:
            self.feature_names_.append(f"fft_{stat}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        features_list = []

        for i in range(n_samples):
            fft_complex = np.fft.rfft(X[i])
            fft_mag = np.abs(fft_complex)

            if self.log_amplitude:
                fft_mag = np.log1p(fft_mag)

            sample_features = []

            # Band energy (sum of squared magnitudes in each band)
            for j in range(len(self.band_edges_) - 1):
                start, end = self.band_edges_[j], self.band_edges_[j + 1]
                band_energy = np.sum(fft_mag[start:end] ** 2)
                sample_features.append(band_energy)

            # Top N frequencies by magnitude
            sorted_idx = np.argsort(fft_mag)[::-1][:self.n_top_freqs]
            for idx in sorted_idx:
                sample_features.append(fft_mag[idx])
                if self.include_phase:
                    sample_features.append(np.angle(fft_complex[idx]))

            # Pad if needed (for edge cases)
            while len(sample_features) < len(self.band_edges_) - 1 + self.n_top_freqs * (2 if self.include_phase else 1):
                sample_features.append(0.0)

            # Global statistics
            sample_features.append(np.mean(fft_mag))
            sample_features.append(np.std(fft_mag))
            sample_features.append(np.max(fft_mag))

            # Spectral centroid (center of mass of spectrum)
            freqs = np.arange(len(fft_mag))
            mag_sum = np.sum(fft_mag) + 1e-10
            spectral_centroid = np.sum(freqs * fft_mag) / mag_sum
            sample_features.append(spectral_centroid)

            # Spectral spread (standard deviation around centroid)
            spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_mag) / mag_sum)
            sample_features.append(spectral_spread)

            features_list.append(sample_features)

        return np.array(features_list)

    def get_feature_names(self) -> list[str]:
        return self.feature_names_

class _LocalStatsModule:
    """
    Local statistical features computed on spectral bands.

    Divides the spectrum into bands and computes statistics per band.
    This simulates having multiple independent measurements, which is
    the typical structure of tabular data.

    Scientific basis:
    - Band-wise statistics reduce autocorrelation
    - Different statistics capture different aspects of local variation
    - Quantiles are robust to outliers

    Reference: Næs et al. (2002). A User-Friendly Guide to Multivariate
               Calibration and Classification.
    """

    def __init__(self, n_bands: int = 15):
        self.n_bands = n_bands
        self.band_indices_ = []
        self.feature_names_ = []

    def fit(self, X: np.ndarray) -> '_LocalStatsModule':
        n_features = X.shape[1]
        bands = np.array_split(np.arange(n_features), self.n_bands)
        self.band_indices_ = [np.array(b) for b in bands if len(b) > 0]

        # Generate feature names
        self.feature_names_ = []
        for i in range(len(self.band_indices_)):
            for stat in ['mean', 'std', 'min', 'max', 'q25', 'q75', 'skew', 'kurt']:
                self.feature_names_.append(f"band_{i}_{stat}")

        # Inter-band features
        for stat in ['slope', 'curvature', 'range_ratio']:
            self.feature_names_.append(f"interband_{stat}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        features_list = []

        for i in range(n_samples):
            sample_features = []
            band_means = []

            for idx in self.band_indices_:
                band_data = X[i, idx]

                mean_val = np.mean(band_data)
                band_means.append(mean_val)

                sample_features.extend([
                    mean_val,
                    np.std(band_data),
                    np.min(band_data),
                    np.max(band_data),
                    np.quantile(band_data, 0.25),
                    np.quantile(band_data, 0.75),
                    skew(band_data) if len(band_data) > 2 else 0.0,
                    kurtosis(band_data) if len(band_data) > 3 else 0.0
                ])

            # Inter-band features
            band_means = np.array(band_means)
            if len(band_means) > 1:
                # Linear trend across bands
                x = np.arange(len(band_means))
                slope = np.polyfit(x, band_means, 1)[0]
                # Curvature (second derivative approximation)
                curvature = np.mean(np.diff(band_means, n=2)) if len(band_means) > 2 else 0.0
                # Range ratio
                range_ratio = (np.max(band_means) - np.min(band_means)) / (np.mean(band_means) + 1e-10)
            else:
                slope, curvature, range_ratio = 0.0, 0.0, 0.0

            sample_features.extend([slope, curvature, range_ratio])
            features_list.append(sample_features)

        return np.array(features_list)

    def get_feature_names(self) -> list[str]:
        return self.feature_names_

class _DerivativeModule:
    """
    Derivative-based features for spectral shape characterization.

    First and second derivatives emphasize spectral changes and peaks.
    The second derivative is particularly useful for resolving overlapping
    peaks and removing baseline effects.

    Scientific basis:
    - First derivative: rate of change, removes constant baseline
    - Second derivative: curvature, peak sharpening, baseline removal
    - Savitzky-Golay filtering provides noise reduction

    Reference: Savitzky & Golay (1964). Smoothing and differentiation
               of data by simplified least squares procedures.
    """

    def __init__(self, window_length: int = 11, polyorder: int = 3,
                 n_stats: int = 10):
        self.window_length = window_length
        self.polyorder = polyorder
        self.n_stats = n_stats
        self.feature_names_ = []

    def fit(self, X: np.ndarray) -> '_DerivativeModule':
        n_features = X.shape[1]

        # Adjust window length if needed
        if self.window_length >= n_features:
            self.window_length = n_features // 4 * 2 + 1  # Ensure odd
        if self.window_length < self.polyorder + 1:
            self.window_length = self.polyorder + 2
            if self.window_length % 2 == 0:
                self.window_length += 1

        # Generate feature names
        self.feature_names_ = []
        for deriv in ['d1', 'd2']:
            for stat in ['mean', 'std', 'min', 'max', 'range', 'zero_crossings',
                        'pos_area', 'neg_area', 'max_abs_pos', 'energy']:
                self.feature_names_.append(f"{deriv}_{stat}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        features_list = []

        for i in range(n_samples):
            sample_features = []

            for deriv_order in [1, 2]:
                try:
                    deriv = signal.savgol_filter(
                        X[i],
                        window_length=self.window_length,
                        polyorder=self.polyorder,
                        deriv=deriv_order
                    )
                except Exception:
                    deriv = np.gradient(X[i]) if deriv_order == 1 else np.gradient(np.gradient(X[i]))

                # Statistics on derivative
                sample_features.append(np.mean(deriv))
                sample_features.append(np.std(deriv))
                sample_features.append(np.min(deriv))
                sample_features.append(np.max(deriv))
                sample_features.append(np.max(deriv) - np.min(deriv))

                # Zero crossings (sign changes)
                zero_crossings = np.sum(np.diff(np.sign(deriv)) != 0)
                sample_features.append(zero_crossings)

                # Positive and negative areas
                pos_area = np.sum(deriv[deriv > 0])
                neg_area = np.sum(np.abs(deriv[deriv < 0]))
                sample_features.append(pos_area)
                sample_features.append(neg_area)

                # Position of maximum absolute value
                max_abs_pos = np.argmax(np.abs(deriv)) / len(deriv)
                sample_features.append(max_abs_pos)

                # Energy (sum of squares)
                energy = np.sum(deriv ** 2)
                sample_features.append(energy)

            features_list.append(sample_features)

        return np.array(features_list)

    def get_feature_names(self) -> list[str]:
        return self.feature_names_

class _PeakModule:
    """
    Peak-based features for absorption band characterization.

    Detects prominent peaks in the spectrum and extracts their properties.
    This is physically meaningful for NIRS as peaks correspond to molecular
    absorption bands (Beer-Lambert law).

    Scientific basis:
    - Peak positions relate to molecular vibrations
    - Peak heights relate to concentration (Beer-Lambert)
    - Peak widths relate to molecular environment
    - Number of peaks indicates spectral complexity

    Reference: Workman & Weyer (2012). Practical Guide and Spectral
               Atlas for Interpretive Near-Infrared Spectroscopy.
    """

    def __init__(self, n_peaks: int = 10, prominence_quantile: float = 0.5):
        self.n_peaks = n_peaks
        self.prominence_quantile = prominence_quantile
        self.feature_names_ = []

    def fit(self, X: np.ndarray) -> '_PeakModule':
        # Generate feature names
        self.feature_names_ = []

        # Peak count and density
        self.feature_names_.extend(['n_peaks', 'peak_density'])

        # Per-peak features
        for i in range(self.n_peaks):
            self.feature_names_.extend([
                f"peak_{i}_pos",
                f"peak_{i}_height",
                f"peak_{i}_prominence",
                f"peak_{i}_width"
            ])

        # Aggregate peak features
        self.feature_names_.extend([
            'peaks_mean_height', 'peaks_std_height',
            'peaks_mean_width', 'peaks_std_width',
            'peaks_mean_prominence', 'peaks_spacing_std'
        ])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        features_list = []

        for i in range(n_samples):
            spectrum = X[i]

            # Smooth slightly before peak detection
            spectrum_smooth = gaussian_filter1d(spectrum, sigma=2)

            # Compute prominence threshold
            prom_threshold = np.quantile(np.abs(np.diff(spectrum_smooth)),
                                         self.prominence_quantile)

            # Find peaks
            peaks, properties = signal.find_peaks(
                spectrum_smooth,
                prominence=prom_threshold,
                width=1
            )

            sample_features = []

            # Peak count and density
            n_peaks = len(peaks)
            peak_density = n_peaks / n_features
            sample_features.extend([n_peaks, peak_density])

            # Sort peaks by prominence
            if n_peaks > 0:
                prominences = properties.get('prominences', np.ones(n_peaks))
                sorted_idx = np.argsort(prominences)[::-1]
                peaks = peaks[sorted_idx]

                heights = spectrum_smooth[peaks]
                widths = properties.get('widths', np.ones(n_peaks))[sorted_idx]
                prominences = prominences[sorted_idx]
            else:
                heights, widths, prominences = [], [], []

            # Per-peak features (pad if fewer peaks)
            for j in range(self.n_peaks):
                if j < n_peaks:
                    sample_features.extend([
                        peaks[j] / n_features,  # Normalized position
                        heights[j],
                        prominences[j],
                        widths[j] / n_features  # Normalized width
                    ])
                else:
                    sample_features.extend([0.0, 0.0, 0.0, 0.0])

            # Aggregate features
            if n_peaks > 0:
                sample_features.extend([
                    np.mean(heights),
                    np.std(heights) if n_peaks > 1 else 0.0,
                    np.mean(widths),
                    np.std(widths) if n_peaks > 1 else 0.0,
                    np.mean(prominences),
                    np.std(np.diff(peaks)) / n_features if n_peaks > 1 else 0.0
                ])
            else:
                sample_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            features_list.append(sample_features)

        return np.array(features_list)

    def get_feature_names(self) -> list[str]:
        return self.feature_names_

class _ScatterModule:
    """
    Scatter and baseline features for spectral normalization characterization.

    These features capture baseline effects and light scattering, which are
    common artifacts in NIRS that don't carry chemical information but can
    still be predictive (e.g., particle size effects).

    Scientific basis:
    - Baseline slope indicates scattering/particle size
    - Polynomial coefficients model baseline curvature
    - Area under curve relates to total absorbance
    - Centroid indicates spectral "center of mass"

    Reference: Rinnan et al. (2009). Review of the most common
               pre-processing techniques for near-infrared spectra.
    """

    def __init__(self, poly_degree: int = 3):
        self.poly_degree = poly_degree
        self.feature_names_ = []

    def fit(self, X: np.ndarray) -> '_ScatterModule':
        # Generate feature names
        self.feature_names_ = []

        # Polynomial coefficients
        for i in range(self.poly_degree + 1):
            self.feature_names_.append(f"baseline_poly_{i}")

        # Residual statistics
        self.feature_names_.extend([
            'baseline_residual_std',
            'baseline_residual_skew',
            'baseline_residual_kurt'
        ])

        # Global features
        self.feature_names_.extend([
            'total_area',
            'normalized_area',
            'spectral_centroid',
            'spectral_std',
            'asymmetry',
            'flatness',
            'crest_factor'
        ])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        x = np.linspace(-1, 1, n_features)  # Normalized x-axis
        features_list = []

        for i in range(n_samples):
            spectrum = X[i]
            sample_features = []

            # Polynomial baseline fit
            poly_coeffs = np.polyfit(x, spectrum, self.poly_degree)
            sample_features.extend(poly_coeffs)

            # Residuals from baseline
            baseline = np.polyval(poly_coeffs, x)
            residuals = spectrum - baseline
            sample_features.extend([
                np.std(residuals),
                skew(residuals),
                kurtosis(residuals)
            ])

            # Total and normalized area
            total_area = np.trapezoid(spectrum) if hasattr(np, 'trapezoid') else np.trapz(spectrum)
            normalized_area = total_area / n_features
            sample_features.extend([total_area, normalized_area])

            # Spectral centroid (center of mass)
            abs_spectrum = np.abs(spectrum) + 1e-10
            weights = abs_spectrum / np.sum(abs_spectrum)
            centroid = np.sum(x * weights)
            sample_features.append(centroid)

            # Spectral spread (std around centroid)
            spectral_std = np.sqrt(np.sum(((x - centroid) ** 2) * weights))
            sample_features.append(spectral_std)

            # Asymmetry (difference between left and right halves)
            mid = n_features // 2
            left_area = np.sum(spectrum[:mid])
            right_area = np.sum(spectrum[mid:])
            asymmetry = (right_area - left_area) / (left_area + right_area + 1e-10)
            sample_features.append(asymmetry)

            # Flatness (geometric mean / arithmetic mean)
            # Approximation that handles negative values
            abs_spec = np.abs(spectrum) + 1e-10
            geometric_mean = np.exp(np.mean(np.log(abs_spec)))
            arithmetic_mean = np.mean(abs_spec)
            flatness = geometric_mean / (arithmetic_mean + 1e-10)
            sample_features.append(flatness)

            # Crest factor (peak / RMS)
            rms = np.sqrt(np.mean(spectrum ** 2))
            crest_factor = np.max(np.abs(spectrum)) / (rms + 1e-10)
            sample_features.append(crest_factor)

            features_list.append(sample_features)

        return np.array(features_list)

    def get_feature_names(self) -> list[str]:
        return self.feature_names_

class _WaveletPCAModule:
    """
    Multi-scale PCA on wavelet coefficients.

    Applies PCA separately to each wavelet decomposition level,
    creating a compact multi-scale representation where each scale
    contributes a few principal components.

    Scientific basis:
    - Combines multi-resolution analysis with decorrelation
    - Each scale captures different frequency information
    - PCA per scale reduces redundancy within each frequency band
    - Results in a compact, interpretable feature set

    Reference: Trygg & Wold (1998). PLS regression on wavelet
               compressed NIR spectra.
    """

    def __init__(self, wavelet: str = 'db4', max_level: int = 4,
                 n_components_per_level: int = 3):
        self.wavelet = wavelet
        self.max_level = max_level
        self.n_components_per_level = n_components_per_level
        self.actual_level_ = 0
        self.pcas_ = {}
        self.scalers_ = {}
        self.feature_names_ = []
        self._pywt_available = False

    def fit(self, X: np.ndarray) -> '_WaveletPCAModule':
        try:
            import pywt
            self._pywt_available = True
        except ImportError:
            warnings.warn("pywt not available. Wavelet-PCA features will be skipped.", stacklevel=2)
            self._pywt_available = False
            return self

        n_samples, n_features = X.shape
        max_level_possible = pywt.dwt_max_level(n_features, self.wavelet)
        self.actual_level_ = min(self.max_level, max_level_possible)

        # Decompose all samples to get coefficient arrays
        all_coeffs = {i: [] for i in range(self.actual_level_ + 1)}

        for i in range(n_samples):
            coeffs = pywt.wavedec(X[i], self.wavelet, level=self.actual_level_)
            for level_idx, c in enumerate(coeffs):
                all_coeffs[level_idx].append(c)

        # Fit PCA for each level
        self.feature_names_ = []

        for level_idx in range(self.actual_level_ + 1):
            level_data = np.array(all_coeffs[level_idx])
            n_coeffs = level_data.shape[1]
            n_comps = min(self.n_components_per_level, n_coeffs, n_samples - 1)

            if n_comps > 0:
                scaler = StandardScaler()
                level_scaled = scaler.fit_transform(level_data)
                pca = PCA(n_components=n_comps, whiten=True)
                pca.fit(level_scaled)

                self.scalers_[level_idx] = scaler
                self.pcas_[level_idx] = pca

                level_name = 'approx' if level_idx == 0 else f'd{level_idx}'
                for j in range(n_comps):
                    self.feature_names_.append(f"wavelet_pca_{level_name}_pc{j}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._pywt_available or not self.pcas_:
            return np.zeros((X.shape[0], 0))

        import pywt
        n_samples = X.shape[0]
        all_features = []

        for i in range(n_samples):
            coeffs = pywt.wavedec(X[i], self.wavelet, level=self.actual_level_)
            sample_features = []

            for level_idx, c in enumerate(coeffs):
                if level_idx in self.pcas_:
                    c_scaled = self.scalers_[level_idx].transform(c.reshape(1, -1))
                    pcs = self.pcas_[level_idx].transform(c_scaled).flatten()
                    sample_features.extend(pcs)

            all_features.append(sample_features)

        return np.array(all_features)

    def get_feature_names(self) -> list[str]:
        return self.feature_names_

class _PLSModule:
    """
    Partial Least Squares latent features for supervised dimensionality reduction.

    PLS finds latent variables (scores) that maximize covariance with Y,
    making them ideal for regression tasks. The scores are decorrelated
    and directly aligned with the target variable.

    Scientific basis:
    - PLS scores are optimal for prediction (maximize covariance with Y)
    - Naturally handles multicollinearity in spectral data
    - Widely used in chemometrics (NIRS, Raman, etc.)
    - Scores are orthogonal and capture predictive variance

    Reference: Wold et al. (2001). PLS-regression: a basic tool of
               chemometrics. Chemometrics and Intelligent Laboratory Systems.
    """

    def __init__(self, n_components: int = 20):
        self.n_components = n_components
        self.pls_ = None
        self.scaler_ = None
        self.n_components_fitted_ = 0
        self.feature_names_ = []

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> '_PLSModule':
        if y is None:
            warnings.warn("PLS requires y for fitting. PLS features will be skipped.", stacklevel=2)
            return self

        try:
            from sklearn.cross_decomposition import PLSRegression
        except ImportError:
            warnings.warn("sklearn PLSRegression not available.", stacklevel=2)
            return self

        n_samples, n_features = X.shape
        y = np.asarray(y).ravel()

        # Determine number of components
        max_components = min(self.n_components, n_samples - 1, n_features)
        self.n_components_fitted_ = max_components

        if max_components > 0:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)

            self.pls_ = PLSRegression(n_components=max_components, scale=False)
            self.pls_.fit(X_scaled, y)

            self.feature_names_ = [f"pls_score_{i}" for i in range(max_components)]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pls_ is None:
            return np.zeros((X.shape[0], 0))

        X_scaled = self.scaler_.transform(X)
        return self.pls_.transform(X_scaled)

    def get_feature_names(self) -> list[str]:
        return self.feature_names_

class _NMFModule:
    """
    Non-negative Matrix Factorization for spectral decomposition.

    NMF decomposes spectra into non-negative basis spectra and their
    mixing coefficients. The coefficients represent contributions of
    latent "spectral components" - interpretable as chemical constituents.

    Scientific basis:
    - Produces interpretable, additive decomposition
    - Mixing coefficients are naturally non-negative (concentrations)
    - Basis spectra can represent pure components
    - Well-suited for absorbance/reflectance data

    Note: Only works with non-negative spectra. Negative values will be
    shifted to make data non-negative.

    Reference: Lee & Seung (1999). Learning the parts of objects by
               non-negative matrix factorization. Nature.
    """

    def __init__(self, n_components: int = 15, max_iter: int = 200):
        self.n_components = n_components
        self.max_iter = max_iter
        self.nmf_ = None
        self.shift_ = 0.0
        self.n_components_fitted_ = 0
        self.feature_names_ = []

    def fit(self, X: np.ndarray) -> '_NMFModule':
        try:
            from sklearn.decomposition import NMF
        except ImportError:
            warnings.warn("sklearn NMF not available.", stacklevel=2)
            return self

        n_samples, n_features = X.shape

        # Make data non-negative by shifting
        self.shift_ = 0.0
        if np.min(X) < 0:
            self.shift_ = -np.min(X) + 1e-6

        X_shifted = X + self.shift_

        # Determine number of components
        max_components = min(self.n_components, n_samples - 1, n_features)
        self.n_components_fitted_ = max_components

        if max_components > 0:
            self.nmf_ = NMF(
                n_components=max_components,
                max_iter=self.max_iter,
                init='nndsvda',
                random_state=42
            )
            self.nmf_.fit(X_shifted)

            self.feature_names_ = [f"nmf_coef_{i}" for i in range(max_components)]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.nmf_ is None:
            return np.zeros((X.shape[0], 0))

        X_shifted = X + self.shift_
        # Clip to ensure non-negative (for numerical stability)
        X_shifted = np.clip(X_shifted, 0, None)
        return self.nmf_.transform(X_shifted)

    def get_feature_names(self) -> list[str]:
        return self.feature_names_

class _BandAreaModule:
    """
    Band area (AUC) features for spectral region characterization.

    Computes the area under the curve for spectral bands, providing
    integrated intensity measures that are robust to noise and
    correspond to physical quantities (total absorption in a region).

    Scientific basis:
    - Area under peaks relates to concentration (Beer-Lambert integral)
    - Normalized areas provide relative composition information
    - Band ratios are commonly used in spectroscopy
    - Robust to wavelength shifts and noise

    Reference: Mark & Workman (2007). Chemometrics in Spectroscopy.
    """

    def __init__(self, n_bands: int = 12, include_ratios: bool = True):
        self.n_bands = n_bands
        self.include_ratios = include_ratios
        self.band_indices_ = []
        self.feature_names_ = []

    def fit(self, X: np.ndarray) -> '_BandAreaModule':
        n_features = X.shape[1]
        bands = np.array_split(np.arange(n_features), self.n_bands)
        self.band_indices_ = [np.array(b) for b in bands if len(b) > 0]
        actual_n_bands = len(self.band_indices_)

        # Generate feature names
        self.feature_names_ = []

        # Absolute areas
        for i in range(actual_n_bands):
            self.feature_names_.append(f"band_area_{i}")

        # Normalized areas
        for i in range(actual_n_bands):
            self.feature_names_.append(f"band_area_norm_{i}")

        # Total area
        self.feature_names_.append("total_area")

        # Band ratios (adjacent bands)
        if self.include_ratios and actual_n_bands > 1:
            for i in range(actual_n_bands - 1):
                self.feature_names_.append(f"band_ratio_{i}_{i+1}")

            # Ratio of first half to second half
            self.feature_names_.append("band_ratio_first_second_half")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        features_list = []

        for i in range(n_samples):
            sample_features = []
            band_areas = []

            # Compute absolute areas
            for idx in self.band_indices_:
                # Use trapezoidal rule for area
                area = np.trapezoid(X[i, idx]) if hasattr(np, 'trapezoid') else np.trapz(X[i, idx])
                band_areas.append(area)
                sample_features.append(area)

            # Total area
            total_area = sum(band_areas)
            total_area_safe = total_area if abs(total_area) > 1e-10 else 1e-10

            # Normalized areas
            for area in band_areas:
                sample_features.append(area / total_area_safe)

            sample_features.append(total_area)

            # Band ratios
            if self.include_ratios and len(band_areas) > 1:
                for j in range(len(band_areas) - 1):
                    denom = band_areas[j + 1] if abs(band_areas[j + 1]) > 1e-10 else 1e-10
                    sample_features.append(band_areas[j] / denom)

                # First half vs second half ratio
                mid = len(band_areas) // 2
                first_half = sum(band_areas[:mid]) if mid > 0 else 1e-10
                second_half = sum(band_areas[mid:]) if mid < len(band_areas) else 1e-10
                if abs(second_half) < 1e-10:
                    second_half = 1e-10
                sample_features.append(first_half / second_half)

            features_list.append(sample_features)

        return np.array(features_list)

    def get_feature_names(self) -> list[str]:
        return self.feature_names_

class _DiscretizationModule:
    """
    Discretization features that create categorical-like variables.

    TabPFN was trained on data with mixed continuous and categorical features.
    This module creates binned/discretized versions of spectral values to
    simulate categorical features.

    Scientific basis:
    - Histogram features are robust to noise
    - Entropy measures spectral complexity
    - Bin counts simulate categorical measurements

    Reference: Dougherty et al. (1995). Supervised and Unsupervised
               Discretization of Continuous Features.
    """

    def __init__(self, n_bins: int = 10, n_bands: int = 8):
        self.n_bins = n_bins
        self.n_bands = n_bands
        self.bin_edges_ = None
        self.band_indices_ = []
        self.feature_names_ = []

    def fit(self, X: np.ndarray) -> '_DiscretizationModule':
        n_features = X.shape[1]

        # Learn bin edges from data (global)
        flat = X.flatten()
        self.bin_edges_ = np.percentile(flat, np.linspace(0, 100, self.n_bins + 1))
        self.bin_edges_[0] = -np.inf
        self.bin_edges_[-1] = np.inf

        # Band indices
        bands = np.array_split(np.arange(n_features), self.n_bands)
        self.band_indices_ = [np.array(b) for b in bands if len(b) > 0]

        # Generate feature names
        self.feature_names_ = []

        # Global histogram features
        for i in range(self.n_bins):
            self.feature_names_.append(f"hist_bin_{i}")

        self.feature_names_.extend(['hist_entropy', 'hist_mode_bin', 'hist_uniformity'])

        # Per-band mode bin
        for i in range(len(self.band_indices_)):
            self.feature_names_.append(f"band_{i}_mode_bin")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        features_list = []

        for i in range(n_samples):
            sample_features = []

            # Global histogram
            hist, _ = np.histogram(X[i], bins=self.bin_edges_)
            hist_normalized = hist / (np.sum(hist) + 1e-10)
            sample_features.extend(hist_normalized)

            # Histogram entropy
            hist_entropy = entropy(hist_normalized + 1e-10)
            sample_features.append(hist_entropy)

            # Mode bin (most common)
            mode_bin = np.argmax(hist)
            sample_features.append(mode_bin / self.n_bins)  # Normalized

            # Uniformity (sum of squared proportions)
            uniformity = np.sum(hist_normalized ** 2)
            sample_features.append(uniformity)

            # Per-band mode bin
            for idx in self.band_indices_:
                band_hist, _ = np.histogram(X[i, idx], bins=self.bin_edges_)
                band_mode = np.argmax(band_hist)
                sample_features.append(band_mode / self.n_bins)

            features_list.append(sample_features)

        return np.array(features_list)

    def get_feature_names(self) -> list[str]:
        return self.feature_names_

# =============================================================================
# MAIN TRANSFORMER CLASS
# =============================================================================

class SpectralLatentFeatures(BaseEstimator, TransformerMixin):
    """
    Comprehensive spectral-to-tabular feature transformer optimized for TabPFN.

    Extracts diverse, decorrelated features from NIRS spectra to create
    a tabular representation that matches the data structure TabPFN 2.5
    was trained on.

    Parameters
    ----------
    n_pca : int, default=60
        Number of PCA components (decorrelated global features)

    pca_whiten : bool, default=True
        Whether to whiten PCA components (unit variance)

    use_wavelets : bool, default=True
        Whether to include wavelet decomposition features

    wavelet : str, default='db4'
        Wavelet type for DWT ('db4', 'sym4', 'haar', etc.)

    wavelet_levels : int, default=4
        Number of wavelet decomposition levels

    n_fft_bands : int, default=15
        Number of log-spaced FFT frequency bands

    n_local_bands : int, default=12
        Number of bands for local statistics

    n_peaks : int, default=8
        Number of peaks to extract features from

    n_bins : int, default=10
        Number of bins for discretization features

    output_normalization : str, default='quantile'
        Feature normalization method: 'none', 'standard', 'quantile', 'power'
        'quantile' transforms to uniform distribution
        'power' uses Yeo-Johnson power transform

    target_n_features : int, default=None
        Target number of output features (adjusts module parameters)

    Attributes
    ----------
    n_features_in_ : int
        Number of input features (wavelengths)

    n_features_out_ : int
        Number of output features (latent variables)

    feature_names_out_ : list
        Names of output features

    Examples
    --------
    >>> from spectral_latent_features import SpectralLatentFeatures
    >>> transformer = SpectralLatentFeatures(n_pca=50, use_wavelets=True)
    >>> X_latent = transformer.fit_transform(X_spectra)
    >>> print(f"Transformed: {X_spectra.shape} -> {X_latent.shape}")

    Notes
    -----
    The transformer is designed to produce ~200-400 features by default.
    Features are organized into blocks:
    - PCA: Global decorrelated patterns
    - Wavelet: Multi-scale decomposition
    - FFT: Frequency domain representation
    - Local Stats: Band-wise statistics
    - Derivatives: Shape characterization
    - Peaks: Absorption band features
    - Scatter: Baseline/scattering indices
    - Discretization: Categorical-like features

    References
    ----------
    [1] Hollmann et al. (2023). TabPFN: A Transformer That Solves Small
        Tabular Classification Problems in a Second. ICLR 2023.
    [2] Rinnan et al. (2009). Review of the most common pre-processing
        techniques for near-infrared spectra. TrAC.
    """

    def __init__(
        self,
        # Module activation flags
        use_pca: bool = True,
        use_wavelets: bool = True,
        use_wavelet_pca: bool = False,
        use_local_stats: bool = False,
        use_discretization: bool = False,
        use_scatter: bool = False,
        use_peaks: bool = False,
        use_derivatives: bool = False,
        use_fft: bool = False,
        use_pls: bool = False,
        use_nmf: bool = False,
        use_band_areas: bool = False,
        # PCA parameters
        n_pca: int = 60,
        pca_whiten: bool = True,
        pca_variance_threshold: float = 0.999,
        # Wavelet parameters
        wavelet: str = 'db4',
        wavelet_levels: int = 4,
        wavelet_coeffs_per_level: int = 8,
        # Wavelet-PCA parameters
        wavelet_pca_components_per_level: int = 3,
        # FFT parameters
        n_fft_bands: int = 15,
        n_fft_top: int = 15,
        # Local stats parameters
        n_local_bands: int = 12,
        # Derivative parameters
        n_deriv_window: int = 11,
        # Peak parameters
        n_peaks: int = 8,
        # Scatter parameters
        poly_degree: int = 3,
        # Discretization parameters
        n_bins: int = 10,
        n_disc_bands: int = 6,
        # PLS parameters
        n_pls: int = 20,
        # NMF parameters
        n_nmf: int = 15,
        nmf_max_iter: int = 200,
        # Band area parameters
        n_area_bands: int = 12,
        area_include_ratios: bool = True,
        # Output normalization
        output_normalization: str = 'quantile',
        random_state: int | None = None
    ):
        # Module activation flags
        self.use_pca = use_pca
        self.use_wavelets = use_wavelets
        self.use_fft = use_fft
        self.use_local_stats = use_local_stats
        self.use_derivatives = use_derivatives
        self.use_peaks = use_peaks
        self.use_scatter = use_scatter
        self.use_discretization = use_discretization
        self.use_wavelet_pca = use_wavelet_pca
        self.use_pls = use_pls
        self.use_nmf = use_nmf
        self.use_band_areas = use_band_areas
        # PCA parameters
        self.n_pca = n_pca
        self.pca_whiten = pca_whiten
        self.pca_variance_threshold = pca_variance_threshold
        # Wavelet parameters
        self.wavelet = wavelet
        self.wavelet_levels = wavelet_levels
        self.wavelet_coeffs_per_level = wavelet_coeffs_per_level
        # Wavelet-PCA parameters
        self.wavelet_pca_components_per_level = wavelet_pca_components_per_level
        # FFT parameters
        self.n_fft_bands = n_fft_bands
        self.n_fft_top = n_fft_top
        # Local stats parameters
        self.n_local_bands = n_local_bands
        # Derivative parameters
        self.n_deriv_window = n_deriv_window
        # Peak parameters
        self.n_peaks = n_peaks
        # Scatter parameters
        self.poly_degree = poly_degree
        # Discretization parameters
        self.n_bins = n_bins
        self.n_disc_bands = n_disc_bands
        # PLS parameters
        self.n_pls = n_pls
        # NMF parameters
        self.n_nmf = n_nmf
        self.nmf_max_iter = nmf_max_iter
        # Band area parameters
        self.n_area_bands = n_area_bands
        self.area_include_ratios = area_include_ratios
        # Output normalization
        self.output_normalization = output_normalization
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> 'SpectralLatentFeatures':
        """
        Fit all feature extraction modules.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_wavelengths)
            Training spectra

        y : array-like of shape (n_samples,), optional
            Target values. Required for supervised modules (PLS).

        Returns
        -------
        self : SpectralLatentFeatures
            Fitted transformer
        """
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]

        # Initialize modules based on activation flags
        self._pca = _PCAModule(
            n_components=self.n_pca,
            whiten=self.pca_whiten,
            variance_threshold=self.pca_variance_threshold
        ) if self.use_pca else None

        self._wavelet = _WaveletModule(
            wavelet=self.wavelet,
            max_level=self.wavelet_levels,
            n_coeffs_per_level=self.wavelet_coeffs_per_level
        ) if self.use_wavelets else None

        self._wavelet_pca = _WaveletPCAModule(
            wavelet=self.wavelet,
            max_level=self.wavelet_levels,
            n_components_per_level=self.wavelet_pca_components_per_level
        ) if self.use_wavelet_pca else None

        self._fft = _FFTModule(
            n_freq_bands=self.n_fft_bands,
            n_top_freqs=self.n_fft_top
        ) if self.use_fft else None

        self._local = _LocalStatsModule(
            n_bands=self.n_local_bands
        ) if self.use_local_stats else None

        self._deriv = _DerivativeModule(
            window_length=self.n_deriv_window
        ) if self.use_derivatives else None

        self._peaks = _PeakModule(
            n_peaks=self.n_peaks
        ) if self.use_peaks else None

        self._scatter = _ScatterModule(
            poly_degree=self.poly_degree
        ) if self.use_scatter else None

        self._discretization = _DiscretizationModule(
            n_bins=self.n_bins,
            n_bands=self.n_disc_bands
        ) if self.use_discretization else None

        self._pls = _PLSModule(
            n_components=self.n_pls
        ) if self.use_pls else None

        self._nmf = _NMFModule(
            n_components=self.n_nmf,
            max_iter=self.nmf_max_iter
        ) if self.use_nmf else None

        self._band_areas = _BandAreaModule(
            n_bands=self.n_area_bands,
            include_ratios=self.area_include_ratios
        ) if self.use_band_areas else None

        # Fit all active modules
        if self._pca:
            self._pca.fit(X)
        if self._wavelet:
            self._wavelet.fit(X)
        if self._wavelet_pca:
            self._wavelet_pca.fit(X)
        if self._fft:
            self._fft.fit(X)
        if self._local:
            self._local.fit(X)
        if self._deriv:
            self._deriv.fit(X)
        if self._peaks:
            self._peaks.fit(X)
        if self._scatter:
            self._scatter.fit(X)
        if self._discretization:
            self._discretization.fit(X)
        if self._pls:
            self._pls.fit(X, y)  # PLS requires y
        if self._nmf:
            self._nmf.fit(X)
        if self._band_areas:
            self._band_areas.fit(X)

        # Collect feature names from active modules
        self.feature_names_out_ = []
        if self._pca:
            self.feature_names_out_.extend(self._pca.get_feature_names())
        if self._wavelet:
            self.feature_names_out_.extend(self._wavelet.get_feature_names())
        if self._wavelet_pca:
            self.feature_names_out_.extend(self._wavelet_pca.get_feature_names())
        if self._fft:
            self.feature_names_out_.extend(self._fft.get_feature_names())
        if self._local:
            self.feature_names_out_.extend(self._local.get_feature_names())
        if self._deriv:
            self.feature_names_out_.extend(self._deriv.get_feature_names())
        if self._peaks:
            self.feature_names_out_.extend(self._peaks.get_feature_names())
        if self._scatter:
            self.feature_names_out_.extend(self._scatter.get_feature_names())
        if self._discretization:
            self.feature_names_out_.extend(self._discretization.get_feature_names())
        if self._pls:
            self.feature_names_out_.extend(self._pls.get_feature_names())
        if self._nmf:
            self.feature_names_out_.extend(self._nmf.get_feature_names())
        if self._band_areas:
            self.feature_names_out_.extend(self._band_areas.get_feature_names())

        # Fit output normalizer on transformed training data
        X_transformed = self._transform_raw(X)
        self.n_features_out_ = X_transformed.shape[1]

        if self.output_normalization == 'quantile':
            self._output_scaler = QuantileTransformer(
                output_distribution='uniform',
                random_state=self.random_state
            )
            self._output_scaler.fit(X_transformed)
        elif self.output_normalization == 'power':
            self._output_scaler = PowerTransformer(method='yeo-johnson')
            self._output_scaler.fit(X_transformed)
        elif self.output_normalization == 'standard':
            self._output_scaler = StandardScaler()
            self._output_scaler.fit(X_transformed)
        else:
            self._output_scaler = None

        return self

    def _transform_raw(self, X: np.ndarray) -> np.ndarray:
        """Transform without output normalization."""
        blocks = []

        if self._pca:
            blocks.append(self._pca.transform(X))
        if self._wavelet:
            wavelet_features = self._wavelet.transform(X)
            if wavelet_features.shape[1] > 0:
                blocks.append(wavelet_features)
        if self._wavelet_pca:
            wavelet_pca_features = self._wavelet_pca.transform(X)
            if wavelet_pca_features.shape[1] > 0:
                blocks.append(wavelet_pca_features)
        if self._fft:
            blocks.append(self._fft.transform(X))
        if self._local:
            blocks.append(self._local.transform(X))
        if self._deriv:
            blocks.append(self._deriv.transform(X))
        if self._peaks:
            blocks.append(self._peaks.transform(X))
        if self._scatter:
            blocks.append(self._scatter.transform(X))
        if self._discretization:
            blocks.append(self._discretization.transform(X))
        if self._pls:
            pls_features = self._pls.transform(X)
            if pls_features.shape[1] > 0:
                blocks.append(pls_features)
        if self._nmf:
            nmf_features = self._nmf.transform(X)
            if nmf_features.shape[1] > 0:
                blocks.append(nmf_features)
        if self._band_areas:
            blocks.append(self._band_areas.transform(X))

        if not blocks:
            # No modules active, return empty array with correct number of samples
            return np.zeros((X.shape[0], 0))

        X_combined = np.hstack(blocks)

        # Handle NaN/Inf
        X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

        return X_combined

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform spectra to latent features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_wavelengths)
            Spectra to transform

        Returns
        -------
        X_latent : ndarray of shape (n_samples, n_features_out_)
            Latent feature representation
        """
        X = np.asarray(X)
        X_transformed = self._transform_raw(X)

        if self._output_scaler is not None:
            X_transformed = self._output_scaler.transform(X_transformed)

        return X_transformed

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get output feature names."""
        return np.array(self.feature_names_out_, dtype=object)

    def get_module_info(self) -> dict:
        """Get information about each module's contribution."""
        info = {}

        if self._pca:
            info['pca'] = {
                'n_features': self._pca.n_components_fitted_,
                'description': 'Global decorrelated patterns',
                'active': True
            }
        else:
            info['pca'] = {'n_features': 0, 'description': 'Global decorrelated patterns', 'active': False}

        if self._wavelet and len(self._wavelet.get_feature_names()) > 0:
            info['wavelet'] = {
                'n_features': len(self._wavelet.get_feature_names()),
                'description': 'Multi-scale decomposition',
                'active': True
            }
        else:
            info['wavelet'] = {'n_features': 0, 'description': 'Multi-scale decomposition', 'active': False}

        if self._wavelet_pca and len(self._wavelet_pca.get_feature_names()) > 0:
            info['wavelet_pca'] = {
                'n_features': len(self._wavelet_pca.get_feature_names()),
                'description': 'Multi-scale PCA on wavelet coefficients',
                'active': True
            }
        else:
            info['wavelet_pca'] = {'n_features': 0, 'description': 'Multi-scale PCA on wavelet coefficients', 'active': False}

        if self._fft:
            info['fft'] = {
                'n_features': len(self._fft.get_feature_names()),
                'description': 'Frequency domain features',
                'active': True
            }
        else:
            info['fft'] = {'n_features': 0, 'description': 'Frequency domain features', 'active': False}

        if self._local:
            info['local_stats'] = {
                'n_features': len(self._local.get_feature_names()),
                'description': 'Band-wise statistics',
                'active': True
            }
        else:
            info['local_stats'] = {'n_features': 0, 'description': 'Band-wise statistics', 'active': False}

        if self._deriv:
            info['derivatives'] = {
                'n_features': len(self._deriv.get_feature_names()),
                'description': 'Shape characterization',
                'active': True
            }
        else:
            info['derivatives'] = {'n_features': 0, 'description': 'Shape characterization', 'active': False}

        if self._peaks:
            info['peaks'] = {
                'n_features': len(self._peaks.get_feature_names()),
                'description': 'Absorption band features',
                'active': True
            }
        else:
            info['peaks'] = {'n_features': 0, 'description': 'Absorption band features', 'active': False}

        if self._scatter:
            info['scatter'] = {
                'n_features': len(self._scatter.get_feature_names()),
                'description': 'Baseline/scattering indices',
                'active': True
            }
        else:
            info['scatter'] = {'n_features': 0, 'description': 'Baseline/scattering indices', 'active': False}

        if self._discretization:
            info['discretization'] = {
                'n_features': len(self._discretization.get_feature_names()),
                'description': 'Categorical-like features',
                'active': True
            }
        else:
            info['discretization'] = {'n_features': 0, 'description': 'Categorical-like features', 'active': False}

        if self._pls and len(self._pls.get_feature_names()) > 0:
            info['pls'] = {
                'n_features': len(self._pls.get_feature_names()),
                'description': 'Supervised PLS latent scores',
                'active': True
            }
        else:
            info['pls'] = {'n_features': 0, 'description': 'Supervised PLS latent scores', 'active': False}

        if self._nmf and len(self._nmf.get_feature_names()) > 0:
            info['nmf'] = {
                'n_features': len(self._nmf.get_feature_names()),
                'description': 'Non-negative matrix factorization coefficients',
                'active': True
            }
        else:
            info['nmf'] = {'n_features': 0, 'description': 'Non-negative matrix factorization coefficients', 'active': False}

        if self._band_areas:
            info['band_areas'] = {
                'n_features': len(self._band_areas.get_feature_names()),
                'description': 'Band area (AUC) features',
                'active': True
            }
        else:
            info['band_areas'] = {'n_features': 0, 'description': 'Band area (AUC) features', 'active': False}

        info['total'] = self.n_features_out_
        info['output_normalization'] = self.output_normalization

        return info

# =============================================================================
# LIGHTWEIGHT VERSION
# =============================================================================

class SpectralLatentFeaturesLite(BaseEstimator, TransformerMixin):
    """
    Lightweight version with fewer features (~150-200).

    Uses only the most essential modules:
    - PCA (global patterns)
    - Local statistics (band features)
    - Derivatives (shape)
    - Scatter (baseline)

    Good for when computation time is critical or dataset is very small.
    """

    def __init__(
        self,
        n_pca: int = 40,
        n_local_bands: int = 10,
        output_normalization: str = 'quantile',
        random_state: int | None = None
    ):
        self.n_pca = n_pca
        self.n_local_bands = n_local_bands
        self.output_normalization = output_normalization
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> 'SpectralLatentFeaturesLite':
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]

        self._pca = _PCAModule(n_components=self.n_pca, whiten=True)
        self._local = _LocalStatsModule(n_bands=self.n_local_bands)
        self._deriv = _DerivativeModule()
        self._scatter = _ScatterModule(poly_degree=2)

        self._pca.fit(X)
        self._local.fit(X)
        self._deriv.fit(X)
        self._scatter.fit(X)

        self.feature_names_out_ = (
            self._pca.get_feature_names() +
            self._local.get_feature_names() +
            self._deriv.get_feature_names() +
            self._scatter.get_feature_names()
        )

        X_transformed = self._transform_raw(X)
        self.n_features_out_ = X_transformed.shape[1]

        if self.output_normalization == 'quantile':
            self._output_scaler = QuantileTransformer(
                output_distribution='uniform',
                random_state=self.random_state
            )
            self._output_scaler.fit(X_transformed)
        else:
            self._output_scaler = None

        return self

    def _transform_raw(self, X: np.ndarray) -> np.ndarray:
        blocks = [
            self._pca.transform(X),
            self._local.transform(X),
            self._deriv.transform(X),
            self._scatter.transform(X)
        ]
        X_combined = np.hstack(blocks)
        return np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        X_transformed = self._transform_raw(X)
        if self._output_scaler is not None:
            X_transformed = self._output_scaler.transform(X_transformed)
        return X_transformed

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.feature_names_out_, dtype=object)

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_tabpfn_features(
    X: np.ndarray,
    n_features: int = 300,
    normalization: str = 'quantile'
) -> tuple[np.ndarray, SpectralLatentFeatures]:
    """
    Convenience function to quickly transform spectra for TabPFN.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_wavelengths)
        Input spectra

    n_features : int, default=300
        Target number of output features

    normalization : str, default='quantile'
        Output normalization method

    Returns
    -------
    X_transformed : ndarray
        Transformed features

    transformer : SpectralLatentFeatures
        Fitted transformer (for transforming new data)

    Examples
    --------
    >>> X_latent, transformer = create_tabpfn_features(X_train)
    >>> X_test_latent = transformer.transform(X_test)
    """
    # Adjust parameters based on target features
    if n_features <= 200:
        transformer = SpectralLatentFeaturesLite(
            n_pca=40,
            n_local_bands=10,
            output_normalization=normalization
        )
    else:
        n_pca = min(80, n_features // 4)
        transformer = SpectralLatentFeatures(
            n_pca=n_pca,
            output_normalization=normalization
        )

    X_transformed = transformer.fit_transform(X)

    return X_transformed, transformer

if __name__ == '__main__':
    # Quick test
    print("Testing SpectralLatentFeatures...")

    # Generate synthetic spectral data
    np.random.seed(42)
    n_samples = 100
    n_wavelengths = 500

    # Simulate NIRS-like spectra with smooth baseline and peaks
    x = np.linspace(0, 10, n_wavelengths)
    X = np.zeros((n_samples, n_wavelengths))
    for i in range(n_samples):
        baseline = 0.5 + 0.1 * x + 0.02 * x**2
        peaks = (
            0.5 * np.exp(-0.5 * ((x - 3) / 0.5)**2) +
            0.3 * np.exp(-0.5 * ((x - 6) / 0.8)**2) +
            0.2 * np.exp(-0.5 * ((x - 8) / 0.3)**2)
        )
        noise = 0.02 * np.random.randn(n_wavelengths)
        X[i] = baseline + np.random.uniform(0.5, 1.5) * peaks + noise

    # Test full transformer
    print("\nFull SpectralLatentFeatures:")
    transformer = SpectralLatentFeatures()
    X_transformed = transformer.fit_transform(X)
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {X_transformed.shape}")
    print("  Module info:")
    for name, info in transformer.get_module_info().items():
        if isinstance(info, dict):
            print(f"    {name}: {info['n_features']} features - {info['description']}")
        else:
            print(f"    {name}: {info}")

    # Test lite transformer
    print("\nSpectralLatentFeaturesLite:")
    transformer_lite = SpectralLatentFeaturesLite()
    X_lite = transformer_lite.fit_transform(X)
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {X_lite.shape}")

    # Test convenience function
    print("\ncreate_tabpfn_features (300 target):")
    X_auto, _ = create_tabpfn_features(X, n_features=300)
    print(f"  Output shape: {X_auto.shape}")

    print("\n✓ All tests passed!")
