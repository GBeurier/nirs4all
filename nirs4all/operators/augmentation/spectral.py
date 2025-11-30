import numpy as np
from scipy import signal, interpolate
from scipy.ndimage import convolve1d
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Union, Tuple, List

from .abc_augmenter import Augmenter

# --- Utility Functions ---


def _get_gaussian_kernel(sigma: float, width: int) -> np.ndarray:
    """Generates a 1D Gaussian kernel."""
    if width % 2 == 0:
        width += 1  # Ensure odd width
    x = np.arange(-(width // 2), (width // 2) + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


def _convolve_1d_batch(X: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolves all rows of a 2D array with a kernel using reflection padding."""
    return convolve1d(X, kernel, axis=1, mode='reflect')


def _convolve_1d(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolves a 1D signal with a kernel using reflection padding."""
    pad_size = len(kernel) // 2
    x_padded = np.pad(x, pad_size, mode='reflect')
    return signal.convolve(x_padded, kernel, mode='valid')


def _safe_interp(x_new: np.ndarray, x_old: np.ndarray, y_old: np.ndarray) -> np.ndarray:
    """Safe 1D interpolation."""
    return np.interp(x_new, x_old, y_old)

# --- 2.1 Additive / Multiplicative Noise ---

class GaussianAdditiveNoise(Augmenter):
    """
    Adds Gaussian noise to the spectra.
    X_aug = X + noise

    Vectorized implementation using batch convolution.
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 sigma: float = 0.01, smoothing_kernel_width: int = 1):
        super().__init__(apply_on, random_state, copy=copy)
        self.sigma = sigma
        self.smoothing_kernel_width = smoothing_kernel_width

    def augment(self, X, apply_on="samples"):
        n_samples, n_features = X.shape

        # Generate noise
        if apply_on == "global":
            # Global std dev
            scale = np.std(X) * self.sigma
            noise = self.random_gen.normal(0, scale, size=X.shape)
        else:
            # Per-sample std dev
            stds = np.std(X, axis=1, keepdims=True)
            noise = self.random_gen.normal(0, 1, size=X.shape) * stds * self.sigma

        # Smooth noise if requested - vectorized batch processing
        if self.smoothing_kernel_width > 1:
            kernel_sigma = self.smoothing_kernel_width / 6.0
            kernel = _get_gaussian_kernel(kernel_sigma, self.smoothing_kernel_width)
            # Batch convolution using scipy.ndimage.convolve1d
            noise = _convolve_1d_batch(noise, kernel)

        return X + noise


class MultiplicativeNoise(Augmenter):
    """
    Multiplies spectra by a random gain factor.
    X_aug = (1 + epsilon) * X
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 sigma_gain: float = 0.05, per_wavelength: bool = False):
        super().__init__(apply_on, random_state, copy=copy)
        self.sigma_gain = sigma_gain
        self.per_wavelength = per_wavelength

    def augment(self, X, apply_on="samples"):
        n_samples, n_features = X.shape

        if self.per_wavelength:
            epsilon = self.random_gen.normal(0, self.sigma_gain, size=X.shape)
        else:
            if apply_on == "global":
                 # One gain for all? Or one gain per sample?
                 # Usually multiplicative noise is per sample or per wavelength.
                 # If apply_on="global", maybe one gain for the whole dataset?
                 # Let's stick to per-sample gain as default behavior for "samples"
                 # and single gain for "global" (though less useful).
                 epsilon = self.random_gen.normal(0, self.sigma_gain)
            else:
                epsilon = self.random_gen.normal(0, self.sigma_gain, size=(n_samples, 1))

        return X * (1 + epsilon)


# --- 2.2 Baseline Shifts and Drifts ---

class LinearBaselineDrift(Augmenter):
    """
    Adds a linear baseline drift.
    X_aug = X + a + b * lambda
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 offset_range: Tuple[float, float] = (-0.1, 0.1),
                 slope_range: Tuple[float, float] = (-0.001, 0.001),
                 lambda_axis: Optional[np.ndarray] = None):
        super().__init__(apply_on, random_state, copy=copy)
        self.offset_range = offset_range
        self.slope_range = slope_range
        self.lambda_axis = lambda_axis

    def augment(self, X, apply_on="samples"):
        n_samples, n_features = X.shape

        if self.lambda_axis is None:
            lambdas = np.arange(n_features)
        else:
            lambdas = self.lambda_axis

        # Center lambdas to avoid correlation between slope and offset
        lambdas_centered = lambdas - np.mean(lambdas)

        offsets = self.random_gen.uniform(self.offset_range[0], self.offset_range[1], size=(n_samples, 1))
        slopes = self.random_gen.uniform(self.slope_range[0], self.slope_range[1], size=(n_samples, 1))

        drift = offsets + slopes * lambdas_centered.reshape(1, -1)
        return X + drift


class PolynomialBaselineDrift(Augmenter):
    """
    Adds a polynomial baseline drift.
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 degree: int = 3,
                 coeff_ranges: Optional[List[Tuple[float, float]]] = None,
                 lambda_axis: Optional[np.ndarray] = None):
        super().__init__(apply_on, random_state, copy=copy)
        self.degree = degree
        self.coeff_ranges = coeff_ranges
        self.lambda_axis = lambda_axis

    def augment(self, X, apply_on="samples"):
        n_samples, n_features = X.shape

        if self.lambda_axis is None:
            lambdas = np.linspace(-1, 1, n_features) # Normalized for stability
        else:
            # Normalize lambda axis to [-1, 1] for polynomial stability
            l_min, l_max = np.min(self.lambda_axis), np.max(self.lambda_axis)
            lambdas = 2 * (self.lambda_axis - l_min) / (l_max - l_min) - 1

        drift = np.zeros_like(X)

        # Default ranges if not provided
        ranges = self.coeff_ranges
        if ranges is None:
            # Decaying ranges for higher orders
            ranges = [(-0.1 / (i+1), 0.1 / (i+1)) for i in range(self.degree + 1)]

        for i in range(self.degree + 1):
            coeffs = self.random_gen.uniform(ranges[i][0], ranges[i][1], size=(n_samples, 1))
            term = coeffs * (lambdas.reshape(1, -1) ** i)
            drift += term

        return X + drift


# --- 2.3 Wavelength Axis Distortions ---

class WavelengthShift(Augmenter):
    """
    Shifts the wavelength axis.

    Vectorized implementation using batch interpolation.
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 shift_range: Tuple[float, float] = (-2.0, 2.0),
                 lambda_axis: Optional[np.ndarray] = None):
        super().__init__(apply_on, random_state, copy=copy)
        self.shift_range = shift_range
        self.lambda_axis = lambda_axis

    def augment(self, X, apply_on="samples"):
        n_samples, n_features = X.shape

        if self.lambda_axis is None:
            lambdas = np.arange(n_features, dtype=float)
        else:
            lambdas = self.lambda_axis.astype(float)

        # Generate all shifts at once
        shifts = self.random_gen.uniform(self.shift_range[0], self.shift_range[1], size=n_samples)

        # Vectorized interpolation using broadcasting
        # Create query coordinates for all samples at once
        # lambdas: (n_features,), shifts: (n_samples,)
        query_lambdas = lambdas[np.newaxis, :] - shifts[:, np.newaxis]  # (n_samples, n_features)

        # Batch interpolation - use np.interp in a vectorized manner
        # For each sample, interpolate at shifted wavelengths
        X_aug = np.empty_like(X)
        # Use np.apply_along_axis for cleaner code, or loop with pre-allocated array
        for i in range(n_samples):
            X_aug[i] = np.interp(query_lambdas[i], lambdas, X[i])

        return X_aug


class WavelengthStretch(Augmenter):
    """
    Stretches or compresses the wavelength axis.

    Vectorized implementation using batch interpolation.
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 stretch_range: Tuple[float, float] = (0.99, 1.01),
                 lambda_axis: Optional[np.ndarray] = None):
        super().__init__(apply_on, random_state, copy=copy)
        self.stretch_range = stretch_range
        self.lambda_axis = lambda_axis

    def augment(self, X, apply_on="samples"):
        n_samples, n_features = X.shape

        if self.lambda_axis is None:
            lambdas = np.arange(n_features, dtype=float)
        else:
            lambdas = self.lambda_axis.astype(float)

        center_lambda = np.mean(lambdas)

        # Generate all stretch factors at once
        factors = self.random_gen.uniform(self.stretch_range[0], self.stretch_range[1], size=n_samples)

        # Vectorized computation of query coordinates
        # l_query = center + (lambda - center) / factor
        lambdas_centered = lambdas - center_lambda  # (n_features,)
        query_lambdas = center_lambda + lambdas_centered[np.newaxis, :] / factors[:, np.newaxis]  # (n_samples, n_features)

        # Batch interpolation
        X_aug = np.empty_like(X)
        for i in range(n_samples):
            X_aug[i] = np.interp(query_lambdas[i], lambdas, X[i])

        return X_aug


class LocalWavelengthWarp(Augmenter):
    """
    Applies a non-linear warp to the wavelength axis.

    Optimized implementation with pre-computed control points.
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 n_control_points: int = 5,
                 max_shift: float = 1.0,
                 lambda_axis: Optional[np.ndarray] = None):
        super().__init__(apply_on, random_state, copy=copy)
        self.n_control_points = n_control_points
        self.max_shift = max_shift
        self.lambda_axis = lambda_axis

    def augment(self, X, apply_on="samples"):
        n_samples, n_features = X.shape

        if self.lambda_axis is None:
            lambdas = np.arange(n_features, dtype=float)
        else:
            lambdas = self.lambda_axis.astype(float)

        X_aug = np.empty_like(X)

        # Control points evenly spaced - pre-compute once
        ctrl_x = np.linspace(lambdas[0], lambdas[-1], self.n_control_points)

        # Generate all random shifts at once for all samples
        all_ctrl_shifts = self.random_gen.uniform(
            -self.max_shift, self.max_shift,
            size=(n_samples, self.n_control_points)
        )

        # Determine spline degree
        k = 3 if self.n_control_points > 3 else 1

        for i in range(n_samples):
            # Use pre-generated shifts
            ctrl_y_shifts = all_ctrl_shifts[i]

            # Interpolate shifts to get shift for every lambda using cubic spline
            tck = interpolate.splrep(ctrl_x, ctrl_y_shifts, s=0, k=k)
            shifts = interpolate.splev(lambdas, tck)

            # Apply warp: f(l - shift(l))
            X_aug[i] = np.interp(lambdas - shifts, lambdas, X[i])

        return X_aug


class SmoothMagnitudeWarp(Augmenter):
    """
    Multiplies the spectrum by a smooth curve.

    Optimized implementation with pre-computed control points.
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 n_control_points: int = 5,
                 gain_range: Tuple[float, float] = (0.9, 1.1),
                 lambda_axis: Optional[np.ndarray] = None):
        super().__init__(apply_on, random_state, copy=copy)
        self.n_control_points = n_control_points
        self.gain_range = gain_range
        self.lambda_axis = lambda_axis

    def augment(self, X, apply_on="samples"):
        n_samples, n_features = X.shape

        if self.lambda_axis is None:
            lambdas = np.arange(n_features, dtype=float)
        else:
            lambdas = self.lambda_axis.astype(float)

        X_aug = np.empty_like(X)
        ctrl_x = np.linspace(lambdas[0], lambdas[-1], self.n_control_points)

        # Generate all random gains at once for all samples
        all_ctrl_gains = self.random_gen.uniform(
            self.gain_range[0], self.gain_range[1],
            size=(n_samples, self.n_control_points)
        )

        # Determine spline degree
        k = 3 if self.n_control_points > 3 else 1

        for i in range(n_samples):
            ctrl_gains = all_ctrl_gains[i]
            tck = interpolate.splrep(ctrl_x, ctrl_gains, s=0, k=k)
            gains = interpolate.splev(lambdas, tck)
            X_aug[i] = X[i] * gains

        return X_aug


class BandPerturbation(Augmenter):
    """
    Perturbs specific bands of the spectrum.

    Optimized with pre-generated random parameters.
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 n_bands: int = 3,
                 bandwidth_range: Tuple[int, int] = (5, 20),
                 gain_range: Tuple[float, float] = (0.9, 1.1),
                 offset_range: Tuple[float, float] = (-0.01, 0.01)):
        super().__init__(apply_on, random_state, copy=copy)
        self.n_bands = n_bands
        self.bandwidth_range = bandwidth_range
        self.gain_range = gain_range
        self.offset_range = offset_range

    def augment(self, X, apply_on="samples"):
        n_samples, n_features = X.shape
        X_aug = X.copy()

        # Pre-generate all random parameters for all samples and bands
        centers = self.random_gen.integers(0, n_features, size=(n_samples, self.n_bands))
        widths = self.random_gen.integers(
            self.bandwidth_range[0], self.bandwidth_range[1], size=(n_samples, self.n_bands)
        )
        gains = self.random_gen.uniform(
            self.gain_range[0], self.gain_range[1], size=(n_samples, self.n_bands)
        )
        offsets = self.random_gen.uniform(
            self.offset_range[0], self.offset_range[1], size=(n_samples, self.n_bands)
        )

        for i in range(n_samples):
            for b in range(self.n_bands):
                center = centers[i, b]
                width = widths[i, b]
                start = max(0, center - width // 2)
                end = min(n_features, center + width // 2)

                if start >= end:
                    continue

                X_aug[i, start:end] = X_aug[i, start:end] * gains[i, b] + offsets[i, b]

        return X_aug


# --- 2.5 Resolution / Smoothing Jitter ---

class GaussianSmoothingJitter(Augmenter):
    """
    Applies Gaussian smoothing with random sigma.

    Optimized with pre-generated random parameters.
    Note: Due to per-sample kernel requirements, this still uses a loop
    but with pre-generated random values.
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 sigma_range: Tuple[float, float] = (0.5, 2.0),
                 kernel_width: int = 11):
        super().__init__(apply_on, random_state, copy=copy)
        self.sigma_range = sigma_range
        self.kernel_width = kernel_width

    def augment(self, X, apply_on="samples"):
        n_samples, n_features = X.shape
        X_aug = np.empty_like(X)

        # Pre-generate all sigma values
        sigmas = self.random_gen.uniform(self.sigma_range[0], self.sigma_range[1], size=n_samples)

        for i in range(n_samples):
            kernel = _get_gaussian_kernel(sigmas[i], self.kernel_width)
            X_aug[i] = _convolve_1d(X[i], kernel)

        return X_aug


class UnsharpSpectralMask(Augmenter):
    """
    Applies unsharp masking (sharpening).
    X_aug = X + k * (X - smooth(X))

    Vectorized implementation using batch convolution.
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 amount_range: Tuple[float, float] = (0.1, 0.5),
                 sigma: float = 1.0,
                 kernel_width: int = 11):
        super().__init__(apply_on, random_state, copy=copy)
        self.amount_range = amount_range
        self.sigma = sigma
        self.kernel_width = kernel_width

    def augment(self, X, apply_on="samples"):
        n_samples, n_features = X.shape

        # Pre-compute kernel once
        kernel = _get_gaussian_kernel(self.sigma, self.kernel_width)

        # Batch smoothing using vectorized convolution
        smoothed = _convolve_1d_batch(X, kernel)

        # Generate all amounts at once
        amounts = self.random_gen.uniform(
            self.amount_range[0], self.amount_range[1], size=(n_samples, 1)
        )

        # Vectorized unsharp mask computation
        X_aug = X + amounts * (X - smoothed)

        return X_aug


# --- 2.6 Spectral Masking and Dropout ---

class BandMasking(Augmenter):
    """
    Masks out bands of the spectrum.

    Optimized with pre-generated random parameters.
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 n_bands_range: Tuple[int, int] = (1, 3),
                 bandwidth_range: Tuple[int, int] = (5, 20),
                 mode: str = "interp"):  # "zero" or "interp"
        super().__init__(apply_on, random_state, copy=copy)
        self.n_bands_range = n_bands_range
        self.bandwidth_range = bandwidth_range
        self.mode = mode

    def augment(self, X, apply_on="samples"):
        n_samples, n_features = X.shape
        X_aug = X.copy()

        # Pre-generate number of bands per sample
        n_bands_per_sample = self.random_gen.integers(
            self.n_bands_range[0], self.n_bands_range[1] + 1, size=n_samples
        )
        max_bands = self.n_bands_range[1]

        # Pre-generate all random parameters for max possible bands
        centers = self.random_gen.integers(0, n_features, size=(n_samples, max_bands))
        widths = self.random_gen.integers(
            self.bandwidth_range[0], self.bandwidth_range[1], size=(n_samples, max_bands)
        )

        for i in range(n_samples):
            for b in range(n_bands_per_sample[i]):
                center = centers[i, b]
                width = widths[i, b]

                start = max(0, center - width // 2)
                end = min(n_features, center + width // 2)

                if start >= end:
                    continue

                if self.mode == "zero":
                    X_aug[i, start:end] = 0
                elif self.mode == "interp":
                    # Linear interpolation between start-1 and end
                    val_start = X_aug[i, start - 1] if start > 0 else X_aug[i, start]
                    val_end = X_aug[i, end] if end < n_features else X_aug[i, end - 1]

                    # Create line
                    x_local = np.arange(end - start)
                    slope = (val_end - val_start) / (end - start + 1) if (end - start + 1) > 0 else 0
                    X_aug[i, start:end] = val_start + slope * (x_local + 1)

        return X_aug


class ChannelDropout(Augmenter):
    """
    Drops individual wavelengths (sets to zero or interpolates).

    Optimized with vectorized mask generation.
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 dropout_prob: float = 0.01,
                 mode: str = "interp"):
        super().__init__(apply_on, random_state, copy=copy)
        self.dropout_prob = dropout_prob
        self.mode = mode

    def augment(self, X, apply_on="samples"):
        n_samples, n_features = X.shape
        X_aug = X.copy()

        # Vectorized mask generation
        mask = self.random_gen.random(size=X.shape) < self.dropout_prob

        if self.mode == "zero":
            X_aug[mask] = 0
        elif self.mode == "interp":
            # For each sample, find dropped indices and interpolate
            for i in range(n_samples):
                dropped_indices = np.where(mask[i])[0]
                if len(dropped_indices) == 0:
                    continue

                kept_indices = np.where(~mask[i])[0]
                if len(kept_indices) == 0:
                    continue  # All dropped, can't interpolate

                X_aug[i, dropped_indices] = np.interp(dropped_indices, kept_indices, X[i, kept_indices])

        return X_aug


# --- 2.7 Rare Structured Artefacts ---

class SpikeNoise(Augmenter):
    """
    Adds spikes to the spectrum.

    Optimized with pre-generated random parameters.
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 n_spikes_range: Tuple[int, int] = (1, 3),
                 amplitude_range: Tuple[float, float] = (-0.5, 0.5)):
        super().__init__(apply_on, random_state, copy=copy)
        self.n_spikes_range = n_spikes_range
        self.amplitude_range = amplitude_range

    def augment(self, X, apply_on="samples"):
        n_samples, n_features = X.shape
        X_aug = X.copy()

        # Pre-generate number of spikes per sample
        n_spikes_per_sample = self.random_gen.integers(
            self.n_spikes_range[0], self.n_spikes_range[1] + 1, size=n_samples
        )
        max_spikes = self.n_spikes_range[1]

        # Pre-generate all spike parameters for maximum possible spikes
        all_indices = self.random_gen.integers(0, n_features, size=(n_samples, max_spikes))
        all_amplitudes = self.random_gen.uniform(
            self.amplitude_range[0], self.amplitude_range[1], size=(n_samples, max_spikes)
        )

        for i in range(n_samples):
            n_spikes = n_spikes_per_sample[i]
            # Use pre-generated values but only first n_spikes
            # Need to ensure unique indices - use first n_spikes and ensure uniqueness
            indices = np.unique(all_indices[i, :n_spikes])
            amplitudes = all_amplitudes[i, :len(indices)]
            X_aug[i, indices] += amplitudes

        return X_aug


class LocalClipping(Augmenter):
    """
    Clips values in a local region to simulate saturation.

    Optimized with pre-generated random parameters.
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 n_regions: int = 1,
                 width_range: Tuple[int, int] = (5, 20)):
        super().__init__(apply_on, random_state, copy=copy)
        self.n_regions = n_regions
        self.width_range = width_range

    def augment(self, X, apply_on="samples"):
        n_samples, n_features = X.shape
        X_aug = X.copy()

        # Pre-generate all random parameters
        centers = self.random_gen.integers(0, n_features, size=(n_samples, self.n_regions))
        widths = self.random_gen.integers(
            self.width_range[0], self.width_range[1], size=(n_samples, self.n_regions)
        )

        for i in range(n_samples):
            for r in range(self.n_regions):
                center = centers[i, r]
                width = widths[i, r]

                start = max(0, center - width // 2)
                end = min(n_features, center + width // 2)

                if start >= end:
                    continue

                # Clip to the 90th percentile of the segment (flattening peaks)
                segment = X_aug[i, start:end]
                limit = np.percentile(segment, 90)
                X_aug[i, start:end] = np.minimum(segment, limit)

        return X_aug


# --- 2.8 Sample Combinations ---

class MixupAugmenter(Augmenter):
    """
    Mixup augmentation.
    Note: This modifies both X and y.
    Standard transform() only returns X.
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 alpha: float = 0.2):
        super().__init__(apply_on, random_state, copy=copy)
        self.alpha = alpha

    def augment(self, X, apply_on="samples"):
        # Without y, we can only mix X.
        # If y is needed, this class needs to be used differently than standard Augmenter.
        # For now, implementing X mixing.
        n_samples = X.shape[0]
        indices = self.random_gen.permutation(n_samples)

        lam = self.random_gen.beta(self.alpha, self.alpha, size=(n_samples, 1))

        X_aug = lam * X + (1 - lam) * X[indices]
        return X_aug


class LocalMixupAugmenter(Augmenter):
    """
    Mixup with nearest neighbors.
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 alpha: float = 0.2, k_neighbors: int = 5):
        super().__init__(apply_on, random_state, copy=copy)
        self.alpha = alpha
        self.k_neighbors = k_neighbors
        self.X_fit_ = None

    def fit(self, X, y=None):
        self.X_fit_ = X
        return self

    def augment(self, X, apply_on="samples"):
        # If fit was called, use X_fit_ to find neighbors.
        # If not, use X itself.
        reference_X = self.X_fit_ if self.X_fit_ is not None else X

        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(reference_X)
        distances, indices = nn.kneighbors(X)

        n_samples = X.shape[0]
        X_aug = np.zeros_like(X)

        for i in range(n_samples):
            # Pick a random neighbor (excluding self which is usually index 0 in results)
            neighbor_idx = self.random_gen.choice(indices[i, 1:])
            neighbor = reference_X[neighbor_idx]

            lam = self.random_gen.beta(self.alpha, self.alpha)
            X_aug[i] = lam * X[i] + (1 - lam) * neighbor

        return X_aug


# --- 2.9 Scattering-based Simulation ---

class ScatterSimulationMSC(Augmenter):
    """
    Simulates scatter variation: x_aug = a + b * x
    """
    def __init__(self, apply_on="samples", random_state=None, *, copy=True,
                 reference_mode: str = "self", # "self", "global_mean"
                 a_range: Tuple[float, float] = (-0.1, 0.1),
                 b_range: Tuple[float, float] = (0.9, 1.1)):
        super().__init__(apply_on, random_state, copy=copy)
        self.reference_mode = reference_mode
        self.a_range = a_range
        self.b_range = b_range
        self.global_mean_ = None

    def fit(self, X, y=None):
        if self.reference_mode == "global_mean":
            self.global_mean_ = np.mean(X, axis=0)
        return self

    def augment(self, X, apply_on="samples"):
        n_samples = X.shape[0]

        a = self.random_gen.uniform(self.a_range[0], self.a_range[1], size=(n_samples, 1))
        b = self.random_gen.uniform(self.b_range[0], self.b_range[1], size=(n_samples, 1))

        if self.reference_mode == "global_mean" and self.global_mean_ is not None:
            # This mode is tricky. Usually MSC corrects X to match Ref.
            # Here we want to simulate scatter, so we take Ref and apply scatter?
            # Or take X (which is assumed to be Ref-like) and apply scatter?
            # "Simulate scatter variation by perturbing a, b in x ~ a + b * x_ref"
            # If X is the input, and we want to add scatter, we can treat X as the "ideal" and add scatter.
            # So X_aug = a + b * X.
            # This is same as "self" mode effectively if we treat X as reference.
            # If reference_mode is global_mean, maybe we assume X is close to mean and we want to deviate it?
            # Let's stick to X_aug = a + b * X for simplicity as it matches the formula structure.
            pass

        # Apply: X_aug = a + b * X
        return a + b * X
