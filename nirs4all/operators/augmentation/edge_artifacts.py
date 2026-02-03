"""
Edge artifacts and spectral boundary effects augmentation operators.

This module provides wavelength-aware augmentation operators that simulate
edge-related artifacts commonly observed in NIR spectra, including:

- Detector sensitivity roll-off at wavelength boundaries
- Stray light effects (more pronounced at spectral edges)
- Edge curvature/bending due to optical aberrations

These artifacts often manifest as deformations at the start or end of spectra
and can significantly impact model performance if not accounted for.

References:
    - Workman, J., & Weyer, L. (2012). Practical Guide and Spectral Atlas for
      Interpretive Near-Infrared Spectroscopy. CRC Press.
    - Burns, D. A., & Ciurczak, E. W. (2007). Handbook of Near-Infrared Analysis.
      CRC Press. Chapter on instrumental considerations.
    - Chalmers, J. M., & Griffiths, P. R. (2001). Mid-Infrared Spectroscopy:
      Anomalies, Artifacts and Common Errors. Wiley.
    - JASCO (2020). Advantages of high-sensitivity InGaAs detector in UV-Vis/NIR
      spectrophotometer. Technical Note.
    - Applied Optics (1975). Resolution and stray light in near infrared
      spectroscopy, 14(8), 1977.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..base import SpectraTransformerMixin


# =============================================================================
# Detector Response Parameters (literature-based)
# =============================================================================

@dataclass
class DetectorModel:
    """
    Parameters for detector sensitivity roll-off modeling.

    Detector sensitivity curves typically follow a profile where sensitivity
    peaks in the middle of the spectral range and rolls off at the edges,
    often following an exponential decay pattern.

    Attributes:
        name: Detector type name.
        optimal_range: Tuple of (min_nm, max_nm) for optimal sensitivity.
        roll_off_rate: Rate of exponential decay at edges (nm^-1).
        min_sensitivity: Minimum relative sensitivity at extreme edges.
    """
    name: str
    optimal_range: Tuple[float, float]
    roll_off_rate: float
    min_sensitivity: float


# Common NIR detector models with their sensitivity characteristics
# Based on manufacturer specifications and literature
DETECTOR_MODELS: Dict[str, DetectorModel] = {
    "ingaas_standard": DetectorModel(
        name="Standard InGaAs",
        optimal_range=(1000, 1600),  # Peak QE >65% in this range
        roll_off_rate=0.008,  # Gradual roll-off
        min_sensitivity=0.3,
    ),
    "ingaas_extended": DetectorModel(
        name="Extended InGaAs",
        optimal_range=(1100, 2200),  # Extended range with lower D*
        roll_off_rate=0.005,
        min_sensitivity=0.2,
    ),
    "pbs": DetectorModel(
        name="PbS",
        optimal_range=(1000, 2800),  # 1-3.3 µm range
        roll_off_rate=0.004,
        min_sensitivity=0.25,
    ),
    "silicon_ccd": DetectorModel(
        name="Silicon CCD",
        optimal_range=(400, 900),  # Drop-off after 1000 nm
        roll_off_rate=0.015,  # Sharp roll-off at NIR
        min_sensitivity=0.1,
    ),
    "generic_nir": DetectorModel(
        name="Generic NIR",
        optimal_range=(900, 1700),
        roll_off_rate=0.006,
        min_sensitivity=0.35,
    ),
}


class DetectorRollOffAugmenter(SpectraTransformerMixin):
    """
    Simulate detector sensitivity roll-off at spectral edges.

    NIR detectors have wavelength-dependent sensitivity curves that typically
    roll off at the edges of their spectral range. This causes:
    - Increased noise at edge wavelengths (lower SNR)
    - Apparent baseline curvature near spectral boundaries
    - Reduced peak heights at the edges

    The effect is modeled as an exponential decay of detector sensitivity
    outside the optimal wavelength range, which manifests as multiplicative
    noise amplification and slight baseline distortion.

    Parameters
    ----------
    detector_model : str, default="generic_nir"
        Detector type to simulate. Available models:
        - "ingaas_standard": Standard InGaAs (1000-1600 nm optimal)
        - "ingaas_extended": Extended InGaAs (1100-2200 nm optimal)
        - "pbs": Lead sulfide (1000-2800 nm optimal)
        - "silicon_ccd": Silicon CCD (400-900 nm optimal)
        - "generic_nir": Generic NIR detector
    effect_strength : float, default=1.0
        Scaling factor for the roll-off effect (0-2).
    noise_amplification : float, default=0.02
        Additional noise added at low-sensitivity wavelengths.
    include_baseline_distortion : bool, default=True
        Whether to include slight baseline distortion at edges.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    _requires_wavelengths : bool
        Always True - this operator requires wavelength information.

    Examples
    --------
    >>> from nirs4all.operators.augmentation import DetectorRollOffAugmenter
    >>> aug = DetectorRollOffAugmenter(detector_model="ingaas_standard")
    >>> X_aug = aug.transform(X, wavelengths=wavelengths)

    >>> # Stronger effect for portable spectrometers
    >>> aug = DetectorRollOffAugmenter(effect_strength=1.5)
    >>> pipeline = [aug, SNV(), PLSRegression(10)]

    References
    ----------
    - JASCO (2020). Advantages of high-sensitivity InGaAs detector.
    - LaserComponents InGaAs Photodiodes specifications.
    """

    _requires_wavelengths: bool = True

    def __init__(
        self,
        detector_model: str = "generic_nir",
        effect_strength: float = 1.0,
        noise_amplification: float = 0.02,
        include_baseline_distortion: bool = True,
        random_state: Optional[int] = None,
    ):
        self.detector_model = detector_model
        self.effect_strength = effect_strength
        self.noise_amplification = noise_amplification
        self.include_baseline_distortion = include_baseline_distortion
        self.random_state = random_state

    def _transform_impl(
        self, X: np.ndarray, wavelengths: np.ndarray
    ) -> np.ndarray:
        """
        Apply detector roll-off effects to spectra.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input spectra.
        wavelengths : ndarray of shape (n_features,)
            Wavelength array in nm.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Spectra with detector roll-off effects applied.
        """
        rng = np.random.default_rng(self.random_state)
        result = X.copy()

        # Get detector model parameters
        if self.detector_model not in DETECTOR_MODELS:
            raise ValueError(
                f"Unknown detector model '{self.detector_model}'. "
                f"Available: {list(DETECTOR_MODELS.keys())}"
            )
        model = DETECTOR_MODELS[self.detector_model]

        # Compute sensitivity curve
        sensitivity = self._compute_sensitivity_curve(wavelengths, model)

        # Apply effects to each sample
        for i in range(result.shape[0]):
            result[i] = self._apply_roll_off_effects(
                result[i], wavelengths, sensitivity, rng
            )

        return result

    def _compute_sensitivity_curve(
        self,
        wavelengths: np.ndarray,
        model: DetectorModel
    ) -> np.ndarray:
        """Compute detector sensitivity curve based on model."""
        opt_min, opt_max = model.optimal_range
        sensitivity = np.ones_like(wavelengths, dtype=np.float64)

        # Roll-off at lower wavelengths
        below_optimal = wavelengths < opt_min
        if np.any(below_optimal):
            distance = opt_min - wavelengths[below_optimal]
            decay = np.exp(-model.roll_off_rate * self.effect_strength * distance)
            sensitivity[below_optimal] = model.min_sensitivity + (1 - model.min_sensitivity) * decay

        # Roll-off at higher wavelengths
        above_optimal = wavelengths > opt_max
        if np.any(above_optimal):
            distance = wavelengths[above_optimal] - opt_max
            decay = np.exp(-model.roll_off_rate * self.effect_strength * distance)
            sensitivity[above_optimal] = model.min_sensitivity + (1 - model.min_sensitivity) * decay

        return sensitivity

    def _apply_roll_off_effects(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        sensitivity: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Apply roll-off effects to a single spectrum."""
        result = spectrum.copy()

        # 1. Amplify noise at low-sensitivity regions
        if self.noise_amplification > 0:
            # Inverse sensitivity -> noise amplification factor
            noise_factor = (1 / np.clip(sensitivity, 0.1, 1.0) - 1) * self.noise_amplification
            noise = rng.normal(0, 1, len(wavelengths)) * noise_factor
            result = result + noise

        # 2. Slight baseline distortion at edges
        if self.include_baseline_distortion:
            baseline_distortion = (1 - sensitivity) * 0.01 * self.effect_strength
            result = result + baseline_distortion

        return result


class StrayLightAugmenter(SpectraTransformerMixin):
    """
    Simulate stray light effects on NIR spectra.

    Stray light is unwanted radiation that reaches the detector without passing
    through the intended optical path. Its effects are most pronounced:
    - At high-absorbance wavelengths (peaks appear truncated)
    - At spectral edges where instrument sensitivity is lower
    - Near the limits of the detector's wavelength range

    The primary effect is a reduction in observed peak height, causing apparent
    negative deviations from Beer's law. This is particularly problematic at
    the edges of spectra where stray light often constitutes a larger fraction
    of the total signal.

    Parameters
    ----------
    stray_light_fraction : float, default=0.001
        Base stray light as fraction of total signal (0.001 = 0.1%).
        Typical values: 0.0001-0.01 depending on instrument quality.
    edge_enhancement : float, default=2.0
        Factor by which stray light increases at spectral edges.
    edge_width : float, default=0.1
        Fraction of spectral range considered "edge" (0-0.5).
    include_peak_truncation : bool, default=True
        Whether to simulate peak height reduction at high absorbance.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    _requires_wavelengths : bool
        Always True - this operator requires wavelength information.

    Examples
    --------
    >>> from nirs4all.operators.augmentation import StrayLightAugmenter
    >>> aug = StrayLightAugmenter(stray_light_fraction=0.005)
    >>> X_aug = aug.transform(X, wavelengths=wavelengths)

    >>> # High stray light (older/portable instruments)
    >>> aug = StrayLightAugmenter(stray_light_fraction=0.01, edge_enhancement=3.0)
    >>> pipeline = [aug, MSC(), PLSRegression(10)]

    Notes
    -----
    The observed transmittance with stray light is:
        T_obs = (T_true + s) / (1 + s)

    where s is the stray light fraction. This causes:
    - At high absorbance (low T_true): T_obs ≈ s, creating a floor effect
    - At low absorbance (high T_true): Minimal effect

    Converting to absorbance:
        A_obs = -log10(T_obs) < A_true

    References
    ----------
    - Applied Optics (1975). Resolution and stray light in near infrared
      spectroscopy, 14(8), 1977.
    - Chalmers & Griffiths (2001). Mid-Infrared Spectroscopy: Anomalies,
      Artifacts and Common Errors.
    """

    _requires_wavelengths: bool = True

    def __init__(
        self,
        stray_light_fraction: float = 0.001,
        edge_enhancement: float = 2.0,
        edge_width: float = 0.1,
        include_peak_truncation: bool = True,
        random_state: Optional[int] = None,
    ):
        self.stray_light_fraction = stray_light_fraction
        self.edge_enhancement = edge_enhancement
        self.edge_width = edge_width
        self.include_peak_truncation = include_peak_truncation
        self.random_state = random_state

    def _transform_impl(
        self, X: np.ndarray, wavelengths: np.ndarray
    ) -> np.ndarray:
        """
        Apply stray light effects to spectra.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input spectra (in absorbance units).
        wavelengths : ndarray of shape (n_features,)
            Wavelength array in nm.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Spectra with stray light effects applied.
        """
        rng = np.random.default_rng(self.random_state)
        result = X.copy()

        # Compute edge-enhanced stray light profile
        stray_profile = self._compute_stray_light_profile(wavelengths)

        # Apply effects to each sample
        for i in range(result.shape[0]):
            result[i] = self._apply_stray_light(result[i], stray_profile, rng)

        return result

    def _compute_stray_light_profile(self, wavelengths: np.ndarray) -> np.ndarray:
        """Compute wavelength-dependent stray light profile."""
        n_wl = len(wavelengths)
        wl_range = wavelengths.max() - wavelengths.min()
        edge_points = int(n_wl * self.edge_width)

        # Base stray light level
        profile = np.ones(n_wl) * self.stray_light_fraction

        # Enhanced stray light at edges using smooth sigmoid transition
        if edge_points > 0:
            # Left edge enhancement
            left_x = np.linspace(0, 5, edge_points)
            left_weight = 1 + (self.edge_enhancement - 1) / (1 + np.exp(left_x - 2.5))
            profile[:edge_points] *= left_weight

            # Right edge enhancement
            right_x = np.linspace(5, 0, edge_points)
            right_weight = 1 + (self.edge_enhancement - 1) / (1 + np.exp(right_x - 2.5))
            profile[-edge_points:] *= right_weight

        return profile

    def _apply_stray_light(
        self,
        spectrum: np.ndarray,
        stray_profile: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Apply stray light effect to a single spectrum."""
        # Add random variation to stray light
        stray = stray_profile * (1 + rng.normal(0, 0.1, len(spectrum)))
        stray = np.clip(stray, 0, 0.1)  # Cap at 10%

        # Convert absorbance to transmittance
        # A = -log10(T), so T = 10^(-A)
        T_true = np.power(10, -spectrum)
        T_true = np.clip(T_true, 1e-10, 1.0)  # Avoid numerical issues

        # Apply stray light: T_obs = (T_true + s) / (1 + s)
        T_obs = (T_true + stray) / (1 + stray)

        # Convert back to absorbance
        result = -np.log10(T_obs)

        # Peak truncation: high absorbance values are more affected
        if self.include_peak_truncation:
            # This is already captured by the stray light model
            # but we can add extra effect at very high absorbance
            high_abs_mask = spectrum > 1.5
            if np.any(high_abs_mask):
                truncation_factor = 1 - 0.05 * np.clip(spectrum[high_abs_mask] - 1.5, 0, 2)
                result[high_abs_mask] *= truncation_factor

        return result


class EdgeCurvatureAugmenter(SpectraTransformerMixin):
    """
    Simulate edge curvature and baseline bending at spectral boundaries.

    Edge curvature can arise from various sources:
    - Optical aberrations in the spectrometer
    - Wavelength-dependent baseline drift
    - Polynomial baseline correction artifacts
    - Sample holder effects

    This operator adds smooth curvature that increases towards the spectral
    edges, mimicking the characteristic "smile" or "frown" patterns often
    seen in real spectra.

    Parameters
    ----------
    curvature_strength : float, default=0.02
        Maximum curvature amplitude (in absorbance units).
    curvature_type : str, default="random"
        Type of curvature pattern:
        - "random": Randomly choose smile/frown/asymmetric
        - "smile": Upward curvature at edges (convex)
        - "frown": Downward curvature at edges (concave)
        - "asymmetric": Different curvature at each edge
    asymmetry : float, default=0.0
        For "asymmetric" type, ratio of left/right curvature (-1 to 1).
        Positive values emphasize left edge, negative emphasize right.
    edge_focus : float, default=0.7
        How concentrated the curvature is at edges (0-1).
        Higher values create sharper edge effects.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    _requires_wavelengths : bool
        Always True - this operator requires wavelength information.

    Examples
    --------
    >>> from nirs4all.operators.augmentation import EdgeCurvatureAugmenter
    >>> aug = EdgeCurvatureAugmenter(curvature_strength=0.03)
    >>> X_aug = aug.transform(X, wavelengths=wavelengths)

    >>> # Simulate baseline correction artifacts
    >>> aug = EdgeCurvatureAugmenter(
    ...     curvature_type="asymmetric",
    ...     asymmetry=0.5,
    ...     edge_focus=0.8
    ... )
    >>> pipeline = [aug, Detrend(), PLSRegression(10)]

    References
    ----------
    - Cao, A., et al. (2007). A robust method for automated background
      subtraction of tissue fluorescence. Journal of Raman Spectroscopy.
    - NIRPY Research (2019). Two methods for baseline correction of
      spectral data.
    """

    _requires_wavelengths: bool = True

    def __init__(
        self,
        curvature_strength: float = 0.02,
        curvature_type: str = "random",
        asymmetry: float = 0.0,
        edge_focus: float = 0.7,
        random_state: Optional[int] = None,
    ):
        self.curvature_strength = curvature_strength
        self.curvature_type = curvature_type
        self.asymmetry = asymmetry
        self.edge_focus = edge_focus
        self.random_state = random_state

    def _transform_impl(
        self, X: np.ndarray, wavelengths: np.ndarray
    ) -> np.ndarray:
        """
        Apply edge curvature effects to spectra.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input spectra.
        wavelengths : ndarray of shape (n_features,)
            Wavelength array in nm.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Spectra with edge curvature applied.
        """
        rng = np.random.default_rng(self.random_state)
        result = X.copy()

        # Normalized wavelength scale [0, 1]
        wl_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())

        for i in range(result.shape[0]):
            curvature = self._generate_curvature(wl_norm, rng)
            result[i] = result[i] + curvature

        return result

    def _generate_curvature(
        self,
        wl_norm: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Generate curvature pattern."""
        # Determine curvature type
        if self.curvature_type == "random":
            ctype = rng.choice(["smile", "frown", "asymmetric"])
        else:
            ctype = self.curvature_type

        # Base curvature shape (centered parabola)
        # x in [0, 1], centered at 0.5
        x = wl_norm - 0.5

        # Focus factor: higher values create sharper edge effects
        focus_power = 2 + 2 * self.edge_focus  # Range: 2-4

        if ctype == "smile":
            # Upward curvature at edges (U-shape)
            curvature = np.power(np.abs(x), focus_power) * 2
            curvature = curvature * self.curvature_strength

        elif ctype == "frown":
            # Downward curvature at edges (∩-shape)
            curvature = -np.power(np.abs(x), focus_power) * 2
            curvature = curvature * self.curvature_strength

        elif ctype == "asymmetric":
            # Different curvature on each side
            asym = self.asymmetry + rng.uniform(-0.2, 0.2)
            asym = np.clip(asym, -1, 1)

            left_strength = 1 + asym
            right_strength = 1 - asym

            curvature = np.zeros_like(wl_norm)
            left_mask = wl_norm < 0.5
            right_mask = ~left_mask

            curvature[left_mask] = np.power(np.abs(x[left_mask]), focus_power) * left_strength
            curvature[right_mask] = np.power(np.abs(x[right_mask]), focus_power) * right_strength

            # Random sign
            sign = rng.choice([-1, 1])
            curvature = sign * curvature * self.curvature_strength

        else:
            raise ValueError(f"Unknown curvature_type: {ctype}")

        # Add small random variation
        curvature = curvature * (1 + rng.normal(0, 0.1))

        return curvature


class TruncatedPeakAugmenter(SpectraTransformerMixin):
    """
    Simulate truncated absorption peaks at spectral boundaries.

    When measuring NIR spectra, absorption bands that have their centers
    outside the measured wavelength range will appear as partial peaks
    at the spectral edges. This creates characteristic rising or falling
    baselines at the spectrum boundaries.

    This effect is common when:
    - The spectrometer range doesn't cover the full absorption band
    - Strong absorbers (e.g., water) have peaks just outside the range
    - Mid-IR absorption bands tail into the NIR region

    Parameters
    ----------
    peak_probability : float, default=0.3
        Probability of adding truncated peaks (0-1).
    amplitude_range : tuple of (float, float), default=(0.01, 0.1)
        Range of peak amplitudes (in absorbance units).
    width_range : tuple of (float, float), default=(50, 200)
        Range of peak widths (in nm). Controls how fast the edge rises/falls.
    left_edge : bool, default=True
        Whether to potentially add truncated peak at left (low wavelength) edge.
    right_edge : bool, default=True
        Whether to potentially add truncated peak at right (high wavelength) edge.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    _requires_wavelengths : bool
        Always True - this operator requires wavelength information.

    Examples
    --------
    >>> from nirs4all.operators.augmentation import TruncatedPeakAugmenter
    >>> aug = TruncatedPeakAugmenter(peak_probability=0.5)
    >>> X_aug = aug.transform(X, wavelengths=wavelengths)

    >>> # Strong truncated peaks (e.g., water band edge)
    >>> aug = TruncatedPeakAugmenter(
    ...     amplitude_range=(0.05, 0.2),
    ...     width_range=(100, 300)
    ... )
    >>> pipeline = [aug, SNV(), PLSRegression(10)]

    Notes
    -----
    The truncated peak is modeled as a Gaussian band with its center
    positioned outside the measured wavelength range. Only the "tail"
    of this band appears in the spectrum.
    """

    _requires_wavelengths: bool = True

    def __init__(
        self,
        peak_probability: float = 0.3,
        amplitude_range: Tuple[float, float] = (0.01, 0.1),
        width_range: Tuple[float, float] = (50, 200),
        left_edge: bool = True,
        right_edge: bool = True,
        random_state: Optional[int] = None,
    ):
        self.peak_probability = peak_probability
        self.amplitude_range = amplitude_range
        self.width_range = width_range
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.random_state = random_state

    def _transform_impl(
        self, X: np.ndarray, wavelengths: np.ndarray
    ) -> np.ndarray:
        """
        Apply truncated peak effects to spectra.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input spectra.
        wavelengths : ndarray of shape (n_features,)
            Wavelength array in nm.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Spectra with truncated peaks at edges.
        """
        rng = np.random.default_rng(self.random_state)
        result = X.copy()

        wl_min, wl_max = wavelengths.min(), wavelengths.max()
        wl_range = wl_max - wl_min

        for i in range(result.shape[0]):
            # Left edge truncated peak
            if self.left_edge and rng.random() < self.peak_probability:
                peak = self._generate_truncated_peak(
                    wavelengths, "left", wl_min, wl_range, rng
                )
                result[i] = result[i] + peak

            # Right edge truncated peak
            if self.right_edge and rng.random() < self.peak_probability:
                peak = self._generate_truncated_peak(
                    wavelengths, "right", wl_max, wl_range, rng
                )
                result[i] = result[i] + peak

        return result

    def _generate_truncated_peak(
        self,
        wavelengths: np.ndarray,
        edge: str,
        edge_wl: float,
        wl_range: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Generate a truncated Gaussian peak at the specified edge."""
        amplitude = rng.uniform(*self.amplitude_range)
        width = rng.uniform(*self.width_range)

        # Position peak center outside the measured range
        # Distance outside: 0.5-1.5 times the width
        offset = width * rng.uniform(0.5, 1.5)

        if edge == "left":
            center = edge_wl - offset
        else:
            center = edge_wl + offset

        # Gaussian profile
        peak = amplitude * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)

        return peak


# =============================================================================
# Combined Edge Artifacts Augmenter
# =============================================================================

class EdgeArtifactsAugmenter(SpectraTransformerMixin):
    """
    Combined augmenter for edge-related spectral artifacts.

    This is a convenience class that combines multiple edge artifact effects:
    - Detector roll-off
    - Stray light
    - Edge curvature
    - Truncated peaks

    Each effect can be individually enabled/disabled.

    Parameters
    ----------
    detector_roll_off : bool, default=True
        Enable detector sensitivity roll-off effect.
    stray_light : bool, default=True
        Enable stray light effect.
    edge_curvature : bool, default=True
        Enable edge curvature/bending effect.
    truncated_peaks : bool, default=True
        Enable truncated peak effect at boundaries.
    overall_strength : float, default=1.0
        Scaling factor for all effects (0-2).
    detector_model : str, default="generic_nir"
        Detector model for roll-off simulation.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    _requires_wavelengths : bool
        Always True - this operator requires wavelength information.

    Examples
    --------
    >>> from nirs4all.operators.augmentation import EdgeArtifactsAugmenter
    >>> aug = EdgeArtifactsAugmenter(overall_strength=0.8)
    >>> X_aug = aug.transform(X, wavelengths=wavelengths)

    >>> # Only detector and stray light effects
    >>> aug = EdgeArtifactsAugmenter(
    ...     detector_roll_off=True,
    ...     stray_light=True,
    ...     edge_curvature=False,
    ...     truncated_peaks=False
    ... )
    >>> pipeline = [aug, SNV(), PLSRegression(10)]
    """

    _requires_wavelengths: bool = True

    def __init__(
        self,
        detector_roll_off: bool = True,
        stray_light: bool = True,
        edge_curvature: bool = True,
        truncated_peaks: bool = True,
        overall_strength: float = 1.0,
        detector_model: str = "generic_nir",
        random_state: Optional[int] = None,
    ):
        self.detector_roll_off = detector_roll_off
        self.stray_light = stray_light
        self.edge_curvature = edge_curvature
        self.truncated_peaks = truncated_peaks
        self.overall_strength = overall_strength
        self.detector_model = detector_model
        self.random_state = random_state

        # Initialize sub-augmenters
        self._init_augmenters()

    def _init_augmenters(self):
        """Initialize sub-augmenters based on settings."""
        rng = np.random.default_rng(self.random_state)

        if self.detector_roll_off:
            self._detector_aug = DetectorRollOffAugmenter(
                detector_model=self.detector_model,
                effect_strength=self.overall_strength,
                random_state=int(rng.integers(0, 2**31)),
            )

        if self.stray_light:
            self._stray_aug = StrayLightAugmenter(
                stray_light_fraction=0.001 * self.overall_strength,
                random_state=int(rng.integers(0, 2**31)),
            )

        if self.edge_curvature:
            self._curvature_aug = EdgeCurvatureAugmenter(
                curvature_strength=0.02 * self.overall_strength,
                random_state=int(rng.integers(0, 2**31)),
            )

        if self.truncated_peaks:
            self._truncated_aug = TruncatedPeakAugmenter(
                peak_probability=0.3,
                amplitude_range=(0.01 * self.overall_strength, 0.1 * self.overall_strength),
                random_state=int(rng.integers(0, 2**31)),
            )

    def _transform_impl(
        self, X: np.ndarray, wavelengths: np.ndarray
    ) -> np.ndarray:
        """
        Apply all enabled edge artifact effects to spectra.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input spectra.
        wavelengths : ndarray of shape (n_features,)
            Wavelength array in nm.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Spectra with edge artifacts applied.
        """
        result = X.copy()

        if self.truncated_peaks:
            result = self._truncated_aug.transform(result, wavelengths=wavelengths)

        if self.edge_curvature:
            result = self._curvature_aug.transform(result, wavelengths=wavelengths)

        if self.stray_light:
            result = self._stray_aug.transform(result, wavelengths=wavelengths)

        if self.detector_roll_off:
            result = self._detector_aug.transform(result, wavelengths=wavelengths)

        return result


__all__ = [
    "DetectorRollOffAugmenter",
    "StrayLightAugmenter",
    "EdgeCurvatureAugmenter",
    "TruncatedPeakAugmenter",
    "EdgeArtifactsAugmenter",
    "DETECTOR_MODELS",
    "DetectorModel",
]
