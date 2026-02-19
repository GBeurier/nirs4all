"""
Forward model components for physical signal-chain reconstruction.

This module implements the forward measurement chain:
    Canonical physical model → Instrument effects → Domain transform → Preprocessing

Key principle: Keep one latent physical model on a canonical grid, then apply
dataset-specific transforms to match observed data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

if TYPE_CHECKING:
    from .environmental import EnvironmentalEffectsModel

# =============================================================================
# Canonical Forward Model
# =============================================================================

@dataclass
class CanonicalForwardModel:
    """
    Physical model on canonical high-resolution wavelength grid.

    Computes absorption coefficient K(λ) from chemical components:
        K(λ) = Σ c_k * ε_k(λ) + K0(λ)

    where:
        - c_k: concentration of component k
        - ε_k(λ): molar absorptivity (from component library)
        - K0(λ): continuum/background absorption (low-frequency)

    Attributes:
        canonical_grid: High-resolution wavelength grid (nm).
        component_names: Names of components to include.
        component_spectra: Pre-computed component spectra on canonical grid.
        baseline_order: Order of Chebyshev baseline polynomial.
        continuum_order: Order of continuum absorption polynomial.
    """

    canonical_grid: np.ndarray
    component_names: list[str] = field(default_factory=list)
    baseline_order: int = 5
    continuum_order: int = 3
    _component_spectra: np.ndarray | None = field(default=None, repr=False)
    _baseline_basis: np.ndarray | None = field(default=None, repr=False)
    _continuum_basis: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize component spectra and basis matrices."""
        self._build_component_spectra()
        self._build_basis_matrices()

    def _build_component_spectra(self) -> None:
        """Pre-compute component spectra on canonical grid."""
        from ..components import get_component

        n_wl = len(self.canonical_grid)
        n_comp = len(self.component_names)

        if n_comp == 0:
            self._component_spectra = np.zeros((0, n_wl))
            return

        self._component_spectra = np.zeros((n_comp, n_wl))
        for k, name in enumerate(self.component_names):
            try:
                comp = get_component(name)
                self._component_spectra[k] = comp.compute(self.canonical_grid)
            except (ValueError, KeyError):
                # Component not found, leave as zeros
                pass

    def _build_basis_matrices(self) -> None:
        """Build Chebyshev polynomial basis matrices."""
        n_wl = len(self.canonical_grid)
        wl_norm = self._normalize_wavelengths(self.canonical_grid)

        # Baseline basis (Chebyshev polynomials)
        self._baseline_basis = np.zeros((self.baseline_order + 1, n_wl))
        for i in range(self.baseline_order + 1):
            self._baseline_basis[i] = np.polynomial.chebyshev.chebval(
                wl_norm, [0] * i + [1]
            )

        # Continuum basis (lower order for smooth background)
        self._continuum_basis = np.zeros((self.continuum_order + 1, n_wl))
        for i in range(self.continuum_order + 1):
            self._continuum_basis[i] = np.polynomial.chebyshev.chebval(
                wl_norm, [0] * i + [1]
            )

    def _normalize_wavelengths(self, wl: np.ndarray) -> np.ndarray:
        """Normalize wavelengths to [-1, 1] for Chebyshev basis."""
        wl_min, wl_max = self.canonical_grid.min(), self.canonical_grid.max()
        return 2 * (wl - wl_min) / (wl_max - wl_min) - 1

    def compute_absorption(
        self,
        concentrations: np.ndarray,
        path_length: float = 1.0,
        baseline_coeffs: np.ndarray | None = None,
        continuum_coeffs: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute absorption coefficient on canonical grid.

        Args:
            concentrations: Component concentrations, shape (n_components,).
            path_length: Optical path length factor.
            baseline_coeffs: Baseline polynomial coefficients.
            continuum_coeffs: Continuum absorption coefficients.

        Returns:
            Absorbance spectrum on canonical grid.
        """
        n_wl = len(self.canonical_grid)

        # Component contribution: A = L * Σ c_k * ε_k(λ)
        absorption = path_length * (concentrations @ self._component_spectra) if self._component_spectra is not None and len(concentrations) > 0 else np.zeros(n_wl)

        # Add baseline
        if baseline_coeffs is not None and self._baseline_basis is not None:
            absorption += baseline_coeffs @ self._baseline_basis

        # Add continuum absorption
        if continuum_coeffs is not None and self._continuum_basis is not None:
            absorption += continuum_coeffs @ self._continuum_basis

        return absorption

    def get_design_matrix(self, path_length: float = 1.0) -> np.ndarray:
        """
        Get full design matrix for linear fitting.

        Returns:
            Design matrix of shape (n_wavelengths, n_components + n_baseline + n_continuum).
        """
        matrices = []

        # Component spectra (scaled by path length)
        if self._component_spectra is not None and self._component_spectra.shape[0] > 0:
            matrices.append(path_length * self._component_spectra.T)

        # Baseline basis
        if self._baseline_basis is not None:
            matrices.append(self._baseline_basis.T)

        # Continuum basis
        if self._continuum_basis is not None:
            matrices.append(self._continuum_basis.T)

        if matrices:
            return np.hstack(matrices)
        return np.zeros((len(self.canonical_grid), 0))

    @property
    def n_components(self) -> int:
        """Number of chemical components."""
        return len(self.component_names)

    @property
    def n_baseline(self) -> int:
        """Number of baseline coefficients."""
        return self.baseline_order + 1

    @property
    def n_continuum(self) -> int:
        """Number of continuum coefficients."""
        return self.continuum_order + 1

    @property
    def n_linear_params(self) -> int:
        """Total number of linear parameters."""
        return self.n_components + self.n_baseline + self.n_continuum

# =============================================================================
# Instrument Model
# =============================================================================

@dataclass
class InstrumentModel:
    """
    Instrument effects: warp, ILS convolution, gain/offset, resampling.

    Transforms spectrum from canonical grid to target instrument grid:
        1. Wavelength warp: λ* → λ' (shift + stretch + optional higher order)
        2. ILS convolution: Gaussian or Voigt line shape
        3. Stray light / gain / offset
        4. Resample to target grid

    Attributes:
        target_grid: Target wavelength grid (dataset grid).
        wl_shift: Wavelength shift in nm (default 0).
        wl_stretch: Wavelength scale factor (default 1).
        wl_poly_coeffs: Higher-order polynomial warp coefficients.
        ils_sigma: Instrument line shape Gaussian sigma in nm.
        stray_light: Stray light fraction (default 0).
        gain: Photometric gain (default 1).
        offset: Photometric offset (default 0).
    """

    target_grid: np.ndarray
    wl_shift: float = 0.0
    wl_stretch: float = 1.0
    wl_poly_coeffs: np.ndarray | None = None
    ils_sigma: float = 4.0
    stray_light: float = 0.0
    gain: float = 1.0
    offset: float = 0.0

    def apply(
        self,
        spectrum: np.ndarray,
        canonical_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Apply instrument chain to transform spectrum.

        Args:
            spectrum: Input spectrum on canonical grid.
            canonical_grid: Canonical wavelength grid.

        Returns:
            Transformed spectrum on target grid.
        """
        # 1. Wavelength warp
        warped_wl = self._apply_wavelength_warp(canonical_grid)

        # 2. ILS convolution
        wl_step = np.median(np.diff(canonical_grid))
        if self.ils_sigma > 0 and wl_step > 0:
            sigma_idx = self.ils_sigma / wl_step
            spectrum_ils = gaussian_filter1d(spectrum, sigma=max(0.5, sigma_idx))
        else:
            spectrum_ils = spectrum

        # 3. Stray light / gain / offset
        spectrum_phot = self.gain * spectrum_ils + self.offset
        if self.stray_light > 0:
            spectrum_phot = spectrum_phot + self.stray_light * np.mean(spectrum_ils)

        # 4. Resample to target grid
        # Use linear interpolation with extrapolation handling
        valid_mask = (warped_wl >= canonical_grid.min()) & (warped_wl <= canonical_grid.max())
        if not np.all(valid_mask):
            # Extend with edge values for extrapolation
            spectrum_resampled = np.interp(
                self.target_grid,
                warped_wl,
                spectrum_phot,
                left=spectrum_phot[0],
                right=spectrum_phot[-1],
            )
        else:
            spectrum_resampled = np.interp(self.target_grid, warped_wl, spectrum_phot)

        return spectrum_resampled

    def _apply_wavelength_warp(self, canonical_grid: np.ndarray) -> np.ndarray:
        """Apply wavelength warp transformation."""
        warped = self.wl_shift + self.wl_stretch * canonical_grid

        if self.wl_poly_coeffs is not None and len(self.wl_poly_coeffs) > 0:
            # Normalize grid for polynomial
            wl_norm = (canonical_grid - canonical_grid.mean()) / (
                canonical_grid.max() - canonical_grid.min()
            )
            for i, coeff in enumerate(self.wl_poly_coeffs):
                warped += coeff * wl_norm ** (i + 2)  # Start from quadratic

        return warped

    def get_jacobian_wrt_wl_shift(
        self,
        spectrum: np.ndarray,
        canonical_grid: np.ndarray,
        eps: float = 0.1,
    ) -> np.ndarray:
        """Numerical Jacobian w.r.t. wavelength shift."""
        orig = self.wl_shift
        self.wl_shift = orig + eps
        spec_plus = self.apply(spectrum, canonical_grid)
        self.wl_shift = orig - eps
        spec_minus = self.apply(spectrum, canonical_grid)
        self.wl_shift = orig
        return (spec_plus - spec_minus) / (2 * eps)

    def get_jacobian_wrt_ils_sigma(
        self,
        spectrum: np.ndarray,
        canonical_grid: np.ndarray,
        eps: float = 0.1,
    ) -> np.ndarray:
        """Numerical Jacobian w.r.t. ILS sigma."""
        orig = self.ils_sigma
        self.ils_sigma = orig + eps
        spec_plus = self.apply(spectrum, canonical_grid)
        self.ils_sigma = max(0.5, orig - eps)
        spec_minus = self.apply(spectrum, canonical_grid)
        self.ils_sigma = orig
        return (spec_plus - spec_minus) / (2 * eps)

    @classmethod
    def from_params(
        cls,
        target_grid: np.ndarray,
        params: dict[str, float],
    ) -> InstrumentModel:
        """Create InstrumentModel from parameter dictionary."""
        return cls(
            target_grid=target_grid,
            wl_shift=params.get("wl_shift", 0.0),
            wl_stretch=params.get("wl_stretch", 1.0),
            ils_sigma=params.get("ils_sigma", 4.0),
            stray_light=params.get("stray_light", 0.0),
            gain=params.get("gain", 1.0),
            offset=params.get("offset", 0.0),
        )

# =============================================================================
# Domain Transform
# =============================================================================

@dataclass
class DomainTransform:
    """
    Transform between physical domains (absorbance, reflectance, etc.).

    For absorbance datasets: A(λ) = absorption coefficient (direct)
    For reflectance datasets: R(λ) computed via Kubelka-Munk or approximation

    Attributes:
        domain: Domain type ('absorbance', 'reflectance', 'transmittance', 'km').
        scatter_coeffs: Scattering coefficients for KM model (reflectance).
        scatter_wavelength_dep: Wavelength-dependent scatter (λ^-n).
    """

    domain: Literal["absorbance", "reflectance", "transmittance", "km"] = "absorbance"
    scatter_coeffs: np.ndarray | None = None
    scatter_wavelength_exp: float = 0.0  # For wavelength-dependent scatter

    def transform(
        self,
        absorption: np.ndarray,
        wavelengths: np.ndarray,
        scatter: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Transform absorption to target domain.

        Args:
            absorption: Absorption coefficient K(λ).
            wavelengths: Wavelength grid.
            scatter: Scattering coefficient S(λ) for reflectance.

        Returns:
            Spectrum in target domain representation.
        """
        if self.domain == "absorbance":
            return absorption

        elif self.domain == "transmittance":
            # T = exp(-A) for optical density
            return np.exp(-np.clip(absorption, 0, 10))

        elif self.domain in ("reflectance", "km"):
            # Kubelka-Munk: f(R) = (1-R)²/(2R) = K/S
            # Solve for R given K and S

            # Get or compute scattering coefficient
            if scatter is not None:
                S = scatter
            elif self.scatter_coeffs is not None:
                # Use provided scatter coefficients with wavelength dependence
                S = self._compute_scatter(wavelengths)
            else:
                # Default scattering (smooth baseline)
                S = np.ones_like(absorption) * 0.5

            # Avoid division by zero
            S = np.maximum(S, 1e-6)
            K = np.maximum(absorption, 0)

            # KM ratio
            km_ratio = K / S

            # Solve quadratic: (1-R)²/(2R) = km_ratio
            # R² - 2R(1 + km_ratio) + 1 = 0
            # R = (1 + km_ratio) - sqrt((1 + km_ratio)² - 1)
            a = 1 + km_ratio
            discriminant = np.maximum(a**2 - 1, 0)
            R = a - np.sqrt(discriminant)

            # Clip to valid reflectance range
            R = np.clip(R, 0.01, 0.99)

            if self.domain == "km":
                # Return KM function value
                return km_ratio
            return R

        return absorption

    def inverse_transform(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        scatter: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Inverse transform from domain to absorption.

        Args:
            spectrum: Spectrum in domain representation.
            wavelengths: Wavelength grid.
            scatter: Scattering coefficient for reflectance.

        Returns:
            Absorption coefficient.
        """
        if self.domain == "absorbance":
            return spectrum

        elif self.domain == "transmittance":
            return -np.log(np.clip(spectrum, 1e-6, 1))

        elif self.domain in ("reflectance", "km"):
            if self.domain == "km":
                km_ratio = spectrum
            else:
                R = np.clip(spectrum, 0.01, 0.99)
                km_ratio = (1 - R) ** 2 / (2 * R)

            if scatter is not None:
                S = scatter
            elif self.scatter_coeffs is not None:
                S = self._compute_scatter(wavelengths)
            else:
                S = np.ones_like(spectrum) * 0.5

            return km_ratio * S

        return spectrum

    def _compute_scatter(self, wavelengths: np.ndarray) -> np.ndarray:
        """Compute wavelength-dependent scattering coefficient."""
        if self.scatter_coeffs is None:
            return np.ones_like(wavelengths) * 0.5

        # Baseline scatter with wavelength dependence
        wl_norm = wavelengths / 1000.0  # Normalize to μm
        S = self.scatter_coeffs[0] * np.ones_like(wavelengths)

        if len(self.scatter_coeffs) > 1:
            S += self.scatter_coeffs[1] * (wl_norm ** (-self.scatter_wavelength_exp))

        if len(self.scatter_coeffs) > 2:
            # Polynomial terms
            wl_centered = (wavelengths - wavelengths.mean()) / 1000.0
            for i, coeff in enumerate(self.scatter_coeffs[2:]):
                S += coeff * wl_centered ** (i + 1)

        return np.maximum(S, 1e-6)

# =============================================================================
# Preprocessing Operator
# =============================================================================

@dataclass
class PreprocessingOperator:
    """
    Apply dataset preprocessing to match stored representation.

    Implements exact preprocessing steps:
        - Savitzky-Golay derivatives (1st, 2nd order)
        - SNV (Standard Normal Variate)
        - MSC (Multiplicative Scatter Correction)
        - Detrend
        - Mean centering

    Attributes:
        preprocessing_type: Type of preprocessing.
        sg_window: Savitzky-Golay window length.
        sg_polyorder: Savitzky-Golay polynomial order.
        sg_deriv: Derivative order (0, 1, 2).
        reference_spectrum: Reference for MSC (mean of calibration set).
    """

    preprocessing_type: Literal[
        "none", "first_derivative", "second_derivative",
        "snv", "msc", "detrend", "mean_centered"
    ] = "none"
    sg_window: int = 15
    sg_polyorder: int = 2
    sg_deriv: int = 0
    reference_spectrum: np.ndarray | None = None

    def apply(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to spectrum.

        Args:
            spectrum: Input spectrum, shape (n_wavelengths,) or (n_samples, n_wavelengths).

        Returns:
            Preprocessed spectrum(a).
        """
        is_1d = spectrum.ndim == 1
        if is_1d:
            spectrum = spectrum.reshape(1, -1)

        result = spectrum.copy()

        if self.preprocessing_type == "none":
            pass

        elif self.preprocessing_type == "first_derivative":
            window = min(self.sg_window, spectrum.shape[1] - 1) | 1
            polyorder = min(self.sg_polyorder, window - 1)
            result = savgol_filter(
                spectrum, window, polyorder, deriv=1, axis=1, mode="interp"
            )

        elif self.preprocessing_type == "second_derivative":
            window = min(self.sg_window, spectrum.shape[1] - 1) | 1
            polyorder = min(self.sg_polyorder + 1, window - 1)
            result = savgol_filter(
                spectrum, window, polyorder, deriv=2, axis=1, mode="interp"
            )

        elif self.preprocessing_type == "snv":
            means = spectrum.mean(axis=1, keepdims=True)
            stds = spectrum.std(axis=1, keepdims=True)
            stds = np.where(stds < 1e-10, 1.0, stds)
            result = (spectrum - means) / stds

        elif self.preprocessing_type == "msc":
            ref = spectrum.mean(axis=0) if self.reference_spectrum is None else self.reference_spectrum

            for i in range(spectrum.shape[0]):
                # Linear fit: spectrum[i] = a * ref + b
                coeffs = np.polyfit(ref, spectrum[i], 1)
                a, b = coeffs[0], coeffs[1]
                if abs(a) > 1e-10:
                    result[i] = (spectrum[i] - b) / a

        elif self.preprocessing_type == "detrend":
            from scipy.signal import detrend
            result = detrend(spectrum, axis=1, type="linear")

        elif self.preprocessing_type == "mean_centered":
            result = spectrum - spectrum.mean(axis=1, keepdims=True)

        if is_1d:
            return result.ravel()
        return result

    def apply_to_matrix(self, X: np.ndarray) -> np.ndarray:
        """Apply preprocessing to design matrix columns."""
        return self.apply(X.T).T

    @classmethod
    def from_detection(
        cls,
        preprocessing_type: str,
        sg_window: int = 15,
        sg_polyorder: int = 2,
    ) -> PreprocessingOperator:
        """Create PreprocessingOperator from detected preprocessing type."""
        type_map = {
            "raw_absorbance": "none",
            "raw_reflectance": "none",
            "first_derivative": "first_derivative",
            "second_derivative": "second_derivative",
            "snv_corrected": "snv",
            "msc_corrected": "msc",
            "mean_centered": "mean_centered",
            "normalized": "none",  # Min-max scaling doesn't need special handling
        }
        prep_type = type_map.get(preprocessing_type, "none")
        return cls(
            preprocessing_type=prep_type,
            sg_window=sg_window,
            sg_polyorder=sg_polyorder,
        )

# =============================================================================
# Forward Chain
# =============================================================================

@dataclass
class ForwardChain:
    """
    Complete forward measurement chain combining all components.

    Chain: CanonicalForwardModel → [EnvironmentalEffects] → DomainTransform → InstrumentModel → PreprocessingOperator

    Attributes:
        canonical_model: Physical model on canonical grid.
        environmental_model: Optional environmental effects (temperature, moisture, scattering).
        instrument_model: Instrument effects.
        domain_transform: Domain conversion.
        preprocessing: Dataset preprocessing.
    """

    canonical_model: CanonicalForwardModel
    instrument_model: InstrumentModel
    domain_transform: DomainTransform
    preprocessing: PreprocessingOperator
    environmental_model: EnvironmentalEffectsModel | None = None

    def forward(
        self,
        concentrations: np.ndarray,
        path_length: float = 1.0,
        baseline_coeffs: np.ndarray | None = None,
        continuum_coeffs: np.ndarray | None = None,
        scatter: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Run full forward chain.

        Args:
            concentrations: Component concentrations.
            path_length: Optical path length factor.
            baseline_coeffs: Baseline polynomial coefficients.
            continuum_coeffs: Continuum absorption coefficients.
            scatter: Scattering coefficients for reflectance.

        Returns:
            Spectrum on target grid with preprocessing applied.
        """
        # 1. Compute absorption on canonical grid
        absorption = self.canonical_model.compute_absorption(
            concentrations=concentrations,
            path_length=path_length,
            baseline_coeffs=baseline_coeffs,
            continuum_coeffs=continuum_coeffs,
        )

        # 2. Apply environmental effects (temperature, moisture, scattering)
        if self.environmental_model is not None and self.environmental_model.enabled:
            absorption = self.environmental_model.apply(
                absorption,
                self.canonical_model.canonical_grid,
            )

        # 3. Apply domain transform (absorbance → reflectance if needed)
        domain_spectrum = self.domain_transform.transform(
            absorption,
            self.canonical_model.canonical_grid,
            scatter=scatter,
        )

        # 4. Apply instrument effects and resample
        instrument_spectrum = self.instrument_model.apply(
            domain_spectrum,
            self.canonical_model.canonical_grid,
        )

        # 5. Apply preprocessing
        preprocessed = self.preprocessing.apply(instrument_spectrum)

        return preprocessed

    def forward_design_matrix(
        self,
        path_length: float = 1.0,
    ) -> np.ndarray:
        """
        Get transformed design matrix for linear fitting.

        Returns the design matrix after applying instrument and preprocessing transforms.
        Note: Domain transform is not applied here as it may be nonlinear (KM).
        """
        # Get canonical design matrix
        A_canonical = self.canonical_model.get_design_matrix(path_length)

        # Apply instrument transform to each column
        A_instrument = np.zeros(
            (len(self.instrument_model.target_grid), A_canonical.shape[1])
        )
        for j in range(A_canonical.shape[1]):
            A_instrument[:, j] = self.instrument_model.apply(
                A_canonical[:, j],
                self.canonical_model.canonical_grid,
            )

        # Apply preprocessing
        A_preprocessed = self.preprocessing.apply_to_matrix(A_instrument)

        return A_preprocessed

    @classmethod
    def create(
        cls,
        canonical_grid: np.ndarray,
        target_grid: np.ndarray,
        component_names: list[str],
        domain: str = "absorbance",
        preprocessing_type: str = "none",
        instrument_params: dict[str, float] | None = None,
        baseline_order: int = 5,
        continuum_order: int = 3,
        sg_window: int = 15,
        sg_polyorder: int = 2,
        include_environmental: bool = False,
    ) -> ForwardChain:
        """
        Convenience factory method to create ForwardChain.

        Args:
            canonical_grid: High-resolution canonical wavelength grid.
            target_grid: Target dataset wavelength grid.
            component_names: Names of components to include.
            domain: Domain type ('absorbance', 'reflectance').
            preprocessing_type: Preprocessing type.
            instrument_params: Instrument parameters dict.
            baseline_order: Baseline polynomial order.
            continuum_order: Continuum polynomial order.
            sg_window: Savitzky-Golay window.
            sg_polyorder: Savitzky-Golay polynomial order.
            include_environmental: Whether to include environmental effects model.

        Returns:
            Configured ForwardChain instance.
        """
        canonical_model = CanonicalForwardModel(
            canonical_grid=canonical_grid,
            component_names=component_names,
            baseline_order=baseline_order,
            continuum_order=continuum_order,
        )

        instrument_params = instrument_params or {}
        instrument_model = InstrumentModel.from_params(target_grid, instrument_params)

        domain_transform = DomainTransform(domain=domain)

        preprocessing = PreprocessingOperator.from_detection(
            preprocessing_type, sg_window, sg_polyorder
        )

        # Create environmental model if requested
        environmental_model = None
        if include_environmental:
            from .environmental import EnvironmentalEffectsModel
            environmental_model = EnvironmentalEffectsModel()

        return cls(
            canonical_model=canonical_model,
            instrument_model=instrument_model,
            domain_transform=domain_transform,
            preprocessing=preprocessing,
            environmental_model=environmental_model,
        )
