"""
Synthetic NIRS Spectra Generator
================================

A physically-motivated synthetic NIRS spectra generator based on Beer-Lambert law,
with realistic instrumental effects and noise models.

Key features:
- Voigt profile peak shapes (Gaussian + Lorentzian convolution)
- Realistic NIR band positions from known spectroscopic databases
- Configurable baseline, scattering, and instrumental effects
- Batch/session effects for domain adaptation research
- Controllable outlier/artifact generation

References:
- Workman Jr, J., & Weyer, L. (2012). Practical Guide and Spectral Atlas for
  Interpretive Near-Infrared Spectroscopy. CRC Press.
- Burns, D. A., & Ciurczak, E. W. (2007). Handbook of Near-Infrared Analysis.
  CRC Press.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import voigt_profile


@dataclass
class NIRBand:
    """
    Represents a single NIR absorption band.

    Attributes:
        center: Central wavelength in nm
        sigma: Gaussian width in nm
        gamma: Lorentzian width in nm (for Voigt profile)
        amplitude: Peak amplitude (absorbance units)
        name: Descriptive name of the band
    """
    center: float
    sigma: float
    gamma: float = 0.0  # 0 = pure Gaussian
    amplitude: float = 1.0
    name: str = ""

    def compute(self, wavelengths: np.ndarray) -> np.ndarray:
        """Compute the band profile at given wavelengths using Voigt profile."""
        if self.gamma <= 0:
            # Pure Gaussian
            return self.amplitude * np.exp(-0.5 * ((wavelengths - self.center) / self.sigma) ** 2)
        else:
            # Voigt profile (convolution of Gaussian and Lorentzian)
            # scipy.special.voigt_profile uses sigma and gamma in different units
            return self.amplitude * voigt_profile(
                wavelengths - self.center, self.sigma, self.gamma
            ) * self.sigma * np.sqrt(2 * np.pi)


@dataclass
class SpectralComponent:
    """
    A spectral component representing a chemical compound or functional group.

    Attributes:
        name: Component name (e.g., "water", "protein", "lipid")
        bands: List of NIRBand objects defining the spectral signature
        correlation_group: Optional group ID for correlated components
    """
    name: str
    bands: List[NIRBand] = field(default_factory=list)
    correlation_group: Optional[int] = None

    def compute(self, wavelengths: np.ndarray) -> np.ndarray:
        """Compute the full component spectrum by summing all bands."""
        spectrum = np.zeros_like(wavelengths, dtype=np.float64)
        for band in self.bands:
            spectrum += band.compute(wavelengths)
        return spectrum


# ============================================================================
# Predefined Spectral Components based on NIR band assignments
# ============================================================================
# Reference: Workman & Weyer, "Practical Guide to Interpretive Near-Infrared
# Spectroscopy" (2012), Table of characteristic NIR absorption bands
# ============================================================================

PREDEFINED_COMPONENTS: Dict[str, SpectralComponent] = {
    "water": SpectralComponent(
        name="water",
        bands=[
            NIRBand(center=1450, sigma=25, gamma=3, amplitude=0.8, name="O-H 1st overtone"),
            NIRBand(center=1940, sigma=30, gamma=4, amplitude=1.0, name="O-H combination"),
            NIRBand(center=2500, sigma=50, gamma=5, amplitude=0.3, name="O-H stretch + bend"),
        ],
        correlation_group=1
    ),
    "protein": SpectralComponent(
        name="protein",
        bands=[
            NIRBand(center=1510, sigma=20, gamma=2, amplitude=0.5, name="N-H 1st overtone"),
            NIRBand(center=1680, sigma=25, gamma=3, amplitude=0.4, name="C-H aromatic"),
            NIRBand(center=2050, sigma=30, gamma=3, amplitude=0.6, name="N-H combination"),
            NIRBand(center=2180, sigma=25, gamma=2, amplitude=0.5, name="Protein C-H"),
            NIRBand(center=2300, sigma=20, gamma=2, amplitude=0.3, name="N-H+Amide III"),
        ],
        correlation_group=2
    ),
    "lipid": SpectralComponent(
        name="lipid",
        bands=[
            NIRBand(center=1210, sigma=20, gamma=2, amplitude=0.4, name="C-H 2nd overtone"),
            NIRBand(center=1390, sigma=15, gamma=1, amplitude=0.3, name="C-H combination"),
            NIRBand(center=1720, sigma=25, gamma=2, amplitude=0.6, name="C-H 1st overtone"),
            NIRBand(center=2310, sigma=20, gamma=2, amplitude=0.5, name="CH2 combination"),
            NIRBand(center=2350, sigma=18, gamma=2, amplitude=0.4, name="CH3 combination"),
        ],
        correlation_group=3
    ),
    "starch": SpectralComponent(
        name="starch",
        bands=[
            NIRBand(center=1460, sigma=25, gamma=3, amplitude=0.5, name="O-H 1st overtone"),
            NIRBand(center=1580, sigma=20, gamma=2, amplitude=0.3, name="Starch combination"),
            NIRBand(center=2100, sigma=30, gamma=3, amplitude=0.6, name="O-H+C-O combination"),
            NIRBand(center=2270, sigma=25, gamma=2, amplitude=0.4, name="C-O+C-C stretch"),
        ],
        correlation_group=4
    ),
    "cellulose": SpectralComponent(
        name="cellulose",
        bands=[
            NIRBand(center=1490, sigma=22, gamma=2, amplitude=0.4, name="O-H 1st overtone"),
            NIRBand(center=1780, sigma=18, gamma=2, amplitude=0.3, name="Cellulose C-H"),
            NIRBand(center=2090, sigma=28, gamma=3, amplitude=0.5, name="O-H combination"),
            NIRBand(center=2280, sigma=22, gamma=2, amplitude=0.4, name="Cellulose C-O"),
            NIRBand(center=2340, sigma=20, gamma=2, amplitude=0.35, name="C-H combination"),
        ],
        correlation_group=4
    ),
    "chlorophyll": SpectralComponent(
        name="chlorophyll",
        bands=[
            NIRBand(center=1070, sigma=15, gamma=1, amplitude=0.3, name="Chl absorption"),
            NIRBand(center=1400, sigma=20, gamma=2, amplitude=0.4, name="C-H 1st overtone"),
            NIRBand(center=2270, sigma=22, gamma=2, amplitude=0.35, name="C-H combination"),
        ],
        correlation_group=5
    ),
    "oil": SpectralComponent(
        name="oil",
        bands=[
            NIRBand(center=1165, sigma=18, gamma=2, amplitude=0.35, name="C-H 2nd overtone"),
            NIRBand(center=1215, sigma=16, gamma=1.5, amplitude=0.3, name="CH2 2nd overtone"),
            NIRBand(center=1410, sigma=20, gamma=2, amplitude=0.45, name="C-H combination"),
            NIRBand(center=1725, sigma=22, gamma=2, amplitude=0.7, name="C-H 1st overtone"),
            NIRBand(center=2140, sigma=25, gamma=2, amplitude=0.4, name="C=C unsaturation"),
            NIRBand(center=2305, sigma=18, gamma=2, amplitude=0.5, name="CH2 combination"),
        ],
        correlation_group=3
    ),
    "nitrogen_compound": SpectralComponent(
        name="nitrogen_compound",
        bands=[
            NIRBand(center=1500, sigma=18, gamma=2, amplitude=0.45, name="N-H 1st overtone"),
            NIRBand(center=2060, sigma=25, gamma=2, amplitude=0.5, name="N-H combination"),
            NIRBand(center=2150, sigma=22, gamma=2, amplitude=0.4, name="N-H+C-N"),
        ],
        correlation_group=2
    ),
}


class ComponentLibrary:
    """
    Library of spectral components that can be used for synthetic generation.

    Supports both predefined components and custom component creation.
    """

    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the component library.

        Args:
            random_state: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(random_state)
        self._components: Dict[str, SpectralComponent] = {}

    @classmethod
    def from_predefined(cls, component_names: Optional[List[str]] = None,
                        random_state: Optional[int] = None) -> 'ComponentLibrary':
        """
        Create a library from predefined components.

        Args:
            component_names: List of component names to include.
                           If None, includes all predefined components.
            random_state: Random seed

        Returns:
            ComponentLibrary instance
        """
        library = cls(random_state=random_state)
        if component_names is None:
            component_names = list(PREDEFINED_COMPONENTS.keys())

        for name in component_names:
            if name in PREDEFINED_COMPONENTS:
                library._components[name] = PREDEFINED_COMPONENTS[name]
            else:
                raise ValueError(f"Unknown predefined component: {name}")

        return library

    def add_random_component(self, name: str, n_bands: int = 3,
                             wavelength_range: Tuple[float, float] = (1000, 2500),
                             zones: Optional[List[Tuple[float, float]]] = None) -> SpectralComponent:
        """
        Generate and add a random spectral component.

        Args:
            name: Component name
            n_bands: Number of absorption bands
            wavelength_range: Wavelength range for band placement
            zones: Optional list of (min, max) zones for band centers

        Returns:
            The generated SpectralComponent
        """
        if zones is None:
            # Default NIR-relevant zones
            zones = [
                (1100, 1300),  # 2nd overtones
                (1400, 1550),  # 1st overtones O-H, N-H
                (1650, 1800),  # 1st overtones C-H
                (1850, 2000),  # Combination O-H
                (2000, 2200),  # Combination N-H
                (2200, 2400),  # Combination C-H
            ]

        bands = []
        for _ in range(n_bands):
            zone = zones[self.rng.integers(0, len(zones))]
            center = self.rng.uniform(*zone)
            sigma = self.rng.uniform(10, 30)
            gamma = self.rng.uniform(0, 5)
            amplitude = self.rng.lognormal(mean=-0.5, sigma=0.5)

            bands.append(NIRBand(
                center=center,
                sigma=sigma,
                gamma=gamma,
                amplitude=amplitude,
                name=f"band_{len(bands)}"
            ))

        component = SpectralComponent(name=name, bands=bands)
        self._components[name] = component
        return component

    def generate_random_library(self, n_components: int = 5,
                                n_bands_range: Tuple[int, int] = (2, 6)) -> 'ComponentLibrary':
        """
        Generate a library of random spectral components.

        Args:
            n_components: Number of components to generate
            n_bands_range: Range for number of bands per component

        Returns:
            Self for chaining
        """
        for i in range(n_components):
            n_bands = self.rng.integers(*n_bands_range)
            self.add_random_component(f"component_{i}", n_bands=n_bands)
        return self

    @property
    def components(self) -> Dict[str, SpectralComponent]:
        """Get all components in the library."""
        return self._components

    @property
    def n_components(self) -> int:
        """Number of components in the library."""
        return len(self._components)

    def compute_all(self, wavelengths: np.ndarray) -> np.ndarray:
        """
        Compute spectra for all components.

        Args:
            wavelengths: Wavelength grid

        Returns:
            Array of shape (n_components, n_wavelengths)
        """
        return np.array([
            comp.compute(wavelengths) for comp in self._components.values()
        ])

    @property
    def component_names(self) -> List[str]:
        """Get list of component names."""
        return list(self._components.keys())


class SyntheticNIRSGenerator:
    """
    Generator for synthetic NIRS spectra with realistic instrumental effects.

    This generator implements a physically-motivated model based on Beer-Lambert law
    with additional effects for baseline, scattering, instrumental response, and noise.

    Model:
        A_i(λ) = L_i * Σ_k c_ik * ε_k(λ) + baseline_i(λ) + scatter_i(λ) + noise_i(λ)

    where:
        - c_ik: concentration of component k in sample i
        - ε_k(λ): molar absorptivity of component k (Voigt profiles)
        - L_i: optical path length factor
        - baseline: polynomial baseline drift
        - scatter: multiplicative/additive scattering effects
        - noise: wavelength-dependent Gaussian noise

    Args:
        wavelength_start: Start wavelength in nm (default: 1000)
        wavelength_end: End wavelength in nm (default: 2500)
        wavelength_step: Wavelength step in nm (default: 2)
        component_library: Optional ComponentLibrary (generates random if None)
        complexity: Complexity level ('simple', 'realistic', 'complex')
        random_state: Random seed for reproducibility

    Example:
        >>> generator = SyntheticNIRSGenerator(random_state=42)
        >>> X, Y, E = generator.generate(n_samples=1000)
        >>> print(X.shape, Y.shape, E.shape)
        (1000, 751) (1000, 5) (5, 751)
    """

    def __init__(
        self,
        wavelength_start: float = 1000,
        wavelength_end: float = 2500,
        wavelength_step: float = 2,
        component_library: Optional[ComponentLibrary] = None,
        complexity: str = "realistic",
        random_state: Optional[int] = None,
    ):
        self.wavelength_start = wavelength_start
        self.wavelength_end = wavelength_end
        self.wavelength_step = wavelength_step
        self.complexity = complexity
        self.rng = np.random.default_rng(random_state)

        # Generate wavelength grid
        self.wavelengths = np.arange(
            wavelength_start, wavelength_end + wavelength_step, wavelength_step
        )
        self.n_wavelengths = len(self.wavelengths)

        # Set up component library
        if component_library is not None:
            self.library = component_library
        else:
            # Use predefined components for realistic mode
            if complexity in ("realistic", "complex"):
                self.library = ComponentLibrary.from_predefined(
                    ["water", "protein", "lipid", "starch", "cellulose"],
                    random_state=random_state
                )
            else:
                self.library = ComponentLibrary(random_state=random_state)
                self.library.generate_random_library(n_components=5)

        # Precompute component spectra
        self.E = self.library.compute_all(self.wavelengths)

        # Set complexity-dependent parameters
        self._set_complexity_params()

    def _set_complexity_params(self):
        """Set parameters based on complexity level."""
        if self.complexity == "simple":
            self.params = {
                "path_length_std": 0.02,
                "baseline_amplitude": 0.01,
                "scatter_alpha_std": 0.02,
                "scatter_beta_std": 0.005,
                "tilt_std": 0.005,
                "global_slope_mean": 0.0,  # Mean global slope (absorbance per 1000nm)
                "global_slope_std": 0.02,  # Std of global slope variation
                "shift_std": 0.2,
                "stretch_std": 0.0005,
                "instrumental_fwhm": 4,
                "noise_base": 0.002,
                "noise_signal_dep": 0.005,
                "artifact_prob": 0.0,
            }
        elif self.complexity == "realistic":
            self.params = {
                "path_length_std": 0.05,
                "baseline_amplitude": 0.02,
                "scatter_alpha_std": 0.05,
                "scatter_beta_std": 0.01,
                "tilt_std": 0.01,
                "global_slope_mean": 0.05,  # Typical upward slope in NIR
                "global_slope_std": 0.03,
                "shift_std": 0.5,
                "stretch_std": 0.001,
                "instrumental_fwhm": 8,
                "noise_base": 0.005,
                "noise_signal_dep": 0.01,
                "artifact_prob": 0.02,
            }
        elif self.complexity == "complex":
            self.params = {
                "path_length_std": 0.08,
                "baseline_amplitude": 0.05,
                "scatter_alpha_std": 0.08,
                "scatter_beta_std": 0.02,
                "tilt_std": 0.02,
                "global_slope_mean": 0.08,
                "global_slope_std": 0.05,
                "shift_std": 1.0,
                "stretch_std": 0.002,
                "instrumental_fwhm": 12,
                "noise_base": 0.008,
                "noise_signal_dep": 0.015,
                "artifact_prob": 0.05,
            }
        else:
            raise ValueError(f"Unknown complexity level: {self.complexity}")

    def generate_concentrations(
        self,
        n_samples: int,
        method: str = "dirichlet",
        alpha: Optional[np.ndarray] = None,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate concentration matrix.

        Args:
            n_samples: Number of samples
            method: Generation method ('dirichlet', 'uniform', 'lognormal', 'correlated')
            alpha: Dirichlet concentration parameters
            correlation_matrix: For 'correlated' method

        Returns:
            Concentration matrix of shape (n_samples, n_components)
        """
        n_components = self.library.n_components

        if method == "dirichlet":
            if alpha is None:
                alpha = np.ones(n_components) * 2.0
            C = self.rng.dirichlet(alpha, size=n_samples)

        elif method == "uniform":
            C = self.rng.uniform(0, 1, size=(n_samples, n_components))

        elif method == "lognormal":
            C = self.rng.lognormal(mean=0, sigma=0.5, size=(n_samples, n_components))
            C = C / C.sum(axis=1, keepdims=True)  # Normalize

        elif method == "correlated":
            # Generate correlated concentrations using Cholesky decomposition
            if correlation_matrix is None:
                # Create a default correlation structure based on component groups
                correlation_matrix = np.eye(n_components)
                # Add some cross-correlations
                for i in range(n_components):
                    for j in range(i + 1, n_components):
                        corr = self.rng.uniform(-0.3, 0.5)
                        correlation_matrix[i, j] = corr
                        correlation_matrix[j, i] = corr

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(correlation_matrix)
            eigvals = np.maximum(eigvals, 0.01)
            correlation_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

            L = np.linalg.cholesky(correlation_matrix)
            Z = self.rng.standard_normal((n_samples, n_components))
            C = Z @ L.T
            # Transform to positive values
            C = np.abs(C)
            C = C / C.sum(axis=1, keepdims=True)

        else:
            raise ValueError(f"Unknown concentration method: {method}")

        return C

    def _apply_beer_lambert(self, C: np.ndarray) -> np.ndarray:
        """Apply Beer-Lambert law: A = C @ E."""
        return C @ self.E

    def _apply_path_length(self, A: np.ndarray) -> np.ndarray:
        """Apply random path length variation."""
        n_samples = A.shape[0]
        L = self.rng.normal(1.0, self.params["path_length_std"], size=n_samples)
        L = np.maximum(L, 0.5)  # Ensure positive
        return A * L[:, np.newaxis]

    def _generate_baseline(self, n_samples: int) -> np.ndarray:
        """Generate polynomial baseline drift."""
        x = (self.wavelengths - self.wavelengths.mean()) / (np.ptp(self.wavelengths) / 2)
        amp = self.params["baseline_amplitude"]

        baseline = np.zeros((n_samples, self.n_wavelengths))
        for i in range(n_samples):
            b0 = self.rng.normal(0, amp)
            b1 = self.rng.normal(0, amp * 0.5)
            b2 = self.rng.normal(0, amp * 0.3)
            b3 = self.rng.normal(0, amp * 0.1)
            baseline[i] = b0 + b1 * x + b2 * x**2 + b3 * x**3

        return baseline

    def _apply_global_slope(self, A: np.ndarray) -> np.ndarray:
        """
        Apply global slope effect commonly observed in NIR spectra.

        This simulates the typical upward trend in absorbance with increasing wavelength,
        caused by scattering effects (particle size, surface roughness) and instrumental
        factors. The slope varies between samples.

        The slope is defined as absorbance change per 1000nm.
        """
        n_samples = A.shape[0]

        # Normalized wavelength (0 to 1 across the range)
        wl_range = np.ptp(self.wavelengths)
        x_norm = (self.wavelengths - self.wavelengths.min()) / wl_range

        # Generate slopes: mean + sample-specific variation
        slope_mean = self.params["global_slope_mean"]
        slope_std = self.params["global_slope_std"]
        slopes = self.rng.normal(slope_mean, slope_std, size=n_samples)

        # Scale to actual wavelength range (slope is per 1000nm)
        scale_factor = wl_range / 1000.0

        # Apply slope to each sample
        for i in range(n_samples):
            A[i] += slopes[i] * scale_factor * x_norm

        return A

    def _apply_scatter(self, A: np.ndarray) -> np.ndarray:
        """Apply multiplicative/additive scatter effects."""
        n_samples = A.shape[0]

        # Multiplicative scatter (SNV/MSC-like before correction)
        alpha = self.rng.normal(1.0, self.params["scatter_alpha_std"], size=n_samples)
        alpha = np.maximum(alpha, 0.7)  # Ensure positive

        # Additive offset
        beta = self.rng.normal(0, self.params["scatter_beta_std"], size=n_samples)

        # Apply
        A_scattered = A * alpha[:, np.newaxis] + beta[:, np.newaxis]

        # Add tilt
        x = (self.wavelengths - self.wavelengths.mean()) / np.ptp(self.wavelengths)
        gamma = self.rng.normal(0, self.params["tilt_std"], size=n_samples)
        tilt = gamma[:, np.newaxis] * x[np.newaxis, :]
        A_scattered += tilt

        return A_scattered

    def _apply_wavelength_shift(self, A: np.ndarray) -> np.ndarray:
        """Apply wavelength calibration shifts/stretches via interpolation."""
        n_samples = A.shape[0]

        shifts = self.rng.normal(0, self.params["shift_std"], size=n_samples)
        stretches = self.rng.normal(1.0, self.params["stretch_std"], size=n_samples)

        A_shifted = np.zeros_like(A)
        for i in range(n_samples):
            # New wavelength grid after shift/stretch
            wl_shifted = stretches[i] * self.wavelengths + shifts[i]
            # Interpolate back to original grid
            A_shifted[i] = np.interp(self.wavelengths, wl_shifted, A[i])

        return A_shifted

    def _apply_instrumental_response(self, A: np.ndarray) -> np.ndarray:
        """Apply instrumental broadening (Gaussian convolution)."""
        fwhm = self.params["instrumental_fwhm"]
        # Convert FWHM to sigma in wavelength step units
        sigma_wl = fwhm / (2 * np.sqrt(2 * np.log(2)))
        sigma_pts = sigma_wl / self.wavelength_step

        A_convolved = np.zeros_like(A)
        for i in range(A.shape[0]):
            A_convolved[i] = gaussian_filter1d(A[i], sigma_pts)

        return A_convolved

    def _add_noise(self, A: np.ndarray) -> np.ndarray:
        """Add wavelength-dependent Gaussian noise."""
        sigma_base = self.params["noise_base"]
        sigma_signal = self.params["noise_signal_dep"]

        # Heteroscedastic noise: higher noise at higher absorbance
        sigma = sigma_base + sigma_signal * np.abs(A)
        noise = self.rng.normal(0, sigma)

        return A + noise

    def _add_artifacts(self, A: np.ndarray, metadata: dict) -> np.ndarray:
        """Add random artifacts (spikes, dead bands)."""
        n_samples = A.shape[0]
        artifact_prob = self.params["artifact_prob"]

        artifact_types = []
        for i in range(n_samples):
            if self.rng.random() < artifact_prob:
                artifact_type = self.rng.choice(["spike", "dead_band", "saturation"])
                artifact_types.append(artifact_type)

                if artifact_type == "spike":
                    # Random spike at 1-3 points
                    n_spikes = self.rng.integers(1, 4)
                    spike_indices = self.rng.choice(self.n_wavelengths, n_spikes, replace=False)
                    spike_values = self.rng.uniform(0.5, 1.5, n_spikes)
                    A[i, spike_indices] += spike_values * np.sign(self.rng.standard_normal(n_spikes))

                elif artifact_type == "dead_band":
                    # Region with increased noise (detector issue)
                    start_idx = self.rng.integers(0, self.n_wavelengths - 20)
                    width = self.rng.integers(10, 30)
                    end_idx = min(start_idx + width, self.n_wavelengths)
                    A[i, start_idx:end_idx] += self.rng.normal(0, 0.05, end_idx - start_idx)

                elif artifact_type == "saturation":
                    # Clipping at high absorbance
                    threshold = self.rng.uniform(0.8, 1.2)
                    A[i] = np.clip(A[i], -np.inf, threshold)
            else:
                artifact_types.append(None)

        metadata["artifact_types"] = artifact_types
        return A

    def generate_batch_effects(
        self,
        n_batches: int,
        samples_per_batch: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate batch/session effects for domain adaptation research.

        Args:
            n_batches: Number of measurement batches/sessions
            samples_per_batch: List of sample counts per batch

        Returns:
            Tuple of (batch_offsets, batch_gains) arrays
        """
        batch_offsets = []
        batch_gains = []

        for _ in range(n_batches):
            # Baseline offset per batch (slow drift)
            x = (self.wavelengths - self.wavelengths.mean()) / np.ptp(self.wavelengths)
            offset = self.rng.normal(0, 0.02) + self.rng.normal(0, 0.01) * x
            batch_offsets.append(offset)

            # Gain variation per batch
            gain = self.rng.normal(1.0, 0.03)
            batch_gains.append(gain)

        return np.array(batch_offsets), np.array(batch_gains)

    def generate(
        self,
        n_samples: int = 1000,
        concentration_method: str = "dirichlet",
        include_batch_effects: bool = False,
        n_batches: int = 1,
        return_metadata: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray, np.ndarray, dict]]:
        """
        Generate synthetic NIRS spectra.

        Args:
            n_samples: Number of spectra to generate
            concentration_method: Method for generating concentrations
                ('dirichlet', 'uniform', 'lognormal', 'correlated')
            include_batch_effects: Whether to add batch/session effects
            n_batches: Number of batches (only if include_batch_effects=True)
            return_metadata: Whether to return additional metadata

        Returns:
            Tuple of:
                - X: Spectra matrix (n_samples, n_wavelengths)
                - Y: Concentration matrix (n_samples, n_components)
                - E: Component spectra (n_components, n_wavelengths)
                - metadata: (optional) Dictionary with generation details
        """
        metadata = {
            "n_samples": n_samples,
            "n_components": self.library.n_components,
            "n_wavelengths": self.n_wavelengths,
            "component_names": self.library.component_names,
            "wavelengths": self.wavelengths.copy(),
            "complexity": self.complexity,
            "concentration_method": concentration_method,
        }

        # 1. Generate concentrations
        C = self.generate_concentrations(n_samples, method=concentration_method)

        # 2. Apply Beer-Lambert law
        A = self._apply_beer_lambert(C)

        # 3. Apply path length variation
        A = self._apply_path_length(A)

        # 4. Generate and add baseline
        baseline = self._generate_baseline(n_samples)
        A = A + baseline

        # 5. Apply global slope (typical NIR upward trend)
        A = self._apply_global_slope(A)

        # 6. Apply scatter effects
        A = self._apply_scatter(A)

        # 7. Apply batch effects if requested
        if include_batch_effects and n_batches > 1:
            samples_per_batch = [n_samples // n_batches] * n_batches
            samples_per_batch[-1] += n_samples % n_batches

            batch_offsets, batch_gains = self.generate_batch_effects(n_batches, samples_per_batch)

            batch_ids = []
            idx = 0
            for batch_id, n_in_batch in enumerate(samples_per_batch):
                batch_ids.extend([batch_id] * n_in_batch)
                A[idx:idx + n_in_batch] = A[idx:idx + n_in_batch] * batch_gains[batch_id] + batch_offsets[batch_id]
                idx += n_in_batch

            metadata["batch_ids"] = np.array(batch_ids)
            metadata["batch_offsets"] = batch_offsets
            metadata["batch_gains"] = batch_gains

        # 8. Apply wavelength shift/stretch
        A = self._apply_wavelength_shift(A)

        # 9. Apply instrumental response
        A = self._apply_instrumental_response(A)

        # 10. Add noise
        A = self._add_noise(A)

        # 10. Add artifacts
        A = self._add_artifacts(A, metadata)

        if return_metadata:
            return A, C, self.E, metadata
        else:
            return A, C, self.E

    def create_dataset(
        self,
        n_train: int = 800,
        n_test: int = 200,
        target_component: Optional[Union[str, int]] = None,
        **generate_kwargs,
    ) -> 'SpectroDataset':
        """
        Create a SpectroDataset from synthetic spectra.

        Args:
            n_train: Number of training samples
            n_test: Number of test samples
            target_component: Which component to use as target (name or index).
                            If None, uses all components as multi-output target.
            **generate_kwargs: Additional arguments for generate()

        Returns:
            SpectroDataset ready for pipeline use
        """
        from nirs4all.data import SpectroDataset

        # Generate all samples
        n_total = n_train + n_test
        X, C, E = self.generate(n_samples=n_total, **generate_kwargs)

        # Determine target
        if target_component is None:
            y = C
        elif isinstance(target_component, str):
            comp_idx = self.library.component_names.index(target_component)
            y = C[:, comp_idx]
        else:
            y = C[:, target_component]

        # Create dataset
        dataset = SpectroDataset(name="synthetic_nirs")

        # Create wavelength headers
        headers = [str(int(wl)) for wl in self.wavelengths]

        # Add training samples
        dataset.add_samples(X[:n_train], indexes={"partition": "train"}, headers=headers, header_unit="nm")
        dataset.add_targets(y[:n_train])

        # Add test samples
        dataset.add_samples(X[n_train:], indexes={"partition": "test"}, headers=headers, header_unit="nm")
        dataset.add_targets(y[n_train:])

        return dataset
