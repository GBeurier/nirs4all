"""
Spectral components for synthetic NIRS spectra generation.

This module provides the core building blocks for defining NIR absorption bands
and spectral components based on physical spectroscopy principles.

Classes:
    NIRBand: Represents a single NIR absorption band with Voigt profile.
    SpectralComponent: A chemical compound or functional group with multiple bands.
    ComponentLibrary: Collection of spectral components for generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import voigt_profile


@dataclass
class NIRBand:
    """
    Represents a single NIR absorption band.

    This class models an absorption band using a Voigt profile, which is
    the convolution of Gaussian (thermal broadening) and Lorentzian
    (pressure broadening) line shapes.

    Attributes:
        center: Central wavelength in nm.
        sigma: Gaussian width (standard deviation) in nm.
        gamma: Lorentzian width (HWHM) in nm. Use 0 for pure Gaussian.
        amplitude: Peak amplitude in absorbance units.
        name: Descriptive name of the band (e.g., "O-H 1st overtone").

    Example:
        >>> band = NIRBand(center=1450, sigma=25, gamma=3, amplitude=0.8)
        >>> wavelengths = np.arange(1400, 1500, 1)
        >>> spectrum = band.compute(wavelengths)
    """

    center: float
    sigma: float
    gamma: float = 0.0
    amplitude: float = 1.0
    name: str = ""

    def compute(self, wavelengths: np.ndarray) -> np.ndarray:
        """
        Compute the band profile at given wavelengths using Voigt profile.

        Args:
            wavelengths: Array of wavelengths in nm at which to evaluate the band.

        Returns:
            Array of absorbance values at each wavelength.

        Note:
            When gamma=0, a pure Gaussian profile is used for efficiency.
            Otherwise, the full Voigt profile (Gaussian ⊗ Lorentzian) is computed.
        """
        if self.gamma <= 0:
            # Pure Gaussian for efficiency
            return self.amplitude * np.exp(-0.5 * ((wavelengths - self.center) / self.sigma) ** 2)
        else:
            # Voigt profile (convolution of Gaussian and Lorentzian)
            return self.amplitude * voigt_profile(
                wavelengths - self.center, self.sigma, self.gamma
            ) * self.sigma * np.sqrt(2 * np.pi)


@dataclass
class SpectralComponent:
    """
    A spectral component representing a chemical compound or functional group.

    Each component consists of multiple absorption bands that together define
    the characteristic NIR signature of the compound.

    Attributes:
        name: Component name (e.g., "water", "protein", "lipid").
        bands: List of NIRBand objects defining the spectral signature.
        correlation_group: Optional group ID for components that should have
            correlated concentrations (e.g., protein and nitrogen compounds).

    Example:
        >>> water = SpectralComponent(
        ...     name="water",
        ...     bands=[
        ...         NIRBand(center=1450, sigma=25, gamma=3, amplitude=0.8),
        ...         NIRBand(center=1940, sigma=30, gamma=4, amplitude=1.0),
        ...     ],
        ...     correlation_group=1
        ... )
        >>> wavelengths = np.arange(1000, 2500, 2)
        >>> spectrum = water.compute(wavelengths)
    """

    name: str
    bands: List[NIRBand] = field(default_factory=list)
    correlation_group: Optional[int] = None

    def compute(self, wavelengths: np.ndarray) -> np.ndarray:
        """
        Compute the full component spectrum by summing all bands.

        Args:
            wavelengths: Array of wavelengths in nm at which to evaluate.

        Returns:
            Array of absorbance values representing the combined spectrum.
        """
        spectrum = np.zeros_like(wavelengths, dtype=np.float64)
        for band in self.bands:
            spectrum += band.compute(wavelengths)
        return spectrum


class ComponentLibrary:
    """
    Library of spectral components for synthetic NIRS generation.

    Supports both predefined components (based on known NIR band assignments)
    and programmatically generated random components for research purposes.

    Attributes:
        rng: NumPy random generator for reproducibility.

    Example:
        >>> # Create from predefined components
        >>> library = ComponentLibrary.from_predefined(
        ...     ["water", "protein", "lipid"],
        ...     random_state=42
        ... )
        >>>
        >>> # Or generate random components
        >>> library = ComponentLibrary(random_state=42)
        >>> library.generate_random_library(n_components=5)
        >>>
        >>> # Compute all component spectra
        >>> wavelengths = np.arange(1000, 2500, 2)
        >>> E = library.compute_all(wavelengths)  # shape: (n_components, n_wavelengths)
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        """
        Initialize the component library.

        Args:
            random_state: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(random_state)
        self._components: Dict[str, SpectralComponent] = {}

    @classmethod
    def from_predefined(
        cls,
        component_names: Optional[List[str]] = None,
        random_state: Optional[int] = None,
    ) -> ComponentLibrary:
        """
        Create a library from predefined spectral components.

        Args:
            component_names: List of component names to include.
                If None, includes all predefined components.
            random_state: Random seed for reproducibility.

        Returns:
            ComponentLibrary instance populated with predefined components.

        Raises:
            ValueError: If an unknown component name is specified.

        Example:
            >>> library = ComponentLibrary.from_predefined(
            ...     ["water", "protein", "lipid"]
            ... )
        """
        from ._constants import get_predefined_components

        library = cls(random_state=random_state)
        predefined = get_predefined_components()

        if component_names is None:
            component_names = list(predefined.keys())

        for name in component_names:
            if name in predefined:
                library._components[name] = predefined[name]
            else:
                available = list(predefined.keys())
                raise ValueError(
                    f"Unknown predefined component: '{name}'. "
                    f"Available components: {available}"
                )

        return library

    def add_component(self, component: SpectralComponent) -> ComponentLibrary:
        """
        Add a spectral component to the library.

        Args:
            component: SpectralComponent to add.

        Returns:
            Self for method chaining.
        """
        self._components[component.name] = component
        return self

    def add_random_component(
        self,
        name: str,
        n_bands: int = 3,
        wavelength_range: Tuple[float, float] = (1000, 2500),
        zones: Optional[List[Tuple[float, float]]] = None,
    ) -> SpectralComponent:
        """
        Generate and add a random spectral component.

        Creates a component with randomly placed absorption bands within
        the specified wavelength range or zones.

        Args:
            name: Component name.
            n_bands: Number of absorption bands to generate.
            wavelength_range: Overall wavelength range for band placement.
            zones: Optional list of (min, max) wavelength zones for band centers.
                If None, uses default NIR-relevant zones.

        Returns:
            The generated SpectralComponent.

        Example:
            >>> library = ComponentLibrary(random_state=42)
            >>> component = library.add_random_component(
            ...     "random_compound",
            ...     n_bands=4,
            ...     wavelength_range=(1000, 2500)
            ... )
        """
        from ._constants import DEFAULT_NIR_ZONES

        if zones is None:
            zones = DEFAULT_NIR_ZONES

        bands = []
        for i in range(n_bands):
            zone = zones[self.rng.integers(0, len(zones))]
            center = self.rng.uniform(*zone)
            sigma = self.rng.uniform(10, 30)
            gamma = self.rng.uniform(0, 5)
            amplitude = self.rng.lognormal(mean=-0.5, sigma=0.5)

            bands.append(
                NIRBand(
                    center=center,
                    sigma=sigma,
                    gamma=gamma,
                    amplitude=amplitude,
                    name=f"band_{i}",
                )
            )

        component = SpectralComponent(name=name, bands=bands)
        self._components[name] = component
        return component

    def generate_random_library(
        self,
        n_components: int = 5,
        n_bands_range: Tuple[int, int] = (2, 6),
    ) -> ComponentLibrary:
        """
        Generate a library of random spectral components.

        Args:
            n_components: Number of components to generate.
            n_bands_range: Range (min, max) for number of bands per component.

        Returns:
            Self for method chaining.

        Example:
            >>> library = ComponentLibrary(random_state=42)
            >>> library.generate_random_library(n_components=5, n_bands_range=(2, 5))
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

    @property
    def component_names(self) -> List[str]:
        """Get list of component names in order."""
        return list(self._components.keys())

    def compute_all(self, wavelengths: np.ndarray) -> np.ndarray:
        """
        Compute spectra for all components at given wavelengths.

        Args:
            wavelengths: Array of wavelengths in nm.

        Returns:
            Array of shape (n_components, n_wavelengths) containing
            the spectrum of each component.

        Example:
            >>> library = ComponentLibrary.from_predefined(["water", "protein"])
            >>> wavelengths = np.arange(1000, 2500, 2)
            >>> E = library.compute_all(wavelengths)
            >>> print(E.shape)
            (2, 751)
        """
        return np.array([comp.compute(wavelengths) for comp in self._components.values()])

    def __len__(self) -> int:
        """Return number of components."""
        return self.n_components

    def __iter__(self):
        """Iterate over components."""
        return iter(self._components.values())

    def __getitem__(self, name: str) -> SpectralComponent:
        """Get component by name."""
        return self._components[name]

    def __contains__(self, name: str) -> bool:
        """Check if component exists by name."""
        return name in self._components
