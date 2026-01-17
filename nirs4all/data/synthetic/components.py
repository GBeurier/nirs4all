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
        category: Primary category (e.g., "carbohydrates", "proteins", "lipids").
        subcategory: More specific classification (e.g., "monosaccharides", "amino_acids").
        synonyms: Alternative names (e.g., ["vitamin C"] for ascorbic_acid).
        formula: Chemical formula (e.g., "C6H12O6" for glucose).
        cas_number: CAS registry number for chemical identification.
        references: Literature citations for band assignments.
        tags: Classification tags (e.g., ["food", "pharma", "agriculture"]).

    Example:
        >>> water = SpectralComponent(
        ...     name="water",
        ...     bands=[
        ...         NIRBand(center=1450, sigma=25, gamma=3, amplitude=0.8),
        ...         NIRBand(center=1940, sigma=30, gamma=4, amplitude=1.0),
        ...     ],
        ...     correlation_group=1,
        ...     category="water_related",
        ...     formula="H2O",
        ... )
        >>> wavelengths = np.arange(1000, 2500, 2)
        >>> spectrum = water.compute(wavelengths)
    """

    name: str
    bands: List[NIRBand] = field(default_factory=list)
    correlation_group: Optional[int] = None

    # Metadata fields (Phase 1 enhancement)
    category: str = ""
    subcategory: str = ""
    synonyms: List[str] = field(default_factory=list)
    formula: str = ""
    cas_number: str = ""
    references: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

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

    def validate(self) -> List[str]:
        """
        Validate component parameters.

        Returns:
            List of validation issues (empty if all valid).

        Example:
            >>> component = SpectralComponent(name="test", bands=[])
            >>> issues = component.validate()
            >>> if issues:
            ...     print("Issues found:", issues)
        """
        issues: List[str] = []

        # Check component has bands
        if not self.bands:
            issues.append(f"Component '{self.name}': no bands defined")
            return issues

        # Check band parameters
        for band in self.bands:
            if band.sigma <= 0:
                issues.append(f"Component '{self.name}', band '{band.name}': sigma must be positive (got {band.sigma})")
            if band.gamma < 0:
                issues.append(f"Component '{self.name}', band '{band.name}': gamma must be non-negative (got {band.gamma})")
            if band.amplitude < 0:
                issues.append(f"Component '{self.name}', band '{band.name}': amplitude must be non-negative (got {band.amplitude})")
            if not (200 < band.center < 3000):
                issues.append(f"Component '{self.name}', band '{band.name}': center {band.center} outside valid range (200-3000 nm)")

        return issues

    def is_normalized(self, tolerance: float = 0.01) -> bool:
        """
        Check if the component's band amplitudes are max-normalized (max amplitude = 1.0).

        Args:
            tolerance: Acceptable deviation from 1.0 for max amplitude.

        Returns:
            True if max amplitude is within tolerance of 1.0.
        """
        if not self.bands:
            return True
        max_amp = max(band.amplitude for band in self.bands)
        return abs(max_amp - 1.0) <= tolerance

    def normalized(self, method: str = "max") -> "SpectralComponent":
        """
        Return a new SpectralComponent with normalized band amplitudes.

        Args:
            method: Normalization method.
                - "max": Scale so max amplitude = 1.0 (default)
                - "sum": Scale so sum of amplitudes = 1.0

        Returns:
            New SpectralComponent with normalized amplitudes.

        Example:
            >>> component = SpectralComponent(name="test", bands=[
            ...     NIRBand(center=1450, sigma=25, amplitude=0.8),
            ...     NIRBand(center=1940, sigma=30, amplitude=2.0),
            ... ])
            >>> normalized = component.normalized()
            >>> print(max(b.amplitude for b in normalized.bands))  # 1.0
        """
        if not self.bands:
            return self

        amplitudes = [band.amplitude for band in self.bands]

        if method == "max":
            max_amp = max(amplitudes)
            if max_amp <= 0:
                return self
            factor = 1.0 / max_amp
        elif method == "sum":
            total = sum(amplitudes)
            if total <= 0:
                return self
            factor = 1.0 / total
        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'max' or 'sum'.")

        normalized_bands = [
            NIRBand(
                center=band.center,
                sigma=band.sigma,
                gamma=band.gamma,
                amplitude=band.amplitude * factor,
                name=band.name,
            )
            for band in self.bands
        ]

        return SpectralComponent(
            name=self.name,
            bands=normalized_bands,
            correlation_group=self.correlation_group,
            category=self.category,
            subcategory=self.subcategory,
            synonyms=self.synonyms.copy() if self.synonyms else [],
            formula=self.formula,
            cas_number=self.cas_number,
            references=self.references.copy() if self.references else [],
            tags=self.tags.copy() if self.tags else [],
        )

    def has_bands_in_range(self, wavelength_range: Tuple[float, float]) -> bool:
        """
        Check if component has any bands with centers in the given wavelength range.

        Args:
            wavelength_range: (min, max) wavelength in nm.

        Returns:
            True if at least one band center is within the range.
        """
        low, high = wavelength_range
        return any(low <= band.center <= high for band in self.bands)

    def info(self) -> str:
        """
        Return formatted information about the component.

        Returns:
            Human-readable string with component details.
        """
        lines = [
            f"Component: {self.name}",
            f"Category: {self.category or 'N/A'}",
            f"Subcategory: {self.subcategory or 'N/A'}",
            f"Formula: {self.formula or 'N/A'}",
        ]
        if self.synonyms:
            lines.append(f"Synonyms: {', '.join(self.synonyms)}")
        if self.cas_number:
            lines.append(f"CAS: {self.cas_number}")
        lines.append(f"Bands ({len(self.bands)}):")
        for band in sorted(self.bands, key=lambda b: b.center):
            lines.append(f"  - {band.center:.0f} nm: {band.name or 'unnamed'} (amp={band.amplitude:.2f})")
        if self.references:
            lines.append("References:")
            for ref in self.references:
                lines.append(f"  - {ref}")
        if self.tags:
            lines.append(f"Tags: {', '.join(self.tags)}")
        return "\n".join(lines)


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

    def add_boundary_component(
        self,
        name: str,
        measurement_range: Tuple[float, float] = (1000, 2500),
        edge: str = "both",
        n_bands: int = 1,
        amplitude_range: Tuple[float, float] = (0.3, 1.0),
        width_range: Tuple[float, float] = (50, 200),
        offset_range: Tuple[float, float] = (0.3, 1.5),
    ) -> SpectralComponent:
        """
        Generate a component with bands outside the measurement range.

        This creates "boundary" or "truncated" peaks - absorption bands whose
        centers lie outside the measured wavelength range, resulting in partial
        peaks visible at the spectral edges. This is a common phenomenon in
        real NIR spectra where absorption bands extend beyond the instrument's
        wavelength range.

        Common causes include:
        - Strong water absorption bands at ~2500 nm affecting NIR edge
        - UV/visible absorption tails at the low wavelength end
        - Mid-IR fundamental bands tailing into NIR at the high end

        Args:
            name: Component name.
            measurement_range: (min, max) wavelength range of the "measurement" (nm).
                Bands will be placed outside this range.
            edge: Which edge(s) to add boundary bands:
                - "left": Only below min wavelength
                - "right": Only above max wavelength
                - "both": Either edge (randomly selected)
            n_bands: Number of boundary bands to generate.
            amplitude_range: Range for peak amplitudes (0-1 scale).
            width_range: Range for band widths (nm). Controls how much of
                the peak is visible in the measurement range.
            offset_range: Range for how far outside the measurement range
                to place the band center, as a fraction of width.
                e.g., 0.5 means center is 0.5*width outside the range.

        Returns:
            The generated SpectralComponent with boundary bands.

        Example:
            >>> library = ComponentLibrary(random_state=42)
            >>> # Add water band tail at long wavelength edge
            >>> boundary = library.add_boundary_component(
            ...     "water_tail",
            ...     measurement_range=(1000, 2400),
            ...     edge="right",
            ...     amplitude_range=(0.5, 1.0),
            ...     width_range=(100, 300)
            ... )

        References:
            - Burns & Ciurczak (2007). Handbook of Near-Infrared Analysis.
              Discussion of wavelength range selection and edge effects.
        """
        wl_min, wl_max = measurement_range
        bands = []

        for i in range(n_bands):
            # Determine which edge
            if edge == "both":
                current_edge = self.rng.choice(["left", "right"])
            else:
                current_edge = edge

            # Generate band parameters
            amplitude = self.rng.uniform(*amplitude_range)
            width = self.rng.uniform(*width_range)
            offset_fraction = self.rng.uniform(*offset_range)

            # Position center outside measurement range
            if current_edge == "left":
                center = wl_min - offset_fraction * width
                band_name = f"boundary_left_{i}"
            else:
                center = wl_max + offset_fraction * width
                band_name = f"boundary_right_{i}"

            # Slight Voigt profile for realism
            gamma = self.rng.uniform(0, width * 0.1)

            bands.append(
                NIRBand(
                    center=center,
                    sigma=width,
                    gamma=gamma,
                    amplitude=amplitude,
                    name=band_name,
                )
            )

        component = SpectralComponent(
            name=name,
            bands=bands,
            category="boundary_effect",
            tags=["synthetic", "boundary", "edge_effect"],
        )
        self._components[name] = component
        return component

    def add_boundary_components_from_known(
        self,
        measurement_range: Tuple[float, float] = (1000, 2500),
    ) -> ComponentLibrary:
        """
        Add known boundary components that affect common NIR measurement ranges.

        Based on literature, certain absorption bands commonly appear as
        truncated peaks at measurement boundaries:

        - Left edge (short wavelengths): Electronic transitions, UV tails
        - Right edge (long wavelengths): Strong water O-H bands, C-H fundamentals

        Args:
            measurement_range: (min, max) wavelength range of measurement (nm).

        Returns:
            Self for method chaining.

        Example:
            >>> library = ComponentLibrary(random_state=42)
            >>> library.add_boundary_components_from_known((1000, 2400))
        """
        wl_min, wl_max = measurement_range

        # Right edge: Water combination band (~2500 nm region)
        if wl_max < 2600 and wl_max > 2000:
            self._components["water_boundary_2500"] = SpectralComponent(
                name="water_boundary_2500",
                bands=[
                    NIRBand(
                        center=2520,
                        sigma=80,
                        gamma=5,
                        amplitude=0.8,
                        name="O-H combination band tail"
                    ),
                ],
                category="boundary_effect",
                tags=["water", "boundary", "combination"],
                references=["Workman & Weyer (2012)"],
            )

        # Right edge: Mid-IR C-H fundamental tails
        if wl_max > 2300 and wl_max < 2600:
            self._components["ch_fundamental_tail"] = SpectralComponent(
                name="ch_fundamental_tail",
                bands=[
                    NIRBand(
                        center=2700,
                        sigma=150,
                        gamma=10,
                        amplitude=0.5,
                        name="C-H fundamental tail"
                    ),
                ],
                category="boundary_effect",
                tags=["hydrocarbon", "boundary", "fundamental"],
                references=["Burns & Ciurczak (2007)"],
            )

        # Left edge: UV/visible absorption tails (if NIR starts below 1000 nm)
        if wl_min < 1000 and wl_min > 700:
            self._components["uv_absorption_tail"] = SpectralComponent(
                name="uv_absorption_tail",
                bands=[
                    NIRBand(
                        center=650,
                        sigma=100,
                        gamma=0,
                        amplitude=0.3,
                        name="UV absorption tail"
                    ),
                ],
                category="boundary_effect",
                tags=["electronic", "boundary", "uv"],
            )

        # Left edge: Silicon detector cutoff effects (~1100 nm)
        if wl_min > 900 and wl_min < 1200:
            self._components["detector_edge_1000"] = SpectralComponent(
                name="detector_edge_1000",
                bands=[
                    NIRBand(
                        center=wl_min - 50,
                        sigma=60,
                        gamma=0,
                        amplitude=0.2,
                        name="Detector edge artifact"
                    ),
                ],
                category="boundary_effect",
                tags=["instrumental", "boundary", "detector"],
            )

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


# ============================================================================
# Discovery API Functions (Phase 1 enhancement)
# ============================================================================


def available_components() -> List[str]:
    """
    Return list of all available predefined component names.

    Returns:
        Sorted list of component names.

    Example:
        >>> names = available_components()
        >>> print(f"Available: {len(names)} components")
        >>> print(names[:5])
    """
    from ._constants import get_predefined_components

    return sorted(get_predefined_components().keys())


def get_component(name: str) -> SpectralComponent:
    """
    Get a single predefined component by name.

    Args:
        name: Component name (e.g., "water", "protein", "lipid").

    Returns:
        SpectralComponent object.

    Raises:
        ValueError: If component name is not found.

    Example:
        >>> water = get_component("water")
        >>> print(water.category)
        >>> print(len(water.bands))
    """
    from ._constants import get_predefined_components

    components = get_predefined_components()
    if name not in components:
        available = available_components()
        raise ValueError(f"Unknown component: '{name}'. Use available_components() to list options. Available: {available[:10]}...")
    return components[name]


def search_components(
    query: Optional[str] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    tags: Optional[List[str]] = None,
    wavelength_range: Optional[Tuple[float, float]] = None,
) -> List[str]:
    """
    Search components by various criteria.

    Args:
        query: Fuzzy match on name or synonyms.
        category: Filter by category (e.g., "proteins", "carbohydrates").
        subcategory: Filter by subcategory (e.g., "monosaccharides").
        tags: Filter by tags (any match).
        wavelength_range: Filter by components with bands in range (min, max).

    Returns:
        List of matching component names.

    Example:
        >>> # Find all protein-related components
        >>> proteins = search_components(category="proteins")
        >>>
        >>> # Find components with bands in visible-NIR region
        >>> vis_nir = search_components(wavelength_range=(400, 1000))
        >>>
        >>> # Find components tagged for pharmaceutical use
        >>> pharma = search_components(tags=["pharma"])
    """
    from ._constants import get_predefined_components

    components = get_predefined_components()
    results = []

    for name, comp in components.items():
        # Filter by query (fuzzy match on name/synonyms)
        if query:
            query_lower = query.lower()
            if query_lower not in name.lower():
                # Check synonyms
                if not any(query_lower in s.lower() for s in (comp.synonyms or [])):
                    continue

        # Filter by category
        if category and comp.category != category:
            continue

        # Filter by subcategory
        if subcategory and comp.subcategory != subcategory:
            continue

        # Filter by tags (any match)
        if tags:
            comp_tags = comp.tags or []
            if not any(t in comp_tags for t in tags):
                continue

        # Filter by wavelength range
        if wavelength_range:
            if not comp.has_bands_in_range(wavelength_range):
                continue

        results.append(name)

    return sorted(results)


def list_categories() -> Dict[str, List[str]]:
    """
    Return dictionary of categories to component names.

    Returns:
        Dictionary mapping category names to lists of component names.

    Example:
        >>> categories = list_categories()
        >>> for cat, components in categories.items():
        ...     print(f"{cat}: {len(components)} components")
    """
    from ._constants import get_predefined_components

    components = get_predefined_components()
    categories: Dict[str, List[str]] = {}

    for name, comp in components.items():
        cat = comp.category or "uncategorized"
        categories.setdefault(cat, []).append(name)

    # Sort components within each category
    for cat in categories:
        categories[cat].sort()

    return categories


def component_info(name: str) -> str:
    """
    Return formatted information about a component.

    Args:
        name: Component name.

    Returns:
        Human-readable string with component details.

    Example:
        >>> print(component_info("water"))
    """
    comp = get_component(name)
    return comp.info()


def validate_predefined_components() -> List[str]:
    """
    Validate all predefined components.

    Returns:
        List of validation warnings/errors (empty if all valid).

    Example:
        >>> issues = validate_predefined_components()
        >>> if issues:
        ...     for issue in issues:
        ...         print(issue)
        ... else:
        ...     print("All components valid!")
    """
    from ._constants import get_predefined_components

    issues: List[str] = []
    components = get_predefined_components()

    # Check for uniqueness
    names = list(components.keys())
    if len(names) != len(set(names)):
        duplicates = [n for n in names if names.count(n) > 1]
        issues.append(f"Duplicate component names: {duplicates}")

    # Validate each component
    for name, comp in components.items():
        # Check name matches key
        if comp.name != name:
            issues.append(f"Component '{name}': name mismatch (comp.name='{comp.name}')")

        # Validate component
        comp_issues = comp.validate()
        issues.extend(comp_issues)

        # Check amplitude normalization (max should be ~1.0)
        if comp.bands:
            amplitudes = [b.amplitude for b in comp.bands]
            max_amp = max(amplitudes)
            if abs(max_amp - 1.0) > 0.1:
                issues.append(f"Component '{name}': max amplitude {max_amp:.2f} not normalized to 1.0")

    return issues


def validate_component_coverage(
    wavelength_range: Tuple[float, float] = (350, 2500),
) -> Dict[str, List[str]]:
    """
    Check which components have bands in the given wavelength range.

    Args:
        wavelength_range: (min, max) wavelength in nm.

    Returns:
        Dictionary with 'covered' and 'not_covered' component lists.

    Example:
        >>> coverage = validate_component_coverage((1000, 2500))
        >>> print(f"Covered: {len(coverage['covered'])}")
        >>> print(f"Not covered: {coverage['not_covered']}")
    """
    from ._constants import get_predefined_components

    components = get_predefined_components()
    covered: List[str] = []
    not_covered: List[str] = []

    for name, comp in components.items():
        if comp.has_bands_in_range(wavelength_range):
            covered.append(name)
        else:
            not_covered.append(name)

    return {"covered": sorted(covered), "not_covered": sorted(not_covered)}


def normalize_component_amplitudes(
    component: SpectralComponent,
    method: str = "max",
) -> SpectralComponent:
    """
    Normalize band amplitudes for a component.

    This is a convenience wrapper around SpectralComponent.normalized().

    Args:
        component: SpectralComponent to normalize.
        method: Normalization method ("max" or "sum").

    Returns:
        New SpectralComponent with normalized amplitudes.

    Example:
        >>> comp = get_component("water")
        >>> normalized = normalize_component_amplitudes(comp)
    """
    return component.normalized(method=method)
