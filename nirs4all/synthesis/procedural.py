"""
Procedural spectral component generation for synthetic NIRS data.

This module provides tools for generating chemically-plausible spectral components
with physically-motivated constraints including:
- Overtone relationships (bands at 2×, 3× wavenumber with anharmonicity correction)
- Combination bands (ν₁ + ν₂ relationships)
- Matrix-induced band shifts (hydrogen bonding, solvation)
- Domain-specific constraints

The procedural generator can create thousands of unique, realistic spectral
components without manual specification of band parameters.

Key Classes:
    - ProceduralComponentConfig: Configuration for procedural generation
    - ProceduralComponentGenerator: Main generator class

Example:
    >>> from nirs4all.synthesis.procedural import (
    ...     ProceduralComponentGenerator,
    ...     ProceduralComponentConfig
    ... )
    >>>
    >>> config = ProceduralComponentConfig(
    ...     n_fundamental_bands=2,
    ...     include_overtones=True,
    ...     include_combinations=True
    ... )
    >>> generator = ProceduralComponentGenerator(random_state=42)
    >>> library = generator.generate_library(n_components=10, config=config)

References:
    - Workman Jr, J., & Weyer, L. (2012). Practical Guide and Spectral Atlas
      for Interpretive Near-Infrared Spectroscopy. CRC Press.
    - Siesler, H. W., Ozaki, Y., Kawata, S., & Heise, H. M. (2002). Near-Infrared
      Spectroscopy: Principles, Instruments, Applications. Wiley-VCH.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Optional, Union

import numpy as np

from .components import ComponentLibrary, NIRBand, SpectralComponent
from .wavenumber import (
    FUNDAMENTAL_VIBRATIONS,
    NIR_ZONES_WAVENUMBER,
    apply_hydrogen_bonding_shift,
    calculate_combination_band,
    calculate_overtone_position,
    convert_bandwidth_to_wavelength,
    estimate_bandwidth_broadening,
    wavelength_to_wavenumber,
    wavenumber_to_wavelength,
)


class FunctionalGroupType(StrEnum):
    """Types of functional groups for component generation."""
    HYDROXYL = "hydroxyl"           # O-H (alcohols, phenols, carboxylic acids)
    AMINE = "amine"                 # N-H (primary, secondary amines, amides)
    METHYL = "methyl"               # CH3
    METHYLENE = "methylene"         # CH2
    AROMATIC_CH = "aromatic_ch"     # Aromatic C-H
    VINYL = "vinyl"                 # =C-H
    CARBONYL = "carbonyl"           # C=O
    CARBOXYL = "carboxyl"           # COOH (combination of C=O and O-H)
    THIOL = "thiol"                 # S-H
    WATER = "water"                 # H2O (special case)

# Functional group fundamental frequencies and properties
FUNCTIONAL_GROUP_PROPERTIES = {
    FunctionalGroupType.HYDROXYL: {
        "fundamental_cm": 3550,       # Average free O-H
        "fundamental_range": (3400, 3700),
        "bandwidth_cm": 80,
        "h_bond_susceptibility": 1.0,  # Very susceptible to H-bonding
        "typical_amplitude": 0.7,
    },
    FunctionalGroupType.AMINE: {
        "fundamental_cm": 3350,
        "fundamental_range": (3200, 3500),
        "bandwidth_cm": 70,
        "h_bond_susceptibility": 0.8,
        "typical_amplitude": 0.5,
    },
    FunctionalGroupType.METHYL: {
        "fundamental_cm": 2920,       # Average of sym/asym
        "fundamental_range": (2850, 2970),
        "bandwidth_cm": 50,
        "h_bond_susceptibility": 0.0,
        "typical_amplitude": 0.6,
    },
    FunctionalGroupType.METHYLENE: {
        "fundamental_cm": 2890,
        "fundamental_range": (2840, 2940),
        "bandwidth_cm": 45,
        "h_bond_susceptibility": 0.0,
        "typical_amplitude": 0.6,
    },
    FunctionalGroupType.AROMATIC_CH: {
        "fundamental_cm": 3060,
        "fundamental_range": (3020, 3100),
        "bandwidth_cm": 40,
        "h_bond_susceptibility": 0.1,
        "typical_amplitude": 0.5,
    },
    FunctionalGroupType.VINYL: {
        "fundamental_cm": 3100,
        "fundamental_range": (3050, 3150),
        "bandwidth_cm": 45,
        "h_bond_susceptibility": 0.1,
        "typical_amplitude": 0.4,
    },
    FunctionalGroupType.CARBONYL: {
        "fundamental_cm": 1720,
        "fundamental_range": (1650, 1780),
        "bandwidth_cm": 35,
        "h_bond_susceptibility": 0.5,  # As acceptor
        "typical_amplitude": 0.4,
    },
    FunctionalGroupType.CARBOXYL: {
        "fundamental_cm": 1710,
        "fundamental_range": (1680, 1750),
        "bandwidth_cm": 100,           # Broad due to H-bonding
        "h_bond_susceptibility": 1.0,
        "typical_amplitude": 0.6,
    },
    FunctionalGroupType.THIOL: {
        "fundamental_cm": 2550,
        "fundamental_range": (2500, 2600),
        "bandwidth_cm": 40,
        "h_bond_susceptibility": 0.3,
        "typical_amplitude": 0.3,
    },
    FunctionalGroupType.WATER: {
        "fundamental_cm": 3400,
        "fundamental_range": (3200, 3600),
        "bandwidth_cm": 150,           # Very broad due to H-bonding
        "h_bond_susceptibility": 1.0,
        "typical_amplitude": 1.0,
        "bending_mode_cm": 1640,       # Special: has bending mode for combinations
    },
}

@dataclass
class ProceduralComponentConfig:
    """
    Configuration for procedural component generation.

    Controls the complexity and characteristics of generated spectral components
    including the number of bands, overtone generation, combination bands, and
    environmental effects.

    Attributes:
        n_fundamental_bands: Number of fundamental vibration bands to generate.
        include_overtones: Whether to generate overtone bands (1st, 2nd, etc.).
        max_overtone_order: Maximum overtone order (2=1st overtone, 3=2nd, etc.).
        include_combinations: Whether to generate combination bands.
        max_combinations: Maximum number of combination bands.
        h_bond_strength: Average hydrogen bonding strength (0-1).
        h_bond_variability: Variability in H-bond strength between samples.
        anharmonicity: Anharmonicity constant for overtone calculations.
        anharmonicity_variability: Variability in anharmonicity.
        amplitude_variability: Random variation in band amplitudes.
        bandwidth_variability: Random variation in band widths.
        wavelength_range: NIR wavelength range for band placement (nm).
        functional_groups: Optional list of specific functional groups to use.
        combination_amplitude_factor: Amplitude reduction for combination bands.

    Example:
        >>> config = ProceduralComponentConfig(
        ...     n_fundamental_bands=3,
        ...     include_overtones=True,
        ...     include_combinations=True,
        ...     h_bond_strength=0.5
        ... )
    """
    n_fundamental_bands: int = 3
    include_overtones: bool = True
    max_overtone_order: int = 3  # Up to 2nd overtone
    include_combinations: bool = True
    max_combinations: int = 3
    h_bond_strength: float = 0.3
    h_bond_variability: float = 0.2
    anharmonicity: float = 0.02
    anharmonicity_variability: float = 0.005
    amplitude_variability: float = 0.3
    bandwidth_variability: float = 0.2
    wavelength_range: tuple[float, float] = (900, 2500)
    functional_groups: list[FunctionalGroupType] | None = None
    combination_amplitude_factor: float = 0.2

class ProceduralComponentGenerator:
    """
    Generator for procedurally-created spectral components.

    Creates chemically-plausible spectral components with physically-motivated
    constraints. Uses wavenumber-space calculations for proper overtone and
    combination band placement.

    Attributes:
        rng: NumPy random generator for reproducibility.

    Example:
        >>> generator = ProceduralComponentGenerator(random_state=42)
        >>>
        >>> # Generate a single component
        >>> component = generator.generate_component("my_compound")
        >>>
        >>> # Generate a library of components
        >>> library = generator.generate_library(n_components=10)
        >>>
        >>> # Generate with specific configuration
        >>> config = ProceduralComponentConfig(n_fundamental_bands=4)
        >>> component = generator.generate_component("complex_compound", config)
    """

    def __init__(self, random_state: int | None = None) -> None:
        """
        Initialize the procedural component generator.

        Args:
            random_state: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(random_state)
        self._random_state = random_state

    def _select_functional_groups(
        self,
        n_groups: int,
        allowed_groups: list[FunctionalGroupType] | None = None
    ) -> list[FunctionalGroupType]:
        """
        Randomly select functional groups for a component.

        Args:
            n_groups: Number of functional groups to select.
            allowed_groups: Optional list of allowed groups. If None, uses all.

        Returns:
            List of selected FunctionalGroupType values.
        """
        if allowed_groups is None:
            allowed_groups = list(FunctionalGroupType)

        # Weight selection towards common organic groups
        weights = {
            FunctionalGroupType.HYDROXYL: 3,
            FunctionalGroupType.AMINE: 2,
            FunctionalGroupType.METHYL: 4,
            FunctionalGroupType.METHYLENE: 4,
            FunctionalGroupType.AROMATIC_CH: 2,
            FunctionalGroupType.VINYL: 1,
            FunctionalGroupType.CARBONYL: 2,
            FunctionalGroupType.CARBOXYL: 2,
            FunctionalGroupType.THIOL: 0.5,
            FunctionalGroupType.WATER: 1,
        }

        raw_weights = [weights.get(g, 1) for g in allowed_groups]
        group_weights = np.array(raw_weights) / sum(raw_weights)

        # Select with replacement allowed (can have multiple of same type)
        indices = self.rng.choice(
            len(allowed_groups),
            size=n_groups,
            replace=True,
            p=group_weights
        )

        return [allowed_groups[i] for i in indices]

    def _generate_fundamental_band(
        self,
        functional_group: FunctionalGroupType,
        config: ProceduralComponentConfig,
        band_index: int
    ) -> tuple[NIRBand | None, float]:
        """
        Generate a fundamental band for a functional group.

        Args:
            functional_group: Type of functional group.
            config: Generation configuration.
            band_index: Index of this band (for naming).

        Returns:
            Tuple of (NIRBand or None, fundamental_wavenumber_cm).
            Returns the band and its fundamental wavenumber for overtone calculation.
        """
        props = FUNCTIONAL_GROUP_PROPERTIES[functional_group]

        # Sample fundamental wavenumber within range
        nu_min: float
        nu_max: float
        nu_min, nu_max = (float(props["fundamental_range"][0]), float(props["fundamental_range"][1]))  # type: ignore[index]
        fundamental_cm: float = float(self.rng.uniform(nu_min, nu_max))

        # Apply hydrogen bonding shift if applicable
        h_bond = float(props["h_bond_susceptibility"])
        if h_bond > 0:
            actual_h_bond = self.rng.normal(
                config.h_bond_strength * h_bond,
                config.h_bond_variability * h_bond
            )
            actual_h_bond = float(np.clip(actual_h_bond, 0, 1))
            fundamental_cm = apply_hydrogen_bonding_shift(
                fundamental_cm,
                actual_h_bond,
                is_donor=functional_group in [FunctionalGroupType.HYDROXYL,
                                               FunctionalGroupType.AMINE,
                                               FunctionalGroupType.WATER]
            )

        # Calculate 1st overtone position (fundamental is in mid-IR, outside NIR)
        # For NIR, we care about 1st and higher overtones
        anharmonicity = self.rng.normal(
            config.anharmonicity,
            config.anharmonicity_variability
        )
        anharmonicity = np.clip(anharmonicity, 0.005, 0.05)

        # Find the lowest overtone that falls in the NIR range
        wl_min, wl_max = config.wavelength_range
        band = None
        band_wavelength = None

        for order in range(2, config.max_overtone_order + 1):
            ot = calculate_overtone_position(fundamental_cm, order, float(anharmonicity))
            if wl_min <= ot.wavelength_nm <= wl_max:
                band_wavelength = ot.wavelength_nm
                amplitude_factor = ot.amplitude_factor
                bandwidth_factor = ot.bandwidth_factor
                break
        else:
            # No overtone in range, try to find any band in range
            return None, fundamental_cm

        # Calculate bandwidth in wavelength units
        base_bandwidth_cm = float(props["bandwidth_cm"])
        if h_bond > 0 and actual_h_bond > 0:
            base_bandwidth_cm = estimate_bandwidth_broadening(
                base_bandwidth_cm, float(actual_h_bond)
            )

        bandwidth_cm = base_bandwidth_cm * bandwidth_factor
        # Apply variability
        bandwidth_cm *= self.rng.uniform(
            1 - config.bandwidth_variability,
            1 + config.bandwidth_variability
        )

        # Convert to wavelength
        sigma_nm = convert_bandwidth_to_wavelength(float(bandwidth_cm), band_wavelength) / 2.355

        # Calculate amplitude
        amplitude = float(props["typical_amplitude"]) * amplitude_factor
        amplitude *= self.rng.uniform(
            1 - config.amplitude_variability,
            1 + config.amplitude_variability
        )
        amplitude = max(0.05, amplitude)

        # Gamma for Voigt profile (Lorentzian component)
        gamma = sigma_nm * self.rng.uniform(0.05, 0.2)

        band = NIRBand(
            center=band_wavelength,
            sigma=sigma_nm,
            gamma=gamma,
            amplitude=amplitude,
            name=f"{functional_group.value}_{band_index}"
        )

        return band, fundamental_cm

    def _generate_overtone_bands(
        self,
        fundamental_cm: float,
        functional_group: FunctionalGroupType,
        config: ProceduralComponentConfig,
        base_index: int
    ) -> list[NIRBand]:
        """
        Generate overtone bands for a fundamental.

        Args:
            fundamental_cm: Fundamental wavenumber in cm⁻¹.
            functional_group: Type of functional group.
            config: Generation configuration.
            base_index: Base index for band naming.

        Returns:
            List of NIRBand objects for overtones in the NIR range.
        """
        bands = []
        props = FUNCTIONAL_GROUP_PROPERTIES[functional_group]
        wl_min, wl_max = config.wavelength_range

        anharmonicity = self.rng.normal(
            config.anharmonicity,
            config.anharmonicity_variability
        )
        anharmonicity = np.clip(anharmonicity, 0.005, 0.05)

        for order in range(2, config.max_overtone_order + 1):
            ot = calculate_overtone_position(fundamental_cm, order, float(anharmonicity))

            if wl_min <= ot.wavelength_nm <= wl_max:
                # Calculate bandwidth
                base_bandwidth_cm = float(props["bandwidth_cm"]) * ot.bandwidth_factor
                base_bandwidth_cm *= self.rng.uniform(
                    1 - config.bandwidth_variability,
                    1 + config.bandwidth_variability
                )

                sigma_nm = convert_bandwidth_to_wavelength(
                    base_bandwidth_cm, ot.wavelength_nm
                ) / 2.355

                # Calculate amplitude
                amplitude = float(props["typical_amplitude"]) * ot.amplitude_factor
                amplitude *= self.rng.uniform(
                    1 - config.amplitude_variability,
                    1 + config.amplitude_variability
                )
                amplitude = max(0.02, amplitude)

                gamma = sigma_nm * self.rng.uniform(0.05, 0.2)

                order_name = {2: "1st", 3: "2nd", 4: "3rd"}.get(order, f"{order-1}th")

                band = NIRBand(
                    center=ot.wavelength_nm,
                    sigma=sigma_nm,
                    gamma=gamma,
                    amplitude=amplitude,
                    name=f"{functional_group.value}_{order_name}_overtone_{base_index}"
                )
                bands.append(band)

        return bands

    def _generate_combination_bands(
        self,
        fundamentals: list[tuple[float, FunctionalGroupType]],
        config: ProceduralComponentConfig
    ) -> list[NIRBand]:
        """
        Generate combination bands from pairs of fundamentals.

        Args:
            fundamentals: List of (wavenumber, functional_group) tuples.
            config: Generation configuration.

        Returns:
            List of NIRBand objects for combination bands in NIR range.
        """
        bands = []
        wl_min, wl_max = config.wavelength_range
        n_combinations = 0

        # Generate sum combinations
        for i, (nu1, fg1) in enumerate(fundamentals):
            for j, (nu2, fg2) in enumerate(fundamentals):
                if i >= j:
                    continue
                if n_combinations >= config.max_combinations:
                    break

                comb = calculate_combination_band(nu1, nu2, "sum")

                if wl_min <= comb.wavelength_nm <= wl_max:
                    props1 = FUNCTIONAL_GROUP_PROPERTIES[fg1]
                    props2 = FUNCTIONAL_GROUP_PROPERTIES[fg2]

                    # Combination band width is roughly the sum of individual widths
                    base_bandwidth_cm = (float(props1["bandwidth_cm"]) + float(props2["bandwidth_cm"])) / 2
                    base_bandwidth_cm *= self.rng.uniform(0.8, 1.2)

                    sigma_nm = convert_bandwidth_to_wavelength(
                        base_bandwidth_cm, comb.wavelength_nm
                    ) / 2.355

                    # Amplitude is reduced for combinations
                    base_amp = (props1["typical_amplitude"] + props2["typical_amplitude"]) / 2
                    amplitude = base_amp * config.combination_amplitude_factor
                    amplitude *= self.rng.uniform(0.5, 1.5)
                    amplitude = max(0.02, amplitude)

                    gamma = sigma_nm * self.rng.uniform(0.05, 0.15)

                    band = NIRBand(
                        center=comb.wavelength_nm,
                        sigma=sigma_nm,
                        gamma=gamma,
                        amplitude=amplitude,
                        name=f"{fg1.value}+{fg2.value}_combination"
                    )
                    bands.append(band)
                    n_combinations += 1

        # Special case: water O-H stretch + O-H bend combination
        for nu, fg in fundamentals:
            if fg == FunctionalGroupType.WATER and n_combinations < config.max_combinations:
                props = FUNCTIONAL_GROUP_PROPERTIES[fg]
                bend_mode = props.get("bending_mode_cm", 1640)

                comb = calculate_combination_band(nu, bend_mode, "sum")

                if wl_min <= comb.wavelength_nm <= wl_max:
                    sigma_nm = convert_bandwidth_to_wavelength(
                        props["bandwidth_cm"], comb.wavelength_nm
                    ) / 2.355

                    amplitude = props["typical_amplitude"] * 0.8  # Strong combination
                    amplitude *= self.rng.uniform(0.8, 1.2)

                    gamma = sigma_nm * self.rng.uniform(0.1, 0.2)

                    band = NIRBand(
                        center=comb.wavelength_nm,
                        sigma=sigma_nm,
                        gamma=gamma,
                        amplitude=amplitude,
                        name="water_stretch+bend_combination"
                    )
                    bands.append(band)
                    n_combinations += 1

        return bands

    def generate_component(
        self,
        name: str,
        config: ProceduralComponentConfig | None = None,
        functional_groups: list[FunctionalGroupType] | None = None,
        correlation_group: int | None = None
    ) -> SpectralComponent:
        """
        Generate a single spectral component.

        Creates a chemically-plausible component with bands following
        physical constraints (overtone relationships, combination bands, etc.).

        Args:
            name: Name for the component.
            config: Generation configuration. If None, uses defaults.
            functional_groups: Specific functional groups to use.
                If None, randomly selects based on config.
            correlation_group: Optional correlation group ID.

        Returns:
            SpectralComponent with generated bands.

        Example:
            >>> generator = ProceduralComponentGenerator(random_state=42)
            >>> component = generator.generate_component("my_compound")
            >>> print(f"Generated {len(component.bands)} bands")
        """
        if config is None:
            config = ProceduralComponentConfig()

        # Select functional groups
        if functional_groups is not None:
            groups = functional_groups
        elif config.functional_groups is not None:
            groups = config.functional_groups
        else:
            groups = self._select_functional_groups(
                config.n_fundamental_bands,
                None
            )

        bands: list[NIRBand] = []
        fundamentals: list[tuple[float, FunctionalGroupType]] = []

        # Generate fundamental-derived bands
        for i, group in enumerate(groups):
            band, fundamental_cm = self._generate_fundamental_band(group, config, i)
            if band is not None:
                bands.append(band)
            fundamentals.append((fundamental_cm, group))

            # Generate additional overtones if requested
            if config.include_overtones:
                overtone_bands = self._generate_overtone_bands(
                    fundamental_cm, group, config, i
                )
                # Avoid duplicates (may overlap with primary band)
                for ob in overtone_bands:
                    if not any(abs(ob.center - b.center) < 10 for b in bands):
                        bands.append(ob)

        # Generate combination bands
        if config.include_combinations and len(fundamentals) >= 2:
            comb_bands = self._generate_combination_bands(fundamentals, config)
            for cb in comb_bands:
                # Avoid duplicates
                if not any(abs(cb.center - b.center) < 10 for b in bands):
                    bands.append(cb)

        # Sort bands by wavelength
        bands.sort(key=lambda b: b.center)

        return SpectralComponent(
            name=name,
            bands=bands,
            correlation_group=correlation_group
        )

    def generate_library(
        self,
        n_components: int,
        config: ProceduralComponentConfig | None = None,
        name_prefix: str = "component"
    ) -> ComponentLibrary:
        """
        Generate a library of procedural components.

        Creates multiple unique components with varied characteristics.

        Args:
            n_components: Number of components to generate.
            config: Generation configuration applied to all components.
            name_prefix: Prefix for component names.

        Returns:
            ComponentLibrary populated with generated components.

        Example:
            >>> generator = ProceduralComponentGenerator(random_state=42)
            >>> library = generator.generate_library(10)
            >>> print(f"Created library with {library.n_components} components")
        """
        if config is None:
            config = ProceduralComponentConfig()

        library = ComponentLibrary(random_state=self._random_state)

        for i in range(n_components):
            # Vary the number of functional groups per component
            component_config = ProceduralComponentConfig(
                n_fundamental_bands=self.rng.integers(1, config.n_fundamental_bands + 2),
                include_overtones=config.include_overtones,
                max_overtone_order=config.max_overtone_order,
                include_combinations=config.include_combinations,
                max_combinations=config.max_combinations,
                h_bond_strength=self.rng.uniform(0, 0.8),
                h_bond_variability=config.h_bond_variability,
                anharmonicity=config.anharmonicity,
                anharmonicity_variability=config.anharmonicity_variability,
                amplitude_variability=config.amplitude_variability,
                bandwidth_variability=config.bandwidth_variability,
                wavelength_range=config.wavelength_range,
                functional_groups=config.functional_groups,
                combination_amplitude_factor=config.combination_amplitude_factor,
            )

            component = self.generate_component(
                name=f"{name_prefix}_{i}",
                config=component_config,
                correlation_group=i % 5  # Assign to one of 5 correlation groups
            )
            library.add_component(component)

        return library

    def generate_variant(
        self,
        base_component: SpectralComponent,
        variation_scale: float = 0.1,
        name: str | None = None
    ) -> SpectralComponent:
        """
        Generate a variant of an existing component.

        Creates a new component with similar characteristics but varied
        band positions, widths, and amplitudes. Useful for simulating
        batch effects or matrix variations.

        Args:
            base_component: Component to base the variant on.
            variation_scale: Scale of random variations (0-1).
            name: Name for the variant. If None, appends "_variant".

        Returns:
            SpectralComponent variant.

        Example:
            >>> generator = ProceduralComponentGenerator(random_state=42)
            >>> base = generator.generate_component("base")
            >>> variant = generator.generate_variant(base, variation_scale=0.15)
        """
        if name is None:
            name = f"{base_component.name}_variant"

        new_bands = []
        for band in base_component.bands:
            # Vary center position (shift)
            center_shift = self.rng.normal(0, variation_scale * 20)  # ~20 nm scale
            new_center = band.center + center_shift

            # Vary sigma (width)
            sigma_factor = self.rng.normal(1.0, variation_scale)
            new_sigma = band.sigma * np.clip(sigma_factor, 0.5, 1.5)

            # Vary gamma
            gamma_factor = self.rng.normal(1.0, variation_scale)
            new_gamma = band.gamma * np.clip(gamma_factor, 0.5, 1.5)

            # Vary amplitude
            amp_factor = self.rng.normal(1.0, variation_scale)
            new_amplitude = band.amplitude * np.clip(amp_factor, 0.5, 1.5)

            new_bands.append(NIRBand(
                center=new_center,
                sigma=new_sigma,
                gamma=new_gamma,
                amplitude=new_amplitude,
                name=band.name
            ))

        return SpectralComponent(
            name=name,
            bands=new_bands,
            correlation_group=base_component.correlation_group
        )

    def generate_from_functional_groups(
        self,
        name: str,
        functional_groups: list[FunctionalGroupType | str],
        config: ProceduralComponentConfig | None = None
    ) -> SpectralComponent:
        """
        Generate a component with specified functional groups.

        Convenience method for creating components with known chemistry.

        Args:
            name: Component name.
            functional_groups: List of functional groups (enum or string).
            config: Optional generation configuration.

        Returns:
            SpectralComponent with bands from specified functional groups.

        Example:
            >>> generator = ProceduralComponentGenerator(random_state=42)
            >>> # Generate an alcohol
            >>> alcohol = generator.generate_from_functional_groups(
            ...     "alcohol",
            ...     ["hydroxyl", "methyl", "methylene"]
            ... )
        """
        # Convert strings to enum
        groups = []
        for fg in functional_groups:
            if isinstance(fg, str):
                fg = FunctionalGroupType(fg)
            groups.append(fg)

        if config is None:
            config = ProceduralComponentConfig(
                n_fundamental_bands=len(groups)
            )

        return self.generate_component(
            name=name,
            config=config,
            functional_groups=groups
        )

# ============================================================================
# Module-level exports
# ============================================================================

__all__ = [
    "FunctionalGroupType",
    "FUNCTIONAL_GROUP_PROPERTIES",
    "ProceduralComponentConfig",
    "ProceduralComponentGenerator",
]
