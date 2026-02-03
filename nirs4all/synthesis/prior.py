"""
Conditional prior sampling for synthetic NIRS data generation.

This module provides structured prior sampling where configuration
parameters are sampled conditionally based on domain, instrument type,
and other hierarchical dependencies.

Phase 4 Features:
    - Domain-weighted sampling
    - Conditional instrument selection given domain
    - Conditional measurement mode given instrument
    - Matrix type conditioning on domain
    - Component set selection based on domain
    - Full configuration sampling from prior

Generative DAG:
    Domain → Instrument Category → Wavelength Range, Resolution, Mode, Noise
           → Matrix Type → Particle Size, Scattering, Water Activity
           → Component Set → Concentration Distributions
           → Target Type

References:
    - Workman Jr, J., & Weyer, L. (2012). Practical Guide and Spectral Atlas
      for Interpretive Near-Infrared Spectroscopy. CRC Press.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .domains import (
    DomainCategory,
    APPLICATION_DOMAINS,
    get_domain_config,
    list_domains,
)
from .instruments import (
    InstrumentCategory,
    INSTRUMENT_ARCHETYPES,
    get_instrument_archetype,
    list_instrument_archetypes,
)
from .measurement_modes import MeasurementMode


# ============================================================================
# Matrix Types
# ============================================================================


class MatrixType(str, Enum):
    """Physical matrix types that affect spectral properties."""
    LIQUID = "liquid"
    POWDER = "powder"
    SOLID = "solid"
    PASTE = "paste"
    EMULSION = "emulsion"
    GEL = "gel"
    TISSUE = "tissue"
    SLURRY = "slurry"
    FILM = "film"
    GRANULAR = "granular"


# ============================================================================
# Prior Configuration
# ============================================================================


@dataclass
class NIRSPriorConfig:
    """
    Configuration for NIRS data generation with conditional sampling.

    This class defines the prior distributions and conditional dependencies
    for sampling complete generation configurations.

    Attributes:
        domain_weights: Prior weights for each domain.
        instrument_given_domain: P(instrument_category | domain).
        mode_given_category: P(measurement_mode | instrument_category).
        matrix_given_domain: P(matrix_type | domain).
        temperature_range: (min, max) temperature in Celsius.
        particle_size_range: (min, max) particle size in microns.
        noise_level_range: (min, max) noise level multiplier.

    Example:
        >>> config = NIRSPriorConfig()
        >>> sampler = PriorSampler(config, random_state=42)
        >>> sample = sampler.sample()
        >>> print(sample["domain"], sample["instrument"])
    """

    # Domain prior weights
    domain_weights: Dict[str, float] = field(default_factory=lambda: {
        "grain": 0.15,
        "forage": 0.08,
        "oilseeds": 0.07,
        "fruit": 0.05,
        "dairy": 0.10,
        "meat": 0.05,
        "beverages": 0.05,
        "baking": 0.03,
        "tablets": 0.10,
        "powders": 0.05,
        "liquids": 0.05,
        "fuel": 0.05,
        "polymers": 0.04,
        "lubricants": 0.02,
        "water_quality": 0.03,
        "soil": 0.03,
        "tissue": 0.02,
        "blood": 0.02,
        "textiles": 0.01,
    })

    # P(instrument_category | domain)
    instrument_given_domain: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        # Agriculture domains prefer robust instruments
        "grain": {
            "benchtop": 0.5, "handheld": 0.2, "process": 0.2,
            "embedded": 0.05, "filter": 0.05
        },
        "forage": {
            "benchtop": 0.3, "handheld": 0.4, "process": 0.2,
            "embedded": 0.05, "filter": 0.05
        },
        "oilseeds": {
            "benchtop": 0.5, "handheld": 0.1, "process": 0.3,
            "embedded": 0.05, "filter": 0.05
        },
        "fruit": {
            "benchtop": 0.3, "handheld": 0.5, "process": 0.1,
            "embedded": 0.05, "filter": 0.05
        },
        # Food domains
        "dairy": {
            "benchtop": 0.4, "process": 0.4, "handheld": 0.1,
            "ft_nir": 0.1
        },
        "meat": {
            "benchtop": 0.4, "handheld": 0.3, "process": 0.2,
            "filter": 0.1
        },
        "beverages": {
            "benchtop": 0.3, "process": 0.4, "handheld": 0.2,
            "ft_nir": 0.1
        },
        "baking": {
            "benchtop": 0.5, "process": 0.3, "handheld": 0.1,
            "filter": 0.1
        },
        # Pharmaceutical domains prefer high-precision
        "tablets": {
            "benchtop": 0.5, "ft_nir": 0.3, "process": 0.1,
            "handheld": 0.1
        },
        "powders": {
            "benchtop": 0.4, "ft_nir": 0.4, "process": 0.1,
            "handheld": 0.1
        },
        "liquids": {
            "benchtop": 0.4, "ft_nir": 0.3, "process": 0.2,
            "diode_array": 0.1
        },
        # Petrochemical
        "fuel": {
            "benchtop": 0.3, "process": 0.5, "ft_nir": 0.1,
            "handheld": 0.1
        },
        "polymers": {
            "benchtop": 0.4, "ft_nir": 0.3, "process": 0.2,
            "handheld": 0.1
        },
        "lubricants": {
            "benchtop": 0.3, "process": 0.5, "ft_nir": 0.1,
            "handheld": 0.1
        },
        # Environmental
        "water_quality": {
            "benchtop": 0.3, "handheld": 0.4, "process": 0.2,
            "embedded": 0.1
        },
        "soil": {
            "benchtop": 0.3, "handheld": 0.5, "process": 0.1,
            "embedded": 0.1
        },
        # Biomedical
        "tissue": {
            "benchtop": 0.4, "ft_nir": 0.3, "handheld": 0.2,
            "embedded": 0.1
        },
        "blood": {
            "benchtop": 0.5, "ft_nir": 0.3, "process": 0.1,
            "embedded": 0.1
        },
        # Industrial
        "textiles": {
            "benchtop": 0.4, "process": 0.3, "handheld": 0.2,
            "filter": 0.1
        },
    })

    # P(measurement_mode | instrument_category)
    mode_given_category: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "benchtop": {
            "reflectance": 0.5, "transmittance": 0.3,
            "transflectance": 0.15, "atr": 0.05
        },
        "handheld": {
            "reflectance": 0.7, "transmittance": 0.1,
            "transflectance": 0.15, "atr": 0.05
        },
        "process": {
            "reflectance": 0.4, "transmittance": 0.3,
            "transflectance": 0.25, "atr": 0.05
        },
        "embedded": {
            "reflectance": 0.6, "transmittance": 0.2,
            "transflectance": 0.15, "atr": 0.05
        },
        "ft_nir": {
            "reflectance": 0.4, "transmittance": 0.35,
            "transflectance": 0.1, "atr": 0.15
        },
        "filter": {
            "reflectance": 0.6, "transmittance": 0.3,
            "transflectance": 0.1, "atr": 0.0
        },
        "diode_array": {
            "reflectance": 0.5, "transmittance": 0.4,
            "transflectance": 0.1, "atr": 0.0
        },
    })

    # P(matrix_type | domain)
    matrix_given_domain: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "grain": {"granular": 0.7, "powder": 0.2, "solid": 0.1},
        "forage": {"solid": 0.5, "powder": 0.3, "granular": 0.2},
        "oilseeds": {"granular": 0.6, "solid": 0.3, "powder": 0.1},
        "fruit": {"solid": 0.6, "paste": 0.2, "liquid": 0.2},
        "dairy": {"liquid": 0.5, "emulsion": 0.3, "powder": 0.2},
        "meat": {"solid": 0.6, "paste": 0.3, "emulsion": 0.1},
        "beverages": {"liquid": 0.9, "emulsion": 0.1},
        "baking": {"powder": 0.5, "paste": 0.3, "solid": 0.2},
        "tablets": {"solid": 0.7, "powder": 0.3},
        "powders": {"powder": 0.9, "granular": 0.1},
        "liquids": {"liquid": 0.9, "gel": 0.1},
        "fuel": {"liquid": 0.9, "gel": 0.1},
        "polymers": {"solid": 0.6, "film": 0.3, "powder": 0.1},
        "lubricants": {"liquid": 0.7, "gel": 0.3},
        "water_quality": {"liquid": 1.0},
        "soil": {"powder": 0.5, "granular": 0.4, "slurry": 0.1},
        "tissue": {"tissue": 0.8, "solid": 0.2},
        "blood": {"liquid": 0.9, "gel": 0.1},
        "textiles": {"solid": 0.8, "film": 0.2},
    })

    # Continuous parameter ranges
    temperature_range: Tuple[float, float] = (15.0, 40.0)
    particle_size_range: Tuple[float, float] = (5.0, 200.0)
    noise_level_range: Tuple[float, float] = (0.5, 2.0)
    n_samples_range: Tuple[int, int] = (100, 2000)

    # Target configuration
    target_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "regression": 0.7,
        "classification": 0.3,
    })

    n_targets_range: Tuple[int, int] = (1, 5)
    n_classes_range: Tuple[int, int] = (2, 5)

    def get_domain_weight(self, domain: str) -> float:
        """Get prior weight for a domain."""
        return self.domain_weights.get(domain, 0.0)

    def normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1."""
        total = sum(weights.values())
        if total == 0:
            return weights
        return {k: v / total for k, v in weights.items()}


# ============================================================================
# Prior Sampler
# ============================================================================


class PriorSampler:
    """
    Sample complete generation configurations from prior distributions.

    This class implements hierarchical sampling where lower-level
    configurations are conditioned on higher-level choices.

    Args:
        config: Prior configuration.
        random_state: Random state for reproducibility.

    Example:
        >>> config = NIRSPriorConfig()
        >>> sampler = PriorSampler(config, random_state=42)
        >>>
        >>> # Sample a single configuration
        >>> sample = sampler.sample()
        >>> print(sample)
        >>>
        >>> # Sample multiple configurations
        >>> samples = sampler.sample_batch(10)
    """

    def __init__(
        self,
        config: Optional[NIRSPriorConfig] = None,
        random_state: Optional[int] = None,
    ):
        self.config = config or NIRSPriorConfig()
        self.rng = np.random.default_rng(random_state)

    def _sample_categorical(
        self,
        weights: Dict[str, float],
    ) -> str:
        """Sample from a categorical distribution defined by weights."""
        # Normalize weights
        total = sum(weights.values())
        if total == 0:
            # Uniform if all zero
            categories = list(weights.keys())
            return self.rng.choice(categories)

        categories = list(weights.keys())
        probs = np.array([weights[c] / total for c in categories])
        idx = self.rng.choice(len(categories), p=probs)
        return categories[idx]

    def sample_domain(self) -> str:
        """Sample a domain from the prior."""
        return self._sample_categorical(self.config.domain_weights)

    def sample_instrument_category(self, domain: str) -> str:
        """Sample an instrument category given the domain."""
        if domain in self.config.instrument_given_domain:
            weights = self.config.instrument_given_domain[domain]
        else:
            # Default uniform over all categories
            weights = {cat: 1.0 for cat in [
                "benchtop", "handheld", "process", "embedded", "ft_nir"
            ]}
        return self._sample_categorical(weights)

    def sample_instrument(self, category: str) -> str:
        """Sample a specific instrument given the category."""
        # Get all instruments of this category
        matching = []
        for name, archetype in INSTRUMENT_ARCHETYPES.items():
            if archetype.category.value == category:
                matching.append(name)

        if not matching:
            # Fall back to any instrument
            matching = list(INSTRUMENT_ARCHETYPES.keys())

        return self.rng.choice(matching)

    def sample_measurement_mode(self, instrument_category: str) -> str:
        """Sample a measurement mode given the instrument category."""
        if instrument_category in self.config.mode_given_category:
            weights = self.config.mode_given_category[instrument_category]
        else:
            weights = {
                "reflectance": 0.5, "transmittance": 0.3,
                "transflectance": 0.15, "atr": 0.05
            }
        return self._sample_categorical(weights)

    def sample_matrix_type(self, domain: str) -> str:
        """Sample a matrix type given the domain."""
        if domain in self.config.matrix_given_domain:
            weights = self.config.matrix_given_domain[domain]
        else:
            weights = {"powder": 0.3, "liquid": 0.3, "solid": 0.4}
        return self._sample_categorical(weights)

    def sample_temperature(self) -> float:
        """Sample a temperature from the prior range."""
        low, high = self.config.temperature_range
        return float(self.rng.uniform(low, high))

    def sample_particle_size(self, matrix_type: str) -> float:
        """Sample particle size based on matrix type."""
        low, high = self.config.particle_size_range

        # Adjust range based on matrix type
        if matrix_type == "powder":
            low, high = 5.0, 100.0
        elif matrix_type == "granular":
            low, high = 50.0, 500.0
        elif matrix_type in ("liquid", "emulsion"):
            low, high = 0.1, 10.0
        elif matrix_type == "solid":
            # Not really applicable, but return a value
            low, high = 100.0, 1000.0

        return float(self.rng.uniform(low, high))

    def sample_noise_level(self, instrument_category: str) -> float:
        """Sample noise level multiplier based on instrument category."""
        low, high = self.config.noise_level_range

        # Handheld instruments typically have higher noise
        if instrument_category == "handheld":
            low, high = 1.0, 3.0
        elif instrument_category == "embedded":
            low, high = 1.5, 3.5
        elif instrument_category == "ft_nir":
            low, high = 0.3, 1.0  # FT-NIR typically lower noise
        elif instrument_category == "benchtop":
            low, high = 0.5, 1.5

        return float(self.rng.uniform(low, high))

    def sample_n_samples(self) -> int:
        """Sample number of samples to generate."""
        low, high = self.config.n_samples_range
        return int(self.rng.integers(low, high + 1))

    def sample_target_config(self) -> Dict[str, Any]:
        """Sample target generation configuration."""
        target_type = self._sample_categorical(self.config.target_type_weights)

        if target_type == "regression":
            n_targets = int(self.rng.integers(
                self.config.n_targets_range[0],
                self.config.n_targets_range[1] + 1
            ))
            return {
                "type": "regression",
                "n_targets": n_targets,
                "nonlinearity": self.rng.choice(["none", "mild", "moderate"]),
            }
        else:
            n_classes = int(self.rng.integers(
                self.config.n_classes_range[0],
                self.config.n_classes_range[1] + 1
            ))
            return {
                "type": "classification",
                "n_classes": n_classes,
                "separation": self.rng.choice(["easy", "moderate", "hard"]),
            }

    def sample_components(self, domain: str, n_components: Optional[int] = None) -> List[str]:
        """Sample component set based on domain."""
        try:
            domain_config = get_domain_config(domain)
            available = domain_config.typical_components
        except Exception:
            # Fallback to generic components
            available = ["water", "protein", "lipid", "carbohydrate", "cellulose"]

        if n_components is None:
            n_components = int(self.rng.integers(3, min(8, len(available) + 1)))

        n_components = min(n_components, len(available))
        return list(self.rng.choice(available, size=n_components, replace=False))

    def sample(self) -> Dict[str, Any]:
        """
        Sample a complete dataset configuration from the prior.

        Returns:
            Dictionary with all configuration parameters.

        Example:
            >>> sampler = PriorSampler(random_state=42)
            >>> config = sampler.sample()
            >>> print(config["domain"])
            >>> print(config["instrument"])
        """
        # Hierarchical sampling following the DAG
        domain = self.sample_domain()
        instrument_category = self.sample_instrument_category(domain)
        instrument = self.sample_instrument(instrument_category)
        measurement_mode = self.sample_measurement_mode(instrument_category)
        matrix_type = self.sample_matrix_type(domain)

        # Get instrument archetype for wavelength range
        archetype = get_instrument_archetype(instrument)

        return {
            # Domain and application
            "domain": domain,
            "domain_category": self._get_domain_category(domain),

            # Instrument configuration
            "instrument": instrument,
            "instrument_category": instrument_category,
            "wavelength_range": (
                archetype.wavelength_range[0],
                archetype.wavelength_range[1]
            ),
            "spectral_resolution": archetype.spectral_resolution,

            # Measurement configuration
            "measurement_mode": measurement_mode,
            "matrix_type": matrix_type,

            # Environmental conditions
            "temperature": self.sample_temperature(),
            "particle_size": self.sample_particle_size(matrix_type),
            "noise_level": self.sample_noise_level(instrument_category),

            # Components
            "components": self.sample_components(domain),

            # Dataset configuration
            "n_samples": self.sample_n_samples(),
            "target_config": self.sample_target_config(),

            # Metadata
            "random_state": int(self.rng.integers(0, 2**31)),
        }

    def _get_domain_category(self, domain: str) -> str:
        """Get category for a domain."""
        try:
            domain_config = get_domain_config(domain)
            return domain_config.category.value
        except Exception:
            return "research"

    def sample_batch(self, n: int) -> List[Dict[str, Any]]:
        """
        Sample multiple configurations from the prior.

        Args:
            n: Number of configurations to sample.

        Returns:
            List of configuration dictionaries.
        """
        return [self.sample() for _ in range(n)]

    def sample_for_domain(
        self,
        domain: str,
        n_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Sample a configuration constrained to a specific domain.

        Args:
            domain: Domain to sample for.
            n_samples: Optional number of samples (uses prior if None).

        Returns:
            Configuration dictionary for the specified domain.
        """
        # Fixed domain, sample rest hierarchically
        instrument_category = self.sample_instrument_category(domain)
        instrument = self.sample_instrument(instrument_category)
        measurement_mode = self.sample_measurement_mode(instrument_category)
        matrix_type = self.sample_matrix_type(domain)

        archetype = get_instrument_archetype(instrument)

        config = {
            "domain": domain,
            "domain_category": self._get_domain_category(domain),
            "instrument": instrument,
            "instrument_category": instrument_category,
            "wavelength_range": archetype.wavelength_range,
            "spectral_resolution": archetype.spectral_resolution,
            "measurement_mode": measurement_mode,
            "matrix_type": matrix_type,
            "temperature": self.sample_temperature(),
            "particle_size": self.sample_particle_size(matrix_type),
            "noise_level": self.sample_noise_level(instrument_category),
            "components": self.sample_components(domain),
            "n_samples": n_samples or self.sample_n_samples(),
            "target_config": self.sample_target_config(),
            "random_state": int(self.rng.integers(0, 2**31)),
        }
        return config

    def sample_for_instrument(
        self,
        instrument: str,
        n_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Sample a configuration constrained to a specific instrument.

        Args:
            instrument: Instrument name to use.
            n_samples: Optional number of samples.

        Returns:
            Configuration dictionary for the specified instrument.
        """
        archetype = get_instrument_archetype(instrument)
        instrument_category = archetype.category.value

        # Sample domain that's compatible with this instrument category
        # (inverse sampling - find domains where this category is likely)
        compatible_domains = []
        for domain, cat_weights in self.config.instrument_given_domain.items():
            if cat_weights.get(instrument_category, 0) > 0.1:
                compatible_domains.append(domain)

        if compatible_domains:
            domain = self.rng.choice(compatible_domains)
        else:
            domain = self.sample_domain()

        measurement_mode = self.sample_measurement_mode(instrument_category)
        matrix_type = self.sample_matrix_type(domain)

        return {
            "domain": domain,
            "domain_category": self._get_domain_category(domain),
            "instrument": instrument,
            "instrument_category": instrument_category,
            "wavelength_range": archetype.wavelength_range,
            "spectral_resolution": archetype.spectral_resolution,
            "measurement_mode": measurement_mode,
            "matrix_type": matrix_type,
            "temperature": self.sample_temperature(),
            "particle_size": self.sample_particle_size(matrix_type),
            "noise_level": self.sample_noise_level(instrument_category),
            "components": self.sample_components(domain),
            "n_samples": n_samples or self.sample_n_samples(),
            "target_config": self.sample_target_config(),
            "random_state": int(self.rng.integers(0, 2**31)),
        }


# ============================================================================
# Convenience Functions
# ============================================================================


def sample_prior(
    domain: Optional[str] = None,
    instrument: Optional[str] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Quick function to sample a single configuration from default prior.

    Args:
        domain: Optional domain constraint.
        instrument: Optional instrument constraint.
        random_state: Random state for reproducibility.

    Returns:
        Configuration dictionary.

    Example:
        >>> config = sample_prior(domain="food", random_state=42)
        >>> print(config["domain"], config["instrument"])
    """
    sampler = PriorSampler(random_state=random_state)
    if domain:
        return sampler.sample_for_domain(domain)
    elif instrument:
        return sampler.sample_for_instrument(instrument)
    else:
        return sampler.sample()


def sample_prior_batch(
    n: int,
    random_state: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Quick function to sample multiple configurations from default prior.

    Args:
        n: Number of configurations to sample.
        random_state: Random state for reproducibility.

    Returns:
        List of configuration dictionaries.

    Example:
        >>> configs = sample_prior_batch(10, random_state=42)
        >>> for c in configs:
        ...     print(c["domain"], c["instrument"])
    """
    sampler = PriorSampler(random_state=random_state)
    return sampler.sample_batch(n)


def get_domain_compatible_instruments(domain: str) -> List[str]:
    """
    Get list of instruments commonly used with a domain.

    Args:
        domain: Domain name.

    Returns:
        List of instrument names.

    Example:
        >>> instruments = get_domain_compatible_instruments("tablets")
        >>> print(instruments)
    """
    config = NIRSPriorConfig()
    if domain not in config.instrument_given_domain:
        return list(INSTRUMENT_ARCHETYPES.keys())

    # Get likely instrument categories
    cat_weights = config.instrument_given_domain[domain]
    likely_categories = [cat for cat, w in cat_weights.items() if w > 0.1]

    # Get instruments in those categories
    instruments = []
    for name, archetype in INSTRUMENT_ARCHETYPES.items():
        if archetype.category.value in likely_categories:
            instruments.append(name)

    return instruments


def get_instrument_typical_modes(instrument: str) -> List[str]:
    """
    Get typical measurement modes for an instrument.

    Args:
        instrument: Instrument name.

    Returns:
        List of measurement mode names.

    Example:
        >>> modes = get_instrument_typical_modes("viavi_micronir")
        >>> print(modes)
    """
    config = NIRSPriorConfig()
    archetype = get_instrument_archetype(instrument)
    category = archetype.category.value

    if category not in config.mode_given_category:
        return ["reflectance", "transmittance"]

    mode_weights = config.mode_given_category[category]
    return [mode for mode, w in mode_weights.items() if w > 0.05]
