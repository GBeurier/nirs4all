"""
Application domain configurations for synthetic NIRS data generation.

This module provides domain-specific priors and configurations for generating
realistic synthetic NIRS data tailored to specific application areas such as
agriculture, pharmaceutical, food processing, petrochemical, and others.

Each domain configuration includes:
- Typical spectral components (chemical compounds)
- Concentration distributions specific to the domain
- Wavelength ranges commonly used
- Typical number of components in samples
- Domain-specific noise and artifact characteristics

Key Features:
    - 15+ predefined application domains
    - Domain-aware component selection
    - Realistic concentration priors
    - Easy integration with generators

Example:
    >>> from nirs4all.synthesis.domains import (
    ...     get_domain_config,
    ...     APPLICATION_DOMAINS,
    ...     DomainConfig
    ... )
    >>>
    >>> # Get configuration for agricultural samples
    >>> config = get_domain_config("agriculture_grain")
    >>> print(config.typical_components)
    ['starch', 'protein', 'moisture', 'lipid', 'cellulose']

References:
    - Burns, D. A., & Ciurczak, E. W. (2007). Handbook of Near-Infrared
      Analysis (3rd ed.). CRC Press.
    - Williams, P. C., & Norris, K. H. (2001). Near-Infrared Technology
      in the Agricultural and Food Industries (2nd ed.). AACC International.
    - Reich, G. (2005). Near-Infrared Spectroscopy and Imaging: Basic Principles
      and Pharmaceutical Applications. Advanced Drug Delivery Reviews.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


class DomainCategory(str, Enum):
    """Top-level domain categories."""
    AGRICULTURE = "agriculture"
    FOOD = "food"
    PHARMACEUTICAL = "pharmaceutical"
    PETROCHEMICAL = "petrochemical"
    TEXTILE = "textile"
    ENVIRONMENTAL = "environmental"
    BIOMEDICAL = "biomedical"
    POLYMER = "polymer"
    BEVERAGE = "beverage"


@dataclass
class ConcentrationPrior:
    """
    Prior distribution for component concentrations.

    Attributes:
        distribution: Distribution type ('uniform', 'normal', 'lognormal', 'beta').
        params: Parameters for the distribution (distribution-specific).
        min_value: Minimum allowed concentration.
        max_value: Maximum allowed concentration.
    """
    distribution: str = "uniform"
    params: Dict[str, float] = field(default_factory=lambda: {"low": 0.0, "high": 1.0})
    min_value: float = 0.0
    max_value: float = 1.0

    def sample(self, rng: np.random.Generator, n_samples: int = 1) -> np.ndarray:
        """Sample from the concentration prior."""
        if self.distribution == "uniform":
            values = rng.uniform(
                self.params.get("low", 0.0),
                self.params.get("high", 1.0),
                size=n_samples
            )
        elif self.distribution == "normal":
            values = rng.normal(
                self.params.get("mean", 0.5),
                self.params.get("std", 0.1),
                size=n_samples
            )
        elif self.distribution == "lognormal":
            values = rng.lognormal(
                self.params.get("mean", -1.0),
                self.params.get("sigma", 0.5),
                size=n_samples
            )
        elif self.distribution == "beta":
            values = rng.beta(
                self.params.get("a", 2.0),
                self.params.get("b", 5.0),
                size=n_samples
            )
        else:
            values = rng.uniform(0, 1, size=n_samples)

        return np.clip(values, self.min_value, self.max_value)


@dataclass
class DomainConfig:
    """
    Configuration for a specific application domain.

    Encapsulates all domain-specific parameters needed for generating
    realistic synthetic NIRS data.

    Attributes:
        name: Human-readable domain name.
        category: Domain category (agriculture, pharmaceutical, etc.).
        description: Brief description of the domain.
        typical_components: List of predefined component names commonly found.
        component_weights: Relative importance of each component (for selection).
        concentration_priors: Per-component concentration distributions.
        wavelength_range: Typical measurement range (nm).
        n_components_range: Range of number of components per sample.
        noise_level: Typical noise level ('low', 'medium', 'high').
        measurement_mode: Typical measurement geometry.
        typical_sample_types: Examples of sample types in this domain.
        complexity: Overall complexity level for generation.
        additional_params: Domain-specific additional parameters.
    """
    name: str
    category: DomainCategory
    description: str = ""
    typical_components: List[str] = field(default_factory=list)
    component_weights: Optional[Dict[str, float]] = None
    concentration_priors: Dict[str, ConcentrationPrior] = field(default_factory=dict)
    wavelength_range: Tuple[float, float] = (1000, 2500)
    n_components_range: Tuple[int, int] = (3, 8)
    noise_level: str = "medium"
    measurement_mode: str = "reflectance"
    typical_sample_types: List[str] = field(default_factory=list)
    complexity: str = "realistic"
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def get_component_weights(self) -> Dict[str, float]:
        """Get normalized component weights for selection."""
        if self.component_weights is not None:
            return self.component_weights

        # Default: equal weights
        n = len(self.typical_components)
        if n == 0:
            return {}
        return {comp: 1.0 / n for comp in self.typical_components}

    def sample_components(
        self,
        rng: np.random.Generator,
        n_components: Optional[int] = None
    ) -> List[str]:
        """
        Sample components for a sample based on domain priors.

        Args:
            rng: Random number generator.
            n_components: Number of components. If None, samples from range.

        Returns:
            List of component names.
        """
        if n_components is None:
            n_components = rng.integers(
                self.n_components_range[0],
                self.n_components_range[1] + 1
            )

        weights = self.get_component_weights()
        components = list(weights.keys())
        probs = np.array(list(weights.values()))
        probs = probs / probs.sum()

        # Sample without replacement if possible
        n_to_sample = min(n_components, len(components))
        selected = rng.choice(
            components,
            size=n_to_sample,
            replace=False,
            p=probs
        )

        return list(selected)

    def sample_concentrations(
        self,
        rng: np.random.Generator,
        components: List[str],
        n_samples: int = 1
    ) -> np.ndarray:
        """
        Sample concentrations for selected components.

        Args:
            rng: Random number generator.
            components: List of component names.
            n_samples: Number of samples.

        Returns:
            Concentration matrix (n_samples, n_components).
        """
        n_components = len(components)
        concentrations = np.zeros((n_samples, n_components))

        for i, comp in enumerate(components):
            if comp in self.concentration_priors:
                prior = self.concentration_priors[comp]
            else:
                # Default prior
                prior = ConcentrationPrior(
                    distribution="beta",
                    params={"a": 2, "b": 5}
                )

            concentrations[:, i] = prior.sample(rng, n_samples)

        return concentrations


# ============================================================================
# Predefined Domain Configurations
# ============================================================================

APPLICATION_DOMAINS: Dict[str, DomainConfig] = {
    # =========================================================================
    # AGRICULTURE DOMAINS
    # =========================================================================
    "agriculture_grain": DomainConfig(
        name="Grain and Cereals",
        category=DomainCategory.AGRICULTURE,
        description="NIR analysis of wheat, corn, barley, rice, and other cereals",
        typical_components=[
            "starch", "protein", "moisture", "lipid", "cellulose",
            "gluten", "hemicellulose", "dietary_fiber"
        ],
        component_weights={
            "starch": 0.25,
            "protein": 0.20,
            "moisture": 0.20,
            "lipid": 0.10,
            "cellulose": 0.10,
            "gluten": 0.08,
            "hemicellulose": 0.05,
            "dietary_fiber": 0.02
        },
        concentration_priors={
            "starch": ConcentrationPrior("normal", {"mean": 0.65, "std": 0.10}, 0.3, 0.8),
            "protein": ConcentrationPrior("normal", {"mean": 0.12, "std": 0.03}, 0.05, 0.25),
            "moisture": ConcentrationPrior("normal", {"mean": 0.12, "std": 0.02}, 0.08, 0.18),
            "lipid": ConcentrationPrior("beta", {"a": 2, "b": 20}, 0.01, 0.10),
        },
        wavelength_range=(1100, 2500),
        n_components_range=(4, 7),
        noise_level="medium",
        measurement_mode="reflectance",
        typical_sample_types=["wheat flour", "corn meal", "whole grain", "ground samples"],
        complexity="realistic",
    ),

    "agriculture_forage": DomainConfig(
        name="Forage and Feed",
        category=DomainCategory.AGRICULTURE,
        description="NIR analysis of hay, silage, and animal feed",
        typical_components=[
            "protein", "moisture", "cellulose", "hemicellulose", "lignin",
            "starch", "lipid", "nitrogen_compound", "dietary_fiber"
        ],
        component_weights={
            "cellulose": 0.20,
            "protein": 0.18,
            "moisture": 0.18,
            "hemicellulose": 0.12,
            "lignin": 0.10,
            "starch": 0.08,
            "lipid": 0.06,
            "nitrogen_compound": 0.05,
            "dietary_fiber": 0.03
        },
        concentration_priors={
            "protein": ConcentrationPrior("normal", {"mean": 0.15, "std": 0.05}, 0.05, 0.30),
            "moisture": ConcentrationPrior("normal", {"mean": 0.15, "std": 0.05}, 0.05, 0.40),
            "cellulose": ConcentrationPrior("normal", {"mean": 0.30, "std": 0.08}, 0.15, 0.50),
        },
        wavelength_range=(1100, 2500),
        n_components_range=(5, 9),
        noise_level="high",
        measurement_mode="reflectance",
        typical_sample_types=["hay", "silage", "TMR", "pasture"],
        complexity="complex",
    ),

    "agriculture_oilseeds": DomainConfig(
        name="Oilseeds",
        category=DomainCategory.AGRICULTURE,
        description="NIR analysis of soybeans, canola, sunflower, and other oilseeds",
        typical_components=[
            "oil", "protein", "moisture", "starch", "cellulose",
            "unsaturated_fat", "saturated_fat"
        ],
        component_weights={
            "oil": 0.25,
            "protein": 0.25,
            "moisture": 0.15,
            "starch": 0.12,
            "cellulose": 0.10,
            "unsaturated_fat": 0.08,
            "saturated_fat": 0.05
        },
        concentration_priors={
            "oil": ConcentrationPrior("normal", {"mean": 0.20, "std": 0.05}, 0.10, 0.45),
            "protein": ConcentrationPrior("normal", {"mean": 0.35, "std": 0.05}, 0.25, 0.50),
            "moisture": ConcentrationPrior("normal", {"mean": 0.10, "std": 0.02}, 0.05, 0.15),
        },
        wavelength_range=(1100, 2500),
        n_components_range=(4, 7),
        noise_level="medium",
        measurement_mode="reflectance",
        typical_sample_types=["soybeans", "canola", "sunflower seeds", "cottonseed"],
    ),

    "agriculture_fruit": DomainConfig(
        name="Fruits and Vegetables",
        category=DomainCategory.AGRICULTURE,
        description="NIR analysis of fresh produce quality",
        typical_components=[
            "water", "glucose", "fructose", "sucrose", "starch",
            "cellulose", "malic_acid", "citric_acid", "carotenoid"
        ],
        component_weights={
            "water": 0.25,
            "glucose": 0.15,
            "fructose": 0.15,
            "sucrose": 0.12,
            "cellulose": 0.10,
            "starch": 0.08,
            "malic_acid": 0.06,
            "citric_acid": 0.05,
            "carotenoid": 0.04
        },
        concentration_priors={
            "water": ConcentrationPrior("normal", {"mean": 0.85, "std": 0.05}, 0.70, 0.95),
            "glucose": ConcentrationPrior("beta", {"a": 2, "b": 10}, 0.02, 0.15),
            "fructose": ConcentrationPrior("beta", {"a": 2, "b": 10}, 0.02, 0.15),
        },
        wavelength_range=(700, 1100),  # Shorter range for fresh produce
        n_components_range=(5, 8),
        noise_level="medium",
        measurement_mode="reflectance",
        typical_sample_types=["apples", "tomatoes", "citrus", "berries"],
    ),

    # =========================================================================
    # FOOD DOMAINS
    # =========================================================================
    "food_dairy": DomainConfig(
        name="Dairy Products",
        category=DomainCategory.FOOD,
        description="NIR analysis of milk, cheese, and dairy products",
        typical_components=[
            "water", "lactose", "casein", "lipid", "moisture", "protein"
        ],
        component_weights={
            "water": 0.25,
            "lactose": 0.20,
            "casein": 0.18,
            "lipid": 0.18,
            "moisture": 0.10,
            "protein": 0.09
        },
        concentration_priors={
            "water": ConcentrationPrior("normal", {"mean": 0.87, "std": 0.02}, 0.80, 0.92),
            "lipid": ConcentrationPrior("normal", {"mean": 0.04, "std": 0.01}, 0.01, 0.08),
            "protein": ConcentrationPrior("normal", {"mean": 0.035, "std": 0.005}, 0.02, 0.05),
            "lactose": ConcentrationPrior("normal", {"mean": 0.048, "std": 0.003}, 0.04, 0.055),
        },
        wavelength_range=(1100, 2500),
        n_components_range=(4, 6),
        noise_level="low",
        measurement_mode="transflectance",
        typical_sample_types=["milk", "cheese", "yogurt", "cream"],
    ),

    "food_meat": DomainConfig(
        name="Meat and Poultry",
        category=DomainCategory.FOOD,
        description="NIR analysis of meat composition and quality",
        typical_components=[
            "water", "protein", "lipid", "moisture", "collagen"
        ],
        component_weights={
            "water": 0.25,
            "protein": 0.30,
            "lipid": 0.25,
            "moisture": 0.12,
            "collagen": 0.08
        },
        concentration_priors={
            "water": ConcentrationPrior("normal", {"mean": 0.70, "std": 0.05}, 0.55, 0.80),
            "protein": ConcentrationPrior("normal", {"mean": 0.20, "std": 0.03}, 0.12, 0.28),
            "lipid": ConcentrationPrior("lognormal", {"mean": -2, "sigma": 0.5}, 0.02, 0.40),
        },
        wavelength_range=(900, 1700),
        n_components_range=(4, 5),
        noise_level="medium",
        measurement_mode="reflectance",
        typical_sample_types=["beef", "pork", "chicken", "ground meat"],
        additional_params={"collagen": "derived from protein group"},
    ),

    "food_bakery": DomainConfig(
        name="Bakery Products",
        category=DomainCategory.FOOD,
        description="NIR analysis of bread, cookies, and baked goods",
        typical_components=[
            "starch", "gluten", "moisture", "lipid", "sucrose",
            "glucose", "protein", "cellulose"
        ],
        component_weights={
            "starch": 0.25,
            "gluten": 0.18,
            "moisture": 0.18,
            "lipid": 0.12,
            "sucrose": 0.10,
            "glucose": 0.07,
            "protein": 0.06,
            "cellulose": 0.04
        },
        wavelength_range=(1100, 2500),
        n_components_range=(5, 8),
        noise_level="medium",
        measurement_mode="reflectance",
        typical_sample_types=["bread", "cookies", "crackers", "pastries"],
    ),

    "food_chocolate": DomainConfig(
        name="Confectionery and Chocolate",
        category=DomainCategory.FOOD,
        description="NIR analysis of chocolate and confectionery products",
        typical_components=[
            "lipid", "sucrose", "moisture", "protein", "starch",
            "caffeine", "unsaturated_fat"
        ],
        component_weights={
            "lipid": 0.25,
            "sucrose": 0.25,
            "moisture": 0.12,
            "protein": 0.12,
            "starch": 0.10,
            "caffeine": 0.08,
            "unsaturated_fat": 0.08
        },
        wavelength_range=(1100, 2500),
        n_components_range=(4, 7),
        noise_level="low",
        measurement_mode="reflectance",
        typical_sample_types=["dark chocolate", "milk chocolate", "cocoa powder"],
    ),

    # =========================================================================
    # PHARMACEUTICAL DOMAINS
    # =========================================================================
    "pharma_tablets": DomainConfig(
        name="Pharmaceutical Tablets",
        category=DomainCategory.PHARMACEUTICAL,
        description="NIR analysis of tablet formulations and API content",
        typical_components=[
            "starch", "cellulose", "lactose", "moisture", "aspirin",
            "paracetamol", "caffeine"
        ],
        component_weights={
            "starch": 0.18,
            "cellulose": 0.18,
            "lactose": 0.18,
            "moisture": 0.12,
            "aspirin": 0.12,
            "paracetamol": 0.12,
            "caffeine": 0.10
        },
        concentration_priors={
            "moisture": ConcentrationPrior("normal", {"mean": 0.02, "std": 0.005}, 0.005, 0.05),
        },
        wavelength_range=(1100, 2500),
        n_components_range=(4, 7),
        noise_level="low",
        measurement_mode="reflectance",
        typical_sample_types=["tablets", "capsules", "granules"],
        complexity="simple",
    ),

    "pharma_powder_blends": DomainConfig(
        name="Pharmaceutical Powder Blends",
        category=DomainCategory.PHARMACEUTICAL,
        description="NIR monitoring of powder blending uniformity",
        typical_components=[
            "starch", "cellulose", "lactose", "moisture",
            "aspirin", "paracetamol", "caffeine"
        ],
        wavelength_range=(1100, 2500),
        n_components_range=(3, 6),
        noise_level="medium",
        measurement_mode="reflectance",
        typical_sample_types=["powder blend", "premix", "granulation"],
    ),

    "pharma_raw_materials": DomainConfig(
        name="Pharmaceutical Raw Materials",
        category=DomainCategory.PHARMACEUTICAL,
        description="NIR identification and verification of raw materials",
        typical_components=[
            "starch", "cellulose", "lactose", "glucose", "sucrose",
            "aspirin", "paracetamol", "caffeine", "urea"
        ],
        wavelength_range=(1100, 2500),
        n_components_range=(1, 3),  # Usually single component verification
        noise_level="low",
        measurement_mode="reflectance",
        typical_sample_types=["excipients", "APIs", "intermediates"],
    ),

    # =========================================================================
    # PETROCHEMICAL DOMAINS
    # =========================================================================
    "petrochem_fuels": DomainConfig(
        name="Petroleum Fuels",
        category=DomainCategory.PETROCHEMICAL,
        description="NIR analysis of gasoline, diesel, and aviation fuels",
        typical_components=[
            "alkane", "aromatic", "oil", "unsaturated_fat", "methanol", "ethanol"
        ],
        component_weights={
            "alkane": 0.35,
            "aromatic": 0.25,
            "oil": 0.20,
            "unsaturated_fat": 0.10,
            "methanol": 0.05,
            "ethanol": 0.05
        },
        wavelength_range=(900, 1700),
        n_components_range=(3, 6),
        noise_level="low",
        measurement_mode="transmission",
        typical_sample_types=["gasoline", "diesel", "jet fuel", "biodiesel"],
    ),

    "petrochem_polymers": DomainConfig(
        name="Petrochemical Polymers",
        category=DomainCategory.PETROCHEMICAL,
        description="NIR analysis of synthetic polymers and plastics",
        typical_components=[
            "polyethylene", "polystyrene", "nylon", "polyester", "natural_rubber"
        ],
        wavelength_range=(1100, 2500),
        n_components_range=(1, 4),
        noise_level="low",
        measurement_mode="reflectance",
        typical_sample_types=["pellets", "films", "fibers", "molded parts"],
    ),

    # =========================================================================
    # TEXTILE DOMAINS
    # =========================================================================
    "textile_natural": DomainConfig(
        name="Natural Fibers",
        category=DomainCategory.TEXTILE,
        description="NIR analysis of cotton, wool, and natural fibers",
        typical_components=[
            "cellulose", "cotton", "moisture", "protein", "waxes", "lignin"
        ],
        component_weights={
            "cellulose": 0.30,
            "cotton": 0.25,
            "moisture": 0.20,
            "protein": 0.10,
            "waxes": 0.10,
            "lignin": 0.05
        },
        wavelength_range=(1100, 2500),
        n_components_range=(3, 6),
        noise_level="medium",
        measurement_mode="reflectance",
        typical_sample_types=["cotton", "wool", "silk", "linen"],
    ),

    "textile_synthetic": DomainConfig(
        name="Synthetic Fibers",
        category=DomainCategory.TEXTILE,
        description="NIR analysis of polyester, nylon, and synthetic fibers",
        typical_components=[
            "polyester", "nylon", "polystyrene", "moisture"
        ],
        wavelength_range=(1100, 2500),
        n_components_range=(2, 4),
        noise_level="low",
        measurement_mode="reflectance",
        typical_sample_types=["polyester", "nylon", "acrylic", "blends"],
    ),

    # =========================================================================
    # ENVIRONMENTAL DOMAINS
    # =========================================================================
    "environmental_soil": DomainConfig(
        name="Soil Analysis",
        category=DomainCategory.ENVIRONMENTAL,
        description="NIR analysis of soil properties and composition",
        typical_components=[
            "moisture", "carbonates", "kaolinite", "gypsum",
            "cellulose", "lignin", "protein"
        ],
        component_weights={
            "moisture": 0.25,
            "carbonates": 0.18,
            "kaolinite": 0.18,
            "cellulose": 0.12,
            "lignin": 0.10,
            "gypsum": 0.10,
            "protein": 0.07
        },
        wavelength_range=(1100, 2500),
        n_components_range=(4, 7),
        noise_level="high",
        measurement_mode="reflectance",
        typical_sample_types=["topsoil", "subsoil", "sediments"],
        complexity="complex",
    ),

    "environmental_water": DomainConfig(
        name="Water Quality",
        category=DomainCategory.ENVIRONMENTAL,
        description="NIR analysis of water quality parameters",
        typical_components=[
            "water", "glucose", "protein", "urea", "acetic_acid"
        ],
        wavelength_range=(900, 1100),  # Short-wave NIR for water
        n_components_range=(2, 4),
        noise_level="medium",
        measurement_mode="transmission",
        typical_sample_types=["surface water", "wastewater", "process water"],
    ),

    # =========================================================================
    # BEVERAGE DOMAINS
    # =========================================================================
    "beverage_wine": DomainConfig(
        name="Wine and Spirits",
        category=DomainCategory.BEVERAGE,
        description="NIR analysis of wine, beer, and alcoholic beverages",
        typical_components=[
            "water", "ethanol", "glucose", "fructose", "glycerol",
            "tartaric_acid", "malic_acid", "tannins"
        ],
        component_weights={
            "water": 0.20,
            "ethanol": 0.25,
            "glucose": 0.10,
            "fructose": 0.10,
            "glycerol": 0.10,
            "tartaric_acid": 0.10,
            "malic_acid": 0.08,
            "tannins": 0.07
        },
        concentration_priors={
            "ethanol": ConcentrationPrior("normal", {"mean": 0.13, "std": 0.02}, 0.08, 0.18),
            "glucose": ConcentrationPrior("lognormal", {"mean": -3, "sigma": 0.8}, 0.0, 0.15),
        },
        wavelength_range=(900, 1700),
        n_components_range=(5, 8),
        noise_level="low",
        measurement_mode="transmission",
        typical_sample_types=["red wine", "white wine", "beer", "spirits"],
    ),

    "beverage_juice": DomainConfig(
        name="Fruit Juices",
        category=DomainCategory.BEVERAGE,
        description="NIR analysis of fruit juices and beverages",
        typical_components=[
            "water", "glucose", "fructose", "sucrose",
            "citric_acid", "malic_acid", "carotenoid"
        ],
        wavelength_range=(900, 1100),
        n_components_range=(4, 7),
        noise_level="low",
        measurement_mode="transmission",
        typical_sample_types=["orange juice", "apple juice", "grape juice"],
    ),

    # =========================================================================
    # BIOMEDICAL DOMAINS
    # =========================================================================
    "biomedical_tissue": DomainConfig(
        name="Tissue Analysis",
        category=DomainCategory.BIOMEDICAL,
        description="NIR spectroscopy of biological tissues",
        typical_components=[
            "water", "lipid", "protein", "glucose", "hemoglobin"
        ],
        component_weights={
            "water": 0.35,
            "lipid": 0.20,
            "protein": 0.25,
            "glucose": 0.10,
            "hemoglobin": 0.10
        },
        wavelength_range=(700, 1100),  # Optical window
        n_components_range=(3, 5),
        noise_level="high",
        measurement_mode="reflectance",
        typical_sample_types=["skin", "muscle", "fat tissue"],
        additional_params={"hemoglobin": "simulated with carotenoid"},
    ),
}


# ============================================================================
# Domain Access Functions
# ============================================================================

def get_domain_config(domain_name: str) -> DomainConfig:
    """
    Get configuration for a specific domain.

    Args:
        domain_name: Name of the domain (key in APPLICATION_DOMAINS).

    Returns:
        DomainConfig for the specified domain.

    Raises:
        ValueError: If domain is not found.

    Example:
        >>> config = get_domain_config("agriculture_grain")
        >>> print(config.name)
        'Grain and Cereals'
    """
    if domain_name not in APPLICATION_DOMAINS:
        available = list(APPLICATION_DOMAINS.keys())
        raise ValueError(
            f"Unknown domain: '{domain_name}'. Available domains: {available}"
        )
    return APPLICATION_DOMAINS[domain_name]


def list_domains(category: Optional[DomainCategory] = None) -> List[str]:
    """
    List available domain names.

    Args:
        category: Optional category filter.

    Returns:
        List of domain names.

    Example:
        >>> list_domains(DomainCategory.AGRICULTURE)
        ['agriculture_grain', 'agriculture_forage', ...]
    """
    domains = []
    for name, config in APPLICATION_DOMAINS.items():
        if category is None or config.category == category:
            domains.append(name)
    return domains


def get_domain_components(domain_name: str) -> List[str]:
    """
    Get typical components for a domain.

    Args:
        domain_name: Name of the domain.

    Returns:
        List of component names.

    Example:
        >>> get_domain_components("food_dairy")
        ['water', 'lactose', 'casein', 'lipid', 'moisture', 'protein']
    """
    config = get_domain_config(domain_name)
    return config.typical_components


def get_domains_for_component(component_name: str) -> List[str]:
    """
    Find domains that typically contain a specific component.

    Args:
        component_name: Name of the component.

    Returns:
        List of domain names containing this component.

    Example:
        >>> get_domains_for_component("protein")
        ['agriculture_grain', 'food_meat', 'biomedical_tissue', ...]
    """
    domains = []
    for name, config in APPLICATION_DOMAINS.items():
        if component_name in config.typical_components:
            domains.append(name)
    return domains


def create_domain_aware_library(
    domain_name: str,
    n_samples: int = 100,
    random_state: Optional[int] = None
) -> Tuple[List[str], np.ndarray]:
    """
    Create component selection and concentrations based on domain priors.

    This function samples components and their concentrations according to
    domain-specific distributions.

    Args:
        domain_name: Name of the domain.
        n_samples: Number of samples to generate concentrations for.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (component_names, concentration_matrix).

    Example:
        >>> components, concentrations = create_domain_aware_library(
        ...     "food_dairy",
        ...     n_samples=50,
        ...     random_state=42
        ... )
        >>> print(components)
        ['water', 'lactose', 'casein', 'lipid']
        >>> print(concentrations.shape)
        (50, 4)
    """
    config = get_domain_config(domain_name)
    rng = np.random.default_rng(random_state)

    # Sample components
    components = config.sample_components(rng)

    # Sample concentrations
    concentrations = config.sample_concentrations(rng, components, n_samples)

    return components, concentrations


# ============================================================================
# Module-level exports
# ============================================================================

__all__ = [
    # Classes
    "DomainCategory",
    "ConcentrationPrior",
    "DomainConfig",
    # Data
    "APPLICATION_DOMAINS",
    # Functions
    "get_domain_config",
    "list_domains",
    "get_domain_components",
    "get_domains_for_component",
    "create_domain_aware_library",
]
