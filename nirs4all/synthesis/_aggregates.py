"""Aggregate spectral components for synthetic NIRS generation.

This module provides predefined mixtures representing common sample types in
NIRS applications. Aggregates define realistic compositions with variability
ranges for generating diverse samples.

Aggregate components are useful for:
    - Generating realistic product samples for model training
    - Creating domain-specific synthetic datasets
    - Testing calibration transfer across product types

Example:
    >>> from nirs4all.synthesis import get_aggregate, expand_aggregate
    >>>
    >>> # Get wheat grain composition
    >>> wheat = get_aggregate("wheat_grain")
    >>> print(wheat.description)
    >>>
    >>> # Expand with variability for training data
    >>> compositions = [expand_aggregate("wheat_grain", variability=True) for _ in range(100)]

Bibliography / references:
    `AggregateComponent.references` stores *bibliography keys* (stable IDs)
    resolved by the project's bibliography backend (e.g., a BibTeX/Sphinx layer).
    Typical keys include:
        - USDA_FDC_2019
        - Osborne_Fearn_Hindle_1993
        - Williams_Norris_2001

    The numerical values in aggregates are intended as *typical proximate
    compositions* (water/moisture, protein, lipid, carbohydrates/fiber proxies)
    with realistic ranges; they are not meant to be exact for a given cultivar,
    origin, or processing condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AggregateComponent:
    """
    Predefined mixture of spectral components for common sample types.

    Each aggregate defines a typical composition for a product type along
    with realistic variability ranges for generating diverse samples.

    Attributes:
        name: Unique identifier for the aggregate.
        components: Base composition as {component_name: weight}.
            Weights should approximately sum to 1.0 (allowing for ash, etc.).
        description: Human-readable description of the aggregate.
        domain: Application domain (e.g., "agriculture", "food", "pharmaceutical").
        category: Product category within domain (e.g., "grain", "dairy", "solid_dosage").
        variability: Optional weight ranges for components with natural variation.
            Format: {component_name: (min_weight, max_weight)}.
        correlations: Optional correlation constraints between components.
            Format: [(comp1, comp2, correlation_coefficient), ...].
        tags: Classification tags for filtering (e.g., ["grain", "cereal"]).
        references: Literature or database citations.

    Example:
        >>> wheat = AggregateComponent(
        ...     name="wheat_grain",
        ...     components={"starch": 0.65, "protein": 0.12, "moisture": 0.12},
        ...     description="Typical wheat grain composition",
        ...     domain="agriculture",
        ...     category="grain",
        ...     variability={"protein": (0.08, 0.18), "moisture": (0.08, 0.15)},
        ... )
    """

    name: str
    components: Dict[str, float]
    description: str
    domain: str
    category: str = ""
    spectral_category: str = ""  # Primary chemical/spectral category (proteins, lipids, carbohydrates, alcohols, minerals, etc.)
    variability: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    correlations: List[Tuple[str, str, float]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)

    def validate(self) -> List[str]:
        """
        Validate aggregate definition.

        Returns:
            List of validation issues (empty if all valid).
        """
        issues = []

        # Check components sum to approximately 1.0
        total = sum(self.components.values())
        if not (0.9 <= total <= 1.1):
            issues.append(f"Component weights sum to {total:.3f}, expected ~1.0")

        # Check all weights are positive
        for comp, weight in self.components.items():
            if weight < 0:
                issues.append(f"Component '{comp}' has negative weight: {weight}")

        # Check variability ranges are valid
        for comp, (low, high) in self.variability.items():
            if comp not in self.components:
                issues.append(f"Variability defined for unknown component: {comp}")
            if low > high:
                issues.append(f"Variability range invalid for '{comp}': ({low}, {high})")
            if low < 0 or high > 1:
                issues.append(f"Variability out of bounds for '{comp}': ({low}, {high})")

        return issues

    def info(self) -> str:
        """
        Return formatted information about the aggregate.

        Returns:
            Human-readable string with aggregate details.
        """
        lines = [
            f"Aggregate: {self.name}",
            f"Description: {self.description}",
            f"Domain: {self.domain}",
            f"Category: {self.category or 'N/A'}",
            f"Spectral Category: {self.spectral_category or 'N/A'}",
            f"Components ({len(self.components)}):",
        ]
        for comp, weight in sorted(self.components.items(), key=lambda x: -x[1]):
            var_str = ""
            if comp in self.variability:
                low, high = self.variability[comp]
                var_str = f" (varies: {low:.1%}-{high:.1%})"
            lines.append(f"  - {comp}: {weight:.1%}{var_str}")
        if self.tags:
            lines.append(f"Tags: {', '.join(self.tags)}")
        return "\n".join(lines)


# =============================================================================
# Predefined Aggregate Components
# =============================================================================

AGGREGATE_COMPONENTS: Dict[str, AggregateComponent] = {
    # =========================================================================
    # AGRICULTURAL - Grains
    # =========================================================================
    "wheat_grain": AggregateComponent(
        name="wheat_grain",
        components={
            "starch": 0.65,
            "protein": 0.12,
            "moisture": 0.12,
            "lipid": 0.02,
            "cellulose": 0.08,
            # Remaining ~1% is ash/minerals (not modeled spectrally)
        },
        description="Typical wheat grain composition",
        domain="agriculture",
        category="grain",
        spectral_category="carbohydrates",
        variability={
            "protein": (0.08, 0.18),  # Feed wheat to high-protein bread wheat
            "moisture": (0.08, 0.15),  # Storage moisture range
            "starch": (0.58, 0.72),
        },
        correlations=[
            ("protein", "starch", -0.85),  # Higher protein = lower starch
        ],
        tags=["grain", "cereal", "agriculture"],
        references=["USDA_FDC_2019", "Osborne_Fearn_Hindle_1993"],
    ),

    "corn_grain": AggregateComponent(
        name="corn_grain",
        components={
            "starch": 0.72,
            "protein": 0.09,
            "moisture": 0.11,
            "lipid": 0.04,
            "cellulose": 0.03,
        },
        description="Typical corn/maize grain composition",
        domain="agriculture",
        category="grain",
        spectral_category="carbohydrates",
        variability={
            "protein": (0.06, 0.12),
            "moisture": (0.10, 0.15),
            "lipid": (0.03, 0.06),
        },
        tags=["grain", "cereal", "agriculture"],
        references=["USDA_FDC_2019"],
    ),

    "soybean": AggregateComponent(
        name="soybean",
        components={
            "protein": 0.36,
            "lipid": 0.20,
            "starch": 0.05,
            "cellulose": 0.15,
            "moisture": 0.10,
            "sucrose": 0.05,
        },
        description="Typical soybean composition",
        domain="agriculture",
        category="legume",
        spectral_category="proteins",
        variability={
            "protein": (0.32, 0.42),
            "lipid": (0.16, 0.24),
            "moisture": (0.08, 0.13),
        },
        correlations=[
            ("protein", "lipid", -0.4),  # Slight inverse relationship
        ],
        tags=["legume", "oilseed", "agriculture"],
        references=["USDA_FDC_2019"],
    ),

    "rice_grain": AggregateComponent(
        name="rice_grain",
        components={
            "starch": 0.78,
            "protein": 0.07,
            "moisture": 0.12,
            "lipid": 0.01,
            "cellulose": 0.01,
        },
        description="Polished rice grain composition",
        domain="agriculture",
        category="grain",
        spectral_category="carbohydrates",
        variability={
            "protein": (0.05, 0.10),
            "moisture": (0.10, 0.14),
            "starch": (0.74, 0.82),
        },
        tags=["grain", "cereal", "agriculture"],
        references=["USDA_FDC_2019"],
    ),

    "barley_grain": AggregateComponent(
        name="barley_grain",
        components={
            "starch": 0.60,
            "protein": 0.11,
            "moisture": 0.11,
            "lipid": 0.02,
            "cellulose": 0.14,
        },
        description="Typical barley grain composition",
        domain="agriculture",
        category="grain",
        spectral_category="carbohydrates",
        variability={
            "protein": (0.08, 0.15),
            "moisture": (0.09, 0.13),
        },
        tags=["grain", "cereal", "agriculture", "malting"],
        references=["USDA_FDC_2019"],
    ),

    "oat_grain": AggregateComponent(
        name="oat_grain",
        components={
            # Oat groats are relatively high in lipid compared to other cereals.
            "starch": 0.58,
            "protein": 0.13,
            "moisture": 0.11,
            "lipid": 0.07,
            "cellulose": 0.10,
        },
        description="Typical oat grain (groats) composition",
        domain="agriculture",
        category="grain",
        spectral_category="carbohydrates",
        variability={
            "protein": (0.10, 0.18),
            "lipid": (0.05, 0.10),
            "moisture": (0.09, 0.14),
            "starch": (0.50, 0.65),
        },
        correlations=[
            ("lipid", "starch", -0.5),
        ],
        tags=["grain", "cereal", "agriculture"],
        references=["USDA_FDC_2019"],
    ),

    "rye_grain": AggregateComponent(
        name="rye_grain",
        components={
            "starch": 0.64,
            "protein": 0.10,
            "moisture": 0.12,
            "lipid": 0.02,
            "cellulose": 0.11,
        },
        description="Typical rye grain composition",
        domain="agriculture",
        category="grain",
        spectral_category="carbohydrates",
        variability={
            "protein": (0.08, 0.14),
            "moisture": (0.09, 0.14),
            "cellulose": (0.08, 0.16),
        },
        tags=["grain", "cereal", "agriculture"],
        references=["USDA_FDC_2019"],
    ),

    "sorghum_grain": AggregateComponent(
        name="sorghum_grain",
        components={
            "starch": 0.70,
            "protein": 0.11,
            "moisture": 0.12,
            "lipid": 0.03,
            "cellulose": 0.04,
        },
        description="Typical sorghum grain composition",
        domain="agriculture",
        category="grain",
        spectral_category="carbohydrates",
        variability={
            "protein": (0.08, 0.14),
            "lipid": (0.02, 0.05),
            "moisture": (0.10, 0.14),
        },
        tags=["grain", "cereal", "agriculture"],
        references=["USDA_FDC_2019"],
    ),

    "rapeseed_canola": AggregateComponent(
        name="rapeseed_canola",
        components={
            # Oilseed: high lipid and moderate protein; fiber proxied by cellulose.
            "lipid": 0.40,
            "protein": 0.20,
            "cellulose": 0.12,
            "moisture": 0.08,
            "sucrose": 0.05,
            "starch": 0.05,
        },
        description="Typical rapeseed/canola seed composition (proximate)",
        domain="agriculture",
        category="oilseed",
        spectral_category="lipids",
        variability={
            "lipid": (0.35, 0.48),
            "protein": (0.16, 0.26),
            "moisture": (0.06, 0.12),
        },
        correlations=[
            ("lipid", "protein", -0.4),
        ],
        tags=["seed", "oilseed", "agriculture"],
        references=["USDA_FDC_2019"],
    ),

    "sunflower_seed": AggregateComponent(
        name="sunflower_seed",
        components={
            "lipid": 0.50,
            "protein": 0.20,
            "cellulose": 0.10,
            "moisture": 0.06,
            "sucrose": 0.07,
            "starch": 0.05,
        },
        description="Typical sunflower seed composition (proximate)",
        domain="agriculture",
        category="oilseed",
        spectral_category="lipids",
        variability={
            "lipid": (0.42, 0.58),
            "protein": (0.16, 0.26),
            "moisture": (0.05, 0.10),
        },
        correlations=[
            ("lipid", "protein", -0.3),
        ],
        tags=["seed", "oilseed", "agriculture"],
        references=["USDA_FDC_2019"],
    ),

    "chickpea": AggregateComponent(
        name="chickpea",
        components={
            "starch": 0.45,
            "protein": 0.19,
            "moisture": 0.11,
            "lipid": 0.06,
            "cellulose": 0.15,
            "sucrose": 0.04,
        },
        description="Typical chickpea (dry) composition",
        domain="agriculture",
        category="legume",
        spectral_category="carbohydrates",
        variability={
            "protein": (0.16, 0.24),
            "lipid": (0.04, 0.09),
            "moisture": (0.09, 0.14),
        },
        correlations=[
            ("protein", "starch", -0.4),
        ],
        tags=["legume", "pulse", "agriculture"],
        references=["USDA_FDC_2019"],
    ),

    "lentil": AggregateComponent(
        name="lentil",
        components={
            "starch": 0.48,
            "protein": 0.25,
            "moisture": 0.11,
            "lipid": 0.02,
            "cellulose": 0.12,
            "sucrose": 0.02,
        },
        description="Typical lentil (dry) composition",
        domain="agriculture",
        category="legume",
        spectral_category="proteins",
        variability={
            "protein": (0.20, 0.30),
            "moisture": (0.09, 0.14),
            "cellulose": (0.08, 0.16),
        },
        tags=["legume", "pulse", "agriculture"],
        references=["USDA_FDC_2019"],
    ),


    # =========================================================================
    # AGRICULTURAL - Plant Tissue
    # =========================================================================
    "leaf_green": AggregateComponent(
        name="leaf_green",
        components={
            "water": 0.70,
            "cellulose": 0.12,
            "protein": 0.05,
            "starch": 0.03,
            "chlorophyll": 0.005,
            "carotenoid": 0.001,
            "hemicellulose": 0.05,
            "lignin": 0.02,
        },
        description="Fresh green leaf tissue",
        domain="agriculture",
        category="plant_tissue",
        spectral_category="pigments",
        variability={
            "water": (0.60, 0.85),
            "chlorophyll": (0.002, 0.010),
            "protein": (0.02, 0.08),
        },
        tags=["leaf", "vegetation", "agriculture"],
        references=["Williams_Norris_2001"],
    ),

    "forage_grass": AggregateComponent(
        name="forage_grass",
        components={
            "water": 0.75,
            "cellulose": 0.10,
            "protein": 0.04,
            "hemicellulose": 0.05,
            "lignin": 0.02,
            "starch": 0.02,
            "lipid": 0.01,
        },
        description="Fresh forage grass for animal feed",
        domain="agriculture",
        category="forage",
        spectral_category="water_related",
        variability={
            "water": (0.65, 0.85),
            "protein": (0.02, 0.06),
            "cellulose": (0.08, 0.15),
        },
        tags=["forage", "grass", "animal_feed"],
        references=["Williams_Norris_2001"],
    ),

    "alfalfa_hay": AggregateComponent(
        name="alfalfa_hay",
        components={
            # Dried forage: lower water, higher fiber fractions.
            "moisture": 0.10,
            "cellulose": 0.25,
            "hemicellulose": 0.20,
            "lignin": 0.08,
            "protein": 0.18,
            "starch": 0.10,
            "lipid": 0.04,
            "chlorophyll": 0.002,
            "carotenoid": 0.001,
        },
        description="Alfalfa hay (dried forage) composition",
        domain="agriculture",
        category="forage",
        spectral_category="proteins",
        variability={
            "moisture": (0.07, 0.15),
            "protein": (0.14, 0.24),
            "cellulose": (0.20, 0.32),
            "lignin": (0.05, 0.12),
        },
        tags=["forage", "hay", "animal_feed"],
        references=["Williams_Norris_2001"],
    ),

    "silage_maize": AggregateComponent(
        name="silage_maize",
        components={
            # Fermented forage: high water + lactic acid from fermentation.
            "water": 0.70,
            "starch": 0.10,
            "cellulose": 0.07,
            "hemicellulose": 0.06,
            "lignin": 0.03,
            "protein": 0.03,
            "lipid": 0.01,
            "lactic_acid": 0.02,
        },
        description="Maize (corn) silage composition (proximate)",
        domain="agriculture",
        category="forage",
        spectral_category="water_related",
        variability={
            "water": (0.60, 0.80),
            "starch": (0.05, 0.18),
            "lactic_acid": (0.005, 0.040),
        },
        tags=["forage", "silage", "fermented", "animal_feed"],
        references=["Williams_Norris_2001"],
    ),

    # =========================================================================
    # FOOD - Dairy
    # =========================================================================
    "milk": AggregateComponent(
        name="milk",
        components={
            "water": 0.87,
            "casein": 0.028,
            "whey": 0.006,
            "lipid": 0.04,
            "lactose": 0.05,
        },
        description="Whole cow's milk composition",
        domain="food",
        category="dairy",
        spectral_category="lipids",
        variability={
            "lipid": (0.005, 0.06),  # Skim to whole
            "casein": (0.024, 0.032),
            "lactose": (0.045, 0.055),
        },
        tags=["dairy", "liquid", "milk"],
        references=["USDA_FDC_2019"],
    ),

    "cheese_cheddar": AggregateComponent(
        name="cheese_cheddar",
        components={
            "casein": 0.25,
            "lipid": 0.33,
            "moisture": 0.36,
            "lactose": 0.02,
        },
        description="Cheddar cheese composition",
        domain="food",
        category="dairy",
        spectral_category="lipids",
        variability={
            "moisture": (0.30, 0.42),  # Aged to fresh
            "lipid": (0.28, 0.38),
            "casein": (0.20, 0.30),
        },
        correlations=[
            ("moisture", "casein", -0.5),  # More moisture = less protein density
        ],
        tags=["dairy", "cheese", "solid"],
        references=["USDA_FDC_2019"],
    ),

    "yogurt": AggregateComponent(
        name="yogurt",
        components={
            "water": 0.85,
            "casein": 0.035,
            "whey": 0.015,
            "lipid": 0.03,
            "lactose": 0.04,
            "lactic_acid": 0.01,
        },
        description="Plain yogurt composition",
        domain="food",
        category="dairy",
        spectral_category="water_related",
        variability={
            "lipid": (0.001, 0.10),  # Non-fat to full-fat
            "lactic_acid": (0.005, 0.015),
        },
        tags=["dairy", "fermented", "yogurt"],
        references=["USDA_FDC_2019"],
    ),

    "butter": AggregateComponent(
        name="butter",
        components={
            "lipid": 0.82,
            "water": 0.16,
            "casein": 0.01,
            "lactose": 0.01,
        },
        description="Butter composition",
        domain="food",
        category="dairy",
        spectral_category="lipids",
        variability={
            "lipid": (0.75, 0.86),
            "water": (0.14, 0.22),
        },
        correlations=[
            ("lipid", "water", -0.9),
        ],
        tags=["dairy", "fat", "butter"],
        references=["USDA_FDC_2019"],
    ),

    "cream": AggregateComponent(
        name="cream",
        components={
            "water": 0.60,
            "lipid": 0.35,
            "casein": 0.02,
            "lactose": 0.03,
        },
        description="Cream (high-fat milk fraction) composition",
        domain="food",
        category="dairy",
        spectral_category="lipids",
        variability={
            "lipid": (0.10, 0.45),
            "water": (0.50, 0.80),
        },
        correlations=[
            ("lipid", "water", -0.85),
        ],
        tags=["dairy", "cream", "fat"],
        references=["USDA_FDC_2019"],
    ),

    # =========================================================================
    # FOOD - Cereals & Bakery
    # =========================================================================
    "wheat_flour": AggregateComponent(
        name="wheat_flour",
        components={
            "starch": 0.74,
            "protein": 0.11,
            "moisture": 0.12,
            "lipid": 0.015,
            "cellulose": 0.015,
        },
        description="Wheat flour (refined) composition",
        domain="food",
        category="bakery",
        spectral_category="carbohydrates",
        variability={
            "protein": (0.08, 0.15),
            "moisture": (0.10, 0.14),
            "starch": (0.68, 0.78),
        },
        correlations=[
            ("protein", "starch", -0.7),
        ],
        tags=["grain", "flour", "bakery"],
        references=["USDA_FDC_2019", "Osborne_Fearn_Hindle_1993"],
    ),

    "corn_flour": AggregateComponent(
        name="corn_flour",
        components={
            "starch": 0.76,
            "protein": 0.07,
            "moisture": 0.12,
            "lipid": 0.03,
            "cellulose": 0.02,
        },
        description="Corn (maize) flour composition",
        domain="food",
        category="bakery",
        spectral_category="carbohydrates",
        variability={
            "protein": (0.05, 0.10),
            "lipid": (0.02, 0.05),
            "moisture": (0.10, 0.14),
        },
        tags=["grain", "flour", "bakery"],
        references=["USDA_FDC_2019"],
    ),

    "bread_white": AggregateComponent(
        name="bread_white",
        components={
            "moisture": 0.38,
            "starch": 0.45,
            "protein": 0.08,
            "lipid": 0.03,
            "cellulose": 0.03,
            "sucrose": 0.03,
        },
        description="White bread composition (baked product)",
        domain="food",
        category="bakery",
        spectral_category="carbohydrates",
        variability={
            "moisture": (0.30, 0.45),
            "protein": (0.06, 0.11),
            "lipid": (0.01, 0.06),
            "sucrose": (0.00, 0.06),
        },
        correlations=[
            ("moisture", "starch", -0.6),
        ],
        tags=["bakery", "bread", "processed"],
        references=["USDA_FDC_2019", "Osborne_Fearn_Hindle_1993"],
    ),

    # =========================================================================
    # FOOD - Oils & Sweeteners
    # =========================================================================
    "olive_oil": AggregateComponent(
        name="olive_oil",
        components={
            "lipid": 0.996,
            "moisture": 0.002,
            "chlorophyll": 0.001,
            "carotenoid": 0.0005,
            "tannins": 0.0005,
        },
        description="Olive oil composition (fat-dominated; minor pigments/phenolics)",
        domain="food",
        category="oil",
        spectral_category="lipids",
        variability={
            "moisture": (0.0005, 0.010),
            "chlorophyll": (0.0002, 0.003),
            "carotenoid": (0.0001, 0.001),
        },
        tags=["oil", "lipid", "olive"],
        references=["USDA_FDC_2019"],
    ),

    "honey": AggregateComponent(
        name="honey",
        components={
            "water": 0.17,
            "fructose": 0.40,
            "glucose": 0.35,
            "sucrose": 0.03,
            "maltose": 0.04,
            "protein": 0.01,
        },
        description="Honey composition (major sugars + water)",
        domain="food",
        category="sweetener",
        spectral_category="carbohydrates",
        variability={
            "water": (0.14, 0.22),
            "fructose": (0.34, 0.45),
            "glucose": (0.28, 0.40),
        },
        correlations=[
            ("water", "glucose", -0.4),
        ],
        tags=["sweetener", "honey", "sugars"],
        references=["USDA_FDC_2019"],
    ),

    # =========================================================================
    # FOOD - Meat
    # =========================================================================
    "meat_beef": AggregateComponent(
        name="meat_beef",
        components={
            "water": 0.73,
            "protein": 0.22,
            "lipid": 0.03,
            "collagen": 0.01,
        },
        description="Lean beef composition",
        domain="food",
        category="meat",
        spectral_category="proteins",
        variability={
            "lipid": (0.01, 0.35),  # Very lean to heavily marbled
            "water": (0.50, 0.78),
            "protein": (0.15, 0.24),
        },
        correlations=[
            ("lipid", "water", -0.8),  # More fat = less water
            ("lipid", "protein", -0.6),
        ],
        tags=["meat", "beef", "protein_source"],
        references=["USDA_FDC_2019"],
    ),

    "meat_pork": AggregateComponent(
        name="meat_pork",
        components={
            "water": 0.72,
            "protein": 0.21,
            "lipid": 0.05,
            "collagen": 0.01,
        },
        description="Lean pork composition",
        domain="food",
        category="meat",
        spectral_category="proteins",
        variability={
            "lipid": (0.02, 0.30),
            "water": (0.55, 0.76),
            "protein": (0.16, 0.23),
        },
        correlations=[
            ("lipid", "water", -0.75),
        ],
        tags=["meat", "pork", "protein_source"],
        references=["USDA_FDC_2019"],
    ),

    "meat_chicken": AggregateComponent(
        name="meat_chicken",
        components={
            "water": 0.75,
            "protein": 0.20,
            "lipid": 0.03,
            "collagen": 0.01,
        },
        description="Chicken breast composition",
        domain="food",
        category="meat",
        spectral_category="proteins",
        variability={
            "lipid": (0.01, 0.15),
            "protein": (0.18, 0.25),
        },
        tags=["meat", "poultry", "protein_source"],
        references=["USDA_FDC_2019"],
    ),

    "fish_white": AggregateComponent(
        name="fish_white",
        components={
            "water": 0.80,
            "protein": 0.17,
            "lipid": 0.01,
            "collagen": 0.01,
        },
        description="White fish (cod-like) composition",
        domain="food",
        category="seafood",
        spectral_category="proteins",
        variability={
            "lipid": (0.005, 0.03),
            "protein": (0.15, 0.20),
        },
        tags=["seafood", "fish", "protein_source"],
        references=["USDA_FDC_2019"],
    ),

    "fish_oily": AggregateComponent(
        name="fish_oily",
        components={
            "water": 0.64,
            "protein": 0.18,
            "lipid": 0.16,
            "collagen": 0.02,
        },
        description="Oily fish (salmon-like) composition",
        domain="food",
        category="seafood",
        spectral_category="lipids",
        variability={
            "lipid": (0.08, 0.25),
            "water": (0.55, 0.72),
            "protein": (0.16, 0.22),
        },
        correlations=[
            ("lipid", "water", -0.8),
        ],
        tags=["seafood", "fish", "oily"],
        references=["USDA_FDC_2019"],
    ),

    # =========================================================================
    # FOOD - Fruits and Vegetables
    # =========================================================================
    "apple_fruit": AggregateComponent(
        name="apple_fruit",
        components={
            "water": 0.86,
            "fructose": 0.06,
            "glucose": 0.025,
            "sucrose": 0.02,
            "cellulose": 0.015,
            "malic_acid": 0.005,
        },
        description="Fresh apple fruit composition",
        domain="food",
        category="fruit",
        spectral_category="carbohydrates",
        variability={
            "fructose": (0.04, 0.08),
            "water": (0.82, 0.88),
            "malic_acid": (0.002, 0.012),
        },
        tags=["fruit", "apple", "produce"],
        references=["USDA_FDC_2019"],
    ),

    "tomato_fruit": AggregateComponent(
        name="tomato_fruit",
        components={
            "water": 0.94,
            "fructose": 0.015,
            "glucose": 0.012,
            "citric_acid": 0.005,
            "cellulose": 0.01,
            "lycopene": 0.0001,
        },
        description="Fresh tomato fruit composition",
        domain="food",
        category="vegetable",
        spectral_category="water_related",
        variability={
            "water": (0.92, 0.96),
            "lycopene": (0.00003, 0.0003),
        },
        tags=["vegetable", "tomato", "produce"],
        references=["USDA_FDC_2019"],
    ),

    "banana_fruit": AggregateComponent(
        name="banana_fruit",
        components={
            "water": 0.75,
            "starch": 0.10,
            "sucrose": 0.06,
            "glucose": 0.03,
            "fructose": 0.03,
            "cellulose": 0.02,
            "malic_acid": 0.01,
        },
        description="Fresh banana fruit composition (ripe; sugar/starch mix)",
        domain="food",
        category="fruit",
        spectral_category="carbohydrates",
        variability={
            "water": (0.70, 0.80),
            "starch": (0.00, 0.20),  # ripening strongly shifts starch -> sugars
            "sucrose": (0.03, 0.10),
        },
        correlations=[
            ("starch", "sucrose", -0.7),
        ],
        tags=["fruit", "banana", "produce"],
        references=["USDA_FDC_2019"],
    ),

    "potato_tuber": AggregateComponent(
        name="potato_tuber",
        components={
            "water": 0.79,
            "starch": 0.17,
            "protein": 0.02,
            "cellulose": 0.01,
            "sucrose": 0.01,
        },
        description="Potato tuber composition (fresh)",
        domain="food",
        category="vegetable",
        spectral_category="carbohydrates",
        variability={
            "water": (0.75, 0.83),
            "starch": (0.12, 0.22),
            "protein": (0.015, 0.03),
        },
        tags=["vegetable", "potato", "tuber"],
        references=["USDA_FDC_2019"],
    ),

    # =========================================================================
    # PHARMACEUTICAL
    # =========================================================================
    "tablet_excipient_base": AggregateComponent(
        name="tablet_excipient_base",
        components={
            "microcrystalline_cellulose": 0.40,
            "starch": 0.30,
            "lactose": 0.20,
            "moisture": 0.05,
        },
        description="Common tablet excipient mixture (without API)",
        domain="pharmaceutical",
        category="solid_dosage",
        spectral_category="carbohydrates",
        variability={
            "microcrystalline_cellulose": (0.30, 0.50),
            "starch": (0.20, 0.40),
            "lactose": (0.10, 0.30),
            "moisture": (0.02, 0.08),
        },
        tags=["pharma", "tablet", "excipient"],
        references=["Reich_2005_ADDR"],
    ),

    "tablet_paracetamol": AggregateComponent(
        name="tablet_paracetamol",
        components={
            "paracetamol": 0.50,
            "microcrystalline_cellulose": 0.25,
            "starch": 0.15,
            "moisture": 0.05,
        },
        description="Paracetamol tablet with excipients",
        domain="pharmaceutical",
        category="solid_dosage",
        spectral_category="pharmaceutical",
        variability={
            "paracetamol": (0.45, 0.55),  # ±10% API content
            "moisture": (0.03, 0.07),
        },
        tags=["pharma", "tablet", "paracetamol", "analgesic"],
        references=["Reich_2005_ADDR"],
    ),

    "tablet_aspirin": AggregateComponent(
        name="tablet_aspirin",
        components={
            "aspirin": 0.50,
            "starch": 0.30,
            "microcrystalline_cellulose": 0.15,
            "moisture": 0.03,
        },
        description="Aspirin tablet with excipients",
        domain="pharmaceutical",
        category="solid_dosage",
        spectral_category="pharmaceutical",
        variability={
            "aspirin": (0.45, 0.55),
            "moisture": (0.02, 0.05),
        },
        tags=["pharma", "tablet", "aspirin", "analgesic"],
        references=["Reich_2005_ADDR"],
    ),

    "capsule_ibuprofen": AggregateComponent(
        name="capsule_ibuprofen",
        components={
            "ibuprofen": 0.40,
            "starch": 0.30,
            "lactose": 0.20,
            "moisture": 0.05,
        },
        description="Ibuprofen capsule fill composition",
        domain="pharmaceutical",
        category="solid_dosage",
        spectral_category="pharmaceutical",
        variability={
            "ibuprofen": (0.35, 0.45),
            "moisture": (0.03, 0.07),
        },
        tags=["pharma", "capsule", "ibuprofen", "nsaid"],
        references=["Reich_2005_ADDR"],
    ),

    # ---------------------------------------------------------------------
    # PHARMACEUTICAL - Additional common dosage-form archetypes
    # ---------------------------------------------------------------------
    "tablet_generic_ir_low_dose": AggregateComponent(
        name="tablet_generic_ir_low_dose",
        components={
            # Low-dose potent API (1-5%) with typical excipients + traces.
            "api_generic": 0.02,
            "microcrystalline_cellulose": 0.45,
            "lactose": 0.30,
            "starch": 0.18,
            "moisture": 0.035,
            "lipid": 0.01,   # lubricant proxy (e.g., magnesium stearate)
            "silica": 0.005, # glidant proxy
        },
        description="Generic immediate-release tablet (low-dose API archetype)",
        domain="pharmaceutical",
        category="solid_dosage",
        spectral_category="carbohydrates",
        variability={
            "api_generic": (0.005, 0.05),
            "moisture": (0.02, 0.06),
            "microcrystalline_cellulose": (0.35, 0.55),
            "lactose": (0.15, 0.40),
            "starch": (0.10, 0.25),
        },
        tags=["pharma", "tablet", "archetype", "immediate_release", "low_dose"],
        references=["Reich_2005_ADDR"],
    ),

    "tablet_generic_ir_medium_dose": AggregateComponent(
        name="tablet_generic_ir_medium_dose",
        components={
            # Medium API load (~10-30%) is common across many IR tablets.
            "api_generic": 0.20,
            "microcrystalline_cellulose": 0.33,
            "lactose": 0.25,
            "starch": 0.15,
            "moisture": 0.055,
            "lipid": 0.01,
            "silica": 0.005,
        },
        description="Generic immediate-release tablet (medium-dose API archetype)",
        domain="pharmaceutical",
        category="solid_dosage",
        spectral_category="carbohydrates",
        variability={
            "api_generic": (0.10, 0.30),
            "moisture": (0.02, 0.08),
            "microcrystalline_cellulose": (0.20, 0.45),
            "lactose": (0.10, 0.35),
            "starch": (0.08, 0.22),
        },
        tags=["pharma", "tablet", "archetype", "immediate_release", "medium_dose"],
        references=["Reich_2005_ADDR"],
    ),

    "tablet_generic_ir_high_load": AggregateComponent(
        name="tablet_generic_ir_high_load",
        components={
            # High-load API tablets (50-80%) appear in e.g. some analgesics,
            # antidiabetics, vitamins; excipients reduced accordingly.
            "api_generic": 0.70,
            "microcrystalline_cellulose": 0.15,
            "starch": 0.08,
            "lactose": 0.03,
            "moisture": 0.025,
            "lipid": 0.01,
            "silica": 0.005,
        },
        description="Generic immediate-release tablet (high-load API archetype)",
        domain="pharmaceutical",
        category="solid_dosage",
        spectral_category="pharmaceutical",
        variability={
            "api_generic": (0.50, 0.80),
            "moisture": (0.01, 0.05),
            "microcrystalline_cellulose": (0.08, 0.25),
            "starch": (0.04, 0.15),
        },
        tags=["pharma", "tablet", "archetype", "immediate_release", "high_load"],
        references=["Reich_2005_ADDR"],
    ),

    "tablet_metformin": AggregateComponent(
        name="tablet_metformin",
        components={
            "metformin_hcl": 0.78,
            "microcrystalline_cellulose": 0.10,
            "starch": 0.07,
            "lactose": 0.02,
            "moisture": 0.015,
            "lipid": 0.01,
            "silica": 0.005,
        },
        description="High-load metformin HCl tablet archetype (API-dominant)",
        domain="pharmaceutical",
        category="solid_dosage",
        spectral_category="pharmaceutical",
        variability={
            "metformin_hcl": (0.70, 0.85),
            "moisture": (0.01, 0.04),
        },
        tags=["pharma", "tablet", "metformin", "high_load"],
        references=["Reich_2005_ADDR"],
    ),

    "tablet_vitamin_c": AggregateComponent(
        name="tablet_vitamin_c",
        components={
            "ascorbic_acid": 0.65,
            "microcrystalline_cellulose": 0.18,
            "starch": 0.10,
            "lactose": 0.03,
            "moisture": 0.03,
            "lipid": 0.01,
        },
        description="Vitamin C (ascorbic acid) tablet archetype",
        domain="pharmaceutical",
        category="solid_dosage",
        spectral_category="organic_acids",
        variability={
            "ascorbic_acid": (0.55, 0.75),
            "moisture": (0.01, 0.06),
        },
        tags=["pharma", "tablet", "vitamin", "ascorbic_acid"],
        references=["Reich_2005_ADDR"],
    ),

    "tablet_loratadine_low_dose": AggregateComponent(
        name="tablet_loratadine_low_dose",
        components={
            "loratadine": 0.015,
            "microcrystalline_cellulose": 0.47,
            "lactose": 0.30,
            "starch": 0.18,
            "moisture": 0.03,
            "lipid": 0.01,
            "silica": 0.005,
        },
        description="Low-dose antihistamine tablet archetype (loratadine-like)",
        domain="pharmaceutical",
        category="solid_dosage",
        spectral_category="carbohydrates",
        variability={
            "loratadine": (0.008, 0.03),
            "moisture": (0.02, 0.06),
        },
        tags=["pharma", "tablet", "antihistamine", "low_dose"],
        references=["Reich_2005_ADDR"],
    ),

    "capsule_generic_powder": AggregateComponent(
        name="capsule_generic_powder",
        components={
            "api_generic": 0.25,
            "lactose": 0.40,
            "starch": 0.25,
            "microcrystalline_cellulose": 0.05,
            "moisture": 0.045,
            "silica": 0.005,
        },
        description="Generic hard-capsule fill (powder blend archetype)",
        domain="pharmaceutical",
        category="solid_dosage",
        spectral_category="carbohydrates",
        variability={
            "api_generic": (0.10, 0.45),
            "moisture": (0.02, 0.08),
        },
        tags=["pharma", "capsule", "archetype", "powder"],
        references=["Reich_2005_ADDR"],
    ),

    "capsule_shell_gelatin": AggregateComponent(
        name="capsule_shell_gelatin",
        components={
            # Gelatin-based shell; represented via protein/collagen + water.
            "collagen": 0.84,
            "water": 0.15,
            "glycerol": 0.01,  # plasticizer proxy (more relevant for softgels)
        },
        description="Gelatin capsule shell archetype (protein + water + plasticizer)",
        domain="pharmaceutical",
        category="packaging",
        spectral_category="proteins",
        variability={
            "water": (0.10, 0.18),
            "glycerol": (0.0, 0.03),
        },
        tags=["pharma", "capsule", "shell", "gelatin"],
        references=["Reich_2005_ADDR"],
    ),

    "tablet_amoxicillin": AggregateComponent(
        name="tablet_amoxicillin",
        components={
            "amoxicillin_trihydrate": 0.55,
            "microcrystalline_cellulose": 0.18,
            "starch": 0.15,
            "lactose": 0.08,
            "moisture": 0.03,
            "silica": 0.005,
            "lipid": 0.005,
        },
        description="Antibiotic tablet archetype (amoxicillin-like; medium/high API load)",
        domain="pharmaceutical",
        category="solid_dosage",
        spectral_category="pharmaceutical",
        variability={
            "amoxicillin_trihydrate": (0.45, 0.70),
            "moisture": (0.01, 0.08),
        },
        tags=["pharma", "tablet", "antibiotic", "amoxicillin"],
        references=["Reich_2005_ADDR"],
    ),

    "tablet_effervescent_archetype": AggregateComponent(
        name="tablet_effervescent_archetype",
        components={
            # Simplified effervescent archetype: organic acid + carbohydrate/filler.
            # (Base/carbonate not explicitly modeled here.)
            "citric_acid": 0.22,
            "lactose": 0.25,
            "sucrose": 0.35,
            "starch": 0.12,
            "moisture": 0.06,
        },
        description="Effervescent tablet archetype (acidulated carbohydrate base)",
        domain="pharmaceutical",
        category="solid_dosage",
        spectral_category="organic_acids",
        variability={
            "citric_acid": (0.10, 0.30),
            "moisture": (0.02, 0.10),
        },
        tags=["pharma", "tablet", "effervescent", "archetype"],
        references=["Reich_2005_ADDR"],
    ),

    # ---------------------------------------------------------------------
    # CHEMISTRY - Solvents, solutions, and common NIR lab materials
    # ---------------------------------------------------------------------
    "solvent_ethanol_water_70": AggregateComponent(
        name="solvent_ethanol_water_70",
        components={
            "ethanol": 0.70,
            "water": 0.30,
        },
        description="Ethanol-water mixture (70% v/v archetype; sanitizing/solvent)",
        domain="chemistry",
        category="solvent",
        spectral_category="alcohols",
        variability={
            "ethanol": (0.60, 0.80),
        },
        tags=["chemistry", "solvent", "ethanol", "aqueous"],
        references=["Osborne_Fearn_Hindle_1993"],
    ),

    "solvent_ethanol_water_40": AggregateComponent(
        name="solvent_ethanol_water_40",
        components={
            "ethanol": 0.40,
            "water": 0.60,
        },
        description="Ethanol-water mixture (40% v/v archetype; beverage/solvent range)",
        domain="chemistry",
        category="solvent",
        spectral_category="alcohols",
        variability={
            "ethanol": (0.30, 0.55),
        },
        tags=["chemistry", "solvent", "ethanol", "aqueous"],
        references=["Osborne_Fearn_Hindle_1993"],
    ),

    "solution_glycerol_water_50": AggregateComponent(
        name="solution_glycerol_water_50",
        components={
            "glycerol": 0.50,
            "water": 0.50,
        },
        description="Glycerol-water solution (humectant / viscosity standard archetype)",
        domain="chemistry",
        category="solution",
        spectral_category="alcohols",
        variability={
            "glycerol": (0.30, 0.70),
        },
        tags=["chemistry", "solution", "glycerol", "aqueous"],
        references=["Osborne_Fearn_Hindle_1993"],
    ),

    "solution_sucrose_water": AggregateComponent(
        name="solution_sucrose_water",
        components={
            "water": 0.35,
            "sucrose": 0.63,
            "citric_acid": 0.01,
            "moisture": 0.01,
        },
        description="Concentrated sucrose solution archetype (syrup-like; acidulated)",
        domain="chemistry",
        category="solution",
        spectral_category="carbohydrates",
        variability={
            "sucrose": (0.50, 0.70),
            "citric_acid": (0.0, 0.02),
        },
        tags=["chemistry", "solution", "sucrose", "aqueous", "syrup"],
        references=["Osborne_Fearn_Hindle_1993"],
    ),

    "powder_microcrystalline_cellulose": AggregateComponent(
        name="powder_microcrystalline_cellulose",
        components={
            "microcrystalline_cellulose": 0.95,
            "moisture": 0.05,
        },
        description="Microcrystalline cellulose powder archetype (common excipient/lab material)",
        domain="chemistry",
        category="powder",
        spectral_category="carbohydrates",
        variability={
            "moisture": (0.01, 0.10),
        },
        tags=["chemistry", "powder", "cellulose", "excipient"],
        references=["Osborne_Fearn_Hindle_1993", "Reich_2005_ADDR"],
    ),

    "powder_lactose": AggregateComponent(
        name="powder_lactose",
        components={
            "lactose": 0.98,
            "moisture": 0.02,
        },
        description="Lactose powder archetype (common excipient)",
        domain="chemistry",
        category="powder",
        spectral_category="carbohydrates",
        variability={
            "moisture": (0.005, 0.06),
        },
        tags=["chemistry", "powder", "lactose", "excipient"],
        references=["Reich_2005_ADDR"],
    ),

    "powder_starch": AggregateComponent(
        name="powder_starch",
        components={
            "starch": 0.98,
            "moisture": 0.02,
        },
        description="Starch powder archetype (binder/disintegrant; common carbohydrate)",
        domain="chemistry",
        category="powder",
        spectral_category="carbohydrates",
        variability={
            "moisture": (0.005, 0.08),
        },
        tags=["chemistry", "powder", "starch", "excipient"],
        references=["Osborne_Fearn_Hindle_1993", "Reich_2005_ADDR"],
    ),

    "powder_silica": AggregateComponent(
        name="powder_silica",
        components={
            "silica": 0.98,
            "moisture": 0.02,
        },
        description="Silica powder archetype (mineral; glidant/filler proxy)",
        domain="chemistry",
        category="powder",
        spectral_category="minerals",
        variability={
            "moisture": (0.0, 0.06),
        },
        tags=["chemistry", "powder", "silica", "mineral"],
        references=["Osborne_Fearn_Hindle_1993"],
    ),

    "powder_kaolin": AggregateComponent(
        name="powder_kaolin",
        components={
            "kaolinite": 0.90,
            "silica": 0.08,
            "moisture": 0.02,
        },
        description="Kaolin/kaolinite-rich mineral powder archetype",
        domain="chemistry",
        category="powder",
        spectral_category="minerals",
        variability={
            "moisture": (0.0, 0.06),
            "kaolinite": (0.75, 0.98),
        },
        tags=["chemistry", "powder", "mineral", "kaolinite", "clay"],
        references=["Osborne_Fearn_Hindle_1993", "Khayamim_2015_JNIRS"],
    ),

    "powder_montmorillonite": AggregateComponent(
        name="powder_montmorillonite",
        components={
            "montmorillonite": 0.88,
            "silica": 0.08,
            "moisture": 0.04,
        },
        description="Montmorillonite-rich clay mineral powder archetype",
        domain="chemistry",
        category="powder",
        spectral_category="minerals",
        variability={
            "moisture": (0.0, 0.10),
            "montmorillonite": (0.70, 0.95),
        },
        tags=["chemistry", "powder", "mineral", "montmorillonite", "clay"],
        references=["Osborne_Fearn_Hindle_1993", "Khayamim_2015_JNIRS"],
    ),

    # =========================================================================
    # ENVIRONMENTAL - Soils
    # =========================================================================
    "soil_agricultural": AggregateComponent(
        name="soil_agricultural",
        components={
            "silica": 0.60,
            "kaolinite": 0.15,
            "moisture": 0.15,
            "cellulose": 0.05,  # Organic matter proxy
            "protein": 0.02,    # Organic nitrogen proxy
        },
        description="Typical agricultural topsoil",
        domain="environmental",
        category="soil",
        spectral_category="minerals",
        variability={
            "moisture": (0.05, 0.30),
            "cellulose": (0.01, 0.10),  # Organic matter variation
            "kaolinite": (0.05, 0.25),  # Clay content
        },
        tags=["soil", "environmental", "agriculture"],
        references=["Khayamim_2015_JNIRS"],
    ),

    "soil_clay": AggregateComponent(
        name="soil_clay",
        components={
            "silica": 0.40,
            "kaolinite": 0.30,
            "montmorillonite": 0.10,
            "moisture": 0.15,
            "cellulose": 0.03,
        },
        description="Clay-rich soil composition",
        domain="environmental",
        category="soil",
        spectral_category="minerals",
        variability={
            "moisture": (0.10, 0.35),
            "kaolinite": (0.20, 0.40),
        },
        tags=["soil", "environmental", "clay"],
        references=["Khayamim_2015_JNIRS"],
    ),

    "soil_sandy": AggregateComponent(
        name="soil_sandy",
        components={
            "silica": 0.75,
            "kaolinite": 0.05,
            "montmorillonite": 0.03,
            "moisture": 0.10,
            "cellulose": 0.05,  # organic matter proxy
            "protein": 0.01,    # organic nitrogen proxy
        },
        description="Sandy soil (high quartz/silica; low clay)",
        domain="environmental",
        category="soil",
        spectral_category="minerals",
        variability={
            "moisture": (0.02, 0.25),
            "silica": (0.60, 0.88),
            "cellulose": (0.00, 0.08),
        },
        tags=["soil", "environmental", "sandy"],
        references=["Khayamim_2015_JNIRS"],
    ),

    "soil_organic": AggregateComponent(
        name="soil_organic",
        components={
            "silica": 0.45,
            "kaolinite": 0.10,
            "montmorillonite": 0.03,
            "moisture": 0.20,
            "cellulose": 0.15,
            "protein": 0.05,
        },
        description="Organic-rich topsoil (higher organic matter proxies)",
        domain="environmental",
        category="soil",
        spectral_category="minerals",
        variability={
            "moisture": (0.05, 0.40),
            "cellulose": (0.05, 0.25),
            "protein": (0.02, 0.10),
        },
        tags=["soil", "environmental", "organic"],
        references=["Khayamim_2015_JNIRS"],
    ),

    # =========================================================================
    # POLYMERS
    # =========================================================================
    "plastic_pet": AggregateComponent(
        name="plastic_pet",
        components={
            "pet": 0.95,
            "moisture": 0.003,
        },
        description="PET (polyethylene terephthalate) plastic",
        domain="industrial",
        category="polymer",
        spectral_category="polymers",
        variability={
            "moisture": (0.001, 0.01),
        },
        tags=["polymer", "plastic", "pet", "recycling"],
        references=["Lachenal_1995_VibSpect"],
    ),

    "plastic_pe": AggregateComponent(
        name="plastic_pe",
        components={
            "polyethylene": 0.98,
            "moisture": 0.001,
        },
        description="Polyethylene plastic",
        domain="industrial",
        category="polymer",
        spectral_category="polymers",
        variability={
            "moisture": (0.0005, 0.005),
        },
        tags=["polymer", "plastic", "pe", "recycling"],
        references=["Lachenal_1995_VibSpect"],
    ),

    # =========================================================================
    # INDUSTRIAL - Lignocellulosic materials
    # =========================================================================
    "wood_softwood": AggregateComponent(
        name="wood_softwood",
        components={
            "cellulose": 0.45,
            "hemicellulose": 0.25,
            "lignin": 0.25,
            "moisture": 0.05,
        },
        description="Softwood composition (cellulose/hemicellulose/lignin)",
        domain="industrial",
        category="lignocellulosic",
        spectral_category="carbohydrates",
        variability={
            "moisture": (0.03, 0.15),
            "lignin": (0.20, 0.32),
        },
        tags=["wood", "cellulose", "lignin"],
        references=["Osborne_Fearn_Hindle_1993"],
    ),

    "paper_pulp": AggregateComponent(
        name="paper_pulp",
        components={
            "cellulose": 0.80,
            "hemicellulose": 0.08,
            "lignin": 0.05,
            "silica": 0.03,
            "moisture": 0.04,
        },
        description="Paper/pulp composition (cellulose-dominant with fillers)",
        domain="industrial",
        category="lignocellulosic",
        spectral_category="carbohydrates",
        variability={
            "moisture": (0.02, 0.10),
            "silica": (0.00, 0.08),
            "lignin": (0.00, 0.10),
        },
        tags=["paper", "pulp", "cellulose"],
        references=["Osborne_Fearn_Hindle_1993"],
    ),

    # =========================================================================
    # BEVERAGES
    # =========================================================================
    "wine_red": AggregateComponent(
        name="wine_red",
        components={
            "water": 0.85,
            "ethanol": 0.12,
            "glucose": 0.002,
            "fructose": 0.002,
            "tartaric_acid": 0.003,
            "malic_acid": 0.002,
            "glycerol": 0.008,
            "anthocyanin": 0.0003,
            "tannins": 0.002,
        },
        description="Dry red wine composition",
        domain="food",
        category="beverage",
        spectral_category="alcohols",
        variability={
            "ethanol": (0.10, 0.15),
            "anthocyanin": (0.0001, 0.0005),
            "tannins": (0.001, 0.004),
        },
        tags=["beverage", "wine", "fermented", "alcohol"],
        references=["MarteloVidal_2014_FoodChem"],
    ),

    "beer": AggregateComponent(
        name="beer",
        components={
            "water": 0.92,
            "ethanol": 0.05,
            "maltose": 0.01,
            "glucose": 0.005,
            "protein": 0.005,
            "glycerol": 0.002,
        },
        description="Typical beer composition",
        domain="food",
        category="beverage",
        spectral_category="alcohols",
        variability={
            "ethanol": (0.03, 0.10),
            "maltose": (0.005, 0.02),
        },
        tags=["beverage", "beer", "fermented", "alcohol"],
        references=["USDA_FDC_2019"],
    ),
}


# =============================================================================
# API Functions
# =============================================================================


def get_aggregate(name: str) -> AggregateComponent:
    """
    Get an aggregate component definition by name.

    Args:
        name: Aggregate name (e.g., "wheat_grain", "milk", "tablet_excipient_base").

    Returns:
        AggregateComponent object.

    Raises:
        ValueError: If aggregate name is not found.

    Example:
        >>> wheat = get_aggregate("wheat_grain")
        >>> print(wheat.description)
        Typical wheat grain composition
    """
    if name not in AGGREGATE_COMPONENTS:
        available = list(AGGREGATE_COMPONENTS.keys())
        raise ValueError(f"Unknown aggregate: '{name}'. Available: {available}")
    return AGGREGATE_COMPONENTS[name]


def list_aggregates(
    domain: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[str]:
    """
    List available aggregate components with optional filtering.

    Args:
        domain: Filter by domain (e.g., "agriculture", "food", "pharmaceutical").
        category: Filter by category (e.g., "grain", "dairy", "solid_dosage").
        tags: Filter by tags (any match).

    Returns:
        Sorted list of aggregate names matching the criteria.

    Example:
        >>> # List all aggregates
        >>> all_aggs = list_aggregates()
        >>>
        >>> # List food aggregates
        >>> food_aggs = list_aggregates(domain="food")
        >>>
        >>> # List grain aggregates
        >>> grain_aggs = list_aggregates(category="grain")
    """
    results = []

    for name, agg in AGGREGATE_COMPONENTS.items():
        if domain and agg.domain != domain:
            continue
        if category and agg.category != category:
            continue
        if tags:
            if not any(t in agg.tags for t in tags):
                continue
        results.append(name)

    return sorted(results)


def expand_aggregate(
    name: str,
    variability: bool = False,
    random_state: Optional[int] = None,
    renormalize: bool = True,
) -> Dict[str, float]:
    """
    Expand an aggregate into component weights.

    Args:
        name: Aggregate name.
        variability: If True, sample from variability ranges instead of using
            fixed base composition.
        random_state: Random seed for variability sampling.
        renormalize: If True, normalize weights to sum to 1.0.

    Returns:
        Dictionary of {component_name: weight}.

    Example:
        >>> # Get fixed composition
        >>> comp = expand_aggregate("wheat_grain")
        >>> print(comp["protein"])
        0.12
        >>>
        >>> # Sample with variability for training data
        >>> comp = expand_aggregate("wheat_grain", variability=True, random_state=42)
        >>> print(comp["protein"])  # Will vary between 0.08 and 0.18
    """
    agg = get_aggregate(name)

    if not variability:
        result = agg.components.copy()
    else:
        rng = np.random.default_rng(random_state)
        result = {}

        # Sample variable components
        sampled = {}
        for comp_name, (low, high) in agg.variability.items():
            sampled[comp_name] = rng.uniform(low, high)

        # Apply correlations (simplified: adjust correlated component proportionally)
        for comp1, comp2, corr in agg.correlations:
            if comp1 in sampled and comp2 in agg.variability:
                # Adjust comp2 based on comp1's deviation from mean
                base1 = agg.components.get(comp1, 0)
                var1_range = agg.variability.get(comp1, (base1, base1))
                var2_range = agg.variability.get(comp2, (0, 0))

                if var1_range[1] > var1_range[0] and var2_range[1] > var2_range[0]:
                    # Normalized deviation of comp1 from center
                    center1 = (var1_range[0] + var1_range[1]) / 2
                    range1 = var1_range[1] - var1_range[0]
                    dev1 = (sampled[comp1] - center1) / (range1 / 2)

                    # Apply correlated adjustment to comp2
                    center2 = (var2_range[0] + var2_range[1]) / 2
                    range2 = var2_range[1] - var2_range[0]

                    # Correlated adjustment (with some random noise)
                    adjustment = corr * dev1 * (range2 / 2)
                    noise = rng.normal(0, range2 * 0.1)  # Small random noise
                    new_val = center2 + adjustment + noise
                    sampled[comp2] = np.clip(new_val, var2_range[0], var2_range[1])

        # Build result: use sampled values where available, base otherwise
        for comp_name, base_weight in agg.components.items():
            if comp_name in sampled:
                result[comp_name] = sampled[comp_name]
            else:
                # Add small variation to fixed components
                noise = rng.normal(0, base_weight * 0.02)
                result[comp_name] = max(0, base_weight + noise)

    # Renormalize if requested
    if renormalize:
        total = sum(result.values())
        if total > 0 and abs(total - 1.0) > 0.01:
            result = {k: v / total for k, v in result.items()}

    return result


def aggregate_info(name: str) -> str:
    """
    Return formatted information about an aggregate.

    Args:
        name: Aggregate name.

    Returns:
        Human-readable string with aggregate details.

    Example:
        >>> print(aggregate_info("wheat_grain"))
    """
    agg = get_aggregate(name)
    return agg.info()


def list_domains() -> List[str]:
    """
    List all unique domains across aggregates.

    Returns:
        Sorted list of domain names.

    Example:
        >>> domains = list_domains()
        >>> print(domains)
        ['agriculture', 'environmental', 'food', 'industrial', 'pharmaceutical']
    """
    domains = set()
    for agg in AGGREGATE_COMPONENTS.values():
        domains.add(agg.domain)
    return sorted(domains)


def list_categories(domain: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List categories and their aggregates, optionally filtered by domain.

    Args:
        domain: Optional domain filter.

    Returns:
        Dictionary of {category: [aggregate_names]}.

    Example:
        >>> cats = list_categories(domain="food")
        >>> for cat, aggs in cats.items():
        ...     print(f"{cat}: {aggs}")
    """
    categories: Dict[str, List[str]] = {}

    for name, agg in AGGREGATE_COMPONENTS.items():
        if domain and agg.domain != domain:
            continue
        cat = agg.category or "uncategorized"
        categories.setdefault(cat, []).append(name)

    # Sort aggregates within each category
    for cat in categories:
        categories[cat].sort()

    return categories


def validate_aggregates() -> List[str]:
    """
    Validate all predefined aggregates.

    Returns:
        List of validation issues (empty if all valid).

    Example:
        >>> issues = validate_aggregates()
        >>> if issues:
        ...     for issue in issues:
        ...         print(f"Warning: {issue}")
    """
    issues = []

    for name, agg in AGGREGATE_COMPONENTS.items():
        agg_issues = agg.validate()
        for issue in agg_issues:
            issues.append(f"Aggregate '{name}': {issue}")

    return issues
