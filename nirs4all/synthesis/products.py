"""
Product-level synthetic NIRS generator for neural network training.

This module provides high-level APIs to generate diverse, realistic product
samples with controlled variability for training neural networks. Unlike
the base SyntheticNIRSGenerator which operates at the component level,
ProductGenerator works with predefined product templates that include
realistic composition variability, component correlations, and bounds.

Key Features:
    - Predefined product templates with realistic composition ranges
    - Controlled variability types (FIXED, UNIFORM, NORMAL, LOGNORMAL, CORRELATED)
    - Composition constraints (sum to 1.0, realistic bounds)
    - Correlation preservation between components
    - Target flexibility (any component as regression target)
    - Efficient batch generation for NN training (10k-100k samples)
    - Integration with custom wavelength grids

Example:
    >>> from nirs4all.synthesis import ProductGenerator, list_product_templates
    >>>
    >>> # List available templates
    >>> print(list_product_templates(category="dairy"))
    ['milk_variable_fat', 'cheese_variable_moisture']
    >>>
    >>> # Generate dairy product samples
    >>> generator = ProductGenerator("milk_variable_fat")
    >>> dataset = generator.generate(n_samples=10000, target="fat")
    >>>
    >>> # High-variability dataset for NN training
    >>> generator = ProductGenerator("food_cholesterol_variable")
    >>> dataset = generator.generate(n_samples=50000, target="cholesterol")

References:
    [1] USDA FoodData Central (https://fdc.nal.usda.gov/)
    [2] Osborne, B. G., Fearn, T., & Hindle, P. H. (1993). Practical NIR Spectroscopy.
    [3] Williams, P. (2001). Implementation of Near-Infrared Technology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .components import ComponentLibrary

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset


class VariationType(Enum):
    """
    Type of variation for component concentrations.

    Attributes:
        FIXED: No variation, use exact specified value.
        UNIFORM: Uniform distribution between min and max.
        NORMAL: Normal (Gaussian) distribution with mean and std.
        LOGNORMAL: Log-normal distribution for non-negative values.
        CORRELATED: Value derived from correlation with another component.
        COMPUTED: Value computed from other components (e.g., 1 - sum(others)).
    """

    FIXED = auto()
    UNIFORM = auto()
    NORMAL = auto()
    LOGNORMAL = auto()
    CORRELATED = auto()
    COMPUTED = auto()


@dataclass
class ComponentVariation:
    """
    Specification for how a component's concentration varies.

    Attributes:
        component: Name of the spectral component (must exist in library).
        variation_type: Type of variation (FIXED, UNIFORM, NORMAL, etc.).
        value: For FIXED type, the exact value.
        min_value: For UNIFORM/NORMAL, the minimum bound.
        max_value: For UNIFORM/NORMAL, the maximum bound.
        mean: For NORMAL/LOGNORMAL, the distribution mean.
        std: For NORMAL/LOGNORMAL, the distribution standard deviation.
        correlated_with: For CORRELATED, the source component name.
        correlation: For CORRELATED, the correlation coefficient.
        compute_as: For COMPUTED, a string describing the computation
            (currently supports "remainder" for 1 - sum(others)).

    Example:
        >>> # Fixed moisture content
        >>> moisture = ComponentVariation("moisture", VariationType.FIXED, value=0.12)
        >>>
        >>> # Variable protein with uniform distribution
        >>> protein = ComponentVariation(
        ...     "protein", VariationType.UNIFORM,
        ...     min_value=0.08, max_value=0.18
        ... )
        >>>
        >>> # Starch negatively correlated with protein
        >>> starch = ComponentVariation(
        ...     "starch", VariationType.CORRELATED,
        ...     correlated_with="protein", correlation=-0.85,
        ...     min_value=0.55, max_value=0.72
        ... )
    """

    component: str
    variation_type: VariationType
    value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    correlated_with: Optional[str] = None
    correlation: Optional[float] = None
    compute_as: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate specification based on variation type."""
        vtype = self.variation_type

        if vtype == VariationType.FIXED:
            if self.value is None:
                raise ValueError("FIXED variation requires 'value'")

        elif vtype == VariationType.UNIFORM:
            if self.min_value is None or self.max_value is None:
                raise ValueError("UNIFORM variation requires 'min_value' and 'max_value'")
            if self.min_value > self.max_value:
                raise ValueError("min_value must be <= max_value")

        elif vtype == VariationType.NORMAL:
            if self.mean is None or self.std is None:
                raise ValueError("NORMAL variation requires 'mean' and 'std'")
            if self.std < 0:
                raise ValueError("std must be non-negative")

        elif vtype == VariationType.LOGNORMAL:
            if self.mean is None or self.std is None:
                raise ValueError("LOGNORMAL variation requires 'mean' and 'std'")

        elif vtype == VariationType.CORRELATED:
            if self.correlated_with is None or self.correlation is None:
                raise ValueError("CORRELATED variation requires 'correlated_with' and 'correlation'")
            if not -1.0 <= self.correlation <= 1.0:
                raise ValueError("correlation must be between -1 and 1")

        elif vtype == VariationType.COMPUTED:
            if self.compute_as is None:
                raise ValueError("COMPUTED variation requires 'compute_as'")


@dataclass
class ProductTemplate:
    """
    Template defining a product type with composition variability.

    A ProductTemplate describes a realistic product type (e.g., wheat grain,
    milk, pharmaceutical tablet) along with specifications for how each
    component's concentration can vary. This enables generation of diverse
    samples suitable for neural network training.

    Attributes:
        name: Unique identifier for the template.
        description: Human-readable description.
        category: Product category (e.g., "dairy", "grain", "pharma").
        domain: Application domain (e.g., "agriculture", "food", "pharmaceutical").
        components: List of ComponentVariation specifications.
        default_target: Default component to use as regression target.
        tags: Classification tags for filtering.
        references: Literature or data source citations.

    Example:
        >>> milk_template = ProductTemplate(
        ...     name="milk_variable_fat",
        ...     description="Milk with variable fat content (skim to whole)",
        ...     category="dairy",
        ...     domain="food",
        ...     components=[
        ...         ComponentVariation("water", VariationType.COMPUTED, compute_as="remainder"),
        ...         ComponentVariation("lipid", VariationType.UNIFORM, min_value=0.005, max_value=0.06),
        ...         ComponentVariation("casein", VariationType.NORMAL, mean=0.028, std=0.003),
        ...         ComponentVariation("whey", VariationType.FIXED, value=0.006),
        ...         ComponentVariation("lactose", VariationType.NORMAL, mean=0.05, std=0.003),
        ...     ],
        ...     default_target="lipid",
        ... )
    """

    name: str
    description: str
    category: str
    domain: str
    components: List[ComponentVariation]
    default_target: str = ""
    tags: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate template consistency."""
        comp_names = [c.component for c in self.components]

        # Check for duplicates
        if len(comp_names) != len(set(comp_names)):
            raise ValueError(f"Duplicate components in template '{self.name}'")

        # Check correlated components reference valid sources
        for comp_var in self.components:
            if comp_var.variation_type == VariationType.CORRELATED:
                if comp_var.correlated_with not in comp_names:
                    raise ValueError(
                        f"Component '{comp_var.component}' correlates with "
                        f"'{comp_var.correlated_with}' which is not in template"
                    )

    @property
    def component_names(self) -> List[str]:
        """Return list of component names in this template."""
        return [c.component for c in self.components]

    def info(self) -> str:
        """Return formatted information about the template."""
        lines = [
            f"ProductTemplate: {self.name}",
            f"Description: {self.description}",
            f"Category: {self.category}",
            f"Domain: {self.domain}",
            f"Default Target: {self.default_target or 'N/A'}",
            f"Components ({len(self.components)}):",
        ]
        for comp in self.components:
            vtype = comp.variation_type.name
            if comp.variation_type == VariationType.FIXED:
                detail = f"= {comp.value:.3f}"
            elif comp.variation_type == VariationType.UNIFORM:
                detail = f"~ U({comp.min_value:.3f}, {comp.max_value:.3f})"
            elif comp.variation_type == VariationType.NORMAL:
                detail = f"~ N({comp.mean:.3f}, {comp.std:.3f})"
            elif comp.variation_type == VariationType.LOGNORMAL:
                detail = f"~ LogN({comp.mean:.3f}, {comp.std:.3f})"
            elif comp.variation_type == VariationType.CORRELATED:
                detail = f"corr={comp.correlation:.2f} with {comp.correlated_with}"
            elif comp.variation_type == VariationType.COMPUTED:
                detail = f"computed as {comp.compute_as}"
            else:
                detail = ""
            lines.append(f"  - {comp.component}: {vtype} {detail}")
        if self.tags:
            lines.append(f"Tags: {', '.join(self.tags)}")
        return "\n".join(lines)


# =============================================================================
# Predefined Product Templates
# =============================================================================

PRODUCT_TEMPLATES: Dict[str, ProductTemplate] = {}


def _register_templates() -> None:
    """Register all predefined product templates."""
    global PRODUCT_TEMPLATES

    # =========================================================================
    # DAIRY
    # =========================================================================
    PRODUCT_TEMPLATES["milk_variable_fat"] = ProductTemplate(
        name="milk_variable_fat",
        description="Milk with variable fat content (skim to whole)",
        category="dairy",
        domain="food",
        components=[
            ComponentVariation("lipid", VariationType.UNIFORM, min_value=0.005, max_value=0.06),
            ComponentVariation("casein", VariationType.NORMAL, mean=0.028, std=0.003, min_value=0.022, max_value=0.034),
            ComponentVariation("whey", VariationType.NORMAL, mean=0.006, std=0.001, min_value=0.004, max_value=0.008),
            ComponentVariation("lactose", VariationType.NORMAL, mean=0.05, std=0.003, min_value=0.045, max_value=0.055),
            ComponentVariation("water", VariationType.COMPUTED, compute_as="remainder"),
        ],
        default_target="lipid",
        tags=["dairy", "milk", "liquid", "nn_training"],
        references=["USDA FoodData Central"],
    )

    PRODUCT_TEMPLATES["cheese_variable_moisture"] = ProductTemplate(
        name="cheese_variable_moisture",
        description="Cheese with variable moisture (aged to fresh)",
        category="dairy",
        domain="food",
        components=[
            ComponentVariation("moisture", VariationType.UNIFORM, min_value=0.28, max_value=0.45),
            ComponentVariation("lipid", VariationType.CORRELATED, correlated_with="moisture", correlation=-0.6,
                               min_value=0.25, max_value=0.40, mean=0.33, std=0.04),
            ComponentVariation("casein", VariationType.CORRELATED, correlated_with="moisture", correlation=-0.5,
                               min_value=0.18, max_value=0.32, mean=0.25, std=0.03),
            ComponentVariation("lactose", VariationType.NORMAL, mean=0.02, std=0.01, min_value=0.005, max_value=0.035),
        ],
        default_target="moisture",
        tags=["dairy", "cheese", "solid", "nn_training"],
        references=["USDA FoodData Central"],
    )

    PRODUCT_TEMPLATES["yogurt_variable_fat"] = ProductTemplate(
        name="yogurt_variable_fat",
        description="Yogurt with variable fat content (non-fat to full-fat)",
        category="dairy",
        domain="food",
        components=[
            ComponentVariation("lipid", VariationType.UNIFORM, min_value=0.001, max_value=0.10),
            ComponentVariation("casein", VariationType.NORMAL, mean=0.035, std=0.005, min_value=0.025, max_value=0.045),
            ComponentVariation("whey", VariationType.NORMAL, mean=0.015, std=0.003, min_value=0.010, max_value=0.020),
            ComponentVariation("lactose", VariationType.NORMAL, mean=0.04, std=0.005, min_value=0.030, max_value=0.050),
            ComponentVariation("lactic_acid", VariationType.UNIFORM, min_value=0.005, max_value=0.015),
            ComponentVariation("water", VariationType.COMPUTED, compute_as="remainder"),
        ],
        default_target="lipid",
        tags=["dairy", "yogurt", "fermented", "nn_training"],
        references=["USDA FoodData Central"],
    )

    # =========================================================================
    # MEAT
    # =========================================================================
    PRODUCT_TEMPLATES["meat_variable_fat"] = ProductTemplate(
        name="meat_variable_fat",
        description="Meat with variable fat content (very lean to marbled)",
        category="meat",
        domain="food",
        components=[
            ComponentVariation("lipid", VariationType.UNIFORM, min_value=0.01, max_value=0.35),
            ComponentVariation("protein", VariationType.CORRELATED, correlated_with="lipid", correlation=-0.6,
                               min_value=0.14, max_value=0.24, mean=0.20, std=0.03),
            ComponentVariation("water", VariationType.CORRELATED, correlated_with="lipid", correlation=-0.8,
                               min_value=0.45, max_value=0.78, mean=0.70, std=0.08),
            ComponentVariation("collagen", VariationType.NORMAL, mean=0.015, std=0.005, min_value=0.005, max_value=0.030),
        ],
        default_target="lipid",
        tags=["meat", "protein_source", "nn_training"],
        references=["USDA FoodData Central"],
    )

    PRODUCT_TEMPLATES["meat_variable_protein"] = ProductTemplate(
        name="meat_variable_protein",
        description="Meat with variable protein content across cuts",
        category="meat",
        domain="food",
        components=[
            ComponentVariation("protein", VariationType.UNIFORM, min_value=0.14, max_value=0.28),
            ComponentVariation("lipid", VariationType.CORRELATED, correlated_with="protein", correlation=-0.5,
                               min_value=0.02, max_value=0.25, mean=0.10, std=0.06),
            ComponentVariation("water", VariationType.COMPUTED, compute_as="remainder"),
            ComponentVariation("collagen", VariationType.NORMAL, mean=0.012, std=0.004, min_value=0.005, max_value=0.025),
        ],
        default_target="protein",
        tags=["meat", "protein_source", "nn_training"],
        references=["USDA FoodData Central"],
    )

    # =========================================================================
    # GRAIN
    # =========================================================================
    PRODUCT_TEMPLATES["wheat_variable_protein"] = ProductTemplate(
        name="wheat_variable_protein",
        description="Wheat grain with variable protein (feed to bread wheat)",
        category="grain",
        domain="agriculture",
        components=[
            ComponentVariation("protein", VariationType.UNIFORM, min_value=0.08, max_value=0.18),
            ComponentVariation("starch", VariationType.CORRELATED, correlated_with="protein", correlation=-0.85,
                               min_value=0.55, max_value=0.72, mean=0.65, std=0.04),
            ComponentVariation("moisture", VariationType.NORMAL, mean=0.12, std=0.02, min_value=0.08, max_value=0.16),
            ComponentVariation("lipid", VariationType.NORMAL, mean=0.02, std=0.005, min_value=0.01, max_value=0.03),
            ComponentVariation("cellulose", VariationType.NORMAL, mean=0.08, std=0.015, min_value=0.05, max_value=0.12),
        ],
        default_target="protein",
        tags=["grain", "cereal", "agriculture", "nn_training"],
        references=["USDA FoodData Central", "Osborne1993"],
    )

    PRODUCT_TEMPLATES["corn_grain"] = ProductTemplate(
        name="corn_grain",
        description="Corn/maize grain with typical composition variation",
        category="grain",
        domain="agriculture",
        components=[
            ComponentVariation("starch", VariationType.NORMAL, mean=0.72, std=0.03, min_value=0.65, max_value=0.78),
            ComponentVariation("protein", VariationType.UNIFORM, min_value=0.06, max_value=0.12),
            ComponentVariation("moisture", VariationType.NORMAL, mean=0.11, std=0.015, min_value=0.08, max_value=0.15),
            ComponentVariation("lipid", VariationType.UNIFORM, min_value=0.03, max_value=0.06),
            ComponentVariation("cellulose", VariationType.NORMAL, mean=0.03, std=0.01, min_value=0.02, max_value=0.05),
        ],
        default_target="protein",
        tags=["grain", "cereal", "agriculture", "nn_training"],
        references=["USDA FoodData Central"],
    )

    PRODUCT_TEMPLATES["soybean"] = ProductTemplate(
        name="soybean",
        description="Soybean with variable protein and oil content",
        category="legume",
        domain="agriculture",
        components=[
            ComponentVariation("protein", VariationType.UNIFORM, min_value=0.32, max_value=0.42),
            ComponentVariation("lipid", VariationType.CORRELATED, correlated_with="protein", correlation=-0.4,
                               min_value=0.16, max_value=0.24, mean=0.20, std=0.02),
            ComponentVariation("moisture", VariationType.NORMAL, mean=0.10, std=0.015, min_value=0.07, max_value=0.14),
            ComponentVariation("starch", VariationType.FIXED, value=0.05),
            ComponentVariation("cellulose", VariationType.NORMAL, mean=0.15, std=0.02, min_value=0.10, max_value=0.20),
            ComponentVariation("sucrose", VariationType.NORMAL, mean=0.05, std=0.01, min_value=0.03, max_value=0.07),
        ],
        default_target="protein",
        tags=["legume", "oilseed", "agriculture", "nn_training"],
        references=["USDA FoodData Central"],
    )

    PRODUCT_TEMPLATES["rice_grain"] = ProductTemplate(
        name="rice_grain",
        description="Polished rice grain composition",
        category="grain",
        domain="agriculture",
        components=[
            ComponentVariation("starch", VariationType.NORMAL, mean=0.78, std=0.02, min_value=0.74, max_value=0.82),
            ComponentVariation("protein", VariationType.UNIFORM, min_value=0.05, max_value=0.10),
            ComponentVariation("moisture", VariationType.NORMAL, mean=0.12, std=0.01, min_value=0.10, max_value=0.14),
            ComponentVariation("lipid", VariationType.FIXED, value=0.01),
            ComponentVariation("cellulose", VariationType.FIXED, value=0.01),
        ],
        default_target="protein",
        tags=["grain", "cereal", "agriculture", "nn_training"],
        references=["USDA FoodData Central"],
    )

    PRODUCT_TEMPLATES["barley_grain"] = ProductTemplate(
        name="barley_grain",
        description="Barley grain for malting and feed",
        category="grain",
        domain="agriculture",
        components=[
            ComponentVariation("starch", VariationType.NORMAL, mean=0.60, std=0.03, min_value=0.54, max_value=0.66),
            ComponentVariation("protein", VariationType.UNIFORM, min_value=0.08, max_value=0.15),
            ComponentVariation("moisture", VariationType.NORMAL, mean=0.11, std=0.01, min_value=0.09, max_value=0.13),
            ComponentVariation("lipid", VariationType.FIXED, value=0.02),
            ComponentVariation("cellulose", VariationType.NORMAL, mean=0.14, std=0.02, min_value=0.10, max_value=0.18),
        ],
        default_target="protein",
        tags=["grain", "cereal", "malting", "agriculture", "nn_training"],
        references=["USDA FoodData Central"],
    )

    # =========================================================================
    # PHARMACEUTICAL
    # =========================================================================
    PRODUCT_TEMPLATES["tablet_variable_api"] = ProductTemplate(
        name="tablet_variable_api",
        description="Tablet with variable API content (process monitoring)",
        category="solid_dosage",
        domain="pharmaceutical",
        components=[
            # API content varies (simulating blend uniformity issues)
            ComponentVariation("paracetamol", VariationType.NORMAL, mean=0.50, std=0.05, min_value=0.40, max_value=0.60),
            ComponentVariation("microcrystalline_cellulose", VariationType.NORMAL, mean=0.25, std=0.03,
                               min_value=0.18, max_value=0.32),
            ComponentVariation("starch", VariationType.NORMAL, mean=0.15, std=0.02, min_value=0.10, max_value=0.20),
            ComponentVariation("moisture", VariationType.UNIFORM, min_value=0.02, max_value=0.08),
        ],
        default_target="paracetamol",
        tags=["pharma", "tablet", "api", "process_monitoring", "nn_training"],
        references=["Reich2005"],
    )

    PRODUCT_TEMPLATES["tablet_moisture_stability"] = ProductTemplate(
        name="tablet_moisture_stability",
        description="Tablet with variable moisture (stability study)",
        category="solid_dosage",
        domain="pharmaceutical",
        components=[
            ComponentVariation("moisture", VariationType.UNIFORM, min_value=0.01, max_value=0.12),
            ComponentVariation("paracetamol", VariationType.NORMAL, mean=0.50, std=0.02, min_value=0.46, max_value=0.54),
            ComponentVariation("microcrystalline_cellulose", VariationType.FIXED, value=0.25),
            ComponentVariation("starch", VariationType.FIXED, value=0.15),
        ],
        default_target="moisture",
        tags=["pharma", "tablet", "stability", "moisture", "nn_training"],
        references=["Reich2005"],
    )

    PRODUCT_TEMPLATES["capsule_blend_uniformity"] = ProductTemplate(
        name="capsule_blend_uniformity",
        description="Capsule fill with blend uniformity variation",
        category="solid_dosage",
        domain="pharmaceutical",
        components=[
            ComponentVariation("ibuprofen", VariationType.NORMAL, mean=0.40, std=0.04, min_value=0.32, max_value=0.48),
            ComponentVariation("starch", VariationType.CORRELATED, correlated_with="ibuprofen", correlation=-0.3,
                               min_value=0.25, max_value=0.40, mean=0.32, std=0.03),
            ComponentVariation("lactose", VariationType.NORMAL, mean=0.20, std=0.02, min_value=0.15, max_value=0.25),
            ComponentVariation("moisture", VariationType.UNIFORM, min_value=0.03, max_value=0.08),
        ],
        default_target="ibuprofen",
        tags=["pharma", "capsule", "blend_uniformity", "nn_training"],
        references=["Reich2005"],
    )

    # =========================================================================
    # HIGH-VARIABILITY NN TRAINING TEMPLATES
    # =========================================================================
    PRODUCT_TEMPLATES["food_cholesterol_variable"] = ProductTemplate(
        name="food_cholesterol_variable",
        description="Food matrix with wide cholesterol variability for robust NN training",
        category="nn_training",
        domain="food",
        components=[
            # Cholesterol with very wide range (eggs, meats, dairy, processed foods)
            ComponentVariation("cholesterol", VariationType.LOGNORMAL, mean=0.01, std=0.015,
                               min_value=0.0001, max_value=0.05),
            ComponentVariation("lipid", VariationType.CORRELATED, correlated_with="cholesterol", correlation=0.7,
                               min_value=0.01, max_value=0.50, mean=0.15, std=0.12),
            ComponentVariation("protein", VariationType.UNIFORM, min_value=0.05, max_value=0.30),
            ComponentVariation("water", VariationType.COMPUTED, compute_as="remainder"),
        ],
        default_target="cholesterol",
        tags=["food", "cholesterol", "high_variability", "nn_training", "robust"],
        references=["USDA FoodData Central"],
    )

    PRODUCT_TEMPLATES["universal_fat_predictor"] = ProductTemplate(
        name="universal_fat_predictor",
        description="Wide fat range across food categories for universal fat NN",
        category="nn_training",
        domain="food",
        components=[
            # Fat spanning skim milk to butter
            ComponentVariation("lipid", VariationType.UNIFORM, min_value=0.001, max_value=0.85),
            ComponentVariation("protein", VariationType.UNIFORM, min_value=0.01, max_value=0.35),
            ComponentVariation("starch", VariationType.UNIFORM, min_value=0.0, max_value=0.40),
            ComponentVariation("water", VariationType.COMPUTED, compute_as="remainder"),
        ],
        default_target="lipid",
        tags=["food", "fat", "universal", "high_variability", "nn_training"],
        references=["USDA FoodData Central"],
    )

    PRODUCT_TEMPLATES["universal_protein_predictor"] = ProductTemplate(
        name="universal_protein_predictor",
        description="Wide protein range across food/feed for universal protein NN",
        category="nn_training",
        domain="food",
        components=[
            # Protein spanning vegetables to pure protein isolates
            ComponentVariation("protein", VariationType.UNIFORM, min_value=0.01, max_value=0.95),
            ComponentVariation("lipid", VariationType.UNIFORM, min_value=0.001, max_value=0.30),
            ComponentVariation("starch", VariationType.UNIFORM, min_value=0.0, max_value=0.50),
            ComponentVariation("cellulose", VariationType.UNIFORM, min_value=0.0, max_value=0.20),
            ComponentVariation("water", VariationType.COMPUTED, compute_as="remainder"),
        ],
        default_target="protein",
        tags=["food", "protein", "universal", "high_variability", "nn_training"],
        references=["USDA FoodData Central"],
    )

    PRODUCT_TEMPLATES["universal_moisture_predictor"] = ProductTemplate(
        name="universal_moisture_predictor",
        description="Wide moisture range for universal moisture NN",
        category="nn_training",
        domain="food",
        components=[
            # Moisture from dried products to liquids
            ComponentVariation("water", VariationType.UNIFORM, min_value=0.02, max_value=0.98),
            ComponentVariation("protein", VariationType.UNIFORM, min_value=0.01, max_value=0.40),
            ComponentVariation("starch", VariationType.UNIFORM, min_value=0.0, max_value=0.80),
            ComponentVariation("lipid", VariationType.UNIFORM, min_value=0.0, max_value=0.50),
        ],
        default_target="water",
        tags=["food", "moisture", "water", "universal", "high_variability", "nn_training"],
        references=["USDA FoodData Central"],
    )

    # =========================================================================
    # FRUITS AND VEGETABLES
    # =========================================================================
    PRODUCT_TEMPLATES["fruit_sugar_variable"] = ProductTemplate(
        name="fruit_sugar_variable",
        description="Fruit with variable sugar content (ripeness variation)",
        category="fruit",
        domain="food",
        components=[
            ComponentVariation("fructose", VariationType.UNIFORM, min_value=0.02, max_value=0.12),
            ComponentVariation("glucose", VariationType.CORRELATED, correlated_with="fructose", correlation=0.9,
                               min_value=0.01, max_value=0.08, mean=0.04, std=0.02),
            ComponentVariation("sucrose", VariationType.UNIFORM, min_value=0.005, max_value=0.08),
            ComponentVariation("malic_acid", VariationType.CORRELATED, correlated_with="fructose", correlation=-0.6,
                               min_value=0.001, max_value=0.015, mean=0.005, std=0.003),
            ComponentVariation("water", VariationType.COMPUTED, compute_as="remainder"),
        ],
        default_target="fructose",
        tags=["fruit", "sugar", "ripeness", "nn_training"],
        references=["USDA FoodData Central"],
    )


# Register templates on module load
_register_templates()


# =============================================================================
# ProductGenerator Class
# =============================================================================


class ProductGenerator:
    """
    Generator for product-level synthetic NIRS spectra.

    ProductGenerator creates realistic synthetic spectra based on predefined
    product templates with controlled composition variability. It handles
    correlation constraints, compositional bounds, and efficient batch
    generation for neural network training.

    Attributes:
        template: The ProductTemplate used for generation.
        library: ComponentLibrary with the required spectral components.
        rng: NumPy random generator for reproducibility.

    Args:
        template: Template name (str) or ProductTemplate object.
        random_state: Random seed for reproducibility.
        wavelength_start: Start wavelength in nm (default: 1000).
        wavelength_end: End wavelength in nm (default: 2500).
        wavelength_step: Wavelength step in nm (default: 2).
        wavelengths: Custom wavelength array (overrides start/end/step).
        instrument_wavelength_grid: Predefined instrument grid name.
        complexity: Spectral complexity ('simple', 'realistic', 'complex').

    Example:
        >>> # Generate milk samples with variable fat
        >>> generator = ProductGenerator("milk_variable_fat", random_state=42)
        >>> dataset = generator.generate(n_samples=1000, target="lipid")
        >>>
        >>> # High-variability training data
        >>> generator = ProductGenerator("universal_protein_predictor")
        >>> dataset = generator.generate(n_samples=50000, target="protein")
        >>>
        >>> # Match specific instrument wavelengths
        >>> generator = ProductGenerator(
        ...     "wheat_variable_protein",
        ...     instrument_wavelength_grid="foss_xds"
        ... )
    """

    def __init__(
        self,
        template: Union[str, ProductTemplate],
        random_state: Optional[int] = None,
        wavelength_start: float = 1000.0,
        wavelength_end: float = 2500.0,
        wavelength_step: float = 2.0,
        wavelengths: Optional[np.ndarray] = None,
        instrument_wavelength_grid: Optional[str] = None,
        complexity: str = "realistic",
    ) -> None:
        """Initialize the product generator."""
        # Get template
        if isinstance(template, str):
            self.template = get_product_template(template)
        else:
            self.template = template

        self._random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.complexity = complexity

        # Store wavelength config
        self._wavelength_start = wavelength_start
        self._wavelength_end = wavelength_end
        self._wavelength_step = wavelength_step
        self._wavelengths = wavelengths
        self._instrument_wavelength_grid = instrument_wavelength_grid

        # Create component library from template components
        self.library = ComponentLibrary.from_predefined(
            self.template.component_names,
            random_state=random_state,
        )

    def _sample_compositions(self, n_samples: int) -> np.ndarray:
        """
        Sample component compositions respecting variability and correlations.

        This method generates realistic concentration matrices by:
        1. Sampling independent components according to their variation types
        2. Applying correlation constraints
        3. Computing "remainder" components
        4. Ensuring non-negative values and reasonable totals

        Args:
            n_samples: Number of composition samples to generate.

        Returns:
            Concentration matrix of shape (n_samples, n_components).
        """
        n_components = len(self.template.components)
        comp_names = self.template.component_names

        # Initialize result matrix
        concentrations = np.zeros((n_samples, n_components))

        # Track which components have been sampled
        sampled = set()

        # First pass: sample independent components
        for i, comp_var in enumerate(self.template.components):
            vtype = comp_var.variation_type

            if vtype == VariationType.FIXED:
                concentrations[:, i] = comp_var.value
                sampled.add(comp_var.component)

            elif vtype == VariationType.UNIFORM:
                concentrations[:, i] = self.rng.uniform(
                    comp_var.min_value, comp_var.max_value, n_samples
                )
                sampled.add(comp_var.component)

            elif vtype == VariationType.NORMAL:
                values = self.rng.normal(comp_var.mean, comp_var.std, n_samples)
                if comp_var.min_value is not None and comp_var.max_value is not None:
                    values = np.clip(values, comp_var.min_value, comp_var.max_value)
                concentrations[:, i] = values
                sampled.add(comp_var.component)

            elif vtype == VariationType.LOGNORMAL:
                # Convert mean/std to log-space parameters
                mean, std = comp_var.mean, comp_var.std
                sigma_sq = np.log(1 + (std / mean) ** 2)
                mu = np.log(mean) - sigma_sq / 2
                sigma = np.sqrt(sigma_sq)
                values = self.rng.lognormal(mu, sigma, n_samples)
                if comp_var.min_value is not None and comp_var.max_value is not None:
                    values = np.clip(values, comp_var.min_value, comp_var.max_value)
                concentrations[:, i] = values
                sampled.add(comp_var.component)

        # Second pass: sample correlated components
        for i, comp_var in enumerate(self.template.components):
            if comp_var.variation_type != VariationType.CORRELATED:
                continue
            if comp_var.component in sampled:
                continue

            # Get source component values
            source_name = comp_var.correlated_with
            source_idx = comp_names.index(source_name)
            source_values = concentrations[:, source_idx]

            # Get source component variation for normalization
            source_comp = None
            for cv in self.template.components:
                if cv.component == source_name:
                    source_comp = cv
                    break

            # Normalize source to [0, 1] range
            if source_comp is not None:
                if source_comp.min_value is not None and source_comp.max_value is not None:
                    source_range = source_comp.max_value - source_comp.min_value
                    source_normalized = (source_values - source_comp.min_value) / source_range
                else:
                    source_normalized = source_values / source_values.max()
            else:
                source_normalized = source_values / source_values.max()

            # Generate correlated values
            correlation = comp_var.correlation
            mean = comp_var.mean if comp_var.mean is not None else 0.5
            std = comp_var.std if comp_var.std is not None else 0.1

            # Use Cholesky decomposition for correlation
            noise = self.rng.normal(0, 1, n_samples)
            correlated = correlation * source_normalized + np.sqrt(1 - correlation**2) * noise

            # Scale to target range
            if comp_var.min_value is not None and comp_var.max_value is not None:
                target_range = comp_var.max_value - comp_var.min_value
                target_center = (comp_var.min_value + comp_var.max_value) / 2

                # Map correlated values to target range
                values = target_center + (correlated - 0.5) * target_range
                values = np.clip(values, comp_var.min_value, comp_var.max_value)
            else:
                # Use mean/std
                values = mean + correlated * std
                values = np.maximum(values, 0)

            concentrations[:, i] = values
            sampled.add(comp_var.component)

        # Third pass: compute "remainder" components
        for i, comp_var in enumerate(self.template.components):
            if comp_var.variation_type != VariationType.COMPUTED:
                continue

            if comp_var.compute_as == "remainder":
                # Sum all other components
                mask = np.ones(n_components, dtype=bool)
                mask[i] = False
                other_sum = concentrations[:, mask].sum(axis=1)
                # Compute remainder
                remainder = 1.0 - other_sum
                # Clip to reasonable bounds
                remainder = np.clip(remainder, 0.01, 0.99)
                concentrations[:, i] = remainder
            else:
                raise ValueError(f"Unknown compute_as: {comp_var.compute_as}")

        return concentrations

    def generate(
        self,
        n_samples: int = 1000,
        target: Optional[str] = None,
        train_ratio: float = 0.8,
        include_batch_effects: bool = False,
        n_batches: int = 1,
        return_concentrations: bool = False,
    ) -> Union["SpectroDataset", Tuple["SpectroDataset", np.ndarray]]:
        """
        Generate synthetic product samples.

        Args:
            n_samples: Number of samples to generate.
            target: Component to use as regression target.
                If None, uses template's default_target.
            train_ratio: Proportion of samples for training partition.
            include_batch_effects: Whether to add batch/session effects.
            n_batches: Number of batches (if include_batch_effects=True).
            return_concentrations: If True, also return the full concentration matrix.

        Returns:
            SpectroDataset with train/test partitions.
            If return_concentrations=True, returns (dataset, concentrations).

        Example:
            >>> generator = ProductGenerator("milk_variable_fat")
            >>> dataset = generator.generate(n_samples=1000, target="lipid")
            >>> print(f"Train: {dataset.n_train}, Test: {dataset.n_test}")
        """
        from .generator import SyntheticNIRSGenerator
        from nirs4all.data.dataset import SpectroDataset

        # Determine target component
        if target is None:
            target = self.template.default_target
        if not target:
            # Use first component if no default
            target = self.template.component_names[0]

        # Sample compositions
        concentrations = self._sample_compositions(n_samples)

        # Create generator with matching library
        generator = SyntheticNIRSGenerator(
            wavelength_start=self._wavelength_start,
            wavelength_end=self._wavelength_end,
            wavelength_step=self._wavelength_step,
            wavelengths=self._wavelengths,
            instrument_wavelength_grid=self._instrument_wavelength_grid,
            component_library=self.library,
            complexity=self.complexity,
            random_state=self._random_state,
        )

        # Generate spectra from concentrations
        X, metadata = generator.generate_from_concentrations(
            concentrations,
            include_batch_effects=include_batch_effects,
            n_batches=n_batches,
        )

        # Get target values
        target_idx = self.template.component_names.index(target)
        y = concentrations[:, target_idx]

        # Calculate split
        n_train = int(n_samples * train_ratio)

        # Create dataset
        dataset = SpectroDataset(name=f"synthetic_{self.template.name}")

        # Create wavelength headers
        headers = [str(int(wl)) for wl in generator.wavelengths]

        # Add training samples
        dataset.add_samples(
            X[:n_train],
            indexes={"partition": "train"},
            headers=headers,
            header_unit="nm",
        )
        dataset.add_targets(y[:n_train])

        # Add test samples
        dataset.add_samples(
            X[n_train:],
            indexes={"partition": "test"},
            headers=headers,
            header_unit="nm",
        )
        dataset.add_targets(y[n_train:])

        if return_concentrations:
            return dataset, concentrations
        return dataset

    def generate_dataset_for_target(
        self,
        target: str,
        n_samples: int = 1000,
        target_range: Optional[Tuple[float, float]] = None,
        **kwargs: Any,
    ) -> "SpectroDataset":
        """
        Generate dataset optimized for a specific target component.

        This is a convenience method that generates a dataset and optionally
        scales the target values to a specified range.

        Args:
            target: Component to use as regression target.
            n_samples: Number of samples to generate.
            target_range: Optional (min, max) to scale target values.
            **kwargs: Additional arguments passed to generate().

        Returns:
            SpectroDataset ready for pipeline use.

        Example:
            >>> generator = ProductGenerator("wheat_variable_protein")
            >>> dataset = generator.generate_dataset_for_target(
            ...     target="protein",
            ...     n_samples=10000,
            ...     target_range=(0, 100)  # Scale to percentage
            ... )
        """
        # Generate with return_concentrations to get the target component
        dataset, concentrations = self.generate(
            n_samples=n_samples,
            target=target,
            return_concentrations=True,
            **kwargs
        )

        if target_range is not None:
            # Get target values from concentrations
            target_idx = self.template.component_names.index(target)
            y = concentrations[:, target_idx]
            y_min, y_max = y.min(), y.max()
            if y_max > y_min:
                target_min, target_max = target_range
                y_scaled = (y - y_min) / (y_max - y_min) * (target_max - target_min) + target_min

                # Re-create dataset with scaled targets
                from nirs4all.data.dataset import SpectroDataset as DS
                from .generator import SyntheticNIRSGenerator

                # Get wavelengths from generator
                generator = SyntheticNIRSGenerator(
                    wavelength_start=self._wavelength_start,
                    wavelength_end=self._wavelength_end,
                    wavelength_step=self._wavelength_step,
                    wavelengths=self._wavelengths,
                    instrument_wavelength_grid=self._instrument_wavelength_grid,
                    component_library=self.library,
                    complexity=self.complexity,
                    random_state=self._random_state,
                )

                # Calculate split
                train_ratio = kwargs.get("train_ratio", 0.8)
                n_train = int(n_samples * train_ratio)

                # Get X from original dataset
                X = dataset.x({}, layout="2d")

                # Create new dataset with scaled y
                new_dataset = DS(name=f"synthetic_{self.template.name}")
                headers = [str(int(wl)) for wl in generator.wavelengths]

                # Add training samples
                new_dataset.add_samples(
                    X[:n_train],
                    indexes={"partition": "train"},
                    headers=headers,
                    header_unit="nm",
                )
                new_dataset.add_targets(y_scaled[:n_train])

                # Add test samples
                new_dataset.add_samples(
                    X[n_train:],
                    indexes={"partition": "test"},
                    headers=headers,
                    header_unit="nm",
                )
                new_dataset.add_targets(y_scaled[n_train:])

                return new_dataset

        return dataset

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ProductGenerator(template='{self.template.name}', "
            f"n_components={len(self.template.components)}, "
            f"default_target='{self.template.default_target}')"
        )


# =============================================================================
# CategoryGenerator Class
# =============================================================================


class CategoryGenerator:
    """
    Generator combining multiple product templates for diverse datasets.

    CategoryGenerator enables creation of training datasets that span
    multiple product types, useful for building robust models that
    generalize across categories.

    Attributes:
        templates: List of ProductTemplate objects.
        generators: List of ProductGenerator objects for each template.

    Args:
        templates: List of template names or ProductTemplate objects.
        random_state: Random seed for reproducibility.
        **kwargs: Additional arguments passed to ProductGenerator.

    Example:
        >>> # Combine dairy products
        >>> gen = CategoryGenerator(["milk_variable_fat", "cheese_variable_moisture"])
        >>> dataset = gen.generate(n_samples=2000, target="lipid")
        >>>
        >>> # Universal fat predictor training
        >>> gen = CategoryGenerator([
        ...     "milk_variable_fat",
        ...     "cheese_variable_moisture",
        ...     "meat_variable_fat",
        ... ])
        >>> dataset = gen.generate(n_samples=10000, target="lipid")
    """

    def __init__(
        self,
        templates: List[Union[str, ProductTemplate]],
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the category generator."""
        self._random_state = random_state
        self.rng = np.random.default_rng(random_state)

        # Convert template names to ProductTemplate objects
        self.templates: List[ProductTemplate] = []
        for template in templates:
            if isinstance(template, str):
                self.templates.append(get_product_template(template))
            else:
                self.templates.append(template)

        # Create generators for each template
        # Use different random states for each generator
        self.generators: List[ProductGenerator] = []
        for i, template in enumerate(self.templates):
            seed = random_state + i if random_state is not None else None
            self.generators.append(
                ProductGenerator(template, random_state=seed, **kwargs)
            )

    def generate(
        self,
        n_samples: int = 1000,
        target: Optional[str] = None,
        samples_per_template: Optional[List[int]] = None,
        train_ratio: float = 0.8,
        shuffle: bool = True,
        include_template_labels: bool = False,
    ) -> "SpectroDataset":
        """
        Generate combined dataset from multiple templates.

        Args:
            n_samples: Total number of samples to generate.
            target: Component to use as regression target.
                Must exist in all templates.
            samples_per_template: Number of samples per template.
                If None, divides equally.
            train_ratio: Proportion of samples for training partition.
            shuffle: Whether to shuffle samples across templates.
            include_template_labels: If True, adds template index as metadata.

        Returns:
            SpectroDataset combining samples from all templates.

        Example:
            >>> gen = CategoryGenerator(["milk_variable_fat", "meat_variable_fat"])
            >>> dataset = gen.generate(n_samples=2000, target="lipid")
        """
        from nirs4all.data.dataset import SpectroDataset

        # Determine samples per template
        if samples_per_template is None:
            n_templates = len(self.templates)
            base_samples = n_samples // n_templates
            samples_per_template = [base_samples] * n_templates
            # Add remainder to last template
            samples_per_template[-1] += n_samples % n_templates

        # Collect data from all templates
        all_X: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        all_template_ids: List[np.ndarray] = []
        wavelengths = None

        for i, (gen, n) in enumerate(zip(self.generators, samples_per_template)):
            # Use target from first template if not specified
            t = target if target else gen.template.default_target

            # Check if target exists in this template
            if t not in gen.template.component_names:
                raise ValueError(
                    f"Target component '{t}' not found in template '{gen.template.name}'. "
                    f"Available components: {gen.template.component_names}"
                )

            # Generate with 100% train ratio (we'll split later)
            dataset, concentrations = gen.generate(
                n_samples=n,
                target=t,
                train_ratio=1.0,
                return_concentrations=True,
            )

            # Get X
            X = dataset.x({}, layout="2d")

            # Get y from concentrations (using the target index for this template)
            target_idx = gen.template.component_names.index(t)
            y = concentrations[:, target_idx]

            all_X.append(X)
            all_y.append(y)
            all_template_ids.append(np.full(n, i))

            if wavelengths is None:
                wavelengths = dataset.wavelengths_nm()

        # Concatenate
        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)
        template_ids = np.concatenate(all_template_ids)

        # Shuffle if requested
        if shuffle:
            indices = self.rng.permutation(len(X_combined))
            X_combined = X_combined[indices]
            y_combined = y_combined[indices]
            template_ids = template_ids[indices]

        # Split train/test
        n_total = len(X_combined)
        n_train = int(n_total * train_ratio)

        # Create dataset
        dataset = SpectroDataset(name="synthetic_category")

        # Create wavelength headers
        headers = [str(int(wl)) for wl in wavelengths]

        # Add training samples
        train_meta = {"partition": "train"}
        if include_template_labels:
            train_meta["template_id"] = template_ids[:n_train]
        dataset.add_samples(
            X_combined[:n_train],
            indexes=train_meta,
            headers=headers,
            header_unit="nm",
        )
        dataset.add_targets(y_combined[:n_train])

        # Add test samples
        test_meta = {"partition": "test"}
        if include_template_labels:
            test_meta["template_id"] = template_ids[n_train:]
        dataset.add_samples(
            X_combined[n_train:],
            indexes=test_meta,
            headers=headers,
            header_unit="nm",
        )
        dataset.add_targets(y_combined[n_train:])

        return dataset

    def __repr__(self) -> str:
        """Return string representation."""
        template_names = [t.name for t in self.templates]
        return f"CategoryGenerator(templates={template_names})"


# =============================================================================
# Convenience Functions
# =============================================================================


def list_product_templates(
    category: Optional[str] = None,
    domain: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[str]:
    """
    List available product templates with optional filtering.

    Args:
        category: Filter by category (e.g., "dairy", "grain", "pharma").
        domain: Filter by domain (e.g., "food", "agriculture", "pharmaceutical").
        tags: Filter by tags (any match).

    Returns:
        Sorted list of template names matching the criteria.

    Example:
        >>> # List all templates
        >>> all_templates = list_product_templates()
        >>>
        >>> # List dairy templates
        >>> dairy = list_product_templates(category="dairy")
        >>>
        >>> # List NN training templates
        >>> nn_templates = list_product_templates(tags=["nn_training"])
    """
    results = []

    for name, template in PRODUCT_TEMPLATES.items():
        if category and template.category != category:
            continue
        if domain and template.domain != domain:
            continue
        if tags:
            if not any(t in template.tags for t in tags):
                continue
        results.append(name)

    return sorted(results)


def get_product_template(name: str) -> ProductTemplate:
    """
    Get a product template by name.

    Args:
        name: Template name.

    Returns:
        ProductTemplate object.

    Raises:
        ValueError: If template name is not found.

    Example:
        >>> template = get_product_template("milk_variable_fat")
        >>> print(template.description)
        Milk with variable fat content (skim to whole)
    """
    if name not in PRODUCT_TEMPLATES:
        available = list(PRODUCT_TEMPLATES.keys())
        raise ValueError(f"Unknown product template: '{name}'. Available: {available}")
    return PRODUCT_TEMPLATES[name]


def generate_product_samples(
    template: Union[str, ProductTemplate],
    n_samples: int = 1000,
    target: Optional[str] = None,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> "SpectroDataset":
    """
    Generate synthetic product samples (convenience function).

    This is a shorthand for creating a ProductGenerator and calling generate().

    Args:
        template: Template name or ProductTemplate object.
        n_samples: Number of samples to generate.
        target: Component to use as regression target.
        random_state: Random seed for reproducibility.
        **kwargs: Additional arguments passed to ProductGenerator.generate().

    Returns:
        SpectroDataset with synthetic samples.

    Example:
        >>> from nirs4all.synthesis import generate_product_samples
        >>>
        >>> # Generate milk samples
        >>> dataset = generate_product_samples(
        ...     "milk_variable_fat",
        ...     n_samples=1000,
        ...     target="lipid",
        ...     random_state=42
        ... )
    """
    generator = ProductGenerator(template, random_state=random_state)
    return generator.generate(n_samples=n_samples, target=target, **kwargs)


def product_template_info(name: str) -> str:
    """
    Return formatted information about a product template.

    Args:
        name: Template name.

    Returns:
        Human-readable string with template details.

    Example:
        >>> print(product_template_info("wheat_variable_protein"))
    """
    template = get_product_template(name)
    return template.info()


def list_product_categories() -> List[str]:
    """
    List all unique product categories.

    Returns:
        Sorted list of category names.

    Example:
        >>> categories = list_product_categories()
        >>> print(categories)
        ['dairy', 'fruit', 'grain', 'legume', 'meat', 'nn_training', 'solid_dosage']
    """
    categories = set()
    for template in PRODUCT_TEMPLATES.values():
        categories.add(template.category)
    return sorted(categories)


def list_product_domains() -> List[str]:
    """
    List all unique product domains.

    Returns:
        Sorted list of domain names.

    Example:
        >>> domains = list_product_domains()
        >>> print(domains)
        ['agriculture', 'food', 'pharmaceutical']
    """
    domains = set()
    for template in PRODUCT_TEMPLATES.values():
        domains.add(template.domain)
    return sorted(domains)
