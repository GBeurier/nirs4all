"""
Top-level generate() API for synthetic NIRS data generation.

This module provides the primary entry points for generating synthetic
NIRS datasets within nirs4all.

Example:
    >>> import nirs4all
    >>>
    >>> # Simple generation
    >>> dataset = nirs4all.generate(n_samples=1000, random_state=42)
    >>>
    >>> # Convenience functions
    >>> dataset = nirs4all.generate.regression(n_samples=500)
    >>> dataset = nirs4all.generate.classification(n_samples=300, n_classes=3)
    >>>
    >>> # Builder access
    >>> builder = nirs4all.generate.builder(n_samples=1000)
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.synthesis import SyntheticDatasetBuilder

def generate(
    n_samples: int = 1000,
    *,
    random_state: int | None = None,
    complexity: Literal["simple", "realistic", "complex"] = "simple",
    wavelength_range: tuple[float, float] | None = None,
    components: list[str] | None = None,
    target_range: tuple[float, float] | None = None,
    train_ratio: float = 0.8,
    as_dataset: bool = True,
    name: str = "synthetic_nirs",
    **kwargs: Any,
) -> SpectroDataset | tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic NIRS dataset.

    This is the primary function for creating synthetic spectroscopic data.
    It provides a simple interface for common use cases while allowing
    full customization through keyword arguments.

    Args:
        n_samples: Number of samples to generate.
        random_state: Random seed for reproducibility.
        complexity: Complexity level affecting noise, scatter, etc.
            Options: 'simple' (fast, minimal noise), 'realistic' (typical NIR),
            'complex' (challenging scenarios).
        wavelength_range: Tuple of (start, end) wavelengths in nm.
            Defaults to (1000, 2500) which covers the full NIR range.
        components: List of predefined component names to use.
            Options: 'water', 'protein', 'lipid', 'starch', 'cellulose',
            'chlorophyll', 'oil', 'nitrogen_compound'.
        target_range: Optional (min, max) range for scaling targets.
        train_ratio: Proportion of samples for training partition.
        as_dataset: If True, returns SpectroDataset. If False, returns (X, y) tuple.
        name: Dataset name.
        **kwargs: Additional arguments passed to SyntheticDatasetBuilder.

    Returns:
        If as_dataset=True: SpectroDataset ready for pipeline use.
        If as_dataset=False: Tuple of (X, y) numpy arrays.

    Example:
        >>> import nirs4all
        >>>
        >>> # Basic usage
        >>> dataset = nirs4all.generate(n_samples=1000, random_state=42)
        >>>
        >>> # Quick arrays for prototyping
        >>> X, y = nirs4all.generate(n_samples=500, as_dataset=False)
        >>>
        >>> # Realistic spectra
        >>> dataset = nirs4all.generate(
        ...     n_samples=1000,
        ...     complexity="realistic",
        ...     components=["water", "protein", "lipid"],
        ...     target_range=(0, 100),
        ...     random_state=42
        ... )

    See Also:
        generate.regression: Convenience function for regression datasets.
        generate.classification: Convenience function for classification datasets.
        generate.builder: Access the full builder API.
    """
    from nirs4all.synthesis import SyntheticDatasetBuilder

    builder = SyntheticDatasetBuilder(
        n_samples=n_samples,
        random_state=random_state,
        name=name,
    )

    # Configure features
    feature_kwargs: dict[str, Any] = {"complexity": complexity}
    if wavelength_range is not None:
        feature_kwargs["wavelength_range"] = wavelength_range
    if components is not None:
        feature_kwargs["components"] = components

    builder.with_features(**feature_kwargs)

    # Configure targets
    if target_range is not None:
        builder.with_targets(range=target_range)

    # Configure partitions
    builder.with_partitions(train_ratio=train_ratio)

    # Configure output
    builder.with_output(as_dataset=as_dataset)

    # Handle additional kwargs for advanced configuration
    if "distribution" in kwargs:
        builder.with_targets(distribution=kwargs.pop("distribution"))

    if "batch_effects" in kwargs:
        builder.with_batch_effects(enabled=kwargs.pop("batch_effects"))

    return builder.build()

def regression(
    n_samples: int = 1000,
    *,
    random_state: int | None = None,
    complexity: Literal["simple", "realistic", "complex"] = "simple",
    target_range: tuple[float, float] | None = None,
    target_component: str | int | None = None,
    distribution: Literal["dirichlet", "uniform", "lognormal", "correlated"] = "dirichlet",
    train_ratio: float = 0.8,
    as_dataset: bool = True,
    name: str = "synthetic_regression",
) -> SpectroDataset | tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic NIRS dataset for regression tasks.

    This convenience function is optimized for regression scenarios,
    with sensible defaults for target distribution and scaling.

    Args:
        n_samples: Number of samples to generate.
        random_state: Random seed for reproducibility.
        complexity: Complexity level ('simple', 'realistic', 'complex').
        target_range: Target value range (min, max) for scaling.
        target_component: Which component to use as target.
            If None, uses all components (multi-output regression).
        distribution: Concentration distribution method.
        train_ratio: Proportion of samples for training partition.
        as_dataset: If True, returns SpectroDataset. If False, returns (X, y).
        name: Dataset name.

    Returns:
        If as_dataset=True: SpectroDataset ready for pipeline use.
        If as_dataset=False: Tuple of (X, y) numpy arrays.

    Example:
        >>> import nirs4all
        >>>
        >>> # Simple regression dataset
        >>> dataset = nirs4all.generate.regression(n_samples=500)
        >>>
        >>> # Single target with scaling
        >>> dataset = nirs4all.generate.regression(
        ...     n_samples=1000,
        ...     target_range=(0, 100),
        ...     target_component="protein",
        ...     random_state=42
        ... )
    """
    from nirs4all.synthesis import SyntheticDatasetBuilder

    builder = SyntheticDatasetBuilder(
        n_samples=n_samples,
        random_state=random_state,
        name=name,
    )

    builder.with_features(complexity=complexity)

    target_kwargs: dict[str, Any] = {"distribution": distribution}
    if target_range is not None:
        target_kwargs["range"] = target_range
    if target_component is not None:
        target_kwargs["component"] = target_component

    builder.with_targets(**target_kwargs)
    builder.with_partitions(train_ratio=train_ratio)
    builder.with_output(as_dataset=as_dataset)

    return builder.build()

def classification(
    n_samples: int = 1000,
    *,
    n_classes: int = 2,
    random_state: int | None = None,
    complexity: Literal["simple", "realistic", "complex"] = "simple",
    class_separation: float = 1.0,
    class_weights: list[float] | None = None,
    train_ratio: float = 0.8,
    as_dataset: bool = True,
    name: str = "synthetic_classification",
) -> SpectroDataset | tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic NIRS dataset for classification tasks.

    This convenience function creates datasets with discrete class labels,
    suitable for classification experiments.

    Args:
        n_samples: Number of samples to generate.
        n_classes: Number of classes (2 for binary, >2 for multiclass).
        random_state: Random seed for reproducibility.
        complexity: Complexity level ('simple', 'realistic', 'complex').
        class_separation: Separation factor between classes.
            Higher values make classes more distinguishable.
        class_weights: Optional class proportions for imbalanced datasets.
            Should sum to 1.0.
        train_ratio: Proportion of samples for training partition.
        as_dataset: If True, returns SpectroDataset. If False, returns (X, y).
        name: Dataset name.

    Returns:
        If as_dataset=True: SpectroDataset ready for pipeline use.
        If as_dataset=False: Tuple of (X, y) numpy arrays where y is integer labels.

    Example:
        >>> import nirs4all
        >>>
        >>> # Binary classification
        >>> dataset = nirs4all.generate.classification(n_samples=500, n_classes=2)
        >>>
        >>> # Multiclass with imbalanced classes
        >>> dataset = nirs4all.generate.classification(
        ...     n_samples=1000,
        ...     n_classes=3,
        ...     class_weights=[0.5, 0.3, 0.2],
        ...     random_state=42
        ... )
    """
    from nirs4all.synthesis import SyntheticDatasetBuilder

    builder = SyntheticDatasetBuilder(
        n_samples=n_samples,
        random_state=random_state,
        name=name,
    )

    builder.with_features(complexity=complexity)
    builder.with_classification(
        n_classes=n_classes,
        separation=class_separation,
        class_weights=class_weights,
    )
    builder.with_partitions(train_ratio=train_ratio)
    builder.with_output(as_dataset=as_dataset)

    return builder.build()

def builder(
    n_samples: int = 1000,
    random_state: int | None = None,
    name: str = "synthetic_nirs",
) -> SyntheticDatasetBuilder:
    """
    Create a SyntheticDatasetBuilder for fine-grained control.

    Use this when you need full control over all generation parameters
    via the fluent builder interface.

    Args:
        n_samples: Number of samples to generate.
        random_state: Random seed for reproducibility.
        name: Dataset name.

    Returns:
        SyntheticDatasetBuilder instance for method chaining.

    Example:
        >>> import nirs4all
        >>>
        >>> dataset = (
        ...     nirs4all.generate.builder(n_samples=1000, random_state=42)
        ...     .with_features(
        ...         wavelength_range=(1000, 2500),
        ...         complexity="realistic",
        ...         components=["water", "protein", "lipid"]
        ...     )
        ...     .with_targets(
        ...         distribution="lognormal",
        ...         range=(5, 50),
        ...         component="protein"
        ...     )
        ...     .with_metadata(n_groups=3)
        ...     .with_partitions(train_ratio=0.8)
        ...     .with_batch_effects(n_batches=3)
        ...     .build()
        ... )
    """
    from nirs4all.synthesis import SyntheticDatasetBuilder

    return SyntheticDatasetBuilder(
        n_samples=n_samples,
        random_state=random_state,
        name=name,
    )

def multi_source(
    n_samples: int = 1000,
    sources: list[dict[str, Any]] | None = None,
    *,
    random_state: int | None = None,
    target_range: tuple[float, float] | None = None,
    train_ratio: float = 0.8,
    as_dataset: bool = True,
    name: str = "multi_source_synthetic",
) -> Any:
    """
    Generate a synthetic multi-source NIRS dataset.

    Multi-source datasets combine different types of data, such as
    multiple NIR spectral ranges or NIR spectra with auxiliary measurements.

    Args:
        n_samples: Number of samples to generate.
        sources: List of source configurations. Each source is a dict with:
            - name: Unique source identifier (required).
            - type: Source type - "nir", "vis", "aux", "markers" (default: "nir").
            - wavelength_range: (start, end) for NIR sources.
            - n_features: Number of features for auxiliary sources.
            - complexity: Complexity level for NIR sources.
            - components: Component names for NIR sources.
        random_state: Random seed for reproducibility.
        target_range: Optional (min, max) for scaling targets.
        train_ratio: Proportion of samples for training partition.
        as_dataset: If True, returns SpectroDataset. If False, returns (X, y).
        name: Dataset name.

    Returns:
        If as_dataset=True: SpectroDataset with multiple sources.
        If as_dataset=False: Tuple of (X, y) where X is concatenated features.

    Example:
        >>> import nirs4all
        >>>
        >>> # NIR + markers
        >>> dataset = nirs4all.generate.multi_source(
        ...     n_samples=500,
        ...     sources=[
        ...         {"name": "NIR", "type": "nir", "wavelength_range": (1000, 2500)},
        ...         {"name": "markers", "type": "aux", "n_features": 15}
        ...     ],
        ...     random_state=42
        ... )
        >>>
        >>> # Multiple NIR ranges
        >>> dataset = nirs4all.generate.multi_source(
        ...     n_samples=500,
        ...     sources=[
        ...         {"name": "VIS-NIR", "type": "nir", "wavelength_range": (400, 1100)},
        ...         {"name": "SWIR", "type": "nir", "wavelength_range": (1100, 2500)}
        ...     ]
        ... )
    """
    from nirs4all.synthesis import generate_multi_source as _generate_multi_source

    if sources is None:
        # Default: NIR + markers
        sources = [
            {"name": "NIR", "type": "nir", "wavelength_range": (1000, 2500)},
            {"name": "markers", "type": "aux", "n_features": 10}
        ]

    return _generate_multi_source(
        n_samples=n_samples,
        sources=sources,
        random_state=random_state,
        target_range=target_range,
        as_dataset=as_dataset,
        train_ratio=train_ratio,
        name=name,
    )

def to_folder(
    path: str | Path,
    n_samples: int = 1000,
    *,
    random_state: int | None = None,
    complexity: Literal["simple", "realistic", "complex"] = "simple",
    train_ratio: float = 0.8,
    format: Literal["standard", "single", "fragmented"] = "standard",
    wavelength_range: tuple[float, float] | None = None,
    components: list[str] | None = None,
    target_range: tuple[float, float] | None = None,
) -> Path:
    """
    Generate synthetic data and export to a folder.

    Creates a folder with CSV files compatible with nirs4all's
    DatasetConfigs loader.

    Args:
        path: Output folder path.
        n_samples: Number of samples to generate.
        random_state: Random seed for reproducibility.
        complexity: Complexity level.
        train_ratio: Train/test split ratio.
        format: Export format ('standard', 'single', 'fragmented').
        wavelength_range: Optional (start, end) wavelengths.
        components: Optional list of component names.
        target_range: Optional (min, max) for target scaling.

    Returns:
        Path to created folder.

    Example:
        >>> import nirs4all
        >>> path = nirs4all.generate.to_folder(
        ...     "data/synthetic",
        ...     n_samples=1000,
        ...     train_ratio=0.8,
        ...     random_state=42
        ... )
    """
    from nirs4all.synthesis import SyntheticDatasetBuilder

    builder = SyntheticDatasetBuilder(
        n_samples=n_samples,
        random_state=random_state,
    )

    # Configure features
    feature_kwargs: dict[str, Any] = {"complexity": complexity}
    if wavelength_range is not None:
        feature_kwargs["wavelength_range"] = wavelength_range
    if components is not None:
        feature_kwargs["components"] = components
    builder.with_features(**feature_kwargs)

    # Configure targets
    if target_range is not None:
        builder.with_targets(range=target_range)

    # Configure partitions
    builder.with_partitions(train_ratio=train_ratio)

    return builder.export(path, format=format)

def to_csv(
    path: str | Path,
    n_samples: int = 1000,
    *,
    random_state: int | None = None,
    complexity: Literal["simple", "realistic", "complex"] = "simple",
    wavelength_range: tuple[float, float] | None = None,
    target_range: tuple[float, float] | None = None,
) -> Path:
    """
    Generate synthetic data and export to a single CSV file.

    Args:
        path: Output file path.
        n_samples: Number of samples to generate.
        random_state: Random seed for reproducibility.
        complexity: Complexity level.
        wavelength_range: Optional (start, end) wavelengths.
        target_range: Optional (min, max) for target scaling.

    Returns:
        Path to created file.

    Example:
        >>> import nirs4all
        >>> path = nirs4all.generate.to_csv("data.csv", n_samples=500)
    """
    from nirs4all.synthesis import SyntheticDatasetBuilder

    builder = SyntheticDatasetBuilder(
        n_samples=n_samples,
        random_state=random_state,
    )

    # Configure features
    feature_kwargs: dict[str, Any] = {"complexity": complexity}
    if wavelength_range is not None:
        feature_kwargs["wavelength_range"] = wavelength_range
    builder.with_features(**feature_kwargs)

    # Configure targets
    if target_range is not None:
        builder.with_targets(range=target_range)

    return builder.export_to_csv(path)

def product(
    template: str,
    n_samples: int = 1000,
    *,
    target: str | None = None,
    random_state: int | None = None,
    wavelength_range: tuple[float, float] | None = None,
    wavelengths: np.ndarray | None = None,
    instrument_wavelength_grid: str | None = None,
    complexity: Literal["simple", "realistic", "complex"] = "realistic",
    train_ratio: float = 0.8,
    target_range: tuple[float, float] | None = None,
) -> SpectroDataset:
    """
    Generate synthetic NIRS dataset from a product template.

    Product templates define realistic compositions with controlled
    variability for specific product types (dairy, grain, meat, pharma).
    This is ideal for generating training data for neural networks.

    Args:
        template: Product template name (e.g., "milk_variable_fat",
            "wheat_variable_protein", "tablet_variable_api").
        n_samples: Number of samples to generate.
        target: Component to use as regression target.
            If None, uses template's default target.
        random_state: Random seed for reproducibility.
        wavelength_range: Tuple of (start, end) wavelengths in nm.
        wavelengths: Custom wavelength array.
        instrument_wavelength_grid: Predefined instrument grid name
            (e.g., "foss_xds", "micronir_onsite").
        complexity: Spectral complexity level.
        train_ratio: Proportion of samples for training partition.
        target_range: Optional (min, max) to scale target values.

    Returns:
        SpectroDataset with train/test partitions.

    Example:
        >>> import nirs4all
        >>>
        >>> # Generate dairy samples for fat prediction
        >>> dataset = nirs4all.generate.product(
        ...     "milk_variable_fat",
        ...     n_samples=1000,
        ...     target="lipid",
        ...     random_state=42
        ... )
        >>>
        >>> # Generate grain samples matching specific instrument
        >>> dataset = nirs4all.generate.product(
        ...     "wheat_variable_protein",
        ...     n_samples=5000,
        ...     instrument_wavelength_grid="foss_xds"
        ... )
        >>>
        >>> # List available templates
        >>> from nirs4all.synthesis import list_product_templates
        >>> print(list_product_templates(category="dairy"))

    See Also:
        generate.category: Generate from multiple product templates.
        list_product_templates: List available templates.
    """
    from nirs4all.synthesis import ProductGenerator

    # Build wavelength kwargs
    wl_kwargs: dict[str, Any] = {"complexity": complexity}
    if wavelength_range is not None:
        wl_kwargs["wavelength_start"] = wavelength_range[0]
        wl_kwargs["wavelength_end"] = wavelength_range[1]
    if wavelengths is not None:
        wl_kwargs["wavelengths"] = wavelengths
    if instrument_wavelength_grid is not None:
        wl_kwargs["instrument_wavelength_grid"] = instrument_wavelength_grid

    generator = ProductGenerator(template, random_state=random_state, **wl_kwargs)

    if target_range is not None:
        return generator.generate_dataset_for_target(
            target=target or generator.template.default_target,
            n_samples=n_samples,
            target_range=target_range,
            train_ratio=train_ratio,
        )
    else:
        result = generator.generate(
            n_samples=n_samples,
            target=target,
            train_ratio=train_ratio,
        )
        return result[0] if isinstance(result, tuple) else result

def category(
    templates: list[str],
    n_samples: int = 1000,
    *,
    target: str | None = None,
    random_state: int | None = None,
    samples_per_template: list[int] | None = None,
    wavelength_range: tuple[float, float] | None = None,
    instrument_wavelength_grid: str | None = None,
    complexity: Literal["simple", "realistic", "complex"] = "realistic",
    train_ratio: float = 0.8,
    shuffle: bool = True,
) -> SpectroDataset:
    """
    Generate synthetic NIRS dataset from multiple product templates.

    This creates training datasets that span multiple product types,
    useful for building robust models that generalize across categories.

    Args:
        templates: List of product template names.
        n_samples: Total number of samples to generate.
        target: Component to use as regression target.
            Must exist in all templates.
        random_state: Random seed for reproducibility.
        samples_per_template: Number of samples per template.
            If None, divides equally.
        wavelength_range: Tuple of (start, end) wavelengths in nm.
        instrument_wavelength_grid: Predefined instrument grid name.
        complexity: Spectral complexity level.
        train_ratio: Proportion of samples for training partition.
        shuffle: Whether to shuffle samples across templates.

    Returns:
        SpectroDataset combining samples from all templates.

    Example:
        >>> import nirs4all
        >>>
        >>> # Combine dairy products for fat prediction
        >>> dataset = nirs4all.generate.category(
        ...     ["milk_variable_fat", "cheese_variable_moisture", "yogurt_variable_fat"],
        ...     n_samples=3000,
        ...     target="lipid",
        ...     random_state=42
        ... )
        >>>
        >>> # Universal protein predictor
        >>> dataset = nirs4all.generate.category(
        ...     ["wheat_variable_protein", "soybean", "meat_variable_protein"],
        ...     n_samples=10000,
        ...     target="protein"
        ... )

    See Also:
        generate.product: Generate from a single product template.
    """
    from nirs4all.synthesis import CategoryGenerator

    # Build wavelength kwargs
    wl_kwargs: dict[str, Any] = {"complexity": complexity}
    if wavelength_range is not None:
        wl_kwargs["wavelength_start"] = wavelength_range[0]
        wl_kwargs["wavelength_end"] = wavelength_range[1]
    if instrument_wavelength_grid is not None:
        wl_kwargs["instrument_wavelength_grid"] = instrument_wavelength_grid

    templates_arg: list[str | Any] = list(templates)
    generator = CategoryGenerator(templates_arg, random_state=random_state, **wl_kwargs)

    return generator.generate(
        n_samples=n_samples,
        target=target,
        samples_per_template=samples_per_template,
        train_ratio=train_ratio,
        shuffle=shuffle,
    )

def from_template(
    template: str | np.ndarray | SpectroDataset,
    n_samples: int = 1000,
    *,
    random_state: int | None = None,
    wavelengths: np.ndarray | None = None,
    as_dataset: bool = True,
) -> SpectroDataset | tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data mimicking a real dataset template.

    Analyzes the template data and generates synthetic spectra
    with similar statistical and spectral properties.

    Args:
        template: Real data to mimic. Can be:
            - Path to dataset folder (str).
            - Numpy array (n_samples, n_wavelengths).
            - SpectroDataset object.
        n_samples: Number of samples to generate.
        random_state: Random seed for reproducibility.
        wavelengths: Wavelength grid (required if template is array).
        as_dataset: If True, returns SpectroDataset. If False, returns (X, y).

    Returns:
        Synthetic dataset or arrays with properties similar to template.

    Example:
        >>> import nirs4all
        >>>
        >>> # From a dataset path
        >>> dataset = nirs4all.generate.from_template(
        ...     "sample_data/regression",
        ...     n_samples=1000
        ... )
        >>>
        >>> # From numpy array
        >>> dataset = nirs4all.generate.from_template(
        ...     X_real,
        ...     n_samples=500,
        ...     wavelengths=wavelengths
        ... )
    """
    from nirs4all.synthesis import RealDataFitter, SyntheticDatasetBuilder

    builder = SyntheticDatasetBuilder(
        n_samples=n_samples,
        random_state=random_state,
    )

    # Handle string path
    if isinstance(template, str):
        from nirs4all.data import DatasetConfigs

        dataset_config = DatasetConfigs(template)
        datasets = dataset_config.get_datasets()
        if not datasets:
            raise ValueError(f"No datasets found at {template}")

        template_ds = datasets[0]
        template_array = np.asarray(template_ds.x({}, layout="2d"))
        with contextlib.suppress(AttributeError, TypeError):
            wavelengths = getattr(template_ds, 'wavelengths', None)
        builder.fit_to(template_array, wavelengths=wavelengths)
    else:
        builder.fit_to(template, wavelengths=wavelengths)

    builder.with_output(as_dataset=as_dataset)
    return builder.build()

class _GenerateNamespace:
    """
    Namespace class that makes generate both callable and a namespace.

    This allows both:
        nirs4all.generate(n_samples=1000)
        nirs4all.generate.regression(n_samples=500)
    """

    # Make the main generate function available as __call__
    __call__ = staticmethod(generate)

    # Convenience functions as class attributes
    regression = staticmethod(regression)
    classification = staticmethod(classification)
    builder = staticmethod(builder)
    multi_source = staticmethod(multi_source)

    # Export functions (Phase 4)
    to_folder = staticmethod(to_folder)
    to_csv = staticmethod(to_csv)
    from_template = staticmethod(from_template)

    # Product-level generation (Phase 7)
    product = staticmethod(product)
    category = staticmethod(category)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            "<nirs4all.generate namespace>\n"
            "  generate(n_samples, ...) - Generate synthetic NIRS dataset\n"
            "  generate.regression(...) - Generate regression dataset\n"
            "  generate.classification(...) - Generate classification dataset\n"
            "  generate.multi_source(...) - Generate multi-source dataset\n"
            "  generate.product(...) - Generate from product template (Phase 7)\n"
            "  generate.category(...) - Generate from multiple templates (Phase 7)\n"
            "  generate.builder(...) - Get fluent builder for full control\n"
            "  generate.to_folder(...) - Generate and export to folder\n"
            "  generate.to_csv(...) - Generate and export to CSV file\n"
            "  generate.from_template(...) - Generate mimicking real data"
        )

# Create the singleton namespace instance
# This replaces the module when imported
generate_namespace = _GenerateNamespace()

# For direct function access
__all__ = [
    "generate",
    "regression",
    "classification",
    "builder",
    "multi_source",
    "to_folder",
    "to_csv",
    "from_template",
    # Phase 7: Product-level generation
    "product",
    "category",
    "generate_namespace",
]
