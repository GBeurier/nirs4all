"""
Fluent builder for synthetic NIRS dataset construction.

This module provides a builder pattern interface for creating synthetic
NIRS datasets with fine-grained control over all generation parameters.

Example:
    >>> from nirs4all.data.synthetic import SyntheticDatasetBuilder
    >>>
    >>> dataset = (
    ...     SyntheticDatasetBuilder(n_samples=1000, random_state=42)
    ...     .with_features(complexity="realistic")
    ...     .with_targets(distribution="lognormal", range=(0, 100))
    ...     .with_partitions(train_ratio=0.8)
    ...     .build()
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from .generator import SyntheticNIRSGenerator
from .components import ComponentLibrary
from .config import (
    SyntheticDatasetConfig,
    FeatureConfig,
    TargetConfig,
    MetadataConfig,
    PartitionConfig,
    BatchEffectConfig,
    OutputConfig,
)
from ._constants import DEFAULT_WAVELENGTH_START, DEFAULT_WAVELENGTH_END, DEFAULT_WAVELENGTH_STEP
from .metadata import MetadataGenerator, MetadataGenerationResult
from .targets import TargetGenerator

if TYPE_CHECKING:
    from pathlib import Path
    from nirs4all.data.dataset import SpectroDataset
    from .sources import SourceConfig


@dataclass
class BuilderState:
    """
    Internal state container for the builder.

    This holds all configuration accumulated through the builder methods.
    """

    n_samples: int = 1000
    random_state: Optional[int] = None
    name: str = "synthetic_nirs"

    # Feature configuration
    wavelength_start: float = DEFAULT_WAVELENGTH_START
    wavelength_end: float = DEFAULT_WAVELENGTH_END
    wavelength_step: float = DEFAULT_WAVELENGTH_STEP
    complexity: Literal["simple", "realistic", "complex"] = "simple"
    component_names: Optional[List[str]] = None
    component_library: Optional[ComponentLibrary] = None

    # Target configuration
    concentration_method: Literal["dirichlet", "uniform", "lognormal", "correlated"] = "dirichlet"
    target_range: Optional[Tuple[float, float]] = None
    target_component: Optional[Union[str, int]] = None
    target_transform: Optional[Literal["log", "sqrt"]] = None

    # Classification configuration
    n_classes: Optional[int] = None
    class_separation: float = 1.5
    class_weights: Optional[List[float]] = None
    class_separation_method: Literal["component", "threshold", "cluster"] = "component"

    # Metadata configuration
    generate_sample_ids: bool = False
    sample_id_prefix: str = "sample"
    n_groups: Optional[int] = None
    n_repetitions: Union[int, Tuple[int, int]] = 1
    group_names: Optional[List[str]] = None

    # Multi-source configuration
    sources: Optional[List[Any]] = None  # List of SourceConfig or dicts

    # Partition configuration
    train_ratio: float = 0.8
    stratify: bool = False
    shuffle: bool = True

    # Batch effect configuration
    batch_effects_enabled: bool = False
    n_batches: int = 3

    # Output configuration
    as_dataset: bool = True
    include_metadata: bool = False

    # Cached generated data
    _X: Optional[np.ndarray] = field(default=None, repr=False)
    _y: Optional[np.ndarray] = field(default=None, repr=False)
    _C: Optional[np.ndarray] = field(default=None, repr=False)  # Concentrations
    _wavelengths: Optional[np.ndarray] = field(default=None, repr=False)
    _metadata: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _sample_metadata: Optional[MetadataGenerationResult] = field(default=None, repr=False)


class SyntheticDatasetBuilder:
    """
    Fluent builder for constructing synthetic NIRS datasets.

    This builder provides a chainable interface for configuring all aspects
    of synthetic data generation, from spectral features to targets and metadata.

    The builder accumulates configuration through method calls, then generates
    the dataset when ``build()`` is called.

    Attributes:
        state: Internal BuilderState containing all configuration.

    Args:
        n_samples: Number of samples to generate.
        random_state: Random seed for reproducibility.
        name: Dataset name.

    Example:
        >>> from nirs4all.data.synthetic import SyntheticDatasetBuilder
        >>>
        >>> # Simple usage
        >>> dataset = SyntheticDatasetBuilder(n_samples=500).build()
        >>>
        >>> # Full configuration
        >>> dataset = (
        ...     SyntheticDatasetBuilder(n_samples=1000, random_state=42)
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
        ...     .with_metadata(
        ...         n_groups=3,
        ...         n_repetitions=(2, 5)
        ...     )
        ...     .with_partitions(train_ratio=0.8)
        ...     .build()
        ... )

    See Also:
        nirs4all.generate: Top-level convenience function.
        SyntheticNIRSGenerator: Core generation engine.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        random_state: Optional[int] = None,
        name: str = "synthetic_nirs",
    ) -> None:
        """
        Initialize the builder with basic configuration.

        Args:
            n_samples: Number of samples to generate.
            random_state: Random seed for reproducibility.
            name: Dataset name.

        Raises:
            ValueError: If n_samples is less than 1.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        self.state = BuilderState(
            n_samples=n_samples,
            random_state=random_state,
            name=name,
        )
        self._built = False

    def with_features(
        self,
        *,
        wavelength_range: Optional[Tuple[float, float]] = None,
        wavelength_step: Optional[float] = None,
        complexity: Optional[Literal["simple", "realistic", "complex"]] = None,
        components: Optional[List[str]] = None,
        component_library: Optional[ComponentLibrary] = None,
    ) -> SyntheticDatasetBuilder:
        """
        Configure spectral feature generation.

        Args:
            wavelength_range: Tuple of (start, end) wavelengths in nm.
            wavelength_step: Wavelength sampling step in nm.
            complexity: Complexity level affecting noise, scatter, etc.
                Options: 'simple', 'realistic', 'complex'.
            components: List of predefined component names to use.
            component_library: Pre-configured ComponentLibrary instance.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If both components and component_library are specified.

        Example:
            >>> builder.with_features(
            ...     wavelength_range=(1000, 2500),
            ...     complexity="realistic",
            ...     components=["water", "protein"]
            ... )
        """
        if components is not None and component_library is not None:
            raise ValueError("Cannot specify both 'components' and 'component_library'")

        if wavelength_range is not None:
            self.state.wavelength_start, self.state.wavelength_end = wavelength_range

        if wavelength_step is not None:
            self.state.wavelength_step = wavelength_step

        if complexity is not None:
            self.state.complexity = complexity

        if components is not None:
            self.state.component_names = components

        if component_library is not None:
            self.state.component_library = component_library

        return self

    def with_targets(
        self,
        *,
        distribution: Optional[Literal["dirichlet", "uniform", "lognormal", "correlated"]] = None,
        range: Optional[Tuple[float, float]] = None,
        component: Optional[Union[str, int]] = None,
        transform: Optional[Literal["log", "sqrt"]] = None,
    ) -> SyntheticDatasetBuilder:
        """
        Configure target variable generation for regression tasks.

        Args:
            distribution: Concentration distribution method.
                Options: 'dirichlet', 'uniform', 'lognormal', 'correlated'.
            range: Target value range (min, max) for scaling.
            component: Which component to use as target.
                If None, uses all components (multi-output).
                If str, uses the component with that name.
                If int, uses the component at that index.
            transform: Optional transformation to apply ('log', 'sqrt').

        Returns:
            Self for method chaining.

        Example:
            >>> builder.with_targets(
            ...     distribution="lognormal",
            ...     range=(5, 50),
            ...     component="protein"
            ... )
        """
        # Clear classification settings when configuring regression
        self.state.n_classes = None

        if distribution is not None:
            self.state.concentration_method = distribution

        if range is not None:
            self.state.target_range = range

        if component is not None:
            self.state.target_component = component

        if transform is not None:
            self.state.target_transform = transform

        return self

    def with_classification(
        self,
        *,
        n_classes: int = 2,
        separation: float = 1.5,
        class_weights: Optional[List[float]] = None,
        separation_method: Literal["component", "threshold", "cluster"] = "component",
    ) -> SyntheticDatasetBuilder:
        """
        Configure target generation for classification tasks.

        This creates discrete class labels with controllable separation
        between classes, enabling classification experiments with varying
        difficulty levels.

        Args:
            n_classes: Number of classes to generate.
            separation: Class separation factor (higher = more separable).
                Values around 0.5-1.0: overlapping classes (challenging).
                Values around 1.5-2.0: moderate separation (realistic).
                Values around 2.5+: well-separated classes (easy).
            class_weights: Optional class weights for imbalanced datasets.
                Should sum to 1.0.
            separation_method: How to create class differences:
                - "component": Different component concentration profiles per class.
                - "threshold": Classes based on concentration thresholds.
                - "cluster": K-means-like cluster assignment.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.with_classification(
            ...     n_classes=3,
            ...     separation=2.0,
            ...     class_weights=[0.5, 0.3, 0.2]
            ... )
        """
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")

        if class_weights is not None:
            if len(class_weights) != n_classes:
                raise ValueError(
                    f"class_weights length ({len(class_weights)}) must match "
                    f"n_classes ({n_classes})"
                )
            if abs(sum(class_weights) - 1.0) > 0.01:
                raise ValueError(f"class_weights must sum to 1.0, got {sum(class_weights)}")

        self.state.n_classes = n_classes
        self.state.class_separation = separation
        self.state.class_weights = class_weights
        self.state.class_separation_method = separation_method

        return self

    def with_metadata(
        self,
        *,
        sample_ids: bool = True,
        sample_id_prefix: Optional[str] = None,
        n_groups: Optional[int] = None,
        n_repetitions: Optional[Union[int, Tuple[int, int]]] = None,
        group_names: Optional[List[str]] = None,
    ) -> SyntheticDatasetBuilder:
        """
        Configure sample metadata generation.

        Generates realistic metadata including sample IDs, biological sample
        groupings (with repetitions), and group assignments.

        Args:
            sample_ids: Whether to generate sample IDs.
            sample_id_prefix: Prefix for sample ID strings.
            n_groups: Number of sample groups (for grouped cross-validation).
            n_repetitions: Repetitions per biological sample. Either a fixed int
                or a (min, max) tuple for random variation. When set, each
                "biological sample" gets multiple spectral measurements.
            group_names: Optional list of group names. If None and n_groups > 0,
                generates names like "Group_0", "Group_1", etc.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.with_metadata(
            ...     n_groups=5,
            ...     n_repetitions=(2, 4),
            ...     group_names=["Field_A", "Field_B", "Field_C", "Field_D", "Field_E"]
            ... )
        """
        self.state.generate_sample_ids = sample_ids

        if sample_id_prefix is not None:
            self.state.sample_id_prefix = sample_id_prefix

        if n_groups is not None:
            self.state.n_groups = n_groups

        if n_repetitions is not None:
            self.state.n_repetitions = n_repetitions

        if group_names is not None:
            self.state.group_names = group_names

        return self

    def with_sources(
        self,
        sources: List[Union[Dict[str, Any], Any]],
    ) -> SyntheticDatasetBuilder:
        """
        Configure multi-source generation.

        Multi-source datasets combine different types of data, such as
        multiple NIR spectral ranges or NIR spectra with auxiliary measurements.

        Args:
            sources: List of source configurations. Each source is a dict with:
                - name: Unique source identifier (required).
                - type: Source type - "nir", "vis", "aux", "markers" (default: "nir").
                - wavelength_range: (start, end) for NIR sources.
                - n_features: Number of features for auxiliary sources.
                - complexity: Complexity level for NIR sources.
                - components: Component names for NIR sources.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.with_sources([
            ...     {"name": "NIR", "type": "nir", "wavelength_range": (1000, 2500)},
            ...     {"name": "markers", "type": "aux", "n_features": 15}
            ... ])
        """
        self.state.sources = sources
        return self

    def with_partitions(
        self,
        *,
        train_ratio: Optional[float] = None,
        stratify: Optional[bool] = None,
        shuffle: Optional[bool] = None,
    ) -> SyntheticDatasetBuilder:
        """
        Configure data partitioning (train/test split).

        Args:
            train_ratio: Proportion of samples for training (0.0-1.0).
            stratify: Whether to stratify by target (for classification).
            shuffle: Whether to shuffle before splitting.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.with_partitions(train_ratio=0.75, shuffle=True)
        """
        if train_ratio is not None:
            if not 0.0 < train_ratio <= 1.0:
                raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")
            self.state.train_ratio = train_ratio

        if stratify is not None:
            self.state.stratify = stratify

        if shuffle is not None:
            self.state.shuffle = shuffle

        return self

    def with_batch_effects(
        self,
        *,
        enabled: bool = True,
        n_batches: int = 3,
    ) -> SyntheticDatasetBuilder:
        """
        Configure batch/session effects simulation.

        Batch effects introduce systematic variations between measurement
        sessions, useful for domain adaptation research.

        Args:
            enabled: Whether to enable batch effects.
            n_batches: Number of measurement batches.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.with_batch_effects(n_batches=5)
        """
        self.state.batch_effects_enabled = enabled
        self.state.n_batches = n_batches
        return self

    def with_output(
        self,
        *,
        as_dataset: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
    ) -> SyntheticDatasetBuilder:
        """
        Configure output format.

        Args:
            as_dataset: If True, returns SpectroDataset. If False, returns tuple.
            include_metadata: Whether to include generation metadata in output.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.with_output(as_dataset=False)  # Returns (X, y) tuple
        """
        if as_dataset is not None:
            self.state.as_dataset = as_dataset

        if include_metadata is not None:
            self.state.include_metadata = include_metadata

        return self

    def _create_generator(self) -> SyntheticNIRSGenerator:
        """Create and configure the generator from current state."""
        # Build component library if needed
        library = self.state.component_library
        if library is None and self.state.component_names is not None:
            library = ComponentLibrary.from_predefined(
                self.state.component_names,
                random_state=self.state.random_state,
            )

        return SyntheticNIRSGenerator(
            wavelength_start=self.state.wavelength_start,
            wavelength_end=self.state.wavelength_end,
            wavelength_step=self.state.wavelength_step,
            component_library=library,
            complexity=self.state.complexity,
            random_state=self.state.random_state,
        )

    def _generate_data(self, generator: SyntheticNIRSGenerator) -> None:
        """Generate raw spectral data using the generator."""
        X, C, _E, metadata = generator.generate(
            n_samples=self.state.n_samples,
            concentration_method=self.state.concentration_method,
            include_batch_effects=self.state.batch_effects_enabled,
            n_batches=self.state.n_batches,
            return_metadata=True,
        )

        # Store wavelengths and concentrations
        self.state._wavelengths = generator.wavelengths.copy()
        self.state._metadata = metadata
        self.state._C = C

        # Generate sample metadata if requested
        if self.state.generate_sample_ids or self.state.n_groups is not None:
            self._generate_sample_metadata()

        # Process targets
        y = self._process_targets(C, generator)

        self.state._X = X
        self.state._y = y

    def _generate_sample_metadata(self) -> None:
        """Generate sample metadata using MetadataGenerator."""
        metadata_gen = MetadataGenerator(random_state=self.state.random_state)
        result = metadata_gen.generate(
            n_samples=self.state.n_samples,
            sample_id_prefix=self.state.sample_id_prefix,
            n_groups=self.state.n_groups,
            group_names=self.state.group_names,
            n_repetitions=self.state.n_repetitions,
        )
        self.state._sample_metadata = result

    def _process_targets(
        self,
        C: np.ndarray,
        generator: SyntheticNIRSGenerator,
    ) -> np.ndarray:
        """Process concentration matrix into target values."""
        # For classification, use the TargetGenerator
        if self.state.n_classes is not None:
            target_gen = TargetGenerator(random_state=self.state.random_state)
            y = target_gen.classification(
                n_samples=C.shape[0],
                concentrations=C,
                n_classes=self.state.n_classes,
                class_weights=self.state.class_weights,
                separation=self.state.class_separation,
                separation_method=self.state.class_separation_method,
            )
            return y

        # For regression, process as before
        # Select target component(s)
        if self.state.target_component is None:
            y = C
        elif isinstance(self.state.target_component, str):
            comp_idx = generator.library.component_names.index(self.state.target_component)
            y = C[:, comp_idx]
        else:
            y = C[:, self.state.target_component]

        # Ensure 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Apply transformation
        if self.state.target_transform == "log":
            y = np.log1p(y)
        elif self.state.target_transform == "sqrt":
            y = np.sqrt(y)

        # Apply range scaling
        if self.state.target_range is not None:
            min_val, max_val = self.state.target_range
            y_min, y_max = y.min(), y.max()
            if y_max > y_min:
                y = (y - y_min) / (y_max - y_min) * (max_val - min_val) + min_val
            else:
                y = np.full_like(y, (min_val + max_val) / 2)

        # Flatten if single target
        if y.shape[1] == 1:
            y = y.ravel()

        return y

    def _build_dataset(self) -> SpectroDataset:
        """Build SpectroDataset from generated data."""
        from nirs4all.data import SpectroDataset

        X = self.state._X
        y = self.state._y
        n_samples = self.state.n_samples
        train_ratio = self.state.train_ratio

        # Calculate partition sizes
        n_train = int(n_samples * train_ratio)
        n_test = n_samples - n_train

        # Create shuffle indices if needed
        rng = np.random.default_rng(self.state.random_state)
        if self.state.shuffle:
            indices = rng.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        # Create dataset
        dataset = SpectroDataset(name=self.state.name)

        # Create wavelength headers
        headers = [str(int(wl)) for wl in self.state._wavelengths]

        # Add training samples
        dataset.add_samples(
            X[train_indices],
            indexes={"partition": "train"},
            headers=headers,
            header_unit="nm",
        )
        if y.ndim == 1:
            dataset.add_targets(y[train_indices])
        else:
            dataset.add_targets(y[train_indices])

        # Add test samples if any
        if n_test > 0:
            dataset.add_samples(
                X[test_indices],
                indexes={"partition": "test"},
                headers=headers,
                header_unit="nm",
            )
            if y.ndim == 1:
                dataset.add_targets(y[test_indices])
            else:
                dataset.add_targets(y[test_indices])

        return dataset

    def _build_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build raw numpy arrays from generated data."""
        return self.state._X.copy(), self.state._y.copy()

    def build(self) -> Union[SpectroDataset, Tuple[np.ndarray, np.ndarray]]:
        """
        Build the synthetic dataset with all configured options.

        This method generates the data and returns it in the configured format.

        Returns:
            If as_dataset=True (default): SpectroDataset instance.
            If as_dataset=False: Tuple of (X, y) numpy arrays.

        Raises:
            RuntimeError: If build() was already called on this builder.

        Example:
            >>> dataset = builder.build()
            >>> print(dataset.num_samples)
            1000
        """
        if self._built:
            raise RuntimeError(
                "build() can only be called once per builder instance. "
                "Create a new builder for additional datasets."
            )

        # Check for multi-source generation
        if self.state.sources is not None:
            return self._build_multi_source()

        # Single-source generation
        generator = self._create_generator()
        self._generate_data(generator)

        self._built = True

        # Return in requested format
        if self.state.as_dataset:
            return self._build_dataset()
        else:
            return self._build_arrays()

    def _build_multi_source(self) -> Union[SpectroDataset, Tuple[np.ndarray, np.ndarray]]:
        """Build multi-source dataset using MultiSourceGenerator."""
        from .sources import MultiSourceGenerator

        generator = MultiSourceGenerator(random_state=self.state.random_state)

        if self.state.as_dataset:
            dataset = generator.create_dataset(
                n_samples=self.state.n_samples,
                sources=self.state.sources,
                train_ratio=self.state.train_ratio,
                target_range=self.state.target_range,
                name=self.state.name,
            )
            self._built = True
            return dataset
        else:
            result = generator.generate(
                n_samples=self.state.n_samples,
                sources=self.state.sources,
                target_range=self.state.target_range,
            )
            self._built = True
            X = result.get_combined_features()
            return X, result.targets

    def build_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build and return raw numpy arrays.

        This is a convenience method equivalent to calling
        ``with_output(as_dataset=False).build()``.

        Returns:
            Tuple of (X, y) numpy arrays.

        Example:
            >>> X, y = builder.build_arrays()
        """
        self.state.as_dataset = False
        return self.build()

    def build_dataset(self) -> SpectroDataset:
        """
        Build and return a SpectroDataset.

        This is a convenience method equivalent to calling
        ``with_output(as_dataset=True).build()``.

        Returns:
            SpectroDataset instance.

        Example:
            >>> dataset = builder.build_dataset()
        """
        self.state.as_dataset = True
        return self.build()

    def get_config(self) -> SyntheticDatasetConfig:
        """
        Get the current configuration as a SyntheticDatasetConfig object.

        Returns:
            SyntheticDatasetConfig with all current settings.

        Example:
            >>> config = builder.get_config()
            >>> print(config.n_samples)
            1000
        """
        return SyntheticDatasetConfig(
            n_samples=self.state.n_samples,
            random_state=self.state.random_state,
            features=FeatureConfig(
                wavelength_start=self.state.wavelength_start,
                wavelength_end=self.state.wavelength_end,
                wavelength_step=self.state.wavelength_step,
                complexity=self.state.complexity,
                component_names=self.state.component_names,
            ),
            targets=TargetConfig(
                distribution=self.state.concentration_method,
                range=self.state.target_range,
                transform=self.state.target_transform,
            ),
            metadata=MetadataConfig(
                generate_sample_ids=self.state.generate_sample_ids,
                sample_id_prefix=self.state.sample_id_prefix,
                n_groups=self.state.n_groups,
                n_repetitions=self.state.n_repetitions,
            ),
            partitions=PartitionConfig(
                train_ratio=self.state.train_ratio,
                stratify=self.state.stratify,
                shuffle=self.state.shuffle,
            ),
            batch_effects=BatchEffectConfig(
                enabled=self.state.batch_effects_enabled,
                n_batches=self.state.n_batches,
            ),
            output=OutputConfig(
                as_dataset=self.state.as_dataset,
                include_metadata=self.state.include_metadata,
            ),
            name=self.state.name,
        )

    @classmethod
    def from_config(
        cls,
        config: SyntheticDatasetConfig,
    ) -> SyntheticDatasetBuilder:
        """
        Create a builder from a SyntheticDatasetConfig object.

        Args:
            config: Configuration object to use.

        Returns:
            Configured SyntheticDatasetBuilder instance.

        Example:
            >>> config = SyntheticDatasetConfig(n_samples=500)
            >>> builder = SyntheticDatasetBuilder.from_config(config)
            >>> dataset = builder.build()
        """
        builder = cls(
            n_samples=config.n_samples,
            random_state=config.random_state,
            name=config.name,
        )

        # Apply feature config
        builder.state.wavelength_start = config.features.wavelength_start
        builder.state.wavelength_end = config.features.wavelength_end
        builder.state.wavelength_step = config.features.wavelength_step
        builder.state.complexity = config.features.complexity
        builder.state.component_names = config.features.component_names

        # Apply target config
        builder.state.concentration_method = config.targets.distribution
        builder.state.target_range = config.targets.range
        builder.state.target_transform = config.targets.transform

        # Apply metadata config
        builder.state.generate_sample_ids = config.metadata.generate_sample_ids
        builder.state.sample_id_prefix = config.metadata.sample_id_prefix
        builder.state.n_groups = config.metadata.n_groups
        builder.state.n_repetitions = config.metadata.n_repetitions

        # Apply partition config
        builder.state.train_ratio = config.partitions.train_ratio
        builder.state.stratify = config.partitions.stratify
        builder.state.shuffle = config.partitions.shuffle

        # Apply batch effect config
        builder.state.batch_effects_enabled = config.batch_effects.enabled
        builder.state.n_batches = config.batch_effects.n_batches

        # Apply output config
        builder.state.as_dataset = config.output.as_dataset
        builder.state.include_metadata = config.output.include_metadata

        return builder

    def export(
        self,
        path: Union[str, "Path"],
        format: Literal["standard", "single", "fragmented"] = "standard",
    ) -> "Path":
        """
        Generate data and export to folder.

        Generates the synthetic data and exports it to a folder structure
        compatible with nirs4all's DatasetConfigs loader.

        Args:
            path: Output folder path.
            format: Export format:
                - 'standard': Xcal, Ycal, Xval, Yval files.
                - 'single': All data in one file with partition column.
                - 'fragmented': Multiple small files (for testing).

        Returns:
            Path to created folder.

        Example:
            >>> builder = SyntheticDatasetBuilder(n_samples=1000)
            >>> path = builder.export("data/synthetic", format="standard")
        """
        from pathlib import Path
        from .exporter import DatasetExporter

        # Generate data if not already done
        if self.state._X is None:
            if self.state.sources is not None:
                # Multi-source - generate and export differently
                result = self._build_multi_source()
                if hasattr(result, 'x'):
                    # It's a dataset
                    X = result.x({}, layout='2d')
                    y = result.y({})
                else:
                    X, y = result
                wavelengths = None  # Multi-source doesn't have simple wavelengths
            else:
                generator = self._create_generator()
                self._generate_data(generator)
                X = self.state._X
                y = self.state._y
                wavelengths = self.state._wavelengths
        else:
            X = self.state._X
            y = self.state._y
            wavelengths = self.state._wavelengths

        # Export
        exporter = DatasetExporter()
        return exporter.to_folder(
            path,
            X, y,
            train_ratio=self.state.train_ratio,
            wavelengths=wavelengths,
            format=format,
            random_state=self.state.random_state,
        )

    def export_to_csv(
        self,
        path: Union[str, "Path"],
        include_targets: bool = True,
    ) -> "Path":
        """
        Generate data and export to a single CSV file.

        Args:
            path: Output file path.
            include_targets: Whether to include target column(s).

        Returns:
            Path to created file.

        Example:
            >>> path = builder.export_to_csv("data.csv")
        """
        from pathlib import Path
        from .exporter import DatasetExporter

        # Generate data if not already done
        if self.state._X is None:
            generator = self._create_generator()
            self._generate_data(generator)

        exporter = DatasetExporter()
        return exporter.to_csv(
            path,
            self.state._X,
            self.state._y,
            wavelengths=self.state._wavelengths,
            include_targets=include_targets,
        )

    def fit_to(
        self,
        template: Union[np.ndarray, "SpectroDataset"],
        wavelengths: Optional[np.ndarray] = None,
        *,
        match_statistics: bool = True,
        match_structure: bool = True,
    ) -> "SyntheticDatasetBuilder":
        """
        Configure builder to generate data similar to a template.

        Analyzes the template data and adjusts generation parameters
        to produce synthetic data with similar properties.

        Args:
            template: Real data to mimic (array or SpectroDataset).
            wavelengths: Wavelength grid (if template is array).
            match_statistics: Match statistical properties (mean, std).
            match_structure: Match PCA structure and complexity.

        Returns:
            Self for method chaining.

        Example:
            >>> builder = SyntheticDatasetBuilder(n_samples=1000)
            >>> builder.fit_to(X_real, wavelengths=wl)
            >>> X_synth, y = builder.build_arrays()
        """
        from .fitter import RealDataFitter

        fitter = RealDataFitter()
        params = fitter.fit(template, wavelengths=wavelengths)

        # Apply fitted wavelength range
        self.state.wavelength_start = params.wavelength_start
        self.state.wavelength_end = params.wavelength_end
        self.state.wavelength_step = params.wavelength_step

        # Apply complexity
        if match_structure:
            self.state.complexity = params.complexity

        return self

    def __repr__(self) -> str:
        """Return string representation of the builder."""
        return (
            f"SyntheticDatasetBuilder("
            f"n_samples={self.state.n_samples}, "
            f"complexity='{self.state.complexity}', "
            f"random_state={self.state.random_state})"
        )
