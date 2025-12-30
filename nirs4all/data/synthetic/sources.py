"""
Multi-source dataset generation for synthetic NIRS data.

This module provides tools for generating synthetic datasets with multiple
data sources, such as combining NIR spectra with molecular markers or
auxiliary measurements.

Example:
    >>> from nirs4all.data.synthetic.sources import MultiSourceGenerator
    >>>
    >>> generator = MultiSourceGenerator(random_state=42)
    >>>
    >>> dataset = generator.generate(
    ...     n_samples=500,
    ...     sources=[
    ...         {"name": "NIR_low", "type": "nir", "wavelength_range": (1000, 1700)},
    ...         {"name": "NIR_high", "type": "nir", "wavelength_range": (1700, 2500)},
    ...         {"name": "markers", "type": "aux", "n_features": 15},
    ...     ]
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from .generator import SyntheticNIRSGenerator
from .components import ComponentLibrary

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset


@dataclass
class SourceConfig:
    """
    Configuration for a single data source.

    Attributes:
        name: Unique identifier for the source.
        source_type: Type of source ('nir', 'vis', 'aux', 'markers').
        n_features: Number of features (auto-calculated for NIR sources).

        # NIR-specific
        wavelength_start: Start wavelength for NIR sources.
        wavelength_end: End wavelength for NIR sources.
        wavelength_step: Wavelength step for NIR sources.
        components: Component names for NIR sources.
        complexity: Complexity level for NIR sources.

        # Auxiliary-specific
        distribution: Distribution for auxiliary features.
        correlation_with_target: How correlated aux features are with target.
    """

    name: str
    source_type: Literal["nir", "vis", "aux", "markers"] = "nir"
    n_features: Optional[int] = None

    # NIR-specific options
    wavelength_start: Optional[float] = None
    wavelength_end: Optional[float] = None
    wavelength_step: float = 2.0
    components: Optional[List[str]] = None
    complexity: Literal["simple", "realistic", "complex"] = "simple"

    # Auxiliary-specific options
    distribution: Literal["normal", "uniform", "lognormal"] = "normal"
    correlation_with_target: float = 0.5

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> SourceConfig:
        """Create SourceConfig from dictionary."""
        # Handle wavelength_range shorthand
        if "wavelength_range" in config:
            wl_range = config.pop("wavelength_range")
            config["wavelength_start"] = wl_range[0]
            config["wavelength_end"] = wl_range[1]

        # Handle type -> source_type mapping
        if "type" in config:
            config["source_type"] = config.pop("type")

        return cls(**config)


@dataclass
class MultiSourceResult:
    """
    Container for multi-source generation results.

    Attributes:
        sources: Dictionary mapping source names to feature arrays.
        targets: Target values.
        source_configs: Source configuration objects.
        wavelengths: Dictionary mapping NIR source names to wavelength arrays.
        metadata: Optional metadata dictionary.
    """

    sources: Dict[str, np.ndarray]
    targets: np.ndarray
    source_configs: List[SourceConfig]
    wavelengths: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None

    @property
    def source_names(self) -> List[str]:
        """Get list of source names."""
        return list(self.sources.keys())

    def get_combined_features(self) -> np.ndarray:
        """Concatenate all sources into single feature matrix."""
        return np.hstack([self.sources[name] for name in self.source_names])

    @property
    def n_samples(self) -> int:
        """Get number of samples."""
        return len(self.targets)

    @property
    def n_features_total(self) -> int:
        """Get total number of features across all sources."""
        return sum(arr.shape[1] for arr in self.sources.values())


class MultiSourceGenerator:
    """
    Generate synthetic multi-source NIRS datasets.

    This class creates datasets combining multiple data sources, such as:
    - Multiple NIR spectral ranges (e.g., visible-NIR + shortwave-NIR)
    - NIR spectra + molecular markers
    - NIR spectra + auxiliary measurements

    The generated sources share common underlying structure through
    component concentrations, creating realistic inter-source correlations.

    Attributes:
        rng: NumPy random generator.

    Args:
        random_state: Random seed for reproducibility.

    Example:
        >>> generator = MultiSourceGenerator(random_state=42)
        >>>
        >>> result = generator.generate(
        ...     n_samples=500,
        ...     sources=[
        ...         {
        ...             "name": "NIR",
        ...             "type": "nir",
        ...             "wavelength_range": (1000, 2500),
        ...             "complexity": "realistic"
        ...         },
        ...         {
        ...             "name": "markers",
        ...             "type": "aux",
        ...             "n_features": 20,
        ...             "correlation_with_target": 0.7
        ...         }
        ...     ],
        ...     target_range=(0, 100)
        ... )
        >>>
        >>> print(result.source_names)
        ['NIR', 'markers']
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        """
        Initialize the multi-source generator.

        Args:
            random_state: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(random_state)
        self._random_state = random_state

    def generate(
        self,
        n_samples: int,
        sources: List[Union[SourceConfig, Dict[str, Any]]],
        *,
        target_range: Optional[Tuple[float, float]] = None,
        concentration_method: str = "dirichlet",
        n_components: int = 5,
    ) -> MultiSourceResult:
        """
        Generate a multi-source dataset.

        All sources share underlying component concentrations, which creates
        realistic correlations between sources. NIR sources generate spectra
        from these concentrations, while auxiliary sources create features
        correlated with the same underlying structure.

        Args:
            n_samples: Number of samples to generate.
            sources: List of source configurations (SourceConfig or dict).
            target_range: Optional (min, max) for scaling target values.
            concentration_method: Method for generating component concentrations.
            n_components: Number of underlying components.

        Returns:
            MultiSourceResult containing all generated data.

        Example:
            >>> result = generator.generate(
            ...     n_samples=300,
            ...     sources=[
            ...         {"name": "VIS-NIR", "type": "nir", "wavelength_range": (400, 1100)},
            ...         {"name": "SWIR", "type": "nir", "wavelength_range": (1100, 2500)},
            ...     ]
            ... )
        """
        # Parse source configurations
        parsed_sources = [
            s if isinstance(s, SourceConfig) else SourceConfig.from_dict(s.copy())
            for s in sources
        ]

        # Validate source names are unique
        names = [s.name for s in parsed_sources]
        if len(names) != len(set(names)):
            raise ValueError("Source names must be unique")

        # Generate shared component concentrations
        concentrations = self._generate_concentrations(
            n_samples, n_components, concentration_method
        )

        # Generate each source
        source_data: Dict[str, np.ndarray] = {}
        wavelengths: Dict[str, np.ndarray] = {}

        for source_config in parsed_sources:
            if source_config.source_type in ("nir", "vis"):
                X, wl = self._generate_nir_source(
                    n_samples, concentrations, source_config
                )
                source_data[source_config.name] = X
                wavelengths[source_config.name] = wl
            elif source_config.source_type in ("aux", "markers"):
                X = self._generate_aux_source(
                    n_samples, concentrations, source_config
                )
                source_data[source_config.name] = X
            else:
                raise ValueError(
                    f"Unknown source type: '{source_config.source_type}'"
                )

        # Generate targets from concentrations
        targets = self._generate_targets(concentrations, target_range)

        return MultiSourceResult(
            sources=source_data,
            targets=targets,
            source_configs=parsed_sources,
            wavelengths=wavelengths,
        )

    def _generate_concentrations(
        self,
        n_samples: int,
        n_components: int,
        method: str,
    ) -> np.ndarray:
        """Generate shared component concentrations."""
        if method == "dirichlet":
            alpha = np.ones(n_components) * 2.0
            return self.rng.dirichlet(alpha, size=n_samples)
        elif method == "uniform":
            return self.rng.uniform(0, 1, size=(n_samples, n_components))
        elif method == "lognormal":
            C = self.rng.lognormal(0, 0.5, size=(n_samples, n_components))
            return C / C.sum(axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown concentration method: '{method}'")

    def _generate_nir_source(
        self,
        n_samples: int,
        concentrations: np.ndarray,
        config: SourceConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate NIR spectral source."""
        # Use default wavelength range if not specified
        wl_start = config.wavelength_start or 1000
        wl_end = config.wavelength_end or 2500
        wl_step = config.wavelength_step

        # Create NIR generator for this source
        library = None
        if config.components:
            library = ComponentLibrary.from_predefined(
                config.components, random_state=self._random_state
            )

        generator = SyntheticNIRSGenerator(
            wavelength_start=wl_start,
            wavelength_end=wl_end,
            wavelength_step=wl_step,
            component_library=library,
            complexity=config.complexity,
            random_state=self._random_state,
        )

        # Generate spectra
        # Note: We use the shared concentrations, but the generator may have
        # different number of components, so we adapt
        n_gen_components = generator.library.n_components
        n_shared_components = concentrations.shape[1]

        if n_gen_components == n_shared_components:
            C_adapted = concentrations
        elif n_gen_components < n_shared_components:
            # Use subset of concentrations
            C_adapted = concentrations[:, :n_gen_components]
            C_adapted = C_adapted / C_adapted.sum(axis=1, keepdims=True)
        else:
            # Extend concentrations with noise
            extra = self.rng.dirichlet(
                np.ones(n_gen_components - n_shared_components) * 0.5,
                size=n_samples
            ) * 0.2
            C_adapted = np.hstack([concentrations * 0.8, extra])
            C_adapted = C_adapted / C_adapted.sum(axis=1, keepdims=True)

        # Generate using the generator's internal methods
        X = generator._apply_beer_lambert(C_adapted)
        X = generator._apply_path_length(X)
        X = X + generator._generate_baseline(n_samples)
        X = generator._apply_global_slope(X)
        X = generator._apply_scatter(X)
        X = generator._apply_wavelength_shift(X)
        X = generator._apply_instrumental_response(X)
        X = generator._add_noise(X)

        return X, generator.wavelengths.copy()

    def _generate_aux_source(
        self,
        n_samples: int,
        concentrations: np.ndarray,
        config: SourceConfig,
    ) -> np.ndarray:
        """Generate auxiliary/marker source."""
        n_features = config.n_features or 10
        correlation = config.correlation_with_target

        # Generate features correlated with concentrations
        # Use a linear combination of concentrations plus noise
        n_components = concentrations.shape[1]

        # Create mixing matrix
        mixing = self.rng.normal(0, 1, size=(n_components, n_features))

        # Base features from concentrations
        base_features = concentrations @ mixing

        # Add noise based on correlation level
        noise_std = np.sqrt((1 - correlation**2) / correlation**2) if correlation > 0 else 1.0
        noise = self.rng.normal(0, noise_std, size=(n_samples, n_features))

        X = base_features + noise * np.std(base_features)

        # Apply distribution transformation
        if config.distribution == "lognormal":
            X = np.exp((X - X.mean()) / X.std())
        elif config.distribution == "uniform":
            # Rank transform to uniform
            for j in range(n_features):
                ranks = np.argsort(np.argsort(X[:, j]))
                X[:, j] = ranks / (n_samples - 1)

        return X

    def _generate_targets(
        self,
        concentrations: np.ndarray,
        target_range: Optional[Tuple[float, float]],
    ) -> np.ndarray:
        """Generate target values from concentrations."""
        # Weighted combination of components
        weights = self.rng.dirichlet(np.ones(concentrations.shape[1]))
        y = concentrations @ weights

        # Scale to range
        if target_range is not None:
            min_val, max_val = target_range
            y_min, y_max = y.min(), y.max()
            if y_max > y_min:
                y = (y - y_min) / (y_max - y_min) * (max_val - min_val) + min_val
            else:
                y = np.full_like(y, (min_val + max_val) / 2)

        return y

    def create_dataset(
        self,
        n_samples: int,
        sources: List[Union[SourceConfig, Dict[str, Any]]],
        *,
        train_ratio: float = 0.8,
        target_range: Optional[Tuple[float, float]] = None,
        name: str = "multi_source_synthetic",
    ) -> SpectroDataset:
        """
        Create a SpectroDataset from multi-source generation.

        Args:
            n_samples: Number of samples to generate.
            sources: List of source configurations.
            train_ratio: Proportion of samples for training.
            target_range: Optional (min, max) for target scaling.
            name: Dataset name.

        Returns:
            SpectroDataset with multiple sources configured.

        Example:
            >>> dataset = generator.create_dataset(
            ...     n_samples=500,
            ...     sources=[
            ...         {"name": "NIR", "type": "nir", "wavelength_range": (1000, 2500)},
            ...         {"name": "markers", "type": "aux", "n_features": 10}
            ...     ],
            ...     train_ratio=0.8
            ... )
        """
        from nirs4all.data import SpectroDataset

        # Generate data
        result = self.generate(
            n_samples=n_samples,
            sources=sources,
            target_range=target_range,
        )

        # Create dataset
        dataset = SpectroDataset(name=name)

        # Calculate partition sizes
        n_train = int(n_samples * train_ratio)

        # Shuffle indices
        indices = self.rng.permutation(n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        # Prepare multi-source data and headers
        # Combine all sources into feature arrays (concatenated)
        X_combined = result.get_combined_features()

        # Get headers from first NIR source if available, else simple feature names
        headers = None
        header_unit = None
        for source_name in result.source_names:
            if source_name in result.wavelengths:
                headers = [str(int(wl)) for wl in result.wavelengths[source_name]]
                header_unit = "nm"
                break

        if headers is None:
            headers = [f"feature_{i}" for i in range(X_combined.shape[1])]

        # Add training samples
        dataset.add_samples(
            X_combined[train_indices],
            indexes={"partition": "train"},
            headers=headers,
            header_unit=header_unit,
        )

        # Add test samples
        if len(test_indices) > 0:
            dataset.add_samples(
                X_combined[test_indices],
                indexes={"partition": "test"},
                headers=headers,
                header_unit=header_unit,
            )

        # Add targets
        y = result.targets
        dataset.add_targets(y[train_indices])
        if len(test_indices) > 0:
            dataset.add_targets(y[test_indices])

        return dataset


def generate_multi_source(
    n_samples: int,
    sources: Optional[List[Dict[str, Any]]] = None,
    *,
    random_state: Optional[int] = None,
    target_range: Optional[Tuple[float, float]] = None,
    as_dataset: bool = True,
    train_ratio: float = 0.8,
    name: str = "multi_source_synthetic",
) -> Union[SpectroDataset, MultiSourceResult]:
    """
    Convenience function for generating multi-source datasets.

    Args:
        n_samples: Number of samples.
        sources: List of source configurations. If None, uses default
            single NIR source with wavelength range (1000, 2500).
        random_state: Random seed.
        target_range: Target value range.
        as_dataset: If True, returns SpectroDataset.
        train_ratio: Training set proportion.
        name: Dataset name.

    Returns:
        SpectroDataset or MultiSourceResult depending on as_dataset.

    Example:
        >>> dataset = generate_multi_source(
        ...     n_samples=500,
        ...     sources=[
        ...         {"name": "NIR", "type": "nir", "wavelength_range": (1000, 2500)},
        ...         {"name": "markers", "type": "aux", "n_features": 15}
        ...     ],
        ...     random_state=42
        ... )
    """
    # Default sources if none provided
    if sources is None:
        sources = [{"name": "NIR", "type": "nir", "wavelength_range": (1000, 2500)}]

    generator = MultiSourceGenerator(random_state=random_state)

    if as_dataset:
        return generator.create_dataset(
            n_samples=n_samples,
            sources=sources,
            train_ratio=train_ratio,
            target_range=target_range,
            name=name,
        )
    else:
        return generator.generate(
            n_samples=n_samples,
            sources=sources,
            target_range=target_range,
        )
