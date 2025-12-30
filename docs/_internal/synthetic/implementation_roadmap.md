# Synthetic Generator - Implementation Roadmap

**Version**: 1.5
**Status**: Phase 6 Complete - All Phases Done
**Created**: 2024-12-30
**Last Updated**: 2025-01-02

---

## Implementation Status

| Phase | Status | Completion Date | Notes |
|-------|--------|-----------------|-------|
| Phase 1: Core Module Migration | ✅ **Complete** | 2024-12-30 | 98% test coverage achieved |
| Phase 2: API Integration | ✅ **Complete** | 2024-12-30 | 93 new tests, all passing |
| Phase 3: Feature Enhancements | ✅ **Complete** | 2024-12-30 | 237 tests, metadata/targets/sources modules |
| Phase 4: Export & Fitting | ✅ **Complete** | 2024-12-31 | 287 tests, exporter/fitter modules |
| Phase 5: Test Migration | ✅ **Complete** | 2024-12-30 | 122 new tests, pytest fixtures in conftest.py |
| Phase 6: Documentation | ✅ **Complete** | 2025-01-02 | User/developer guides, examples, API reference |

---

## Design Decisions (Confirmed)

| Decision | Resolution | Rationale |
|----------|------------|-----------|
| **API Name** | `nirs4all.generate()` | Standard ML naming (sklearn: `make_regression`, etc.) |
| **Default Complexity** | `"simple"` for unit tests, `"realistic"` for integration | Balance speed vs realism; max 5s overhead |
| **Visualization** | Merge into `nirs4all.visualization` | Spectra are spectra - unified plotting API |
| **Implementation** | Single-shot implementation | All phases implemented together |

---

## Overview

This document provides a detailed implementation roadmap for integrating and enhancing the synthetic NIRS data generator into nirs4all. It complements the [main specification](./synthetic_generator_specification.md) with actionable tasks, code patterns, and acceptance criteria.

> **Implementation Note**: All phases will be implemented together in a single-shot approach.

---

## Phase 1: Core Module Migration (Week 1-2)

### 1.1 Create Module Structure

**Task**: Set up the synthetic module within nirs4all

```
nirs4all/data/synthetic/
├── __init__.py           # Public API exports
├── generator.py          # Core SyntheticNIRSGenerator
├── components.py         # NIRBand, SpectralComponent, ComponentLibrary
├── config.py             # Configuration dataclasses
├── builder.py            # SyntheticDatasetBuilder
├── targets.py            # TargetGenerator
├── metadata.py           # MetadataGenerator
├── sources.py            # MultiSourceGenerator
├── exporter.py           # DatasetExporter
├── fitter.py             # RealDataFitter
├── validation.py         # Data validation utilities
└── _constants.py         # Predefined components, defaults

nirs4all/visualization/
└── synthetic.py          # Synthetic spectra visualization (merged into existing module)
```

**Implementation Steps**:

1. Create directory structure
2. Move `NIRBand`, `SpectralComponent`, `ComponentLibrary` to `components.py`
3. Move `PREDEFINED_COMPONENTS` to `_constants.py`
4. Move `SyntheticNIRSGenerator` to `generator.py`
5. Create `config.py` with dataclasses from specification
6. Update imports and ensure backward compatibility

**Files to Create**:

#### `nirs4all/data/synthetic/__init__.py`

```python
"""
Synthetic NIRS Data Generation Module.

This module provides tools for generating realistic synthetic NIRS spectra
for testing, examples, benchmarking, and ML research.

Quick Start:
    >>> import nirs4all
    >>> dataset = nirs4all.generate(n_samples=1000, random_state=42)

    >>> # Or use the builder for more control
    >>> from nirs4all.data.synthetic import SyntheticDatasetBuilder
    >>> dataset = (
    ...     SyntheticDatasetBuilder(n_samples=500)
    ...     .with_features(complexity="realistic")
    ...     .with_metadata(n_groups=3)
    ...     .build()
    ... )

See Also:
    - nirs4all.generate: Top-level generation API
    - SyntheticDatasetBuilder: Fluent builder interface
    - SyntheticNIRSGenerator: Core generation engine
"""

from .generator import SyntheticNIRSGenerator
from .components import (
    NIRBand,
    SpectralComponent,
    ComponentLibrary,
    PREDEFINED_COMPONENTS,
)
from .config import (
    SyntheticDatasetConfig,
    FeatureConfig,
    TargetConfig,
    MetadataConfig,
    SourceConfig,
    OutputConfig,
)
from .builder import SyntheticDatasetBuilder
from .targets import TargetGenerator
from .metadata import MetadataGenerator
from .exporter import DatasetExporter, CSVVariationGenerator
from .fitter import RealDataFitter, FittedParameters

__all__ = [
    # Core generator
    "SyntheticNIRSGenerator",
    # Components
    "NIRBand",
    "SpectralComponent",
    "ComponentLibrary",
    "PREDEFINED_COMPONENTS",
    # Configuration
    "SyntheticDatasetConfig",
    "FeatureConfig",
    "TargetConfig",
    "MetadataConfig",
    "SourceConfig",
    "OutputConfig",
    # Builder
    "SyntheticDatasetBuilder",
    # Generators
    "TargetGenerator",
    "MetadataGenerator",
    # Export
    "DatasetExporter",
    "CSVVariationGenerator",
    # Fitting
    "RealDataFitter",
    "FittedParameters",
]
```

### 1.2 Refactor Generator

**Task**: Clean up and enhance the core generator

**Changes Required**:

1. **Type Hints**: Add complete type annotations
2. **Docstrings**: Ensure Google-style docstrings on all methods
3. **Error Handling**: Add input validation and meaningful errors
4. **Configuration**: Use dataclasses instead of dict parameters
5. **Logging**: Add debug logging for generation steps

**Code Pattern**:

```python
# generator.py

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np

from nirs4all.core.logging import get_logger
from .components import ComponentLibrary, PREDEFINED_COMPONENTS
from .config import FeatureConfig

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset

logger = get_logger(__name__)


class SyntheticNIRSGenerator:
    """
    Generate synthetic NIRS spectra with realistic instrumental effects.

    This generator implements a physically-motivated model based on Beer-Lambert
    law with additional effects for baseline, scattering, instrumental response,
    and noise.

    Args:
        wavelength_start: Start wavelength in nm.
        wavelength_end: End wavelength in nm.
        wavelength_step: Wavelength step in nm.
        component_library: Optional ComponentLibrary. If None, uses defaults.
        complexity: Complexity level ("simple", "realistic", "complex").
        random_state: Random seed for reproducibility.

    Attributes:
        wavelengths: Array of wavelength values.
        n_wavelengths: Number of wavelength points.
        library: Component library used for generation.

    Example:
        >>> generator = SyntheticNIRSGenerator(random_state=42)
        >>> X, Y, E = generator.generate(n_samples=1000)
        >>> print(f"Generated {X.shape[0]} samples with {X.shape[1]} features")

    See Also:
        SyntheticDatasetBuilder: Higher-level builder interface.
        nirs4all.generate: Top-level API function.
    """

    def __init__(
        self,
        wavelength_start: float = 1000,
        wavelength_end: float = 2500,
        wavelength_step: float = 2,
        component_library: Optional[ComponentLibrary] = None,
        complexity: str = "realistic",
        random_state: Optional[int] = None,
    ) -> None:
        # Validate inputs
        if wavelength_start >= wavelength_end:
            raise ValueError(
                f"wavelength_start ({wavelength_start}) must be less than "
                f"wavelength_end ({wavelength_end})"
            )
        if complexity not in ("simple", "realistic", "complex"):
            raise ValueError(
                f"complexity must be 'simple', 'realistic', or 'complex', "
                f"got '{complexity}'"
            )

        # ... rest of initialization
        logger.debug(
            f"Initialized generator: wavelengths={wavelength_start}-{wavelength_end}nm, "
            f"complexity={complexity}"
        )
```

### 1.3 Unit Tests for Core Module

**Task**: Achieve 80%+ test coverage for core module

**Test Files**:

```
tests/unit/data/synthetic/
├── __init__.py
├── test_generator.py
├── test_components.py
├── test_config.py
└── conftest.py
```

**Test Pattern**:

```python
# tests/unit/data/synthetic/test_generator.py

import pytest
import numpy as np
from nirs4all.data.synthetic import SyntheticNIRSGenerator


class TestSyntheticNIRSGenerator:
    """Tests for SyntheticNIRSGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a generator with fixed random state."""
        return SyntheticNIRSGenerator(random_state=42)

    def test_init_default_parameters(self, generator):
        """Test initialization with default parameters."""
        assert generator.wavelength_start == 1000
        assert generator.wavelength_end == 2500
        assert generator.n_wavelengths > 0
        assert generator.library is not None

    def test_init_invalid_wavelength_range(self):
        """Test that invalid wavelength range raises error."""
        with pytest.raises(ValueError, match="wavelength_start"):
            SyntheticNIRSGenerator(wavelength_start=2500, wavelength_end=1000)

    def test_init_invalid_complexity(self):
        """Test that invalid complexity raises error."""
        with pytest.raises(ValueError, match="complexity"):
            SyntheticNIRSGenerator(complexity="invalid")

    def test_generate_output_shapes(self, generator):
        """Test that generate returns correct shapes."""
        X, Y, E = generator.generate(n_samples=100)

        assert X.shape == (100, generator.n_wavelengths)
        assert Y.shape[0] == 100
        assert E.shape[1] == generator.n_wavelengths

    def test_generate_reproducibility(self):
        """Test that same random_state produces same output."""
        gen1 = SyntheticNIRSGenerator(random_state=42)
        gen2 = SyntheticNIRSGenerator(random_state=42)

        X1, Y1, _ = gen1.generate(n_samples=50)
        X2, Y2, _ = gen2.generate(n_samples=50)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(Y1, Y2)

    def test_generate_different_complexity_levels(self, generator):
        """Test that different complexity levels produce different noise."""
        gen_simple = SyntheticNIRSGenerator(complexity="simple", random_state=42)
        gen_complex = SyntheticNIRSGenerator(complexity="complex", random_state=42)

        X_simple, _, _ = gen_simple.generate(n_samples=100)
        X_complex, _, _ = gen_complex.generate(n_samples=100)

        # Complex should have higher variance
        assert X_complex.std() > X_simple.std()

    @pytest.mark.parametrize("method", ["dirichlet", "uniform", "lognormal", "correlated"])
    def test_concentration_methods(self, generator, method):
        """Test all concentration generation methods."""
        X, Y, _ = generator.generate(n_samples=50, concentration_method=method)

        assert X.shape[0] == 50
        assert Y.shape[0] == 50
        assert np.all(Y >= 0)  # Concentrations should be non-negative

    def test_create_dataset_returns_spectrodataset(self, generator):
        """Test create_dataset returns proper SpectroDataset."""
        dataset = generator.create_dataset(n_train=80, n_test=20)

        from nirs4all.data import SpectroDataset
        assert isinstance(dataset, SpectroDataset)

        X_train = dataset.x({"partition": "train"})
        X_test = dataset.x({"partition": "test"})

        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
```

### 1.4 Deprecation in bench/synthetic

**Task**: Maintain backward compatibility while deprecating old location

```python
# bench/synthetic/__init__.py (updated)

"""
DEPRECATED: This module has moved to nirs4all.data.synthetic.

This location is kept for backward compatibility but will be removed in v1.0.
Please update your imports:

    # Old (deprecated)
    from bench.synthetic import SyntheticNIRSGenerator

    # New (recommended)
    from nirs4all.data.synthetic import SyntheticNIRSGenerator

    # Or use the top-level API
    import nirs4all
    dataset = nirs4all.generate(n_samples=1000)
"""

import warnings

def _warn_deprecated():
    warnings.warn(
        "Importing from bench.synthetic is deprecated. "
        "Use nirs4all.data.synthetic instead.",
        DeprecationWarning,
        stacklevel=3
    )

# Re-export from new location with deprecation warning
from nirs4all.data.synthetic import (
    SyntheticNIRSGenerator,
    ComponentLibrary,
    NIRBand,
    PREDEFINED_COMPONENTS,
)

_warn_deprecated()
```

### 1.5 Acceptance Criteria - Phase 1

- [x] All classes moved to `nirs4all/data/synthetic/`
- [x] Type hints on all public methods
- [x] Google-style docstrings complete
- [x] Unit test coverage >= 80% (achieved: **98%**)
- [x] `bench/synthetic` imports work with deprecation warning
- [x] All existing functionality preserved

**Phase 1 Completion Notes** (2024-12-30):
- Core module fully migrated to `nirs4all/data/synthetic/`
- 108 unit tests with 98% coverage
- All 3615 project tests pass
- Deprecation warning properly emitted from `bench/synthetic`
- Ready to proceed to Phase 2: API Integration

---

## Phase 2: API Integration (Week 2-3)

### 2.1 Implement SyntheticDatasetBuilder

**Task**: Create the fluent builder interface

**File**: `nirs4all/data/synthetic/builder.py`

```python
"""Fluent builder for synthetic dataset construction."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

from nirs4all.data.dataset import SpectroDataset
from .generator import SyntheticNIRSGenerator
from .components import ComponentLibrary
from .config import (
    SyntheticDatasetConfig,
    FeatureConfig,
    TargetConfig,
    MetadataConfig,
    SourceConfig,
)


class SyntheticDatasetBuilder:
    """
    Fluent builder for complex synthetic dataset configurations.

    This builder provides a chainable API for constructing synthetic
    datasets with custom features, targets, metadata, and export options.

    Args:
        n_samples: Total number of samples to generate.
        task_type: Task type ("regression", "binary", "multiclass").
        random_state: Random seed for reproducibility.

    Example:
        >>> dataset = (
        ...     SyntheticDatasetBuilder(n_samples=1000, random_state=42)
        ...     .with_features(
        ...         wavelength_range=(1000, 2500),
        ...         complexity="realistic"
        ...     )
        ...     .with_targets(
        ...         distribution="lognormal",
        ...         range=(5, 50)
        ...     )
        ...     .with_metadata(
        ...         n_groups=3,
        ...         n_repetitions=(2, 5)
        ...     )
        ...     .with_partitions(train_ratio=0.8)
        ...     .build()
        ... )
    """

    def __init__(
        self,
        n_samples: int = 1000,
        task_type: str = "regression",
        random_state: Optional[int] = None,
    ) -> None:
        self._n_samples = n_samples
        self._task_type = task_type
        self._random_state = random_state
        self._rng = np.random.default_rng(random_state)

        # Configuration containers
        self._feature_config = FeatureConfig()
        self._target_config = TargetConfig()
        self._metadata_config = MetadataConfig()
        self._source_configs: List[SourceConfig] = []
        self._partition_config: Dict[str, float] = {}

        # Fitting template
        self._template = None
        self._template_options: Dict[str, bool] = {}

    def with_features(
        self,
        wavelength_range: Tuple[float, float] = (1000, 2500),
        wavelength_step: float = 2,
        complexity: str = "realistic",
        components: Optional[List[str]] = None,
        include_baseline: bool = True,
        include_scatter: bool = True,
        noise_level: Optional[float] = None,
        batch_effects: bool = False,
        n_batches: int = 1,
    ) -> SyntheticDatasetBuilder:
        """
        Configure spectral feature generation.

        Args:
            wavelength_range: (start, end) wavelength in nm.
            wavelength_step: Step between wavelengths in nm.
            complexity: Generation complexity level.
            components: List of component names to use. None for defaults.
            include_baseline: Add baseline drift.
            include_scatter: Add scatter effects.
            noise_level: Override noise level (None = use complexity default).
            batch_effects: Include batch/session effects.
            n_batches: Number of batches if batch_effects=True.

        Returns:
            Self for method chaining.
        """
        self._feature_config = FeatureConfig(
            wavelength_start=wavelength_range[0],
            wavelength_end=wavelength_range[1],
            wavelength_step=wavelength_step,
            complexity=complexity,
            components=components,
            include_baseline=include_baseline,
            include_scatter=include_scatter,
            noise_level=noise_level,
        )
        self._batch_effects = batch_effects
        self._n_batches = n_batches
        return self

    def with_targets(
        self,
        n_targets: int = 1,
        distribution: str = "uniform",
        range: Tuple[float, float] = (0, 100),
        correlation: float = 0.8,
        noise: float = 0.1,
        n_classes: int = 2,
        class_balance: Optional[List[float]] = None,
    ) -> SyntheticDatasetBuilder:
        """
        Configure target variable generation.

        Args:
            n_targets: Number of target variables (for multi-output).
            distribution: Target distribution type.
            range: (min, max) target value range.
            correlation: How strongly features predict target (0-1).
            noise: Noise level in target values.
            n_classes: Number of classes (classification only).
            class_balance: Class proportions (classification only).

        Returns:
            Self for method chaining.
        """
        self._target_config = TargetConfig(
            n_targets=n_targets,
            distribution=distribution,
            range=range,
            correlation_with_features=correlation,
            n_classes=n_classes,
            class_balance=class_balance,
        )
        return self

    def with_metadata(
        self,
        n_groups: int = 0,
        group_names: Optional[List[str]] = None,
        n_repetitions: Union[int, Tuple[int, int]] = 1,
        extra_columns: Optional[Dict[str, Any]] = None,
    ) -> SyntheticDatasetBuilder:
        """
        Configure metadata generation.

        Args:
            n_groups: Number of sample groups (e.g., farms, sites).
            group_names: Names for groups. Auto-generated if None.
            n_repetitions: Repetitions per biological sample.
                Can be int (fixed) or (min, max) tuple (random).
            extra_columns: Additional metadata columns to generate.

        Returns:
            Self for method chaining.
        """
        if isinstance(n_repetitions, int):
            n_repetitions = (n_repetitions, n_repetitions)

        self._metadata_config = MetadataConfig(
            n_groups=n_groups,
            group_names=group_names,
            n_repetitions=n_repetitions,
            extra_columns=extra_columns,
        )
        return self

    def with_sources(
        self,
        sources: List[Dict[str, Any]],
    ) -> SyntheticDatasetBuilder:
        """
        Configure multi-source generation.

        Args:
            sources: List of source configurations, each containing:
                - name: Source name (required)
                - type: "nir", "vis", "markers", or "aux"
                - wavelength_range: (start, end) for spectral sources
                - n_features: Number of features

        Returns:
            Self for method chaining.

        Example:
            >>> builder.with_sources([
            ...     {"name": "NIR", "type": "nir", "wavelength_range": (1000, 2000)},
            ...     {"name": "markers", "type": "aux", "n_features": 15}
            ... ])
        """
        self._source_configs = [
            SourceConfig(**source) for source in sources
        ]
        return self

    def with_partitions(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.0,
        stratify: bool = False,
    ) -> SyntheticDatasetBuilder:
        """
        Configure train/val/test partitioning.

        Args:
            train_ratio: Proportion of samples for training.
            val_ratio: Proportion for validation (0 = no validation set).
            stratify: Stratify by target (classification) or group.

        Returns:
            Self for method chaining.
        """
        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio < 0:
            raise ValueError("train_ratio + val_ratio must be <= 1.0")

        self._partition_config = {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "stratify": stratify,
        }
        return self

    def fit_to(
        self,
        template: Union[str, SpectroDataset, np.ndarray],
        match_statistics: bool = True,
        match_structure: bool = True,
    ) -> SyntheticDatasetBuilder:
        """
        Fit generation parameters to match a real dataset.

        Args:
            template: Real data to mimic (path, SpectroDataset, or array).
            match_statistics: Match mean, std, noise characteristics.
            match_structure: Match PCA structure and patterns.

        Returns:
            Self for method chaining.
        """
        self._template = template
        self._template_options = {
            "match_statistics": match_statistics,
            "match_structure": match_structure,
        }
        return self

    def build(self) -> SpectroDataset:
        """
        Build and return a SpectroDataset.

        Returns:
            Configured SpectroDataset with generated data.
        """
        # Generate raw data
        X, y, metadata = self._generate_data()

        # Create dataset
        dataset = self._create_dataset(X, y, metadata)

        return dataset

    def build_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build and return (X, y) numpy arrays.

        Returns:
            Tuple of (features, targets) arrays.
        """
        X, y, _ = self._generate_data()
        return X, y

    def export(
        self,
        path: Union[str, Path],
        format: str = "standard",
    ) -> Path:
        """
        Build and export to files.

        Args:
            path: Output folder path.
            format: File layout format.

        Returns:
            Path to created folder.
        """
        from .exporter import DatasetExporter

        X, y, metadata = self._generate_data()
        exporter = DatasetExporter(
            X=X,
            y=y,
            wavelengths=self._get_wavelengths(),
            metadata=metadata,
            partitions=self._generate_partitions(len(X)),
        )
        return exporter.to_folder(path, format=format)

    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Generate all data components."""
        # Apply template fitting if specified
        if self._template is not None:
            self._apply_template_fitting()

        # Generate features
        X, concentrations = self._generate_features()

        # Generate targets
        y = self._generate_targets(concentrations)

        # Generate metadata
        metadata = self._generate_metadata()

        return X, y, metadata

    # ... (additional private methods)
```

### 2.2 Create Top-Level API

**Task**: Implement `nirs4all.generate()` and convenience functions

**File**: `nirs4all/api/generate.py`

```python
"""
Top-level generate() API for synthetic data generation.

This module provides the primary entry points for generating synthetic
NIRS datasets within nirs4all.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pathlib import Path
import numpy as np

from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.synthetic import SyntheticDatasetBuilder


def generate(
    n_samples: int = 1000,
    *,
    task_type: Literal["regression", "binary", "multiclass"] = "regression",
    n_features: int = 751,
    n_targets: int = 1,
    complexity: Literal["simple", "realistic", "complex"] = "realistic",
    random_state: Optional[int] = None,
    output: Literal["dataset", "arrays"] = "dataset",
    **kwargs: Any,
) -> Union[SpectroDataset, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate a synthetic NIRS dataset.

    This is the primary entry point for synthetic data generation in nirs4all.
    For simple use cases, call directly. For complex configurations, use
    the builder via `generate.builder()`.

    Args:
        n_samples: Total number of samples to generate.
        task_type: Type of ML task.
        n_features: Number of spectral features (wavelengths).
        n_targets: Number of target variables (regression only).
        complexity: Noise and effect complexity level.
        random_state: Random seed for reproducibility.
        output: Output format ("dataset" or "arrays").
        **kwargs: Additional configuration passed to builder.

    Returns:
        SpectroDataset or (X, y) tuple depending on output parameter.

    Example:
        >>> import nirs4all
        >>> dataset = nirs4all.generate(n_samples=1000, random_state=42)
        >>> X, y = nirs4all.generate(n_samples=500, output="arrays")
    """
    # Create builder
    builder = SyntheticDatasetBuilder(
        n_samples=n_samples,
        task_type=task_type,
        random_state=random_state,
    )

    # Apply feature configuration
    wavelength_range = kwargs.pop("wavelength_range", None)
    if wavelength_range is None:
        # Calculate range to achieve desired n_features
        step = kwargs.get("wavelength_step", 2)
        wavelength_range = (1000, 1000 + (n_features - 1) * step)

    builder.with_features(
        wavelength_range=wavelength_range,
        complexity=complexity,
        **{k: v for k, v in kwargs.items() if k in (
            "wavelength_step", "components", "include_baseline",
            "include_scatter", "noise_level", "batch_effects", "n_batches"
        )}
    )

    # Apply target configuration
    if task_type == "regression":
        builder.with_targets(
            n_targets=n_targets,
            **{k: v for k, v in kwargs.items() if k in (
                "distribution", "range", "correlation", "noise"
            )}
        )
    else:
        builder.with_targets(
            n_classes=kwargs.get("n_classes", 2 if task_type == "binary" else 3),
            class_balance=kwargs.get("class_balance"),
        )

    # Apply metadata if specified
    if any(k in kwargs for k in ("n_groups", "n_repetitions", "group_names")):
        builder.with_metadata(
            n_groups=kwargs.get("n_groups", 0),
            group_names=kwargs.get("group_names"),
            n_repetitions=kwargs.get("n_repetitions", 1),
        )

    # Apply partitions if specified
    if any(k in kwargs for k in ("train_ratio", "val_ratio")):
        builder.with_partitions(
            train_ratio=kwargs.get("train_ratio", 0.8),
            val_ratio=kwargs.get("val_ratio", 0.0),
        )

    # Build output
    if output == "arrays":
        return builder.build_arrays()
    else:
        return builder.build()


# Module-level convenience functions accessible as generate.function()
class _GenerateModule:
    """
    Namespace for generate convenience functions.

    This allows calling:
        - nirs4all.generate.regression(...)
        - nirs4all.generate.classification(...)
        - nirs4all.generate.builder(...)
    """

    def __call__(self, *args, **kwargs):
        """Allow calling as nirs4all.generate(...)."""
        return generate(*args, **kwargs)

    @staticmethod
    def regression(
        n_samples: int = 1000,
        n_targets: int = 1,
        target_range: Tuple[float, float] = (0.0, 100.0),
        target_distribution: str = "uniform",
        **kwargs: Any,
    ) -> Union[SpectroDataset, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate a regression dataset.

        Args:
            n_samples: Number of samples.
            n_targets: Number of target variables.
            target_range: (min, max) range for targets.
            target_distribution: Distribution type for targets.
            **kwargs: Additional arguments passed to generate().

        Returns:
            SpectroDataset or arrays.
        """
        return generate(
            n_samples=n_samples,
            task_type="regression",
            n_targets=n_targets,
            range=target_range,
            distribution=target_distribution,
            **kwargs
        )

    @staticmethod
    def classification(
        n_samples: int = 1000,
        n_classes: int = 2,
        class_balance: Optional[List[float]] = None,
        class_separation: float = 0.8,
        **kwargs: Any,
    ) -> Union[SpectroDataset, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate a classification dataset.

        Args:
            n_samples: Number of samples.
            n_classes: Number of classes.
            class_balance: Proportion per class (sums to 1).
            class_separation: How separable classes are (0-1).
            **kwargs: Additional arguments passed to generate().

        Returns:
            SpectroDataset or arrays.
        """
        task_type = "binary" if n_classes == 2 else "multiclass"
        return generate(
            n_samples=n_samples,
            task_type=task_type,
            n_classes=n_classes,
            class_balance=class_balance,
            class_separation=class_separation,
            **kwargs
        )

    @staticmethod
    def multi_source(
        n_samples: int = 1000,
        sources: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> SpectroDataset:
        """
        Generate a multi-source dataset.

        Args:
            n_samples: Number of samples.
            sources: List of source configurations.
            **kwargs: Additional arguments.

        Returns:
            SpectroDataset with multiple sources.
        """
        if sources is None:
            sources = [
                {"name": "NIR", "type": "nir", "wavelength_range": (1000, 2500)},
            ]

        builder = SyntheticDatasetBuilder(
            n_samples=n_samples,
            random_state=kwargs.get("random_state"),
        )
        builder.with_sources(sources)

        if "train_ratio" in kwargs:
            builder.with_partitions(train_ratio=kwargs["train_ratio"])

        return builder.build()

    @staticmethod
    def from_template(
        template: Union[str, SpectroDataset, np.ndarray],
        n_samples: int = 1000,
        match_statistics: bool = True,
        match_structure: bool = True,
        **kwargs: Any,
    ) -> SpectroDataset:
        """
        Generate data mimicking a real dataset template.

        Args:
            template: Real data to mimic.
            n_samples: Number of samples to generate.
            match_statistics: Match statistical properties.
            match_structure: Match PCA structure.
            **kwargs: Additional arguments.

        Returns:
            SpectroDataset with similar properties to template.
        """
        builder = SyntheticDatasetBuilder(
            n_samples=n_samples,
            random_state=kwargs.get("random_state"),
        )
        builder.fit_to(
            template,
            match_statistics=match_statistics,
            match_structure=match_structure,
        )
        return builder.build()

    @staticmethod
    def to_folder(
        path: Union[str, Path],
        n_samples: int = 1000,
        train_ratio: float = 0.8,
        format: Literal["standard", "single", "fragmented"] = "standard",
        **kwargs: Any,
    ) -> Path:
        """
        Generate and export dataset to folder.

        Args:
            path: Output folder path.
            n_samples: Number of samples.
            train_ratio: Train/test split ratio.
            format: File layout format.
            **kwargs: Arguments passed to generate().

        Returns:
            Path to created folder.
        """
        builder = SyntheticDatasetBuilder(
            n_samples=n_samples,
            random_state=kwargs.get("random_state"),
        )
        builder.with_partitions(train_ratio=train_ratio)

        if "complexity" in kwargs:
            builder.with_features(complexity=kwargs["complexity"])

        return builder.export(path, format=format)

    @staticmethod
    def to_csv(
        path: Union[str, Path],
        n_samples: int = 1000,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> Path:
        """
        Generate and export to single CSV file.

        Args:
            path: Output file path.
            n_samples: Number of samples.
            include_metadata: Include metadata columns.
            **kwargs: Additional arguments.

        Returns:
            Path to created file.
        """
        return _GenerateModule.to_folder(
            path, n_samples, format="single", **kwargs
        )

    @staticmethod
    def builder(**kwargs: Any) -> SyntheticDatasetBuilder:
        """
        Get a builder for advanced customization.

        Args:
            **kwargs: Arguments passed to SyntheticDatasetBuilder.

        Returns:
            SyntheticDatasetBuilder instance.
        """
        return SyntheticDatasetBuilder(**kwargs)


# Replace module with callable class instance
import sys
sys.modules[__name__] = _GenerateModule()
```

### 2.3 Update nirs4all.__init__.py

**Task**: Expose generate in public API

```python
# nirs4all/__init__.py (additions)

# Add generate to imports
from .api import generate

# Update __all__
__all__ = [
    # ... existing exports
    "generate",
]
```

### 2.4 Acceptance Criteria - Phase 2

- [x] `nirs4all.generate()` works as documented
- [x] `nirs4all.generate.regression()` works
- [x] `nirs4all.generate.classification()` works
- [x] `nirs4all.generate.builder()` returns builder
- [x] Integration with `nirs4all.run()` works
- [x] API tests pass

**Phase 2 Completion Notes** (2024-12-30):
- Created `SyntheticDatasetBuilder` with full fluent API in `nirs4all/data/synthetic/builder.py`
- Created `nirs4all.generate()` top-level API with convenience methods (regression, classification, builder)
- `generate` is both callable and a namespace: `nirs4all.generate(...)` and `nirs4all.generate.regression(...)`
- 60 builder unit tests + 33 generate API tests = 93 new tests, all passing
- Full integration with `nirs4all.run()` verified
- All 3708 project unit tests pass
- Ready to proceed to Phase 3: Feature Enhancements

---

## Phase 3: Feature Enhancements (Week 3-4)

### 3.1 Metadata Generator

**File**: `nirs4all/data/synthetic/metadata.py`

Key implementation:
- Sample ID generation
- Biological sample grouping with repetitions
- Group assignment
- Custom column generation

### 3.2 Classification Generator

**File**: `nirs4all/data/synthetic/targets.py`

Key implementation:
- Multi-class target generation
- Class separation via component profiles
- Balanced/imbalanced class generation

### 3.3 Multi-Source Generator

**File**: `nirs4all/data/synthetic/sources.py`

Key implementation:
- Multiple wavelength ranges
- Different source types (NIR, markers, aux)
- Correlated sources

### 3.4 Target Distribution Enhancements

**File**: `nirs4all/data/synthetic/targets.py`

Key implementation:
- Uniform, normal, lognormal, bimodal distributions
- Mixture models
- Empirical distribution matching

---

## Phase 4: Export & Fitting (Week 4-5)

### 4.1 DatasetExporter

**File**: `nirs4all/data/synthetic/exporter.py`

**Status**: ✅ **Complete**

Key implementation:
- `DatasetExporter` class with configurable export options
- `to_folder()` method supporting "standard", "single", and "fragmented" formats
- `to_csv()` method for single-file exports
- `to_numpy()` method for .npz format exports
- Multi-target support with proper file naming

### 4.2 CSVVariationGenerator

**File**: `nirs4all/data/synthetic/exporter.py`

**Status**: ✅ **Complete**

Key implementation:
- Generate multiple CSV format variations for testing data loaders
- Configurable delimiters (semicolon, comma, tab)
- Configurable precision levels (2, 4, 6, 8 decimal places)
- Optional header inclusion/exclusion
- Systematic variation naming

### 4.3 RealDataFitter

**File**: `nirs4all/data/synthetic/fitter.py`

**Status**: ✅ **Complete**

Key implementation:
- `RealDataFitter` class for analyzing real datasets
- `SpectralProperties` dataclass for holding analysis results
- `FittedParameters` dataclass with save/load JSON serialization
- `fit()` method that extracts:
  - Mean spectrum and standard deviation
  - Global slope statistics
  - Noise estimation via Savitzky-Golay filtering
  - SNR estimation
  - PCA explained variance and component count
- `evaluate_similarity()` method for comparing synthetic to real data
- `get_tuning_recommendations()` for parameter adjustment suggestions
- `compute_spectral_properties()` convenience function
- `compare_datasets()` convenience function
- `fit_to_real_data()` convenience function

### 4.4 API Integration

**Status**: ✅ **Complete**

- `SyntheticDatasetBuilder.export()` - Export dataset to folder
- `SyntheticDatasetBuilder.export_to_csv()` - Export to single CSV
- `SyntheticDatasetBuilder.fit_to()` - Fit parameters to template data
- `nirs4all.generate.to_folder()` - Top-level export function
- `nirs4all.generate.to_csv()` - Top-level CSV export
- `nirs4all.generate.from_template()` - Generate from real data template

### 4.5 Unit Tests

**Status**: ✅ **Complete** - 50 new tests (287 total synthetic tests)

Test files:
- `tests/unit/data/synthetic/test_exporter.py` - 23 tests
- `tests/unit/data/synthetic/test_fitter.py` - 27 tests

Coverage:
- All export formats tested (standard, single, fragmented, numpy)
- CSV variation generation tested
- Fitter analysis and similarity evaluation tested
- API integration tested through builder and generate functions

### 4.6 Acceptance Criteria - Phase 4

- [x] `DatasetExporter.to_folder()` works with all formats
- [x] `DatasetExporter.to_csv()` creates valid CSV files
- [x] `DatasetExporter.to_numpy()` creates valid .npz files
- [x] `CSVVariationGenerator` generates all format variations
- [x] `RealDataFitter.fit()` extracts parameters from real data
- [x] `RealDataFitter.evaluate_similarity()` compares datasets
- [x] `FittedParameters` serializes to/from JSON
- [x] `SyntheticDatasetBuilder.export()` integrates with exporter
- [x] `SyntheticDatasetBuilder.fit_to()` integrates with fitter
- [x] `nirs4all.generate.to_folder()` works as documented
- [x] `nirs4all.generate.from_template()` works as documented
- [x] All 287 synthetic module tests pass

**Phase 4 Completion Notes** (2024-12-31):
- Created comprehensive export system supporting multiple formats
- Implemented real data fitting with statistical analysis
- Added similarity evaluation and tuning recommendations
- Full integration with builder and top-level API
- 50 new unit tests bringing total to 287 tests
- All tests pass in 1.58s
- Ready to proceed to Phase 5: Test Migration

---

## Phase 5: Test Migration (Week 5-6)

### 5.1 Add Fixtures to conftest.py

### 5.2 Update Existing Tests

### 5.3 Add CSV Loader Variation Tests

---

## Phase 6: Documentation (Week 6)

### 6.1 API Reference

Location: `docs/source/api/synthetic.rst`

### 6.2 User Guide

Location: `docs/source/user_guide/synthetic_data.md`

### 6.3 Example Updates

Files to update:
- Create `examples/user/Q50_synthetic_data.py`
- Update `examples/developer/` with synthetic examples

---

## Appendix: File Checklist

### New Files to Create

```
nirs4all/data/synthetic/
├── __init__.py           ✅ Phase 1
├── generator.py          ✅ Phase 1 (move + refactor)
├── components.py         ✅ Phase 1 (move)
├── config.py             ✅ Phase 1
├── builder.py            ✅ Phase 2
├── targets.py            ✅ Phase 3
├── metadata.py           ✅ Phase 3
├── sources.py            ✅ Phase 3
├── exporter.py           ✅ Phase 4
├── fitter.py             ✅ Phase 4
├── validation.py         ✅ Phase 1
└── _constants.py         ✅ Phase 1

nirs4all/api/
└── generate.py           ✅ Phase 2

nirs4all/visualization/
└── synthetic.py          □ Phase 2 (integrated with existing visualization)

tests/unit/data/synthetic/
├── __init__.py           ✅ Phase 1
├── conftest.py           ✅ Phase 1
├── test_generator.py     ✅ Phase 1
├── test_components.py    ✅ Phase 1
├── test_config.py        ✅ Phase 1
├── test_builder.py       ✅ Phase 2
├── test_metadata.py      ✅ Phase 3
├── test_targets.py       ✅ Phase 3
├── test_sources.py       ✅ Phase 3
├── test_validation.py    ✅ Phase 3
├── test_exporter.py      ✅ Phase 4
└── test_fitter.py        ✅ Phase 4

tests/unit/api/
└── test_generate.py      ✅ Phase 2

docs/source/
├── api/synthetic.rst     □ Phase 6
└── user_guide/
    └── synthetic_data.md □ Phase 6

examples/user/
└── Q50_synthetic_data.py □ Phase 6
```

### Files Modified

```
nirs4all/__init__.py              ✅ Phase 2 (add generate)
nirs4all/api/__init__.py          ✅ Phase 2 (add generate)
nirs4all/data/__init__.py         ✅ Phase 1 (add synthetic)
tests/conftest.py                 □ Phase 5 (add fixtures)
bench/synthetic/__init__.py       ✅ Phase 1 (deprecation)
```

---

## Review Notes

### Self-Review Completed

1. **Architecture**: Module structure follows nirs4all patterns ✓
2. **API Design**: Consistent with existing `nirs4all.run()` patterns ✓
3. **Type Safety**: All public functions have type hints ✓
4. **Documentation**: Google-style docstrings specified ✓
5. **Testing**: Comprehensive test strategy defined ✓
6. **Migration**: Backward compatibility maintained ✓

### Potential Issues Identified

1. **Builder complexity**: The fluent API may become unwieldy with many options
   - **Mitigation**: Use config dataclasses for internal state

2. **Performance for large datasets**: Need to verify generation speed
   - **Mitigation**: Add benchmarks in Phase 1

3. **Multi-source complexity**: Integration with existing SpectroDataset sources
   - **Mitigation**: Review SpectroDataset.add_samples() before Phase 3

4. **Template fitting accuracy**: May require iterative refinement
   - **Mitigation**: Start with statistical matching, add PCA in later iteration
