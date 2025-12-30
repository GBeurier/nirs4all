# Synthetic NIRS Data Generator - Full Specification

**Version**: 1.0
**Status**: Approved
**Authors**: nirs4all Team
**Created**: 2024-12-30
**Last Updated**: 2024-12-30

---

## Design Decisions (Confirmed)

| Decision | Resolution | Rationale |
|----------|------------|-----------|
| **API Name** | `nirs4all.generate()` | Standard ML naming (sklearn: `make_regression`, etc.) |
| **Default Complexity** | `"simple"` for unit tests, `"realistic"` for integration | Balance speed vs realism; max 5s overhead |
| **Visualization** | Merge into `nirs4all.visualization` | Spectra are spectra - unified plotting API |
| **Implementation** | Single-shot implementation | All phases implemented together |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Goals and Requirements](#3-goals-and-requirements)
4. [Architecture Design](#4-architecture-design)
5. [API Specification](#5-api-specification)
6. [Enhancement Specifications](#6-enhancement-specifications)
7. [Dataset Export Specification](#7-dataset-export-specification)
8. [Test Integration Specification](#8-test-integration-specification)
9. [Migration Strategy](#9-migration-strategy)
10. [Roadmap](#10-roadmap)
11. [Appendices](#appendices)

---

## 1. Executive Summary

### 1.1 Purpose

This document specifies the integration and enhancement of a synthetic NIRS spectra generator into the nirs4all library. The goal is to provide a robust, production-ready tool for:

- **Testing**: Generate reproducible datasets for unit/integration tests
- **Examples**: Create on-the-fly datasets for documentation examples
- **Research**: Enable controlled experiments with known ground truth
- **Benchmarking**: Compare algorithm performance on standardized synthetic data

### 1.2 Scope

This specification covers:

1. **API Integration**: Clean integration into `nirs4all` public API
2. **Generator Enhancement**: More realistic spectra generation
3. **Real Data Fitting**: Ability to mimic real dataset characteristics
4. **Dataset Export**: Generate nirs4all-compatible files and objects
5. **Test Modernization**: Replace static fixtures with synthetic data

### 1.3 Key Design Principles

| Principle | Description |
|-----------|-------------|
| **Consistency** | Follow existing nirs4all API patterns and coding conventions |
| **Reproducibility** | All generated data must be reproducible via random seeds |
| **Flexibility** | Support simple to complex use cases |
| **Performance** | Efficient generation for large datasets |
| **Validation** | Validate generated data against expected properties |

---

## 2. Current State Analysis

### 2.1 Existing Generator (bench/synthetic/)

The current synthetic generator resides in `bench/synthetic/` and includes:

| File | Purpose | Lines | Maturity |
|------|---------|-------|----------|
| `generator.py` | Core `SyntheticNIRSGenerator` class | ~824 | Beta |
| `visualizer.py` | Visualization tools | ~880 | Beta |
| `comparator.py` | Real vs synthetic comparison | ~800 | Alpha |
| `__init__.py` | Module exports | ~65 | Stable |

**Current Features:**
- ✅ Beer-Lambert law based generation
- ✅ Voigt profile peak shapes
- ✅ Predefined components (water, protein, lipid, etc.)
- ✅ Batch effects simulation
- ✅ Multiple complexity levels (simple, realistic, complex)
- ✅ Various concentration methods (Dirichlet, uniform, lognormal, correlated)
- ✅ Baseline, scatter, noise effects
- ✅ `create_dataset()` method for SpectroDataset output

**Current Limitations:**
- ❌ Not integrated into nirs4all public API
- ❌ No metadata generation (sample IDs, groups, repetitions)
- ❌ No multi-source support
- ❌ No file export capabilities
- ❌ No real data fitting/mimicking
- ❌ Limited distribution options for targets
- ❌ No classification task support

### 2.2 Existing Test Generator (tests/fixtures/data_generators.py)

A simpler generator exists in tests with:

| Class | Purpose |
|-------|---------|
| `SyntheticNIRSDataGenerator` | Simple pattern-based generation |
| `TestDataManager` | File I/O for test datasets |

**Features:**
- ✅ Regression, classification, multi-target, multi-source
- ✅ File export (CSV, gzip)
- ✅ Reproducible random seeds

**Limitations:**
- ❌ Not physically motivated (simple patterns only)
- ❌ No realistic spectral properties
- ❌ Separate from main generator

### 2.3 Integration Points

```
nirs4all/
├── api/
│   ├── __init__.py     ← ADD: generate() function
│   └── generate.py     ← NEW: Generation API
├── data/
│   ├── synthetic/      ← NEW: Generator module
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── components.py
│   │   ├── metadata.py
│   │   ├── fitter.py
│   │   └── exporter.py
│   └── loaders/
│       └── synthetic_loader.py  ← NEW: Config-based generation
```

---

## 3. Goals and Requirements

### 3.1 Functional Requirements

#### FR-1: API Integration

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.1 | Expose `nirs4all.generate()` as top-level function | P0 |
| FR-1.2 | Support `nirs4all.generate.regression()` for quick regression datasets | P0 |
| FR-1.3 | Support `nirs4all.generate.classification()` for classification datasets | P0 |
| FR-1.4 | Support `nirs4all.generate.from_template()` for mimicking real data | P1 |
| FR-1.5 | Return `SpectroDataset` or tuple `(X, y)` based on parameter | P0 |

#### FR-2: Generator Enhancement

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-2.1 | Generate realistic metadata (sample_id, group, repetition) | P0 |
| FR-2.2 | Support multi-source generation (e.g., NIR + markers) | P0 |
| FR-2.3 | Support classification tasks with separable classes | P0 |
| FR-2.4 | Provide configurable target distributions | P1 |
| FR-2.5 | Support temporal/batch correlation structures | P1 |
| FR-2.6 | Add more predefined component libraries | P2 |

#### FR-3: Real Data Fitting

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-3.1 | Analyze real dataset statistical properties | P1 |
| FR-3.2 | Fit generator parameters to match real data | P1 |
| FR-3.3 | Export fitted parameters for reproducibility | P2 |
| FR-3.4 | Provide tuning recommendations | P2 |

#### FR-4: Export Capabilities

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-4.1 | Export as `SpectroDataset` object | P0 |
| FR-4.2 | Export as numpy arrays `(X, y)` | P0 |
| FR-4.3 | Export as CSV files (nirs4all standard format) | P0 |
| FR-4.4 | Export as multi-file datasets (Xcal, Ycal, Xval, Yval) | P0 |
| FR-4.5 | Export as single-file datasets (all data + metadata) | P1 |
| FR-4.6 | Export fragmented datasets (for loader testing) | P1 |

#### FR-5: Test Integration

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-5.1 | Replace static test fixtures with generator calls | P1 |
| FR-5.2 | Provide pytest fixtures for common dataset types | P0 |
| FR-5.3 | Generate complex CSV variations for loader tests | P1 |
| FR-5.4 | Support deterministic generation in CI environments | P0 |

### 3.2 Non-Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-1 | Generation of 10,000 samples < 1 second | P1 |
| NFR-2 | 100% test coverage for generator module | P1 |
| NFR-3 | Type hints on all public functions | P0 |
| NFR-4 | Google-style docstrings | P0 |
| NFR-5 | No additional required dependencies | P0 |

---

## 4. Architecture Design

### 4.1 Module Structure

```
nirs4all/
├── data/
│   └── synthetic/
│       ├── __init__.py           # Public exports
│       ├── generator.py          # SyntheticNIRSGenerator (core)
│       ├── components.py         # NIRBand, SpectralComponent, ComponentLibrary
│       ├── targets.py            # Target distribution generators
│       ├── metadata.py           # MetadataGenerator for sample IDs, groups
│       ├── sources.py            # MultiSourceGenerator
│       ├── fitter.py             # RealDataFitter for parameter estimation
│       ├── exporter.py           # DatasetExporter for file/object output
│       ├── presets.py            # Predefined configurations
│       └── validation.py         # Generated data validation
├── api/
│   └── generate.py               # Top-level generate() function
├── visualization/
│   └── synthetic.py              # Synthetic spectra visualization (merged)
└── __init__.py                   # Add generate to nirs4all namespace
```

> **Note**: Visualization tools for synthetic spectra are integrated into the existing
> `nirs4all.visualization` module rather than being a separate submodule. This reflects
> the design decision that "spectra are spectra" - synthetic and real spectra share
> the same plotting infrastructure.

### 4.2 Class Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           nirs4all.generate()                            │
│                    (Top-level convenience function)                      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       SyntheticDatasetBuilder                            │
│  ─────────────────────────────────────────────────────────────────────  │
│  + n_samples: int                                                        │
│  + task_type: str                                                        │
│  + complexity: str                                                       │
│  + random_state: int                                                     │
│  ─────────────────────────────────────────────────────────────────────  │
│  + with_features(**kwargs) → Self                                        │
│  + with_targets(**kwargs) → Self                                         │
│  + with_metadata(**kwargs) → Self                                        │
│  + with_sources(**kwargs) → Self                                         │
│  + with_partitions(**kwargs) → Self                                      │
│  + fit_to(real_data) → Self                                              │
│  + build() → SpectroDataset                                              │
│  + build_arrays() → Tuple[np.ndarray, np.ndarray]                        │
│  + export(path, format) → Path                                           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ uses
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
          ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ FeatureGenerator │  │  TargetGenerator │  │ MetadataGenerator│
│ (SyntheticNIRS   │  │                   │  │                   │
│  Generator)      │  │                   │  │                   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │
          ▼
┌─────────────────┐
│ ComponentLibrary │
│ (NIRBand,        │
│  SpectralComp.)  │
└─────────────────┘
```

### 4.3 Integration with Existing Systems

```
                    ┌─────────────────────┐
                    │    User Code        │
                    └─────────┬───────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ nirs4all.run()  │  │nirs4all.generate│  │DatasetConfigs   │
│                 │  │     ()          │  │  (path)         │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         │                    ▼                    │
         │           ┌─────────────────┐           │
         │           │SyntheticDataset │           │
         │           │   Builder       │           │
         │           └────────┬────────┘           │
         │                    │                    │
         │         ┌──────────┴──────────┐         │
         │         ▼                     ▼         │
         │  ┌─────────────┐      ┌─────────────┐   │
         │  │SpectroDataset│      │ CSV/Files  │   │
         │  └──────┬──────┘      └──────┬──────┘   │
         │         │                    │          │
         └─────────┴────────────────────┴──────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ PipelineRunner  │
                   └─────────────────┘
```

### 4.4 Configuration Schema

```python
@dataclass
class SyntheticDatasetConfig:
    """Configuration for synthetic dataset generation."""

    # Core settings
    n_samples: int = 1000
    n_train: Optional[int] = None  # If set, creates train/test split
    n_test: Optional[int] = None
    task_type: Literal["regression", "binary", "multiclass"] = "regression"
    random_state: Optional[int] = None

    # Feature generation
    features: FeatureConfig = field(default_factory=FeatureConfig)

    # Target generation
    targets: TargetConfig = field(default_factory=TargetConfig)

    # Metadata generation
    metadata: MetadataConfig = field(default_factory=MetadataConfig)

    # Multi-source configuration
    sources: Optional[List[SourceConfig]] = None

    # Output configuration
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class FeatureConfig:
    """Configuration for spectral feature generation."""
    wavelength_start: float = 1000
    wavelength_end: float = 2500
    wavelength_step: float = 2
    complexity: Literal["simple", "realistic", "complex"] = "realistic"
    components: Optional[List[str]] = None  # None = default library
    n_components: int = 5

    # Effects
    include_baseline: bool = True
    include_scatter: bool = True
    include_noise: bool = True
    noise_level: Optional[float] = None  # Override complexity default


@dataclass
class TargetConfig:
    """Configuration for target variable generation."""
    n_targets: int = 1
    target_names: Optional[List[str]] = None
    distribution: Literal["uniform", "normal", "lognormal", "bimodal"] = "uniform"
    range: Tuple[float, float] = (0.0, 100.0)
    correlation_with_features: float = 0.8  # How much features predict target

    # For classification
    n_classes: int = 2
    class_names: Optional[List[str]] = None
    class_balance: Optional[List[float]] = None  # e.g., [0.7, 0.3]


@dataclass
class MetadataConfig:
    """Configuration for metadata generation."""
    include_sample_id: bool = True
    sample_id_prefix: str = "S"

    # Repetition structure (multiple spectra per biological sample)
    n_repetitions: Tuple[int, int] = (1, 1)  # (min, max) repetitions
    repetition_column: str = "repetition"
    bio_sample_column: str = "bio_sample_id"

    # Groups
    n_groups: int = 0
    group_column: str = "group"
    group_names: Optional[List[str]] = None

    # Additional metadata columns
    extra_columns: Optional[Dict[str, Any]] = None


@dataclass
class SourceConfig:
    """Configuration for a single data source in multi-source datasets."""
    name: str
    source_type: Literal["nir", "markers", "aux"] = "nir"
    n_features: int = 100

    # For NIR sources
    wavelength_range: Optional[Tuple[float, float]] = None

    # For marker sources
    marker_names: Optional[List[str]] = None


@dataclass
class OutputConfig:
    """Configuration for output format."""
    format: Literal["dataset", "arrays", "csv", "folder"] = "dataset"
    path: Optional[str] = None

    # CSV options
    delimiter: str = ";"
    compression: Optional[str] = "gzip"  # None, "gzip", "zip"
    include_headers: bool = True
    header_unit: str = "nm"

    # File layout
    file_layout: Literal["standard", "single", "fragmented"] = "standard"
```

---

## 5. API Specification

### 5.1 Top-Level API (`nirs4all.generate`)

#### 5.1.1 Main Entry Point

```python
def generate(
    n_samples: int = 1000,
    *,
    task_type: Literal["regression", "binary", "multiclass"] = "regression",
    n_features: int = 751,
    n_targets: int = 1,
    complexity: Literal["simple", "realistic", "complex"] = "realistic",
    random_state: Optional[int] = None,
    output: Literal["dataset", "arrays"] = "dataset",
    **kwargs
) -> Union[SpectroDataset, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate a synthetic NIRS dataset.

    This is the primary entry point for synthetic data generation in nirs4all.
    It provides a simple interface for common use cases while supporting
    advanced customization through keyword arguments.

    Args:
        n_samples: Total number of samples to generate.
        task_type: Type of ML task ("regression", "binary", "multiclass").
        n_features: Number of spectral features (wavelengths).
        n_targets: Number of target variables (for multi-output regression).
        complexity: Noise and effect complexity level.
        random_state: Random seed for reproducibility.
        output: Output format ("dataset" for SpectroDataset, "arrays" for numpy).
        **kwargs: Additional configuration options (see SyntheticDatasetConfig).

    Returns:
        SpectroDataset or (X, y) tuple depending on output parameter.

    Examples:
        >>> import nirs4all

        # Quick regression dataset
        >>> dataset = nirs4all.generate(n_samples=1000)

        # Classification with arrays output
        >>> X, y = nirs4all.generate(
        ...     n_samples=500,
        ...     task_type="multiclass",
        ...     n_classes=3,
        ...     output="arrays"
        ... )

        # Custom complexity with reproducibility
        >>> dataset = nirs4all.generate(
        ...     n_samples=2000,
        ...     complexity="complex",
        ...     random_state=42
        ... )
    """
```

#### 5.1.2 Convenience Functions

```python
# nirs4all.generate.regression()
def regression(
    n_samples: int = 1000,
    n_targets: int = 1,
    target_range: Tuple[float, float] = (0.0, 100.0),
    target_distribution: str = "uniform",
    **kwargs
) -> Union[SpectroDataset, Tuple[np.ndarray, np.ndarray]]:
    """Generate a regression dataset with specified target characteristics."""


# nirs4all.generate.classification()
def classification(
    n_samples: int = 1000,
    n_classes: int = 2,
    class_balance: Optional[List[float]] = None,
    class_separation: float = 0.8,
    **kwargs
) -> Union[SpectroDataset, Tuple[np.ndarray, np.ndarray]]:
    """Generate a classification dataset with separable classes."""


# nirs4all.generate.multi_source()
def multi_source(
    n_samples: int = 1000,
    sources: List[Dict[str, Any]] = None,
    **kwargs
) -> SpectroDataset:
    """Generate a multi-source dataset (e.g., NIR + molecular markers)."""


# nirs4all.generate.from_template()
def from_template(
    template: Union[str, SpectroDataset, np.ndarray],
    n_samples: int = 1000,
    match_statistics: bool = True,
    match_structure: bool = True,
    **kwargs
) -> SpectroDataset:
    """
    Generate synthetic data that mimics a real dataset template.

    Args:
        template: Real dataset to mimic (path, SpectroDataset, or numpy array).
        n_samples: Number of samples to generate.
        match_statistics: Match mean, std, slope, noise characteristics.
        match_structure: Match PCA structure and component patterns.
    """


# nirs4all.generate.builder()
def builder(**kwargs) -> SyntheticDatasetBuilder:
    """Get a builder for advanced customization with fluent API."""
```

#### 5.1.3 Export Functions

```python
# nirs4all.generate.to_folder()
def to_folder(
    path: Union[str, Path],
    n_samples: int = 1000,
    train_ratio: float = 0.8,
    format: Literal["standard", "single", "fragmented"] = "standard",
    **generate_kwargs
) -> Path:
    """
    Generate and export a synthetic dataset to a folder.

    Args:
        path: Output folder path.
        n_samples: Total samples to generate.
        train_ratio: Proportion of samples for training.
        format: File layout format:
            - "standard": Xcal.csv.gz, Ycal.csv.gz, Xval.csv.gz, Yval.csv.gz
            - "single": data.csv with all columns
            - "fragmented": Multiple files per partition (for testing)

    Returns:
        Path to the created folder.

    Example:
        >>> import nirs4all
        >>> path = nirs4all.generate.to_folder(
        ...     "test_data/synthetic_wheat",
        ...     n_samples=1000,
        ...     task_type="regression",
        ...     random_state=42
        ... )
        >>> # Now usable with DatasetConfigs
        >>> dataset = DatasetConfigs(path).get_datasets()[0]
    """


# nirs4all.generate.to_csv()
def to_csv(
    path: Union[str, Path],
    n_samples: int = 1000,
    include_metadata: bool = True,
    **generate_kwargs
) -> Path:
    """Generate and export to a single CSV file."""
```

### 5.2 Builder API

```python
class SyntheticDatasetBuilder:
    """
    Fluent builder for complex synthetic dataset configurations.

    Provides a chainable API for building datasets with custom
    features, targets, metadata, and export options.

    Example:
        >>> from nirs4all.data.synthetic import SyntheticDatasetBuilder
        >>>
        >>> dataset = (
        ...     SyntheticDatasetBuilder(n_samples=1000, random_state=42)
        ...     .with_features(
        ...         wavelength_range=(1000, 2500),
        ...         complexity="realistic",
        ...         components=["water", "protein", "starch"]
        ...     )
        ...     .with_targets(
        ...         distribution="lognormal",
        ...         range=(5, 50),
        ...         correlation=0.85
        ...     )
        ...     .with_metadata(
        ...         n_groups=3,
        ...         n_repetitions=(2, 5),
        ...         group_names=["field_A", "field_B", "field_C"]
        ...     )
        ...     .with_partitions(train_ratio=0.8)
        ...     .build()
        ... )
    """

    def __init__(
        self,
        n_samples: int = 1000,
        task_type: str = "regression",
        random_state: Optional[int] = None
    ):
        """Initialize builder with core parameters."""

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
        n_batches: int = 1
    ) -> 'SyntheticDatasetBuilder':
        """Configure spectral feature generation."""

    def with_targets(
        self,
        n_targets: int = 1,
        distribution: str = "uniform",
        range: Tuple[float, float] = (0, 100),
        correlation: float = 0.8,
        noise: float = 0.1,
        # Classification-specific
        n_classes: int = 2,
        class_balance: Optional[List[float]] = None
    ) -> 'SyntheticDatasetBuilder':
        """Configure target variable generation."""

    def with_metadata(
        self,
        n_groups: int = 0,
        group_names: Optional[List[str]] = None,
        n_repetitions: Union[int, Tuple[int, int]] = 1,
        extra_columns: Optional[Dict[str, Any]] = None
    ) -> 'SyntheticDatasetBuilder':
        """Configure metadata generation (sample IDs, groups, repetitions)."""

    def with_sources(
        self,
        sources: List[Dict[str, Any]]
    ) -> 'SyntheticDatasetBuilder':
        """
        Configure multi-source generation.

        Args:
            sources: List of source configurations:
                [
                    {"name": "NIR", "type": "nir", "wavelength_range": (1000, 1700)},
                    {"name": "markers", "type": "aux", "n_features": 10}
                ]
        """

    def with_partitions(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.0,
        stratify: bool = False
    ) -> 'SyntheticDatasetBuilder':
        """Configure train/val/test partitioning."""

    def fit_to(
        self,
        template: Union[str, SpectroDataset, np.ndarray],
        match_statistics: bool = True,
        match_structure: bool = True
    ) -> 'SyntheticDatasetBuilder':
        """Fit generation parameters to match a real dataset."""

    def build(self) -> SpectroDataset:
        """Build and return a SpectroDataset."""

    def build_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build and return (X, y) numpy arrays."""

    def export(
        self,
        path: Union[str, Path],
        format: str = "standard"
    ) -> Path:
        """Build and export to files."""
```

### 5.3 Integration with `nirs4all.run()`

The synthetic generator integrates seamlessly with the existing run API:

```python
import nirs4all
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

# Direct generation in run()
result = nirs4all.run(
    pipeline=[StandardScaler(), PLSRegression(10)],
    dataset=nirs4all.generate(n_samples=1000, random_state=42),
    verbose=1
)

# Using builder for complex datasets
dataset = (
    nirs4all.generate.builder(n_samples=500)
    .with_metadata(n_groups=3, n_repetitions=(2, 4))
    .with_features(complexity="complex", batch_effects=True)
    .build()
)

result = nirs4all.run(
    pipeline=[...],
    dataset=dataset
)
```

---

## 6. Enhancement Specifications

### 6.1 Metadata Generation

#### 6.1.1 Sample Identity

```python
class MetadataGenerator:
    """Generate realistic metadata for synthetic datasets."""

    def generate(
        self,
        n_samples: int,
        config: MetadataConfig
    ) -> Dict[str, np.ndarray]:
        """
        Generate metadata arrays for all samples.

        Returns:
            Dictionary with metadata columns:
            - "sample_id": Unique sample identifiers
            - "bio_sample_id": Biological sample IDs (for repetitions)
            - "repetition": Repetition number within bio sample
            - "group": Group assignments
            - Additional custom columns
        """
```

#### 6.1.2 Repetition Structure

Supports the common NIRS scenario where multiple spectra are acquired per biological sample:

```python
# Example: 100 biological samples with 2-5 repetitions each
config = MetadataConfig(
    n_repetitions=(2, 5),  # Random between 2 and 5 repetitions
    bio_sample_column="bio_sample_id",
    repetition_column="rep"
)

# Resulting metadata:
# sample_id | bio_sample_id | rep | group
# S0001     | B001          | 1   | A
# S0002     | B001          | 2   | A
# S0003     | B001          | 3   | A
# S0004     | B002          | 1   | B
# S0005     | B002          | 2   | B
# ...
```

### 6.2 Multi-Source Generation

#### 6.2.1 Source Types

| Type | Description | Features |
|------|-------------|----------|
| `nir` | NIR spectral data | Wavelengths, absorption bands |
| `vis` | Visible spectral data | Different wavelength range |
| `markers` | Molecular markers | Discrete or continuous values |
| `aux` | Auxiliary numerical data | Any numerical features |

#### 6.2.2 Multi-Source Builder

```python
dataset = (
    nirs4all.generate.builder(n_samples=500)
    .with_sources([
        {
            "name": "NIR_low",
            "type": "nir",
            "wavelength_range": (1000, 1700),
            "n_features": 350,
            "components": ["water", "protein"]
        },
        {
            "name": "NIR_high",
            "type": "nir",
            "wavelength_range": (1700, 2500),
            "n_features": 400,
            "components": ["lipid", "starch"]
        },
        {
            "name": "markers",
            "type": "aux",
            "n_features": 15,
            "distribution": "lognormal"
        }
    ])
    .build()
)

# Access sources:
X_nir_low = dataset.x({"partition": "train"}, source="NIR_low")
X_markers = dataset.x({"partition": "train"}, source="markers")
```

### 6.3 Classification Support

#### 6.3.1 Class Separation

```python
class ClassificationGenerator:
    """Generate classification datasets with separable classes."""

    def __init__(
        self,
        n_classes: int = 2,
        class_separation: float = 0.8,
        separation_method: str = "component"
    ):
        """
        Args:
            n_classes: Number of classes.
            class_separation: How separable classes are (0=overlapping, 1=distinct).
            separation_method: How to create class differences:
                - "component": Different component concentrations per class
                - "shift": Spectral shifts between classes
                - "intensity": Different absorption intensities
        """
```

#### 6.3.2 Class Balance

```python
# Balanced classes (default)
dataset = nirs4all.generate.classification(n_samples=300, n_classes=3)
# → 100 samples per class

# Imbalanced classes
dataset = nirs4all.generate.classification(
    n_samples=300,
    n_classes=3,
    class_balance=[0.6, 0.3, 0.1]
)
# → 180, 90, 30 samples
```

### 6.4 Target Distribution Enhancement

#### 6.4.1 Distribution Types

| Distribution | Use Case | Parameters |
|--------------|----------|------------|
| `uniform` | Equal probability across range | `range=(min, max)` |
| `normal` | Gaussian distribution | `mean`, `std` |
| `lognormal` | Right-skewed (concentrations) | `mean`, `sigma` |
| `bimodal` | Two populations | `modes`, `weights` |
| `mixture` | Custom mixture | `components` |
| `empirical` | Match real data | `template` |

#### 6.4.2 Implementation

```python
class TargetGenerator:
    """Generate target variables with specified distributions."""

    def generate(
        self,
        n_samples: int,
        concentrations: np.ndarray,
        config: TargetConfig
    ) -> np.ndarray:
        """
        Generate targets based on concentrations and config.

        For regression:
            y = f(concentrations) + noise

        For classification:
            y = classify(concentrations, thresholds)
        """

    def _generate_regression_targets(
        self,
        n_samples: int,
        concentrations: np.ndarray,
        correlation: float,
        distribution: str,
        range: Tuple[float, float]
    ) -> np.ndarray:
        """Generate continuous targets correlated with concentrations."""

        # Linear combination of component concentrations
        weights = self.rng.dirichlet(np.ones(concentrations.shape[1]))
        base_target = concentrations @ weights

        # Scale to desired range
        base_target = self._scale_to_range(base_target, range)

        # Add noise to achieve desired correlation
        target = self._add_noise_for_correlation(base_target, correlation)

        # Transform to desired distribution
        target = self._transform_distribution(target, distribution, range)

        return target
```

### 6.5 Real Data Fitting

#### 6.5.1 RealDataFitter Class

```python
class RealDataFitter:
    """
    Fit synthetic generator parameters to match real data characteristics.

    Analyzes statistical properties of real spectra and configures
    the generator to produce similar synthetic data.

    Example:
        >>> from nirs4all.data.synthetic import RealDataFitter
        >>> from nirs4all.data import DatasetConfigs
        >>>
        >>> # Load real data
        >>> real_dataset = DatasetConfigs("sample_data/wheat").get_datasets()[0]
        >>> X_real = real_dataset.x({"partition": "train"})
        >>>
        >>> # Fit parameters
        >>> fitter = RealDataFitter()
        >>> params = fitter.fit(X_real, wavelengths=real_dataset.wavelengths)
        >>>
        >>> # Generate similar synthetic data
        >>> generator = SyntheticNIRSGenerator(**params.to_generator_kwargs())
        >>> X_synth, _, _ = generator.generate(n_samples=1000)
    """

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        wavelengths: Optional[np.ndarray] = None
    ) -> 'FittedParameters':
        """
        Analyze real data and fit generation parameters.

        Extracts:
            - Noise characteristics (level, wavelength-dependency)
            - Baseline properties (polynomial coefficients)
            - Global slope statistics
            - PCA structure (variance distribution)
            - Peak positions and widths
        """

    def get_recommendations(self) -> Dict[str, Any]:
        """Get human-readable tuning recommendations."""


@dataclass
class FittedParameters:
    """Container for fitted generation parameters."""

    # Noise
    noise_base: float
    noise_signal_dep: float

    # Baseline
    baseline_amplitude: float

    # Slope
    global_slope_mean: float
    global_slope_std: float

    # Scatter
    scatter_alpha_std: float
    scatter_beta_std: float

    # PCA structure
    n_effective_components: int
    variance_ratios: np.ndarray

    # Wavelength range
    wavelength_start: float
    wavelength_end: float
    wavelength_step: float

    def to_generator_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for SyntheticNIRSGenerator."""

    def to_yaml(self) -> str:
        """Export as YAML configuration."""

    @classmethod
    def from_yaml(cls, path: str) -> 'FittedParameters':
        """Load from YAML configuration."""
```

---

## 7. Dataset Export Specification

### 7.1 Export Formats

#### 7.1.1 Standard Folder Format

```
output_folder/
├── Xcal.csv.gz       # Training features
├── Ycal.csv.gz       # Training targets
├── Xval.csv.gz       # Validation/test features
├── Yval.csv.gz       # Validation/test targets
└── metadata.csv      # Optional: sample metadata
```

#### 7.1.2 Single File Format

```
output_folder/
└── data.csv.gz
    # Columns: wavelength_1, ..., wavelength_N, target_1, ..., target_M, partition, sample_id, ...
```

#### 7.1.3 Fragmented Format (for loader testing)

```
output_folder/
├── train/
│   ├── features_part1.csv
│   ├── features_part2.csv
│   ├── targets.csv
│   └── meta.csv
├── test/
│   ├── features.csv
│   └── targets.csv
└── config.yaml         # Dataset configuration
```

### 7.2 DatasetExporter Class

```python
class DatasetExporter:
    """Export synthetic datasets to various file formats."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        wavelengths: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, np.ndarray]] = None,
        partitions: Optional[np.ndarray] = None
    ):
        """Initialize with generated data."""

    def to_folder(
        self,
        path: Union[str, Path],
        format: str = "standard",
        delimiter: str = ";",
        compression: Optional[str] = "gzip",
        include_headers: bool = True,
        header_unit: str = "nm"
    ) -> Path:
        """
        Export to a nirs4all-compatible folder.

        Args:
            path: Output folder path.
            format: Layout format ("standard", "single", "fragmented").
            delimiter: CSV delimiter.
            compression: Compression type (None, "gzip", "zip").
            include_headers: Whether to include wavelength headers.
            header_unit: Header unit ("nm", "cm-1").

        Returns:
            Path to created folder.
        """

    def to_spectrodataset(self) -> SpectroDataset:
        """Convert to SpectroDataset object."""

    def to_config_dict(self) -> Dict[str, Any]:
        """Generate a DatasetConfigs-compatible configuration dict."""
```

### 7.3 CSV Format Variations for Testing

```python
class CSVVariationGenerator:
    """Generate various CSV format variations for loader testing."""

    @staticmethod
    def standard(X, y, path, **kwargs) -> Path:
        """Standard semicolon-separated, gzip compressed."""

    @staticmethod
    def comma_separated(X, y, path, **kwargs) -> Path:
        """Comma-separated values."""

    @staticmethod
    def tab_separated(X, y, path, **kwargs) -> Path:
        """Tab-separated values."""

    @staticmethod
    def no_compression(X, y, path, **kwargs) -> Path:
        """Uncompressed CSV."""

    @staticmethod
    def zip_compression(X, y, path, **kwargs) -> Path:
        """ZIP compressed."""

    @staticmethod
    def with_text_headers(X, y, path, **kwargs) -> Path:
        """Text feature names instead of wavelengths."""

    @staticmethod
    def with_index_column(X, y, path, **kwargs) -> Path:
        """Include row index column."""

    @staticmethod
    def european_decimals(X, y, path, **kwargs) -> Path:
        """Comma decimal separator, semicolon field separator."""

    @staticmethod
    def with_missing_values(X, y, path, missing_ratio=0.01, **kwargs) -> Path:
        """Include NaN values at random positions."""

    @staticmethod
    def multi_file_x(X, y, path, n_files=3, **kwargs) -> Path:
        """Split X across multiple files (multi-source simulation)."""

    @staticmethod
    def single_file_all(X, y, path, metadata=None, **kwargs) -> Path:
        """All data in single CSV file."""
```

---

## 8. Test Integration Specification

### 8.1 Pytest Fixtures

```python
# tests/conftest.py (additions)

import pytest
from nirs4all.data.synthetic import SyntheticDatasetBuilder

@pytest.fixture(scope="session")
def synthetic_regression_dataset():
    """Provide a standard synthetic regression dataset for testing."""
    return (
        SyntheticDatasetBuilder(n_samples=200, random_state=42)
        .with_features(complexity="simple")
        .with_partitions(train_ratio=0.8)
        .build()
    )


@pytest.fixture(scope="session")
def synthetic_classification_dataset():
    """Provide a standard synthetic classification dataset."""
    return (
        SyntheticDatasetBuilder(n_samples=150, task_type="multiclass", random_state=42)
        .with_targets(n_classes=3, class_balance=[0.4, 0.35, 0.25])
        .with_partitions(train_ratio=0.8)
        .build()
    )


@pytest.fixture(scope="session")
def synthetic_multi_source_dataset():
    """Provide a multi-source synthetic dataset."""
    return (
        SyntheticDatasetBuilder(n_samples=200, random_state=42)
        .with_sources([
            {"name": "NIR", "type": "nir", "wavelength_range": (1000, 2000)},
            {"name": "markers", "type": "aux", "n_features": 10}
        ])
        .with_partitions(train_ratio=0.8)
        .build()
    )


@pytest.fixture(scope="function")
def synthetic_dataset_path(tmp_path):
    """Create a temporary synthetic dataset folder."""
    import nirs4all
    return nirs4all.generate.to_folder(
        tmp_path / "synthetic",
        n_samples=100,
        random_state=42
    )


@pytest.fixture(scope="session")
def synthetic_data_generator():
    """Provide access to the generator for custom configurations."""
    from nirs4all.data.synthetic import SyntheticDatasetBuilder
    return SyntheticDatasetBuilder


# Fixture for CSV loader testing
@pytest.fixture(params=[
    "standard",
    "comma_separated",
    "no_compression",
    "with_text_headers",
    "european_decimals",
    "with_missing_values"
])
def csv_variation_path(request, tmp_path):
    """Parametrized fixture for various CSV formats."""
    from nirs4all.data.synthetic import CSVVariationGenerator
    from nirs4all.data.synthetic import SyntheticDatasetBuilder

    builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)
    X, y = builder.build_arrays()

    generator_method = getattr(CSVVariationGenerator, request.param)
    return generator_method(X, y, tmp_path / request.param)
```

### 8.2 Test Migration Plan

#### 8.2.1 Tests to Update

| Test File | Current Approach | New Approach |
|-----------|------------------|--------------|
| `tests/unit/data/loaders/test_csv_loader.py` | Manual temp files | `CSVVariationGenerator` fixtures |
| `tests/unit/data/test_dataset.py` | Static fixtures | `synthetic_regression_dataset` |
| `tests/integration/pipeline/` | `sample_data/` | Synthetic datasets |
| `tests/integration/data/` | Manual creation | Fixtures |

#### 8.2.2 New Test Files

```
tests/
├── unit/
│   └── data/
│       └── synthetic/
│           ├── test_generator.py
│           ├── test_metadata.py
│           ├── test_targets.py
│           ├── test_exporter.py
│           └── test_fitter.py
└── integration/
    └── data/
        ├── test_synthetic_pipeline.py
        └── test_csv_variations.py
```

### 8.3 CSV Loader Test Enhancement

```python
# tests/unit/data/loaders/test_csv_variations.py

class TestCSVLoaderWithSyntheticData:
    """Test CSV loader with generated format variations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Generate base synthetic data."""
        builder = SyntheticDatasetBuilder(n_samples=100, random_state=42)
        self.X, self.y = builder.build_arrays()
        self.wavelengths = np.linspace(1000, 2500, self.X.shape[1])

    def test_standard_format(self, tmp_path):
        """Test loading standard semicolon-separated gzip format."""
        path = CSVVariationGenerator.standard(self.X, self.y, tmp_path)
        # ... assertions

    def test_comma_separated(self, tmp_path):
        """Test loading comma-separated format."""
        path = CSVVariationGenerator.comma_separated(self.X, self.y, tmp_path)
        # ... assertions

    def test_european_decimals(self, tmp_path):
        """Test loading with European decimal format."""
        path = CSVVariationGenerator.european_decimals(self.X, self.y, tmp_path)
        # ... assertions

    def test_missing_values_handling(self, tmp_path):
        """Test NA handling with controlled missing values."""
        path = CSVVariationGenerator.with_missing_values(
            self.X, self.y, tmp_path, missing_ratio=0.02
        )
        # ... assertions about NA handling

    def test_multi_file_loading(self, tmp_path):
        """Test loading fragmented X files."""
        path = CSVVariationGenerator.multi_file_x(self.X, self.y, tmp_path, n_files=3)
        # ... assertions about multi-source loading

    def test_single_file_all_columns(self, tmp_path):
        """Test loading single file with X, y, and metadata."""
        metadata = {"group": np.repeat(["A", "B"], 50)}
        path = CSVVariationGenerator.single_file_all(
            self.X, self.y, tmp_path, metadata=metadata
        )
        # ... assertions
```

---

## 9. Migration Strategy

### 9.1 Phase 1: Core Module (Week 1-2)

**Goal**: Move and restructure generator code into nirs4all

1. Create `nirs4all/data/synthetic/` module structure
2. Move core classes from `bench/synthetic/`:
   - `generator.py` → `generator.py`
   - Component classes → `components.py`
3. Add comprehensive type hints
4. Write unit tests
5. Ensure backward compatibility via deprecation in `bench/synthetic/`

### 9.2 Phase 2: API Integration (Week 2-3)

**Goal**: Integrate with nirs4all public API

1. Implement `SyntheticDatasetBuilder`
2. Create `nirs4all/api/generate.py`
3. Add `generate` to `nirs4all.__init__.py`
4. Write API tests
5. Update `nirs4all.run()` to accept generated datasets

### 9.3 Phase 3: Enhancement (Week 3-4)

**Goal**: Add new features

1. Implement `MetadataGenerator`
2. Implement classification support
3. Add multi-source generation
4. Enhance target distributions
5. Write tests for new features

### 9.4 Phase 4: Export & Fitting (Week 4-5)

**Goal**: File export and real data fitting

1. Implement `DatasetExporter`
2. Implement `CSVVariationGenerator`
3. Implement `RealDataFitter`
4. Write integration tests

### 9.5 Phase 5: Test Migration (Week 5-6)

**Goal**: Update test suite

1. Add synthetic data fixtures to `conftest.py`
2. Update existing tests to use fixtures
3. Add CSV loader variation tests
4. Remove redundant test data files

### 9.6 Phase 6: Documentation (Week 6)

**Goal**: Complete documentation

1. API reference documentation
2. User guide with examples
3. Update existing examples
4. Migration guide from `bench/synthetic/`

---

## 10. Roadmap

### 10.1 Timeline Overview

```
Week 1-2: Core Module
├── Move code to nirs4all/data/synthetic/
├── Refactor for production quality
├── Add type hints and docstrings
└── Unit tests (80% coverage)

Week 2-3: API Integration
├── SyntheticDatasetBuilder class
├── nirs4all.generate() function
├── Integration with nirs4all.run()
└── API tests

Week 3-4: Enhancements
├── Metadata generation
├── Classification support
├── Multi-source generation
├── Target distribution options
└── Enhancement tests

Week 4-5: Export & Fitting
├── DatasetExporter class
├── CSVVariationGenerator
├── RealDataFitter class
└── Integration tests

Week 5-6: Test Migration
├── Pytest fixtures
├── Update existing tests
├── CSV loader tests
└── Remove static fixtures

Week 6: Documentation
├── API reference
├── User guide
├── Examples update
└── Migration guide
```

### 10.2 Milestones

| Milestone | Target Date | Deliverables |
|-----------|-------------|--------------|
| M1: Core Ready | Week 2 | Generator in nirs4all, basic tests |
| M2: API Ready | Week 3 | `nirs4all.generate()` working |
| M3: Enhanced | Week 4 | Metadata, classification, multi-source |
| M4: Export Ready | Week 5 | File export, fitting |
| M5: Tests Updated | Week 6 | All tests using synthetic data |
| M6: Documented | Week 6 | Full documentation |

### 10.3 Success Criteria

- [ ] `nirs4all.generate()` works for all documented use cases
- [ ] 100% of examples work with synthetic data
- [ ] Test suite runs without external data dependencies
- [ ] CSV loader tests cover 10+ format variations
- [ ] Documentation complete with API reference
- [ ] No regressions in existing functionality

---

## Appendices

### A. Predefined Component Library

| Component | Bands | Primary Use |
|-----------|-------|-------------|
| water | O-H 1450, 1940, 2500 nm | Moisture content |
| protein | N-H 1510, 2050, 2180 nm | Protein analysis |
| lipid | C-H 1210, 1720, 2310 nm | Fat/oil content |
| starch | O-H 1460, 2100 nm | Carbohydrate analysis |
| cellulose | O-H 1490, 2090 nm | Fiber content |
| chlorophyll | 1070, 1400 nm | Plant analysis |
| oil | C-H 1165, 1725, 2305 nm | Oil industry |
| nitrogen_compound | N-H 1500, 2060 nm | Feed analysis |

### B. Example Configurations

#### B.1 Simple Regression

```python
dataset = nirs4all.generate(
    n_samples=1000,
    task_type="regression",
    complexity="realistic",
    random_state=42
)
```

#### B.2 Multi-Target with Metadata

```python
dataset = (
    nirs4all.generate.builder(n_samples=500, random_state=42)
    .with_features(complexity="realistic", components=["protein", "starch", "water"])
    .with_targets(n_targets=3, correlation=0.85)
    .with_metadata(
        n_groups=4,
        group_names=["farm_A", "farm_B", "farm_C", "farm_D"],
        n_repetitions=(2, 5)
    )
    .with_partitions(train_ratio=0.7, val_ratio=0.15)
    .build()
)
```

#### B.3 Classification with Imbalance

```python
dataset = nirs4all.generate.classification(
    n_samples=600,
    n_classes=4,
    class_balance=[0.4, 0.3, 0.2, 0.1],
    class_separation=0.7,
    random_state=42
)
```

#### B.4 Export to Files

```python
path = nirs4all.generate.to_folder(
    "data/synthetic_wheat",
    n_samples=2000,
    train_ratio=0.8,
    format="standard",
    complexity="complex",
    random_state=42
)
```

### C. API Quick Reference

```python
# Top-level functions
nirs4all.generate(n_samples, task_type, ...)  → SpectroDataset
nirs4all.generate.regression(...)             → SpectroDataset
nirs4all.generate.classification(...)         → SpectroDataset
nirs4all.generate.multi_source(...)           → SpectroDataset
nirs4all.generate.from_template(...)          → SpectroDataset
nirs4all.generate.to_folder(...)              → Path
nirs4all.generate.to_csv(...)                 → Path
nirs4all.generate.builder(...)                → SyntheticDatasetBuilder

# Builder methods
builder.with_features(...)                    → SyntheticDatasetBuilder
builder.with_targets(...)                     → SyntheticDatasetBuilder
builder.with_metadata(...)                    → SyntheticDatasetBuilder
builder.with_sources(...)                     → SyntheticDatasetBuilder
builder.with_partitions(...)                  → SyntheticDatasetBuilder
builder.fit_to(...)                           → SyntheticDatasetBuilder
builder.build()                               → SpectroDataset
builder.build_arrays()                        → Tuple[np.ndarray, np.ndarray]
builder.export(...)                           → Path
```

---

**Document Status**: Draft - Awaiting Review

**Next Actions**:
1. Review and validate architecture decisions
2. Confirm API naming conventions
3. Prioritize features for Phase 1
4. Begin implementation
