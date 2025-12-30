# Test Integration Strategy for Synthetic Data

**Version**: 1.0
**Status**: Draft
**Created**: 2024-12-30

---

## Overview

This document details the strategy for integrating synthetic data generation into the nirs4all test suite. The goal is to:

1. Replace static test fixtures with reproducible synthetic data
2. Enable comprehensive CSV loader testing with format variations
3. Provide consistent, deterministic test data across CI environments
4. Reduce test maintenance burden

---

## Current Test Infrastructure

### Existing Test Data Sources

| Source | Location | Purpose |
|--------|----------|---------|
| `examples/sample_data/` | Folder | Example datasets for documentation |
| `tests/fixtures/data_generators.py` | Python | Simple synthetic generator for tests |
| Manual temp files | In-test | Ad-hoc test data creation |

### Issues with Current Approach

1. **Duplication**: Two synthetic generators exist (`bench/synthetic/`, `tests/fixtures/`)
2. **Inconsistency**: Different tests create data differently
3. **Limited Coverage**: CSV loader tests only cover basic formats
4. **Maintenance**: Static fixtures require manual updates
5. **CI Dependencies**: Some tests rely on external files

---

## Design Decisions

### Complexity Defaults

| Test Type | Complexity | Rationale |
|-----------|------------|-----------|
| **Unit Tests** | `"simple"` | Fast execution, deterministic, minimal overhead |
| **Integration Tests** | `"realistic"` | More realistic spectra, max 5 second overhead acceptable |

This ensures unit tests remain fast while integration tests validate behavior on realistic data.

### Visualization

Synthetic spectra visualization is merged into `nirs4all.visualization.synthetic` module, reusing existing spectral plotting infrastructure since synthetic spectra are structurally identical to real spectra.

---

## Target Architecture

### Test Data Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                      Test Data Sources                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Session-Scoped │    │ Function-Scoped │                    │
│  │     Fixtures     │    │    Fixtures     │                    │
│  │                 │    │                 │                    │
│  │ - Shared across │    │ - Fresh per test │                    │
│  │   all tests     │    │ - Isolated state │                    │
│  │ - Generated once│    │ - Custom configs │                    │
│  └────────┬────────┘    └────────┬────────┘                    │
│           │                      │                              │
│           └──────────┬───────────┘                              │
│                      │                                          │
│                      ▼                                          │
│           ┌──────────────────────┐                              │
│           │ SyntheticDataset     │                              │
│           │     Builder          │                              │
│           └──────────────────────┘                              │
│                      │                                          │
│      ┌───────────────┼───────────────┐                          │
│      ▼               ▼               ▼                          │
│ ┌──────────┐  ┌──────────────┐  ┌──────────────┐               │
│ │SpectroDS │  │ Numpy Arrays │  │  CSV Files   │               │
│ │ Objects  │  │   (X, y)     │  │  (tmp_path)  │               │
│ └──────────┘  └──────────────┘  └──────────────┘               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Fixture Hierarchy

```python
# Fixture Scopes and Reuse

@pytest.fixture(scope="session")
def synthetic_generator():
    """Shared generator instance for all tests."""
    # Created once per test session
    pass

@pytest.fixture(scope="session")
def standard_regression_dataset(synthetic_generator):
    """Standard regression dataset reused across tests."""
    # Created once, shared by all tests needing regression data
    pass

@pytest.fixture(scope="module")
def module_specific_data(synthetic_generator):
    """Data specific to a test module."""
    # Created once per test file
    pass

@pytest.fixture(scope="function")
def isolated_dataset(synthetic_generator):
    """Fresh dataset for each test function."""
    # Created for each test, allows modification
    pass

@pytest.fixture(scope="function")
def temp_csv_dataset(tmp_path, synthetic_generator):
    """Temporary CSV files, cleaned up after test."""
    pass
```

---

## Pytest Fixtures Specification

### Core Fixtures (tests/conftest.py)

```python
"""
Core pytest fixtures for synthetic data generation.

These fixtures provide consistent, reproducible test data across
the entire test suite using nirs4all's synthetic data generator.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Tuple, Generator

# Import synthetic module after it's integrated
from nirs4all.data.synthetic import SyntheticDatasetBuilder
from nirs4all.data import SpectroDataset


# ============================================================================
# Session-Scoped Fixtures (Shared Across All Tests)
# ============================================================================

@pytest.fixture(scope="session")
def synthetic_builder_factory():
    """
    Factory for creating SyntheticDatasetBuilder instances.

    Use this when you need custom configurations but want
    consistent random state handling.

    Example:
        def test_custom_data(synthetic_builder_factory):
            dataset = (
                synthetic_builder_factory(n_samples=100)
                .with_features(complexity="complex")
                .build()
            )
    """
    def _factory(n_samples: int = 200, random_state: int = 42, **kwargs):
        return SyntheticDatasetBuilder(
            n_samples=n_samples,
            random_state=random_state,
            **kwargs
        )
    return _factory


@pytest.fixture(scope="session")
def standard_regression_dataset() -> SpectroDataset:
    """
    Standard regression dataset for testing.

    Properties:
        - 200 samples (160 train, 40 test)
        - Simple complexity (fast, deterministic)
        - Single target
        - Wavelength range: 1000-2500 nm

    Shared across all tests - do not modify!
    """
    return (
        SyntheticDatasetBuilder(n_samples=200, random_state=42)
        .with_features(complexity="simple")
        .with_targets(distribution="uniform", range=(10, 50))
        .with_partitions(train_ratio=0.8)
        .build()
    )


@pytest.fixture(scope="session")
def standard_classification_dataset() -> SpectroDataset:
    """
    Standard classification dataset for testing.

    Properties:
        - 150 samples (120 train, 30 test)
        - 3 classes with balanced distribution
        - Simple complexity
    """
    return (
        SyntheticDatasetBuilder(
            n_samples=150,
            task_type="multiclass",
            random_state=42
        )
        .with_features(complexity="simple")
        .with_targets(n_classes=3)
        .with_partitions(train_ratio=0.8)
        .build()
    )


@pytest.fixture(scope="session")
def standard_binary_dataset() -> SpectroDataset:
    """
    Standard binary classification dataset.

    Properties:
        - 100 samples (80 train, 20 test)
        - 2 classes
        - Good class separation
    """
    return (
        SyntheticDatasetBuilder(
            n_samples=100,
            task_type="binary",
            random_state=42
        )
        .with_features(complexity="simple")
        .with_targets(n_classes=2, class_balance=[0.5, 0.5])
        .with_partitions(train_ratio=0.8)
        .build()
    )


@pytest.fixture(scope="session")
def multi_target_dataset() -> SpectroDataset:
    """
    Multi-target regression dataset.

    Properties:
        - 200 samples
        - 3 target variables
    """
    return (
        SyntheticDatasetBuilder(n_samples=200, random_state=42)
        .with_features(complexity="simple")
        .with_targets(n_targets=3)
        .with_partitions(train_ratio=0.8)
        .build()
    )


@pytest.fixture(scope="session")
def multi_source_dataset() -> SpectroDataset:
    """
    Multi-source dataset for testing source handling.

    Properties:
        - 200 samples
        - 3 sources: NIR_low, NIR_high, markers
    """
    return (
        SyntheticDatasetBuilder(n_samples=200, random_state=42)
        .with_sources([
            {"name": "NIR_low", "type": "nir", "wavelength_range": (1000, 1700)},
            {"name": "NIR_high", "type": "nir", "wavelength_range": (1700, 2500)},
            {"name": "markers", "type": "aux", "n_features": 15},
        ])
        .with_partitions(train_ratio=0.8)
        .build()
    )


@pytest.fixture(scope="session")
def dataset_with_metadata() -> SpectroDataset:
    """
    Dataset with rich metadata for testing aggregation, grouping.

    Properties:
        - 300 samples
        - 4 groups
        - 2-5 repetitions per biological sample
        - ~80 biological samples
    """
    return (
        SyntheticDatasetBuilder(n_samples=300, random_state=42)
        .with_features(complexity="simple")
        .with_metadata(
            n_groups=4,
            group_names=["site_A", "site_B", "site_C", "site_D"],
            n_repetitions=(2, 5),
        )
        .with_partitions(train_ratio=0.8)
        .build()
    )


@pytest.fixture(scope="session")
def dataset_with_batch_effects() -> SpectroDataset:
    """
    Dataset with simulated batch/session effects.

    Useful for testing preprocessing robustness.
    """
    return (
        SyntheticDatasetBuilder(n_samples=300, random_state=42)
        .with_features(
            complexity="realistic",
            batch_effects=True,
            n_batches=3
        )
        .with_partitions(train_ratio=0.8)
        .build()
    )


# ============================================================================
# Function-Scoped Fixtures (Fresh Per Test)
# ============================================================================

@pytest.fixture
def fresh_regression_dataset(synthetic_builder_factory) -> SpectroDataset:
    """
    Fresh regression dataset for tests that modify data.

    Use this when your test modifies the dataset object.
    """
    return (
        synthetic_builder_factory(n_samples=100)
        .with_features(complexity="simple")
        .with_partitions(train_ratio=0.8)
        .build()
    )


@pytest.fixture
def regression_arrays(synthetic_builder_factory) -> Tuple[np.ndarray, np.ndarray]:
    """
    Regression data as numpy arrays.

    Returns:
        (X, y) tuple of numpy arrays
    """
    return (
        synthetic_builder_factory(n_samples=100)
        .with_features(complexity="simple")
        .build_arrays()
    )


@pytest.fixture
def classification_arrays(synthetic_builder_factory) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classification data as numpy arrays.

    Returns:
        (X, y) tuple where y contains class labels
    """
    return (
        synthetic_builder_factory(n_samples=100, task_type="multiclass")
        .with_targets(n_classes=3)
        .build_arrays()
    )


# ============================================================================
# File-Based Fixtures (For Loader Testing)
# ============================================================================

@pytest.fixture
def synthetic_dataset_folder(tmp_path, synthetic_builder_factory) -> Path:
    """
    Create a temporary folder with synthetic dataset files.

    Creates standard nirs4all folder structure:
        - Xcal.csv.gz
        - Ycal.csv.gz
        - Xval.csv.gz
        - Yval.csv.gz

    Returns:
        Path to the temporary dataset folder
    """
    return (
        synthetic_builder_factory(n_samples=100)
        .with_partitions(train_ratio=0.8)
        .export(tmp_path / "dataset", format="standard")
    )


@pytest.fixture
def synthetic_single_file(tmp_path, synthetic_builder_factory) -> Path:
    """
    Create a single CSV file with all data.

    Returns:
        Path to the CSV file
    """
    return (
        synthetic_builder_factory(n_samples=100)
        .export(tmp_path / "dataset", format="single")
    )


# ============================================================================
# Parametrized Fixtures for Format Variations
# ============================================================================

@pytest.fixture(params=["simple", "realistic", "complex"])
def dataset_all_complexities(request, synthetic_builder_factory) -> SpectroDataset:
    """
    Parametrized fixture providing all complexity levels.

    Tests using this fixture run 3x with different complexity.
    """
    return (
        synthetic_builder_factory(n_samples=100)
        .with_features(complexity=request.param)
        .build()
    )


@pytest.fixture(params=[
    {"n_classes": 2, "task_type": "binary"},
    {"n_classes": 3, "task_type": "multiclass"},
    {"n_classes": 5, "task_type": "multiclass"},
])
def classification_variations(request, synthetic_builder_factory) -> SpectroDataset:
    """
    Parametrized fixture for classification with varying class counts.
    """
    return (
        synthetic_builder_factory(
            n_samples=100,
            task_type=request.param["task_type"]
        )
        .with_targets(n_classes=request.param["n_classes"])
        .build()
    )
```

### CSV Loader Test Fixtures (tests/unit/data/loaders/conftest.py)

```python
"""
CSV loader test fixtures with format variations.

These fixtures generate CSV files in various formats to test
the loader's ability to handle different file configurations.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from nirs4all.data.synthetic import SyntheticDatasetBuilder, CSVVariationGenerator


@pytest.fixture(scope="module")
def base_synthetic_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Base synthetic data for all CSV format tests.

    Returns:
        (X, y, wavelengths) tuple
    """
    builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)
    builder.with_features(
        wavelength_range=(1000, 2000),
        wavelength_step=10,
        complexity="simple"
    )
    X, y = builder.build_arrays()
    wavelengths = np.arange(1000, 2001, 10)
    return X, y, wavelengths


# ============================================================================
# Individual Format Fixtures
# ============================================================================

@pytest.fixture
def csv_standard(tmp_path, base_synthetic_data) -> Path:
    """Standard semicolon-separated gzip compressed CSV."""
    X, y, wavelengths = base_synthetic_data
    return CSVVariationGenerator.standard(
        X, y, tmp_path / "standard",
        wavelengths=wavelengths
    )


@pytest.fixture
def csv_comma_separated(tmp_path, base_synthetic_data) -> Path:
    """Comma-separated CSV."""
    X, y, wavelengths = base_synthetic_data
    return CSVVariationGenerator.comma_separated(
        X, y, tmp_path / "comma",
        wavelengths=wavelengths
    )


@pytest.fixture
def csv_tab_separated(tmp_path, base_synthetic_data) -> Path:
    """Tab-separated CSV."""
    X, y, wavelengths = base_synthetic_data
    return CSVVariationGenerator.tab_separated(
        X, y, tmp_path / "tab",
        wavelengths=wavelengths
    )


@pytest.fixture
def csv_no_compression(tmp_path, base_synthetic_data) -> Path:
    """Uncompressed CSV."""
    X, y, wavelengths = base_synthetic_data
    return CSVVariationGenerator.no_compression(
        X, y, tmp_path / "uncompressed",
        wavelengths=wavelengths
    )


@pytest.fixture
def csv_zip_compression(tmp_path, base_synthetic_data) -> Path:
    """ZIP compressed CSV."""
    X, y, wavelengths = base_synthetic_data
    return CSVVariationGenerator.zip_compression(
        X, y, tmp_path / "zip",
        wavelengths=wavelengths
    )


@pytest.fixture
def csv_text_headers(tmp_path, base_synthetic_data) -> Path:
    """CSV with text column headers instead of wavelengths."""
    X, y, _ = base_synthetic_data
    return CSVVariationGenerator.with_text_headers(
        X, y, tmp_path / "text_headers",
        header_names=[f"feature_{i}" for i in range(X.shape[1])]
    )


@pytest.fixture
def csv_no_headers(tmp_path, base_synthetic_data) -> Path:
    """CSV without header row."""
    X, y, _ = base_synthetic_data
    return CSVVariationGenerator.no_headers(
        X, y, tmp_path / "no_headers"
    )


@pytest.fixture
def csv_with_index(tmp_path, base_synthetic_data) -> Path:
    """CSV with row index column."""
    X, y, wavelengths = base_synthetic_data
    return CSVVariationGenerator.with_index_column(
        X, y, tmp_path / "with_index",
        wavelengths=wavelengths
    )


@pytest.fixture
def csv_european_decimals(tmp_path, base_synthetic_data) -> Path:
    """CSV with European decimal format (comma decimal, semicolon separator)."""
    X, y, wavelengths = base_synthetic_data
    return CSVVariationGenerator.european_decimals(
        X, y, tmp_path / "european",
        wavelengths=wavelengths
    )


@pytest.fixture
def csv_with_missing_values(tmp_path, base_synthetic_data) -> Path:
    """CSV with random missing values (NaN)."""
    X, y, wavelengths = base_synthetic_data
    return CSVVariationGenerator.with_missing_values(
        X, y, tmp_path / "missing",
        wavelengths=wavelengths,
        missing_ratio=0.02
    )


@pytest.fixture
def csv_multi_file_x(tmp_path, base_synthetic_data) -> Path:
    """CSV with X split across multiple files (multi-source simulation)."""
    X, y, wavelengths = base_synthetic_data
    return CSVVariationGenerator.multi_file_x(
        X, y, tmp_path / "multi_x",
        wavelengths=wavelengths,
        n_files=3
    )


@pytest.fixture
def csv_single_file_all(tmp_path, base_synthetic_data) -> Path:
    """Single CSV with X, y, and metadata columns."""
    X, y, wavelengths = base_synthetic_data
    metadata = {
        "sample_id": [f"S{i:04d}" for i in range(len(X))],
        "group": np.random.choice(["A", "B", "C"], size=len(X))
    }
    return CSVVariationGenerator.single_file_all(
        X, y, tmp_path / "single_all",
        wavelengths=wavelengths,
        metadata=metadata
    )


# ============================================================================
# Parametrized Fixtures for Comprehensive Testing
# ============================================================================

@pytest.fixture(params=[
    "standard",
    "comma_separated",
    "tab_separated",
    "no_compression",
    "with_text_headers",
    "no_headers",
    "with_index",
])
def csv_basic_variations(request, tmp_path, base_synthetic_data) -> Tuple[Path, dict]:
    """
    Parametrized fixture for basic CSV format variations.

    Returns:
        (path, metadata) tuple where metadata describes the format
    """
    X, y, wavelengths = base_synthetic_data
    generator = getattr(CSVVariationGenerator, request.param)

    format_info = {
        "standard": {"delimiter": ";", "compression": "gzip", "has_header": True},
        "comma_separated": {"delimiter": ",", "compression": "gzip", "has_header": True},
        "tab_separated": {"delimiter": "\t", "compression": "gzip", "has_header": True},
        "no_compression": {"delimiter": ";", "compression": None, "has_header": True},
        "with_text_headers": {"delimiter": ";", "compression": "gzip", "has_header": True, "header_unit": "text"},
        "no_headers": {"delimiter": ";", "compression": "gzip", "has_header": False},
        "with_index": {"delimiter": ";", "compression": "gzip", "has_header": True, "has_index": True},
    }

    path = generator(X, y, tmp_path / request.param, wavelengths=wavelengths)
    return path, format_info[request.param]


@pytest.fixture(params=[
    "standard",
    "european_decimals",
    "with_missing_values",
])
def csv_edge_cases(request, tmp_path, base_synthetic_data) -> Tuple[Path, dict]:
    """
    Parametrized fixture for edge case CSV formats.

    Tests format variations that require special handling.
    """
    X, y, wavelengths = base_synthetic_data
    generator = getattr(CSVVariationGenerator, request.param)

    edge_info = {
        "standard": {"has_special": False},
        "european_decimals": {"decimal_separator": ",", "delimiter": ";"},
        "with_missing_values": {"has_nan": True, "missing_ratio": 0.02},
    }

    path = generator(X, y, tmp_path / request.param, wavelengths=wavelengths)
    return path, edge_info[request.param]
```

---

## Test Migration Guide

### Step 1: Identify Tests Using Static Data

Run grep to find tests using old patterns:

```bash
# Find tests using old generator
grep -r "SyntheticNIRSDataGenerator" tests/

# Find tests using manual temp file creation
grep -r "tempfile.NamedTemporaryFile" tests/
grep -r "tmp_path.*\.write" tests/

# Find tests importing from fixtures/data_generators
grep -r "from tests.fixtures.data_generators" tests/
```

### Step 2: Migration Pattern

**Before (old pattern):**

```python
# tests/unit/data/test_something.py

import tempfile
import numpy as np
from tests.fixtures.data_generators import SyntheticNIRSDataGenerator

class TestSomething:
    def test_feature(self):
        gen = SyntheticNIRSDataGenerator(random_state=42)
        X, y = gen.generate_regression_data(n_samples=100)

        # Create temp files manually
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            np.savetxt(f, X, delimiter=';')
            temp_path = f.name

        # Test logic
        ...
```

**After (new pattern):**

```python
# tests/unit/data/test_something.py

class TestSomething:
    def test_feature(self, standard_regression_dataset):
        """Test using session-scoped fixture."""
        X = standard_regression_dataset.x({"partition": "train"})
        y = standard_regression_dataset.y({"partition": "train"})

        # Test logic - no manual file handling needed
        ...

    def test_feature_with_file(self, synthetic_dataset_folder):
        """Test using file-based fixture."""
        from nirs4all.data import DatasetConfigs

        dataset = DatasetConfigs(synthetic_dataset_folder).get_datasets()[0]
        # Test logic
        ...

    def test_custom_configuration(self, synthetic_builder_factory):
        """Test needing custom data configuration."""
        dataset = (
            synthetic_builder_factory(n_samples=50)
            .with_features(complexity="complex")
            .with_metadata(n_groups=2)
            .build()
        )
        # Test logic
        ...
```

### Step 3: Update Test Files

Priority order for migration:

1. **High Priority** (frequently failing or slow tests):
   - `tests/unit/data/loaders/test_csv_loader.py`
   - `tests/integration/pipeline/test_basic_pipeline.py`

2. **Medium Priority** (functional tests):
   - `tests/unit/data/test_dataset.py`
   - `tests/unit/data/test_config.py`
   - `tests/integration/data/`

3. **Low Priority** (stable tests):
   - Other unit tests using static data

---

## CSVVariationGenerator Specification

### Implementation

```python
# nirs4all/data/synthetic/exporter.py

class CSVVariationGenerator:
    """
    Generate CSV files in various formats for loader testing.

    Each method creates a complete dataset folder with the specified
    format characteristics. Used primarily for testing the data loader's
    ability to handle different file formats.
    """

    @staticmethod
    def standard(
        X: np.ndarray,
        y: np.ndarray,
        path: Path,
        wavelengths: Optional[np.ndarray] = None,
        train_ratio: float = 0.8,
    ) -> Path:
        """
        Create standard nirs4all format.

        Format:
            - Semicolon delimiter
            - Gzip compression
            - Wavelength headers
            - Separate train/test files

        Files created:
            - Xcal.csv.gz, Ycal.csv.gz
            - Xval.csv.gz, Yval.csv.gz
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Split data
        n_train = int(len(X) * train_ratio)
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # Create headers
        if wavelengths is not None:
            headers = [str(int(w)) for w in wavelengths]
        else:
            headers = [str(i) for i in range(X.shape[1])]

        # Save files
        _save_csv(path / "Xcal.csv.gz", X_train, headers, ";", "gzip")
        _save_csv(path / "Ycal.csv.gz", y_train.reshape(-1, 1), None, ";", "gzip")
        _save_csv(path / "Xval.csv.gz", X_test, headers, ";", "gzip")
        _save_csv(path / "Yval.csv.gz", y_test.reshape(-1, 1), None, ";", "gzip")

        return path

    @staticmethod
    def comma_separated(X, y, path, **kwargs) -> Path:
        """Comma-separated values (delimiter=',')."""
        # Implementation

    @staticmethod
    def tab_separated(X, y, path, **kwargs) -> Path:
        """Tab-separated values (delimiter='\\t')."""
        # Implementation

    @staticmethod
    def no_compression(X, y, path, **kwargs) -> Path:
        """Uncompressed CSV files."""
        # Implementation

    @staticmethod
    def zip_compression(X, y, path, **kwargs) -> Path:
        """ZIP compressed files."""
        # Implementation

    @staticmethod
    def with_text_headers(X, y, path, header_names: List[str], **kwargs) -> Path:
        """Text column headers instead of wavelengths."""
        # Implementation

    @staticmethod
    def no_headers(X, y, path, **kwargs) -> Path:
        """CSV without header row."""
        # Implementation

    @staticmethod
    def with_index_column(X, y, path, **kwargs) -> Path:
        """Include row index as first column."""
        # Implementation

    @staticmethod
    def european_decimals(X, y, path, **kwargs) -> Path:
        """
        European decimal format.

        - Decimal separator: comma (,)
        - Field separator: semicolon (;)
        - Example: 1,234;5,678;9,012
        """
        # Implementation

    @staticmethod
    def with_missing_values(X, y, path, missing_ratio: float = 0.01, **kwargs) -> Path:
        """
        CSV with random NaN values.

        Useful for testing NA handling.
        """
        X_copy = X.copy()
        mask = np.random.random(X_copy.shape) < missing_ratio
        X_copy[mask] = np.nan
        # Save with NaN handling

    @staticmethod
    def multi_file_x(X, y, path, n_files: int = 3, **kwargs) -> Path:
        """
        Split X across multiple files.

        Creates:
            - Xcal_1.csv.gz, Xcal_2.csv.gz, Xcal_3.csv.gz
            - Ycal.csv.gz (single file)

        Simulates multi-source data from separate instruments.
        """
        # Implementation

    @staticmethod
    def single_file_all(
        X, y, path,
        metadata: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> Path:
        """
        All data in a single CSV file.

        Columns: [wavelengths..., target, metadata_cols...]
        """
        # Implementation

    @staticmethod
    def fragmented_partitions(X, y, path, **kwargs) -> Path:
        """
        Separate folders for each partition.

        Creates:
            train/
                X.csv.gz
                y.csv.gz
            test/
                X.csv.gz
                y.csv.gz
        """
        # Implementation
```

---

## Test File Updates Required

### tests/conftest.py Additions

```python
# Add imports
from nirs4all.data.synthetic import SyntheticDatasetBuilder

# Add fixtures from specification above
# ...

# Deprecate old generator fixture
@pytest.fixture
def legacy_data_generator():
    """
    DEPRECATED: Use synthetic_builder_factory instead.

    Kept for backward compatibility during migration.
    """
    import warnings
    warnings.warn(
        "legacy_data_generator is deprecated, use synthetic_builder_factory",
        DeprecationWarning
    )
    from tests.fixtures.data_generators import SyntheticNIRSDataGenerator
    return SyntheticNIRSDataGenerator(random_state=42)
```

### tests/fixtures/data_generators.py Update

```python
"""
DEPRECATED: This module is deprecated.

Use nirs4all.data.synthetic instead:

    # Old
    from tests.fixtures.data_generators import SyntheticNIRSDataGenerator

    # New
    from nirs4all.data.synthetic import SyntheticDatasetBuilder

    # Or use fixtures in tests
    def test_something(standard_regression_dataset):
        ...

This module will be removed in a future version.
"""

import warnings

warnings.warn(
    "tests.fixtures.data_generators is deprecated. "
    "Use nirs4all.data.synthetic or conftest fixtures.",
    DeprecationWarning,
    stacklevel=2
)

# Keep old code for backward compatibility
# ... existing implementation ...
```

---

## Acceptance Criteria

### Phase 5 Completion Checklist

- [ ] All core fixtures implemented in `tests/conftest.py`
- [ ] CSV variation fixtures implemented
- [ ] At least 10 test files migrated to new fixtures
- [ ] `tests/unit/data/loaders/test_csv_loader.py` using variations
- [ ] `tests/fixtures/data_generators.py` deprecated
- [ ] No test failures after migration
- [ ] CI pipeline passes
- [ ] Test execution time not increased by >10%

### Quality Metrics

| Metric | Target |
|--------|--------|
| Test files using new fixtures | >80% |
| CSV format variations tested | 10+ |
| Fixture code coverage | 100% |
| Synthetic module test coverage | >90% |

---

## Appendix: Fixture Quick Reference

### Session-Scoped (Shared, Read-Only)

| Fixture | Purpose | Samples |
|---------|---------|---------|
| `standard_regression_dataset` | Basic regression | 200 |
| `standard_classification_dataset` | 3-class classification | 150 |
| `standard_binary_dataset` | Binary classification | 100 |
| `multi_target_dataset` | Multi-output regression | 200 |
| `multi_source_dataset` | Multiple data sources | 200 |
| `dataset_with_metadata` | Groups & repetitions | 300 |
| `dataset_with_batch_effects` | Batch simulation | 300 |

### Function-Scoped (Fresh, Modifiable)

| Fixture | Purpose | Returns |
|---------|---------|---------|
| `fresh_regression_dataset` | Modifiable regression | SpectroDataset |
| `regression_arrays` | Raw arrays | (X, y) |
| `classification_arrays` | Classification arrays | (X, y) |
| `synthetic_dataset_folder` | Temp file folder | Path |
| `synthetic_single_file` | Single CSV file | Path |

### Parametrized (Multiple Runs)

| Fixture | Variations | Use Case |
|---------|------------|----------|
| `dataset_all_complexities` | simple, realistic, complex | Complexity testing |
| `classification_variations` | 2, 3, 5 classes | Class count testing |
| `csv_basic_variations` | 7 formats | Loader format testing |
| `csv_edge_cases` | 3 formats | Edge case testing |

### Factory

| Fixture | Purpose |
|---------|---------|
| `synthetic_builder_factory` | Create custom builders |
