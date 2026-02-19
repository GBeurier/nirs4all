"""
Pytest configuration for nirs4all tests.

This file configures the test environment, including matplotlib backend
for headless/GUI-less test execution and provides core pytest fixtures
for synthetic data generation.

Fixture Strategy:
    - Session-scoped fixtures: Shared across all tests (read-only data)
    - Function-scoped fixtures: Fresh per test (modifiable data)
    - Parametrized fixtures: Run tests with multiple configurations

Complexity Defaults:
    - Unit tests: "simple" complexity (fast, deterministic)
    - Integration tests: "realistic" complexity (more realistic spectra)
"""

# =============================================================================
# IMPORTANT: Set test workspace BEFORE any nirs4all imports
# This must happen at module load time, before pytest_configure
# =============================================================================
import os
import tempfile

# Create the test workspace directory immediately at module load
# This ensures it's set before any test file imports nirs4all
_TEST_WORKSPACE_DIR = tempfile.mkdtemp(prefix="nirs4all_test_")
os.environ["NIRS4ALL_WORKSPACE"] = _TEST_WORKSPACE_DIR

# Now safe to import other modules
from pathlib import Path

import matplotlib
import numpy as np
import pytest


def pytest_configure(config):
    """
    Configure pytest environment before tests run.

    Sets matplotlib to use non-interactive 'Agg' backend to avoid
    GUI-related errors in test environments (CI/CD, headless systems).

    Args:
        config: pytest config object (required by pytest hook)
    """
    # Use non-interactive backend for all tests
    matplotlib.use('Agg')

def pytest_unconfigure(config):
    """
    Clean up after all tests complete.

    Removes the temporary workspace directory.

    Args:
        config: pytest config object (required by pytest hook)
    """
    global _TEST_WORKSPACE_DIR
    if _TEST_WORKSPACE_DIR and os.path.exists(_TEST_WORKSPACE_DIR):
        import shutil
        try:
            shutil.rmtree(_TEST_WORKSPACE_DIR)
        except Exception:
            pass  # Best effort cleanup
    if "NIRS4ALL_WORKSPACE" in os.environ:
        del os.environ["NIRS4ALL_WORKSPACE"]

# =============================================================================
# Lazy imports for synthetic module (avoid import errors if module missing)
# =============================================================================

def _get_builder():
    """Lazy import of SyntheticDatasetBuilder."""
    from nirs4all.synthesis import SyntheticDatasetBuilder
    return SyntheticDatasetBuilder

def _get_generator():
    """Lazy import of SyntheticNIRSGenerator."""
    from nirs4all.synthesis import SyntheticNIRSGenerator
    return SyntheticNIRSGenerator

def _get_csv_variation_generator():
    """Lazy import of CSVVariationGenerator."""
    from nirs4all.synthesis import CSVVariationGenerator
    return CSVVariationGenerator

# =============================================================================
# Session-Scoped Fixtures (Shared Across All Tests)
# =============================================================================

@pytest.fixture(scope="session")
def test_workspace():
    """
    Get the test workspace directory path.

    This directory is automatically cleaned up after all tests complete.
    Use this fixture when you need explicit access to the shared test workspace.

    Returns:
        Path: Path to the shared test workspace directory.
    """
    return Path(_TEST_WORKSPACE_DIR)

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

    Returns:
        Callable that creates SyntheticDatasetBuilder instances.
    """
    SyntheticDatasetBuilder = _get_builder()

    def _factory(n_samples: int = 200, random_state: int = 42, **kwargs):
        return SyntheticDatasetBuilder(
            n_samples=n_samples,
            random_state=random_state,
            **kwargs
        )
    return _factory

@pytest.fixture(scope="session")
def synthetic_generator_factory():
    """
    Factory for creating SyntheticNIRSGenerator instances.

    Use this for low-level generator access.

    Returns:
        Callable that creates SyntheticNIRSGenerator instances.
    """
    SyntheticNIRSGenerator = _get_generator()

    def _factory(complexity: str = "simple", random_state: int = 42, **kwargs):
        return SyntheticNIRSGenerator(
            complexity=complexity,
            random_state=random_state,
            **kwargs
        )
    return _factory

@pytest.fixture(scope="session")
def standard_regression_dataset():
    """
    Standard regression dataset for testing.

    Properties:
        - 200 samples (160 train, 40 test)
        - Simple complexity (fast, deterministic)
        - Single target, range (10, 50)
        - Wavelength range: 1000-2500 nm

    Returns:
        SpectroDataset: Shared across all tests - do not modify!

    Example:
        def test_preprocessing(standard_regression_dataset):
            X = standard_regression_dataset.x({"partition": "train"})
            y = standard_regression_dataset.y({"partition": "train"})
    """
    SyntheticDatasetBuilder = _get_builder()
    return (
        SyntheticDatasetBuilder(n_samples=200, random_state=42)
        .with_features(complexity="simple")
        .with_targets(distribution="uniform", range=(10, 50))
        .with_partitions(train_ratio=0.8)
        .build()
    )

@pytest.fixture(scope="session")
def standard_classification_dataset():
    """
    Standard classification dataset for testing.

    Properties:
        - 150 samples (120 train, 30 test)
        - 3 classes with balanced distribution
        - Simple complexity, good class separation

    Returns:
        SpectroDataset: Shared across all tests - do not modify!
    """
    SyntheticDatasetBuilder = _get_builder()
    return (
        SyntheticDatasetBuilder(n_samples=150, random_state=42)
        .with_features(complexity="simple")
        .with_classification(n_classes=3, separation=2.0)
        .with_partitions(train_ratio=0.8)
        .build()
    )

@pytest.fixture(scope="session")
def standard_binary_dataset():
    """
    Standard binary classification dataset.

    Properties:
        - 100 samples (80 train, 20 test)
        - 2 classes with balanced distribution
        - Good class separation for reliable tests

    Returns:
        SpectroDataset: Shared across all tests - do not modify!
    """
    SyntheticDatasetBuilder = _get_builder()
    return (
        SyntheticDatasetBuilder(n_samples=100, random_state=42)
        .with_features(complexity="simple")
        .with_classification(n_classes=2, separation=2.5)
        .with_partitions(train_ratio=0.8)
        .build()
    )

@pytest.fixture(scope="session")
def multi_target_dataset():
    """
    Multi-target regression dataset.

    Properties:
        - 200 samples
        - Multiple target variables (from concentration matrix)
        - Simple complexity

    Returns:
        SpectroDataset: Shared across all tests - do not modify!
    """
    SyntheticDatasetBuilder = _get_builder()
    return (
        SyntheticDatasetBuilder(n_samples=200, random_state=42)
        .with_features(
            complexity="simple",
            components=["water", "protein", "lipid"],
        )
        .with_targets(distribution="dirichlet")  # Multiple targets
        .with_partitions(train_ratio=0.8)
        .build()
    )

@pytest.fixture(scope="session")
def multi_source_dataset():
    """
    Multi-source dataset for testing source handling.

    Properties:
        - 200 samples
        - 3 sources: NIR_low, NIR_high, markers
        - Different wavelength ranges and feature types

    Returns:
        SpectroDataset: Shared across all tests - do not modify!
    """
    SyntheticDatasetBuilder = _get_builder()
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
def dataset_with_metadata():
    """
    Dataset with rich metadata for testing aggregation, grouping.

    Properties:
        - 300 samples
        - 4 groups with descriptive names
        - 2-5 repetitions per biological sample
        - Sample IDs generated

    Returns:
        SpectroDataset: Shared across all tests - do not modify!
    """
    SyntheticDatasetBuilder = _get_builder()
    return (
        SyntheticDatasetBuilder(n_samples=300, random_state=42)
        .with_features(complexity="simple")
        .with_metadata(
            sample_ids=True,
            n_groups=4,
            group_names=["site_A", "site_B", "site_C", "site_D"],
            n_repetitions=(2, 5),
        )
        .with_partitions(train_ratio=0.8)
        .build()
    )

@pytest.fixture(scope="session")
def dataset_with_batch_effects():
    """
    Dataset with simulated batch/session effects.

    Properties:
        - 300 samples
        - Realistic complexity with batch variations
        - 3 measurement batches

    Useful for testing preprocessing robustness.

    Returns:
        SpectroDataset: Shared across all tests - do not modify!
    """
    SyntheticDatasetBuilder = _get_builder()
    return (
        SyntheticDatasetBuilder(n_samples=300, random_state=42)
        .with_features(complexity="realistic")
        .with_batch_effects(enabled=True, n_batches=3)
        .with_partitions(train_ratio=0.8)
        .build()
    )

@pytest.fixture(scope="session")
def small_regression_arrays():
    """
    Small regression arrays for quick tests.

    Properties:
        - 50 samples (no split)
        - Simple complexity
        - Returns (X, y) tuple

    Returns:
        Tuple[np.ndarray, np.ndarray]: Shared arrays - do not modify!
    """
    SyntheticDatasetBuilder = _get_builder()
    return (
        SyntheticDatasetBuilder(n_samples=50, random_state=42)
        .with_features(complexity="simple")
        .with_output(as_dataset=False)
        .build()
    )

@pytest.fixture(scope="session")
def sample_wavelengths():
    """
    Standard wavelength array (1000-2500nm, 2nm step).

    Returns:
        np.ndarray: 751 wavelength values.
    """
    return np.arange(1000, 2502, 2)

# =============================================================================
# Function-Scoped Fixtures (Fresh Per Test)
# =============================================================================

@pytest.fixture
def fresh_regression_dataset(synthetic_builder_factory):
    """
    Fresh regression dataset for tests that modify data.

    Use this when your test modifies the dataset object.

    Returns:
        SpectroDataset: Fresh instance, safe to modify.
    """
    return (
        synthetic_builder_factory(n_samples=100)
        .with_features(complexity="simple")
        .with_targets(distribution="uniform", range=(10, 50))
        .with_partitions(train_ratio=0.8)
        .build()
    )

@pytest.fixture
def fresh_classification_dataset(synthetic_builder_factory):
    """
    Fresh classification dataset for tests that modify data.

    Returns:
        SpectroDataset: Fresh instance, safe to modify.
    """
    return (
        synthetic_builder_factory(n_samples=100)
        .with_features(complexity="simple")
        .with_classification(n_classes=3)
        .with_partitions(train_ratio=0.8)
        .build()
    )

@pytest.fixture
def regression_arrays(synthetic_builder_factory) -> tuple[np.ndarray, np.ndarray]:
    """
    Regression data as numpy arrays.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) tuple of numpy arrays.
    """
    return (
        synthetic_builder_factory(n_samples=100)
        .with_features(complexity="simple")
        .with_targets(distribution="uniform", range=(10, 50))
        .build_arrays()
    )

@pytest.fixture
def classification_arrays(synthetic_builder_factory) -> tuple[np.ndarray, np.ndarray]:
    """
    Classification data as numpy arrays.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) tuple where y contains class labels.
    """
    return (
        synthetic_builder_factory(n_samples=100)
        .with_features(complexity="simple")
        .with_classification(n_classes=3)
        .build_arrays()
    )

# =============================================================================
# File-Based Fixtures (For Loader Testing)
# =============================================================================

@pytest.fixture
def synthetic_dataset_folder(tmp_path, synthetic_builder_factory) -> Path:
    """
    Create a temporary folder with synthetic dataset files.

    Creates standard nirs4all folder structure:
        - Xcal.csv
        - Ycal.csv
        - Xval.csv
        - Yval.csv

    Args:
        tmp_path: Pytest's temporary path fixture.
        synthetic_builder_factory: Builder factory fixture.

    Returns:
        Path: Path to the temporary dataset folder.
    """
    return (
        synthetic_builder_factory(n_samples=100)
        .with_features(complexity="simple")
        .with_partitions(train_ratio=0.8)
        .export(tmp_path / "dataset", format="standard")
    )

@pytest.fixture
def synthetic_single_file_folder(tmp_path, synthetic_builder_factory) -> Path:
    """
    Create a folder with single CSV file containing all data.

    Returns:
        Path: Path to the folder containing data.csv.
    """
    return (
        synthetic_builder_factory(n_samples=100)
        .with_features(complexity="simple")
        .with_partitions(train_ratio=0.8)
        .export(tmp_path / "dataset", format="single")
    )

@pytest.fixture
def synthetic_csv_file(tmp_path, synthetic_builder_factory) -> Path:
    """
    Create a single CSV file with all data.

    Returns:
        Path: Path to the CSV file.
    """
    return (
        synthetic_builder_factory(n_samples=100)
        .with_features(complexity="simple")
        .export_to_csv(tmp_path / "data.csv")
    )

# =============================================================================
# Base Data Fixtures for Format Variation Testing
# =============================================================================

@pytest.fixture(scope="module")
def base_synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Base synthetic data for CSV format tests.

    Session-scoped for efficiency in format variation tests.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (X, y, wavelengths) tuple.
    """
    SyntheticDatasetBuilder = _get_builder()
    builder = (
        SyntheticDatasetBuilder(n_samples=50, random_state=42)
        .with_features(
            wavelength_range=(1000, 2000),
            wavelength_step=10,
            complexity="simple"
        )
        .with_targets(distribution="uniform", range=(10, 50))
    )
    X, y = builder.build_arrays()
    wavelengths = np.arange(1000, 2001, 10)
    return X, y, wavelengths

# =============================================================================
# Parametrized Fixtures for Comprehensive Testing
# =============================================================================

@pytest.fixture(params=["simple", "realistic", "complex"])
def dataset_all_complexities(request, synthetic_builder_factory):
    """
    Parametrized fixture providing all complexity levels.

    Tests using this fixture run 3x with different complexity.

    Args:
        request: Pytest request object with param.
        synthetic_builder_factory: Builder factory fixture.

    Returns:
        SpectroDataset: Dataset with specified complexity.
    """
    return (
        synthetic_builder_factory(n_samples=100)
        .with_features(complexity=request.param)
        .with_partitions(train_ratio=0.8)
        .build()
    )

@pytest.fixture(params=[2, 3, 5])
def classification_n_classes(request, synthetic_builder_factory):
    """
    Parametrized fixture for classification with varying class counts.

    Tests using this fixture run 3x with different n_classes.

    Args:
        request: Pytest request object with param.
        synthetic_builder_factory: Builder factory fixture.

    Returns:
        SpectroDataset: Classification dataset with specified n_classes.
    """
    return (
        synthetic_builder_factory(n_samples=100)
        .with_features(complexity="simple")
        .with_classification(n_classes=request.param)
        .with_partitions(train_ratio=0.8)
        .build()
    )

# =============================================================================
# CSV Variation Fixtures (For Loader Testing)
# =============================================================================

@pytest.fixture
def csv_variation_generator():
    """
    Get CSVVariationGenerator instance.

    Returns:
        CSVVariationGenerator: For creating CSV format variations.
    """
    CSVVariationGenerator = _get_csv_variation_generator()
    return CSVVariationGenerator()

@pytest.fixture
def csv_semicolon_format(tmp_path, base_synthetic_data, csv_variation_generator) -> Path:
    """CSV with semicolon delimiter (nirs4all default)."""
    X, y, wavelengths = base_synthetic_data
    return csv_variation_generator.with_semicolon_delimiter(
        tmp_path / "semicolon",
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8,
        random_state=42,
    )

@pytest.fixture
def csv_comma_format(tmp_path, base_synthetic_data, csv_variation_generator) -> Path:
    """CSV with comma delimiter."""
    X, y, wavelengths = base_synthetic_data
    return csv_variation_generator.with_comma_delimiter(
        tmp_path / "comma",
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8,
        random_state=42,
    )

@pytest.fixture
def csv_tab_format(tmp_path, base_synthetic_data, csv_variation_generator) -> Path:
    """CSV with tab delimiter (.tsv)."""
    X, y, wavelengths = base_synthetic_data
    return csv_variation_generator.with_tab_delimiter(
        tmp_path / "tab",
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8,
        random_state=42,
    )

@pytest.fixture
def csv_no_headers_format(tmp_path, base_synthetic_data, csv_variation_generator) -> Path:
    """CSV without column headers."""
    X, y, _ = base_synthetic_data
    return csv_variation_generator.without_headers(
        tmp_path / "no_headers",
        X, y,
        train_ratio=0.8,
        random_state=42,
    )

@pytest.fixture
def csv_with_index_format(tmp_path, base_synthetic_data, csv_variation_generator) -> Path:
    """CSV with row index column."""
    X, y, wavelengths = base_synthetic_data
    return csv_variation_generator.with_row_index(
        tmp_path / "with_index",
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8,
        random_state=42,
    )

@pytest.fixture
def csv_all_variations(tmp_path, base_synthetic_data, csv_variation_generator) -> dict:
    """
    Generate all CSV format variations.

    Returns:
        dict: Mapping of variation name to Path.
    """
    X, y, wavelengths = base_synthetic_data
    return csv_variation_generator.generate_all_variations(
        tmp_path / "variations",
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8,
        random_state=42,
    )

