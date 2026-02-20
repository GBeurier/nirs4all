"""
CSV loader test fixtures with format variations.

These fixtures generate CSV files in various formats to test
the loader's ability to handle different file configurations.

This module provides specialized fixtures for testing data loaders with
different CSV format variations. It uses the synthetic data generator
from nirs4all.synthesis to create consistent, reproducible test data.

Fixture Naming Convention:
    csv_{format}_format: Individual format fixtures
    csv_all_formats: Parametrized fixture for all formats
    csv_basic_variations: Common format variations
    csv_edge_cases: Edge case formats

See Also:
    tests/conftest.py: Core fixtures including base_synthetic_data
    nirs4all.synthesis.CSVVariationGenerator: CSV variation generator
"""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from nirs4all.synthesis import CSVVariationGenerator, SyntheticDatasetBuilder

# =============================================================================
# Module-scoped Base Data (Shared Across All Tests in This Module)
# =============================================================================

@pytest.fixture(scope="module")
def loader_base_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Base synthetic data for all loader CSV format tests.

    Module-scoped to generate data once and reuse across all loader tests.

    Returns:
        Tuple containing:
            - X: Feature matrix (50 samples, 101 features)
            - y: Target values (50,) - single target
            - wavelengths: Wavelength grid (1000-2000nm, step 10)
    """
    builder = (
        SyntheticDatasetBuilder(n_samples=50, random_state=42)
        .with_features(
            wavelength_range=(1000, 2000),
            wavelength_step=10,
            complexity="simple"
        )
        .with_targets(distribution="uniform", range=(10, 50), component=0)
    )
    X, y = builder.build_arrays()
    wavelengths = np.arange(1000, 2001, 10)
    # Ensure y is 1D for single-target loader tests
    if y.ndim > 1:
        y = y[:, 0]
    return X, y, wavelengths

@pytest.fixture(scope="module")
def loader_multi_target_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-target data for loader tests.

    Returns:
        Tuple containing X, y (multi-column), wavelengths.
    """
    builder = (
        SyntheticDatasetBuilder(n_samples=50, random_state=42)
        .with_features(
            wavelength_range=(1000, 2000),
            wavelength_step=10,
            complexity="simple",
            components=["water", "protein", "lipid"],
        )
        .with_targets(distribution="dirichlet")
    )
    X, y = builder.build_arrays()
    wavelengths = np.arange(1000, 2001, 10)
    # Ensure y is 2D for multi-target
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return X, y, wavelengths

@pytest.fixture(scope="module")
def loader_csv_generator() -> CSVVariationGenerator:
    """Get a shared CSVVariationGenerator instance."""
    return CSVVariationGenerator()

# =============================================================================
# Individual Format Fixtures
# =============================================================================

@pytest.fixture
def csv_standard_format(tmp_path, loader_base_data, loader_csv_generator) -> Path:
    """
    Standard nirs4all CSV format.

    Format:
        - Semicolon delimiter
        - Wavelength headers
        - Separate Xcal/Ycal/Xval/Yval files
    """
    X, y, wavelengths = loader_base_data
    return cast(Path, loader_csv_generator.with_semicolon_delimiter(
        tmp_path / "standard",
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8,
        random_state=42,
    ))

@pytest.fixture
def csv_comma_delimiter(tmp_path, loader_base_data, loader_csv_generator) -> Path:
    """CSV with comma delimiter (common international format)."""
    X, y, wavelengths = loader_base_data
    return cast(Path, loader_csv_generator.with_comma_delimiter(
        tmp_path / "comma",
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8,
        random_state=42,
    ))

@pytest.fixture
def csv_tab_delimiter(tmp_path, loader_base_data, loader_csv_generator) -> Path:
    """CSV with tab delimiter (.tsv format)."""
    X, y, wavelengths = loader_base_data
    return cast(Path, loader_csv_generator.with_tab_delimiter(
        tmp_path / "tab",
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8,
        random_state=42,
    ))

@pytest.fixture
def csv_no_headers(tmp_path, loader_base_data, loader_csv_generator) -> Path:
    """CSV without column headers."""
    X, y, _ = loader_base_data
    return cast(Path, loader_csv_generator.without_headers(
        tmp_path / "no_headers",
        X, y,
        train_ratio=0.8,
        random_state=42,
    ))

@pytest.fixture
def csv_with_index(tmp_path, loader_base_data, loader_csv_generator) -> Path:
    """CSV with row index as first column."""
    X, y, wavelengths = loader_base_data
    return cast(Path, loader_csv_generator.with_row_index(
        tmp_path / "with_index",
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8,
        random_state=42,
    ))

@pytest.fixture
def csv_single_file(tmp_path, loader_base_data, loader_csv_generator) -> Path:
    """Single CSV file with all data and partition column."""
    X, y, wavelengths = loader_base_data
    return cast(Path, loader_csv_generator.as_single_file(
        tmp_path / "single_file",
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8,
        random_state=42,
    ))

@pytest.fixture
def csv_fragmented(tmp_path, loader_base_data, loader_csv_generator) -> Path:
    """Fragmented dataset with multiple small files."""
    X, y, wavelengths = loader_base_data
    return cast(Path, loader_csv_generator.as_fragmented(
        tmp_path / "fragmented",
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8,
        random_state=42,
    ))

@pytest.fixture
def csv_low_precision(tmp_path, loader_base_data, loader_csv_generator) -> Path:
    """CSV with low floating point precision (2 decimals)."""
    X, y, wavelengths = loader_base_data
    return cast(Path, loader_csv_generator.with_precision(
        tmp_path / "low_precision",
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8,
        random_state=42,
        precision=2,
    ))

@pytest.fixture
def csv_high_precision(tmp_path, loader_base_data, loader_csv_generator) -> Path:
    """CSV with high floating point precision (10 decimals)."""
    X, y, wavelengths = loader_base_data
    return cast(Path, loader_csv_generator.with_precision(
        tmp_path / "high_precision",
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8,
        random_state=42,
        precision=10,
    ))

# =============================================================================
# Edge Case Fixtures
# =============================================================================

@pytest.fixture
def csv_with_missing_values(tmp_path, loader_base_data) -> tuple[Path, float]:
    """
    CSV with random missing values (NaN).

    Returns:
        Tuple of (path, missing_ratio).
    """
    X, y, wavelengths = loader_base_data
    X_copy = X.copy()

    # Introduce random missing values (2% of data)
    missing_ratio = 0.02
    rng = np.random.default_rng(42)
    mask = rng.random(X_copy.shape) < missing_ratio
    X_copy[mask] = np.nan

    # Create folder and save manually
    path = tmp_path / "missing_values"
    path.mkdir(parents=True, exist_ok=True)

    n_train = int(len(X_copy) * 0.8)

    # Save with missing values
    df_train = pd.DataFrame(X_copy[:n_train])
    df_train.to_csv(path / "Xcal.csv", sep=";", index=False)

    df_y_train = pd.DataFrame(y[:n_train])
    df_y_train.to_csv(path / "Ycal.csv", sep=";", index=False)

    df_test = pd.DataFrame(X_copy[n_train:])
    df_test.to_csv(path / "Xval.csv", sep=";", index=False)

    df_y_test = pd.DataFrame(y[n_train:])
    df_y_test.to_csv(path / "Yval.csv", sep=";", index=False)

    return cast(Path, path), missing_ratio

@pytest.fixture
def csv_with_text_headers(tmp_path, loader_base_data) -> Path:
    """CSV with text column headers instead of wavelengths."""
    X, y, _ = loader_base_data
    n_features = X.shape[1]

    path = tmp_path / "text_headers"
    path.mkdir(parents=True, exist_ok=True)

    # Create text headers
    headers = [f"feature_{i}" for i in range(n_features)]

    n_train = int(len(X) * 0.8)

    # Ensure y is 1D for DataFrame
    y_1d = y.ravel() if y.ndim > 1 else y

    df_train = pd.DataFrame(X[:n_train], columns=headers)
    df_train.to_csv(path / "Xcal.csv", sep=";", index=False)

    df_y_train = pd.DataFrame({"target": y_1d[:n_train]})
    df_y_train.to_csv(path / "Ycal.csv", sep=";", index=False)

    df_test = pd.DataFrame(X[n_train:], columns=headers)
    df_test.to_csv(path / "Xval.csv", sep=";", index=False)

    df_y_test = pd.DataFrame({"target": y_1d[n_train:]})
    df_y_test.to_csv(path / "Yval.csv", sep=";", index=False)

    return cast(Path, path)

@pytest.fixture
def csv_european_decimals(tmp_path, loader_base_data) -> Path:
    """
    CSV with European decimal format.

    Format:
        - Decimal separator: comma (,)
        - Field separator: semicolon (;)
        - Example: 1,234;5,678;9,012
    """
    X, y, wavelengths = loader_base_data

    path = tmp_path / "european"
    path.mkdir(parents=True, exist_ok=True)

    n_train = int(len(X) * 0.8)

    # Save with comma decimal separator
    def save_european(filepath, data, headers=None):
        with open(filepath, 'w') as f:
            if headers:
                f.write(";".join(headers) + "\n")
            for row in data:
                # Convert floats to European format (comma as decimal)
                row_str = ";".join(f"{val:.6f}".replace(".", ",") for val in row)
                f.write(row_str + "\n")

    headers = [str(int(wl)) for wl in wavelengths]
    save_european(path / "Xcal.csv", X[:n_train], headers)
    save_european(path / "Ycal.csv", y[:n_train].reshape(-1, 1), ["target"])
    save_european(path / "Xval.csv", X[n_train:], headers)
    save_european(path / "Yval.csv", y[n_train:].reshape(-1, 1), ["target"])

    return cast(Path, path)

@pytest.fixture
def csv_multi_source(tmp_path, loader_base_data) -> Path:
    """
    Multi-source CSV format with X split across multiple files.

    Creates:
        - Xcal_1.csv, Xcal_2.csv, Xcal_3.csv (features split)
        - Ycal.csv (single file)
        - Xval_1.csv, Xval_2.csv, Xval_3.csv
        - Yval.csv
    """
    X, y, wavelengths = loader_base_data

    path = tmp_path / "multi_source"
    path.mkdir(parents=True, exist_ok=True)

    n_train = int(len(X) * 0.8)
    n_features = X.shape[1]
    n_files = 3
    chunk_size = n_features // n_files

    # Split features across multiple files
    for i in range(n_files):
        start = i * chunk_size
        end = start + chunk_size if i < n_files - 1 else n_features

        X_chunk = X[:, start:end]
        chunk_wavelengths = wavelengths[start:end]
        headers = [str(int(wl)) for wl in chunk_wavelengths]

        # Training data
        df_train = pd.DataFrame(X_chunk[:n_train], columns=headers)
        df_train.to_csv(path / f"Xcal_{i+1}.csv", sep=";", index=False)

        # Test data
        df_test = pd.DataFrame(X_chunk[n_train:], columns=headers)
        df_test.to_csv(path / f"Xval_{i+1}.csv", sep=";", index=False)

    # Save Y files - ensure y is 1D
    y_1d = y.ravel() if y.ndim > 1 else y
    df_y_train = pd.DataFrame({"target": y_1d[:n_train]})
    df_y_train.to_csv(path / "Ycal.csv", sep=";", index=False)

    df_y_test = pd.DataFrame({"target": y_1d[n_train:]})
    df_y_test.to_csv(path / "Yval.csv", sep=";", index=False)

    return cast(Path, path)

@pytest.fixture
def csv_single_file_all_data(tmp_path, loader_base_data) -> Path:
    """
    Single CSV with X, y, and metadata columns all in one file.

    Columns: [sample_id, group, wavelengths..., target]
    """
    X, y, wavelengths = loader_base_data

    path = tmp_path / "single_all"
    path.mkdir(parents=True, exist_ok=True)

    n_samples = len(X)
    n_train = int(n_samples * 0.8)

    # Create partition assignment
    rng = np.random.default_rng(42)
    indices = rng.permutation(n_samples)
    partition = np.array(["train"] * n_samples)
    partition[indices[n_train:]] = "test"

    # Ensure y is 1D
    y_1d = y.ravel() if y.ndim > 1 else y

    # Build data dict
    data = {
        "sample_id": [f"S{i:04d}" for i in range(n_samples)],
        "partition": partition,
        "group": np.random.choice(["A", "B", "C"], size=n_samples),
    }

    # Add features
    for i, wl in enumerate(wavelengths):
        data[str(int(wl))] = X[:, i]

    # Add target
    data["target"] = y_1d

    df = pd.DataFrame(data)
    df.to_csv(path / "data.csv", sep=";", index=False)

    return cast(Path, path)

# =============================================================================
# Parametrized Fixtures for Comprehensive Testing
# =============================================================================

@pytest.fixture(params=[
    "standard",
    "comma",
    "tab",
    "no_headers",
    "with_index",
])
def csv_basic_variations(
    request,
    tmp_path,
    loader_base_data,
    loader_csv_generator,
) -> tuple[Path, dict]:
    """
    Parametrized fixture for basic CSV format variations.

    Tests using this fixture run 5x with different formats.

    Returns:
        Tuple of (path, format_info) where format_info describes the format.
    """
    X, y, wavelengths = loader_base_data

    format_configs = {
        "standard": {
            "method": loader_csv_generator.with_semicolon_delimiter,
            "info": {"delimiter": ";", "has_header": True, "has_index": False},
        },
        "comma": {
            "method": loader_csv_generator.with_comma_delimiter,
            "info": {"delimiter": ",", "has_header": True, "has_index": False},
        },
        "tab": {
            "method": loader_csv_generator.with_tab_delimiter,
            "info": {"delimiter": "\t", "has_header": True, "has_index": False},
        },
        "no_headers": {
            "method": loader_csv_generator.without_headers,
            "info": {"delimiter": ";", "has_header": False, "has_index": False},
        },
        "with_index": {
            "method": loader_csv_generator.with_row_index,
            "info": {"delimiter": ";", "has_header": True, "has_index": True},
        },
    }

    config = format_configs[request.param]

    if request.param == "no_headers":
        path = config["method"](
            tmp_path / request.param,
            X, y,
            train_ratio=0.8,
            random_state=42,
        )
    else:
        path = config["method"](
            tmp_path / request.param,
            X, y,
            wavelengths=wavelengths,
            train_ratio=0.8,
            random_state=42,
        )

    return cast(Path, path), config["info"]

@pytest.fixture(params=["standard", "single_file", "fragmented"])
def csv_file_structures(
    request,
    tmp_path,
    loader_base_data,
    loader_csv_generator,
) -> tuple[Path, str]:
    """
    Parametrized fixture for different file structure variations.

    Tests using this fixture run 3x with different structures.

    Returns:
        Tuple of (path, structure_type).
    """
    X, y, wavelengths = loader_base_data

    structure_methods = {
        "standard": loader_csv_generator.with_semicolon_delimiter,
        "single_file": loader_csv_generator.as_single_file,
        "fragmented": loader_csv_generator.as_fragmented,
    }

    method = structure_methods[request.param]
    path = method(
        tmp_path / request.param,
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8,
        random_state=42,
    )

    return cast(Path, path), request.param

@pytest.fixture(params=[2, 6, 10])
def csv_precision_levels(
    request,
    tmp_path,
    loader_base_data,
    loader_csv_generator,
) -> tuple[Path, int]:
    """
    Parametrized fixture for different precision levels.

    Tests using this fixture run 3x with different precisions.

    Returns:
        Tuple of (path, precision).
    """
    X, y, wavelengths = loader_base_data

    path = loader_csv_generator.with_precision(
        tmp_path / f"precision_{request.param}",
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8,
        random_state=42,
        precision=request.param,
    )

    return cast(Path, path), request.param

# =============================================================================
# All Variations Fixture
# =============================================================================

@pytest.fixture
def csv_all_loader_variations(
    tmp_path,
    loader_base_data,
    loader_csv_generator,
) -> dict[str, Path]:
    """
    Generate all CSV format variations for comprehensive testing.

    Returns:
        Dict mapping variation name to Path.
    """
    X, y, wavelengths = loader_base_data
    return cast(dict[str, Path], loader_csv_generator.generate_all_variations(
        tmp_path / "all_variations",
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8,
        random_state=42,
    ))
