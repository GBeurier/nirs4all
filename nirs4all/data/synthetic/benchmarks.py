"""
Benchmark dataset utilities for synthetic data validation.

This module provides information about standard NIR benchmark datasets
that can be used to validate synthetic data quality.

Phase 4 Features:
    - Benchmark dataset registry with metadata
    - Dataset characteristic summaries
    - Reference spectral properties
    - Loader utilities for common formats

Note:
    This module provides metadata and loading utilities for benchmark datasets.
    The actual dataset files need to be obtained from their respective sources
    due to licensing restrictions.

References:
    - Corn (Cargill): M5spec competition dataset
    - Tecator (meat): StatLib - meat protein/fat/moisture
    - Shootout 2002: IDRC shootout pharmaceutical tablets
    - Wheat: Hard red wheat kernels dataset
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# ============================================================================
# Benchmark Dataset Registry
# ============================================================================


class BenchmarkDomain(str, Enum):
    """Domains for benchmark datasets."""
    AGRICULTURE = "agriculture"
    FOOD = "food"
    PHARMACEUTICAL = "pharmaceutical"
    PETROCHEMICAL = "petrochemical"
    ENVIRONMENTAL = "environmental"
    GENERAL = "general"


@dataclass
class BenchmarkDatasetInfo:
    """
    Metadata for a benchmark dataset.

    Attributes:
        name: Dataset name/identifier.
        full_name: Full descriptive name.
        domain: Application domain.
        n_samples: Number of samples (approximate if variable).
        n_wavelengths: Number of wavelength points.
        wavelength_range: (min, max) wavelength in nm.
        targets: List of target variable names.
        sample_type: Description of sample type.
        measurement_mode: Typical measurement mode.
        source_url: URL to obtain the dataset.
        reference: Publication or source reference.
        license: License information.
        typical_snr: Typical signal-to-noise ratio range.
        typical_peak_density: Typical peaks per 100 nm.
        notes: Additional notes.
    """
    name: str
    full_name: str
    domain: BenchmarkDomain
    n_samples: int
    n_wavelengths: int
    wavelength_range: Tuple[float, float]
    targets: List[str]
    sample_type: str
    measurement_mode: str
    source_url: str
    reference: str
    license: str = "Unknown"
    typical_snr: Tuple[float, float] = (50, 500)
    typical_peak_density: Tuple[float, float] = (1.0, 5.0)
    notes: str = ""

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Dataset: {self.full_name} ({self.name})",
            f"Domain: {self.domain.value}",
            f"Samples: {self.n_samples}",
            f"Wavelengths: {self.n_wavelengths} ({self.wavelength_range[0]}-{self.wavelength_range[1]} nm)",
            f"Targets: {', '.join(self.targets)}",
            f"Sample Type: {self.sample_type}",
            f"Measurement: {self.measurement_mode}",
            f"Source: {self.source_url}",
            f"Reference: {self.reference}",
        ]
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)


# Registry of benchmark datasets
BENCHMARK_DATASETS: Dict[str, BenchmarkDatasetInfo] = {
    "corn": BenchmarkDatasetInfo(
        name="corn",
        full_name="Corn/Maize M5spec Dataset",
        domain=BenchmarkDomain.AGRICULTURE,
        n_samples=80,
        n_wavelengths=700,
        wavelength_range=(1100, 2498),
        targets=["moisture", "oil", "protein", "starch"],
        sample_type="Ground corn samples",
        measurement_mode="reflectance",
        source_url="http://www.eigenvector.com/data/Corn/",
        reference="Eigenvector Research, Cargill Inc.",
        license="Free for research use",
        typical_snr=(100, 500),
        typical_peak_density=(1.5, 4.0),
        notes="Classic small sample size calibration challenge. 3 instruments.",
    ),

    "tecator": BenchmarkDatasetInfo(
        name="tecator",
        full_name="Tecator Meat Dataset",
        domain=BenchmarkDomain.FOOD,
        n_samples=215,
        n_wavelengths=100,
        wavelength_range=(850, 1050),
        targets=["fat", "moisture", "protein"],
        sample_type="Finely chopped meat samples",
        measurement_mode="transmittance",
        source_url="http://lib.stat.cmu.edu/datasets/tecator",
        reference="Tecator Infratec Food and Feed Analyzer",
        license="Free for academic use",
        typical_snr=(200, 1000),
        typical_peak_density=(0.5, 2.0),
        notes="Wet meat samples. Narrow wavelength range.",
    ),

    "shootout2002": BenchmarkDatasetInfo(
        name="shootout2002",
        full_name="IDRC Shootout 2002 Pharmaceutical Tablets",
        domain=BenchmarkDomain.PHARMACEUTICAL,
        n_samples=654,
        n_wavelengths=404,
        wavelength_range=(600, 1898),
        targets=["api_content", "hardness", "active_weight"],
        sample_type="Pharmaceutical tablets",
        measurement_mode="reflectance",
        source_url="http://www.idrc-chambersburg.org/shootout.html",
        reference="IDRC (International Diffuse Reflectance Conference)",
        license="Free for research use",
        typical_snr=(100, 400),
        typical_peak_density=(2.0, 5.0),
        notes="Blend uniformity challenge. Multiple manufacturing lots.",
    ),

    "wheat_kernels": BenchmarkDatasetInfo(
        name="wheat_kernels",
        full_name="Hard Red Wheat Kernels",
        domain=BenchmarkDomain.AGRICULTURE,
        n_samples=155,
        n_wavelengths=100,
        wavelength_range=(1100, 2498),
        targets=["protein", "moisture", "hardness"],
        sample_type="Intact wheat kernels",
        measurement_mode="reflectance",
        source_url="http://www.eigenvector.com/data/Wheat/",
        reference="Eigenvector Research",
        license="Free for research use",
        typical_snr=(50, 300),
        typical_peak_density=(1.0, 3.0),
        notes="Intact kernel analysis (not ground). High scatter variation.",
    ),

    "diesel": BenchmarkDatasetInfo(
        name="diesel",
        full_name="Diesel Fuel NIR Dataset",
        domain=BenchmarkDomain.PETROCHEMICAL,
        n_samples=245,
        n_wavelengths=401,
        wavelength_range=(750, 1550),
        targets=["cetane", "density", "viscosity", "total_aromatics"],
        sample_type="Diesel fuel samples",
        measurement_mode="transmittance",
        source_url="http://www.eigenvector.com/data/SWRI/",
        reference="Southwest Research Institute",
        license="Free for research use",
        typical_snr=(300, 1000),
        typical_peak_density=(0.5, 2.0),
        notes="Clear liquid samples. Low scattering.",
    ),

    "tablet_api": BenchmarkDatasetInfo(
        name="tablet_api",
        full_name="Tablet Active Pharmaceutical Ingredient Dataset",
        domain=BenchmarkDomain.PHARMACEUTICAL,
        n_samples=310,
        n_wavelengths=650,
        wavelength_range=(1100, 2498),
        targets=["api_concentration"],
        sample_type="Intact pharmaceutical tablets",
        measurement_mode="reflectance",
        source_url="Various publications",
        reference="Multiple sources",
        license="Various",
        typical_snr=(80, 400),
        typical_peak_density=(2.0, 6.0),
        notes="Typical intact tablet analysis scenario.",
    ),

    "milk": BenchmarkDatasetInfo(
        name="milk",
        full_name="Milk Composition Dataset",
        domain=BenchmarkDomain.FOOD,
        n_samples=300,
        n_wavelengths=1050,
        wavelength_range=(400, 2500),
        targets=["fat", "protein", "lactose"],
        sample_type="Raw milk samples",
        measurement_mode="transflectance",
        source_url="Various dairy research",
        reference="Dairy research literature",
        license="Various",
        typical_snr=(200, 800),
        typical_peak_density=(1.5, 4.0),
        notes="Emulsion samples. Water dominates spectrum.",
    ),

    "olive_oil": BenchmarkDatasetInfo(
        name="olive_oil",
        full_name="Olive Oil Authenticity Dataset",
        domain=BenchmarkDomain.FOOD,
        n_samples=120,
        n_wavelengths=1050,
        wavelength_range=(400, 2500),
        targets=["adulterant_fraction", "acidity", "peroxide_value"],
        sample_type="Olive oil samples",
        measurement_mode="transmittance",
        source_url="Various food authenticity research",
        reference="Food authenticity literature",
        license="Various",
        typical_snr=(400, 1200),
        typical_peak_density=(0.5, 2.0),
        notes="Clear liquid. Classification and regression tasks.",
    ),
}


# ============================================================================
# Dataset Loader Utilities
# ============================================================================


@dataclass
class LoadedBenchmarkDataset:
    """
    Container for a loaded benchmark dataset.

    Attributes:
        info: Dataset metadata.
        X: Spectral data (n_samples, n_wavelengths).
        y: Target values (n_samples, n_targets) or (n_samples,).
        wavelengths: Wavelength array.
        sample_ids: Optional sample identifiers.
        metadata: Optional additional metadata.
    """
    info: BenchmarkDatasetInfo
    X: np.ndarray
    y: np.ndarray
    wavelengths: np.ndarray
    sample_ids: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def list_benchmark_datasets() -> List[str]:
    """
    List all registered benchmark datasets.

    Returns:
        List of dataset names.

    Example:
        >>> datasets = list_benchmark_datasets()
        >>> print(datasets)
    """
    return list(BENCHMARK_DATASETS.keys())


def get_benchmark_info(name: str) -> BenchmarkDatasetInfo:
    """
    Get information about a benchmark dataset.

    Args:
        name: Dataset name.

    Returns:
        BenchmarkDatasetInfo for the dataset.

    Raises:
        KeyError: If dataset not found.

    Example:
        >>> info = get_benchmark_info("corn")
        >>> print(info.summary())
    """
    if name not in BENCHMARK_DATASETS:
        available = ", ".join(BENCHMARK_DATASETS.keys())
        raise KeyError(f"Unknown benchmark dataset '{name}'. Available: {available}")
    return BENCHMARK_DATASETS[name]


def get_datasets_by_domain(domain: Union[str, BenchmarkDomain]) -> List[str]:
    """
    Get benchmark datasets for a specific domain.

    Args:
        domain: Domain name or enum.

    Returns:
        List of dataset names in that domain.

    Example:
        >>> pharma_datasets = get_datasets_by_domain("pharmaceutical")
        >>> print(pharma_datasets)
    """
    if isinstance(domain, str):
        domain = BenchmarkDomain(domain)

    return [
        name for name, info in BENCHMARK_DATASETS.items()
        if info.domain == domain
    ]


def load_benchmark_dataset(
    name: str,
    data_dir: Optional[Union[str, Path]] = None,
    format: str = "auto",
) -> LoadedBenchmarkDataset:
    """
    Load a benchmark dataset from disk.

    Args:
        name: Dataset name from registry.
        data_dir: Directory containing dataset files.
        format: File format ("auto", "csv", "mat", "jdx").

    Returns:
        LoadedBenchmarkDataset with data.

    Raises:
        FileNotFoundError: If dataset files not found.
        KeyError: If dataset name not in registry.

    Example:
        >>> dataset = load_benchmark_dataset("corn", data_dir="./datasets/")
        >>> print(dataset.X.shape, dataset.y.shape)

    Note:
        Dataset files must be obtained separately from their sources.
        This function provides standardized loading once files are available.
    """
    info = get_benchmark_info(name)

    if data_dir is None:
        raise FileNotFoundError(
            f"Dataset '{name}' requires data_dir parameter. "
            f"Please obtain the dataset from: {info.source_url}"
        )

    data_dir = Path(data_dir)

    # Try common file patterns
    possible_files = [
        data_dir / f"{name}.csv",
        data_dir / f"{name}.mat",
        data_dir / f"{name}_spectra.csv",
        data_dir / name / "spectra.csv",
        data_dir / name / f"{name}.csv",
    ]

    data_file = None
    for f in possible_files:
        if f.exists():
            data_file = f
            break

    if data_file is None:
        raise FileNotFoundError(
            f"Could not find dataset files for '{name}' in {data_dir}. "
            f"Tried: {[str(f) for f in possible_files]}"
        )

    # Load based on format
    if format == "auto":
        format = data_file.suffix.lstrip(".")

    if format == "csv":
        return _load_csv_dataset(data_file, info)
    elif format == "mat":
        return _load_mat_dataset(data_file, info)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _load_csv_dataset(
    filepath: Path,
    info: BenchmarkDatasetInfo,
) -> LoadedBenchmarkDataset:
    """Load dataset from CSV format."""
    import csv

    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            data.append([float(x) if x else np.nan for x in row])

    data = np.array(data)

    # Assume first few columns are targets, rest are spectra
    n_targets = len(info.targets)
    y = data[:, :n_targets]
    X = data[:, n_targets:]

    # Generate wavelength array
    wl_start, wl_end = info.wavelength_range
    wavelengths = np.linspace(wl_start, wl_end, X.shape[1])

    return LoadedBenchmarkDataset(
        info=info,
        X=X,
        y=y,
        wavelengths=wavelengths,
    )


def _load_mat_dataset(
    filepath: Path,
    info: BenchmarkDatasetInfo,
) -> LoadedBenchmarkDataset:
    """Load dataset from MATLAB .mat format."""
    from scipy.io import loadmat

    mat_data = loadmat(str(filepath))

    # Common variable names in .mat files
    X = None
    y = None
    wavelengths = None

    for key in ['X', 'spectra', 'Spectra', 'NIR']:
        if key in mat_data:
            X = mat_data[key]
            break

    for key in ['Y', 'y', 'targets', 'Targets', 'reference']:
        if key in mat_data:
            y = mat_data[key]
            break

    for key in ['wavelengths', 'wl', 'Wavelengths', 'nm']:
        if key in mat_data:
            wavelengths = mat_data[key].flatten()
            break

    if X is None:
        raise ValueError(f"Could not find spectral data in {filepath}")
    if y is None:
        # Create dummy targets
        y = np.zeros((X.shape[0], 1))

    if wavelengths is None:
        wl_start, wl_end = info.wavelength_range
        wavelengths = np.linspace(wl_start, wl_end, X.shape[1])

    return LoadedBenchmarkDataset(
        info=info,
        X=X,
        y=y,
        wavelengths=wavelengths,
    )


# ============================================================================
# Synthetic Dataset Generation Matching Benchmark
# ============================================================================


def get_benchmark_spectral_properties(name: str) -> Dict[str, Any]:
    """
    Get spectral properties to match when generating synthetic data.

    Args:
        name: Benchmark dataset name.

    Returns:
        Dictionary of properties suitable for synthetic generator.

    Example:
        >>> props = get_benchmark_spectral_properties("corn")
        >>> generator = SyntheticNIRSGenerator(**props)
    """
    info = get_benchmark_info(name)

    # Map domain to likely components
    domain_components = {
        BenchmarkDomain.AGRICULTURE: ["water", "protein", "starch", "cellulose", "lipid"],
        BenchmarkDomain.FOOD: ["water", "protein", "lipid", "glucose", "lactose"],
        BenchmarkDomain.PHARMACEUTICAL: ["cellulose", "starch", "paracetamol", "water"],
        BenchmarkDomain.PETROCHEMICAL: ["alkane", "aromatic", "oil"],
    }

    return {
        "wavelength_start": info.wavelength_range[0],
        "wavelength_end": info.wavelength_range[1],
        "wavelength_step": (info.wavelength_range[1] - info.wavelength_range[0]) / info.n_wavelengths,
        "measurement_mode": info.measurement_mode,
        "typical_components": domain_components.get(info.domain, ["water", "protein"]),
        "n_samples": info.n_samples,
        "expected_snr": info.typical_snr,
        "expected_peak_density": info.typical_peak_density,
    }


def create_synthetic_matching_benchmark(
    benchmark_name: str,
    n_samples: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic data matching benchmark dataset properties.

    Args:
        benchmark_name: Name of benchmark dataset to match.
        n_samples: Number of samples (uses benchmark size if None).
        random_state: Random state for reproducibility.

    Returns:
        Tuple of (spectra, concentrations, component_spectra).

    Example:
        >>> X, C, E = create_synthetic_matching_benchmark("corn", random_state=42)
        >>> print(X.shape)
    """
    # Import here to avoid circular imports
    from .generator import SyntheticNIRSGenerator
    from .components import ComponentLibrary

    props = get_benchmark_spectral_properties(benchmark_name)

    # Create component library
    library = ComponentLibrary.from_predefined(props["typical_components"])

    # Create generator
    generator = SyntheticNIRSGenerator(
        component_library=library,
        wavelength_start=props["wavelength_start"],
        wavelength_end=props["wavelength_end"],
        wavelength_step=props["wavelength_step"],
        random_state=random_state,
    )

    # Generate
    if n_samples is None:
        n_samples = props["n_samples"]

    X, C, E = generator.generate(n_samples=n_samples)

    return X, C, E
