"""
Generate 30 test datasets with matching YAML configurations.

This script generates synthetic NIRS datasets covering all DatasetConfig schema options,
along with YAML configuration files that can load them. The datasets are designed to test
the full capability of the dataset wizard in nirs4all_webapp.

Usage:
    python -m examples.scripts.generate_test_datasets [--output-dir PATH] [--seed INT]

Categories:
    A: File Format & Delimiter Variations (5 configs)
    B: Loading Parameters (5 configs)
    C: File Structures (5 configs)
    D: Partition & Split Strategies (5 configs)
    E: Multi-Source & Aggregation (5 configs)
    F: Task Types & Variations (5 configs)
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from nirs4all.synthesis import SyntheticDatasetBuilder, DatasetExporter, ExportConfig


@dataclass
class DatasetSpec:
    """Specification for a single test dataset."""

    name: str
    category: str
    description: str
    n_samples: int = 60
    n_train: int = 48
    n_test: int = 12
    n_features: int = 200
    wavelength_start: float = 1100.0
    wavelength_end: float = 2100.0
    # Export options
    delimiter: str = ";"
    decimal_separator: str = "."
    has_header: bool = True
    file_extension: str = ".csv"
    compression: Optional[str] = None
    # Dataset structure
    structure: Literal["standard", "single", "legacy", "fragmented"] = "standard"
    # Task type
    task_type: Literal["regression", "binary", "multiclass"] = "regression"
    n_classes: int = 2
    # Header unit
    header_unit: str = "nm"
    signal_type: str = "absorbance"
    # Encoding
    encoding: str = "utf-8"
    skip_rows: int = 0
    # Multi-source
    multi_source: bool = False
    sources: Optional[List[Dict[str, Any]]] = None
    # Partition
    partition_method: Optional[str] = None
    # Repetition (was: Aggregation)
    repetition: bool = False
    repetition_column: Optional[str] = None
    repetition_method: str = "mean"
    # Variations
    variations: bool = False
    variation_mode: Optional[str] = None
    # Folds
    custom_folds: bool = False
    # Metadata
    include_metadata: bool = False
    # Extra config options
    extra_config: Dict[str, Any] = field(default_factory=dict)


class TestDatasetGenerator:
    """Generates all 30 test datasets with matching YAML configs."""

    def __init__(
        self,
        output_dir: Path,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.verbose = verbose
        self.specs: List[DatasetSpec] = []
        self.datasets_dir = self.output_dir / "sample_datasets"
        self.configs_dir = self.output_dir / "sample_configs" / "datasets"

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def register_all_specs(self) -> None:
        """Register all 30 dataset specifications."""
        self._register_category_a()  # File formats & delimiters
        self._register_category_b()  # Loading parameters
        self._register_category_c()  # File structures
        self._register_category_d()  # Partitions
        self._register_category_e()  # Multi-source & aggregation
        self._register_category_f()  # Task types & variations

    def generate_all(self) -> None:
        """Generate all datasets and configs."""
        self.register_all_specs()

        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

        for i, spec in enumerate(self.specs, 1):
            self.log(f"[{i:02d}/30] Generating {spec.name}...")
            self._generate_dataset(spec)

        self.log(f"\nGenerated {len(self.specs)} datasets in {self.datasets_dir}")
        self.log(f"Generated {len(self.specs)} configs in {self.configs_dir}")

    def _generate_dataset(self, spec: DatasetSpec) -> None:
        """Generate a single dataset and its config."""
        dataset_dir = self.datasets_dir / spec.name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(self.seed + hash(spec.name) % 10000)

        # Generate synthetic data
        wavelengths = np.linspace(spec.wavelength_start, spec.wavelength_end, spec.n_features)
        X, y, metadata = self._generate_synthetic_data(spec, wavelengths, rng)

        # Export based on structure type
        if spec.multi_source:
            self._export_multi_source(spec, dataset_dir, X, y, wavelengths, metadata, rng)
        elif spec.variations:
            self._export_variations(spec, dataset_dir, X, y, wavelengths, metadata, rng)
        elif spec.structure == "standard":
            self._export_standard(spec, dataset_dir, X, y, wavelengths, metadata, rng)
        elif spec.structure == "single":
            self._export_single(spec, dataset_dir, X, y, wavelengths, metadata, rng)
        elif spec.structure == "legacy":
            self._export_legacy(spec, dataset_dir, X, y, wavelengths, metadata, rng)
        elif spec.structure == "fragmented":
            self._export_fragmented(spec, dataset_dir, X, y, wavelengths, metadata, rng)

        # Generate YAML config
        config = self._generate_config(spec, dataset_dir)
        config_path = self.configs_dir / f"{spec.name}.yaml"
        self._write_yaml(config_path, config, spec.description)

    def _generate_synthetic_data(
        self,
        spec: DatasetSpec,
        wavelengths: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Generate synthetic spectral data."""
        n_samples = spec.n_samples
        n_features = len(wavelengths)

        # Generate base spectra using simplified physics
        n_components = 4
        concentrations = rng.dirichlet([1.0] * n_components, size=n_samples)

        # Create absorption profiles for each component
        component_spectra = []
        for i in range(n_components):
            center = wavelengths[0] + (i + 1) * (wavelengths[-1] - wavelengths[0]) / (n_components + 1)
            width = (wavelengths[-1] - wavelengths[0]) / 8
            profile = np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
            component_spectra.append(profile)
        component_spectra = np.array(component_spectra)

        # Beer-Lambert law: A = C @ epsilon
        X = concentrations @ component_spectra

        # Add baseline and noise
        baseline = rng.normal(0, 0.01, (n_samples, 1)) * np.linspace(0, 1, n_features)
        noise = rng.normal(0, 0.005, X.shape)
        X = X + baseline + noise

        # Scale to reasonable absorbance range
        X = X * 0.5 + 0.5

        # Generate targets
        if spec.task_type == "regression":
            y = concentrations[:, 0] * 50 + concentrations[:, 1] * 30 + rng.normal(0, 2, n_samples)
            y = np.clip(y, 5, 55)
        elif spec.task_type == "binary":
            threshold = np.median(concentrations[:, 0])
            y = (concentrations[:, 0] > threshold).astype(int)
        else:  # multiclass
            y = np.digitize(concentrations[:, 0], np.linspace(0, 1, spec.n_classes + 1)[1:-1])

        # Generate metadata
        metadata = {}
        metadata["sample_id"] = np.array([f"sample_{i:04d}" for i in range(n_samples)])
        metadata["group"] = np.array([f"group_{i % 5}" for i in range(n_samples)])

        if spec.aggregation:
            # Create repeated measurements
            n_unique = n_samples // 3
            metadata["sample_id"] = np.array([f"sample_{i % n_unique:04d}" for i in range(n_samples)])
            metadata["rep"] = np.array([f"rep_{i // n_unique}" for i in range(n_samples)])

        return X, y, metadata

    def _export_standard(
        self,
        spec: DatasetSpec,
        dataset_dir: Path,
        X: np.ndarray,
        y: np.ndarray,
        wavelengths: np.ndarray,
        metadata: Dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        """Export to standard Xcal/Ycal/Xval/Yval structure."""
        indices = rng.permutation(len(X))
        train_idx = indices[:spec.n_train]
        test_idx = indices[spec.n_train:]

        # Create column headers
        if spec.header_unit == "nm":
            headers = [str(int(wl)) for wl in wavelengths]
        elif spec.header_unit == "cm-1":
            # Convert nm to cm-1 (wavenumber)
            headers = [str(int(1e7 / wl)) for wl in wavelengths]
        elif spec.header_unit == "index":
            headers = [f"f_{i}" for i in range(len(wavelengths))]
        else:
            headers = [str(int(wl)) for wl in wavelengths]

        # Export files
        self._write_csv(
            dataset_dir / f"Xcal{spec.file_extension}",
            X[train_idx],
            headers if spec.has_header else None,
            spec,
        )
        self._write_csv(
            dataset_dir / f"Xval{spec.file_extension}",
            X[test_idx],
            headers if spec.has_header else None,
            spec,
        )

        y_headers = ["target"] if spec.has_header else None
        y_train = y[train_idx].reshape(-1, 1) if y.ndim == 1 else y[train_idx]
        y_test = y[test_idx].reshape(-1, 1) if y.ndim == 1 else y[test_idx]

        self._write_csv(dataset_dir / f"Ycal{spec.file_extension}", y_train, y_headers, spec)
        self._write_csv(dataset_dir / f"Yval{spec.file_extension}", y_test, y_headers, spec)

        # Export metadata if needed
        if spec.include_metadata:
            meta_headers = list(metadata.keys()) if spec.has_header else None
            meta_train = np.column_stack([metadata[k][train_idx] for k in metadata])
            meta_test = np.column_stack([metadata[k][test_idx] for k in metadata])

            self._write_csv(dataset_dir / f"Mcal{spec.file_extension}", meta_train, meta_headers, spec, is_text=True)
            self._write_csv(dataset_dir / f"Mval{spec.file_extension}", meta_test, meta_headers, spec, is_text=True)

    def _export_single(
        self,
        spec: DatasetSpec,
        dataset_dir: Path,
        X: np.ndarray,
        y: np.ndarray,
        wavelengths: np.ndarray,
        metadata: Dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        """Export all data to a single file with partition column."""
        indices = rng.permutation(len(X))
        train_idx = indices[:spec.n_train]
        test_idx = indices[spec.n_train:]

        # Create partition column
        partition = np.array(["train"] * len(X))
        partition[test_idx] = "test"

        # Build combined data
        headers = ["partition", "sample_id"]
        headers += [str(int(wl)) for wl in wavelengths]
        headers += ["target"]

        data = np.column_stack([
            partition,
            metadata["sample_id"],
            X,
            y.reshape(-1, 1) if y.ndim == 1 else y,
        ])

        self._write_csv(dataset_dir / f"data{spec.file_extension}", data, headers, spec, is_text=True)

    def _export_legacy(
        self,
        spec: DatasetSpec,
        dataset_dir: Path,
        X: np.ndarray,
        y: np.ndarray,
        wavelengths: np.ndarray,
        metadata: Dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        """Export to legacy separate train_x, train_y, test_x, test_y structure."""
        indices = rng.permutation(len(X))
        train_idx = indices[:spec.n_train]
        test_idx = indices[spec.n_train:]

        headers = [str(int(wl)) for wl in wavelengths] if spec.has_header else None

        self._write_csv(dataset_dir / f"train_x{spec.file_extension}", X[train_idx], headers, spec)
        self._write_csv(dataset_dir / f"test_x{spec.file_extension}", X[test_idx], headers, spec)

        y_headers = ["target"] if spec.has_header else None
        self._write_csv(dataset_dir / f"train_y{spec.file_extension}", y[train_idx].reshape(-1, 1), y_headers, spec)
        self._write_csv(dataset_dir / f"test_y{spec.file_extension}", y[test_idx].reshape(-1, 1), y_headers, spec)

        if spec.include_metadata:
            meta_headers = list(metadata.keys()) if spec.has_header else None
            meta_train = np.column_stack([metadata[k][train_idx] for k in metadata])
            meta_test = np.column_stack([metadata[k][test_idx] for k in metadata])

            self._write_csv(dataset_dir / f"train_group{spec.file_extension}", meta_train, meta_headers, spec, is_text=True)
            self._write_csv(dataset_dir / f"test_group{spec.file_extension}", meta_test, meta_headers, spec, is_text=True)

    def _export_fragmented(
        self,
        spec: DatasetSpec,
        dataset_dir: Path,
        X: np.ndarray,
        y: np.ndarray,
        wavelengths: np.ndarray,
        metadata: Dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        """Export to fragmented multiple files structure."""
        indices = rng.permutation(len(X))
        train_idx = indices[:spec.n_train]
        test_idx = indices[spec.n_train:]

        headers = [str(int(wl)) for wl in wavelengths] if spec.has_header else None

        # Split training into 3 parts
        n_per_part = len(train_idx) // 3
        for i in range(3):
            start = i * n_per_part
            end = start + n_per_part if i < 2 else len(train_idx)
            part_idx = train_idx[start:end]

            self._write_csv(dataset_dir / f"X_train_part{i}{spec.file_extension}", X[part_idx], headers, spec)

        # Test as single file
        self._write_csv(dataset_dir / f"X_test{spec.file_extension}", X[test_idx], headers, spec)

        # Y files
        y_headers = ["target"] if spec.has_header else None
        self._write_csv(dataset_dir / f"Y_train{spec.file_extension}", y[train_idx].reshape(-1, 1), y_headers, spec)
        self._write_csv(dataset_dir / f"Y_test{spec.file_extension}", y[test_idx].reshape(-1, 1), y_headers, spec)

    def _export_multi_source(
        self,
        spec: DatasetSpec,
        dataset_dir: Path,
        X: np.ndarray,
        y: np.ndarray,
        wavelengths: np.ndarray,
        metadata: Dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        """Export multi-source dataset with prefixed filenames."""
        indices = rng.permutation(len(X))
        train_idx = indices[:spec.n_train]
        test_idx = indices[spec.n_train:]

        source_configs = spec.sources or [
            {"name": "NIR", "range": (0, spec.n_features // 2)},
            {"name": "MIR", "range": (spec.n_features // 2, spec.n_features)},
        ]

        for src in source_configs:
            name = src["name"]
            feat_range = src.get("range", (0, spec.n_features))

            X_src = X[:, feat_range[0]:feat_range[1]]
            wl_src = wavelengths[feat_range[0]:feat_range[1]]

            headers = [str(int(wl)) for wl in wl_src] if spec.has_header else None

            self._write_csv(dataset_dir / f"{name}_train{spec.file_extension}", X_src[train_idx], headers, spec)
            self._write_csv(dataset_dir / f"{name}_test{spec.file_extension}", X_src[test_idx], headers, spec)

        # Shared targets
        y_headers = ["target"] if spec.has_header else None
        self._write_csv(dataset_dir / f"Y_train{spec.file_extension}", y[train_idx].reshape(-1, 1), y_headers, spec)
        self._write_csv(dataset_dir / f"Y_test{spec.file_extension}", y[test_idx].reshape(-1, 1), y_headers, spec)

        # Shared metadata with sample_id for linking
        meta_headers = list(metadata.keys()) if spec.has_header else None
        meta_train = np.column_stack([metadata[k][train_idx] for k in metadata])
        meta_test = np.column_stack([metadata[k][test_idx] for k in metadata])

        self._write_csv(dataset_dir / f"metadata_train{spec.file_extension}", meta_train, meta_headers, spec, is_text=True)
        self._write_csv(dataset_dir / f"metadata_test{spec.file_extension}", meta_test, meta_headers, spec, is_text=True)

    def _export_variations(
        self,
        spec: DatasetSpec,
        dataset_dir: Path,
        X: np.ndarray,
        y: np.ndarray,
        wavelengths: np.ndarray,
        metadata: Dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        """Export feature variations (raw + preprocessed)."""
        indices = rng.permutation(len(X))
        train_idx = indices[:spec.n_train]
        test_idx = indices[spec.n_train:]

        headers = [str(int(wl)) for wl in wavelengths] if spec.has_header else None

        # Raw variation
        self._write_csv(dataset_dir / f"X_raw_train{spec.file_extension}", X[train_idx], headers, spec)
        self._write_csv(dataset_dir / f"X_raw_test{spec.file_extension}", X[test_idx], headers, spec)

        # SNV-like preprocessed variation (mean-center, unit variance)
        X_snv = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
        self._write_csv(dataset_dir / f"X_snv_train{spec.file_extension}", X_snv[train_idx], headers, spec)
        self._write_csv(dataset_dir / f"X_snv_test{spec.file_extension}", X_snv[test_idx], headers, spec)

        # Targets
        y_headers = ["target"] if spec.has_header else None
        self._write_csv(dataset_dir / f"Y_train{spec.file_extension}", y[train_idx].reshape(-1, 1), y_headers, spec)
        self._write_csv(dataset_dir / f"Y_test{spec.file_extension}", y[test_idx].reshape(-1, 1), y_headers, spec)

    def _write_csv(
        self,
        path: Path,
        data: np.ndarray,
        headers: Optional[List[str]],
        spec: DatasetSpec,
        is_text: bool = False,
    ) -> None:
        """Write data to CSV file with specified format."""
        # Handle skip_rows by adding empty lines at start
        prefix_lines = ""
        if spec.skip_rows > 0:
            prefix_lines = "\n" * spec.skip_rows

        # Build CSV content
        lines = []

        if headers is not None:
            lines.append(spec.delimiter.join(headers))

        for row in data:
            if is_text:
                row_str = spec.delimiter.join(str(v) for v in row)
            else:
                row_str = spec.delimiter.join(
                    f"{float(v):.6f}".replace(".", spec.decimal_separator) for v in row
                )
            lines.append(row_str)

        content = prefix_lines + "\n".join(lines) + "\n"

        # Handle compression
        if spec.compression == "gzip" or path.suffix == ".gz":
            actual_path = path if path.suffix == ".gz" else Path(str(path) + ".gz")
            with gzip.open(actual_path, "wt", encoding=spec.encoding) as f:
                f.write(content)
        else:
            with open(path, "w", encoding=spec.encoding) as f:
                f.write(content)

    def _generate_config(self, spec: DatasetSpec, dataset_dir: Path) -> Dict[str, Any]:
        """Generate YAML configuration for the dataset."""
        # Use path relative to examples/ folder (where user typically runs from)
        rel_path = f"sample_datasets/{spec.name}"

        config: Dict[str, Any] = {
            "name": spec.name,
        }

        # Task type
        if spec.task_type == "binary":
            config["task_type"] = "binary_classification"
        elif spec.task_type == "multiclass":
            config["task_type"] = "multiclass_classification"
        else:
            config["task_type"] = "regression"

        # Signal type at top level (applies only to X data)
        config["signal_type"] = spec.signal_type

        # Global params - for loading settings
        global_params: Dict[str, Any] = {}

        if spec.delimiter != ";":
            global_params["delimiter"] = spec.delimiter
        else:
            global_params["delimiter"] = ";"

        if spec.decimal_separator != ".":
            global_params["decimal_separator"] = spec.decimal_separator

        global_params["has_header"] = spec.has_header
        global_params["header_unit"] = spec.header_unit

        if spec.encoding != "utf-8":
            global_params["encoding"] = spec.encoding

        # NA policy uses valid values: 'remove', 'abort', 'auto'
        global_params["na_policy"] = "remove"

        config["global_params"] = global_params

        # File paths based on structure
        ext = spec.file_extension
        if spec.compression == "gzip":
            ext = ext + ".gz" if not ext.endswith(".gz") else ext

        if spec.multi_source:
            self._add_multi_source_config(config, spec, rel_path, ext)
        elif spec.variations:
            self._add_variations_config(config, spec, rel_path, ext)
        elif spec.structure == "standard":
            config["train_x"] = f"{rel_path}/Xcal{ext}"
            config["train_y"] = f"{rel_path}/Ycal{ext}"
            config["test_x"] = f"{rel_path}/Xval{ext}"
            config["test_y"] = f"{rel_path}/Yval{ext}"
            if spec.include_metadata:
                config["train_group"] = f"{rel_path}/Mcal{ext}"
                config["test_group"] = f"{rel_path}/Mval{ext}"
        elif spec.structure == "single":
            config["train_x"] = f"{rel_path}/data{ext}"
            config["partition"] = {
                "column": "partition",
                "train_values": ["train"],
                "test_values": ["test"],
            }
        elif spec.structure == "legacy":
            config["train_x"] = f"{rel_path}/train_x{ext}"
            config["train_y"] = f"{rel_path}/train_y{ext}"
            config["test_x"] = f"{rel_path}/test_x{ext}"
            config["test_y"] = f"{rel_path}/test_y{ext}"
            if spec.include_metadata:
                config["train_group"] = f"{rel_path}/train_group{ext}"
                config["test_group"] = f"{rel_path}/test_group{ext}"

        # Partition configuration
        if spec.partition_method:
            self._add_partition_config(config, spec)

        # Repetition
        if spec.repetition and spec.repetition_column:
            config["repetition"] = spec.repetition_column
            config["repetition_method"] = spec.repetition_method

        # Folds
        if spec.custom_folds:
            config["folds"] = [
                {"train": list(range(0, 32)), "val": list(range(32, 48))},
                {"train": list(range(16, 48)), "val": list(range(0, 16))},
            ]

        # Merge extra config
        config.update(spec.extra_config)

        return config

    def _add_multi_source_config(
        self,
        config: Dict[str, Any],
        spec: DatasetSpec,
        rel_path: str,
        ext: str,
    ) -> None:
        """Add multi-source configuration."""
        source_configs = spec.sources or [
            {"name": "NIR", "range": (0, spec.n_features // 2)},
            {"name": "MIR", "range": (spec.n_features // 2, spec.n_features)},
        ]

        sources = []
        for src in source_configs:
            name = src["name"]
            source_entry = {
                "name": name,
                "train_x": f"{rel_path}/{name}_train{ext}",
                "test_x": f"{rel_path}/{name}_test{ext}",
            }
            if "params" in src:
                source_entry["params"] = src["params"]
            sources.append(source_entry)

        config["sources"] = sources
        config["shared_targets"] = {
            "path": f"{rel_path}/Y_train{ext}",
            "link_by": "sample_id",
        }

        # Add train_y/test_y for backward compatibility
        config["train_y"] = f"{rel_path}/Y_train{ext}"
        config["test_y"] = f"{rel_path}/Y_test{ext}"
        config["train_group"] = f"{rel_path}/metadata_train{ext}"
        config["test_group"] = f"{rel_path}/metadata_test{ext}"

    def _add_variations_config(
        self,
        config: Dict[str, Any],
        spec: DatasetSpec,
        rel_path: str,
        ext: str,
    ) -> None:
        """Add feature variations configuration."""
        config["variations"] = [
            {
                "name": "raw",
                "description": "Raw spectral data",
                "train_x": f"{rel_path}/X_raw_train{ext}",
                "test_x": f"{rel_path}/X_raw_test{ext}",
            },
            {
                "name": "snv",
                "description": "SNV preprocessed spectra",
                "train_x": f"{rel_path}/X_snv_train{ext}",
                "test_x": f"{rel_path}/X_snv_test{ext}",
                "preprocessing_applied": [
                    {"type": "SNV", "description": "Standard Normal Variate"}
                ],
            },
        ]
        config["variation_mode"] = spec.variation_mode or "separate"
        config["train_y"] = f"{rel_path}/Y_train{ext}"
        config["test_y"] = f"{rel_path}/Y_test{ext}"

    def _add_partition_config(self, config: Dict[str, Any], spec: DatasetSpec) -> None:
        """Add partition configuration."""
        if spec.partition_method == "column":
            config["partition"] = {
                "column": "partition",
                "train_values": ["train"],
                "test_values": ["test"],
            }
        elif spec.partition_method == "percentage":
            config["partition"] = {
                "train": "80%",
                "test": "20%",
                "shuffle": True,
                "random_state": 42,
            }
        elif spec.partition_method == "stratified":
            config["partition"] = {
                "train": "70%",
                "test": "30%",
                "stratify": "target",
                "shuffle": True,
                "random_state": 42,
            }
        elif spec.partition_method == "index":
            config["partition"] = {
                "train": list(range(0, spec.n_train)),
                "test": list(range(spec.n_train, spec.n_samples)),
            }

    def _write_yaml(self, path: Path, config: Dict[str, Any], description: str) -> None:
        """Write YAML configuration file."""
        header = f"# {description}\n#\n# Auto-generated test configuration\n\n"

        with open(path, "w") as f:
            f.write(header)
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # =========================================================================
    # Category Registration Methods
    # =========================================================================

    def _register_category_a(self) -> None:
        """Category A: File Format & Delimiter Variations (5 configs)."""

        # A01: Standard semicolon CSV
        self.specs.append(DatasetSpec(
            name="A01_csv_semicolon",
            category="A",
            description="Standard semicolon-delimited CSV",
            delimiter=";",
            structure="standard",
        ))

        # A02: Comma-delimited CSV
        self.specs.append(DatasetSpec(
            name="A02_csv_comma",
            category="A",
            description="Comma-delimited CSV",
            delimiter=",",
            structure="standard",
        ))

        # A03: Tab-separated values
        self.specs.append(DatasetSpec(
            name="A03_csv_tab",
            category="A",
            description="Tab-separated values",
            delimiter="\t",
            file_extension=".tsv",
            structure="standard",
        ))

        # A04: Pipe delimiter
        self.specs.append(DatasetSpec(
            name="A04_csv_pipe",
            category="A",
            description="Pipe-delimited CSV",
            delimiter="|",
            structure="standard",
        ))

        # A05: European format (semicolon delimiter, comma decimal)
        self.specs.append(DatasetSpec(
            name="A05_csv_european",
            category="A",
            description="European CSV format (semicolon delimiter, comma decimal)",
            delimiter=";",
            decimal_separator=",",
            structure="standard",
        ))

    def _register_category_b(self) -> None:
        """Category B: Loading Parameters (5 configs)."""

        # B01: No header
        self.specs.append(DatasetSpec(
            name="B01_no_header",
            category="B",
            description="CSV without column headers",
            has_header=False,
            header_unit="index",
            structure="standard",
        ))

        # B02: Wavenumber headers (cm-1)
        self.specs.append(DatasetSpec(
            name="B02_wavenumber",
            category="B",
            description="Wavenumber headers (cm-1)",
            header_unit="cm-1",
            structure="standard",
        ))

        # B03: Wavelength headers with absorbance
        self.specs.append(DatasetSpec(
            name="B03_wavelength",
            category="B",
            description="Wavelength headers (nm) with absorbance",
            header_unit="nm",
            signal_type="absorbance",
            structure="standard",
        ))

        # B04: Reflectance signal type
        self.specs.append(DatasetSpec(
            name="B04_reflectance",
            category="B",
            description="Reflectance percentage signal type",
            signal_type="reflectance%",
            structure="standard",
        ))

        # B05: Latin-1 encoding with skip rows
        self.specs.append(DatasetSpec(
            name="B05_encoding_skiprows",
            category="B",
            description="Latin-1 encoding with 2 skip rows",
            encoding="latin-1",
            skip_rows=2,
            structure="standard",
        ))

    def _register_category_c(self) -> None:
        """Category C: File Structures (5 configs)."""

        # C01: Legacy separate files
        self.specs.append(DatasetSpec(
            name="C01_legacy_separate",
            category="C",
            description="Legacy format with separate train_x, train_y files",
            structure="legacy",
        ))

        # C02: Single combined file with partition column
        self.specs.append(DatasetSpec(
            name="C02_combined_single",
            category="C",
            description="Single file with partition column",
            structure="single",
            delimiter=",",
        ))

        # C03: Standard Xcal/Ycal/Xval/Yval folder
        self.specs.append(DatasetSpec(
            name="C03_standard_folder",
            category="C",
            description="Standard folder structure (Xcal, Ycal, Xval, Yval)",
            structure="standard",
        ))

        # C04: With metadata files
        self.specs.append(DatasetSpec(
            name="C04_with_metadata",
            category="C",
            description="Standard structure with metadata files",
            structure="standard",
            include_metadata=True,
        ))

        # C05: Compressed gzip files
        self.specs.append(DatasetSpec(
            name="C05_compressed",
            category="C",
            description="Gzip compressed CSV files",
            compression="gzip",
            file_extension=".csv.gz",
            structure="standard",
        ))

    def _register_category_d(self) -> None:
        """Category D: Partition & Split Strategies (5 configs)."""

        # D01: Column-based partition
        self.specs.append(DatasetSpec(
            name="D01_column_partition",
            category="D",
            description="Column-based partition (split column)",
            structure="single",
            partition_method="column",
            delimiter=",",
        ))

        # D02: Percentage-based partition
        self.specs.append(DatasetSpec(
            name="D02_percentage_partition",
            category="D",
            description="Percentage-based partition (80/20 split)",
            structure="single",
            partition_method="percentage",
            delimiter=",",
        ))

        # D03: Index-based partition
        self.specs.append(DatasetSpec(
            name="D03_index_partition",
            category="D",
            description="Index-based partition (explicit indices)",
            structure="single",
            partition_method="index",
            delimiter=",",
        ))

        # D04: Stratified partition (classification)
        self.specs.append(DatasetSpec(
            name="D04_stratified_partition",
            category="D",
            description="Stratified partition for classification",
            structure="single",
            partition_method="stratified",
            task_type="binary",
            delimiter=",",
        ))

        # D05: Custom cross-validation folds
        self.specs.append(DatasetSpec(
            name="D05_custom_folds",
            category="D",
            description="Custom cross-validation fold definitions",
            structure="standard",
            custom_folds=True,
        ))

    def _register_category_e(self) -> None:
        """Category E: Multi-Source & Aggregation (5 configs)."""

        # E01: Dual NIR sources
        self.specs.append(DatasetSpec(
            name="E01_dual_source",
            category="E",
            description="Dual NIR sources (NIR + MIR)",
            multi_source=True,
            sources=[
                {"name": "NIR", "range": (0, 100)},
                {"name": "MIR", "range": (100, 200)},
            ],
            include_metadata=True,
        ))

        # E02: NIR + auxiliary markers
        self.specs.append(DatasetSpec(
            name="E02_nir_markers",
            category="E",
            description="NIR spectra with auxiliary markers",
            multi_source=True,
            sources=[
                {"name": "NIR", "range": (0, 180)},
                {"name": "markers", "range": (180, 200)},
            ],
            include_metadata=True,
        ))

        # E03: Shared targets configuration
        self.specs.append(DatasetSpec(
            name="E03_shared_targets",
            category="E",
            description="Multi-source with shared targets file",
            multi_source=True,
            sources=[
                {"name": "source1", "range": (0, 100)},
                {"name": "source2", "range": (100, 200)},
            ],
            include_metadata=True,
        ))

        # E04: Mean repetition by sample_id
        self.specs.append(DatasetSpec(
            name="E04_aggregate_mean",
            category="E",
            description="Mean aggregation by sample_id",
            structure="standard",
            repetition=True,
            repetition_column="sample_id",
            repetition_method="mean",
            include_metadata=True,
        ))

        # E05: Repetition with outlier exclusion
        self.specs.append(DatasetSpec(
            name="E05_aggregate_outliers",
            category="E",
            description="Aggregation with outlier exclusion",
            structure="standard",
            repetition=True,
            repetition_column="sample_id",
            repetition_method="mean",
            include_metadata=True,
            extra_config={"repetition_exclude_outliers": True},
        ))

    def _register_category_f(self) -> None:
        """Category F: Task Types & Variations (5 configs)."""

        # F01: Regression task
        self.specs.append(DatasetSpec(
            name="F01_regression",
            category="F",
            description="Regression task type",
            task_type="regression",
            structure="standard",
        ))

        # F02: Binary classification
        self.specs.append(DatasetSpec(
            name="F02_binary_class",
            category="F",
            description="Binary classification task",
            task_type="binary",
            structure="standard",
        ))

        # F03: Multiclass classification
        self.specs.append(DatasetSpec(
            name="F03_multiclass",
            category="F",
            description="Multiclass classification (5 classes)",
            task_type="multiclass",
            n_classes=5,
            structure="standard",
        ))

        # F04: Feature variations (separate mode)
        self.specs.append(DatasetSpec(
            name="F04_variations_separate",
            category="F",
            description="Feature variations with separate mode",
            variations=True,
            variation_mode="separate",
        ))

        # F05: Feature variations (concat mode)
        self.specs.append(DatasetSpec(
            name="F05_variations_concat",
            category="F",
            description="Feature variations with concat mode",
            variations=True,
            variation_mode="concat",
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate test datasets and configs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Output directory (default: examples/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output messages",
    )

    args = parser.parse_args()

    generator = TestDatasetGenerator(
        output_dir=args.output_dir,
        seed=args.seed,
        verbose=not args.quiet,
    )

    generator.generate_all()


if __name__ == "__main__":
    main()
