"""
Dataset export utilities for synthetic NIRS data.

This module provides tools for exporting synthetic datasets to various
file formats and folder structures compatible with nirs4all loaders.

Key Features:
    - Export to CSV files (single or multi-file format)
    - Export to nirs4all standard folder structure (Xcal, Ycal, Xval, Yval)
    - Export with metadata (sample IDs, groups, etc.)
    - Generate CSV variations for loader testing

Example:
    >>> from nirs4all.data.synthetic import SyntheticDatasetBuilder, DatasetExporter
    >>>
    >>> builder = SyntheticDatasetBuilder(n_samples=1000, random_state=42)
    >>> X, y = builder.build_arrays()
    >>>
    >>> exporter = DatasetExporter()
    >>> path = exporter.to_folder(
    ...     "output/synthetic_data",
    ...     X, y,
    ...     train_ratio=0.8,
    ...     wavelengths=builder.state._wavelengths
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class ExportConfig:
    """
    Configuration for dataset export.

    Attributes:
        format: Export format ('standard', 'single', 'fragmented').
            - 'standard': Separate Xcal, Ycal, Xval, Yval files.
            - 'single': All data in one file with partition column.
            - 'fragmented': Multiple small files (for loader testing).
        separator: CSV delimiter character.
        float_precision: Decimal precision for floating point values.
        include_headers: Whether to include column headers in CSV.
        include_index: Whether to include row index.
        compression: Optional compression ('gzip', 'zip', None).
        file_extension: File extension to use.
    """

    format: Literal["standard", "single", "fragmented"] = "standard"
    separator: str = ";"
    float_precision: int = 6
    include_headers: bool = True
    include_index: bool = False
    compression: Optional[Literal["gzip", "zip"]] = None
    file_extension: str = ".csv"


class DatasetExporter:
    """
    Export synthetic datasets to various file formats.

    This class provides methods for exporting synthetic NIRS datasets
    to files and folders compatible with nirs4all's data loaders.

    Attributes:
        config: Export configuration settings.

    Args:
        config: Optional ExportConfig. Uses defaults if None.

    Example:
        >>> exporter = DatasetExporter()
        >>>
        >>> # Export to standard folder structure
        >>> path = exporter.to_folder(
        ...     "output/data",
        ...     X, y,
        ...     train_ratio=0.8,
        ...     wavelengths=wavelengths
        ... )
        >>>
        >>> # Export to single CSV
        >>> path = exporter.to_csv(
        ...     "output/all_data.csv",
        ...     X, y,
        ...     wavelengths=wavelengths
        ... )
    """

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        """
        Initialize the exporter.

        Args:
            config: Export configuration. Uses defaults if None.
        """
        self.config = config or ExportConfig()

    def to_folder(
        self,
        path: Union[str, Path],
        X: np.ndarray,
        y: np.ndarray,
        *,
        train_ratio: float = 0.8,
        wavelengths: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, np.ndarray]] = None,
        random_state: Optional[int] = None,
        format: Optional[Literal["standard", "single", "fragmented"]] = None,
    ) -> Path:
        """
        Export dataset to a folder structure.

        Creates a folder with CSV files compatible with nirs4all's
        DatasetConfigs loader.

        Args:
            path: Output folder path.
            X: Feature matrix (n_samples, n_features).
            y: Target values (n_samples,) or (n_samples, n_targets).
            train_ratio: Proportion for training set.
            wavelengths: Optional wavelength values for column headers.
            metadata: Optional dict of metadata arrays (same length as X).
            random_state: Random seed for train/test split.
            format: Override config format for this export.

        Returns:
            Path to created folder.

        Raises:
            ValueError: If X and y have incompatible shapes.
            ImportError: If pandas is not available.

        Example:
            >>> exporter.to_folder(
            ...     "data/synthetic",
            ...     X, y,
            ...     train_ratio=0.8,
            ...     wavelengths=np.arange(1000, 2500, 2)
            ... )
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for CSV export")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Validate inputs
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}"
            )

        export_format = format or self.config.format

        if export_format == "standard":
            return self._export_standard(path, X, y, train_ratio, wavelengths, metadata, random_state)
        elif export_format == "single":
            return self._export_single(path, X, y, train_ratio, wavelengths, metadata, random_state)
        elif export_format == "fragmented":
            return self._export_fragmented(path, X, y, train_ratio, wavelengths, metadata, random_state)
        else:
            raise ValueError(f"Unknown format: {export_format}")

    def _export_standard(
        self,
        path: Path,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float,
        wavelengths: Optional[np.ndarray],
        metadata: Optional[Dict[str, np.ndarray]],
        random_state: Optional[int],
    ) -> Path:
        """Export to standard Xcal/Ycal/Xval/Yval structure."""
        n_samples = X.shape[0]
        n_train = int(n_samples * train_ratio)

        # Create train/test split
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        # Create feature column names
        if wavelengths is not None:
            columns = [str(int(wl)) for wl in wavelengths]
        else:
            columns = [f"feature_{i}" for i in range(X.shape[1])]

        # Create target column names
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y_columns = [f"target_{i}" if y.shape[1] > 1 else "target" for i in range(y.shape[1])]
        if y.shape[1] == 1:
            y_columns = ["target"]

        # Export training data
        X_train = pd.DataFrame(X[train_idx], columns=columns)
        X_train.to_csv(
            path / f"Xcal{self.config.file_extension}",
            sep=self.config.separator,
            index=self.config.include_index,
            float_format=f"%.{self.config.float_precision}f",
        )

        y_train = pd.DataFrame(y[train_idx], columns=y_columns)
        y_train.to_csv(
            path / f"Ycal{self.config.file_extension}",
            sep=self.config.separator,
            index=self.config.include_index,
            float_format=f"%.{self.config.float_precision}f",
        )

        # Export test data
        if len(test_idx) > 0:
            X_test = pd.DataFrame(X[test_idx], columns=columns)
            X_test.to_csv(
                path / f"Xval{self.config.file_extension}",
                sep=self.config.separator,
                index=self.config.include_index,
                float_format=f"%.{self.config.float_precision}f",
            )

            y_test = pd.DataFrame(y[test_idx], columns=y_columns)
            y_test.to_csv(
                path / f"Yval{self.config.file_extension}",
                sep=self.config.separator,
                index=self.config.include_index,
                float_format=f"%.{self.config.float_precision}f",
            )

        # Export metadata if provided
        if metadata:
            self._export_metadata(path, metadata, train_idx, test_idx)

        return path

    def _export_single(
        self,
        path: Path,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float,
        wavelengths: Optional[np.ndarray],
        metadata: Optional[Dict[str, np.ndarray]],
        random_state: Optional[int],
    ) -> Path:
        """Export all data to a single CSV file with partition column."""
        n_samples = X.shape[0]
        n_train = int(n_samples * train_ratio)

        # Create train/test split
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        # Create feature column names
        if wavelengths is not None:
            feature_columns = [str(int(wl)) for wl in wavelengths]
        else:
            feature_columns = [f"feature_{i}" for i in range(X.shape[1])]

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y_columns = [f"target_{i}" if y.shape[1] > 1 else "target" for i in range(y.shape[1])]
        if y.shape[1] == 1:
            y_columns = ["target"]

        # Build combined DataFrame
        data = {}

        # Add partition column
        partition = np.array(["train"] * n_samples)
        partition[test_idx] = "test"
        # Reorder by indices
        reorder = np.zeros(n_samples, dtype=int)
        reorder[indices] = np.arange(n_samples)
        data["partition"] = partition

        # Add sample IDs if in metadata
        if metadata and "sample_id" in metadata:
            data["sample_id"] = metadata["sample_id"]

        # Add features
        for i, col in enumerate(feature_columns):
            data[col] = X[:, i]

        # Add targets
        for i, col in enumerate(y_columns):
            data[col] = y[:, i]

        # Add remaining metadata
        if metadata:
            for key, values in metadata.items():
                if key != "sample_id":  # Already added
                    data[key] = values

        df = pd.DataFrame(data)
        df.to_csv(
            path / f"data{self.config.file_extension}",
            sep=self.config.separator,
            index=self.config.include_index,
            float_format=f"%.{self.config.float_precision}f",
        )

        return path

    def _export_fragmented(
        self,
        path: Path,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float,
        wavelengths: Optional[np.ndarray],
        metadata: Optional[Dict[str, np.ndarray]],
        random_state: Optional[int],
    ) -> Path:
        """Export to multiple small files (for loader testing)."""
        n_samples = X.shape[0]
        n_train = int(n_samples * train_ratio)

        # Create train/test split
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        # Create feature column names
        if wavelengths is not None:
            columns = [str(int(wl)) for wl in wavelengths]
        else:
            columns = [f"feature_{i}" for i in range(X.shape[1])]

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y_columns = [f"target_{i}" if y.shape[1] > 1 else "target" for i in range(y.shape[1])]
        if y.shape[1] == 1:
            y_columns = ["target"]

        # Create train folder with fragmented files
        train_path = path / "train"
        train_path.mkdir(parents=True, exist_ok=True)

        # Split training data into chunks
        chunk_size = max(10, len(train_idx) // 5)  # At least 5 chunks
        for i, start in enumerate(range(0, len(train_idx), chunk_size)):
            end = min(start + chunk_size, len(train_idx))
            chunk_idx = train_idx[start:end]

            # Export features
            df_x = pd.DataFrame(X[chunk_idx], columns=columns)
            df_x.to_csv(
                train_path / f"X_part{i}{self.config.file_extension}",
                sep=self.config.separator,
                index=self.config.include_index,
                float_format=f"%.{self.config.float_precision}f",
            )

            # Export targets
            df_y = pd.DataFrame(y[chunk_idx], columns=y_columns)
            df_y.to_csv(
                train_path / f"Y_part{i}{self.config.file_extension}",
                sep=self.config.separator,
                index=self.config.include_index,
                float_format=f"%.{self.config.float_precision}f",
            )

        # Create test folder
        if len(test_idx) > 0:
            test_path = path / "test"
            test_path.mkdir(parents=True, exist_ok=True)

            df_x = pd.DataFrame(X[test_idx], columns=columns)
            df_x.to_csv(
                test_path / f"X{self.config.file_extension}",
                sep=self.config.separator,
                index=self.config.include_index,
                float_format=f"%.{self.config.float_precision}f",
            )

            df_y = pd.DataFrame(y[test_idx], columns=y_columns)
            df_y.to_csv(
                test_path / f"Y{self.config.file_extension}",
                sep=self.config.separator,
                index=self.config.include_index,
                float_format=f"%.{self.config.float_precision}f",
            )

        return path

    def _export_metadata(
        self,
        path: Path,
        metadata: Dict[str, np.ndarray],
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> None:
        """Export metadata to separate CSV files."""
        meta_df = pd.DataFrame(metadata)

        # Training metadata
        meta_train = meta_df.iloc[train_idx]
        meta_train.to_csv(
            path / f"metadata_cal{self.config.file_extension}",
            sep=self.config.separator,
            index=self.config.include_index,
        )

        # Test metadata
        if len(test_idx) > 0:
            meta_test = meta_df.iloc[test_idx]
            meta_test.to_csv(
                path / f"metadata_val{self.config.file_extension}",
                sep=self.config.separator,
                index=self.config.include_index,
            )

    def to_csv(
        self,
        path: Union[str, Path],
        X: np.ndarray,
        y: np.ndarray,
        *,
        wavelengths: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, np.ndarray]] = None,
        include_targets: bool = True,
    ) -> Path:
        """
        Export dataset to a single CSV file.

        Creates a CSV file with features (and optionally targets) combined.

        Args:
            path: Output file path.
            X: Feature matrix (n_samples, n_features).
            y: Target values (n_samples,) or (n_samples, n_targets).
            wavelengths: Optional wavelength values for column headers.
            metadata: Optional dict of metadata arrays.
            include_targets: Whether to include target column(s).

        Returns:
            Path to created file.

        Example:
            >>> exporter.to_csv("data.csv", X, y, wavelengths=wavelengths)
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for CSV export")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Validate inputs
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}"
            )

        # Create feature column names
        if wavelengths is not None:
            feature_columns = [str(int(wl)) for wl in wavelengths]
        else:
            feature_columns = [f"feature_{i}" for i in range(X.shape[1])]

        # Build DataFrame
        data = {}

        # Add metadata first (sample IDs, etc.)
        if metadata:
            for key, values in metadata.items():
                data[key] = values

        # Add features
        for i, col in enumerate(feature_columns):
            data[col] = X[:, i]

        # Add targets
        if include_targets:
            if y.ndim == 1:
                data["target"] = y
            else:
                for i in range(y.shape[1]):
                    data[f"target_{i}" if y.shape[1] > 1 else "target"] = y[:, i]

        df = pd.DataFrame(data)
        df.to_csv(
            path,
            sep=self.config.separator,
            index=self.config.include_index,
            float_format=f"%.{self.config.float_precision}f",
        )

        return path

    def to_numpy(
        self,
        path: Union[str, Path],
        X: np.ndarray,
        y: np.ndarray,
        *,
        wavelengths: Optional[np.ndarray] = None,
        compressed: bool = False,
    ) -> Path:
        """
        Export dataset to numpy .npy or .npz format.

        Args:
            path: Output file path (without extension).
            X: Feature matrix (n_samples, n_features).
            y: Target values.
            wavelengths: Optional wavelength values.
            compressed: Whether to use compressed format (.npz).

        Returns:
            Path to created file.

        Example:
            >>> exporter.to_numpy("data", X, y, compressed=True)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {"X": X, "y": y}
        if wavelengths is not None:
            arrays["wavelengths"] = wavelengths

        if compressed:
            save_path = path.with_suffix(".npz")
            np.savez_compressed(save_path, **arrays)
        else:
            save_path = path.with_suffix(".npz")
            np.savez(save_path, **arrays)

        return save_path


class CSVVariationGenerator:
    """
    Generate CSV files with various format variations for loader testing.

    This class creates CSV files with different delimiters, encodings,
    header formats, and other variations to test the robustness of
    CSV loaders.

    Attributes:
        base_exporter: DatasetExporter for actual file writing.

    Example:
        >>> generator = CSVVariationGenerator()
        >>>
        >>> # Generate all variations
        >>> paths = generator.generate_all_variations(
        ...     "test_data",
        ...     X, y,
        ...     wavelengths=wavelengths
        ... )
        >>>
        >>> # Generate specific variation
        >>> path = generator.with_semicolon_delimiter(
        ...     "data_semicolon",
        ...     X, y
        ... )
    """

    def __init__(self) -> None:
        """Initialize the variation generator."""
        self.base_exporter = DatasetExporter()

    def generate_all_variations(
        self,
        base_path: Union[str, Path],
        X: np.ndarray,
        y: np.ndarray,
        *,
        wavelengths: Optional[np.ndarray] = None,
        train_ratio: float = 0.8,
        random_state: Optional[int] = None,
    ) -> Dict[str, Path]:
        """
        Generate CSV files with all format variations.

        Creates multiple versions of the dataset with different CSV
        format options for comprehensive loader testing.

        Args:
            base_path: Base output folder path.
            X: Feature matrix.
            y: Target values.
            wavelengths: Optional wavelength values.
            train_ratio: Train/test split ratio.
            random_state: Random seed.

        Returns:
            Dictionary mapping variation name to created path.

        Example:
            >>> paths = generator.generate_all_variations(
            ...     "test_variations",
            ...     X, y,
            ...     random_state=42
            ... )
            >>> print(paths.keys())
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Standard format (semicolon separator)
        paths["standard_semicolon"] = self.with_semicolon_delimiter(
            base_path / "standard_semicolon",
            X, y,
            wavelengths=wavelengths,
            train_ratio=train_ratio,
            random_state=random_state,
        )

        # Comma separator
        paths["comma_separated"] = self.with_comma_delimiter(
            base_path / "comma_separated",
            X, y,
            wavelengths=wavelengths,
            train_ratio=train_ratio,
            random_state=random_state,
        )

        # Tab separated
        paths["tab_separated"] = self.with_tab_delimiter(
            base_path / "tab_separated",
            X, y,
            wavelengths=wavelengths,
            train_ratio=train_ratio,
            random_state=random_state,
        )

        # No headers
        paths["no_headers"] = self.without_headers(
            base_path / "no_headers",
            X, y,
            train_ratio=train_ratio,
            random_state=random_state,
        )

        # With index
        paths["with_index"] = self.with_row_index(
            base_path / "with_index",
            X, y,
            wavelengths=wavelengths,
            train_ratio=train_ratio,
            random_state=random_state,
        )

        # Single file format
        paths["single_file"] = self.as_single_file(
            base_path / "single_file",
            X, y,
            wavelengths=wavelengths,
            train_ratio=train_ratio,
            random_state=random_state,
        )

        # Fragmented files
        paths["fragmented"] = self.as_fragmented(
            base_path / "fragmented",
            X, y,
            wavelengths=wavelengths,
            train_ratio=train_ratio,
            random_state=random_state,
        )

        # Low precision
        paths["low_precision"] = self.with_precision(
            base_path / "low_precision",
            X, y,
            wavelengths=wavelengths,
            train_ratio=train_ratio,
            random_state=random_state,
            precision=2,
        )

        # High precision
        paths["high_precision"] = self.with_precision(
            base_path / "high_precision",
            X, y,
            wavelengths=wavelengths,
            train_ratio=train_ratio,
            random_state=random_state,
            precision=10,
        )

        return paths

    def with_semicolon_delimiter(
        self,
        path: Union[str, Path],
        X: np.ndarray,
        y: np.ndarray,
        *,
        wavelengths: Optional[np.ndarray] = None,
        train_ratio: float = 0.8,
        random_state: Optional[int] = None,
    ) -> Path:
        """Create CSV with semicolon delimiter (nirs4all default)."""
        config = ExportConfig(separator=";")
        exporter = DatasetExporter(config)
        return exporter.to_folder(
            path, X, y,
            wavelengths=wavelengths,
            train_ratio=train_ratio,
            random_state=random_state,
        )

    def with_comma_delimiter(
        self,
        path: Union[str, Path],
        X: np.ndarray,
        y: np.ndarray,
        *,
        wavelengths: Optional[np.ndarray] = None,
        train_ratio: float = 0.8,
        random_state: Optional[int] = None,
    ) -> Path:
        """Create CSV with comma delimiter."""
        config = ExportConfig(separator=",")
        exporter = DatasetExporter(config)
        return exporter.to_folder(
            path, X, y,
            wavelengths=wavelengths,
            train_ratio=train_ratio,
            random_state=random_state,
        )

    def with_tab_delimiter(
        self,
        path: Union[str, Path],
        X: np.ndarray,
        y: np.ndarray,
        *,
        wavelengths: Optional[np.ndarray] = None,
        train_ratio: float = 0.8,
        random_state: Optional[int] = None,
    ) -> Path:
        """Create CSV with tab delimiter."""
        config = ExportConfig(separator="\t", file_extension=".tsv")
        exporter = DatasetExporter(config)
        return exporter.to_folder(
            path, X, y,
            wavelengths=wavelengths,
            train_ratio=train_ratio,
            random_state=random_state,
        )

    def without_headers(
        self,
        path: Union[str, Path],
        X: np.ndarray,
        y: np.ndarray,
        *,
        train_ratio: float = 0.8,
        random_state: Optional[int] = None,
    ) -> Path:
        """Create CSV without column headers."""
        config = ExportConfig(include_headers=False)
        exporter = DatasetExporter(config)

        # Need to manually write since pandas always writes headers by default
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        n_samples = X.shape[0]
        n_train = int(n_samples * train_ratio)

        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        # Ensure y is 2D
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y

        np.savetxt(
            path / "Xcal.csv",
            X[train_idx],
            delimiter=config.separator,
            fmt=f"%.{config.float_precision}f",
        )
        np.savetxt(
            path / "Ycal.csv",
            y_2d[train_idx],
            delimiter=config.separator,
            fmt=f"%.{config.float_precision}f",
        )

        if len(test_idx) > 0:
            np.savetxt(
                path / "Xval.csv",
                X[test_idx],
                delimiter=config.separator,
                fmt=f"%.{config.float_precision}f",
            )
            np.savetxt(
                path / "Yval.csv",
                y_2d[test_idx],
                delimiter=config.separator,
                fmt=f"%.{config.float_precision}f",
            )

        return path

    def with_row_index(
        self,
        path: Union[str, Path],
        X: np.ndarray,
        y: np.ndarray,
        *,
        wavelengths: Optional[np.ndarray] = None,
        train_ratio: float = 0.8,
        random_state: Optional[int] = None,
    ) -> Path:
        """Create CSV with row index column."""
        config = ExportConfig(include_index=True)
        exporter = DatasetExporter(config)
        return exporter.to_folder(
            path, X, y,
            wavelengths=wavelengths,
            train_ratio=train_ratio,
            random_state=random_state,
        )

    def as_single_file(
        self,
        path: Union[str, Path],
        X: np.ndarray,
        y: np.ndarray,
        *,
        wavelengths: Optional[np.ndarray] = None,
        train_ratio: float = 0.8,
        random_state: Optional[int] = None,
    ) -> Path:
        """Create single CSV file with all data and partition column."""
        config = ExportConfig(format="single")
        exporter = DatasetExporter(config)
        return exporter.to_folder(
            path, X, y,
            wavelengths=wavelengths,
            train_ratio=train_ratio,
            random_state=random_state,
            format="single",
        )

    def as_fragmented(
        self,
        path: Union[str, Path],
        X: np.ndarray,
        y: np.ndarray,
        *,
        wavelengths: Optional[np.ndarray] = None,
        train_ratio: float = 0.8,
        random_state: Optional[int] = None,
    ) -> Path:
        """Create fragmented dataset with multiple small files."""
        exporter = DatasetExporter()
        return exporter.to_folder(
            path, X, y,
            wavelengths=wavelengths,
            train_ratio=train_ratio,
            random_state=random_state,
            format="fragmented",
        )

    def with_precision(
        self,
        path: Union[str, Path],
        X: np.ndarray,
        y: np.ndarray,
        *,
        wavelengths: Optional[np.ndarray] = None,
        train_ratio: float = 0.8,
        random_state: Optional[int] = None,
        precision: int = 6,
    ) -> Path:
        """Create CSV with specified floating point precision."""
        config = ExportConfig(float_precision=precision)
        exporter = DatasetExporter(config)
        return exporter.to_folder(
            path, X, y,
            wavelengths=wavelengths,
            train_ratio=train_ratio,
            random_state=random_state,
        )


def export_to_folder(
    path: Union[str, Path],
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_ratio: float = 0.8,
    wavelengths: Optional[np.ndarray] = None,
    format: Literal["standard", "single", "fragmented"] = "standard",
    random_state: Optional[int] = None,
) -> Path:
    """
    Quick function to export synthetic data to folder.

    Convenience function for simple export use cases.

    Args:
        path: Output folder path.
        X: Feature matrix.
        y: Target values.
        train_ratio: Train/test split ratio.
        wavelengths: Optional wavelength values.
        format: Export format.
        random_state: Random seed.

    Returns:
        Path to created folder.

    Example:
        >>> path = export_to_folder(
        ...     "data/synthetic",
        ...     X, y,
        ...     train_ratio=0.8,
        ...     wavelengths=wavelengths
        ... )
    """
    exporter = DatasetExporter()
    return exporter.to_folder(
        path, X, y,
        train_ratio=train_ratio,
        wavelengths=wavelengths,
        format=format,
        random_state=random_state,
    )


def export_to_csv(
    path: Union[str, Path],
    X: np.ndarray,
    y: np.ndarray,
    *,
    wavelengths: Optional[np.ndarray] = None,
) -> Path:
    """
    Quick function to export synthetic data to single CSV.

    Args:
        path: Output file path.
        X: Feature matrix.
        y: Target values.
        wavelengths: Optional wavelength values.

    Returns:
        Path to created file.

    Example:
        >>> path = export_to_csv("data.csv", X, y)
    """
    exporter = DatasetExporter()
    return exporter.to_csv(path, X, y, wavelengths=wavelengths)
