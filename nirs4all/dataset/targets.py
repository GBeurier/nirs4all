"""
Target management for SpectroDataset.

This module contains a simplified Target class that manages target values
with their transformations, similar to Features but focused on targets.
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np
import polars as pl
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import warnings


class Targets:
    """
    Target management class that stores targets and their transformations.

    Unlike Features, Target keeps all processing versions to enable inverse transforms
    for predictions and evaluations.
    """

    def __init__(self):
        """Initialize empty Target."""
        # Index DataFrame tracking rows, samples, and processing versions
        self.index = pl.DataFrame({
            "row": pl.Series([], dtype=pl.Int32),
            "sample": pl.Series([], dtype=pl.Int32),
            "processing": pl.Series([], dtype=pl.String),
            "partition": pl.Series([], dtype=pl.String),
        })

        # Store target data for each processing version
        # Key format: "processing_name" -> numpy array
        self._data: Dict[str, np.ndarray] = {}

        # Store transformers and their source processing
        # Key format: "processing_name" -> (TransformerMixin, source_processing)
        self.transformers_dict: Dict[str, tuple[TransformerMixin, str]] = {}

        # Cache for filtered results
        self._filtered_cache: Dict[str, pl.DataFrame] = {}

    @property
    def n_samples(self) -> int:
        """Get number of samples."""
        if len(self.index) == 0:
            return 0
        return self.index["sample"].n_unique()

    @property
    def processing_versions(self) -> List[str]:
        """Get available processing versions."""
        if len(self.index) == 0:
            return []
        return sorted(self.index["processing"].unique().to_list())

    def add_targets(self, y: np.ndarray, overrides: Optional[Dict[str, Any]] = None) -> None:
        """
        Add raw targets to the dataset.

        Args:
            y: Target array of shape (n_samples, n_targets) or (n_samples,)
            overrides: Optional overrides for index columns
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = y.shape[0]
        overrides = overrides or {}

        # Build index for raw processing
        indices = self._add_to_index(n_samples, "raw", overrides)

        # Store raw data
        self._data["raw"] = y.copy()

        # Automatically create numeric version
        self._create_numeric_version(y, indices)

    def _add_to_index(self, n_samples: int, processing: str, overrides: Dict[str, Any]) -> None:
        """Add rows to index for a processing version."""
        # Determine next indices
        next_row = self.index["row"].max() + 1 if len(self.index) > 0 else 0
        next_sample = overrides.get("sample", list(range(self.n_samples, self.n_samples + n_samples)))

        if isinstance(next_sample, int):
            next_sample = [next_sample]
        elif len(next_sample) != n_samples:
            raise ValueError(f"Sample override must have {n_samples} elements")

        next_sample = np.array(next_sample, dtype=np.int32)
        next_rows = np.arange(next_row, next_row + n_samples, dtype=np.int32)
        # Create new rows
        new_rows = pl.DataFrame({
            "row": next_rows,
            "sample": next_sample,
            "processing": [processing] * n_samples,
            "partition": overrides.get("partition", "train") if "partition" in overrides else "train",
        })

        # Add to index
        self.index = pl.concat([self.index, new_rows], how="vertical")
        return next_rows

    def _create_numeric_version(self, y_raw: np.ndarray, indices: np.ndarray) -> None:
        """
        Automatically create a numeric version of targets.

        Converts string columns to categories and ensures numeric data for models.
        """
        y_numeric = y_raw.copy()
        transformers = []

        # Handle each column
        for col_idx in range(y_raw.shape[1]):
            col_data = y_raw[:, col_idx]

            # Check if column contains strings
            if col_data.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
                # Use LabelEncoder for string data
                le = LabelEncoder()
                y_numeric[:, col_idx] = le.fit_transform(col_data)
                transformers.append((f'col_{col_idx}', le, [col_idx]))
            elif not np.issubdtype(col_data.dtype, np.number):
                # Try to convert to numeric
                try:
                    y_numeric[:, col_idx] = col_data.astype(np.float32)
                except (ValueError, TypeError):
                    # Fall back to LabelEncoder
                    le = LabelEncoder()
                    y_numeric[:, col_idx] = le.fit_transform(col_data.astype(str))
                    transformers.append((f'col_{col_idx}', le, [col_idx]))

        # Create composite transformer if needed
        if transformers:
            transformer = ColumnTransformer(transformers, remainder='passthrough')
            transformer.fit(y_raw)
        else:
            # No transformation needed, create identity transformer
            from sklearn.preprocessing import FunctionTransformer
            transformer = FunctionTransformer(validate=False)
            transformer.fit(y_raw)

        # Store numeric version
        self._data["numeric"] = y_numeric.astype(np.float32)
        self.transformers_dict["numeric"] = (transformer, "raw")

        # Add to index
        self._add_to_index(y_raw.shape[0], "numeric", {"sample": indices})

    def _get_samples(self, processing: str) -> List[int]:
        """Get sample IDs for a specific processing version."""
        filtered = self.index.filter(pl.col("processing") == processing)
        return filtered["sample"].to_list()

    def y(self, filter_dict: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Get target arrays with filtering.

        Args:
            filter_dict: Dictionary of column: value pairs for filtering
                        Must include 'processing' key to specify version

        Returns:
            Filtered target array
        """
        if not self._data:
            raise ValueError("No targets available")

        filter_dict = filter_dict or {}

        # Default to numeric processing if not specified
        processing = filter_dict.get("y", "numeric")

        if processing not in self._data:
            available = list(self._data.keys())
            raise ValueError(f"Processing '{processing}' not found. Available: {available}")

        # Get data for this processing version
        y_data = self._data[processing]

        # Apply filtering if specified
        if filter_dict:
            filtered_indices = self._apply_filter(filter_dict)
            if len(filtered_indices) == 0:
                return np.array([]).reshape(0, y_data.shape[1])

            # Convert to row indices for this processing
            # sample_indices = self._samples_to_rows(filtered_indices, processing)
            return y_data[filtered_indices]

        return y_data

    def set_y(self, filter_dict: Dict[str, Any], y: np.ndarray, transformer: TransformerMixin, new_processing: str) -> None:
        """
        Add a new processing version of targets.

        Args:
            filter_dict: Dictionary specifying source processing and samples
            y: Transformed target array
            transformer: sklearn TransformerMixin used for transformation
            new_processing: Name for the new processing version
        """
        if new_processing in self._data:
            warnings.warn(f"Processing '{new_processing}' already exists, overwriting")

        source_processing = filter_dict.get("processing", "numeric")

        if source_processing not in self._data:
            raise ValueError(f"Source processing '{source_processing}' not found")

        # Validate shape
        source_samples = self._get_samples(source_processing)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if y.shape[0] != len(source_samples):
            raise ValueError(f"Target shape mismatch: expected {len(source_samples)} samples, got {y.shape[0]}")

        # Store new data and transformer
        self._data[new_processing] = y.copy()
        self.transformers_dict[new_processing] = (transformer, source_processing)

        # Add to index
        self._add_to_index(y.shape[0], new_processing, {"sample": source_samples})

        # Clear cache
        self._filtered_cache.clear()

    def invert_transform(self, y_predict: np.ndarray, from_processing: str,
                        to_processing: str = "raw") -> np.ndarray:
        """
        Chain inverse transforms from one processing version to another.

        Args:
            y_predict: Predictions to inverse transform
            from_processing: Starting processing version
            to_processing: Target processing version (usually "raw" or "numeric")

        Returns:
            Inverse transformed predictions
        """
        if from_processing == to_processing:
            return y_predict

        if from_processing not in self.transformers_dict:
            raise ValueError(f"No transformer found for processing '{from_processing}'")

        current_y = y_predict.copy()
        current_processing = from_processing

        # Chain inverse transforms
        visited = set()
        while current_processing != to_processing:
            if current_processing in visited:
                raise ValueError(f"Circular dependency detected in transformers: {visited}")
            visited.add(current_processing)

            if current_processing not in self.transformers_dict:
                raise ValueError(f"Cannot find transformer chain from '{from_processing}' to '{to_processing}'")

            transformer, source_processing = self.transformers_dict[current_processing]

            # Apply inverse transform
            current_y = transformer.inverse_transform(current_y)
            current_processing = source_processing

        return current_y

    def _apply_filter(self, filter_dict: Dict[str, Any]) -> List[int]:
        """Apply filters and return sample indices."""
        cache_key = str(sorted(filter_dict.items()))
        if cache_key in self._filtered_cache:
            return self._filtered_cache[cache_key]["sample"].to_list()

        # Start with full index
        filtered = self.index

        # Apply each filter
        for column, value in filter_dict.items():
            if column not in filtered.columns and column != "processing":
                continue

            if isinstance(value, (list, tuple, np.ndarray)):
                filtered = filtered.filter(pl.col(column).is_in(value))
            else:
                filtered = filtered.filter(pl.col(column) == value)

        # Cache result
        self._filtered_cache[cache_key] = filtered
        return filtered["sample"].unique().to_list()

    def _samples_to_rows(self, sample_indices: List[int], processing: str) -> np.ndarray:
        """Convert sample indices to row indices for a specific processing version."""
        filtered = self.index.filter(
            (pl.col("sample").is_in(sample_indices)) &
            (pl.col("processing") == processing)
        )

        # Get mapping from sample to row position in the processing data
        samples_in_processing = self._get_samples(processing)
        sample_to_pos = {sample: i for i, sample in enumerate(samples_in_processing)}

        # Convert to positions
        positions = [sample_to_pos[sample] for sample in sample_indices if sample in sample_to_pos]
        return np.array(positions)

    def get_processing_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all processing versions."""
        info = {}
        for processing in self.processing_versions:
            data = self._data[processing]
            info[processing] = {
                "shape": data.shape,
                "dtype": str(data.dtype),
                "samples": len(self._get_samples(processing))
            }

            if processing in self.transformers_dict:
                transformer, source = self.transformers_dict[processing]
                info[processing]["transformer"] = type(transformer).__name__
                info[processing]["source_processing"] = source

        return info

    def __repr__(self):
        if not self._data:
            return "Target(empty)"

        n_samples = self.n_samples
        processings = self.processing_versions
        return f"Target(samples={n_samples}, processings={processings})"

    def __str__(self):
        if not self._data:
            return "Target: No data"

        lines = [f"Target: {self.n_samples} samples"]
        for processing in self.processing_versions:
            shape = self._data[processing].shape
            lines.append(f"  {processing}: {shape}")

        return "\n".join(lines)