"""
Lazy loading support for datasets.

This module provides lazy loading capabilities to defer data loading
until it is actually needed, improving startup time and memory usage.

Phase 8 Implementation - Dataset Configuration Roadmap
Section 8.5: Performance Optimization - Lazy Loading
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)


class LazyArray:
    """A lazy-loading array wrapper.

    Defers loading until array data is actually accessed.
    Supports numpy array interface for compatibility.

    Example:
        ```python
        # Create lazy array
        lazy = LazyArray(
            loader=lambda: np.load("large_file.npy"),
            shape=(10000, 500),
            dtype=np.float32
        )

        # Array not loaded yet
        print(lazy.shape)  # (10000, 500)

        # Triggers loading on first access
        data = lazy[0:100]  # Now loads the data

        # Explicit loading
        lazy.load()
        full_data = lazy.data
        ```
    """

    def __init__(
        self,
        loader: Callable[[], np.ndarray],
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[np.dtype] = None,
        source_path: Optional[str] = None,
    ):
        """Initialize lazy array.

        Args:
            loader: Function to load the array data.
            shape: Expected shape (for metadata before loading).
            dtype: Expected dtype (for metadata before loading).
            source_path: Path to source file (for cache key).
        """
        self._loader = loader
        self._shape = shape
        self._dtype = dtype
        self._source_path = source_path
        self._data: Optional[np.ndarray] = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if data has been loaded."""
        return self._loaded

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Get array shape (may trigger load if unknown)."""
        if self._shape is not None:
            return self._shape
        self.load()
        return self._data.shape if self._data is not None else None

    @property
    def dtype(self) -> Optional[np.dtype]:
        """Get array dtype (may trigger load if unknown)."""
        if self._dtype is not None:
            return self._dtype
        self.load()
        return self._data.dtype if self._data is not None else None

    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        shape = self.shape
        return len(shape) if shape else 0

    @property
    def data(self) -> np.ndarray:
        """Get the loaded data (triggers load if needed)."""
        self.load()
        return self._data

    def load(self) -> np.ndarray:
        """Load the data if not already loaded.

        Returns:
            The loaded numpy array.
        """
        if not self._loaded:
            logger.debug(f"Loading lazy array from {self._source_path or 'memory'}")
            self._data = self._loader()
            self._loaded = True

            # Update shape/dtype from actual data
            if self._data is not None:
                self._shape = self._data.shape
                self._dtype = self._data.dtype

        return self._data

    def unload(self) -> None:
        """Unload data to free memory."""
        if self._loaded:
            logger.debug(f"Unloading lazy array from {self._source_path or 'memory'}")
            self._data = None
            self._loaded = False

    def __len__(self) -> int:
        """Get length (first dimension)."""
        shape = self.shape
        return shape[0] if shape else 0

    def __getitem__(self, key):
        """Get item from array (triggers load)."""
        return self.data[key]

    def __array__(self, dtype=None):
        """Support numpy array conversion."""
        data = self.data
        if dtype is not None:
            return np.asarray(data, dtype=dtype)
        return data

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"LazyArray(shape={self._shape}, dtype={self._dtype}, {status})"


class LazyDataset:
    """A lazy-loading dataset wrapper.

    Wraps multiple data components (X, y, metadata) as lazy arrays
    that load on demand.

    Example:
        ```python
        # Create from loader functions
        dataset = LazyDataset(
            x_loader=lambda: load_features("X.csv"),
            y_loader=lambda: load_targets("Y.csv"),
            metadata_loader=lambda: load_metadata("M.csv")
        )

        # Nothing loaded yet
        print(dataset.x_shape)  # Returns cached shape if known

        # Triggers X loading only
        X_data = dataset.X

        # Load everything
        dataset.load_all()
        ```
    """

    def __init__(
        self,
        x_loader: Optional[Callable[[], np.ndarray]] = None,
        y_loader: Optional[Callable[[], np.ndarray]] = None,
        metadata_loader: Optional[Callable[[], Any]] = None,
        x_shape: Optional[Tuple[int, ...]] = None,
        y_shape: Optional[Tuple[int, ...]] = None,
        name: str = "dataset",
    ):
        """Initialize lazy dataset.

        Args:
            x_loader: Function to load features.
            y_loader: Function to load targets.
            metadata_loader: Function to load metadata.
            x_shape: Expected shape of X (for pre-load info).
            y_shape: Expected shape of y (for pre-load info).
            name: Dataset name.
        """
        self.name = name
        self._x_loader = x_loader
        self._y_loader = y_loader
        self._metadata_loader = metadata_loader

        self._X: Optional[LazyArray] = None
        self._y: Optional[LazyArray] = None
        self._metadata: Optional[Any] = None
        self._metadata_loaded = False

        if x_loader:
            self._X = LazyArray(x_loader, shape=x_shape)
        if y_loader:
            self._y = LazyArray(y_loader, shape=y_shape)

    @property
    def X(self) -> Optional[np.ndarray]:
        """Get features (triggers load if needed)."""
        if self._X is None:
            return None
        return self._X.data

    @property
    def y(self) -> Optional[np.ndarray]:
        """Get targets (triggers load if needed)."""
        if self._y is None:
            return None
        return self._y.data

    @property
    def metadata(self) -> Optional[Any]:
        """Get metadata (triggers load if needed)."""
        if self._metadata_loader is not None and not self._metadata_loaded:
            logger.debug(f"Loading metadata for {self.name}")
            self._metadata = self._metadata_loader()
            self._metadata_loaded = True
        return self._metadata

    @property
    def x_shape(self) -> Optional[Tuple[int, ...]]:
        """Get X shape without loading."""
        return self._X.shape if self._X else None

    @property
    def y_shape(self) -> Optional[Tuple[int, ...]]:
        """Get y shape without loading."""
        return self._y.shape if self._y else None

    @property
    def n_samples(self) -> int:
        """Get number of samples."""
        shape = self.x_shape
        return shape[0] if shape else 0

    @property
    def n_features(self) -> int:
        """Get number of features."""
        shape = self.x_shape
        return shape[1] if shape and len(shape) > 1 else 0

    @property
    def is_x_loaded(self) -> bool:
        """Check if X is loaded."""
        return self._X is not None and self._X.is_loaded

    @property
    def is_y_loaded(self) -> bool:
        """Check if y is loaded."""
        return self._y is not None and self._y.is_loaded

    @property
    def is_metadata_loaded(self) -> bool:
        """Check if metadata is loaded."""
        return self._metadata_loaded

    def load_all(self) -> None:
        """Load all data components."""
        if self._X:
            self._X.load()
        if self._y:
            self._y.load()
        if self._metadata_loader and not self._metadata_loaded:
            self._metadata = self._metadata_loader()
            self._metadata_loaded = True

    def unload_all(self) -> None:
        """Unload all data to free memory."""
        if self._X:
            self._X.unload()
        if self._y:
            self._y.unload()
        self._metadata = None
        self._metadata_loaded = False

    def __repr__(self) -> str:
        x_status = "loaded" if self.is_x_loaded else "not loaded"
        y_status = "loaded" if self.is_y_loaded else "not loaded"
        return (
            f"LazyDataset(name={self.name}, "
            f"X={self.x_shape} {x_status}, "
            f"y={self.y_shape} {y_status})"
        )


def create_lazy_dataset(
    train_x_path: Optional[str] = None,
    train_y_path: Optional[str] = None,
    train_group_path: Optional[str] = None,
    load_params: Optional[Dict[str, Any]] = None,
) -> LazyDataset:
    """Create a lazy dataset from file paths.

    Args:
        train_x_path: Path to training features.
        train_y_path: Path to training targets.
        train_group_path: Path to training metadata.
        load_params: Loading parameters.

    Returns:
        LazyDataset instance.
    """
    from nirs4all.data.loaders.csv_loader import load_csv

    load_params = load_params or {}

    def make_x_loader():
        if train_x_path is None:
            return None
        def loader():
            result = load_csv(train_x_path, data_type='x', **load_params)
            return result[0].values if result[0] is not None else None
        return loader

    def make_y_loader():
        if train_y_path is None:
            return None
        def loader():
            result = load_csv(train_y_path, data_type='y', **load_params)
            return result[0].values if result[0] is not None else None
        return loader

    def make_metadata_loader():
        if train_group_path is None:
            return None
        def loader():
            result = load_csv(train_group_path, data_type='x', **load_params)
            return result[0] if result[0] is not None else None
        return loader

    return LazyDataset(
        x_loader=make_x_loader(),
        y_loader=make_y_loader(),
        metadata_loader=make_metadata_loader(),
        name=Path(train_x_path).stem if train_x_path else "dataset",
    )
