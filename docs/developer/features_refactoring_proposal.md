# Features Refactoring Proposal - Complete Modularization

## Overview

This proposal outlines a comprehensive refactoring of `features.py` and `feature_source.py` to improve modularity, maintainability, and type safety while ensuring **100% backward compatibility**.

---

## 1. Module Structure & Imports

### New Directory Structure

```
nirs4all/data/
├── feature_components/           # NEW MODULE
│   ├── __init__.py              # Exports all components + constants
│   ├── feature_constants.py     # Enums and constants
│   ├── feature_source.py        # Refactored FeatureSource (moved here)
│   ├── array_storage.py
│   ├── processing_manager.py
│   ├── header_manager.py
│   ├── layout_transformer.py
│   ├── update_strategy.py
│   └── augmentation_handler.py
├── features.py                   # Features class (imports from feature_components)
├── dataset.py                    # Dataset (imports from feature_components)
└── __init__.py                   # Provides backward-compatible imports
```

### Backward-Compatible Import Strategy

```python
# File: nirs4all/data/__init__.py

# Provide shortcut imports for common classes
from nirs4all.data.feature_components import (
    FeatureSource,           # Main class
    FeatureLayout,           # Enum
    HeaderUnit,              # Enum
    feature_constants,       # Module for direct access
)

# This allows both:
# from nirs4all.data import FeatureSource, FeatureLayout
# from nirs4all.data.feature_components import FeatureSource, FeatureLayout
```

```python
# File: nirs4all/data/feature_components/__init__.py

"""Feature management components and utilities."""

from .feature_constants import (
    FeatureLayout,
    HeaderUnit,
    LayoutType,
    HeaderUnitType,
    DEFAULT_PROCESSING,
    normalize_layout,
    normalize_header_unit,
)
from .feature_source import FeatureSource
from .array_storage import ArrayStorage
from .processing_manager import ProcessingManager
from .header_manager import HeaderManager
from .layout_transformer import LayoutTransformer
from .update_strategy import UpdateStrategy, ReplacementOperation, AdditionOperation
from .augmentation_handler import AugmentationHandler

__all__ = [
    # Constants and enums
    "FeatureLayout",
    "HeaderUnit",
    "LayoutType",
    "HeaderUnitType",
    "DEFAULT_PROCESSING",
    "normalize_layout",
    "normalize_header_unit",
    # Main class
    "FeatureSource",
    # Internal components
    "ArrayStorage",
    "ProcessingManager",
    "HeaderManager",
    "LayoutTransformer",
    "UpdateStrategy",
    "ReplacementOperation",
    "AdditionOperation",
    "AugmentationHandler",
]
```

---

## 2. Constants & Enums Module

### File: `nirs4all/data/feature_components/feature_constants.py`

```python
"""Constants and enums for feature management."""

from enum import Enum
from typing import Literal

# Default processing identifier
DEFAULT_PROCESSING = "raw"


class FeatureLayout(str, Enum):
    """Feature data layout formats.

    String values ensure backward compatibility with existing pipelines
    that use layout="3d_transpose" as strings.
    """
    FLAT_2D = "2d"                      # (samples, processings * features)
    FLAT_2D_INTERLEAVED = "2d_interleaved"  # (samples, features * processings)
---

## 3. Feature Components Implementation

### File: `nirs4all/data/feature_components/array_storage.py`tibility.

    Args:
        layout: Layout as string or enum

    Returns:
        FeatureLayout enum value

    Raises:
        ValueError: If layout string is invalid
    """
    if isinstance(layout, FeatureLayout):
        return layout

    try:
        return FeatureLayout(layout)
    except ValueError:
        valid = [e.value for e in FeatureLayout]
        raise ValueError(f"Invalid layout '{layout}'. Valid options: {valid}")


def normalize_header_unit(unit: Union[str, HeaderUnit]) -> HeaderUnit:
    """Convert string header unit to enum.

    Args:
        unit: Unit as string or enum

    Returns:
        HeaderUnit enum value

    Raises:
        ValueError: If unit string is invalid
    """
    if isinstance(unit, HeaderUnit):
        return unit

    try:
        return HeaderUnit(unit)
    except ValueError:
        valid = [e.value for e in HeaderUnit]
        raise ValueError(f"Invalid header unit '{unit}'. Valid options: {valid}")
```

### Backward Compatibility Strategy

```python
# OLD CODE (still works):
dataset.x({"partition": "train"}, layout="3d_transpose")

# NEW CODE (also works):
from nirs4all.data.feature_constants import FeatureLayout
dataset.x({"partition": "train"}, layout=FeatureLayout.VOLUME_3D_TRANSPOSE)

# Both produce identical results due to str(FeatureLayout.VOLUME_3D_TRANSPOSE) == "3d_transpose"
```

---

## 2. Feature Components Module

### File: `nirs4all/data/feature_components/__init__.py`

```python
"""Modular components for feature management."""

from .array_storage import ArrayStorage
from .processing_manager import ProcessingManager
from .header_manager import HeaderManager
from .layout_transformer import LayoutTransformer
from .update_strategy import UpdateStrategy, ReplacementOperation, AdditionOperation
from .augmentation_handler import AugmentationHandler

__all__ = [
    "ArrayStorage",
    "ProcessingManager",
    "HeaderManager",
    "LayoutTransformer",
    "UpdateStrategy",
    "ReplacementOperation",
    "AdditionOperation",
    "AugmentationHandler",
]
```

---

### File: `nirs4all/data/feature_components/array_storage.py`

```python
"""Low-level 3D array storage with padding support."""

import numpy as np
from typing import Optional


class ArrayStorage:
    """Manages the 3D numpy array (samples, processings, features) with padding.

    This is the lowest-level component responsible only for array manipulation
    and memory management.

    Attributes:
        padding: Whether to allow padding when adding features with fewer dimensions.
        pad_value: Value to use for padding.
    """

    def __init__(self, padding: bool = True, pad_value: float = 0.0, dtype=np.float32):
        """Initialize empty array storage.

        Args:
            padding: If True, allow padding features to match existing dimensions.
            pad_value: Value to use for padding missing features.
            dtype: NumPy data type for the array.
        """
        self.padding = padding
        self.pad_value = pad_value
        self.dtype = dtype
        self._array = np.empty((0, 1, 0), dtype=dtype)

    @property
    def array(self) -> np.ndarray:
        """Get the underlying array."""
        return self._array

    @property
    def shape(self) -> tuple:
        """Get array shape (samples, processings, features)."""
        return self._array.shape

    @property
    def num_samples(self) -> int:
        """Number of samples (first dimension)."""
        return self._array.shape[0]

    @property
    def num_processings(self) -> int:
        """Number of processings (second dimension)."""
        return self._array.shape[1]

    @property
    def num_features(self) -> int:
        """Number of features (third dimension)."""
        return self._array.shape[2]

    def add_samples(self, data: np.ndarray) -> None:
        """Append samples along first dimension.

        Args:
            data: 3D array of shape (n_new_samples, processings, features)
                  or 2D array of shape (n_new_samples, features) which will be
                  expanded to (n_new_samples, 1, features).
        """
        if data.ndim == 2:
            data = data[:, None, :]

        if self.num_samples == 0:
            self._array = data
        else:
            # Ensure compatible processing and feature dimensions
            if data.shape[1] != self.num_processings:
                raise ValueError(
                    f"Processing dimension mismatch: expected {self.num_processings}, "
                    f"got {data.shape[1]}"
                )

            prepared = self._prepare_for_storage(data)
            self._array = np.concatenate((self._array, prepared), axis=0)

    def add_processings(self, data: np.ndarray) -> None:
        """Append processings along second dimension.

        Args:
            data: 3D array of shape (samples, n_new_processings, features)
        """
        if data.shape[0] != self.num_samples:
            raise ValueError(
                f"Sample dimension mismatch: expected {self.num_samples}, "
                f"got {data.shape[0]}"
            )

        prepared = self._prepare_for_storage(data)

        if self.num_samples == 0:
            self._array = prepared
        else:
            self._array = np.concatenate((self._array, prepared), axis=1)

    def replace_processing(self, proc_index: int, data: np.ndarray) -> None:
        """Replace data at specific processing index.

        Args:
            proc_index: Processing dimension index to replace.
            data: 2D array of shape (samples, features).
        """
        prepared = self._prepare_for_storage(data)
        self._array[:, proc_index, :] = prepared

    def resize_features(self, new_num_features: int) -> None:
        """Resize feature dimension, preserving data where possible.

        Args:
            new_num_features: New size for feature dimension.
        """
        if new_num_features == self.num_features:
            return

        if self.num_samples == 0:
            self._array = np.empty((0, self.num_processings, new_num_features), dtype=self.dtype)
            return

        new_shape = (self.num_samples, self.num_processings, new_num_features)
        new_array = np.full(new_shape, self.pad_value, dtype=self.dtype)

        # Copy existing data
        min_features = min(self.num_features, new_num_features)
        new_array[:, :, :min_features] = self._array[:, :, :min_features]

        self._array = new_array

    def get_samples(self, indices: np.ndarray, processing_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Retrieve samples with optional processing filtering.

        Args:
            indices: Sample indices to retrieve.
            processing_indices: Optional processing indices to include.

        Returns:
            3D array subset.
        """
        if processing_indices is None:
            return self._array[indices, :, :]
        return self._array[indices, :, :][:, processing_indices, :]

    def _prepare_for_storage(self, data: np.ndarray) -> np.ndarray:
        """Prepare data for storage by handling padding.

        Args:
            data: Array to prepare (can be 2D or 3D).

        Returns:
            Prepared array with matching dimensions.

        Raises:
            ValueError: If padding disabled and dimensions don't match.
        """
        if self.num_samples == 0:
            return data

        # Determine feature dimension based on input shape
        feature_dim = data.shape[-1]

        if feature_dim == self.num_features:
            return data

        if feature_dim < self.num_features and self.padding:
            # Pad to match existing dimension
            if data.ndim == 2:
                padded = np.full((data.shape[0], self.num_features), self.pad_value, dtype=self.dtype)
                padded[:, :feature_dim] = data
            else:  # 3D
                padded = np.full(
                    (data.shape[0], data.shape[1], self.num_features),
                    self.pad_value,
                    dtype=self.dtype
                )
                padded[:, :, :feature_dim] = data
            return padded

        if not self.padding:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.num_features}, "
                f"got {feature_dim}. Padding is disabled."
            )

        return data

    def __repr__(self) -> str:
        return f"ArrayStorage(shape={self.shape}, dtype={self.dtype})"
```

---

### File: `nirs4all/data/feature_components/processing_manager.py`

```python
"""Processing ID management and tracking."""

from typing import List, Dict, Optional
from nirs4all.data.feature_constants import DEFAULT_PROCESSING


class ProcessingManager:
    """Manages processing IDs and their mapping to array indices.

    Keeps track of which processing is stored at which index in the
    second dimension of the 3D array.
    """

    def __init__(self, default_processing: str = DEFAULT_PROCESSING):
        """Initialize with default processing.

        Args:
            default_processing: Initial processing name (typically "raw").
        """
        self._processing_ids: List[str] = [default_processing]
        self._id_to_index: Dict[str, int] = {default_processing: 0}

    @property
    def processing_ids(self) -> List[str]:
        """Get copy of processing ID list."""
        return self._processing_ids.copy()

    @property
    def num_processings(self) -> int:
        """Get number of processings."""
        return len(self._processing_ids)

    def get_index(self, processing_id: str) -> Optional[int]:
        """Get index for a processing ID.

        Args:
            processing_id: Processing identifier.

        Returns:
            Index in array, or None if not found.
        """
        return self._id_to_index.get(processing_id)

    def has_processing(self, processing_id: str) -> bool:
        """Check if processing exists.

        Args:
            processing_id: Processing identifier.

        Returns:
            True if processing exists.
        """
        return processing_id in self._id_to_index

    def add_processing(self, processing_id: str) -> int:
        """Add new processing and return its index.

        Args:
            processing_id: New processing identifier.

        Returns:
            Index assigned to the new processing.

        Raises:
            ValueError: If processing already exists.
        """
        if processing_id in self._id_to_index:
            raise ValueError(f"Processing '{processing_id}' already exists")

        new_index = len(self._processing_ids)
        self._processing_ids.append(processing_id)
        self._id_to_index[processing_id] = new_index
        return new_index

    def rename_processing(self, old_id: str, new_id: str) -> None:
        """Rename a processing.

        Args:
            old_id: Current processing identifier.
            new_id: New processing identifier.

        Raises:
            ValueError: If old_id doesn't exist or new_id already exists.
        """
        if old_id not in self._id_to_index:
            raise ValueError(f"Processing '{old_id}' does not exist")

        if new_id in self._id_to_index and new_id != old_id:
            raise ValueError(f"Processing '{new_id}' already exists")

        if old_id == new_id:
            return

        index = self._id_to_index[old_id]
        self._processing_ids[index] = new_id
        del self._id_to_index[old_id]
        self._id_to_index[new_id] = index

    def __repr__(self) -> str:
        return f"ProcessingManager(ids={self._processing_ids})"
```

---

### File: `nirs4all/data/feature_components/header_manager.py`

```python
"""Feature header and unit management."""

from typing import Optional, List
from nirs4all.data.feature_constants import HeaderUnit, normalize_header_unit


class HeaderManager:
    """Manages feature headers and their units.

    Headers can be wavelengths, wavenumbers, or arbitrary feature names.
    """

    def __init__(self):
        """Initialize empty header manager."""
        self._headers: Optional[List[str]] = None
        self._unit: HeaderUnit = HeaderUnit.WAVENUMBER

    @property
    def headers(self) -> Optional[List[str]]:
        """Get feature headers."""
        return self._headers

    @property
    def unit(self) -> HeaderUnit:
        """Get header unit type."""
        return self._unit

    @property
    def unit_str(self) -> str:
        """Get header unit as string (backward compatibility)."""
        return str(self._unit)

    def set_headers(self, headers: Optional[List[str]], unit: Optional[str] = None) -> None:
        """Set feature headers with optional unit.

        Args:
            headers: List of header strings.
            unit: Unit type (defaults to existing unit if None).
        """
        self._headers = headers
        if unit is not None:
            self._unit = normalize_header_unit(unit)

    def clear_headers(self) -> None:
        """Clear headers (e.g., after feature dimension change)."""
        self._headers = None

    def validate_headers(self, num_features: int) -> bool:
        """Check if headers match feature count.

        Args:
            num_features: Expected number of features.

        Returns:
            True if headers match or are None.
        """
        if self._headers is None:
            return True
        return len(self._headers) == num_features

    def __repr__(self) -> str:
        n = len(self._headers) if self._headers else 0
        return f"HeaderManager(headers={n}, unit={self._unit})"
```

---

### File: `nirs4all/data/feature_components/layout_transformer.py`

```python
"""Layout transformation for feature arrays."""

import numpy as np
from nirs4all.data.feature_constants import FeatureLayout, normalize_layout
from typing import Union


class LayoutTransformer:
    """Transforms 3D feature arrays to different layouts.

    Handles all layout conversions without modifying the source array.
    """

    @staticmethod
    def transform(data: np.ndarray, layout: Union[str, FeatureLayout]) -> np.ndarray:
        """Transform 3D array to specified layout.

        Args:
            data: Input 3D array of shape (samples, processings, features).
            layout: Target layout (string or enum).

        Returns:
            Transformed array in requested layout.

        Raises:
            ValueError: If layout is invalid or data is not 3D.
        """
        if data.ndim != 3:
            raise ValueError(f"Expected 3D input array, got {data.ndim}D")

        layout_enum = normalize_layout(layout)

        if layout_enum == FeatureLayout.FLAT_2D:
            # (samples, processings, features) -> (samples, processings * features)
            return data.reshape(data.shape[0], -1)

        elif layout_enum == FeatureLayout.FLAT_2D_INTERLEAVED:
            # (samples, processings, features) -> (samples, features, processings) -> (samples, features * processings)
            return np.transpose(data, (0, 2, 1)).reshape(data.shape[0], -1)

        elif layout_enum == FeatureLayout.VOLUME_3D:
            # Keep as-is
            return data

        elif layout_enum == FeatureLayout.VOLUME_3D_TRANSPOSE:
            # (samples, processings, features) -> (samples, features, processings)
            return np.transpose(data, (0, 2, 1))

        else:
            raise ValueError(f"Unsupported layout: {layout_enum}")

    @staticmethod
    def get_empty_array(layout: Union[str, FeatureLayout], num_processings: int, num_features: int, dtype=np.float32) -> np.ndarray:
        """Create empty array with correct shape for layout.

        Args:
            layout: Target layout.
            num_processings: Number of processings.
            num_features: Number of features.
            dtype: Array data type.

        Returns:
            Empty array with shape (0, ...) appropriate for layout.
        """
        layout_enum = normalize_layout(layout)

        if layout_enum in [FeatureLayout.FLAT_2D, FeatureLayout.FLAT_2D_INTERLEAVED]:
            total_features = num_processings * num_features
            return np.empty((0, total_features), dtype=dtype)
        else:  # 3D layouts
            return np.empty((0, num_processings, num_features), dtype=dtype)
```

---

### File: `nirs4all/data/feature_components/update_strategy.py`

```python
"""Strategy pattern for feature update operations."""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ReplacementOperation:
    """Represents a processing replacement operation."""
    proc_index: int
    new_data: np.ndarray
    new_proc_name: str


@dataclass
class AdditionOperation:
    """Represents a processing addition operation."""
    new_data: np.ndarray
    new_proc_name: str


class UpdateStrategy:
    """Categorizes and validates feature update operations.

    This strategy pattern separates the logic of determining what
    operations to perform from actually performing them.
    """

    def __init__(self, processing_manager, array_storage):
        """Initialize strategy with dependencies.

        Args:
            processing_manager: ProcessingManager instance.
            array_storage: ArrayStorage instance.
        """
        self.processing_manager = processing_manager
        self.array_storage = array_storage

    def categorize_operations(
        self,
        features: List[np.ndarray],
        source_processings: List[str],
        target_processings: List[str]
    ) -> Tuple[List[ReplacementOperation], List[AdditionOperation]]:
        """Categorize update operations into replacements and additions.

        Args:
            features: List of feature arrays.
            source_processings: List of existing processing names ("" = add new).
            target_processings: List of target processing names.

        Returns:
            Tuple of (replacements, additions).

        Raises:
            ValueError: If operations are invalid.
        """
        # Normalize source_processings
        if len(source_processings) == 0:
            source_processings = [""] * len(target_processings)

        if len(features) != len(source_processings) or len(features) != len(target_processings):
            raise ValueError(
                f"Length mismatch: features={len(features)}, "
                f"source_processings={len(source_processings)}, "
                f"target_processings={len(target_processings)}"
            )

        replacements = []
        additions = []

        for arr, source_proc, target_proc in zip(features, source_processings, target_processings):
            if source_proc == "":
                # Add new processing
                if self.processing_manager.has_processing(target_proc):
                    raise ValueError(f"Processing '{target_proc}' already exists, cannot add")
                additions.append(AdditionOperation(new_data=arr, new_proc_name=target_proc))
            else:
                # Replace existing processing
                proc_index = self.processing_manager.get_index(source_proc)
                if proc_index is None:
                    raise ValueError(f"Source processing '{source_proc}' does not exist")

                if target_proc != source_proc and self.processing_manager.has_processing(target_proc):
                    raise ValueError(f"Target processing '{target_proc}' already exists")

                replacements.append(ReplacementOperation(
                    proc_index=proc_index,
                    new_data=arr,
                    new_proc_name=target_proc
                ))

        return replacements, additions

    def should_resize_features(self, replacements: List[ReplacementOperation], additions: List[AdditionOperation]) -> Optional[int]:
        """Determine if feature dimension should be resized.

        Resizing is needed when:
        - All processings are being replaced
        - All have same new feature dimension
        - New dimension differs from current

        Args:
            replacements: List of replacement operations.
            additions: List of addition operations.

        Returns:
            New feature dimension if resize needed, None otherwise.
        """
        if not replacements or additions:
            return None

        # Get new feature dimensions
        new_feature_dims = [op.new_data.shape[1] for op in replacements]

        # Check all same dimension and different from current
        if len(set(new_feature_dims)) == 1:
            new_dim = new_feature_dims[0]
            if new_dim != self.array_storage.num_features:
                return new_dim

        return None
```

---

### File: `nirs4all/data/feature_components/augmentation_handler.py`

```python
"""Handles sample augmentation operations."""

import numpy as np
from typing import List


class AugmentationHandler:
    """Manages sample augmentation for feature sources.

    Augmentation creates new samples by duplicating existing ones
    and adding new processing data.
    """

    def __init__(self, array_storage, processing_manager):
        """Initialize handler with dependencies.

        Args:
            array_storage: ArrayStorage instance.
            processing_manager: ProcessingManager instance.
        """
        self.array_storage = array_storage
        self.processing_manager = processing_manager

    def augment_samples(
        self,
        sample_indices: List[int],
        data: np.ndarray,
        processings: List[str],
        count_list: List[int]
    ) -> None:
        """Create augmented samples.

        Args:
            sample_indices: Original sample indices to augment.
            data: Augmented feature data (total_augmented_samples, n_features).
            processings: Processing names for augmented data.
            count_list: Number of augmentations per sample.

        Raises:
            ValueError: If data shape doesn't match expected augmentation count.
        """
        if not sample_indices:
            return

        total_augmentations = sum(count_list)
        if total_augmentations == 0:
            return

        # Validate input
        if data.ndim != 2:
            raise ValueError(f"data must be 2D array, got {data.ndim}D")

        if data.shape[0] != total_augmentations:
            raise ValueError(
                f"data must have {total_augmentations} samples, got {data.shape[0]}"
            )

        # Prepare data
        prep_data = self.array_storage._prepare_for_storage(data)

        # Expand array to accommodate new samples
        self._expand_for_augmented_samples(sample_indices, count_list)

        # Add new processing(s)
        for proc_name in processings:
            if not self.processing_manager.has_processing(proc_name):
                self._add_augmentation_processing(proc_name, prep_data, total_augmentations)

    def _expand_for_augmented_samples(self, sample_indices: List[int], count_list: List[int]) -> None:
        """Expand array to fit augmented samples."""
        total_augmentations = sum(count_list)
        new_num_samples = self.array_storage.num_samples + total_augmentations
        current_processings = self.array_storage.num_processings

        new_shape = (new_num_samples, current_processings, self.array_storage.num_features)
        expanded_array = np.full(new_shape, self.array_storage.pad_value, dtype=self.array_storage.dtype)

        # Copy existing data
        expanded_array[:self.array_storage.num_samples, :, :] = self.array_storage.array

        # Duplicate original samples for augmentation
        sample_idx = 0
        for orig_idx, aug_count in zip(sample_indices, count_list):
            for _ in range(aug_count):
                expanded_array[self.array_storage.num_samples + sample_idx, :, :] = \
                    self.array_storage.array[orig_idx, :, :]
                sample_idx += 1

        # Update storage
        self.array_storage._array = expanded_array

    def _add_augmentation_processing(self, proc_name: str, data: np.ndarray, total_augmentations: int) -> None:
        """Add new processing dimension for augmented samples."""
        current_processings = self.array_storage.num_processings

        # Expand to include new processing
        new_shape = (self.array_storage.num_samples, current_processings + 1, self.array_storage.num_features)
        expanded_array = np.full(new_shape, self.array_storage.pad_value, dtype=self.array_storage.dtype)

        # Copy existing data
        expanded_array[:, :current_processings, :] = self.array_storage.array

        # Add new processing data to augmented samples only
        augmented_start_idx = self.array_storage.num_samples - total_augmentations
        for i in range(total_augmentations):
            augmented_sample_idx = augmented_start_idx + i
            expanded_array[augmented_sample_idx, current_processings, :] = data[i, :]

        # Update storage and manager
        self.array_storage._array = expanded_array
        self.processing_manager.add_processing(proc_name)
```

---

## 3. Refactored FeatureSource

### File: `nirs4all/data/feature_source.py` (Refactored)

```python
"""Refactored FeatureSource using component-based architecture."""

import numpy as np
from typing import List, Optional, Union
from nirs4all.data.helpers import InputFeatures, ProcessingList, SampleIndices
from nirs4all.data.feature_constants import FeatureLayout, HeaderUnit, normalize_layout
from nirs4all.data.feature_components import (
    ArrayStorage,
    ProcessingManager,
    HeaderManager,
    LayoutTransformer,
    UpdateStrategy,
    AugmentationHandler,
)


class FeatureSource:
    """Manages a 3D numpy array of features using modular components.

---

## 4. Refactored FeatureSource

### File: `nirs4all/data/feature_components/feature_source.py` (Moved & Refactored)
    - UpdateStrategy: Update operation logic
    - AugmentationHandler: Sample augmentation

    Attributes:
        padding: Whether to allow padding when adding features with fewer dimensions.
        pad_value: Value to use for padding.
    """

    def __init__(self, padding: bool = True, pad_value: float = 0.0):
        """Initialize an empty FeatureSource.

        Args:
            padding: If True, allow padding features to match existing dimensions.
            pad_value: Value to use for padding missing features.
        """
        # Initialize components
        self._storage = ArrayStorage(padding=padding, pad_value=pad_value)
        self._processing_mgr = ProcessingManager()
        self._header_mgr = HeaderManager()
        self._layout_transformer = LayoutTransformer()
        self._update_strategy = UpdateStrategy(self._processing_mgr, self._storage)
        self._augmentation_handler = AugmentationHandler(self._storage, self._processing_mgr)

        # Expose for backward compatibility
        self.padding = padding
        self.pad_value = pad_value

    # ========== Properties (backward compatibility) ==========

    @property
    def headers(self) -> Optional[List[str]]:
        """Get the feature headers."""
        return self._header_mgr.headers

    @property
    def header_unit(self) -> str:
        """Get the unit type of the headers (as string for BC)."""
        return self._header_mgr.unit_str

    @property
    def num_samples(self) -> int:
        """Get the number of samples."""
        return self._storage.num_samples

    @property
    def num_processings(self) -> int:
        """Get the number of processing stages."""
        return self._processing_mgr.num_processings

    @property
    def num_features(self) -> int:
        """Get the number of features per processing."""
        return self._storage.num_features

    @property
    def num_2d_features(self) -> int:
        """Get total features when flattened to 2D."""
        return self._storage.num_processings * self._storage.num_features

    @property
    def processing_ids(self) -> List[str]:
        """Get a copy of the processing ID list."""
        return self._processing_mgr.processing_ids

    # ========== Public API ==========

    def add_samples(self, new_samples: np.ndarray, headers: Optional[List[str]] = None) -> None:
        """Add new samples to the feature source.

        Only allowed when there's only one processing (raw).

        Args:
            new_samples: 2D array of shape (n_samples, n_features).
            headers: Optional list of feature header names.

        Raises:
            ValueError: If dataset already has multiple processings or new_samples is not 2D.
        """
        if self.num_processings > 1:
            raise ValueError("Cannot add new samples to a dataset that already has been processed.")

        if new_samples.ndim != 2:
            raise ValueError(f"new_samples must be a 2D array, got {new_samples.ndim} dimensions")

        X = np.asarray(new_samples, dtype=self._storage.dtype)

        # Convert to 3D and add
        X_3d = X[:, None, :]
        self._storage.add_samples(X_3d)

        # Update headers
        if headers is not None:
            self._header_mgr.set_headers(headers)

    def set_headers(self, headers: Optional[List[str]], unit: str = "cm-1") -> None:
        """Set feature headers with unit metadata.

        Args:
            headers: List of header strings.
            unit: Unit type ("cm-1", "nm", "none", "text", "index").
        """
        self._header_mgr.set_headers(headers, unit)

    def update_features(self, source_processings: ProcessingList, features: InputFeatures, processings: ProcessingList) -> None:
        """Add new features or replace existing ones.

        Args:
            source_processings: List of existing processing names to replace ("" = add new).
            features: Feature arrays to add or replace.
            processings: Target processing names for the features.

        Example:
            # Add new 'savgol' and 'detrend', replace 'raw' with 'msc'
            update_features(["", "raw", ""],
                           [savgol_data, msc_data, detrend_data],
                           ["savgol", "msc", "detrend"])
        """
        # Normalize features to list of arrays
        feature_list = self._normalize_feature_input(features)
        if not feature_list:
            return

        # Categorize operations
        replacements, additions = self._update_strategy.categorize_operations(
            feature_list, source_processings, processings
        )

        # Check if feature dimension should be resized
        new_feature_dim = self._update_strategy.should_resize_features(replacements, additions)
        if new_feature_dim is not None:
            self._storage.resize_features(new_feature_dim)
            self._header_mgr.clear_headers()

        # Apply operations
        self._apply_replacements(replacements)
        self._apply_additions(additions)

    def augment_samples(
        self,
        sample_indices: List[int],
        data: np.ndarray,
        processings: List[str],
        count_list: List[int]
    ) -> None:
        """Create augmented samples by duplicating and adding processing data.

        Args:
            sample_indices: Original sample indices to augment.
            data: Augmented feature data (total_augmented_samples, n_features).
            processings: Processing names for the augmented data.
            count_list: Number of augmentations per sample.
        """
        if isinstance(processings, str):
            processings = [processings]

        self._augmentation_handler.augment_samples(
            sample_indices, data, processings, count_list
        )

    def x(self, indices: SampleIndices, layout: Union[str, FeatureLayout]) -> np.ndarray:
        """Retrieve feature data in specified layout.

        Args:
            indices: Sample indices to retrieve.
            layout: Output format ("2d", "2d_interleaved", "3d", "3d_transpose" or enum).

        Returns:
            Feature array in requested layout.

        Raises:
            ValueError: If layout is unknown.
        """
        # Handle empty indices
        if len(indices) == 0:
            return self._layout_transformer.get_empty_array(
                layout, self.num_processings, self.num_features, self._storage.dtype
            )

        # Get 3D data
        indices_array = np.asarray(indices)
        data_3d = self._storage.get_samples(indices_array)

        # Transform to requested layout
        return self._layout_transformer.transform(data_3d, layout)

    # ========== Private helpers ==========

    def _normalize_feature_input(self, features: InputFeatures) -> List[np.ndarray]:
        """Normalize various feature input types to list of arrays."""
        if isinstance(features, np.ndarray):
            return [features]
        elif isinstance(features, list):
            if not features:
                return []
            # Handle list of lists (multi-source case) - take first source
            if isinstance(features[0], list):
                return list(features[0])
            elif isinstance(features[0], np.ndarray):
                return list(features)
        return []

    def _apply_replacements(self, replacements) -> None:
        """Apply replacement operations."""
        for op in replacements:
            # Prepare data and replace
            prepared = self._storage._prepare_for_storage(op.new_data)
            self._storage.replace_processing(op.proc_index, prepared)

            # Update processing name if changed
            old_name = self.processing_ids[op.proc_index]
            if op.new_proc_name != old_name:
                self._processing_mgr.rename_processing(old_name, op.new_proc_name)

    def _apply_additions(self, additions) -> None:
        """Apply addition operations."""
        if not additions:
            return

        addition_arrays = []
        for op in additions:
            prepared = self._storage._prepare_for_storage(op.new_data)
            addition_arrays.append(prepared[:, None, :])
            self._processing_mgr.add_processing(op.new_proc_name)

        # Concatenate and add to storage
        new_data_3d = np.concatenate(addition_arrays, axis=1)
        self._storage.add_processings(new_data_3d)

    # ========== String representations ==========

    def __repr__(self) -> str:
        return (
            f"FeatureSource(shape={self._storage.shape}, dtype={self._storage.dtype}, "
            f"processing_ids={self.processing_ids})"
        )

    def __str__(self) -> str:
        arr = self._storage.array
        if arr.size > 0:
            mean_val = round(float(np.mean(arr)), 3)
            var_val = round(float(np.var(arr)), 3)
            min_val = round(float(np.min(arr)), 3)
            max_val = round(float(np.max(arr)), 3)
        else:
            mean_val = var_val = min_val = max_val = 0.0

        return (
            f"{self._storage.shape}, processings={self.processing_ids}, "
            f"min={min_val}, max={max_val}, mean={mean_val}, var={var_val}"
        )
```

---

## 4. Source Index Inconsistencies

### Current Issues

```python
# In features.py
def update_features(self, ..., source: int = -1) -> None:
    self.sources[source if source >= 0 else 0].update_features(...)

# Inconsistency: -1 defaults to 0, but this is confusing
# Better: Use Optional[int] = None with explicit default
```

### Proposed Fix

```python
class Features:
    def update_features(
        self,
        source_processings: ProcessingList,
        features: InputFeatures,
        processings: ProcessingList,
        source: Optional[int] = None  # Changed from source: int = -1
    ) -> None:
---

## 5. Updating All Imports Across Codebase

### Strategy for Zero Regression

All files that currently use layout strings or header unit strings must be updated to import from the new location while maintaining backward compatibility.

### Files Requiring Updates

Based on grep search, the following files need import updates:

1. **`nirs4all/data/features.py`**
   ```python
   # Add at top
   from nirs4all.data.feature_components import FeatureSource, FeatureLayout
   ```

2. **`nirs4all/data/dataset.py`**
   ```python
   # Update import
   from nirs4all.data.helpers import Layout  # Keep for type hints
   from nirs4all.data.feature_components import FeatureLayout  # Add for validation
   ```

3. **`nirs4all/data/dataset_components/feature_accessor.py`**
   ```python
   # Add imports
   from nirs4all.data.feature_components import FeatureLayout, HeaderUnit
   ```

4. **`nirs4all/data/loaders/loader.py`**
   ```python
   # Add imports
   from nirs4all.data.feature_components import HeaderUnit
   ```

5. **`nirs4all/utils/model_builder.py`**
   ```python
   # Add imports
   from nirs4all.data.feature_components import FeatureLayout

   # Update code to use enum while maintaining string BC
   layout = FeatureLayout.VOLUME_3D_TRANSPOSE if framework == 'tensorflow' else FeatureLayout.FLAT_2D
   # This still works because FeatureLayout.__str__() returns the string value
   ```

6. **`nirs4all/pipeline/runner.py`**
   ```python
   # Add imports
   from nirs4all.data.feature_components import FeatureLayout

   # Can optionally update string literals to enums:
   x_train = dataset.x({"partition": "train"}, layout=FeatureLayout.FLAT_2D)
   # Or keep as strings - both work due to normalization
   ```

### Validation Strategy

```python
# File: tests/test_backward_compatibility.py

def test_all_layout_usages():
---

## 7. Multi-Source API Design_builder import ModelBuilder
    # ... test that it still works

    # Test pipeline runner
    from nirs4all.pipeline.runner import PipelineRunner
    # ... test that it still works

    # Test dataset
    from nirs4all.data.dataset import SpectroDataset
    # ... test that it still works
```

### Helpers.py Type Hints

The `Layout` type hint in `helpers.py` should remain unchanged for backward compatibility:

```python
# File: nirs4all/data/helpers.py

# Keep existing Literal for type hints (BC)
Layout = Literal["2d", "3d", "2d_t", "3d_i"]

# Type hints in function signatures continue to work
def x(self, selector: Selector, layout: Layout = "2d") -> OutputData:
    # Internally, normalize to enum if needed
    from nirs4all.data.feature_components import normalize_layout
    layout_enum = normalize_layout(layout)
```

### Import Migration Checklist

- [ ] Create `feature_components/` directory
- [ ] Move and refactor `feature_source.py`
- [ ] Create all component files
- [ ] Update `feature_components/__init__.py` with exports
- [ ] Update `data/__init__.py` with shortcuts
- [ ] Update `features.py` imports
- [ ] Update `dataset.py` imports
- [ ] Update `dataset_components/feature_accessor.py` imports
- [ ] Update `loaders/loader.py` imports
- [ ] Update `utils/model_builder.py` imports
- [ ] Update `pipeline/runner.py` imports (optional, can stay as strings)
- [ ] Run full test suite
- [ ] Check all examples still work

---

## 6. Source Index Inconsistencies
            source_processings: List of existing processing names to replace.
            features: Feature arrays to add or replace.
            processings: Target processing names for the features.
            source: Source index to update. If None, defaults to first source (0).
                   Use 0 explicitly for first source.
        """
        if not features:
            return

---

## 8. Implementation Phases
        # Validate source index
### Phase 1: Directory Structure & Constants (Low Risk)
1. Create `nirs4all/data/feature_components/` directory
2. Create `feature_constants.py` with enums
---

## 9. Testing Strategy__.py` with shortcut imports
6. **No breaking changes**

### Phase 2: Components Implementation (Internal)
1. Implement all 6 component classes in `feature_components/`
2. Keep existing `feature_source.py` untouched for now
3. Run isolated component tests
4. **No user-facing changes**

### Phase 3: FeatureSource Refactoring & Migration
1. Create new `feature_components/feature_source.py` using components
2. Run comprehensive tests comparing old vs new implementation
3. Remove old `data/feature_source.py`
4. Verify import shortcuts work: `from nirs4all.data import FeatureSource`
5. **Still backward compatible**

### Phase 4: Update All Import Sites
1. Update `features.py` to import from `feature_components`
2. Update `dataset.py` to import from `feature_components`
3. Update `dataset_components/feature_accessor.py`
4. Update `loaders/loader.py`
5. Update `utils/model_builder.py`
6. Optionally update `pipeline/runner.py` (can keep strings)
7. **No API changes, just import paths**

### Phase 5: Features Class Enhancements
1. Fix source index inconsistency (use `Optional[int]`)
2. Add explicit `_per_source` properties
3. Update docstrings with new import examples
4. **Minor API enhancement, backward compatible**

### Phase 6: Comprehensive Testing
1. Run full test suite with all examples
2. Test backward compatibility explicitly
3. Verify no regressions in pipeline runs
4. **Validation phase**

### Phase 7: Deprecation Warnings (Optional, Long-term)
1. Add soft deprecation warnings for `source=-1` pattern
2. Suggest enum usage in documentation (no warnings)
3. Remove warnings after 2+ major versions
---

## 10. Documentation Updates
1. **Single-source convenience**: Users don't need to unpack `[42]` to get `42`
2. **Natural ergonomics**: `if dataset.num_features == 100` vs `if dataset.num_features[0] == 100`
3. **Backward compatibility**: Existing single-source code doesn't break

### Alternative Design (For Consideration)

If you want more consistency, add **explicit** single/multi properties:

```python
class Features:
    # Original (auto-unwrap)
    @property
    def num_features(self) -> Union[int, List[int]]:
        """Backward compatible: unwraps for single source."""
        res = [src.num_features for src in self.sources]
        return res[0] if len(res) == 1 else res

    # Explicit multi-source (always list)
    @property
    def num_features_per_source(self) -> List[int]:
        """Always returns list, even for single source."""
        return [src.num_features for src in self.sources]

    # Explicit single-source (raises if multi)
    @property
    def num_features_single(self) -> int:
        """Returns int only if single source."""
        if len(self.sources) != 1:
            raise ValueError(
                f"Expected single source, got {len(self.sources)}. "
                "Use num_features_per_source for multi-source datasets."
            )
        return self.sources[0].num_features
```

### Recommendation

**Keep your current approach** with these additions:
- Document the auto-unwrap behavior clearly in docstrings
- Add explicit `_per_source` variants for users who want consistency
---

## 11. Critical: Import Path Changes

### Benefits of This Refactoring

1. **Modularity**: Clear separation of concerns (storage, processing, headers, layout)
2. **Testability**: Each component can be tested independently
3. **Extensibility**: Easy to add new layouts or processing strategies
4. **Type Safety**: Enums prevent typos while maintaining string compatibility
5. **Maintainability**: Smaller, focused classes are easier to understand
6. **Backward Compatibility**: 100% compatible with existing user code
7. **Organized Structure**: Related code grouped in `feature_components/` module
### What NOT to Do

1. ❌ Don't force users to use enums (accept both strings and enums)
2. ❌ Don't break the multi-source auto-unwrap behavior
3. ❌ Don't add logging yet (as requested)
4. ❌ Don't change serialization format without migration path
5. ❌ Don't update imports without updating ALL affected files
6. ❌ Don't remove old import paths without deprecation warnings

### What TO Do

1. ✅ Move `FeatureSource` to `feature_components/` module
2. ✅ Create all 6 component classes in `feature_components/`
3. ✅ Add enums in `feature_components/feature_constants.py`
4. ✅ Provide shortcut imports via `data/__init__.py`
5. ✅ Update ALL files using layouts/header units to import from new location
6. ✅ Add deprecation stub at old `feature_source.py` location
7. ✅ Run comprehensive tests to ensure zero regressions
8. ✅ Update documentation with new import patterns

### Implementation Checklist

- [ ] **Phase 1**: Create directory structure and constants
- [ ] **Phase 2**: Implement all component classes
- [ ] **Phase 3**: Refactor and move FeatureSource
- [ ] **Phase 4**: Update all import sites (critical for no regressions)
  - [ ] `features.py`
  - [ ] `dataset.py`
  - [ ] `dataset_components/feature_accessor.py`
  - [ ] `loaders/loader.py`
  - [ ] `utils/model_builder.py`
  - [ ] `pipeline/runner.py`
- [ ] **Phase 5**: Enhance Features class API
- [ ] **Phase 6**: Run full test suite + all examples
- [ ] **Phase 7**: Add deprecation warnings (optional)

### Final Recommendation

**Proceed with Phases 1-6** for complete refactoring with zero breaking changes. The critical addition is **Phase 4** to update all import sites across the codebase, ensuring no regressions when layout strings and header units are referenced throughout the project.

The component-based architecture combined with centralized constants will make future enhancements much easier while keeping the public API stable and ensuring type safety across the entire codebase.
```

### Old Import Paths (Will Break)

```python
# BEFORE (will break after migration):
from nirs4all.data.feature_source import FeatureSource  # ❌ File moved

# AFTER (backward compatible via __init__.py):
from nirs4all.data import FeatureSource  # ✅ Works via shortcut
from nirs4all.data.feature_components import FeatureSource  # ✅ Direct path
```

### Migration Strategy for External Users

If external code imports `FeatureSource` directly, we can provide a temporary compatibility shim:

```python
# File: nirs4all/data/feature_source.py (deprecated stub)

"""
Deprecated: FeatureSource has moved to feature_components.

This module provides backward compatibility for old imports.
Will be removed in version 0.5.0.
"""

import warnings
from nirs4all.data.feature_components import FeatureSource

warnings.warn(
    "Importing from nirs4all.data.feature_source is deprecated. "
    "Use: from nirs4all.data import FeatureSource",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["FeatureSource"]
```

This ensures:
- Old `from nirs4all.data.feature_source import FeatureSource` still works
- Users get a clear deprecation warning with migration path
- No immediate breakage for existing codebases

---

## 12. Summary
from typing import TypeGuard

def is_single_source(features: Features) -> TypeGuard[int]:
    """Type guard for single-source feature blocks."""
    return len(features.sources) == 1
```

---

## 6. Implementation Phases

### Phase 1: Constants & Foundation (Low Risk)
1. Create `feature_constants.py` with enums
2. Add backward-compatible normalization functions
3. Update type hints to accept both `str` and `Enum`
4. **No breaking changes**

### Phase 2: Components (Internal Refactoring)
1. Create `feature_components/` module
2. Implement all component classes
3. Keep old `FeatureSource` untouched
4. **No user-facing changes**

### Phase 3: FeatureSource Refactoring
1. Create `feature_source_v2.py` with new implementation
2. Run comprehensive tests comparing v1 vs v2 outputs
3. Switch import when validated
4. **Still backward compatible**

### Phase 4: Features Class Updates
1. Fix source index inconsistency (use `Optional[int]`)
2. Add explicit `_per_source` properties
3. Update docstrings
4. **Minor API enhancement, backward compatible**

### Phase 5: Deprecation (Optional, Long-term)
1. Add deprecation warnings for `-1` as default
2. Suggest enum usage over strings (warnings)
3. Remove warnings after 2+ major versions

---

## 7. Testing Strategy

```python
# tests/test_feature_refactoring.py

def test_backward_compatibility_layout_strings():
    """Ensure old string layouts still work."""
    source = FeatureSource()
    source.add_samples(np.random.rand(10, 100))

    # Old way (strings)
    x_old = source.x(range(5), layout="3d_transpose")

    # New way (enums)
    from nirs4all.data.feature_constants import FeatureLayout
    x_new = source.x(range(5), layout=FeatureLayout.VOLUME_3D_TRANSPOSE)

    np.testing.assert_array_equal(x_old, x_new)


def test_backward_compatibility_header_units():
    """Ensure old string units still work."""
    source = FeatureSource()

    # Old way
    source.set_headers(["800", "850", "900"], unit="nm")
    assert source.header_unit == "nm"

    # New way
    from nirs4all.data.feature_constants import HeaderUnit
    source.set_headers(["800", "850", "900"], unit=HeaderUnit.WAVELENGTH)
    assert source.header_unit == "nm"  # Still returns string


def test_refactored_feature_source_parity():
    """Ensure refactored version produces identical results."""
    # Create identical scenarios with old and new implementations
    # Compare all outputs
    pass
```

---

## 8. Documentation Updates

### Migration Guide

````markdown
# Migration Guide: Features Refactoring

## Enums vs Strings (Recommended, not required)

```python
# Before (still works)
X = dataset.x({"partition": "train"}, layout="3d_transpose")

# After (recommended for type safety)
from nirs4all.data.feature_constants import FeatureLayout
X = dataset.x({"partition": "train"}, layout=FeatureLayout.VOLUME_3D_TRANSPOSE)
```

## Source Index (Breaking in v1.0.0)

```python
# Before (deprecated)
dataset.update_features([], [data], ["msc"], source=-1)

# After (recommended)
dataset.update_features([], [data], ["msc"], source=None)
# or
dataset.update_features([], [data], ["msc"])  # None is default
```

## Multi-Source Properties (New)

```python
# Auto-unwrap (backward compatible)
n_features = dataset.num_features  # int if single source, List[int] if multi

# Explicit list (always consistent)
n_features_list = dataset.num_features_per_source  # Always List[int]
```
````

---

## Summary

### Benefits of This Refactoring

1. **Modularity**: Clear separation of concerns (storage, processing, headers, layout)
2. **Testability**: Each component can be tested independently
3. **Extensibility**: Easy to add new layouts or processing strategies
4. **Type Safety**: Enums prevent typos while maintaining string compatibility
5. **Maintainability**: Smaller, focused classes are easier to understand
6. **Backward Compatibility**: 100% compatible with existing user code

### What NOT to Do

1. ❌ Don't force users to use enums (accept both)
2. ❌ Don't break the multi-source auto-unwrap behavior
3. ❌ Don't add logging yet (as you mentioned)
4. ❌ Don't change serialization format without migration path

### Final Recommendation

**Proceed with Phases 1-3** for maximum impact with zero breaking changes. The component-based architecture will make future enhancements much easier while keeping the public API stable.
