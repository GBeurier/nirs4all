"""
Target management for SpectroDataset.

This module contains TargetBlock for managing target values with zero-copy,
type-aware architecture. Supports regression, classification, and multilabel
targets with lazy encoding and efficient filtering.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import warnings


class TargetType(Enum):
    """Target type enumeration."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    MULTILABEL = "multilabel"


class TargetSource(ABC):
    """Abstract base class for target sources with zero-copy capability."""

    def __init__(self, name: str, target_type: TargetType, samples: np.ndarray,
                 processing: str = "raw"):
        """
        Initialize target source.

        Args:
            name: Name of the target source
            target_type: Type of target (regression, classification, multilabel)
            samples: Sample IDs as numpy array
            processing: Processing version identifier
        """
        self.name = name
        self.target_type = target_type
        self.samples = samples
        self.processing = processing

        # Lazy encoding attributes
        self._label_encoder = None
        self._classes = None
        self._encoded_data = None

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get shape of target data."""
        return self._get_shape()

    @property
    def classes(self) -> Optional[np.ndarray]:
        """Get unique classes for classification targets."""
        if self.target_type == TargetType.CLASSIFICATION and self._classes is None:
            self._classes = np.unique(self.get_raw_data())
        return self._classes

    @abstractmethod
    def _get_shape(self) -> Tuple[int, ...]:
        """Get shape of target data."""
        pass

    @abstractmethod
    def get_raw_data(self) -> np.ndarray:
        """Get raw target data."""
        pass

    @abstractmethod
    def get_subset(self, indices: np.ndarray) -> 'TargetSource':
        """Get subset of target source (zero-copy if possible)."""
        pass

    def get_encoded_data(self) -> np.ndarray:
        """Get encoded target data (lazy encoding for classification)."""
        if self.target_type == TargetType.REGRESSION:
            return self.get_raw_data()
        elif self.target_type == TargetType.CLASSIFICATION:
            return self._get_encoded_classification()
        elif self.target_type == TargetType.MULTILABEL:
            return self._get_encoded_multilabel()
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")

    def _get_encoded_classification(self) -> np.ndarray:
        """Get encoded classification data."""
        if self._encoded_data is None:
            raw_data = self.get_raw_data()
            if raw_data.dtype.kind in 'ui':  # Integer types
                self._encoded_data = raw_data
            else:  # String types
                from sklearn.preprocessing import LabelEncoder
                self._label_encoder = LabelEncoder()
                self._encoded_data = self._label_encoder.fit_transform(raw_data)
        return np.asarray(self._encoded_data)

    def _get_encoded_multilabel(self) -> np.ndarray:
        """Get encoded multilabel data (one-hot encoding)."""
        if self._encoded_data is None:
            raw_data = self.get_raw_data()
            if raw_data.ndim == 1:
                # Single label per sample - convert to one-hot
                from sklearn.preprocessing import MultiLabelBinarizer
                mlb = MultiLabelBinarizer()                # Assume comma-separated labels for string data
                if raw_data.dtype.kind in 'us':  # String types
                    labels = [str(label).split(',') for label in raw_data]
                else:
                    labels = [[label] for label in raw_data]
                encoded_matrix = mlb.fit_transform(labels)                # Convert sparse matrix to dense if needed
                if hasattr(encoded_matrix, 'toarray'):
                    encoded_matrix = encoded_matrix.toarray()
                self._encoded_data = encoded_matrix.astype(np.int32)
            else:
                # Already binary encoded
                self._encoded_data = raw_data.astype(np.int32)
        return np.asarray(self._encoded_data, dtype=np.int32)

    def get_one_hot_encoded(self) -> np.ndarray:
        """Get one-hot encoded data for classification targets."""
        if self.target_type != TargetType.CLASSIFICATION:
            raise ValueError("One-hot encoding only available for classification targets")

        from sklearn.preprocessing import OneHotEncoder
        encoded_data = self.get_encoded_data().reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False)
        return encoder.fit_transform(encoded_data)

    def __repr__(self):
        return f"TargetSource(name='{self.name}', type={self.target_type.value}, shape={self.shape})"


class RegressionTargetSource(TargetSource):
    """Target source for regression targets."""

    def __init__(self, name: str, data: np.ndarray, samples: np.ndarray,
                 processing: str = "raw"):
        """
        Initialize regression target source.

        Args:
            name: Name of the target source
            data: Target data as numpy array
            samples: Sample IDs as numpy array
            processing: Processing version identifier
        """
        super().__init__(name, TargetType.REGRESSION, samples, processing)
        self.data = data.astype(np.float32)

    def _get_shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def get_raw_data(self) -> np.ndarray:
        return self.data

    def get_subset(self, indices: np.ndarray) -> 'RegressionTargetSource':
        """Get subset with zero-copy view if indices are contiguous."""
        if len(indices) > 0 and np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            # Contiguous slice - use view (zero-copy)
            start, stop = indices[0], indices[-1] + 1
            subset_data = self.data[start:stop]
            subset_samples = self.samples[start:stop]
        else:
            # Non-contiguous - copy required
            subset_data = self.data[indices]
            subset_samples = self.samples[indices]

        return RegressionTargetSource(
            name=self.name,
            data=subset_data,
            samples=subset_samples,
            processing=self.processing
        )


class ClassificationTargetSource(TargetSource):
    """Target source for classification targets."""

    def __init__(self, name: str, data: np.ndarray, samples: np.ndarray,
                 processing: str = "raw"):
        """
        Initialize classification target source.

        Args:
            name: Name of the target source
            data: Target data as numpy array (strings or integers)
            samples: Sample IDs as numpy array
            processing: Processing version identifier
        """
        super().__init__(name, TargetType.CLASSIFICATION, samples, processing)
        self.data = data

    def _get_shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def get_raw_data(self) -> np.ndarray:
        return self.data

    def get_subset(self, indices: np.ndarray) -> 'ClassificationTargetSource':
        """Get subset with zero-copy view if indices are contiguous."""
        if len(indices) > 0 and np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            # Contiguous slice - use view (zero-copy)
            start, stop = indices[0], indices[-1] + 1
            subset_data = self.data[start:stop]
            subset_samples = self.samples[start:stop]
        else:
            # Non-contiguous - copy required
            subset_data = self.data[indices]
            subset_samples = self.samples[indices]

        return ClassificationTargetSource(
            name=self.name,
            data=subset_data,
            samples=subset_samples,
            processing=self.processing
        )


class MultilabelTargetSource(TargetSource):
    """Target source for multilabel targets."""

    def __init__(self, name: str, data: np.ndarray, samples: np.ndarray,
                 processing: str = "raw"):
        """
        Initialize multilabel target source.

        Args:
            name: Name of the target source
            data: Target data as numpy array (binary matrix or label strings)
            samples: Sample IDs as numpy array
            processing: Processing version identifier
        """
        super().__init__(name, TargetType.MULTILABEL, samples, processing)
        self.data = data

    def _get_shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def get_raw_data(self) -> np.ndarray:
        return self.data

    def get_subset(self, indices: np.ndarray) -> 'MultilabelTargetSource':
        """Get subset with zero-copy view if indices are contiguous."""
        if len(indices) > 0 and np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            # Contiguous slice - use view (zero-copy)
            start, stop = indices[0], indices[-1] + 1
            subset_data = self.data[start:stop]
            subset_samples = self.samples[start:stop]
        else:
            # Non-contiguous - copy required
            subset_data = self.data[indices]
            subset_samples = self.samples[indices]

        return MultilabelTargetSource(
            name=self.name,
            data=subset_data,
            samples=subset_samples,
            processing=self.processing
        )


class TargetBlock:
    """
    Zero-copy, type-aware target management.

    Manages multiple target sources with different types and processing versions.
    Supports efficient filtering and zero-copy operations where possible.
    """

    def __init__(self):
        """Initialize empty target block."""
        self.sources: Dict[str, TargetSource] = {}
        self.sample_mapping: Dict[int, List[str]] = {}  # sample_id -> [source_names]

    def add_regression_targets(self, name: str, data: np.ndarray, samples: np.ndarray,
                              processing: str = "raw") -> None:
        """Add regression targets."""
        source = RegressionTargetSource(name, data, samples, processing)
        self._add_source(source)

    def add_classification_targets(self, name: str, data: np.ndarray, samples: np.ndarray,
                                  processing: str = "raw") -> None:
        """Add classification targets."""
        source = ClassificationTargetSource(name, data, samples, processing)
        self._add_source(source)

    def add_multilabel_targets(self, name: str, data: np.ndarray, samples: np.ndarray,
                              processing: str = "raw") -> None:
        """Add multilabel targets."""
        source = MultilabelTargetSource(name, data, samples, processing)
        self._add_source(source)

    def _add_source(self, source: TargetSource) -> None:
        """Add a target source to the block."""
        key = f"{source.name}_{source.processing}"
        self.sources[key] = source

        # Update sample mapping
        for sample_id in source.samples:
            if sample_id not in self.sample_mapping:
                self.sample_mapping[sample_id] = []
            if key not in self.sample_mapping[sample_id]:
                self.sample_mapping[sample_id].append(key)

    def get_target_names(self) -> List[str]:
        """Get list of available target names."""
        names = set()
        for key in self.sources.keys():
            name = key.rsplit('_', 1)[0]  # Remove processing suffix
            names.add(name)
        return sorted(names)

    def get_processing_versions(self, target_name: str) -> List[str]:
        """Get available processing versions for a target."""
        versions = []
        for key in self.sources.keys():
            if key.startswith(f"{target_name}_"):
                version = key[len(target_name) + 1:]
                versions.append(version)
        return sorted(versions)

    def y(self, filter_dict: Dict[str, Any],
          target_name: Optional[str] = None,
          processing: str = "raw",
          encoded: bool = True) -> np.ndarray:
        """
        Get target arrays with filtering.

        Args:
            filter_dict: Dictionary of column: value pairs for filtering
            target_name: Name of target to retrieve (if None, uses first available)
            processing: Processing version to use (can be overridden by filter_dict['processing'])
            encoded: Whether to return encoded data (for classification/multilabel)

        Returns:
            Filtered target array
        """
        if not self.sources:
            raise ValueError("No targets available")

        # Extract processing from filter_dict if provided
        if 'processing' in filter_dict:
            processing = filter_dict['processing']        # Determine target name if not specified
        if target_name is None:
            target_name = self.get_target_names()[0]
        # Check if target name exists
        available_names = self.get_target_names()
        if target_name not in available_names:
            raise ValueError(f"Target '{target_name}' not found. Available targets: {available_names}")

        # Get the target source
        key = f"{target_name}_{processing}"
        if key not in self.sources:
            # Check if target exists with different processing
            available_processing = self.get_processing_versions(target_name)
            if available_processing:
                # Target exists but processing version doesn't - warn and return empty array
                msg = (f"Target '{target_name}' with processing '{processing}' not found. "
                       f"Available processing versions: {available_processing}. Returning empty array.")
                warnings.warn(msg, UserWarning, stacklevel=3)
                return np.array([]).reshape(0, 1).astype(np.float32)
            else:
                raise ValueError(f"Target '{target_name}' not found")

        source = self.sources[key]

        # Apply sample filtering
        if 'sample' in filter_dict:
            sample_filter = filter_dict['sample']
            if isinstance(sample_filter, (list, tuple, np.ndarray)):
                # Find indices of samples that match
                mask = np.isin(source.samples, sample_filter)
                indices = np.where(mask)[0]
            else:
                # Single sample
                indices = np.where(source.samples == sample_filter)[0]

            if len(indices) == 0:
                # Return empty array with correct shape
                if source.target_type == TargetType.REGRESSION:
                    return np.array([], dtype=np.float32).reshape(0, source.shape[1] if len(source.shape) > 1 else 1)
                else:
                    return np.array([], dtype=np.int32).reshape(0, source.shape[1] if len(source.shape) > 1 else 1)

            # Get subset (zero-copy if contiguous)
            filtered_source = source.get_subset(indices)
        else:
            filtered_source = source

        # Apply additional filtering (processing, etc.)
        # For now, we assume filtering by processing is handled by selecting the right source

        # Return data
        if encoded:
            return filtered_source.get_encoded_data()
        else:
            return filtered_source.get_raw_data()

    def get_target_info(self, target_name: str, processing: str = "raw") -> Dict[str, Any]:
        """Get information about a target."""
        key = f"{target_name}_{processing}"
        if key not in self.sources:
            raise ValueError(f"Target '{target_name}' with processing '{processing}' not found")

        source = self.sources[key]
        info = {
            'name': source.name,
            'type': source.target_type.value,
            'shape': source.shape,
            'processing': source.processing,
            'sample_count': len(source.samples)
        }

        if source.target_type == TargetType.CLASSIFICATION:
            info['classes'] = source.classes
            info['num_classes'] = len(source.classes) if source.classes is not None else 0

        return info

    def __repr__(self):
        if not self.sources:
            return "TargetBlock(empty)"

        target_names = self.get_target_names()
        return f"TargetBlock(targets={target_names}, sources={len(self.sources)})"
