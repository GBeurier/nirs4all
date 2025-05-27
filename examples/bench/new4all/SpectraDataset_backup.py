import numpy as np
import polars as pl
from typing import Any, Sequence, Union, List, Optional, Dict
from .SpectraFeatures import SpectraFeatures
from .DatasetView import DatasetView
from .TargetManager import TargetManager


class SpectraDataset:
    """Main dataset class with efficient operations and clear interface."""
    
    def __init__(self, float64: bool = True, task_type: str = "auto"):
        self.float64 = float64

        # Core data
        self.features: Optional[SpectraFeatures] = None
        self.indices = pl.DataFrame({
            "row": pl.Series([], dtype=pl.Int64),
            "sample": pl.Series([], dtype=pl.Int64),
            "origin": pl.Series([], dtype=pl.Int64),
            "partition": pl.Series([], dtype=pl.Utf8),
            "group": pl.Series([], dtype=pl.Int64),
            "branch": pl.Series([], dtype=pl.Int64),
            "processing": pl.Series([], dtype=pl.Utf8),
        })

        # Target management
        self.target_manager = TargetManager(task_type=task_type)

        # Counters
        self._next_row = 0
        self._next_sample = 0

    def __len__(self) -> int:
        return len(self.indices)

    def add_data(self,
                 features: Union[np.ndarray, List[np.ndarray]],
                 targets: Optional[np.ndarray] = None,
                 partition: str = "train",
                 group: int = 0,
                 branch: int = 0,
                 processing: str = "raw",
                 origin: Optional[Union[int, List[int]]] = None) -> List[int]:
        """Add new data and return sample IDs."""

        if isinstance(features, np.ndarray):
            features = [features]

        n_samples = len(features[0])

        # Generate sample IDs
        sample_ids = list(range(self._next_sample, self._next_sample + n_samples))
        row_ids = list(range(self._next_row, self._next_row + n_samples))

        # Add features
        if self.features is None:
            self.features = SpectraFeatures(features)
        else:
            self.features.append(features)

        # Add indices
        if origin is None:
            origin = sample_ids
        elif isinstance(origin, int):
            origin = [origin] * n_samples

        new_indices = pl.DataFrame({
            "row": row_ids,
            "sample": sample_ids,
            "origin": origin,
            "partition": [partition] * n_samples,
            "group": [group] * n_samples,
            "branch": [branch] * n_samples,
            "processing": [processing] * n_samples,
        })

        self.indices = pl.concat([self.indices, new_indices])

        # Add targets if provided
        if targets is not None:
            self.target_manager.add_targets(sample_ids, targets)

        # Update counters
        self._next_row += n_samples
        self._next_sample += n_samples

        return sample_ids

    def select(self, **filters) -> 'DatasetView':
        """Create an efficient view of the dataset with filters applied."""
        return DatasetView(self, filters)

    def get_features(self, row_indices: np.ndarray,
                    source_indices: Optional[Union[int, List[int]]] = None,
                    concatenate: bool = True) -> Union[np.ndarray, List[np.ndarray]]:        """Direct feature access by row indices."""
        if self.features is None:
            return np.array([])
        return self.features.get_by_rows(row_indices, source_indices, concatenate)

    def get_targets(self, sample_ids: List[int], 
                   representation: str = "auto",
                   transformer_key: Optional[str] = None) -> np.ndarray:
        """Get targets for specific samples using TargetManager."""
        return self.target_manager.get_targets(sample_ids, representation, transformer_key)

    def update_features(self, row_indices: np.ndarray,
                       new_features: Union[np.ndarray, List[np.ndarray]],
                       source_indices: Optional[Union[int, List[int]]] = None):
        """Update features in-place."""
        if self.features is not None:
            self.features.update_rows(row_indices, new_features, source_indices)

    def update_processing(self, sample_ids: List[int], processing_tag: str):
        """Update processing tags for samples."""
        mask = pl.col("sample").is_in(sample_ids)
        self.indices = self.indices.with_columns(
            pl.when(mask).then(pl.lit(processing_tag)).otherwise(pl.col("processing")).alias("processing")
        )

    # Target management methods
    def fit_transform_targets(self, sample_ids: List[int], 
                            transformers: List[Any],
                            representation: str = "auto",
                            transformer_key: str = "default") -> np.ndarray:
        """Fit target transformers and return transformed targets."""
        return self.target_manager.fit_transform_targets(
            sample_ids, transformers, representation, transformer_key)
    
    def inverse_transform_predictions(self, predictions: np.ndarray,
                                    representation: str = "auto",
                                    transformer_key: str = "default",
                                    to_original: bool = True) -> np.ndarray:
        """Inverse transform predictions back to original format."""
        return self.target_manager.inverse_transform_predictions(
            predictions, representation, transformer_key, to_original)
    
    def get_target_info(self) -> Dict[str, Any]:
        """Get information about targets."""
        return self.target_manager.get_info()
    
    @property
    def task_type(self) -> str:
        """Get the task type."""
        return self.target_manager.task_type
    
    @property 
    def n_classes(self) -> int:
        """Get number of classes for classification tasks."""
        return self.target_manager.n_classes_
    
    @property
    def classes_(self) -> Optional[np.ndarray]:
        """Get class labels for classification tasks."""
        return self.target_manager.classes_
