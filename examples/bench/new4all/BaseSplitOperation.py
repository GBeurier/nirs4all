"""
Base class for split operations with inheritance hierarchy
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np
import polars as pl
from PipelineOperation import PipelineOperation
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext


class BaseSplitOperation(PipelineOperation, ABC):
    """Abstract base class for all splitting operations."""

    def __init__(self, random_state: int = 42, **split_params):
        """
        Initialize base split operation

        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        **split_params : dict
            Additional parameters for splitting strategy
        """
        super().__init__()
        self.random_state = random_state
        self.split_params = split_params

    def can_execute(self, dataset: SpectraDataset, context: PipelineContext) -> bool:
        """Check if split can be executed - base implementation."""
        return len(dataset) > 1

    @abstractmethod
    def execute(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute split operation - must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get operation name - must be implemented by subclasses."""
        pass

    def apply_splits_to_dataset(self, dataset: SpectraDataset, split_indices: Dict[str, np.ndarray]) -> None:
        """Apply splits to dataset by updating partition labels in indices."""

        if dataset.indices is None or len(dataset.indices) == 0:
            raise ValueError("Cannot split empty dataset")

        for partition_name, sample_indices in split_indices.items():
            # Update partition labels for the specified samples
            mask = dataset.indices["row"].is_in(sample_indices)
            dataset.indices = dataset.indices.with_columns([
                pl.when(mask)
                .then(pl.lit(partition_name))
                .otherwise(pl.col("partition"))
                .alias("partition")
            ])

    def get_n_samples(self, dataset: SpectraDataset) -> int:
        """Get number of samples in dataset."""
        return len(dataset)
