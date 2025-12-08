"""
Fold-based split operations for cross-validation
"""
import numpy as np
import polars as pl
from typing import Dict, List, Optional, Any, Union
from sklearn.model_selection import (
    StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit,
    RepeatedStratifiedKFold, RepeatedKFold
)

try:
    from BaseSplitOperation import BaseSplitOperation
    from SpectraDataset import SpectraDataset
    from PipelineContext import PipelineContext
except ImportError:
    from BaseSplitOperation import BaseSplitOperation
    from SpectraDataset import SpectraDataset
    from PipelineContext import PipelineContext


class FoldSplitOperation(BaseSplitOperation):
    """Operation for fold-based data splitting (cross-validation)."""

    def __init__(self,
                 fold_strategy: str = "StratifiedKFold",
                 n_splits: int = 5,
                 n_repeats: int = 1,
                 test_size: Optional[float] = None,
                 random_state: int = 42,
                 shuffle: bool = True,
                 **fold_params):
        """
        Initialize fold split operation

        Parameters:
        -----------
        fold_strategy : str
            Folding strategy: "StratifiedKFold", "KFold", "GroupKFold", "TimeSeriesSplit",
            "RepeatedStratifiedKFold", "RepeatedKFold"
        n_splits : int
            Number of folds
        n_repeats : int
            Number of repetitions (for Repeated strategies)
        test_size : float, optional
            If provided, creates train/test split first, then folds on training data
        random_state : int
            Random state for reproducibility
        shuffle : bool
            Whether to shuffle data before splitting
        **fold_params : dict
            Additional parameters for folding strategy
        """
        super().__init__(random_state, **fold_params)
        self.fold_strategy = fold_strategy
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.test_size = test_size
        self.shuffle = shuffle

    def execute(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute fold split operation."""
        if not self.can_execute(dataset, context):
            raise ValueError("Cannot execute split - no data available")

        # Get data for splitting
        n_samples = self.get_n_samples(dataset)

        # Get targets and groups if needed
        all_sample_ids = dataset.indices["sample"].to_list()
        y_for_split = None
        groups_for_split = None

        if hasattr(dataset, 'target_manager') and dataset.target_manager is not None:
            if all_sample_ids:
                y_for_split = dataset.target_manager.get_targets(all_sample_ids)

        if "GroupKFold" in self.fold_strategy:
            groups_for_split = self.split_params.get("groups")
            if groups_for_split is None:
                raise ValueError("GroupKFold requires 'groups' parameter")

        # Handle train/test split if test_size is provided
        if self.test_size:
            train_indices, test_indices = self._create_train_test_split(
                n_samples, y_for_split
            )

            # Apply test split to dataset
            self._apply_train_test_split(dataset, train_indices, test_indices)

            # Generate folds only on training data
            fold_definitions = self.generate_fold_definitions(
                len(train_indices), y_for_split[train_indices] if y_for_split is not None else None,
                groups_for_split[train_indices] if groups_for_split is not None else None,
                base_indices=train_indices
            )
        else:
            # Generate folds on all data
            fold_definitions = self.generate_fold_definitions(
                n_samples, y_for_split, groups_for_split
            )

        # Store fold definitions in dataset
        dataset.add_folds(fold_definitions)

        # Update context with fold information
        context.data_splits = {
            'strategy': self.fold_strategy,
            'n_splits': self.n_splits,
            'n_repeats': self.n_repeats,
            'n_folds': len(fold_definitions),
            'type': 'fold',
            'has_test_holdout': self.test_size is not None
        }

        print(f"Fold split completed using {self.fold_strategy}:")
        print(f"  {len(fold_definitions)} folds created")
        if self.test_size:
            print(f"  Test holdout: {self.test_size * 100:.1f}% of data")

    def get_name(self) -> str:
        """Get operation name."""
        repeats_str = f"_R{self.n_repeats}" if self.n_repeats > 1 else ""
        test_str = f"_TestHO{self.test_size}" if self.test_size else ""
        return f"FoldSplit({self.fold_strategy}_{self.n_splits}{repeats_str}{test_str})"

    def generate_fold_definitions(self,
                                 n_samples: int,
                                 y: Optional[np.ndarray] = None,
                                 groups: Optional[np.ndarray] = None,
                                 base_indices: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Generate fold definitions based on strategy."""

        if base_indices is None:
            base_indices = np.arange(n_samples)

        # Create the appropriate fold splitter
        splitter = self._create_fold_splitter()

        fold_definitions = []
        fold_id = 0

        # Generate splits
        if groups is not None:
            splits = splitter.split(base_indices, y, groups)
        else:
            splits = splitter.split(base_indices, y)

        for train_idx, val_idx in splits:
            # Convert to actual sample indices if using base_indices
            train_indices = base_indices[train_idx]
            val_indices = base_indices[val_idx]

            fold_def = {
                "fold_id": fold_id,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "strategy": self.fold_strategy,
                "n_train": len(train_indices),
                "n_val": len(val_indices)
            }

            fold_definitions.append(fold_def)
            fold_id += 1

        return fold_definitions

    def _create_fold_splitter(self):
        """Create the appropriate sklearn fold splitter."""

        common_params = {
            'n_splits': self.n_splits,
            'random_state': self.random_state
        }

        if self.fold_strategy == "StratifiedKFold":
            if self.shuffle:
                common_params['shuffle'] = True
            return StratifiedKFold(**common_params)

        elif self.fold_strategy == "KFold":
            if self.shuffle:
                common_params['shuffle'] = True
            return KFold(**common_params)

        elif self.fold_strategy == "GroupKFold":
            # GroupKFold doesn't support random_state or shuffle
            return GroupKFold(n_splits=self.n_splits)

        elif self.fold_strategy == "TimeSeriesSplit":
            # TimeSeriesSplit doesn't support random_state or shuffle
            return TimeSeriesSplit(n_splits=self.n_splits)

        elif self.fold_strategy == "RepeatedStratifiedKFold":
            return RepeatedStratifiedKFold(
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                random_state=self.random_state
            )

        elif self.fold_strategy == "RepeatedKFold":
            return RepeatedKFold(
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                random_state=self.random_state
            )

        else:
            raise ValueError(f"Unknown fold strategy: {self.fold_strategy}")

    def _create_train_test_split(self, n_samples: int, y: Optional[np.ndarray] = None):
        """Create initial train/test split if test_size is provided."""
        from sklearn.model_selection import train_test_split

        indices = np.arange(n_samples)

        if y is not None:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=self.test_size,
                stratify=y,
                random_state=self.random_state
            )
        else:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=self.test_size,
                random_state=self.random_state
            )

        return train_idx, test_idx

    def _apply_train_test_split(self, dataset: SpectraDataset, train_indices: np.ndarray, test_indices: np.ndarray):
        """Apply train/test split to dataset."""

        # Create a copy of indices to work with
        updated_indices = dataset.indices.clone()

        # Set train partition
        train_mask = pl.col("row").is_in(train_indices.tolist())
        updated_indices = updated_indices.with_columns([
            pl.when(train_mask).then(pl.lit("train")).otherwise(pl.col("partition")).alias("partition")
        ])

        # Set test partition
        test_mask = pl.col("row").is_in(test_indices.tolist())
        updated_indices = updated_indices.with_columns([
            pl.when(test_mask).then(pl.lit("test")).otherwise(pl.col("partition")).alias("partition")
        ])

        # Replace the dataset indices
        dataset.indices = updated_indices


class FoldStrategy:
    """Strategy factory for common fold scenarios."""

    @classmethod
    def stratified_k_fold(cls, n_splits: int = 5, shuffle: bool = True,
                         random_state: int = 42) -> FoldSplitOperation:
        """Create stratified k-fold split."""
        return FoldSplitOperation(
            fold_strategy="StratifiedKFold",
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

    @classmethod
    def k_fold(cls, n_splits: int = 5, shuffle: bool = True,
              random_state: int = 42) -> FoldSplitOperation:
        """Create k-fold split."""
        return FoldSplitOperation(
            fold_strategy="KFold",
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

    @classmethod
    def repeated_stratified_k_fold(cls, n_splits: int = 5, n_repeats: int = 2,
                                  random_state: int = 42) -> FoldSplitOperation:
        """Create repeated stratified k-fold split."""
        return FoldSplitOperation(
            fold_strategy="RepeatedStratifiedKFold",
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state
        )

    @classmethod
    def group_k_fold(cls, groups: np.ndarray, n_splits: int = 5) -> FoldSplitOperation:
        """Create group k-fold split."""
        return FoldSplitOperation(
            fold_strategy="GroupKFold",
            n_splits=n_splits,
            groups=groups
        )

    @classmethod
    def time_series_split(cls, n_splits: int = 5) -> FoldSplitOperation:
        """Create time series split."""
        return FoldSplitOperation(
            fold_strategy="TimeSeriesSplit",
            n_splits=n_splits
        )

    @classmethod
    def stratified_with_holdout(cls, n_splits: int = 5, test_size: float = 0.2,
                               shuffle: bool = True, random_state: int = 42) -> FoldSplitOperation:
        """Create stratified k-fold with test holdout."""
        return FoldSplitOperation(
            fold_strategy="StratifiedKFold",
            n_splits=n_splits,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state
        )
