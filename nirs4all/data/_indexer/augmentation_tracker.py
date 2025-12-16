"""
Track origin/augmented sample relationships.

This module provides the AugmentationTracker class for managing the
relationships between base samples and their augmented versions.
"""

from typing import List, Optional
import numpy as np
import polars as pl


class AugmentationTracker:
    """
    Manage origin and augmented sample relationships.

    This class encapsulates the critical two-phase selection pattern that
    prevents data leakage across cross-validation folds when working with
    augmented samples.

    Key concepts:
    - Base samples: origin == sample (self-referencing)
    - Augmented samples: origin != sample (references base sample)
    - Two-phase selection: Select base samples first, then their augmentations

    The two-phase approach ensures that if a base sample is in the test set,
    all its augmentations also stay in test (preventing leakage).
    """

    def __init__(self, store, query_builder):
        """
        Initialize the augmentation tracker.

        Args:
            store: IndexStore instance for querying.
            query_builder: QueryBuilder instance for building filters.
        """
        self._store = store
        self._query_builder = query_builder

    def get_augmented_for_origins(self, origin_ids: List[int], additional_filter: Optional[pl.Expr] = None) -> np.ndarray:
        """
        Get all augmented samples for given origin sample IDs.

        This method retrieves augmented versions of base samples, enabling
        two-phase selection that prevents data leakage.

        Args:
            origin_ids: List of origin sample IDs to find augmented versions for.
            additional_filter: Optional additional filter to apply (e.g., exclusion filter).

        Returns:
            np.ndarray: Array of augmented sample IDs (dtype: np.int32). Only
                       includes samples where origin is in origin_ids AND
                       sample != origin (actual augmented samples).

        Examples:
            >>> # Get base samples
            >>> base_samples = [0, 1, 2]
            >>> # Get their augmented versions
            >>> augmented = tracker.get_augmented_for_origins(base_samples)
            >>> # augmented might be [100, 101, 102, 103] (if samples 0,1,2 have augmentations)
            >>>
            >>> # Combine for full dataset
            >>> all_samples = np.concatenate([np.array(base_samples), augmented])

        Note:
            This method does not filter by partition, group, or other criteria.
            It returns ALL augmented samples for the given origins, regardless
            of their other attributes.
        """
        if not origin_ids:
            return np.array([], dtype=np.int32)

        # Build filter: origin in list AND sample != origin (augmented only)
        origin_filter = self._query_builder.build_origin_filter(origin_ids)
        augmented_filter = self._query_builder.build_augmented_samples_filter()
        condition = origin_filter & augmented_filter

        # Apply additional filter if provided (e.g., exclusion filter)
        if additional_filter is not None:
            condition = condition & additional_filter

        # Query and return sample IDs
        augmented_df = self._store.query(condition)
        return augmented_df.select(pl.col("sample")).to_series().to_numpy().astype(np.int32)

    def get_origin_for_sample(self, sample_id: int) -> Optional[int]:
        """
        Get origin sample ID for a given sample.

        All samples have an origin:
        - Base samples: origin == sample (self-referencing)
        - Augmented samples: origin != sample (references base sample)

        Args:
            sample_id: Sample ID to look up.

        Returns:
            Optional[int]: Origin sample ID, or None if sample not found.

        Examples:
            >>> # For augmented sample
            >>> tracker.get_origin_for_sample(100)  # Returns 10 (its origin)
            >>>
            >>> # For base sample
            >>> tracker.get_origin_for_sample(10)   # Returns 10 (itself)
            >>>
            >>> # For non-existent sample
            >>> tracker.get_origin_for_sample(999)  # Returns None

        Note:
            This is a single-sample lookup. For batch operations, use other
            methods which are more efficient.
        """
        condition = pl.col("sample") == sample_id
        row = self._store.query(condition)

        if len(row) == 0:
            return None

        return int(row.select(pl.col("origin")).item())

    def is_augmented(self, sample_id: int) -> bool:
        """
        Check if a sample is augmented (origin != sample).

        Args:
            sample_id: Sample ID to check.

        Returns:
            bool: True if sample is augmented, False if base or not found.

        Example:
            >>> tracker.is_augmented(100)  # True (augmented)
            >>> tracker.is_augmented(10)   # False (base sample)
        """
        condition = pl.col("sample") == sample_id
        row = self._store.query(condition)

        if len(row) == 0:
            return False

        sample_val = row.select(pl.col("sample")).item()
        origin_val = row.select(pl.col("origin")).item()
        return sample_val != origin_val

    def get_base_samples(self, condition: pl.Expr) -> np.ndarray:
        """
        Get base sample IDs matching a condition.

        Args:
            condition: Polars expression for filtering.

        Returns:
            np.ndarray: Array of base sample IDs (where sample == origin).

        Example:
            >>> # Get base train samples
            >>> train_condition = pl.col("partition") == "train"
            >>> base_train = tracker.get_base_samples(train_condition)
        """
        base_filter = self._query_builder.build_base_samples_filter()
        combined = condition & base_filter

        filtered_df = self._store.query(combined)
        return filtered_df.select(pl.col("sample")).to_series().to_numpy().astype(np.int32)

    def get_all_samples_with_augmentations(self, base_condition: pl.Expr, additional_filter: Optional[pl.Expr] = None) -> np.ndarray:
        """
        Get base samples and their augmentations in two phases (leak prevention).

        This method implements the critical two-phase selection:
        1. Phase 1: Select base samples matching the condition
        2. Phase 2: Get all augmented versions of those base samples

        Args:
            base_condition: Polars expression for selecting base samples.
            additional_filter: Optional additional filter to apply to augmented samples
                             (e.g., exclusion filter). This filter is applied in both
                             phases to ensure excluded augmented samples are filtered.

        Returns:
            np.ndarray: Combined array of base sample IDs and their augmented versions.

        Example:
            >>> # Get all train samples (base + augmented)
            >>> train_condition = pl.col("partition") == "train"
            >>> all_train = tracker.get_all_samples_with_augmentations(train_condition)

        Note:
            This ensures that augmented samples from other partitions/groups are NOT
            included, preventing data leakage in cross-validation scenarios.
        """
        # Phase 1: Get base samples
        base_samples = self.get_base_samples(base_condition)

        if len(base_samples) == 0:
            return base_samples

        # Phase 2: Get augmented versions (apply additional_filter to exclude marked samples)
        augmented = self.get_augmented_for_origins(base_samples.tolist(), additional_filter=additional_filter)

        if len(augmented) == 0:
            return base_samples

        # Combine and return
        return np.concatenate([base_samples, augmented])
