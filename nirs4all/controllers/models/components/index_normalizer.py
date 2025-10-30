"""
Index Normalizer - Normalize and validate sample indices

This component handles conversion of sample indices to consistent format
and validates them. Extracted from launch_training() lines 448-454.
"""

from typing import List, Optional, Union
import numpy as np


class IndexNormalizer:
    """Normalizes sample indices to consistent format.

    Converts numpy int types to Python int and validates indices
    are within valid ranges.

    Example:
        >>> normalizer = IndexNormalizer()
        >>> indices = normalizer.normalize([np.int64(0), np.int64(1), np.int64(2)])
        >>> indices
        [0, 1, 2]
    """

    def normalize(
        self,
        indices: Optional[Union[List, np.ndarray]],
        n_samples: int,
        default_range: bool = True,
        validate: bool = False
    ) -> List[int]:
        """Normalize indices to Python int list.

        Args:
            indices: Input indices (may be None, list, or numpy array)
            n_samples: Total number of samples (for validation and defaults)
            default_range: If True and indices is None, return range(n_samples)
            validate: If True, validate indices are within bounds

        Returns:
            List of Python integers

        Raises:
            ValueError: If validate=True and indices are out of bounds
        """
        # Handle None case
        if indices is None:
            if default_range:
                return list(range(n_samples))
            return []

        # Convert to list if numpy array
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()

        # Convert numpy int types to Python int
        normalized = [int(idx) for idx in indices]

        # Validate indices if requested
        if validate:
            self._validate_indices(normalized, n_samples)

        return normalized

    def _validate_indices(self, indices: List[int], n_samples: int) -> None:
        """Validate that indices are within valid range.

        Args:
            indices: List of indices to validate
            n_samples: Total number of samples

        Raises:
            ValueError: If any index is out of bounds
        """
        if not indices:
            return

        min_idx = min(indices)
        max_idx = max(indices)

        if min_idx < 0:
            raise ValueError(f"Negative index found: {min_idx}")

        if max_idx >= n_samples:
            raise ValueError(
                f"Index {max_idx} out of bounds for {n_samples} samples"
            )

    def normalize_batch(
        self,
        indices_dict: dict,
        n_samples_dict: dict
    ) -> dict:
        """Normalize a dictionary of indices for multiple partitions.

        Args:
            indices_dict: Dictionary with keys like 'train', 'val', 'test'
                         and values as index lists/arrays
            n_samples_dict: Dictionary with same keys and values as sample counts

        Returns:
            Dictionary with same keys but normalized indices
        """
        result = {}
        for key, indices in indices_dict.items():
            n_samples = n_samples_dict.get(key, 0)
            result[key] = self.normalize(indices, n_samples)
        return result
