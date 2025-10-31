"""Handles sample augmentation operations."""

import numpy as np
from typing import List


class AugmentationHandler:
    """Manages sample augmentation for feature sources.

    Handles the logic for creating augmented samples from existing ones,
    including data validation and processing management.
    """

    @staticmethod
    def validate_augmentation_inputs(
        sample_indices: List[int],
        data: np.ndarray,
        count_list: List[int],
        num_samples: int
    ) -> int:
        """Validate augmentation inputs and return total augmentations.

        Args:
            sample_indices: List of sample indices to augment.
            data: Augmented feature data.
            count_list: Number of augmentations per sample.
            num_samples: Current number of samples in the dataset.

        Returns:
            Total number of augmentations.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not sample_indices:
            return 0

        total_augmentations = sum(count_list)
        if total_augmentations == 0:
            return 0

        # Validate data shape
        if data.ndim != 2:
            raise ValueError(f"data must be a 2D array, got {data.ndim} dimensions")

        if data.shape[0] != total_augmentations:
            raise ValueError(
                f"data must have {total_augmentations} samples, got {data.shape[0]}"
            )

        # Validate sample indices
        for idx in sample_indices:
            if idx < 0 or idx >= num_samples:
                raise ValueError(
                    f"Sample index {idx} out of range [0, {num_samples})"
                )

        # Validate count_list
        if len(count_list) != len(sample_indices):
            raise ValueError(
                "count_list must have the same length as sample_indices"
            )

        return total_augmentations

    @staticmethod
    def normalize_processings(processings: any) -> List[str]:
        """Normalize processings to list of strings.

        Args:
            processings: Processing name(s), either string or list of strings.

        Returns:
            List of processing names.
        """
        if isinstance(processings, str):
            return [processings]
        return list(processings)

    def __repr__(self) -> str:
        return "AugmentationHandler()"
