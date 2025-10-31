"""Strategy pattern for feature update operations."""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class ReplacementOperation:
    """Represents a processing replacement operation."""
    proc_idx: int
    new_data: np.ndarray
    new_proc_name: str


@dataclass
class AdditionOperation:
    """Represents a processing addition operation."""
    new_data: np.ndarray
    new_proc_name: str


class UpdateStrategy:
    """Categorizes and validates feature update operations.

    Separates update requests into replacements (updating existing processings)
    and additions (adding new processings).
    """

    @staticmethod
    def categorize_operations(
        features: List[np.ndarray],
        source_processings: List[str],
        processings: List[str],
        processing_id_to_index: dict
    ) -> Tuple[List[ReplacementOperation], List[AdditionOperation]]:
        """Separate operations into replacements and additions.

        Args:
            features: List of feature arrays to add or replace.
            source_processings: List of existing processing names to replace ("" = add new).
            processings: List of target processing names.
            processing_id_to_index: Current mapping of processing names to indices.

        Returns:
            Tuple of (replacements, additions) where:
                - replacements: List of ReplacementOperation objects
                - additions: List of AdditionOperation objects

        Raises:
            ValueError: If source processing doesn't exist, or target processing already exists.
        """
        replacements = []
        additions = []

        # Handle case where source_processings is empty
        if len(source_processings) == 0:
            source_processings = [""] * len(processings)

        for arr, source_proc, target_proc in zip(features, source_processings, processings):
            if source_proc == "":
                # Add new processing
                if target_proc in processing_id_to_index:
                    raise ValueError(
                        f"Processing '{target_proc}' already exists, cannot add"
                    )
                additions.append(AdditionOperation(new_data=arr, new_proc_name=target_proc))
            else:
                # Replace existing processing
                if source_proc not in processing_id_to_index:
                    raise ValueError(
                        f"Source processing '{source_proc}' does not exist"
                    )
                if target_proc != source_proc and target_proc in processing_id_to_index:
                    raise ValueError(
                        f"Target processing '{target_proc}' already exists"
                    )

                source_idx = processing_id_to_index[source_proc]
                replacements.append(
                    ReplacementOperation(
                        proc_idx=source_idx,
                        new_data=arr,
                        new_proc_name=target_proc
                    )
                )

        return replacements, additions

    @staticmethod
    def should_resize_features(
        replacements: List[ReplacementOperation],
        additions: List[AdditionOperation],
        current_num_features: int
    ) -> Tuple[bool, int]:
        """Determine if feature dimension should be resized.

        Returns True if all processings are being replaced with same new dimension.

        Args:
            replacements: List of replacement operations.
            additions: List of addition operations.
            current_num_features: Current feature dimension.

        Returns:
            Tuple of (should_resize, new_num_features).
        """
        if replacements and not additions:
            new_feature_dims = [op.new_data.shape[1] for op in replacements]
            if len(set(new_feature_dims)) == 1 and new_feature_dims[0] != current_num_features:
                return True, new_feature_dims[0]

        return False, current_num_features

    def __repr__(self) -> str:
        return "UpdateStrategy()"
