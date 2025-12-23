"""Processing ID management and tracking."""

from typing import List, Dict, Optional
from nirs4all.data._features.feature_constants import DEFAULT_PROCESSING


class ProcessingManager:
    """Manages processing IDs and their mapping to array indices.

    Tracks which processing stages exist and their positions in the array's
    second dimension.

    Attributes:
        processing_ids: List of processing names in order.
        processing_id_to_index: Dictionary mapping processing names to indices.
    """

    def __init__(self):
        """Initialize with default 'raw' processing."""
        self._processing_ids: List[str] = [DEFAULT_PROCESSING]
        self._processing_id_to_index: Dict[str, int] = {DEFAULT_PROCESSING: 0}

    @property
    def processing_ids(self) -> List[str]:
        """Get a copy of the processing ID list.

        Returns:
            List of processing identifiers.
        """
        return self._processing_ids.copy()

    @property
    def num_processings(self) -> int:
        """Get the number of processing stages.

        Returns:
            Number of unique processings.
        """
        return len(self._processing_ids)

    def get_index(self, processing_id: str) -> Optional[int]:
        """Get the index for a processing ID.

        Args:
            processing_id: Processing name to lookup.

        Returns:
            Index of the processing, or None if not found.
        """
        return self._processing_id_to_index.get(processing_id)

    def has_processing(self, processing_id: str) -> bool:
        """Check if a processing exists.

        Args:
            processing_id: Processing name to check.

        Returns:
            True if the processing exists.
        """
        return processing_id in self._processing_id_to_index

    def add_processing(self, processing_id: str) -> int:
        """Add a new processing ID.

        Args:
            processing_id: Name of the new processing.

        Returns:
            Index assigned to the new processing.

        Raises:
            ValueError: If processing_id already exists.
        """
        if self.has_processing(processing_id):
            raise ValueError(f"Processing '{processing_id}' already exists")

        new_index = len(self._processing_ids)
        self._processing_ids.append(processing_id)
        self._processing_id_to_index[processing_id] = new_index
        return new_index

    def rename_processing(self, old_id: str, new_id: str) -> None:
        """Rename a processing ID.

        Args:
            old_id: Current processing name.
            new_id: New processing name.

        Raises:
            ValueError: If old_id doesn't exist or new_id already exists.
        """
        if not self.has_processing(old_id):
            raise ValueError(f"Processing '{old_id}' does not exist")

        if old_id != new_id and self.has_processing(new_id):
            raise ValueError(f"Processing '{new_id}' already exists")

        if old_id == new_id:
            return  # No change needed

        idx = self._processing_id_to_index[old_id]
        self._processing_ids[idx] = new_id
        del self._processing_id_to_index[old_id]
        self._processing_id_to_index[new_id] = idx

    def reset_processings(self, new_processings: List[str]) -> None:
        """Reset processing IDs to a new list.

        Args:
            new_processings: List of new processing names.
        """
        self._processing_ids = list(new_processings)
        self._processing_id_to_index = {
            pid: idx for idx, pid in enumerate(new_processings)
        }

    def __repr__(self) -> str:
        return f"ProcessingManager(processing_ids={self._processing_ids})"
