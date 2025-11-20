"""
Sample ID generation and tracking.

This module provides the SampleManager class for generating unique sample
and row IDs within the indexer system.
"""

from typing import Optional


class SampleManager:
    """
    Manage sample and row ID generation.

    This class provides simple, focused functionality for generating unique
    IDs within the indexer. It delegates storage to IndexStore and provides
    auto-incrementing ID generation.

    The SampleManager is stateless and queries the IndexStore for current
    maximum values, ensuring consistency even after manual DataFrame modifications.
    """

    def __init__(self, store):
        """
        Initialize the sample manager.

        Args:
            store: IndexStore instance for querying max values.
        """
        self._store = store

    def next_row_id(self) -> int:
        """
        Get the next available row ID.

        Returns:
            int: Next row ID (max + 1, or 0 if empty).

        Example:
            >>> next_id = manager.next_row_id()
            >>> # Use next_id for new row
        """
        max_val = self._store.get_max("row")
        return (max_val + 1) if max_val is not None else 0

    def next_sample_id(self) -> int:
        """
        Get the next available sample ID.

        Returns:
            int: Next sample ID (max + 1, or 0 if empty).

        Example:
            >>> next_id = manager.next_sample_id()
            >>> # Use next_id for new sample
        """
        max_val = self._store.get_max("sample")
        return (max_val + 1) if max_val is not None else 0

    def generate_row_ids(self, count: int) -> list[int]:
        """
        Generate a list of consecutive row IDs.

        Args:
            count: Number of IDs to generate.

        Returns:
            list[int]: List of consecutive row IDs.

        Example:
            >>> ids = manager.generate_row_ids(5)
            >>> # ids: [0, 1, 2, 3, 4] (if starting from empty)
        """
        start_id = self.next_row_id()
        return list(range(start_id, start_id + count))

    def generate_sample_ids(self, count: int) -> list[int]:
        """
        Generate a list of consecutive sample IDs.

        Args:
            count: Number of IDs to generate.

        Returns:
            list[int]: List of consecutive sample IDs.

        Example:
            >>> ids = manager.generate_sample_ids(3)
            >>> # ids: [10, 11, 12] (if current max is 9)
        """
        start_id = self.next_sample_id()
        return list(range(start_id, start_id + count))
