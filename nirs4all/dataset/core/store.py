from __future__ import annotations
import numpy as np
from collections import OrderedDict
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .blocks import Block
    from .views import TensorView


class FeatureStore:
    """
    (source_id, processing_id) → Block
    LRU + refcount per processing_id.
    """

    def __init__(self, max_blocks: int = 64) -> None:
        self.max_blocks = max_blocks
        self._blocks: OrderedDict[tuple[int, int], Block] = OrderedDict()
        self._refcounts: dict[int, int] = {}

    # 🔹 initial refcount utility (called by SpectraDataset.load)
    def sync_refcount(self, processing_id: int, n_users: int):
        self._refcounts[processing_id] = n_users

    def view(self, rows, source_id: int, processing_id: int) -> TensorView:
        """Create a view on an existing block."""
        # Local import to avoid circular import
        from .views import TensorView
        block = self._blocks[(source_id, processing_id)]
        return TensorView(block, rows)

    def add_block(self,
                  block: Block,
                  source_id: int,
                  processing_id: int,
                  n_users: int) -> None:
        """Add a block to the store."""
        key = (source_id, processing_id)
        self._blocks[key] = block
        self._blocks.move_to_end(key)  # LRU: most recent
        self.inc_ref(processing_id, n_users)
        self._evict_if_needed()

    def inc_ref(self, processing_id: int, n: int = 1) -> None:
        """Increase ref-count for a processing_id."""
        self._refcounts[processing_id] = self._refcounts.get(processing_id, 0) + n

    def dec_ref(self, processing_id: int, n: int = 1) -> None:
        """
        Decrease ref-count for a processing_id.

        When the counter reaches zero the entry is removed so that
        prune_unused() can get rid of the corresponding blocks.
        """
        current = self._refcounts.get(processing_id, 0)
        new_val = max(0, current - n)
        if new_val == 0:
            self._refcounts.pop(processing_id, None)
        else:
            self._refcounts[processing_id] = new_val

    def prune_unused(self) -> None:
        """Remove unreferenced blocks."""
        to_remove = []
        for (source_id, processing_id) in self._blocks.keys():
            if self._refcounts.get(processing_id, 0) == 0:
                to_remove.append((source_id, processing_id))

        for key in to_remove:
            del self._blocks[key]

    def concat_blocks(self,
                      blocks: Sequence[Block],
                      new_source_id: int,
                      new_processing_id: int,
                      axis: int = 1) -> Block:
        """Permanently concatenate several blocks (features)."""
        from .blocks import Block
        data_arrays = [block.data for block in blocks]
        concatenated = np.concatenate(data_arrays, axis=axis)
        return Block(concatenated, new_source_id, new_processing_id)

    def slice_block(self,
                    block: Block,
                    col_slice: slice,
                    new_source_id: int,
                    new_processing_id: int) -> Block:
        """Create a new block view → unique C-contiguous copy."""
        from .blocks import Block
        sliced_data = block.data[:, col_slice].copy()
        return Block(sliced_data, new_source_id, new_processing_id)

    def list_blocks(self) -> list[tuple[int, int]]:
        """List the keys (source_id, processing_id) of available blocks."""
        return list(self._blocks.keys())

    def has_block(self, source_id: int, processing_id: int) -> bool:
        """Check if a block exists."""
        return (source_id, processing_id) in self._blocks

    def get_block(self, source_id: int, processing_id: int) -> Block:
        """Retrieve a block by its identifiers."""
        key = (source_id, processing_id)
        if key not in self._blocks:
            raise KeyError(f"Block {key} not found")
        # Move to end for LRU
        self._blocks.move_to_end(key)
        return self._blocks[key]

    def _evict_if_needed(self) -> None:
        """LRU eviction if too many blocks."""
        while len(self._blocks) > self.max_blocks:
            # Remove the oldest (FIFO order in OrderedDict)
            oldest_key = next(iter(self._blocks))
            del self._blocks[oldest_key]

    def unique_source_ids(self) -> list[int]:
        """List of unique source IDs."""
        return list({src for src, _ in self._blocks})

    def unique_processing_ids(self, source_ids: Sequence[int] | None = None) -> list[int]:
        """List of unique processing IDs for given source IDs."""
        if source_ids is None:
            return list({proc for _, proc in self._blocks})
        return list({proc for src, proc in self._blocks if src in source_ids})

    def iter_blocks(self):
        """Yield ((source_id, processing_id), Block) pairs in LRU order."""
        return self._blocks.items()