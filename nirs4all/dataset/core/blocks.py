# blocks.py
from __future__ import annotations
import numpy as np
from typing import Final, Any


class Block:
    """
    C-contiguous chunk (n_samples × n_features), strictly immutable.

    * The attributes `_data`, `_source_id`, `_processing_id` are frozen
      after construction;
    * The NumPy buffer is marked WRITEABLE=False, so any attempt
      to write from the user side immediately raises an error;
    * No method exposes a mutator.
    """
    __slots__: Final = ("_data", "_source_id", "_processing_id", "_locked")

    # ---------- construction -------------------------------------------------
    def __init__(
        self,
        data: np.ndarray,
        source_id: int,
        processing_id: int,
    ) -> None:
        # Ensure C-contiguity
        if not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)

        # Make the buffer read-only
        data.setflags(write=False)

        # Assignments: use object.__setattr__ to bypass
        # the blocking __setattr__ defined below.
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_source_id", int(source_id))
        object.__setattr__(self, "_processing_id", int(processing_id))

        # Permanently lock the instance
        object.__setattr__(self, "_locked", True)

    # ---------- post-init protection -----------------------------------------
    def __setattr__(self, name: str, value: Any) -> None:  # noqa: D401
        """
        Forbids any modification after initialization.

        Always raises AttributeError, except *during* __init__,
        thanks to the `_locked` flag.
        """
        if getattr(self, "_locked", False):
            raise AttributeError(
                f"{self.__class__.__name__} is immutable; "
                f"attempt to modify attribute '{name}'"
            )
        # During __init__
        object.__setattr__(self, name, value)

    # ---------- read-only access ---------------------------------------------
    @property
    def data(self) -> np.ndarray:
        """NumPy buffer (read-only)."""
        return self._data

    @property
    def source_id(self) -> int:
        return self._source_id

    @property
    def processing_id(self) -> int:
        return self._processing_id

    @property
    def n_samples(self) -> int:
        return self._data.shape[0]

    @property
    def n_features(self) -> int:
        return self._data.shape[1]

    # ---------- functional API ----------------------------------------------
    def slice(self, rows: np.ndarray | slice):
        """
        "Zero-copy" view on the requested rows.

        The returned array remains non-writeable since it is derived from a
        parent with WRITEABLE=False.
        """
        return self._data[rows]

    # ---------- representation & hash ---------------------------------------
    def __repr__(self) -> str:
        return (
            f"Block(source_id={self._source_id}, "
            f"processing_id={self._processing_id}, "
            f"shape={self._data.shape}, dtype={self._data.dtype})"
        )

    def __hash__(self) -> int:  # Allows use as a dict key if needed
        return hash((self._source_id, self._processing_id))
