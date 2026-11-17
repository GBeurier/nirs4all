"""Deterministic content identity for numpy arrays.

``compute_data_hash`` is the single algorithm behind dataset content identities:
SHA-256 over the raw bytes of the C-contiguous array. The identity is persisted
(workspace runs, bundle fingerprints, Studio provenance) and compared across
sessions and machines, so it never depends on optional installed packages.
"""

import hashlib

import numpy as np


def compute_data_hash(X: np.ndarray) -> str:
    """Compute the SHA-256 hex digest of a numpy array's raw bytes.

    The array is made contiguous before hashing so that the result is
    independent of memory layout.

    Args:
        X: Numpy array to hash.

    Returns:
        64-character lowercase hex digest.
    """
    return hashlib.sha256(np.ascontiguousarray(X).data.tobytes()).hexdigest()
