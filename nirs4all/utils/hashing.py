"""Fast, deterministic content hashing for numpy arrays.

Provides ``compute_data_hash`` which hashes the raw bytes of a numpy array
using xxhash (preferred for speed) with a SHA-256 fallback.
"""

import hashlib

import numpy as np

try:
    import xxhash
    _HAS_XXHASH = True
except ImportError:
    _HAS_XXHASH = False

def compute_data_hash(X: np.ndarray) -> str:
    """Compute a deterministic hex digest for a numpy array.

    Uses xxhash (xxh128) when available for speed, otherwise falls back
    to SHA-256.  The array is made contiguous before hashing so that
    the result is independent of memory layout.

    Args:
        X: Numpy array to hash.

    Returns:
        Hex digest string.
    """
    data = np.ascontiguousarray(X).data.tobytes()
    if _HAS_XXHASH:
        return xxhash.xxh128(data).hexdigest()
    return hashlib.sha256(data).hexdigest()
