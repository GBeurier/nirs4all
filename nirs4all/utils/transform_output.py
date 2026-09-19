"""Adapt transformer results to dense spectral storage with bounded allocation."""

from typing import Any

MAX_DENSE_TRANSFORM_BYTES = 256 * 1024 * 1024


def normalize_transform_output(output: Any, operator_name: str, *, max_dense_bytes: int = MAX_DENSE_TRANSFORM_BYTES) -> Any:
    """Preserve dense arrays and convert sparse matrices only within the budget."""
    from scipy.sparse import issparse

    if not issparse(output):
        return output
    dense_bytes = int(output.shape[0]) * int(output.shape[1]) * int(output.dtype.itemsize)
    if dense_bytes > max_dense_bytes:
        raise ValueError(
            f"{operator_name} returned sparse features with shape {output.shape}; "
            f"dense spectral storage would require {dense_bytes} bytes, exceeding the {max_dense_bytes}-byte "
            "sparse conversion limit. Reduce the output feature count or dataset size."
        )
    return output.toarray()
