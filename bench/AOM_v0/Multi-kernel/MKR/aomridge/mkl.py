"""MKL-light supervised block weights for AOM-Ridge.

Replaces the hand-set scales ``s_b`` of the superblock with weights
``w_b >= 0``, ``sum_b w_b = 1`` learned fold-locally from kernel-target
alignment (Cristianini, KTA). The combined kernel is *linear* in the
weights (no squaring):

```text
K_mkl = sum_b w_b K_b,        K_b = (s_b X A_b^T)(s_b X A_b^T)^T
U_mkl = sum_b w_b A_b^T A_b X^T   (with the s_b^2 already absorbed)
```

so the standard dual-Ridge identities still hold and ``coef_ = U_mkl @ C``
lives in the original feature space.

The closed-form alignment weight is

```text
align_b = <K_b, Y Y^T>_F / (||K_b||_F * ||Y Y^T||_F)
w_b     = max(align_b, 0) / sum_b max(align_b, 0)
```

restricted to the ``top_k`` highest-aligned operators (others get
``w_b = 0``). Identity is *not* force-kept; if no operator has a positive
alignment we fall back to a uniform weight on the top-k.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from aompls.operators import LinearSpectralOperator

# ----------------------------------------------------------------------
# Alignment scoring
# ----------------------------------------------------------------------


def kta_score(K_b: np.ndarray, YYt: np.ndarray) -> float:
    """Kernel-target alignment ``<K_b, Y Y^T>_F / (||K_b||_F * ||Y Y^T||_F)``.

    Scale-invariant in [-1, 1]. Returns 0.0 when either operand has zero
    Frobenius norm.
    """
    K_norm = float(np.linalg.norm(K_b, ord="fro"))
    Y_norm = float(np.linalg.norm(YYt, ord="fro"))
    if K_norm < 1e-30 or Y_norm < 1e-30:
        return 0.0
    return float(np.sum(K_b * YYt) / (K_norm * Y_norm))


def _block_kernel(
    Xc: np.ndarray, op: LinearSpectralOperator, scale: float
) -> np.ndarray:
    """Return ``K_b = (s_b X A_b^T)(s_b X A_b^T)^T`` for one operator."""
    AXt = op.apply_cov(Xc.T)                # (p, n)
    AtAXt = op.adjoint_vec(AXt)             # (p, n)
    K_b = float(scale) ** 2 * (Xc @ AtAXt)  # (n, n)
    return 0.5 * (K_b + K_b.T)


# ----------------------------------------------------------------------
# Weight learning
# ----------------------------------------------------------------------


def learn_block_weights(
    operators: Sequence[LinearSpectralOperator],
    Xc: np.ndarray,
    Yc: np.ndarray,
    scales: np.ndarray,
    top_k: int = 6,
    mode: str = "alignment",
) -> np.ndarray:
    """Learn supervised block weights on the simplex.

    Parameters
    ----------
    operators
        Fitted operator clones (one per block).
    Xc
        Centred / scaled training matrix (training fold only — never the
        full dataset in CV).
    Yc
        Centred target matrix, shape ``(n, q)``.
    scales
        Per-operator scales ``s_b`` already absorbed into the block kernels.
        Pass ``np.ones(B)`` if no per-block scaling is desired.
    top_k
        Keep at most this many blocks; the rest get weight 0.
    mode
        Currently only ``"alignment"``: ``w_b ∝ max(align_b, 0)``.

    Returns
    -------
    weights : np.ndarray of shape (B,)
        Non-negative weights summing to 1.

    Notes
    -----
    The weights live on the simplex by construction. If no operator has a
    positive alignment (e.g. uncorrelated synthetic data), the top-k
    operators by ``|align|`` receive a uniform weight so the kernel is
    well-defined.
    """
    if mode != "alignment":
        raise ValueError(f"unknown mkl mode {mode!r}; expected 'alignment'")
    if len(operators) != len(scales):
        raise ValueError("operators and scales must have the same length")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    YYt = Yc @ Yc.T
    n_ops = len(operators)
    aligns = np.zeros(n_ops, dtype=float)
    for b, (op, s) in enumerate(zip(operators, scales, strict=False)):
        K_b = _block_kernel(Xc, op, float(s))
        aligns[b] = kta_score(K_b, YYt)

    pos = np.maximum(aligns, 0.0)
    # Mask everything outside the top_k by descending alignment.
    if n_ops > top_k:
        cutoff_idx = np.argsort(-aligns)[:top_k]
        mask = np.zeros(n_ops, dtype=bool)
        mask[cutoff_idx] = True
        pos = np.where(mask, pos, 0.0)
    total = float(pos.sum())
    if total > 0.0:
        return pos / total
    # Fallback: no positive alignment among the top-k -> uniform on the top-k.
    weights = np.zeros(n_ops, dtype=float)
    if n_ops <= top_k:
        weights[:] = 1.0 / n_ops
    else:
        cutoff_idx = np.argsort(-aligns)[:top_k]
        weights[cutoff_idx] = 1.0 / top_k
    return weights


# ----------------------------------------------------------------------
# Combined kernel and U matrix
# ----------------------------------------------------------------------


def mkl_kernel_train(
    Xc: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    weights: np.ndarray,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the linear-in-weight combined kernel and ``U`` matrix.

    ``K_mkl = sum_b w_b * (s_b X A_b^T)(s_b X A_b^T)^T``
    ``U_mkl = sum_b w_b * s_b^2 * A_b^T A_b X^T``

    so that ``K_mkl = X @ U_mkl`` and ``coef_ = U_mkl @ C`` is original-space.
    ``scales`` defaults to all ones (raw block kernels).
    """
    if scales is None:
        scales_arr = np.ones(len(operators), dtype=float)
    else:
        scales_arr = np.asarray(scales, dtype=float)
    if len(operators) != len(weights):
        raise ValueError("operators and weights must have the same length")
    if len(operators) != len(scales_arr):
        raise ValueError("operators and scales must have the same length")
    p, n = Xc.shape[1], Xc.shape[0]
    Xt = Xc.T
    U = np.zeros((p, n), dtype=float)
    for op, w, s in zip(operators, weights, scales_arr, strict=False):
        if w == 0.0:
            continue
        AXt = op.apply_cov(Xt)              # (p, n)
        AtAXt = op.adjoint_vec(AXt)         # (p, n)
        U += float(w) * float(s) ** 2 * AtAXt
    K = Xc @ U
    K = 0.5 * (K + K.T)
    return K, U


def mkl_kernel_cross(X_left_c: np.ndarray, U_train: np.ndarray) -> np.ndarray:
    """Cross-kernel ``K_cross = X_left_c @ U_train`` for prediction."""
    return X_left_c @ U_train
