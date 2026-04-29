"""Strict-linear kernel utilities for AOM-Ridge.

All routines work with row spectra. A strict linear operator
``A_b in R^{p x p}`` acts on row spectra as ``X_b = X A_b^T``.

The superblock kernel is

```text
K_super = sum_b s_b^2 Xc A_b^T A_b Xc^T
```

and the matrix used to recover an original-space coefficient is

```text
U = sum_b s_b^2 A_b^T A_b Xc^T
beta = U C, with C = (K + alpha I)^-1 Yc
```

Operator banks live in ``bench/AOM_v0/aompls`` and are imported here.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence

import numpy as np
from aompls.banks import bank_by_name
from aompls.operators import IdentityOperator, LinearSpectralOperator

OperatorBankSpec = str | Sequence[LinearSpectralOperator]
BlockScaling = str


# ----------------------------------------------------------------------
# Y / target shape helpers
# ----------------------------------------------------------------------


def as_2d_y(Y: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ``Y`` as a 2D ``(n, q)`` array and a flag ``was_1d``.

    The flag is used to restore the 1D shape on prediction.
    """
    Y = np.asarray(Y)
    if Y.ndim == 1:
        return Y.reshape(-1, 1).astype(float, copy=False), True
    if Y.ndim != 2:
        raise ValueError("Y must be 1D or 2D")
    return Y.astype(float, copy=False), False


# ----------------------------------------------------------------------
# Operator bank resolution and cloning
# ----------------------------------------------------------------------


def resolve_operator_bank(
    operator_bank: OperatorBankSpec, p: int | None = None
) -> list[LinearSpectralOperator]:
    """Resolve a bank spec to a list of operator instances.

    The resolved bank is a deep copy: callers may mutate it freely without
    affecting cached presets. Identity is guaranteed to appear exactly once:
    duplicate ``IdentityOperator`` entries are deduplicated (first one kept),
    and identity is prepended only if absent.
    """
    if isinstance(operator_bank, str):
        ops = bank_by_name(operator_bank, p=p)
    elif isinstance(operator_bank, Iterable):
        ops = list(operator_bank)
    else:
        raise TypeError("operator_bank must be a name or a sequence of operators")
    if not ops:
        raise ValueError("operator_bank resolved to an empty list")
    cloned = clone_operator_bank(ops, p=p)
    # Deduplicate identity operators: keep the first occurrence, drop the rest.
    seen_identity = False
    deduped: list[LinearSpectralOperator] = []
    for op in cloned:
        if isinstance(op, IdentityOperator):
            if seen_identity:
                continue
            seen_identity = True
        deduped.append(op)
    if not seen_identity:
        deduped = [IdentityOperator(p=p)] + deduped
    return deduped


def clone_operator_bank(
    operators: Sequence[LinearSpectralOperator], p: int | None = None
) -> list[LinearSpectralOperator]:
    """Deep-copy operator instances and (optionally) reset their ``p``.

    Cached matrices are dropped so each clone re-derives them when fitted.
    """
    out: list[LinearSpectralOperator] = []
    for op in operators:
        clone = copy.deepcopy(op)
        clone._matrix_cache = None
        if p is not None:
            clone.p = p
        else:
            clone.p = None
        out.append(clone)
    return out


def fit_operator_bank(
    operators: Sequence[LinearSpectralOperator],
    X: np.ndarray,
    Y: np.ndarray | None = None,
) -> list[LinearSpectralOperator]:
    """Bind every operator to the dimensionality of ``X``.

    The operators in this phase are strict-linear and do not learn parameters
    from data; ``fit`` only stores ``p``. Returns the same list for chaining.
    """
    for op in operators:
        op.fit(X, Y)
    return list(operators)


# ----------------------------------------------------------------------
# Block scales and metric application
# ----------------------------------------------------------------------


def _validate_block_scaling(name: str) -> str:
    name = name.lower()
    if name not in ("rms", "none", "scale_power"):
        raise ValueError("block_scaling must be 'rms', 'none', or 'scale_power'")
    return name


def _frob_norm(M: np.ndarray) -> float:
    return float(np.linalg.norm(M, ord="fro"))


def compute_block_scales_from_xt(
    Xt: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    block_scaling: BlockScaling = "rms",
    eps: float = 1e-12,
    scale_power: float = 1.0,
) -> np.ndarray:
    """Compute per-operator block scales ``s_b`` from ``X^T``.

    With ``Xt = Xc^T`` (shape ``p x n``), ``A_b @ Xt`` equals ``X_b^T``,
    whose Frobenius norm matches ``||Xc A_b^T||_F``.

    ``block_scaling``:

    - ``"none"``: ``s_b = 1`` (raw blocks; informative blocks dominate).
    - ``"rms"``: ``s_b = 1 / (RMS(X_b) + eps)`` (equal Frobenius norm per block).
    - ``"scale_power"``: ``s_b = (RMS_target / (RMS(X_b) + eps)) ** scale_power``
      with ``RMS_target = 1`` (chosen so that ``scale_power=1`` reduces to
      ``"rms"`` and ``scale_power=0`` reduces to ``"none"``). Intermediate
      values give a soft equalisation.
    """
    block_scaling = _validate_block_scaling(block_scaling)
    p, n = Xt.shape
    if block_scaling == "none":
        return np.ones(len(operators), dtype=float)
    denom = max(np.sqrt(float(n) * float(p)), 1.0)
    rms_b = np.empty(len(operators), dtype=float)
    for i, op in enumerate(operators):
        AXt = op.apply_cov(Xt)
        rms_b[i] = _frob_norm(AXt) / denom
    inv = 1.0 / (rms_b + eps)
    if block_scaling == "rms":
        return inv
    # scale_power
    if scale_power == 0.0:
        return np.ones(len(operators), dtype=float)
    return inv ** float(scale_power)


def metric_times_xt(
    Xt: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    block_scales: np.ndarray,
) -> np.ndarray:
    """Compute ``U = sum_b s_b^2 A_b^T A_b Xc^T`` without materializing ``M``.

    ``Xt`` is ``Xc^T`` with shape ``(p, n)``. Returns ``U`` with shape
    ``(p, n)``. The caller can then form ``K = Xc @ U`` or
    ``K_cross = X_left_c @ U``.
    """
    if len(operators) != len(block_scales):
        raise ValueError("operators and block_scales must have the same length")
    p, n = Xt.shape
    U = np.zeros((p, n), dtype=float)
    for op, s in zip(operators, block_scales, strict=False):
        AXt = op.apply_cov(Xt)             # A_b @ Xt, shape (p, n)
        AtAXt = op.adjoint_vec(AXt)        # A_b^T @ (A_b @ Xt), shape (p, n)
        U += float(s) ** 2 * AtAXt
    return U


# ----------------------------------------------------------------------
# Train / cross kernels
# ----------------------------------------------------------------------


def linear_operator_kernel_train(
    Xc: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    block_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the train superblock kernel ``K`` and the matrix ``U``.

    ``K = Xc @ U`` and ``U = sum_b s_b^2 A_b^T A_b Xc^T``. The kernel is
    symmetrized to suppress floating-point asymmetry.
    """
    Xt = Xc.T
    U = metric_times_xt(Xt, operators, block_scales)
    K = Xc @ U
    K = 0.5 * (K + K.T)
    return K, U


def linear_operator_kernel_cross(
    X_left_c: np.ndarray, U_train: np.ndarray
) -> np.ndarray:
    """Compute ``K_cross = X_left_c @ U_train`` (shape ``n_left x n_train``)."""
    return X_left_c @ U_train


# ----------------------------------------------------------------------
# Explicit superblock (tests only)
# ----------------------------------------------------------------------


def explicit_superblock(
    Xc: np.ndarray,
    operators: Sequence[LinearSpectralOperator],
    block_scales: np.ndarray,
) -> np.ndarray:
    """Materialize ``Phi = [s_1 X A_1^T | ... | s_B X A_B^T]`` for tests.

    Used only by the equivalence tests; never called from estimators.
    """
    blocks = []
    for op, s in zip(operators, block_scales, strict=False):
        Z_b = op.transform(Xc)
        blocks.append(float(s) * Z_b)
    if not blocks:
        raise ValueError("operators must be non-empty")
    return np.concatenate(blocks, axis=1)
