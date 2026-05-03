"""Centred + trace-normalised AOM block kernels for mkR / MKM / BLUP.

Each AOM block produces a raw kernel ``K_b_raw = X (A_b^T A_b) X^T`` then
two standardisation steps are applied:

1. **Centring** — ``K_b_c = H K_b_raw H`` with ``H = I - 1 1^T / n``. After
   centring, ``K_b_c @ 1 == 0``.
2. **Trace normalisation** — ``K_b = (n / tr(K_b_c)) K_b_c``. After this,
   ``tr(K_b) / n == 1``, so per-block weights ``eta_b`` (mkR) and variance
   components ``sigma_b^2`` (MKM) are directly comparable across blocks.

Cross-kernel double-centring uses **only training-side moments** so that
test rows never bleed into kernel construction. Specifically, with
``mu_train`` the row mean and ``nu_train`` the global mean of the training
raw kernel, the cross-kernel centring formula is

```text
K_b_cross_c = K_b_cross_raw
            - K_b_cross_raw.mean(axis=1)[:, None]   # test-row mean (computed from training cross)
            - mu_train[None, :]                      # training row mean (stored at fit time)
            + nu_train                               # training global mean (stored at fit time)
```

The test-row mean is computed from ``K_b_cross_raw`` which is itself a
function of test data and training data; this is the standard kernel-PCA /
"feature centring at training mean" construction and is **not** test-data
leakage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from aompls.operators import LinearSpectralOperator

from .kernels import (
    clone_operator_bank,
    fit_operator_bank,
    resolve_operator_bank,
)

__all__ = [
    "AOMKernelizer",
    "BlockKernelStats",
    "kernel_alignment_matrix",
]


# ----------------------------------------------------------------------
# Stats dataclass
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BlockKernelStats:
    """Statistics needed to centre a cross kernel using training data only."""

    mu: np.ndarray  # (n_train,) row mean of training raw kernel
    nu: float       # global mean of training raw kernel
    tau: float      # trace scale  n / max(tr(K_c), eps)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _check_2d(X: np.ndarray, name: str) -> np.ndarray:
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim != 2:
        raise ValueError(f"{name} must be a 2D array (got {Xa.ndim}D)")
    return Xa


def _block_kernel_train(
    Xc: np.ndarray, op: LinearSpectralOperator
) -> np.ndarray:
    """Compute the raw block kernel ``K = Xc (A^T A) Xc^T``."""
    AXt = op.apply_cov(Xc.T)             # (p, n)
    AtAXt = op.adjoint_vec(AXt)          # (p, n)
    K = Xc @ AtAXt                        # (n, n)
    return 0.5 * (K + K.T)


def _block_kernel_cross(
    X_left_c: np.ndarray, X_train_c: np.ndarray, op: LinearSpectralOperator
) -> np.ndarray:
    """Compute the raw cross kernel ``K_left = X_left_c (A^T A) X_train_c^T``."""
    AXt = op.apply_cov(X_train_c.T)       # (p, n_train)
    AtAXt = op.adjoint_vec(AXt)           # (p, n_train)
    return X_left_c @ AtAXt               # (n_left, n_train)


def _double_center_train(
    K_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply ``H K_raw H``; return ``(K_c, mu, nu)``."""
    mu = K_raw.mean(axis=1)               # (n,) — row mean
    nu = float(mu.mean())                 # scalar (== K.mean())
    K_c = K_raw - mu[:, None] - mu[None, :] + nu
    return 0.5 * (K_c + K_c.T), mu, nu


def _double_center_cross(
    K_raw_cross: np.ndarray, mu_train: np.ndarray, nu_train: float
) -> np.ndarray:
    """Centre a ``(n_left, n_train)`` cross kernel using training stats only.

    See module docstring for the formula. The test-side row mean of
    ``K_raw_cross`` is computed at transform time from data already needed
    for the cross kernel; this is not test-data leakage.
    """
    if K_raw_cross.shape[1] != mu_train.shape[0]:
        raise ValueError("K_raw_cross columns must match training size")
    row_mean_test = K_raw_cross.mean(axis=1)
    return K_raw_cross - row_mean_test[:, None] - mu_train[None, :] + nu_train


# ----------------------------------------------------------------------
# Diagnostic helper
# ----------------------------------------------------------------------


def kernel_alignment_matrix(K_blocks: Sequence[np.ndarray]) -> np.ndarray:
    """Pairwise normalised Frobenius inner product of a list of kernels.

    ``A_ij = <K_i, K_j>_F / (||K_i||_F * ||K_j||_F)``.

    The matrix has 1.0 on the diagonal (numerical noise aside). Off-diagonal
    values approaching 1 indicate that the corresponding kernels carry
    nearly identical information; their associated weights / variance
    components will not be separately identifiable.
    """
    B = len(K_blocks)
    if B == 0:
        return np.zeros((0, 0), dtype=float)
    norms = np.array([np.linalg.norm(K, ord="fro") for K in K_blocks], dtype=float)
    A = np.eye(B, dtype=float)
    for i in range(B):
        for j in range(i + 1, B):
            denom = norms[i] * norms[j]
            if denom < 1e-30:
                A[i, j] = A[j, i] = 0.0
            else:
                A[i, j] = A[j, i] = float(np.sum(K_blocks[i] * K_blocks[j]) / denom)
    return A


# ----------------------------------------------------------------------
# AOMKernelizer
# ----------------------------------------------------------------------


class AOMKernelizer:
    """Build centred + trace-normalised AOM block kernels (sklearn-style).

    Parameters
    ----------
    operator_bank : str | sequence
        Bank name (e.g. ``"compact"``, ``"default"``) or explicit operator
        sequence. See ``aompls.banks.bank_by_name``.
    center : bool, default True
        Apply double-centring at the kernel level.
    normalize : str, default "trace"
        ``"trace"`` rescales each block so that ``tr(K_b)/n == 1``;
        ``"none"`` skips normalisation.
    eps : float, default 1e-12
        Floor for trace denominator.

    Attributes (set by ``fit``)
    ---------------------------
    operators_ : list of fitted operator clones (one per block).
    block_names_ : list[str]
    n_train_ : int
    K_train_blocks_ : list[ndarray (n_train, n_train)]
        Centred + normalised training block kernels.
    block_stats_ : list[BlockKernelStats]
        Statistics needed for cross-kernel centring at transform time.
    x_mean_ : ndarray (p,)
        Training-mean row used to centre features before kernel computation.
    """

    def __init__(
        self,
        operator_bank: object = "compact",
        *,
        center: bool = True,
        normalize: str = "trace",
        eps: float = 1e-12,
        zero_trace_policy: str = "raise",
        zero_trace_threshold: float = 1e-12,
        top_k_active: int | None = None,
        screen_score_method: str = "norm",
        screen_diversity_threshold: float = 0.98,
        screen_keep_identity: bool = True,
    ) -> None:
        if normalize not in ("trace", "none"):
            raise ValueError("normalize must be 'trace' or 'none'")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        if zero_trace_policy not in ("raise", "drop", "warn_keep"):
            raise ValueError(
                "zero_trace_policy must be 'raise', 'drop', or 'warn_keep'"
            )
        if zero_trace_threshold <= 0.0:
            raise ValueError("zero_trace_threshold must be > 0")
        if top_k_active is not None and top_k_active < 1:
            raise ValueError("top_k_active must be >= 1 or None")
        if screen_score_method not in ("norm", "kta", "blend"):
            raise ValueError(
                "screen_score_method must be 'norm', 'kta', or 'blend'"
            )
        self.operator_bank = operator_bank
        self.center = bool(center)
        self.normalize = normalize
        self.eps = float(eps)
        self.zero_trace_policy = zero_trace_policy
        self.zero_trace_threshold = float(zero_trace_threshold)
        self.top_k_active = top_k_active
        self.screen_score_method = screen_score_method
        self.screen_diversity_threshold = float(screen_diversity_threshold)
        self.screen_keep_identity = bool(screen_keep_identity)
        # Fitted state
        self.operators_: list[LinearSpectralOperator] | None = None
        self.block_names_: list[str] | None = None
        self.n_train_: int = 0
        self.K_train_blocks_: list[np.ndarray] | None = None
        self.block_stats_: list[BlockKernelStats] | None = None
        self.x_mean_: np.ndarray | None = None
        self._X_train_c_: np.ndarray | None = None  # for transform() cross kernels

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AOMKernelizer":
        """Fit operators to ``X`` and store training kernel statistics.

        If ``top_k_active`` is set and ``y`` is provided, the operator bank is
        pre-screened using the supplied ``screen_score_method`` and only the
        top-k operators (by alignment with ``y``) are kept. This lets the
        caller use a large bank like ``"default"`` (100 ops) without paying
        the full kernel-construction cost or suffering from selection
        variance during weight learning.
        """
        Xa = _check_2d(X, "X")
        n, p = Xa.shape
        x_mean = Xa.mean(axis=0)
        Xc = Xa - x_mean

        ops_template = resolve_operator_bank(self.operator_bank, p=p)
        ops = clone_operator_bank(ops_template, p=p)
        fit_operator_bank(ops, Xc)

        # Active screening: if top_k_active < len(ops) and y is supplied,
        # use the existing `screen_active_operators` to keep only the most
        # informative subset BEFORE we materialize the kernels.
        if (
            self.top_k_active is not None
            and y is not None
            and len(ops) > self.top_k_active
        ):
            from .selection import screen_active_operators
            y_arr = np.asarray(y, dtype=float)
            if y_arr.ndim == 1:
                y_2d = y_arr.reshape(-1, 1)
            else:
                y_2d = y_arr
            active_indices, _, _ = screen_active_operators(
                Xa, y_2d, ops,
                top_m=self.top_k_active,
                diversity_threshold=self.screen_diversity_threshold,
                keep_identity=self.screen_keep_identity,
                score_method=self.screen_score_method,
            )
            ops = [ops[i] for i in active_indices]
            # Re-fit the (already-fitted) clones to bind p again — defensive.
            fit_operator_bank(ops, Xc)
        self._n_active_ = len(ops)

        K_blocks: list[np.ndarray] = []
        stats: list[BlockKernelStats] = []
        kept_indices: list[int] = []
        zero_threshold = float(n) * self.zero_trace_threshold
        for idx, op in enumerate(ops):
            K_raw = _block_kernel_train(Xc, op)
            if self.center:
                K_c, mu, nu = _double_center_train(K_raw)
            else:
                K_c, mu, nu = K_raw, np.zeros(n, dtype=float), 0.0
            if self.normalize == "trace":
                tr = float(np.trace(K_c))
                if tr < zero_threshold:
                    if self.zero_trace_policy == "raise":
                        raise ValueError(
                            f"block {op.name!r} has near-zero trace "
                            f"(tr={tr:.3e}, threshold={zero_threshold:.3e}); "
                            "trace normalisation would amplify numerical noise"
                        )
                    if self.zero_trace_policy == "drop":
                        continue
                    # warn_keep: floor with eps to avoid exception (still risky)
                    tr = max(tr, self.eps)
                tau = float(n) / max(tr, self.eps)
                K_norm = tau * K_c
            else:
                tau = 1.0
                K_norm = K_c
            K_blocks.append(0.5 * (K_norm + K_norm.T))
            stats.append(BlockKernelStats(mu=mu, nu=nu, tau=tau))
            kept_indices.append(idx)

        if self.zero_trace_policy == "drop":
            ops = [ops[i] for i in kept_indices]
        if not ops:
            raise ValueError("all blocks dropped (zero trace); cannot fit")

        self.operators_ = ops
        self.block_names_ = [op.name for op in ops]
        self.n_train_ = n
        self.K_train_blocks_ = K_blocks
        self.block_stats_ = stats
        self.x_mean_ = x_mean
        self._X_train_c_ = Xc
        return self

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> list[np.ndarray]:
        """Fit and return the list of training block kernels."""
        self.fit(X, y)
        assert self.K_train_blocks_ is not None
        return list(self.K_train_blocks_)

    def transform(self, X: np.ndarray) -> list[np.ndarray]:
        """Return list of cross block kernels for ``X``.

        Each output has shape ``(n_test, n_train)``. Centring uses **only**
        training-side moments stored at ``fit`` time.
        """
        if (
            self.operators_ is None
            or self.block_stats_ is None
            or self.x_mean_ is None
            or self._X_train_c_ is None
        ):
            raise RuntimeError("AOMKernelizer must be fitted before transform")
        Xa = _check_2d(X, "X")
        if Xa.shape[1] != self.x_mean_.shape[0]:
            raise ValueError(
                f"X has {Xa.shape[1]} features; kernelizer expects "
                f"{self.x_mean_.shape[0]}"
            )
        Xc_left = Xa - self.x_mean_
        out: list[np.ndarray] = []
        for op, stats in zip(self.operators_, self.block_stats_, strict=False):
            K_raw_cross = _block_kernel_cross(Xc_left, self._X_train_c_, op)
            if self.center:
                K_c_cross = _double_center_cross(K_raw_cross, stats.mu, stats.nu)
            else:
                K_c_cross = K_raw_cross
            if self.normalize == "trace":
                K_norm_cross = stats.tau * K_c_cross
            else:
                K_norm_cross = K_c_cross
            out.append(K_norm_cross)
        return out

    def get_params(self, deep: bool = True) -> dict:
        """Sklearn-compatible parameter dict (not used by AOMKernelizer
        directly; kept for cloning)."""
        del deep
        return {
            "operator_bank": self.operator_bank,
            "center": self.center,
            "normalize": self.normalize,
            "eps": self.eps,
        }
