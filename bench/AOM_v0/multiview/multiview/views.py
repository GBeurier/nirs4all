"""View abstraction and bank construction for AOM-multiview.

A view is a strict-linear operator `A in R^{p x p}` acting on row spectra
(`X_b = X A^T`). This module adds:

- `BlockMaskOperator`: diagonal mask restricting features to a contiguous
  wavelength block.
- `ViewBuilder`: factory assembling operator banks for `preproc_only`,
  `blocks_only`, or `combined` (preproc x block) modes.

See `bench/AOM_v0/multiview/docs/DESIGN_VIEWS.md` for the design rationale,
boundary semantics (block-mask after preprocessing, §5), edge-case rejection
rules (§3.4), and Codex review disposition (§9).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from aompls.banks import bank_by_name
from aompls.operators import (
    ComposedOperator,
    IdentityOperator,
    LinearSpectralOperator,
)


# ---------------------------------------------------------------------------
# Block mask operator
# ---------------------------------------------------------------------------


class BlockMaskOperator(LinearSpectralOperator):
    """Diagonal mask restricting a spectrum to a contiguous wavelength block.

    For block `[start, end) ⊂ [0, p)`, the operator is
    `M = diag(m)` with `m[i] = 1` if `start <= i < end` else `0`.
    `M` is symmetric (`M = M^T`), idempotent (`M^2 = M`), strict-linear,
    and parameter-free.

    Cross-covariance identity (used by `simpls_covariance` /
    `nipals_adjoint`): `(X M^T)^T Y = M X^T Y` since `M = M^T`.
    """

    def __init__(self, start: int, end: int, p: int, name: Optional[str] = None) -> None:
        if not isinstance(p, (int, np.integer)) or int(p) <= 0:
            raise ValueError(f"p must be a positive int, got {p!r}")
        p = int(p)
        if not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)):
            raise ValueError("start and end must be integers")
        start = int(start)
        end = int(end)
        if start < 0 or end > p:
            raise ValueError(f"block [{start}, {end}) out of bounds for p={p}")
        if start >= end:
            raise ValueError(f"block must be non-empty: got [{start}, {end})")
        if start == 0 and end == p:
            raise ValueError("full-cover block is degenerate; use IdentityOperator")
        block_name = name if name is not None else f"mask_{start}_{end}"
        super().__init__(name=block_name, p=p)
        self.start = start
        self.end = end

    def fit(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> "BlockMaskOperator":
        if X is not None:
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError("X must be 2-dimensional")
            if X.shape[1] != self.p:
                raise ValueError(
                    f"X has {X.shape[1]} features; operator {self.name} expects {self.p}"
                )
        return self

    def _transform_impl(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros_like(X)
        out[:, self.start : self.end] = X[:, self.start : self.end]
        return out

    def _apply_cov_impl(self, S: np.ndarray) -> np.ndarray:
        out = np.zeros_like(S)
        out[self.start : self.end, :] = S[self.start : self.end, :]
        return out

    def _adjoint_vec_impl(self, v: np.ndarray) -> np.ndarray:
        out = np.zeros_like(v)
        out[self.start : self.end] = v[self.start : self.end]
        return out

    def _matrix_impl(self, p: int) -> np.ndarray:
        if p != self.p:
            raise ValueError(
                f"BlockMaskOperator built for p={self.p}, got matrix request for p={p}"
            )
        m = np.zeros(p, dtype=float)
        m[self.start : self.end] = 1.0
        return np.diag(m)


# ---------------------------------------------------------------------------
# Block strategies
# ---------------------------------------------------------------------------


def _equal_width_blocks(p: int, K: int) -> List[Tuple[int, int]]:
    """Return K contiguous, approximately equal-width blocks of [0, p)."""
    edges = [int(round(k * p / K)) for k in range(K + 1)]
    edges[0] = 0
    edges[-1] = p
    return [(edges[k], edges[k + 1]) for k in range(K)]


def _resolve_blocks(K: int, p: int, strategy: str) -> List[Tuple[int, int]]:
    if not isinstance(K, (int, np.integer)) or int(K) < 2:
        raise ValueError(f"K must be an int >= 2, got {K!r}")
    K = int(K)
    if K > p:
        raise ValueError(f"K={K} exceeds p={p}; would produce empty blocks")
    if K == p:
        raise ValueError(f"K=p={p} produces single-feature blocks; degenerate")
    if p < 2 * K:
        raise ValueError(
            f"p={p} < 2*K={2 * K}; blocks too narrow for stable views"
        )
    if strategy == "equal_width":
        return _equal_width_blocks(p, K)
    if strategy == "quantile_width":
        raise NotImplementedError("strategy='quantile_width' is a Phase 2 stub")
    if strategy == "chemistry_NIR":
        raise NotImplementedError("strategy='chemistry_NIR' is a Phase 2 stub")
    raise ValueError(f"unknown strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# ViewBuilder
# ---------------------------------------------------------------------------


@dataclass
class ViewBuilder:
    """Factory for operator banks composed of preprocessing and/or block views.

    Use the `.preproc_only`, `.blocks_only`, or `.combined` classmethods to
    construct a builder, then call `.build(p)` to materialize the bank.

    All banks always include `IdentityOperator` first, matching the existing
    `bench/AOM_v0/aompls/estimators.py:_resolve_bank` convention. All
    operators are required to be strict-linear; `ViewBuilder.build` raises
    if a non-strict-linear operator is encountered (e.g. via a custom user
    bank).
    """

    mode: str
    bank_name: Optional[str] = None
    K: Optional[int] = None
    strategy: Optional[str] = None
    include_global: bool = False

    @classmethod
    def preproc_only(cls, bank_name: str = "compact") -> "ViewBuilder":
        return cls(mode="preproc_only", bank_name=bank_name)

    @classmethod
    def blocks_only(cls, K: int = 3, strategy: str = "equal_width") -> "ViewBuilder":
        return cls(mode="blocks_only", K=K, strategy=strategy)

    @classmethod
    def combined(
        cls,
        bank_name: str = "compact",
        K: int = 3,
        strategy: str = "equal_width",
        include_global: bool = True,
    ) -> "ViewBuilder":
        return cls(
            mode="combined",
            bank_name=bank_name,
            K=K,
            strategy=strategy,
            include_global=include_global,
        )

    def build(self, p: int) -> List[LinearSpectralOperator]:
        if not isinstance(p, (int, np.integer)) or int(p) <= 0:
            raise ValueError(f"p must be a positive int, got {p!r}")
        p = int(p)

        if self.mode == "preproc_only":
            preproc_bank = list(bank_by_name(self.bank_name, p=p))
            self._enforce_strict_linear(preproc_bank)
            return self._ensure_identity_first(preproc_bank, p)

        if self.mode == "blocks_only":
            blocks = _resolve_blocks(self.K, p, self.strategy)
            bank: List[LinearSpectralOperator] = [IdentityOperator(p=p)]
            for start, end in blocks:
                bank.append(BlockMaskOperator(start=start, end=end, p=p))
            return bank

        if self.mode == "combined":
            blocks = _resolve_blocks(self.K, p, self.strategy)
            preproc_bank = list(bank_by_name(self.bank_name, p=p))
            self._enforce_strict_linear(preproc_bank)
            non_identity_preproc = [
                op for op in preproc_bank if not isinstance(op, IdentityOperator)
            ]
            bank = [IdentityOperator(p=p)]
            for start, end in blocks:
                bank.append(BlockMaskOperator(start=start, end=end, p=p))
            if self.include_global:
                bank.extend(non_identity_preproc)
            for start, end in blocks:
                mask = BlockMaskOperator(start=start, end=end, p=p)
                for op in non_identity_preproc:
                    composed = ComposedOperator(
                        [op, mask], name=f"{mask.name}__{op.name}"
                    )
                    bank.append(composed)
            return bank

        raise ValueError(f"unknown mode: {self.mode!r}")

    @staticmethod
    def _enforce_strict_linear(bank: Sequence[LinearSpectralOperator]) -> None:
        for op in bank:
            if not getattr(op, "is_strict_linear", False):
                raise ValueError(
                    f"operator {op.name!r} is not strict-linear; "
                    "ViewBuilder requires strict-linear operators only"
                )

    @staticmethod
    def _ensure_identity_first(
        bank: Sequence[LinearSpectralOperator], p: int
    ) -> List[LinearSpectralOperator]:
        ops = list(bank)
        if any(isinstance(op, IdentityOperator) for op in ops):
            return ops
        return [IdentityOperator(p=p)] + ops
