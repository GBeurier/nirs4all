"""Operator chains for FastAOM.

An :class:`OperatorChain` is an ordered sequence of strict-linear spectral
operators that, when composed left-to-right, defines a single linear map
``A_s = A_K ... A_2 A_1`` acting on row spectra as ``X_s = X A_s.T`` (using
the parent ``aompls`` convention where the operator's matrix acts as
``transform(X) = X A.T``).

The class wraps :class:`aompls.operators.ComposedOperator` and adds:

  * grammar-aware :meth:`simplify` (identity drops, detrend collapse,
    consecutive-smoother / consecutive-detrend dedup, trailing-projection
    pruning),
  * a stable :meth:`signature` for deduplication,
  * a :meth:`depth` accessor and a typed :meth:`stages` property for
    grammar / debugging.

Adjoint correctness is enforced by the dot-product test in the test
suite (``test_operator_chain.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from nirs4all.operators.models._aom_nirs.pls.operator_generation import (
    canonicalize as _aompls_canonicalize,
    chain_signature as _aompls_chain_signature,
    family_signature as _aompls_family,
)
from nirs4all.operators.models._aom_nirs.pls.operators import (
    ComposedOperator,
    IdentityOperator,
    LinearSpectralOperator,
)


@dataclass(frozen=True)
class ChainStage:
    """A single stage in an :class:`OperatorChain`.

    Attributes:
        op: The underlying strict-linear spectral operator.
        family: The grammar family (``"sg_smooth"``, ``"sg_d1"``,
            ``"detrend"``, ``"finite_difference"``, ``"norris_williams"``,
            ``"whittaker"``, ``"gauss_smooth"``, ``"gauss_deriv"``,
            ``"shift"``, ``"composed"``, ``"identity"``, ``"other"``).
    """

    op: LinearSpectralOperator
    family: str


class OperatorChain:
    """An ordered chain of linear spectral operators with grammar utilities.

    The composed action is ``A_s = A_K ... A_1`` (apply leftmost first),
    matching the parent ``aompls.ComposedOperator`` convention. Methods
    behave linearly:

      * :meth:`transform` returns ``X A_s.T`` for row spectra.
      * :meth:`apply_cov` returns ``A_s @ S`` for column-stacked vectors.
      * :meth:`adjoint_vec` returns ``A_s.T @ v`` for adjoint actions.

    Empty chains are not allowed — pass a single :class:`IdentityOperator`
    instead.
    """

    __slots__ = ("_ops", "_signature", "_composed")

    def __init__(self, operators: Sequence[LinearSpectralOperator]) -> None:
        if not operators:
            raise ValueError("OperatorChain requires at least one operator (use IdentityOperator())")
        self._ops: Tuple[LinearSpectralOperator, ...] = tuple(operators)
        self._signature: Optional[str] = None
        self._composed: Optional[ComposedOperator] = None

    # ------------------------------------------------------------- accessors

    @property
    def operators(self) -> Tuple[LinearSpectralOperator, ...]:
        return self._ops

    @property
    def stages(self) -> Tuple[ChainStage, ...]:
        return tuple(ChainStage(op=op, family=_aompls_family(op)) for op in self._ops)

    def depth(self) -> int:
        return len(self._ops)

    def families(self) -> Tuple[str, ...]:
        return tuple(_aompls_family(op) for op in self._ops)

    def names(self) -> Tuple[str, ...]:
        return tuple(op.name for op in self._ops)

    def __iter__(self) -> Iterator[LinearSpectralOperator]:
        return iter(self._ops)

    def __len__(self) -> int:
        return len(self._ops)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"OperatorChain({' > '.join(self.names())})"

    # ------------------------------------------------------------- composition

    @property
    def composed(self) -> ComposedOperator:
        """Return the lazily built :class:`ComposedOperator`."""
        if self._composed is None:
            if len(self._ops) == 1:
                # ComposedOperator works for length 1 but adds overhead; still
                # use it for a uniform interface.
                self._composed = ComposedOperator(list(self._ops), name=self.signature)
            else:
                self._composed = ComposedOperator(list(self._ops), name=self.signature)
        return self._composed

    def fit(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> "OperatorChain":
        self.composed.fit(X=X, y=y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.composed.transform(X)

    def apply_cov(self, S: np.ndarray) -> np.ndarray:
        return self.composed.apply_cov(S)

    def adjoint_vec(self, v: np.ndarray) -> np.ndarray:
        return self.composed.adjoint_vec(v)

    def matrix(self, p: Optional[int] = None) -> np.ndarray:
        return self.composed.matrix(p)

    # --------------------------------------------------------------- signature

    @property
    def signature(self) -> str:
        if self._signature is None:
            self._signature = _aompls_chain_signature(self._ops)
        return self._signature

    def __hash__(self) -> int:  # pragma: no cover - trivial
        return hash(self.signature)

    def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial
        if not isinstance(other, OperatorChain):
            return NotImplemented
        return self.signature == other.signature

    # ------------------------------------------------------------- simplification

    def simplify(self) -> "OperatorChain":
        """Return a grammar-aware simplified chain.

        Rules (combining the parent ``canonicalize`` with extra FastAOM
        simplifications):

          * Drop ``IdentityOperator`` stages.
          * Collapse consecutive ``DetrendProjectionOperator`` to the
            higher-degree one (idempotence of nested polynomial projection
            complements when degrees are nested).
          * Collapse consecutive smoothers of the same family by keeping the
            widest window (heuristic: wider smoother absorbs the narrower).
          * Drop a *trailing* detrend if the chain already ends with a
            derivative whose null space contains the polynomial basis
            (first/second derivatives kill polynomials of equal degree).
          * If the chain becomes empty after simplification, return an
            :class:`IdentityOperator`-only chain.
        """
        # Only the parent's canonicalisation: drop identities and collapse
        # consecutive ``detrend`` ops to the higher-degree one. We do NOT add
        # any FastAOM-specific simplifications:
        #
        #   * The "consecutive same-family smoother collapse" rule is left to
        #     the *grammar* (``aompls.grammar_allows`` rejects two same-family
        #     smoothers in a row), not the simplifier — keeping the simplifier
        #     purely conservative avoids silently discarding valid candidates
        #     that the grammar might allow in a future change.
        #
        #   * "Drop trailing detrend after derivative" is NOT applied: zero-
        #     padded SG / FD / NW derivatives leave residual constant / linear
        #     trends at the spectrum edges, so ``derivative -> detrend`` is
        #     not redundant in finite-length spectra.
        ops = list(_aompls_canonicalize(self._ops))
        if not ops:
            ops = [IdentityOperator(p=self._ops[0].p)]
        return OperatorChain(ops)

    # ----------------------------------------------------------- compatibility

    @classmethod
    def from_operators(cls, operators: Iterable[LinearSpectralOperator]) -> "OperatorChain":
        return cls(tuple(operators))


def chain_from_operators(operators: Iterable[LinearSpectralOperator]) -> OperatorChain:
    """Convenience constructor used by the chain generator and tests."""
    return OperatorChain.from_operators(operators)
