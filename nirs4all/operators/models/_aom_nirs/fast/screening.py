"""Fast covariance screening of operator chains.

For each ``(base, chain)`` candidate, compute the dimensionless score

    score(j, s) = ||A_s^T B_j(X)^T y||^2 / (||B_j(X) A_s^T||_F^2 ||y||^2)

following the user's preprocessing convention ``X_s = X @ A_s``. Under
the parent codebase convention ``transform(X) = X @ A.T`` (where ``A``
is the operator's :meth:`matrix`), the user's ``A_s`` corresponds to
codebase's ``A.T``, so the user's ``A_s^T = codebase A``. The numerator
becomes

    ||A_s^T (B_j(X)^T y)||^2 = ||A @ g_0||^2 = ||chain.apply_cov(g_0)||^2

with ``g_0 = B_j(X)^T y``. The denominator is computed from the truncated
SVD ``B_j(X) ≈ U S V^T``: ``||B_j(X) A.T||_F^2 ≈ ||F_s||_F^2`` with
``F_s = A V diag(S) = chain.apply_cov(V * S[None, :])`` (shape ``p × r``).

The score is bounded in ``[0, 1]`` (Cauchy-Schwarz) and is a
preprocessing-invariant alignment between the latent direction of
``B_j(X) A_s^T`` and ``y``. Higher = more informative.

The :func:`diversity_topk` filter keeps the top-``k`` chains per
``(base, family)`` to avoid swamping the finalist pool with
near-duplicates from the same operator family (e.g. dozens of slightly
different SG-d1 chains).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from nirs4all.operators.models._aom_nirs.pls.operator_generation import family_signature as _family

from .lowrank import LowRankBase
from .operator_chain import OperatorChain


@dataclass
class ScreeningCandidate:
    """A scored ``(base, chain)`` candidate.

    Attributes:
        base_index: Index of the source base in the input bank.
        base_name: Convenience signature of the base.
        chain: The candidate chain.
        score: Screening score (``[0, 1]``).
        numerator: Raw numerator (``||A^T g_0||^2``); informative for
            debugging.
        denominator: Raw denominator (``||F_s||_F^2 * ||y||^2``).
        chain_norm_F_sq: Frobenius norm squared of the transformed
            matrix.
        family_tag: Coarse signature for the chain's "dominant" family,
            used by :func:`diversity_topk` for stratified selection.
    """

    base_index: int
    base_name: str
    chain: OperatorChain
    score: float
    numerator: float
    denominator: float
    chain_norm_F_sq: float
    family_tag: str = field(default="")


def _chain_family_tag(chain: OperatorChain) -> str:
    """Heuristic stratification tag for the chain.

    Concatenates the chain's operator families (e.g. ``"sg_smooth+sg_d1"``)
    so chains differing only in window length share the same tag, while
    structurally different chains stratify into different buckets.
    """
    fams = chain.families()
    return "+".join(fams) if fams else "identity"


def fast_covariance_screen(
    bases: Sequence[LowRankBase],
    chains: Sequence[OperatorChain],
    eps: float = 1e-12,
) -> List[ScreeningCandidate]:
    """Screen every ``(base, chain)`` pair and return all candidates with scores.

    Args:
        bases: Pre-fitted :class:`LowRankBase` list (one per nonlinear
            base).
        chains: Chains to screen.
        eps: Numerical floor for the denominator.

    Returns:
        Flat list of :class:`ScreeningCandidate`, sorted by descending
        score.
    """
    candidates: List[ScreeningCandidate] = []
    for j, base in enumerate(bases):
        y_norm_sq = float(np.dot(base.y_centred, base.y_centred))
        if y_norm_sq <= eps:
            # Constant-y fold: emit a single identity candidate with score 0 so
            # downstream orchestrators can still construct a fallback (mean)
            # model instead of crashing on empty finalists.
            for chain in chains:
                candidates.append(
                    ScreeningCandidate(
                        base_index=j,
                        base_name=base.base.signature,
                        chain=chain,
                        score=0.0,
                        numerator=0.0,
                        denominator=eps,
                        chain_norm_F_sq=base.chain_norm_F_sq(chain),
                        family_tag=_chain_family_tag(chain),
                    )
                )
            continue
        for chain in chains:
            # Numerator: ||A_s^T B_j(X)^T y||^2 (user's notation, X_s = X @ A_s).
            # Under codebase convention (transform = X @ A.T), this is
            # ||A @ g_0||^2 = ||chain.apply_cov(g_0)||^2.
            g_s = chain.apply_cov(base.g0)
            num = float(np.dot(g_s, g_s))
            # Denominator: ||F_s||_F^2 = ||A_s V diag(S)||_F^2
            chain_F = base.chain_norm_F_sq(chain)
            denom = chain_F * y_norm_sq + eps
            score = num / denom
            candidates.append(
                ScreeningCandidate(
                    base_index=j,
                    base_name=base.base.signature,
                    chain=chain,
                    score=score,
                    numerator=num,
                    denominator=denom,
                    chain_norm_F_sq=chain_F,
                    family_tag=_chain_family_tag(chain),
                )
            )
    candidates.sort(key=lambda c: -c.score)
    return candidates


def diversity_topk(
    candidates: Sequence[ScreeningCandidate],
    top_k_global: int,
    top_k_per_family: Optional[int] = None,
    top_k_per_base: Optional[int] = None,
) -> List[ScreeningCandidate]:
    """Return up to ``top_k_global`` candidates with diversity constraints.

    Args:
        candidates: Pre-scored candidates (typically sorted by score
            already; if not, this routine re-sorts a copy).
        top_k_global: Maximum number of returned candidates.
        top_k_per_family: Maximum number per family tag (e.g. cap how
            many ``sg_smooth+sg_d1`` chains we keep). ``None`` disables.
        top_k_per_base: Maximum number per base. ``None`` disables.

    Returns:
        Sorted list with at most ``top_k_global`` entries.
    """
    sorted_candidates = sorted(candidates, key=lambda c: -c.score)
    by_family: Dict[str, int] = {}
    by_base: Dict[int, int] = {}
    picked: List[ScreeningCandidate] = []
    for cand in sorted_candidates:
        if len(picked) >= top_k_global:
            break
        if top_k_per_family is not None:
            if by_family.get(cand.family_tag, 0) >= top_k_per_family:
                continue
        if top_k_per_base is not None:
            if by_base.get(cand.base_index, 0) >= top_k_per_base:
                continue
        picked.append(cand)
        by_family[cand.family_tag] = by_family.get(cand.family_tag, 0) + 1
        by_base[cand.base_index] = by_base.get(cand.base_index, 0) + 1
    return picked
