"""Beam search over operator chains in covariance space.

Three-tier architecture:

- **Candidate bank**: large primitive grid generated from `operator_generation`.
- **Active bank**: small, fold-specific bank produced by beam search and
  similarity pruning (this module).
- **Selection bank**: the active bank passed to exact AOM/POP CV/PRESS
  selection.

The beam search:

1. Starts from a single state with the empty chain and `response = S`.
2. At each depth, expands every state by every allowed primitive (`grammar_allows`).
3. Computes `new_response = op.apply_cov(state.response)` lazily.
4. Scores `-||response|| / gain`.
5. Keeps the top diverse states using probe-based or response-based cosine.

The final active bank is wrapped into `ComposedOperator` instances so it can
be passed unmodified to the existing AOM/POP estimators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .operator_generation import (
    canonicalize,
    chain_signature,
    family_signature,
    grammar_allows,
    primitive_bank,
)
from .operator_similarity import keep_top_diverse, response_cosine
from .operators import ComposedOperator, IdentityOperator, LinearSpectralOperator


@dataclass
class CandidateState:
    """A single beam-search state."""

    chain: Tuple[LinearSpectralOperator, ...]
    response: np.ndarray
    score: float = 0.0
    gain: float = 1.0
    family_signature: str = ""
    signature: str = ""

    def to_operator(self) -> LinearSpectralOperator:
        if not self.chain:
            return IdentityOperator()
        if len(self.chain) == 1:
            return self.chain[0]
        return ComposedOperator(list(self.chain))


def _normalised_score(response: np.ndarray, gain: float) -> float:
    norm = float(np.linalg.norm(response))
    if gain < 1e-9:
        return 0.0
    return -norm / (gain + 1e-9)


def _operator_gain(op: LinearSpectralOperator, p: int, dim: int) -> float:
    """Cheap gain proxy: norm of the operator response on a single Gaussian probe."""
    rng = np.random.default_rng(7)
    z = rng.standard_normal(p)
    z = z / (np.linalg.norm(z) + 1e-12)
    out = op.apply_cov(z)
    return float(np.linalg.norm(out))


def _chain_gain(chain: Tuple[LinearSpectralOperator, ...], p: int, dim: int) -> float:
    if not chain:
        return 1.0
    rng = np.random.default_rng(13)
    z = rng.standard_normal(p)
    z = z / (np.linalg.norm(z) + 1e-12)
    out = z
    for op in chain:
        out = op.apply_cov(out)
    g = float(np.linalg.norm(out))
    return max(g, 1e-9)


def explore_active_bank(
    S: np.ndarray,
    primitive_ops: Sequence[LinearSpectralOperator],
    max_degree: int = 2,
    beam_width: int = 32,
    final_top_m: int = 20,
    cosine_threshold: float = 0.98,
    per_family_limit: int = 4,
    include_identity: bool = True,
) -> List[LinearSpectralOperator]:
    """Run a deterministic beam search over operator chains.

    Returns a list of `LinearSpectralOperator` instances (single primitives or
    `ComposedOperator` chains) suitable for passing to AOM / POP.

    The search is deterministic given the inputs (no random pruning).
    """
    if S.ndim == 1:
        S2 = S.reshape(-1, 1)
    else:
        S2 = S
    p, q = S2.shape
    # Ensure operators are bound to p.
    for op in primitive_ops:
        op.fit(np.zeros((1, p)))
    # Initial state: identity (chain=())
    initial_resp = S2
    initial_gain = 1.0
    initial = CandidateState(
        chain=(),
        response=initial_resp,
        score=_normalised_score(initial_resp, initial_gain),
        gain=initial_gain,
        family_signature="identity",
        signature="identity",
    )
    # Depth 1: every primitive applied to S
    states: List[CandidateState] = [initial]
    kept_by_depth: List[CandidateState] = [initial] if include_identity else []
    seen_signatures = {initial.signature}
    for depth in range(1, max_degree + 1):
        candidates: List[CandidateState] = []
        for state in states:
            for op in primitive_ops:
                if not grammar_allows(state.chain, op):
                    continue
                new_chain = canonicalize(state.chain + (op,))
                if not new_chain:
                    continue
                sig = chain_signature(new_chain)
                if sig in seen_signatures:
                    continue
                response = op.apply_cov(state.response)
                gain = state.gain * (_operator_gain(op, p, q) + 1e-9)
                score = _normalised_score(response, gain)
                candidates.append(
                    CandidateState(
                        chain=new_chain,
                        response=response,
                        score=score,
                        gain=gain,
                        family_signature=family_signature(op),
                        signature=sig,
                    )
                )
                seen_signatures.add(sig)
        # Family quotas + diversity
        candidates_sorted = sorted(candidates, key=lambda s: s.score)
        bucketed: List[CandidateState] = []
        family_counts: dict = {}
        for cand in candidates_sorted:
            if family_counts.get(cand.family_signature, 0) >= per_family_limit:
                continue
            bucketed.append(cand)
            family_counts[cand.family_signature] = family_counts.get(cand.family_signature, 0) + 1
            if len(bucketed) >= beam_width * 4:
                break
        diverse_input = [(c.score, c.response, c) for c in bucketed]
        diverse = keep_top_diverse(diverse_input, top_m=beam_width, cosine_threshold=cosine_threshold)
        states = [payload for _, _, payload in diverse]
        kept_by_depth.extend(states)
    final_input = [(c.score, c.response, c) for c in kept_by_depth]
    final = keep_top_diverse(final_input, top_m=final_top_m, cosine_threshold=cosine_threshold)
    out: List[LinearSpectralOperator] = []
    seen_outputs = set()
    for _, _, c in final:
        sig = chain_signature(c.chain)
        if sig in seen_outputs:
            continue
        seen_outputs.add(sig)
        out.append(c.to_operator())
    if include_identity and not any(family_signature(op) == "identity" for op in out):
        out.insert(0, IdentityOperator())
    return out


def build_active_bank_from_training(
    X: np.ndarray,
    Y: np.ndarray,
    max_degree: int = 2,
    beam_width: int = 32,
    final_top_m: int = 20,
    cosine_threshold: float = 0.98,
    per_family_limit: int = 4,
    primitive_ops: Optional[Sequence[LinearSpectralOperator]] = None,
) -> List[LinearSpectralOperator]:
    """Build a fold-specific active bank from `(X_train, Y_train)`.

    The cross-covariance `S = X^T Y` is the only data leak point: callers
    must pass training data only.
    """
    Xc = np.asarray(X, dtype=float)
    Yc = np.asarray(Y, dtype=float)
    if Yc.ndim == 1:
        Yc = Yc.reshape(-1, 1)
    p = Xc.shape[1]
    if primitive_ops is None:
        primitive_ops = primitive_bank(p)
    Xc_centered = Xc - Xc.mean(axis=0)
    Yc_centered = Yc - Yc.mean(axis=0)
    S = Xc_centered.T @ Yc_centered
    return explore_active_bank(
        S=S,
        primitive_ops=primitive_ops,
        max_degree=max_degree,
        beam_width=beam_width,
        final_top_m=final_top_m,
        cosine_threshold=cosine_threshold,
        per_family_limit=per_family_limit,
    )
