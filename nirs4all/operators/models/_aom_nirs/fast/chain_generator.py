"""Enumerate operator chains under the FastAOM grammar.

Given a primitive bank (e.g. the parent ``aompls.compact_bank``) and a
:class:`ChainGrammar`, this module enumerates all chains up to a maximum
depth and de-duplicates them by stable signature. Identity-only chains
are kept (they correspond to the "no preprocessing" branch) and reduce
trivially to PLS-Ridge in the downstream models.

The enumeration is depth-first with grammar-driven pruning, so it does
not generate chains that would be rejected (no need to materialise huge
intermediate lists). A simple beam-search option caps the breadth of
each level for very large primitive banks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from nirs4all.operators.models._aom_nirs.pls.operators import IdentityOperator, LinearSpectralOperator
from .grammar import ChainGrammar, default_grammar
from .operator_chain import OperatorChain


@dataclass
class ChainGenerationConfig:
    """Parameters for chain enumeration.

    Attributes:
        max_depth: Maximum chain depth (overrides the grammar's max_depth
            if smaller).
        include_identity_chain: If True, the chain ``[IdentityOperator]``
            is included (matches "no preprocessing").
        beam_width: Optional cap on the number of chains kept at each
            depth level. ``None`` means no cap. Useful when the primitive
            bank is very large.
        simplify_each: If True, every emitted chain is passed through
            :meth:`OperatorChain.simplify` before deduplication. This is
            usually a good idea because the grammar can still emit
            redundant chains (e.g. ``smoother -> derivative -> detrend``
            that collapses).
    """

    max_depth: int = 4
    include_identity_chain: bool = True
    beam_width: Optional[int] = None
    simplify_each: bool = True


def _generate_recursive(
    bank: Sequence[LinearSpectralOperator],
    grammar: ChainGrammar,
    cfg: ChainGenerationConfig,
    current: List[LinearSpectralOperator],
    out: List[Tuple[LinearSpectralOperator, ...]],
) -> None:
    if current:
        out.append(tuple(current))
    if len(current) >= cfg.max_depth:
        return
    if len(current) >= grammar.max_depth:
        return
    for op in bank:
        if not grammar.is_extension_valid(current, op):
            continue
        current.append(op)
        _generate_recursive(bank, grammar, cfg, current, out)
        current.pop()


def generate_chains(
    bank: Sequence[LinearSpectralOperator],
    grammar: Optional[ChainGrammar] = None,
    cfg: Optional[ChainGenerationConfig] = None,
    feature_dim: Optional[int] = None,
) -> List[OperatorChain]:
    """Enumerate :class:`OperatorChain` instances under ``grammar``.

    Args:
        bank: Primitive operator bank to draw from. Each operator should
            already be initialised for the spectral dimension (use
            ``aompls.fit_bank`` if needed).
        grammar: Chain grammar (default: :func:`default_grammar`).
        cfg: Generation parameters.
        feature_dim: Optional override for the feature dimensionality
            (the grammar treats the bank as already shaped, but identity
            chains need a value of ``p`` to bind to).

    Returns:
        Deduplicated list of :class:`OperatorChain`, sorted by depth
        then by signature for reproducibility.
    """
    grammar = grammar or default_grammar()
    cfg = cfg or ChainGenerationConfig()

    raw_chains: List[Tuple[LinearSpectralOperator, ...]] = []
    _generate_recursive(bank, grammar, cfg, current=[], out=raw_chains)

    chains: List[OperatorChain] = []
    if cfg.include_identity_chain:
        chains.append(OperatorChain([IdentityOperator(p=feature_dim)]))

    for tup in raw_chains:
        chain = OperatorChain(tup)
        if cfg.simplify_each:
            chain = chain.simplify()
        chains.append(chain)

    # Beam pruning by depth (keep first ``beam_width`` per depth)
    if cfg.beam_width is not None:
        by_depth: Dict[int, List[OperatorChain]] = {}
        for chain in chains:
            by_depth.setdefault(chain.depth(), []).append(chain)
        pruned: List[OperatorChain] = []
        for depth in sorted(by_depth.keys()):
            pruned.extend(by_depth[depth][: cfg.beam_width])
        chains = pruned

    # Deduplicate by signature, preserving first-seen order then sorting
    seen: Dict[str, OperatorChain] = {}
    for chain in chains:
        seen.setdefault(chain.signature, chain)
    deduped = sorted(seen.values(), key=lambda c: (c.depth(), c.signature))
    return deduped
