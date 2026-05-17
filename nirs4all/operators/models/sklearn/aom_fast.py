"""FastAOM public entry point for `nirs4all`.

Re-exports the canonical FastAOM implementation from
``nirs4all.operators.models._aom_nirs.fast`` (vendored copy of the
``aom-nirs`` package; once ``aom-nirs`` is on PyPI this file will
switch to ``from aom_nirs.fast import ...``).

FastAOM screens millions of preprocessing chains via adjoint-only
covariance scoring + diversity-aware top-k + low-rank kernel
evaluation, then fits one of four AOM-style models on the survivors.
The paper-headline FastAOM variant is ``SparseMultiKernelRidge``
(``FastAOMPLSRidge(model='sparse_mkr_compact')``) — median rel-RMSEP
1.022 with 2.48 s median fit time on the 50-dataset cohort.
"""

from __future__ import annotations

from nirs4all.operators.models._aom_nirs.fast import (
    AbsorbanceBase,
    ASLSBase,
    BaseTransform,
    ChainGenerationConfig,
    ChainGrammar,
    EMSCBase,
    FastAOMConfig,
    FastAOMPLSRidge,
    HardAOMChainPLSRidge,
    LowRankBase,
    MSCBase,
    OperatorChain,
    OSCBase,
    RawBase,
    ScreeningCandidate,
    SingleChainPLSRidge,
    SNVBase,
    SNVOSCBase,
    SoftAOMChainPLSRidge,
    SparseMultiKernelRidge,
    WhittakerBaseLine,
    build_base_bank,
    chain_from_operators,
    default_grammar,
    diversity_topk,
    fast_covariance_screen,
    fit_lowrank_bases,
    generate_chains,
)

__all__ = [
    # models
    "FastAOMPLSRidge",
    "FastAOMConfig",
    "SingleChainPLSRidge",
    "HardAOMChainPLSRidge",
    "SoftAOMChainPLSRidge",
    "SparseMultiKernelRidge",
    # screening
    "ScreeningCandidate",
    "fast_covariance_screen",
    "diversity_topk",
    # bases
    "BaseTransform",
    "RawBase",
    "AbsorbanceBase",
    "SNVBase",
    "MSCBase",
    "EMSCBase",
    "ASLSBase",
    "OSCBase",
    "SNVOSCBase",
    "WhittakerBaseLine",
    "build_base_bank",
    # grammar / chains
    "ChainGrammar",
    "default_grammar",
    "ChainGenerationConfig",
    "generate_chains",
    "OperatorChain",
    "chain_from_operators",
    # low-rank
    "LowRankBase",
    "fit_lowrank_bases",
]
