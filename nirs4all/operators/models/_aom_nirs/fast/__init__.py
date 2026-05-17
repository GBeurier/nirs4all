"""FastAOM: Fast multi-operator AOM PLS-Ridge for NIRS regression.

The framework explores a very large space of preprocessing chains without
naively fitting one model per chain, by:

  1. Treating a chain of linear preprocessing operators as a single
     composed linear operator ``A_s`` so that ``X_s = X A_s.T`` (matching
     the parent ``aompls`` convention where ``transform(X) = X A.T``).
  2. Generating chains from a typed grammar over a few nonlinear bases
     (raw, SNV, MSC, EMSC, absorbance, ASLS-selected) up to depth 4.
  3. Screening millions of chains with adjoint-only covariance scores
     ``score(s) = ||A^T B(X)^T y||^2 / (||B(X) A.T||_F^2 ||y||^2)``
     where ``A = chain.matrix()``, ``B`` is the nonlinear base, and the
     denominator is approximated from a truncated SVD of ``B(X)``.
  4. Evaluating finalists with low-rank kernel-vector products via the
     SVD basis ``K_s ≈ U F.T F U.T`` where ``F = (A V) diag(S)``.
  5. Fitting one of four AOM-style models on the surviving banks:

       - :class:`SingleChainPLSRidge` — best chain + PLS-Ridge.
       - :class:`HardAOMChainPLSRidge` — one chain per latent component.
       - :class:`SoftAOMChainPLSRidge` — sparse non-negative chain mixture
         per latent component.
       - :class:`SparseMultiKernelRidge` — sparse non-negative kernel mixture
         ``K_θ = Σ θ_s K_s`` with Ridge.

The package is a sibling of ``bench/AOM_v0/aompls`` and reuses its
``LinearSpectralOperator`` subclasses, ``ComposedOperator``, NIPALS/SIMPLS
engines, and metrics module. New material is the chain grammar with
nonlinear bases, the fast screening with diversity, the low-rank
evaluator, and the four AOM models above.
"""

from __future__ import annotations

from .operator_chain import OperatorChain, chain_from_operators
from .bases import (
    BaseTransform,
    RawBase,
    AbsorbanceBase,
    SNVBase,
    MSCBase,
    EMSCBase,
    ASLSBase,
    OSCBase,
    SNVOSCBase,
    WhittakerBaseLine,
    build_base_bank,
)
from .grammar import ChainGrammar, default_grammar
from .chain_generator import generate_chains, ChainGenerationConfig

# The remaining modules are imported lazily so incremental development
# (writing operator infrastructure first, then screening / low-rank / models)
# does not break package-level imports.
try:  # pragma: no cover - import-time guard
    from .screening import (
        ScreeningCandidate,
        fast_covariance_screen,
        diversity_topk,
    )
except ImportError:  # pragma: no cover
    pass
try:  # pragma: no cover - import-time guard
    from .lowrank import LowRankBase, fit_lowrank_bases
except ImportError:  # pragma: no cover
    pass
try:  # pragma: no cover - import-time guard
    from .models import (
        SingleChainPLSRidge,
        HardAOMChainPLSRidge,
        SoftAOMChainPLSRidge,
        SparseMultiKernelRidge,
        FastAOMConfig,
        FastAOMPLSRidge,
    )
except ImportError:  # pragma: no cover
    pass

__version__ = "0.1.0"

__all__ = [
    # operator chain
    "OperatorChain",
    "chain_from_operators",
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
    # grammar
    "ChainGrammar",
    "default_grammar",
    # generation
    "generate_chains",
    "ChainGenerationConfig",
    # screening
    "ScreeningCandidate",
    "fast_covariance_screen",
    "diversity_topk",
    # low-rank
    "LowRankBase",
    "fit_lowrank_bases",
    # models
    "SingleChainPLSRidge",
    "HardAOMChainPLSRidge",
    "SoftAOMChainPLSRidge",
    "SparseMultiKernelRidge",
    "FastAOMConfig",
    "FastAOMPLSRidge",
]
