"""Operator bank presets.

A bank is a list of `LinearSpectralOperator` instances. Each preset starts
from the identity (so AOM/POP can always reduce to standard PLS) and adds a
small, well-behaved set of strict linear operators relevant to NIRS.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .operators import (
    ComposedOperator,
    DetrendProjectionOperator,
    FiniteDifferenceOperator,
    IdentityOperator,
    LinearSpectralOperator,
    NorrisWilliamsOperator,
    SavitzkyGolayOperator,
    WhittakerOperator,
)


def compact_bank(p: Optional[int] = None) -> List[LinearSpectralOperator]:
    """Small, well-tested bank suitable for benchmark runs and smoke tests.

    Contains identity, two SG smoothers, two SG first-derivatives, one
    SG second-derivative, two detrend projections, and two finite-difference
    operators. All entries are cheap, strictly linear, and zero-padded.
    """
    return [
        IdentityOperator(p=p),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0, p=p),
        SavitzkyGolayOperator(window_length=21, polyorder=3, deriv=0, p=p),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=p),
        SavitzkyGolayOperator(window_length=21, polyorder=3, deriv=1, p=p),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=2, p=p),
        DetrendProjectionOperator(degree=1, p=p),
        DetrendProjectionOperator(degree=2, p=p),
        FiniteDifferenceOperator(order=1, p=p),
    ]


def default_bank(p: Optional[int] = None) -> List[LinearSpectralOperator]:
    """Default benchmark bank — exact 1:1 mirror of the production nirs4all
    `default_operator_bank`.

    The bank has 100 operators:

    - 1 identity (always first);
    - 32 "savgol-family" entries: SG (deriv 0/1/2 across polyorder 2 and 3)
      + FD order 1/2 + 8 Norris-Williams variants;
    - 3 detrend projections (degree 0, 1, 2);
    - 64 composed operators (every savgol-family entry composed with
      detrend degree 0 and detrend degree 1).
    """
    savgol_family: List[LinearSpectralOperator] = []
    # SG smoothing
    for w in (11, 15, 21, 31):
        savgol_family.append(SavitzkyGolayOperator(window_length=w, polyorder=2, deriv=0, p=p))
    # SG first derivative, polyorder 2
    for w in (11, 15, 21, 31, 41):
        savgol_family.append(SavitzkyGolayOperator(window_length=w, polyorder=2, deriv=1, p=p))
    # SG second derivative, polyorder 2
    for w in (11, 15, 21, 31, 41):
        savgol_family.append(SavitzkyGolayOperator(window_length=w, polyorder=2, deriv=2, p=p))
    # SG second derivative, polyorder 3
    for w in (11, 15, 21, 31):
        savgol_family.append(SavitzkyGolayOperator(window_length=w, polyorder=3, deriv=2, p=p))
    # SG first derivative, polyorder 3
    for w in (11, 15, 21, 31):
        savgol_family.append(SavitzkyGolayOperator(window_length=w, polyorder=3, deriv=1, p=p))
    # Finite differences
    savgol_family.append(FiniteDifferenceOperator(order=1, p=p))
    savgol_family.append(FiniteDifferenceOperator(order=2, p=p))
    # Norris-Williams gap derivatives
    savgol_family.append(NorrisWilliamsOperator(gap=3, smoothing=1, order=1, p=p))
    savgol_family.append(NorrisWilliamsOperator(gap=5, smoothing=1, order=1, p=p))
    savgol_family.append(NorrisWilliamsOperator(gap=11, smoothing=1, order=1, p=p))
    savgol_family.append(NorrisWilliamsOperator(gap=5, smoothing=5, order=1, p=p))
    savgol_family.append(NorrisWilliamsOperator(gap=11, smoothing=5, order=1, p=p))
    savgol_family.append(NorrisWilliamsOperator(gap=5, smoothing=1, order=2, p=p))
    savgol_family.append(NorrisWilliamsOperator(gap=11, smoothing=1, order=2, p=p))
    savgol_family.append(NorrisWilliamsOperator(gap=5, smoothing=5, order=2, p=p))
    # Composed: every family entry composed with detrend(0) and detrend(1)
    detrend_d0 = lambda: DetrendProjectionOperator(degree=0, p=p)
    detrend_d1 = lambda: DetrendProjectionOperator(degree=1, p=p)
    composed_ops: List[LinearSpectralOperator] = []
    for sg in savgol_family:
        for dt_factory in (detrend_d0, detrend_d1):
            dt = dt_factory()
            composed_ops.append(ComposedOperator([sg, dt], name=f"compose({sg.name}|{dt.name})"))
    bank: List[LinearSpectralOperator] = [IdentityOperator(p=p)]
    bank.extend(savgol_family)
    bank.append(DetrendProjectionOperator(degree=0, p=p))
    bank.append(DetrendProjectionOperator(degree=1, p=p))
    bank.append(DetrendProjectionOperator(degree=2, p=p))
    bank.extend(composed_ops)
    return bank


def extended_bank(p: Optional[int] = None) -> List[LinearSpectralOperator]:
    """Extended bank: production-equivalent default + Whittaker smoothers."""
    bank = default_bank(p=p)
    bank.extend(
        [
            WhittakerOperator(lam=10.0, p=p),
            WhittakerOperator(lam=1e3, p=p),
            WhittakerOperator(lam=1e5, p=p),
        ]
    )
    return bank


def deep_bank(p: Optional[int] = None, max_degree: int = 3) -> List[LinearSpectralOperator]:
    """Bank with chains of compositions up to `max_degree` strict-linear primitives.

    Builds on the production-equivalent default bank by adding multi-stage
    chains (order >= 3) over the savgol family. This is the "order 3 or 4"
    deep bank requested by the benchmark protocol; it pairs smoothing,
    derivatives, and detrend in 3- or 4-step chains.

    The resulting bank has more candidates and is meant for ablation studies
    on how much extra capacity a deeper composition grammar buys.
    """
    if max_degree < 2:
        raise ValueError("deep_bank requires max_degree >= 2")
    base = default_bank(p=p)
    if max_degree == 2:
        return base
    # Pick a small set of "stage primitives" for chains of order >= 3.
    smoothers: List[LinearSpectralOperator] = [
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0, p=p),
        SavitzkyGolayOperator(window_length=21, polyorder=2, deriv=0, p=p),
    ]
    derivatives: List[LinearSpectralOperator] = [
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=p),
        SavitzkyGolayOperator(window_length=21, polyorder=2, deriv=1, p=p),
        SavitzkyGolayOperator(window_length=21, polyorder=3, deriv=2, p=p),
        FiniteDifferenceOperator(order=1, p=p),
    ]
    detrend = [
        DetrendProjectionOperator(degree=1, p=p),
        DetrendProjectionOperator(degree=2, p=p),
    ]
    deep_chains: List[LinearSpectralOperator] = []
    # Order 3: smoothing -> derivative -> detrend
    for sm in smoothers:
        for d in derivatives:
            for dt in detrend:
                deep_chains.append(
                    ComposedOperator([sm, d, dt], name=f"order3({sm.name}|{d.name}|{dt.name})")
                )
    if max_degree >= 4:
        # Order 4: detrend -> smoothing -> derivative -> detrend
        for dt0 in detrend:
            for sm in smoothers:
                for d in derivatives:
                    for dt1 in detrend:
                        if dt0.degree == dt1.degree:
                            continue  # canonicalise: skip detrend chained with itself
                        deep_chains.append(
                            ComposedOperator([dt0, sm, d, dt1],
                                             name=f"order4({dt0.name}|{sm.name}|{d.name}|{dt1.name})")
                        )
    return base + deep_chains


def family_pruned_default(p: Optional[int] = None, max_per_family: int = 2) -> List[LinearSpectralOperator]:
    """Default bank pruned to at most `max_per_family` operators per family.

    Family = first underscore-separated token of the operator name (e.g.
    `sg_smooth`, `sg_d1`, `sg_d2`, `fd`, `nw`, `detrend`, `compose`,
    `identity`). Within a family, operators are kept in their natural
    insertion order. This reduces the multiple-comparison bias by
    eliminating near-duplicate operators within each preprocessing
    family while keeping cross-family diversity.
    """
    base = default_bank(p=p)
    seen: dict = {}
    pruned: List[LinearSpectralOperator] = []
    for op in base:
        name = op.name
        family = name.split("(")[0].split("_")[0]
        # Group SG variants together by deriv: sg_d1, sg_d2, sg_smooth -> use full prefix
        if name.startswith("sg_smooth"):
            family = "sg_smooth"
        elif name.startswith("sg_d1"):
            family = "sg_d1"
        elif name.startswith("sg_d2"):
            family = "sg_d2"
        elif name.startswith("compose"):
            family = "compose"
        elif name.startswith("nw"):
            family = "nw"
        elif name.startswith("detrend"):
            family = "detrend"
        elif name.startswith("fd"):
            family = "fd"
        count = seen.get(family, 0)
        if count < max_per_family:
            pruned.append(op)
            seen[family] = count + 1
    return pruned


def response_dedup_default(
    p: Optional[int] = None, cosine_threshold: float = 0.995, random_state: int = 0
) -> List[LinearSpectralOperator]:
    """Default bank deduplicated by intrinsic-response cosine similarity.

    Builds the default bank, then prunes operators whose response on a
    random probe basis is within `cosine_threshold` of an already-kept
    operator. Reuses the existing
    `aompls.operator_similarity.prune_by_intrinsic_similarity`. This is
    a data-independent reduction (no y leak) that typically trims the
    100-operator default to 20-30 operators, killing the multiple-
    comparison inflation without touching cross-family diversity.
    """
    if p is None:
        p = 256  # arbitrary default if not provided
    from .operator_similarity import prune_by_intrinsic_similarity
    base = default_bank(p=p)
    return prune_by_intrinsic_similarity(
        base, p=p, cosine_threshold=cosine_threshold, random_state=random_state
    )


def bank_by_name(name: str, p: Optional[int] = None) -> List[LinearSpectralOperator]:
    """Resolve a bank preset by name."""
    name = name.lower()
    if name == "compact":
        return compact_bank(p)
    if name == "default":
        return default_bank(p)
    if name == "extended":
        return extended_bank(p)
    if name == "deep3":
        return deep_bank(p, max_degree=3)
    if name == "deep4":
        return deep_bank(p, max_degree=4)
    if name == "identity":
        return [IdentityOperator(p=p)]
    if name == "family_pruned":
        return family_pruned_default(p=p, max_per_family=2)
    if name == "response_dedup":
        return response_dedup_default(p=p, cosine_threshold=0.995)
    raise ValueError(f"unknown bank preset: {name!r}")


def fit_bank(bank: Sequence[LinearSpectralOperator], X, y=None) -> List[LinearSpectralOperator]:
    """Bind every operator in `bank` to the dimensionality of `X`."""
    fitted: List[LinearSpectralOperator] = []
    for op in bank:
        op.fit(X, y)
        fitted.append(op)
    return fitted
