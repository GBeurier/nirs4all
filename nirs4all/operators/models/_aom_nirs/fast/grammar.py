"""Typed chain grammar for FastAOM.

The grammar groups primitive operators by their *role* in a spectral
preprocessing chain:

    BASELINE_SCATTER  : detrend / Whittaker / linear-baseline operators
    SMOOTHING         : Savitzky-Golay (deriv=0) / Gaussian (order=0) /
                        Whittaker (very large lambda)
    DERIVATIVE        : SG (deriv=1,2) / finite difference / Norris-Williams /
                        Gaussian-derivative
    PROJECTION_MASK   : detrend / FCK / shift / FFT-bandpass (used as
                        post-projection at the chain end)

A chain follows the rough ordering
``[BaselineScatter?] -> [Smoothing?] -> [Derivative?] -> [ProjectionMask?]``
which respects the heuristic that derivatives should follow smoothing,
and that a projection / mask is typically the last step. The grammar
accepts chains that violate the strict ordering as long as they obey the
parent ``aompls.operator_generation.grammar_allows`` rules (no two
consecutive smoothers of the same family, no double derivatives, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from nirs4all.operators.models._aom_nirs.pls.operator_generation import (
    family_signature as _family,
    grammar_allows as _aompls_grammar_allows,
)
from nirs4all.operators.models._aom_nirs.pls.operators import LinearSpectralOperator


# Role tags ------------------------------------------------------------------

ROLE_BASELINE = "baseline_scatter"
ROLE_SMOOTHING = "smoothing"
ROLE_DERIVATIVE = "derivative"
ROLE_PROJECTION = "projection_mask"

ALL_ROLES: Tuple[str, ...] = (ROLE_BASELINE, ROLE_SMOOTHING, ROLE_DERIVATIVE, ROLE_PROJECTION)


# Default family-to-role mapping --------------------------------------------

_DEFAULT_FAMILY_ROLE: Dict[str, str] = {
    "identity": ROLE_BASELINE,  # treated as no-op; allowed in any slot
    "detrend": ROLE_PROJECTION,
    "whittaker": ROLE_SMOOTHING,
    "sg_smooth": ROLE_SMOOTHING,
    "gauss_smooth": ROLE_SMOOTHING,
    "sg_d1": ROLE_DERIVATIVE,
    "sg_d2": ROLE_DERIVATIVE,
    "finite_difference": ROLE_DERIVATIVE,
    "norris_williams": ROLE_DERIVATIVE,
    "gauss_deriv": ROLE_DERIVATIVE,
    "shift": ROLE_PROJECTION,
    "composed": ROLE_SMOOTHING,  # a composed primitive is usually smoothing-like
    "other": ROLE_PROJECTION,
}


def role_of(op: LinearSpectralOperator) -> str:
    """Return the role tag for an operator (defaults to :data:`ROLE_PROJECTION`)."""
    family = _family(op)
    return _DEFAULT_FAMILY_ROLE.get(family, ROLE_PROJECTION)


# Grammar --------------------------------------------------------------------


@dataclass(frozen=True)
class ChainGrammar:
    """Typed chain grammar with depth and role constraints.

    Attributes:
        max_depth: Maximum chain depth (number of operators).
        allowed_roles_per_position: Optional ``(role-set,)`` per chain
            position. If ``None`` (default), any role is allowed at any
            position, subject to the parent ``aompls.grammar_allows``
            rules. If provided, the role at position ``i`` must be in
            ``allowed_roles_per_position[i]``.
        allow_repeats_in_role: If True, repeats of the same role
            (e.g. two consecutive smoothings of different families) are
            allowed. If False (default), each role appears at most once.
        forbidden_role_sequences: Pairs of consecutive role tags that are
            forbidden. By default we forbid ``(derivative -> baseline)``
            because a baseline-scatter step after a derivative is rarely
            helpful.
        max_per_role: Optional per-role cap. ``None`` means no cap.
    """

    max_depth: int = 4
    allowed_roles_per_position: Optional[Tuple[FrozenSet[str], ...]] = None
    allow_repeats_in_role: bool = False
    forbidden_role_sequences: Tuple[Tuple[str, str], ...] = ((ROLE_DERIVATIVE, ROLE_BASELINE),)
    max_per_role: Optional[Dict[str, int]] = None

    def is_extension_valid(
        self,
        chain: Sequence[LinearSpectralOperator],
        candidate: LinearSpectralOperator,
    ) -> bool:
        """Return True if appending ``candidate`` to ``chain`` is legal."""
        if len(chain) >= self.max_depth:
            return False

        # 1. Parent aompls grammar rules (no double smoothers/detrends/etc.)
        if not _aompls_grammar_allows(tuple(chain), candidate):
            return False

        cand_role = role_of(candidate)

        # 2. Allowed role at position constraint
        if self.allowed_roles_per_position is not None:
            pos = len(chain)
            if pos < len(self.allowed_roles_per_position):
                allowed = self.allowed_roles_per_position[pos]
                if cand_role not in allowed:
                    return False

        # 3. Repeats in role
        if not self.allow_repeats_in_role and chain:
            roles_so_far = [role_of(op) for op in chain]
            if cand_role in roles_so_far:
                return False

        # 4. Per-role caps
        if self.max_per_role:
            cap = self.max_per_role.get(cand_role)
            if cap is not None:
                count = sum(1 for op in chain if role_of(op) == cand_role) + 1
                if count > cap:
                    return False

        # 5. Forbidden role sequences
        if chain:
            last_role = role_of(chain[-1])
            if (last_role, cand_role) in self.forbidden_role_sequences:
                return False

        return True

    def chain_role_summary(self, chain: Sequence[LinearSpectralOperator]) -> Tuple[str, ...]:
        return tuple(role_of(op) for op in chain)


def default_grammar(max_depth: int = 4) -> ChainGrammar:
    """Return the default FastAOM grammar.

    No fixed role-per-position; up to ``max_depth`` operators; each role
    appears at most once (so we do not get two derivatives, two
    smoothings, etc.); the ``derivative -> baseline`` pair is forbidden.
    """
    return ChainGrammar(
        max_depth=max_depth,
        allowed_roles_per_position=None,
        allow_repeats_in_role=False,
        forbidden_role_sequences=((ROLE_DERIVATIVE, ROLE_BASELINE),),
        max_per_role={
            ROLE_BASELINE: 1,
            ROLE_SMOOTHING: 1,
            ROLE_DERIVATIVE: 1,
            ROLE_PROJECTION: 2,  # allow up to 2 projections (detrend + shift)
        },
    )
