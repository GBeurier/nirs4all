"""Tests for the FastAOM chain grammar and generator."""

from __future__ import annotations

import pytest

from aompls.banks import compact_bank
from aompls.operators import (
    DetrendProjectionOperator,
    FiniteDifferenceOperator,
    IdentityOperator,
    SavitzkyGolayOperator,
)
from FastAOM.chain_generator import (
    ChainGenerationConfig,
    generate_chains,
)
from FastAOM.grammar import (
    ChainGrammar,
    ROLE_BASELINE,
    ROLE_DERIVATIVE,
    ROLE_PROJECTION,
    ROLE_SMOOTHING,
    default_grammar,
    role_of,
)


@pytest.fixture
def feature_dim() -> int:
    return 64


@pytest.fixture
def bank(feature_dim: int):
    return compact_bank(p=feature_dim)


def test_role_assignments() -> None:
    sg_smooth = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0, p=32)
    sg_d1 = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=32)
    detrend = DetrendProjectionOperator(degree=1, p=32)
    fd = FiniteDifferenceOperator(order=1, p=32)
    assert role_of(sg_smooth) == ROLE_SMOOTHING
    assert role_of(sg_d1) == ROLE_DERIVATIVE
    assert role_of(detrend) == ROLE_PROJECTION
    assert role_of(fd) == ROLE_DERIVATIVE


def test_default_grammar_forbids_double_derivative(bank) -> None:
    grammar = default_grammar(max_depth=3)
    # Chain ending with one derivative -> any second derivative-role op should fail
    sg_d1 = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=64)
    sg_d2 = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=2, p=64)
    chain = [sg_d1]
    assert not grammar.is_extension_valid(chain, sg_d2)


def test_default_grammar_forbids_double_smoother(bank) -> None:
    grammar = default_grammar(max_depth=3)
    sg_sm_a = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0, p=64)
    sg_sm_b = SavitzkyGolayOperator(window_length=21, polyorder=2, deriv=0, p=64)
    assert not grammar.is_extension_valid([sg_sm_a], sg_sm_b)


def test_default_grammar_forbids_derivative_then_baseline(bank) -> None:
    grammar = default_grammar(max_depth=4)
    sg_d1 = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=64)
    # Whittaker is mapped to smoothing role; let's manually test that derivative -> baseline pair
    # is impossible by emitting role pair (derivative -> baseline).
    # In the default mapping there's no direct baseline operator, so we test the role logic via
    # an artificial setup using identity (which maps to ROLE_BASELINE in our mapping).
    identity = IdentityOperator(p=64)
    assert not grammar.is_extension_valid([sg_d1], identity)


def test_generator_returns_at_least_identity(bank, feature_dim) -> None:
    chains = generate_chains(bank, feature_dim=feature_dim)
    assert len(chains) >= 1
    # Identity chain should be present
    identity_present = any(c.depth() == 1 and c.operators[0].name == "identity" for c in chains)
    assert identity_present


def test_generator_respects_max_depth(bank, feature_dim) -> None:
    cfg = ChainGenerationConfig(max_depth=2, include_identity_chain=True)
    chains = generate_chains(bank, default_grammar(max_depth=2), cfg, feature_dim=feature_dim)
    assert all(c.depth() <= 2 for c in chains)
    # We should have more than just identity given the bank size
    assert len(chains) > 1


def test_generator_dedup_signatures(bank, feature_dim) -> None:
    chains = generate_chains(bank, feature_dim=feature_dim)
    signatures = [c.signature for c in chains]
    assert len(signatures) == len(set(signatures))


def test_generator_beam_caps_breadth(bank, feature_dim) -> None:
    cfg_no_cap = ChainGenerationConfig(max_depth=3, include_identity_chain=True)
    cfg_capped = ChainGenerationConfig(max_depth=3, include_identity_chain=True, beam_width=3)
    chains_no_cap = generate_chains(bank, default_grammar(max_depth=3), cfg_no_cap, feature_dim=feature_dim)
    chains_capped = generate_chains(bank, default_grammar(max_depth=3), cfg_capped, feature_dim=feature_dim)
    assert len(chains_capped) <= len(chains_no_cap)


def test_grammar_per_position_constraint(bank, feature_dim) -> None:
    """Restrict position 0 to smoothing role only; chain must start with smoother or identity-only chain."""
    allowed = (frozenset({ROLE_SMOOTHING}), frozenset({ROLE_DERIVATIVE}))
    grammar = ChainGrammar(
        max_depth=2,
        allowed_roles_per_position=allowed,
        allow_repeats_in_role=False,
        max_per_role={ROLE_SMOOTHING: 1, ROLE_DERIVATIVE: 1},
    )
    cfg = ChainGenerationConfig(max_depth=2, include_identity_chain=False)
    chains = generate_chains(bank, grammar, cfg, feature_dim=feature_dim)
    for chain in chains:
        roles = [role_of(op) for op in chain.operators]
        if len(roles) >= 1:
            assert roles[0] == ROLE_SMOOTHING
        if len(roles) >= 2:
            assert roles[1] == ROLE_DERIVATIVE


def test_generator_simplify_each_collapses(bank, feature_dim) -> None:
    """Simplification should collapse chains that the grammar accepts but are redundant."""
    cfg_no_simp = ChainGenerationConfig(max_depth=3, include_identity_chain=False, simplify_each=False)
    cfg_simp = ChainGenerationConfig(max_depth=3, include_identity_chain=False, simplify_each=True)
    chains_no_simp = generate_chains(bank, default_grammar(max_depth=3), cfg_no_simp, feature_dim=feature_dim)
    chains_simp = generate_chains(bank, default_grammar(max_depth=3), cfg_simp, feature_dim=feature_dim)
    # Simplification merges duplicates (same signature), so we have fewer or equal chains
    assert len(chains_simp) <= len(chains_no_simp)
