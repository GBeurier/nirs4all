"""Tests for ``aomridge.guards.check_no_selector_branch_leak``.

D-A-008 (see ``bench/AOM_v0/Ridge/docs/HEADLINE_SPXY3_NESTED_AUDIT.md`` §10).

The guard prevents a leakage path where a selector variant
(``auto_select`` / ``blender`` / ``residual_tabpfn``) declares
variant-level ``branch_preproc``. The runner would then fit the
preprocessor on the full training set before the selector's outer-CV
runs, leaking preprocessing state across the selector's folds.

These tests pin the contract:

1. A selector variant with non-empty ``branch_preproc`` raises ``ValueError``.
2. A selector variant with ``branch_preproc=None`` passes silently.
3. A non-selector variant with non-empty ``branch_preproc`` passes silently
   (current valid use case for ``superblock`` / ``global`` / ``ridge_pls``
   variants).
4. The error message names the variant label, the offending selection, the
   offending preproc, and points at the audit doc.
"""

from __future__ import annotations

import pytest
from aomridge.guards import SELECTOR_VARIANTS, check_no_selector_branch_leak

# ----------------------------------------------------------------------
# Contract — selector + branch_preproc must raise
# ----------------------------------------------------------------------


@pytest.mark.parametrize("selection", sorted(SELECTOR_VARIANTS))
@pytest.mark.parametrize("branch_preproc", ["snv", "msc", "asls", "emsc2", "osc"])
def test_selector_with_branch_preproc_raises(
    selection: str, branch_preproc: str
) -> None:
    with pytest.raises(ValueError, match="selector-level branch_preproc"):
        check_no_selector_branch_leak(
            label=f"AOMRidge-{selection}-{branch_preproc}",
            selection=selection,
            branch_preproc=branch_preproc,
        )


# ----------------------------------------------------------------------
# Contract — selector with no branch_preproc must pass
# ----------------------------------------------------------------------


@pytest.mark.parametrize("selection", sorted(SELECTOR_VARIANTS))
def test_selector_without_branch_preproc_passes(selection: str) -> None:
    # Both None and empty string should pass; runner treats falsy as "no preproc".
    check_no_selector_branch_leak(
        label=f"AOMRidge-{selection}-none",
        selection=selection,
        branch_preproc=None,
    )
    check_no_selector_branch_leak(
        label=f"AOMRidge-{selection}-empty",
        selection=selection,
        branch_preproc="",
    )


# ----------------------------------------------------------------------
# Contract — non-selector variants with branch_preproc must pass
# ----------------------------------------------------------------------


# Enumeration of every ``selection`` value used in the runner's variant
# tables (HEADLINE_VARIANTS + LEAN_VARIANTS + SMOKE_VARIANTS + FULL_VARIANTS),
# minus the three SELECTOR_VARIANTS. Sourced by grep on
# ``run_aomridge_benchmark.py`` (D-A-008 round 2 review). Update this list
# when a new ``selection=`` literal lands; the union assertion below will
# fail and force the test author to confirm the new value's classification.
NON_SELECTOR_VARIANTS: frozenset[str] = frozenset({
    "superblock",
    "global",
    "branch_global",
    "active_superblock",
    "ridge_pls",
    "aom_pls",
    "multi_branch_mkl",
    "local_ridge",
})


@pytest.mark.parametrize("selection", sorted(NON_SELECTOR_VARIANTS))
@pytest.mark.parametrize("branch_preproc", ["snv", "msc", "asls"])
def test_non_selector_variant_with_branch_preproc_passes(
    selection: str, branch_preproc: str
) -> None:
    check_no_selector_branch_leak(
        label=f"AOMRidge-{selection}-{branch_preproc}",
        selection=selection,
        branch_preproc=branch_preproc,
    )


# ----------------------------------------------------------------------
# Error message must be actionable
# ----------------------------------------------------------------------


def test_error_message_names_label_selection_preproc_and_doc() -> None:
    label = "AOMRidge-AutoSelect-headline-spxy3-with-leaky-snv"
    with pytest.raises(ValueError) as excinfo:
        check_no_selector_branch_leak(
            label=label, selection="auto_select", branch_preproc="snv"
        )
    msg = str(excinfo.value)
    assert label in msg
    assert "auto_select" in msg
    assert "snv" in msg
    assert "HEADLINE_SPXY3_NESTED_AUDIT.md" in msg
    assert "D-A-008" in msg


# ----------------------------------------------------------------------
# Constant — frozen and includes the three known selectors
# ----------------------------------------------------------------------


def test_selector_variants_constant() -> None:
    assert isinstance(SELECTOR_VARIANTS, frozenset)
    assert {"auto_select", "blender", "residual_tabpfn"} == SELECTOR_VARIANTS


# ----------------------------------------------------------------------
# Exhaustiveness — the union of selectors and known non-selectors must
# match every ``selection=`` literal used in the runner's Variant tables.
# This protects against a future variant landing with a typo
# (e.g. ``"auto-select"`` with hyphen) that the guard would silently miss.
# ----------------------------------------------------------------------


def test_selection_universe_is_exhaustively_classified() -> None:
    import re
    from pathlib import Path

    runner_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "run_aomridge_benchmark.py"
    )
    source = runner_path.read_text()

    # Collect every ``selection="..."`` literal used to construct Variant.
    pattern = re.compile(r'selection\s*=\s*"([^"]+)"')
    used_selections = set(pattern.findall(source))

    universe = SELECTOR_VARIANTS | NON_SELECTOR_VARIANTS
    unclassified = used_selections - universe
    assert not unclassified, (
        f"Runner uses ``selection=`` literals that are neither in "
        f"SELECTOR_VARIANTS nor in NON_SELECTOR_VARIANTS: "
        f"{sorted(unclassified)}. Add each to one of the two sets after "
        f"checking whether the new variant runs an internal outer-CV "
        f"(SELECTOR_VARIANTS) or runs as a flat estimator "
        f"(NON_SELECTOR_VARIANTS)."
    )
    dead = universe - used_selections
    assert not dead, (
        f"SELECTOR_VARIANTS / NON_SELECTOR_VARIANTS list dead entries "
        f"that no Variant in the runner currently declares as ``selection=``: "
        f"{sorted(dead)}. Either remove the dead entries or add a Variant "
        f"that exercises them so the guard's contract stays meaningful."
    )
    # Bidirectional equality: the universe matches the runner exactly.
    assert universe == used_selections, (
        f"Universe mismatch: universe={sorted(universe)} "
        f"runner={sorted(used_selections)}"
    )
    # Disjointness: a selection cannot belong to both classes.
    assert not (SELECTOR_VARIANTS & NON_SELECTOR_VARIANTS), (
        "SELECTOR_VARIANTS and NON_SELECTOR_VARIANTS must be disjoint."
    )


# ----------------------------------------------------------------------
# Escape hatch — opt-in flag must allow selector-level branch_preproc
# (for future Codex-approved dataset-level preprocessing variants).
# ----------------------------------------------------------------------


@pytest.mark.parametrize("selection", sorted(SELECTOR_VARIANTS))
def test_escape_hatch_allows_selector_branch_preproc(selection: str) -> None:
    # Without the flag → raises.
    with pytest.raises(ValueError, match="selector-level branch_preproc"):
        check_no_selector_branch_leak(
            label="dataset-level-on-purpose",
            selection=selection,
            branch_preproc="signal_type_detect",
        )
    # With the flag → no raise.
    check_no_selector_branch_leak(
        label="dataset-level-on-purpose",
        selection=selection,
        branch_preproc="signal_type_detect",
        allow_selector_level_branch_preproc=True,
    )
