"""Runtime guards for the AOM-Ridge benchmark runner.

These guards exist to prevent silent leakage paths in the multi-variant
benchmark. They are intentionally cheap and run at the top of
``run_aomridge_benchmark._run_variant`` so that misconfigured variants fail
fast, before any estimator state is built.

Guards in this module enforce code-level invariants documented in
``bench/AOM_v0/Ridge/docs/HEADLINE_SPXY3_NESTED_AUDIT.md``.
"""

from __future__ import annotations

SELECTOR_VARIANTS: frozenset[str] = frozenset(
    {"auto_select", "blender", "residual_tabpfn"}
)
"""Variant ``selection`` values whose ``fit`` runs an internal outer-CV.

These selectors materialise their own folds inside ``fit``; any preprocessing
that should live inside those folds must be declared at the **candidate**
level (inside the spec dict's ``branch_preproc`` field), not at the
**variant** level (``Variant.branch_preproc``). The variant-level field is
fitted on the full training set in ``_run_variant`` before the selector
runs, which would leak preprocessing state across the selector's folds.
"""


def check_no_selector_branch_leak(
    label: str,
    selection: str,
    branch_preproc: str | None,
    *,
    allow_selector_level_branch_preproc: bool = False,
) -> None:
    """Raise ``ValueError`` if a selector variant declares variant-level branch_preproc.

    Args:
        label: Human-readable variant label (used in the error message).
        selection: ``Variant.selection`` value (e.g., ``"auto_select"``).
        branch_preproc: ``Variant.branch_preproc`` value (``None`` or a
            preproc identifier such as ``"snv"``, ``"msc"``, ``"asls"``).
        allow_selector_level_branch_preproc: opt-in escape hatch (default
            ``False``). When ``True``, the guard becomes a no-op even for
            selector variants. Use only when the preprocessing is
            **deliberately** dataset-level (e.g., signal-type detection
            that must be deterministic across folds) AND the variant author
            has documented why fold-level preprocessing is not appropriate
            in the variant's registry ``notes`` field. Codex must approve
            any registry entry that sets this flag.

    Raises:
        ValueError: when ``branch_preproc`` is non-empty, ``selection``
            names a selector variant, AND
            ``allow_selector_level_branch_preproc`` is ``False``.

    See ``bench/AOM_v0/Ridge/docs/HEADLINE_SPXY3_NESTED_AUDIT.md`` §10
    (D-A-008) for the rationale and the leakage trace.
    """
    if allow_selector_level_branch_preproc:
        return
    if branch_preproc and selection in SELECTOR_VARIANTS:
        raise ValueError(
            f"selector-level branch_preproc would leak across folds "
            f"(variant={label!r}, selection={selection!r}, "
            f"branch_preproc={branch_preproc!r}); move branch_preproc "
            f"into the candidate spec's `branch_preproc:` field inside "
            f"`extra` instead, or pass "
            f"`allow_selector_level_branch_preproc=True` if dataset-level "
            f"preprocessing is intentional and Codex-approved. See "
            f"bench/AOM_v0/Ridge/docs/HEADLINE_SPXY3_NESTED_AUDIT.md "
            f"§10 (D-A-008)."
        )


__all__ = ["SELECTOR_VARIANTS", "check_no_selector_branch_leak"]
