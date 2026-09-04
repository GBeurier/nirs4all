"""PYREF native-vs-refusal coverage meter (B-010 / DML-003).

Classifies every registered :class:`PipelineCase` into exactly one disposition
bucket and rolls the partition up into the ``coverage_meter`` summary that
``docs/compatibility.json`` publishes. That summary is the **LOCK-DROP D1
instrument**: unsupported native shapes are explicit refusals and can never be
silently re-executed by the legacy engine. Each PR therefore shows the refusal
count moving toward zero.

Disposition buckets (the per-case partition — every case lands in exactly one):

* ``native`` — concrete single-pipeline shape the dag-ml backend runs itself.
* ``python_expanded`` — a generator shape whose variant SET is enumerated /
  projected Python-side before each concrete route runs (A3 §9); native at
  execution, Python-orchestrated at expansion.
* ``python_pre_materialized`` — a rep-fusion / augmentation shape the host
  reshapes / materializes in Python before the native CV/scoring phases (A3
  coverage matrix).
* ``expected_refusal`` — on the :data:`EXPECTED_REFUSAL` allowlist: the dag-ml
  path legitimately rejects the shape today without running legacy (owner L5).
* ``unexpected_refusal`` — observed to refuse but NOT on the allowlist: a
  native-coverage REGRESSION. Always empty in this STATIC meter (see below).
* ``xfail`` — a strict-xfail: either a ``KNOWN_DIVERGENCES`` cross-engine
  divergence (runs native, diverges) or a registry ``legacy_bug`` case (no
  legacy oracle, non-runnable).
* ``skip`` — a registry ``fixture`` / ``unknown_semantics`` skip (non-runnable).

STATIC vs DYNAMIC
-----------------
This meter is **static**: it classifies from the declared parity structures (the
case registry + the :data:`EXPECTED_REFUSAL` allowlist + ``KNOWN_DIVERGENCES``
+ the registry skip kinds) WITHOUT running either engine. The DYNAMIC truth —
that no runnable case refuses OFF the allowlist — is enforced
per-case by ``test_conformance_dual_engine.py::test_native_refusal_boundary``,
which runs the real dag-ml leg. The static meter trusts that guard, so
``unexpected_refusal`` is 0 here. A caller that already has dynamic observations
(e.g. the boundary test) may pass ``observed_refusal=`` to compute it
without this module running anything itself.

CLI::

    python -m tests.integration.parity.coverage_meter            # markdown to stdout
    python -m tests.integration.parity.coverage_meter --json out.json --md out.md
    python -m tests.integration.parity.coverage_meter --check    # diff vs ledger; exit 1 on drift
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

# Side-effect imports: each `cases_*` module registers its cases at top level.
# These MUST run before `all_cases()` / the `test_conformance_dual_engine`
# import (whose parametrize decorators enumerate the registry at import time).
from . import (  # noqa: F401
    cases_aggregation_reps,
    cases_augmentation,
    cases_baseline,
    cases_branches_merges,
    cases_generators,
    cases_generators_conformance,
    cases_multi_source,
    cases_refit_predict,
    cases_tags_exclude,
)
from ._registry import CANONICAL_KEYWORDS, PipelineCase, all_cases
from .test_conformance_dual_engine import (
    EXPECTED_REFUSAL,
    KNOWN_DIVERGENCES,
    NUM_PREDICTIONS_DIVERGENCE,
    UNSEEDED_NONDETERMINISTIC_CASES,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPATIBILITY_JSON = REPO_ROOT / "docs" / "compatibility.json"

# ---------------------------------------------------------------------------
# Bucket vocabulary.
# ---------------------------------------------------------------------------
NATIVE = "native"
PYTHON_EXPANDED = "python_expanded"
PYTHON_PRE_MATERIALIZED = "python_pre_materialized"
EXPECTED_REFUSAL_BUCKET = "expected_refusal"
UNEXPECTED_REFUSAL = "unexpected_refusal"
XFAIL = "xfail"
SKIP = "skip"

#: The seven mutually-exclusive leaf buckets every case partitions into.
LEAF_BUCKETS: tuple[str, ...] = (
    NATIVE,
    PYTHON_EXPANDED,
    PYTHON_PRE_MATERIALIZED,
    EXPECTED_REFUSAL_BUCKET,
    UNEXPECTED_REFUSAL,
    XFAIL,
    SKIP,
)

#: The exact key set of ``compatibility.json["coverage_meter"]`` — the summary
#: face the ledger publishes and ``_authority.py`` validates.
LEDGER_SUMMARY_KEYS: tuple[str, ...] = (
    "registered",
    "non_runnable",
    "runnable",
    "refusal",
    "native",
    "xfail_strict",
    "skip",
    "num_predictions_divergence",
    "run_only_nondeterministic",
    "expected_refusal_target",
)

# Keywords that mark a runnable native-route case as Python-pre-materialized:
# rep-fusion reshapes the dataset and augmentation materializes train samples
# host-side before the native CV/scoring phases (A3 coverage matrix, §9).
_PRE_MATERIALIZED_KEYWORDS: frozenset[str] = frozenset({
    "rep_to_sources",
    "rep_to_pp",
    "sample_augmentation",
    "feature_augmentation",
    "concat_transform",
})

# Generator keywords: the variant set is enumerated / projected Python-side
# (A3 §9) even where each concrete route then executes natively.
_GENERATOR_KEYWORDS: frozenset[str] = frozenset({
    "_or_",
    "_range_",
    "_log_range_",
    "_grid_",
    "_cartesian_",
    "_zip_",
    "_chain_",
    "_sample_",
})


@dataclass(frozen=True)
class CaseClassification:
    """One case's disposition under the static meter."""

    name: str
    bucket: str
    subtype: str
    runnable: bool
    basis: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "bucket": self.bucket,
            "subtype": self.subtype,
            "runnable": self.runnable,
            "basis": self.basis,
        }


def _native_route(case: PipelineCase) -> tuple[str, str]:
    """Sub-classify a runnable, non-divergent, non-refusal case by its shape.

    Pre-materialization wins over expansion: a generator over an augmented
    dataset is still gated on the host materialization.
    """
    keywords = set(case.keywords)
    pre = sorted(keywords & _PRE_MATERIALIZED_KEYWORDS)
    if pre:
        return PYTHON_PRE_MATERIALIZED, f"host pre-materialization keyword(s) {pre} (rep-fusion / augmentation reshaped before native CV)"
    generators = sorted(keywords & _GENERATOR_KEYWORDS)
    if generators:
        return PYTHON_EXPANDED, f"generator keyword(s) {generators} (Python-side variant expansion + legacy-name projection, A3 §9)"
    return NATIVE, "concrete native dag-ml route"


def classify_case(case: PipelineCase, observed_refusal: frozenset[str] | None = None) -> CaseClassification:
    """Assign ``case`` to exactly one leaf bucket (precedence-ordered).

    Precedence: registry skip (legacy_bug → xfail, else skip) ▸ KNOWN_DIVERGENCES
    (xfail) ▸ observed off-allowlist refusal (unexpected_refusal) ▸
    EXPECTED_REFUSAL (expected_refusal) ▸ native route. ``observed_refusal``
    is only consulted for the unexpected decision; when ``None`` (static mode)
    the meter trusts the allowlist and ``unexpected_refusal`` stays empty.
    """
    name = case.name
    if case.skip_reason:
        if case.skip_kind == "legacy_bug":
            return CaseClassification(name, XFAIL, "legacy_bug", False, f"registry legacy_bug → strict-xfail (no legacy oracle): {case.skip_reason}")
        return CaseClassification(name, SKIP, case.skip_kind or "unknown", False, f"registry {case.skip_kind or 'unknown'} skip: {case.skip_reason}")
    if name in KNOWN_DIVERGENCES:
        return CaseClassification(name, XFAIL, "known_divergence", True, f"KNOWN_DIVERGENCES strict-xfail: {KNOWN_DIVERGENCES[name]}")
    if observed_refusal is not None and name in observed_refusal and name not in EXPECTED_REFUSAL:
        return CaseClassification(name, UNEXPECTED_REFUSAL, "", True, "dag-ml refused OFF the EXPECTED_REFUSAL allowlist — native-coverage regression")
    if name in EXPECTED_REFUSAL:
        return CaseClassification(name, EXPECTED_REFUSAL_BUCKET, "", True, "EXPECTED_REFUSAL allowlist (explicit fail-closed dag-ml boundary)")
    bucket, basis = _native_route(case)
    return CaseClassification(name, bucket, "", True, basis)


@dataclass(frozen=True)
class CoverageReport:
    """The classified inventory + its summary / artifact projections."""

    cases: tuple[CaseClassification, ...]

    def names_in(self, bucket: str) -> list[str]:
        return sorted(c.name for c in self.cases if c.bucket == bucket)

    def leaf_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(LEAF_BUCKETS, 0)
        for case in self.cases:
            counts[case.bucket] += 1
        return counts

    def bucket_counts(self) -> dict[str, int]:
        """Return the mutually exclusive disposition counts."""
        return self.leaf_counts()

    def summary(self) -> dict[str, int]:
        """The ``coverage_meter`` ledger face (exactly :data:`LEDGER_SUMMARY_KEYS`)."""
        leaves = self.leaf_counts()
        registered = len(self.cases)
        non_runnable = sum(1 for c in self.cases if not c.runnable)
        runnable = registered - non_runnable
        refusal = leaves[EXPECTED_REFUSAL_BUCKET] + leaves[UNEXPECTED_REFUSAL]
        return {
            "registered": registered,
            "non_runnable": non_runnable,
            "runnable": runnable,
            "refusal": refusal,
            "native": runnable - refusal,
            "xfail_strict": leaves[XFAIL],
            "skip": leaves[SKIP],
            "num_predictions_divergence": len(NUM_PREDICTIONS_DIVERGENCE),
            "run_only_nondeterministic": len(UNSEEDED_NONDETERMINISTIC_CASES),
            "expected_refusal_target": 0,
        }

    def to_inventory(self) -> dict[str, object]:
        """Full machine-readable inventory (summary + buckets + per-case rows)."""
        return {
            "schema": "nirs4all.pyref.coverage_meter.v1",
            "summary": self.summary(),
            "buckets": self.bucket_counts(),
            "cases": [c.as_dict() for c in self.cases],
        }

    def to_markdown(self) -> str:
        summary = self.summary()
        buckets = self.bucket_counts()
        lines = [
            "# PYREF native-vs-refusal coverage meter",
            "",
            "Fail-closed boundary instrument (B-010 / DML-003): unsupported shapes refuse without legacy execution.",
            "",
            "| metric | count |",
            "|---|---|",
            f"| registered | {summary['registered']} |",
            f"| runnable | {summary['runnable']} |",
            f"| native (reach) | {summary['native']} |",
            f"| refusal | {summary['refusal']} |",
            f"| — expected_refusal | {buckets[EXPECTED_REFUSAL_BUCKET]} |",
            f"| — unexpected_refusal | {buckets[UNEXPECTED_REFUSAL]} |",
            f"| xfail_strict | {summary['xfail_strict']} |",
            f"| skip | {summary['skip']} |",
            f"| run_only_nondeterministic | {summary['run_only_nondeterministic']} |",
            f"| expected_refusal_target | {summary['expected_refusal_target']} |",
            "",
            "## Disposition partition (one bucket per case)",
            "",
            "| bucket | count |",
            "|---|---|",
        ]
        lines.extend(f"| {bucket} | {buckets[bucket]} |" for bucket in LEAF_BUCKETS)
        refusal_names = self.names_in(EXPECTED_REFUSAL_BUCKET)
        if refusal_names:
            lines.extend(["", "## expected_refusal (shrink target, owner L5)", ""])
            lines.extend(f"- {name}" for name in refusal_names)
        unexpected_names = self.names_in(UNEXPECTED_REFUSAL)
        if unexpected_names:
            lines.extend(["", "## unexpected refusal (native-coverage REGRESSION)", ""])
            lines.extend(f"- {name}" for name in unexpected_names)
        return "\n".join(lines) + "\n"


def build_report(observed_refusal: Iterable[str] | None = None) -> CoverageReport:
    """Classify the whole registry into a :class:`CoverageReport`."""
    observed = frozenset(observed_refusal) if observed_refusal is not None else None
    return CoverageReport(tuple(classify_case(case, observed) for case in all_cases()))


def load_ledger_coverage_meter(path: Path = COMPATIBILITY_JSON) -> dict[str, int]:
    """Read the published ``coverage_meter`` summary from the ledger."""
    with path.open("r", encoding="utf-8") as handle:
        return cast("dict[str, int]", json.load(handle)["coverage_meter"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PYREF native-vs-refusal coverage meter (B-010 / DML-003).")
    parser.add_argument("--json", type=Path, default=None, help="write the full inventory JSON to this path")
    parser.add_argument("--md", type=Path, default=None, help="write the markdown summary to this path")
    parser.add_argument("--check", action="store_true", help="compare the meter summary to the ledger; exit 1 on drift")
    args = parser.parse_args(argv)

    report = build_report()
    if args.json is not None:
        args.json.write_text(json.dumps(report.to_inventory(), indent=2) + "\n", encoding="utf-8")
    if args.md is not None:
        args.md.write_text(report.to_markdown(), encoding="utf-8")

    exit_code = 0
    if args.check:
        live = report.summary()
        published = load_ledger_coverage_meter()
        if live != published:
            print("coverage_meter DRIFT — ledger does not match the live meter:")
            print(f"  meter  = {live}")
            print(f"  ledger = {published}")
            exit_code = 1
        else:
            print(f"coverage_meter OK (refusal={live['refusal']}, target={live['expected_refusal_target']})")

    if args.json is None and args.md is None and not args.check:
        print(report.to_markdown())
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
