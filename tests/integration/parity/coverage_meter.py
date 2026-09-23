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
import ast
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
from ._public_surface import surface_drift
from ._registry import CANONICAL_KEYWORDS, PipelineCase, all_cases
from .test_conformance_dual_engine import (
    EXPECTED_REFUSAL,
    KNOWN_DIVERGENCES,
    NUM_PREDICTIONS_DIVERGENCE,
    UNSEEDED_NONDETERMINISTIC_CASES,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPATIBILITY_JSON = REPO_ROOT / "docs" / "compatibility.json"
FEATURE_GAPS_JSON = Path(__file__).with_name("feature_gaps.json")
CONTROLLER_COVERAGE_JSON = Path(__file__).with_name("controller_coverage.json")
MODEL_CHECKPOINT_COVERAGE_JSON = Path(__file__).with_name("model_checkpoint_coverage.json")

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


def checkpoint_inventory_status(path: Path = MODEL_CHECKPOINT_COVERAGE_JSON) -> tuple[bool, list[str], list[str]]:
    """Validate reviewed model-checkpoint positions and their public evidence."""
    inventory = json.loads(path.read_text(encoding="utf-8"))
    open_patterns: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    expected_stages = set(inventory["candidate_intervening_stages"])
    observed_stages: set[str] = set()
    for row in inventory["patterns"]:
        identifier = row["id"]
        if identifier in seen:
            invalid.append(f"duplicate checkpoint pattern {identifier}")
        seen.add(identifier)
        stage = row["intervening_stage"]
        if stage in observed_stages:
            invalid.append(f"duplicate checkpoint stage {stage}")
        observed_stages.add(stage)
        legacy, coverage = row["legacy"], row["coverage"]
        if (legacy == "success" and coverage not in {"verified", "gap", "unverified"}) or (legacy == "unverified" and coverage != "unverified") or (legacy == "failure" and coverage != "not_applicable") or legacy not in {"success", "unverified", "failure"}:
            invalid.append(f"{identifier}: invalid legacy/coverage status")
        if legacy == "unverified" or (legacy == "success" and coverage != "verified"):
            open_patterns.append(identifier)
            continue
        if legacy != "failure" and coverage != "verified":
            continue
        if not row["evidence"]:
            invalid.append(f"{identifier}: verified/failure status without public test evidence")
        for reference in row["evidence"]:
            file_name, separator, test_name = reference.partition("::")
            file_path = REPO_ROOT / file_name
            if not separator or not file_path.is_file():
                invalid.append(f"{identifier}: invalid evidence {reference}")
                continue
            try:
                syntax = ast.parse(file_path.read_text(encoding="utf-8"))
            except SyntaxError:
                invalid.append(f"{identifier}: evidence does not parse {reference}")
                continue
            if test_name not in {node.name for node in ast.walk(syntax) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}:
                invalid.append(f"{identifier}: missing test function {reference}")
    missing_stages = expected_stages - observed_stages
    if missing_stages:
        invalid.append(f"missing checkpoint stages: {sorted(missing_stages)}")
    unknown_stages = observed_stages - expected_stages - {"meta_model"}
    if unknown_stages:
        invalid.append(f"unexpected checkpoint stages: {sorted(unknown_stages)}")
    return bool(inventory["inventory_complete"]), open_patterns, invalid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PYREF native-vs-refusal coverage meter (B-010 / DML-003).")
    parser.add_argument("--json", type=Path, default=None, help="write the full inventory JSON to this path")
    parser.add_argument("--md", type=Path, default=None, help="write the markdown summary to this path")
    parser.add_argument("--check", action="store_true", help="compare the meter and legacy public surface to their reviewed snapshots; exit 1 on drift")
    parser.add_argument("--require-zero-refusals", action="store_true", help="fail if any registered case still refuses DAG-ML")
    parser.add_argument("--require-feature-complete", action="store_true", help="fail until the feature inventory is complete and all registered gaps are closed")
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

    if args.check or args.require_feature_complete:
        drift = surface_drift()
        if drift:
            print(f"DAG-ML legacy surface DRIFT in {drift}; review public_surface.json and parity evidence")
            exit_code = 1
        checkpoint_complete, open_checkpoints, invalid_checkpoints = checkpoint_inventory_status()
        if invalid_checkpoints:
            print("DAG-ML model-checkpoint inventory invalid: " + "; ".join(invalid_checkpoints))
            exit_code = 1

    if args.require_zero_refusals and report.summary()["refusal"]:
        refused = report.names_in(EXPECTED_REFUSAL_BUCKET) + report.names_in(UNEXPECTED_REFUSAL)
        print("DAG-ML parity gate failed; registered cases still refuse: " + ", ".join(refused))
        exit_code = 1

    if args.require_feature_complete:
        ledger = json.loads(FEATURE_GAPS_JSON.read_text(encoding="utf-8"))
        gaps = ledger["gaps"]
        unverified = ledger["unverified"]
        controllers = json.loads(CONTROLLER_COVERAGE_JSON.read_text(encoding="utf-8"))["controllers"]
        uncovered_controllers = sorted(
            name for name, entry in controllers.items()
            if entry["coverage"] not in {"public", "not_pipeline"}
        )
        if not ledger["inventory_complete"] or gaps or unverified or report.summary()["refusal"] or uncovered_controllers or not checkpoint_complete or open_checkpoints:
            print(f"DAG-ML feature gate failed: inventory_complete={ledger['inventory_complete']}, open_gaps={len(gaps)}, unverified={len(unverified)}, registered_refusals={report.summary()['refusal']}, uncovered_controllers={uncovered_controllers}, checkpoint_inventory_complete={checkpoint_complete}, open_checkpoint_patterns={open_checkpoints}")
            exit_code = 1

    if args.json is None and args.md is None and not args.check and not args.require_zero_refusals and not args.require_feature_complete:
        print(report.to_markdown())
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
