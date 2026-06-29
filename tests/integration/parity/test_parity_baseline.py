"""Layer 1–2 parity test: capture / enforce the legacy gold baseline.

**Capture** (writes ``baselines/<case>.json`` for every runnable case):

    pytest tests/integration/parity/test_parity_baseline.py --parity-capture -m slow

**Enforce** (default; re-runs the legacy backend and diffs against the captured
baseline within each case's ``metric_tolerances``):

    pytest tests/integration/parity/test_parity_baseline.py -m slow

Cases without a captured baseline are *skipped*, so capture is a deliberate,
reviewable step (the baselines are committed artifacts). This is the exact
comparison the dag-ml backend will face once the bridge is wired
(``dag-ml/docs/migration-nirs4all/PARITY_AND_PERF_HARNESS.md``, Layer 3); only
the backend that produces the observation changes.
"""

from __future__ import annotations

import pytest

import nirs4all
from nirs4all.data import DatasetConfigs

from . import _oracle
from ._datasets import dataset_path
from ._registry import PipelineCase, all_cases

pytestmark = [pytest.mark.parity]


def test_every_runnable_case_has_committed_baseline() -> None:
    """Gate integrity (fast): no runnable case may silently lack a gold baseline.

    Without this, adding a parity case without capturing its baseline would make
    that case skip on a clean checkout — a silent hole in the non-regression gate.
    Marked ``parity`` only (not ``slow``): it runs no pipeline, just checks files.
    """
    missing = [c.name for c in all_cases() if not c.skip_reason and _oracle.load_baseline(c.name) is None]
    assert not missing, (
        "runnable parity cases without a committed gold baseline "
        f"(capture with --parity-capture and commit): {missing}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", all_cases(), ids=lambda c: c.name)
def test_legacy_gold_baseline(case: PipelineCase, request: pytest.FixtureRequest) -> None:
    """Capture or enforce the legacy gold baseline for one parity case."""
    if case.skip_reason:
        pytest.skip(f"[{case.skip_kind or 'unknown'}] {case.skip_reason}")

    pipeline = case.pipeline
    fingerprint = _oracle.pipeline_fingerprint(pipeline)

    if request.config.getoption("--parity-capture"):
        dataset = DatasetConfigs(dataset_path(case.dataset_key), **case.dataset_kwargs)
        # ENGINE PINNED TO LEGACY: the gold is the LEGACY oracle (ADR-01). Since the
        # ADR-17 flip the DEFAULT engine is dag-ml, so an unqualified ``nirs4all.run``
        # would capture a DAG-ML observation mislabeled "legacy" — and the enforce
        # path (also pinned below) would then compare dag-ml↔dag-ml, silently hiding
        # any legacy↔dag-ml divergence. The dual-engine conformance suite owns the
        # legacy↔dag-ml comparison; this layer is the legacy non-regression alarm.
        result = nirs4all.run(pipeline=pipeline, dataset=dataset, verbose=0, engine="legacy")
        path = _oracle.save_baseline(case.name, fingerprint, _oracle.observe(result, case.task))
        pytest.skip(f"captured legacy baseline -> {path.name}")

    gold = _oracle.load_baseline(case.name)
    if gold is None:
        pytest.skip(f"no gold baseline for {case.name!r}; run with --parity-capture")
    if gold.get("pipeline_fingerprint") != fingerprint:
        pytest.fail(
            f"{case.name}: gold baseline is stale "
            f"(pipeline_fingerprint {gold.get('pipeline_fingerprint')} != {fingerprint}); "
            f"recapture with --parity-capture"
        )

    dataset = DatasetConfigs(dataset_path(case.dataset_key), **case.dataset_kwargs)
    # ENGINE PINNED TO LEGACY (matches the legacy-pinned capture above): the gold
    # is the LEGACY oracle. An unqualified run would resolve to the DEFAULT engine
    # (dag-ml since the ADR-17 flip) and compare dag-ml↔dag-ml, vacuously passing
    # for every shape where the two engines diverge. The legacy↔dag-ml comparison
    # is owned by test_conformance_dual_engine; this is the legacy-only alarm.
    result = nirs4all.run(pipeline=pipeline, dataset=dataset, verbose=0, engine="legacy")
    observed = _oracle.observe(result, case.task)
    violations = _oracle.compare(gold, observed, case.metric_tolerances)
    assert not violations, f"{case.name}: parity violations:\n  " + "\n  ".join(violations)
