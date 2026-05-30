"""End-to-end smoke test: every runnable parity case actually runs.

This is the gold-standard test the dag-ml bridge will be measured against
later. It runs every case that does not carry a `skip_reason`, asserts
the result object reports at least `expected_min_predictions` entries,
and validates the public-API tagged cases (`round_trip`, `predict_path`,
`explain`, `retrain`, `session`) hit their respective entry points.

The full suite is slow (minutes per case × dozens of cases) — gated by
the `slow` marker. Run with:

    pytest tests/integration/parity/test_parity_smoke.py -m slow -v

A subset can be selected by tag:

    pytest tests/integration/parity/test_parity_smoke.py -k round_trip
"""

from __future__ import annotations

import pytest

import nirs4all
from nirs4all.data import DatasetConfigs

from ._datasets import dataset_path
from ._registry import PipelineCase, all_cases, by_tag

pytestmark = pytest.mark.slow


def _make_dataset(case: PipelineCase) -> DatasetConfigs:
    """Resolve a case's dataset_key + dataset_kwargs into a `DatasetConfigs`."""
    return DatasetConfigs(dataset_path(case.dataset_key), **case.dataset_kwargs)


@pytest.mark.parametrize("case", all_cases(), ids=lambda c: c.name)
def test_pipeline_runs_end_to_end(case: PipelineCase, tmp_path) -> None:
    """Each non-skipped case runs to completion and reports predictions."""
    if case.skip_reason:
        # TODO(parity-manifest-followup): convert `skip_kind="legacy_bug"`
        # to `pytest.mark.xfail(strict=True)` once the test param map exposes
        # the skip_kind at collection time (see Codex review on the Phase-3
        # close-out). Today pytest.xfail() at runtime reports as FAIL, not
        # XFAIL, which would mask the legacy fix.
        pytest.skip(f"[{case.skip_kind or 'unknown'}] {case.skip_reason}")

    dataset = _make_dataset(case)
    result = nirs4all.run(
        pipeline=case.pipeline,
        dataset=dataset,
        verbose=0,
    )
    assert result is not None, f"{case.name}: nirs4all.run returned None"
    assert result.num_predictions >= case.expected_min_predictions, (
        f"{case.name}: expected >= {case.expected_min_predictions} predictions, "
        f"got {result.num_predictions}"
    )


@pytest.mark.parametrize("case", by_tag("round_trip"), ids=lambda c: c.name)
def test_round_trip_bundle_export_load_predict(case: PipelineCase, tmp_path) -> None:
    """Cases tagged `round_trip` must train → export `.n4a` → reload → `nirs4all.predict()`."""
    if case.skip_reason:
        # TODO(parity-manifest-followup): convert `skip_kind="legacy_bug"`
        # to `pytest.mark.xfail(strict=True)` once the test param map exposes
        # the skip_kind at collection time (see Codex review on the Phase-3
        # close-out). Today pytest.xfail() at runtime reports as FAIL, not
        # XFAIL, which would mask the legacy fix.
        pytest.skip(f"[{case.skip_kind or 'unknown'}] {case.skip_reason}")

    dataset = _make_dataset(case)
    result = nirs4all.run(
        pipeline=case.pipeline,
        dataset=dataset,
        verbose=0,
    )
    bundle_path = tmp_path / f"{case.name}.n4a"
    result.export(str(bundle_path))
    assert bundle_path.exists(), f"{case.name}: bundle was not exported"

    preds = nirs4all.predict(str(bundle_path), dataset)
    assert preds is not None, f"{case.name}: predict returned None"


@pytest.mark.parametrize("case", by_tag("explain"), ids=lambda c: c.name)
def test_explain_path(case: PipelineCase, tmp_path) -> None:
    """Cases tagged `explain` must complete `nirs4all.explain()` on the trained bundle."""
    if case.skip_reason:
        # TODO(parity-manifest-followup): convert `skip_kind="legacy_bug"`
        # to `pytest.mark.xfail(strict=True)` once the test param map exposes
        # the skip_kind at collection time (see Codex review on the Phase-3
        # close-out). Today pytest.xfail() at runtime reports as FAIL, not
        # XFAIL, which would mask the legacy fix.
        pytest.skip(f"[{case.skip_kind or 'unknown'}] {case.skip_reason}")

    dataset = _make_dataset(case)
    result = nirs4all.run(
        pipeline=case.pipeline,
        dataset=dataset,
        verbose=0,
    )
    bundle_path = tmp_path / f"{case.name}.n4a"
    result.export(str(bundle_path))

    explanation = nirs4all.explain(str(bundle_path), dataset)
    assert explanation is not None


@pytest.mark.parametrize("case", by_tag("retrain"), ids=lambda c: c.name)
def test_retrain_path(case: PipelineCase, tmp_path) -> None:
    """Cases tagged `retrain` must complete `nirs4all.retrain()` on a fresh dataset."""
    if case.skip_reason:
        # TODO(parity-manifest-followup): convert `skip_kind="legacy_bug"`
        # to `pytest.mark.xfail(strict=True)` once the test param map exposes
        # the skip_kind at collection time (see Codex review on the Phase-3
        # close-out). Today pytest.xfail() at runtime reports as FAIL, not
        # XFAIL, which would mask the legacy fix.
        pytest.skip(f"[{case.skip_kind or 'unknown'}] {case.skip_reason}")

    dataset = _make_dataset(case)
    result = nirs4all.run(
        pipeline=case.pipeline,
        dataset=dataset,
        verbose=0,
    )
    bundle_path = tmp_path / f"{case.name}.n4a"
    result.export(str(bundle_path))

    retrained = nirs4all.retrain(str(bundle_path), dataset)
    assert retrained is not None


@pytest.mark.parametrize("case", by_tag("session"), ids=lambda c: c.name)
def test_session_api(case: PipelineCase, tmp_path) -> None:
    """Cases tagged `session` must complete the stateful `nirs4all.session()` workflow."""
    if case.skip_reason:
        # TODO(parity-manifest-followup): convert `skip_kind="legacy_bug"`
        # to `pytest.mark.xfail(strict=True)` once the test param map exposes
        # the skip_kind at collection time (see Codex review on the Phase-3
        # close-out). Today pytest.xfail() at runtime reports as FAIL, not
        # XFAIL, which would mask the legacy fix.
        pytest.skip(f"[{case.skip_kind or 'unknown'}] {case.skip_reason}")

    dataset = _make_dataset(case)
    with nirs4all.session(
        pipeline=case.pipeline,
        name=case.name,
    ) as sess:
        result = sess.run(dataset)
        assert result is not None
