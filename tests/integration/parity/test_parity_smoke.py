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

from collections.abc import Iterable

import pytest

import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.pipeline.dagml.rt import RtError

from ._datasets import dataset_path
from ._registry import PipelineCase, all_cases, by_tag

pytestmark = pytest.mark.slow


def _make_dataset(case: PipelineCase) -> DatasetConfigs:
    """Resolve a case's dataset_key + dataset_kwargs into a `DatasetConfigs`."""
    return DatasetConfigs(dataset_path(case.dataset_key), **case.dataset_kwargs)


def _case_params(cases: Iterable[PipelineCase]) -> list[object]:
    """Attach manifest debt at collection time so skip/xfail headlines stay honest."""
    params: list[object] = []
    for case in cases:
        marks = []
        if case.skip_reason:
            if case.skip_kind == "legacy_bug":
                marks.append(pytest.mark.xfail(reason=f"[legacy_bug] {case.skip_reason}", strict=True))
            else:
                marks.append(pytest.mark.skip(reason=f"[{case.skip_kind or 'unknown'}] {case.skip_reason}"))
        params.append(pytest.param(case, id=case.name, marks=marks))
    return params


def _export_bundle_for_smoke(result: object, bundle_path) -> None:
    """Export for legacy bundle/retrain smoke while preserving V1 dag-ml refusal visibility."""
    try:
        result.export(str(bundle_path))  # type: ignore[attr-defined]
        return
    except RtError as exc:
        is_dagml = bool(result._is_dagml_engine())  # type: ignore[attr-defined]  # noqa: SLF001
        if not is_dagml:
            raise
        payload = exc.to_dict()
        assert payload["cause"] == "unsupported_capability"
        assert payload["unsupported_capability"] == "dagml_native_export"
        assert "compatibility='legacy-refit'" in payload["mitigation"]
        assert not bundle_path.exists(), "default dag-ml export refusal must not leave a partial bundle"

    result.export(str(bundle_path), compatibility="legacy-refit")  # type: ignore[attr-defined]


@pytest.mark.parametrize("case", _case_params(all_cases()))
def test_pipeline_runs_end_to_end(case: PipelineCase, tmp_path) -> None:
    """Each non-skipped case runs to completion and reports predictions."""
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


@pytest.mark.parametrize("case", _case_params(by_tag("round_trip")))
def test_round_trip_bundle_export_load_predict(case: PipelineCase, tmp_path) -> None:
    """Cases tagged `round_trip` must train → export `.n4a` → reload → `nirs4all.predict()`."""
    dataset = _make_dataset(case)
    result = nirs4all.run(
        pipeline=case.pipeline,
        dataset=dataset,
        verbose=0,
    )
    bundle_path = tmp_path / f"{case.name}.n4a"
    _export_bundle_for_smoke(result, bundle_path)
    assert bundle_path.exists(), f"{case.name}: bundle was not exported"

    preds = nirs4all.predict(str(bundle_path), dataset)
    assert preds is not None, f"{case.name}: predict returned None"


@pytest.mark.parametrize("case", _case_params(by_tag("explain")))
def test_explain_path(case: PipelineCase, tmp_path) -> None:
    """Cases tagged `explain` must complete `nirs4all.explain()` on the trained bundle."""
    # nirs4all.explain() builds a ShapAnalyzer, which raises if SHAP is absent.
    pytest.importorskip("shap")

    dataset = _make_dataset(case)
    result = nirs4all.run(
        pipeline=case.pipeline,
        dataset=dataset,
        verbose=0,
    )
    bundle_path = tmp_path / f"{case.name}.n4a"
    _export_bundle_for_smoke(result, bundle_path)

    explanation = nirs4all.explain(str(bundle_path), dataset)
    assert explanation is not None


@pytest.mark.parametrize("case", _case_params(by_tag("retrain")))
def test_retrain_path(case: PipelineCase, tmp_path) -> None:
    """Cases tagged `retrain` must complete `nirs4all.retrain()` on a fresh dataset."""
    dataset = _make_dataset(case)
    result = nirs4all.run(
        pipeline=case.pipeline,
        dataset=dataset,
        verbose=0,
    )
    bundle_path = tmp_path / f"{case.name}.n4a"
    _export_bundle_for_smoke(result, bundle_path)

    retrained = nirs4all.retrain(str(bundle_path), dataset)
    assert retrained is not None


@pytest.mark.parametrize("case", _case_params(by_tag("session")))
def test_session_api(case: PipelineCase, tmp_path) -> None:
    """Cases tagged `session` must complete the stateful `nirs4all.session()` workflow."""
    dataset = _make_dataset(case)
    with nirs4all.session(
        pipeline=case.pipeline,
        name=case.name,
    ) as sess:
        result = sess.run(dataset)
        assert result is not None
