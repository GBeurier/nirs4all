"""CONFORMANCE: workspace / native-results CROSS-ENGINE parity (B-011 / SW5 §6b, PYREF-009b).

The two engines write NON-overlapping on-disk formats — legacy: ``store.sqlite``
+ ``arrays/*.parquet`` + ``runs/manifest.yaml``; dag-ml: an additive,
off-by-default native-results triple (``manifest.json`` + ``score_set.json`` +
``predictions.parquet``) the legacy engine ignores
(``test_dagml_native_results.py:210-219``). So the cross-engine question is NOT
byte-identity of the stores — it is that each engine's persisted results read
back to the SAME projection.

This module asserts, for representative native single-model shapes:

* **Read-back fidelity (same-engine).** A dag-ml run's native triple, read back
  from disk via :func:`~nirs4all.pipeline.dagml.native_results.read_native_results`
  (the verify-then-load reader), reproduces the LIVE run's final-(test) ``y_pred``
  exactly — the on-disk triple is a faithful projection of the run, and its
  ``manifest.engine`` records ``dag-ml``.

* **Cross-engine agreement through the read path.** The dag-ml triple read back
  FROM DISK agrees with the LEGACY run within the cross-impl bands —
  ``cross_impl_ypred`` (``1e-3``) on the per-sample final-(test) ``y_pred`` and
  ``cross_impl_score`` (``1e-3``) on the selected ``best_score``. This is the
  workspace analogue of the dual-engine numeric parity: the persisted native
  results are interchangeable with the legacy oracle, not just the in-memory
  ``RunResult``.

* **Legacy workspace inspectable.** The legacy run exposes the runtime V1 read
  surface (a finite selected ``best_score`` + per-sample final-(test)
  predictions), so a legacy-written workspace is inspectable/predictable through
  the same projection the runtime reads.

* **Runtime projection parity.** The native-results directory also projects
  through the public ``RtResult.from_native_dir`` seam to the same ScoreSet hash,
  reports, and final-test predictions as the live dag-ml ``RunResult``.

Slow: each case runs twice. Gated by ``slow``.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import nirs4all
from nirs4all.pipeline.dagml.native_results import read_native_results
from nirs4all.pipeline.dagml.rt import RtResult

from . import _conformance_helpers as H
from ._registry import PipelineCase, get

pytestmark = [pytest.mark.parity, pytest.mark.slow]


# Same-engine disk round-trip is a faithful projection (band native_export_reproduce).
_FIDELITY_TOL = 1e-6
# Cross-impl bands (compatibility.json): score + per-sample prediction.
_CROSS_IMPL_SCORE_TOL = H._DEFAULT_SCORE_TOL  # noqa: SLF001
_CROSS_IMPL_YPRED_TOL = H._DEFAULT_YPRED_TOL  # noqa: SLF001

_WORKSPACE_CASES = (
    "baseline_vertical_slice",
    "generator_range_n_components",
)


def _final_test_map(predictions: object) -> dict[int, np.ndarray]:
    """Map sample-id → final-(test) y_pred for a bare ``Predictions`` (read-back).

    Reuses the audited :func:`H._final_test_pred_by_sample` by wrapping the
    ``Predictions`` in the ``.predictions`` shape that helper expects.
    """
    return H._final_test_pred_by_sample(SimpleNamespace(predictions=predictions))  # noqa: SLF001


def _max_delta(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> float:
    """Max per-sample |Δ| over the shared sample ids (caller asserts the id sets)."""
    return max(float(np.max(np.abs(a[s] - b[s]))) for s in a)


def _rt_final_test_map(rt_result: RtResult) -> dict[int, np.ndarray]:
    """Map sample-id → final-test y_pred from an ``RtResult`` projection."""
    mapped: dict[int, np.ndarray] = {}
    for row in rt_result.predictions:
        if row.get("partition") != "test" or row.get("fold_id") != "final":
            continue
        values = row.get("y_pred")
        if values is None:
            continue
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        sample_indices = [int(sample_id) for sample_id in row.get("sample_indices", [])]
        assert len(sample_indices) == arr.shape[0], "RtResult prediction row sample_indices/y_pred length mismatch"
        for sample_id, prediction in zip(sample_indices, arr, strict=True):
            mapped[sample_id] = np.asarray(prediction, dtype=float)
    return mapped


@pytest.mark.parametrize("case_name", _WORKSPACE_CASES)
def test_native_results_triple_round_trips_and_agrees_cross_engine(case_name: str, tmp_path: Path) -> None:
    """The dag-ml native triple reads back faithfully AND agrees with legacy cross-engine.

    Skips (not fails) when dag-ml runs the legacy fallback — there is no native
    triple to read on that build.
    """
    case: PipelineCase = get(case_name)

    legacy = nirs4all.run(pipeline=case.pipeline, dataset=H.make_dataset(case), verbose=0, engine="legacy")
    dagml = nirs4all.run(
        pipeline=case.pipeline, dataset=H.make_dataset(case), verbose=0, engine="dag-ml",
        results_path=str(tmp_path / "res"),
    )
    if not dagml._is_dagml_engine():  # noqa: SLF001
        pytest.skip(f"{case_name}: dag-ml ran the legacy fallback on this build; no native triple to read")
    run_dir = dagml._dagml_results_dir  # noqa: SLF001
    assert run_dir is not None and Path(run_dir).is_dir(), f"{case_name}: dag-ml run wrote no native results dir"

    read = read_native_results(run_dir)

    # The triple records the engine that wrote it + carries a canonical ScoreSet.
    assert read["manifest"].get("engine") == "dag-ml", f"{case_name}: native manifest engine != dag-ml"
    assert isinstance(read["score_set"], dict) and read["score_set"], f"{case_name}: empty/absent score_set.json"

    live_map = H._final_test_pred_by_sample(dagml)  # noqa: SLF001
    read_map = _final_test_map(read["predictions"])
    legacy_map = H._final_test_pred_by_sample(legacy)  # noqa: SLF001
    assert read_map, f"{case_name}: native triple read back no final-(test) predictions"

    # 0. Runtime projection parity: the native dir and live RunResult expose the
    # same RT ScoreSet anchor + final-test prediction rows.
    disk_rt = RtResult.from_native_dir(run_dir)
    live_rt = dagml.to_rt_result()
    assert disk_rt.manifest["engine"] == live_rt.manifest["engine"] == "dag-ml"
    assert disk_rt.manifest["fingerprints"]["score_set_hash"] == live_rt.manifest["fingerprints"]["score_set_hash"], (
        f"{case_name}: native-dir RtResult and live RunResult RtResult disagree on score_set_hash"
    )
    assert disk_rt.reports == live_rt.reports, f"{case_name}: native-dir RtResult reports drift from live RunResult reports"
    disk_rt_map = _rt_final_test_map(disk_rt)
    live_rt_map = _rt_final_test_map(live_rt)
    assert set(disk_rt_map) == set(live_rt_map) == set(read_map), (
        f"{case_name}: RtResult final-test sample ids diverge from native read-back "
        f"(|disk_rt|={len(disk_rt_map)} |live_rt|={len(live_rt_map)} |read|={len(read_map)})"
    )
    rt_delta = _max_delta(disk_rt_map, live_rt_map)
    assert rt_delta <= _FIDELITY_TOL, (
        f"{case_name}: native-dir RtResult != live RunResult RtResult final-test y_pred; "
        f"max |Δ| = {rt_delta:.3e} > {_FIDELITY_TOL:.3e}"
    )

    # 1. Read-back fidelity: the disk triple == the live dag-ml run (same engine).
    assert set(read_map) == set(live_map), (
        f"{case_name}: read-back final-(test) sample ids diverge from the live run "
        f"(|read|={len(read_map)} |live|={len(live_map)})"
    )
    fidelity_delta = _max_delta(live_map, read_map)
    assert fidelity_delta <= _FIDELITY_TOL, (
        f"{case_name}: native triple read-back != live dag-ml final-(test) y_pred; "
        f"max |Δ| = {fidelity_delta:.3e} > {_FIDELITY_TOL:.3e} (the on-disk projection is not faithful)"
    )

    # 2. Cross-engine agreement through the read path: disk triple vs legacy oracle.
    assert set(read_map) == set(legacy_map), (
        f"{case_name}: read-back vs legacy final-(test) sample ids diverge "
        f"(|read|={len(read_map)} |legacy|={len(legacy_map)})"
    )
    cross_delta = _max_delta(read_map, legacy_map)
    assert cross_delta <= _CROSS_IMPL_YPRED_TOL, (
        f"{case_name}: persisted dag-ml triple vs legacy final-(test) y_pred; "
        f"max |Δ| = {cross_delta:.3e} > band cross_impl_ypred {_CROSS_IMPL_YPRED_TOL:.3e}"
    )
    score_delta = abs(dagml.best_score - legacy.best_score)
    assert score_delta <= _CROSS_IMPL_SCORE_TOL, (
        f"{case_name}: selected best_score cross-engine Δ = {score_delta:.3e} > "
        f"band cross_impl_score {_CROSS_IMPL_SCORE_TOL:.3e}"
    )

    # 3. Legacy workspace inspectable via the runtime V1 read surface.
    assert legacy_map, f"{case_name}: legacy workspace exposed no final-(test) predictions to inspect"
    assert not math.isnan(legacy.best_score), f"{case_name}: legacy workspace exposed no finite best_score"
