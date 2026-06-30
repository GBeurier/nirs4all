"""CONFORMANCE: ``.n4a`` bundle CROSS-ENGINE round-trip (B-011 / SW5 §6a, PYREF-009a).

``test_conformance_export_roundtrip.py`` already proves a NATIVE single-model
dag-ml export reproduces *its own* final-(test) ``y_pred`` within ``1e-6``
(band ``native_export_reproduce``). That is a same-engine self-consistency
claim. The *cross*-engine leg — that a ``.n4a`` written by one engine and a
``.n4a`` written by the other are INTERCHANGEABLE at the predict boundary, and
that either reproduces what the dag-ml NATIVE (Rust) engine actually scored — is
unproven by that test. This module fills that gap.

For each representative native single-model regression shape:

* run the case on **legacy** (sklearn) and export a ``.n4a`` bundle;
* run the case on **dag-ml** (Rust, native results enabled) and export a
  ``.n4a`` bundle (transitional: the dag-ml export bridge legacy-refits the
  scored pipeline — A3 §8 — so this pins the BRIDGE round-trip today and
  tightens to ``native_export_reproduce`` when native ``.n4a`` export lands,
  DML-008/W3);
* load BOTH bundles via the public ``nirs4all.predict(model="...n4a", data=X)``
  path (detached, no workspace) and predict on the raw held-out test X;
* assert the two engines' bundle predictions agree within band
  ``cross_impl_ypred`` (``1e-3``) — the INTERCHANGE leg;
* assert the dag-ml bundle's predictions reproduce the dag-ml NATIVE run's
  final-(test) ``y_pred`` (mapped by sample id) within ``cross_impl_ypred`` —
  the CROSS-ENGINE round-trip leg (a portable bundle predicts what the Rust
  engine scored).

Scoped to regression PLS-family shapes so the comparison stays inside the
measured cross-impl noise floor (~``6e-4``); a case that runs the legacy
fallback or is not a single-artifact native run is SKIPPED (not failed) — the
cross-engine claim only applies when dag-ml ran natively. Slow: each case runs
twice. Gated by ``slow``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nirs4all
from nirs4all.data import DatasetConfigs

from . import _conformance_helpers as H
from ._datasets import dataset_path
from ._registry import PipelineCase, get

pytestmark = [pytest.mark.parity, pytest.mark.slow]


# Band cross_impl_ypred (compatibility.json): cross-implementation per-sample
# prediction tolerance. 1e-3 sits above the measured ~6e-4 PLS+inverse noise.
_CROSS_IMPL_YPRED_TOL = H._DEFAULT_YPRED_TOL  # noqa: SLF001

# Representative native single-model REGRESSION shapes: a no-preprocessing PLS,
# an n_components sweep, and a y_processing-inverse pipeline (the ~6e-4 noise
# case). Classification / RF shapes are intentionally excluded — their
# cross-engine row-order divergence is a separate (label) band.
_CROSS_ENGINE_CASES = (
    "baseline_vertical_slice",
    "generator_range_n_components",
    "round_trip_with_y_processing_inverse",
)


def _test_x(dataset_key: str) -> tuple[list[int], np.ndarray]:
    """The held-out test sample ids + their 2D feature matrix (raw, pre-preprocessing).

    Returned at the dataset's NATIVE storage dtype (float32) — the dtype the
    captured bundle predicts on. Mirrors ``test_conformance_export_roundtrip``.
    """
    base = DatasetConfigs(dataset_path(dataset_key)).get_dataset_at(0)
    ids = [int(s) for s in base.index_column("sample", {"partition": "test"})]
    return ids, np.asarray(base.x_rows(ids, layout="2d"))


def _bundle_predict(bundle_path: Path, x_test: np.ndarray) -> np.ndarray:
    """Load a ``.n4a`` bundle via the public predict path and predict on raw X.

    ``nirs4all.predict(model=path, data=X)`` is the documented detached
    (no-workspace) bundle prediction surface; the bundle embeds preprocessing,
    so it predicts on RAW test X and returns y_pred in test-sample order.
    """
    result = nirs4all.predict(model=str(bundle_path), data=x_test)
    return np.asarray(result.y_pred, dtype=float).ravel()


@pytest.mark.parametrize("case_name", _CROSS_ENGINE_CASES)
def test_n4a_bundle_cross_engine_round_trip(case_name: str, tmp_path: Path) -> None:
    """A legacy ``.n4a`` and a dag-ml ``.n4a`` predict alike, and reproduce dag-ml-native y_pred.

    Skips (not fails) when dag-ml runs the legacy fallback or is not a
    single-artifact native run — the cross-engine claim only holds for a native
    dag-ml run (mirrors ``test_conformance_export_roundtrip``).
    """
    case: PipelineCase = get(case_name)

    # engine EXPLICIT on both legs: resolve_engine honors $N4A_ENGINE, so an
    # engine=None dag-ml leg under N4A_ENGINE=legacy would silently run legacy
    # and make this a fake legacy-vs-legacy pass.
    legacy = nirs4all.run(pipeline=case.pipeline, dataset=H.make_dataset(case), verbose=0, engine="legacy")
    dagml = nirs4all.run(
        pipeline=case.pipeline, dataset=H.make_dataset(case), verbose=0, engine="dag-ml",
        results_path=str(tmp_path / "res"),
    )
    if not dagml._is_dagml_engine():  # noqa: SLF001
        pytest.skip(f"{case_name}: dag-ml ran the legacy fallback on this build; cross-engine N/A")
    if len(dagml._dagml_refit_artifacts) != 1:  # noqa: SLF001
        pytest.skip(f"{case_name}: not a single-artifact native run; covered by the bridge round-trip test")

    leg_n4a = legacy.export(tmp_path / "legacy.n4a")
    dml_n4a = dagml.export(tmp_path / "dagml.n4a")
    assert Path(leg_n4a).exists() and Path(dml_n4a).exists(), f"{case_name}: a .n4a export wrote no file"

    ids, x_test = _test_x(case.dataset_key)
    y_legacy_bundle = _bundle_predict(Path(leg_n4a), x_test)
    y_dagml_bundle = _bundle_predict(Path(dml_n4a), x_test)
    assert y_legacy_bundle.shape == y_dagml_bundle.shape == (len(ids),), (
        f"{case_name}: bundle y_pred shape mismatch — legacy={y_legacy_bundle.shape} "
        f"dag-ml={y_dagml_bundle.shape} n_test={len(ids)}"
    )

    # INTERCHANGE leg: the two engines' bundles predict alike on the same X.
    interchange_delta = float(np.max(np.abs(y_legacy_bundle - y_dagml_bundle)))
    assert interchange_delta <= _CROSS_IMPL_YPRED_TOL, (
        f"{case_name}: legacy vs dag-ml .n4a bundle predictions diverge; "
        f"max |Δ| = {interchange_delta:.3e} > band cross_impl_ypred {_CROSS_IMPL_YPRED_TOL:.3e}"
    )

    # CROSS-ENGINE round-trip leg: a portable bundle reproduces what the dag-ml
    # NATIVE (Rust) engine actually scored — mapped by sample id, not row order.
    native_by_sample = H._final_test_pred_by_sample(dagml)  # noqa: SLF001
    rows = [(i, ids[i]) for i in range(len(ids)) if ids[i] in native_by_sample]
    assert rows, f"{case_name}: no common dag-ml-native final-(test) samples to compare"
    expected = np.array([native_by_sample[sid].ravel()[0] for _, sid in rows])
    actual = np.array([y_dagml_bundle.reshape(len(ids), -1)[i][0] for i, _ in rows])
    native_delta = float(np.max(np.abs(expected - actual)))
    assert native_delta <= _CROSS_IMPL_YPRED_TOL, (
        f"{case_name}: dag-ml .n4a bundle predict != dag-ml NATIVE final-(test) y_pred; "
        f"max |Δ| = {native_delta:.3e} > band cross_impl_ypred {_CROSS_IMPL_YPRED_TOL:.3e}"
    )
