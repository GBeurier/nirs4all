"""CONFORMANCE: cross-engine ``.n4a`` BUNDLE round-trip parity (B-011).

The B-011 gap RV8 flagged: "workspace/artifact ``.n4a`` round-trip cross-engine"
is not covered by the oracle. The closest existing coverage stops short:

* ``test_parity_smoke.test_round_trip_bundle_export_load_predict`` round-trips an
  ``.n4a`` on the DEFAULT engine only and asserts merely ``preds is not None`` —
  no second engine, no numerical parity.
* ``test_conformance_export_roundtrip`` pins the ``export_model`` (joblib, single
  estimator) path and the ``.n4a`` *refusal* contract — never a full ``.n4a``
  bundle export → ``BundleLoader`` → predict compared across engines.

This module closes that gap for the two registry cases that declare the bundle-IO
contract (``RunResult.export()`` → ``BundleLoader.load()``).

dag-ml ``.n4a`` export has NO native path: ``RunResult.export(format="n4a")`` for
a dag-ml run re-fits the frozen pipeline through the legacy engine (the P1c
bridge — see ``RunResult.export`` / ``run_backend._attach_export_spec``). For a
fully-seeded deterministic single-model pipeline that refit is EXACT, which lets
us pin three independent links of the round-trip:

* **A — transitional bridge contract.** The dag-ml run's ``.n4a``, reloaded,
  reproduces the dag-ml run's NATIVE final-(test) ``y_pred`` within the
  cross-engine ``y_pred`` tolerance. This is the real "the exported bundle
  represents what dag-ml computed" claim (the bridge faithfully reconstructs the
  native run). Measured Δ ≤ 7.8e-4 (the y_processing-inverse + Ridge shape).
* **B — same-engine bundle-IO fidelity.** The legacy run's ``.n4a``, reloaded,
  reproduces the legacy run's final-(test) ``y_pred`` to float identity (it
  replays the same refit estimator on the same raw test X). Measured Δ = 0.0.
* **C — cross-engine bundle equivalence.** The legacy ``.n4a`` and the dag-ml
  ``.n4a`` predict identically on the same holdout X. Measured Δ = 0.0.

Skips (not fails) if the dag-ml engine ran the legacy fallback on this build —
then A/C are legacy-vs-legacy and trivially true; the contract under test is
dag-ml-native. The native/fallback truth is the suite's own
``dual_engine_runner`` signal (no fallback warning AND the per_dataset engine
marker), so this never silently widens the native boundary.

Slow: each case runs on both engines plus a bridge refit at export. Gated by
``slow``.

    pytest tests/integration/parity/test_conformance_n4a_bundle_parity.py -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline.bundle import BundleLoader

from . import _conformance_helpers as H
from ._datasets import dataset_path
from ._registry import PipelineCase, get

pytestmark = [pytest.mark.parity, pytest.mark.slow]

# Same-engine bundle-IO fidelity tolerance: the legacy ``.n4a`` replays the SAME
# refit estimator on the SAME raw test X, so its reload-predict must equal the
# legacy run's stored final-(test) y_pred to float identity. Measured Δ = 0.0;
# 1e-6 mirrors the exactness bar of test_conformance_export_roundtrip's native
# joblib path.
_BUNDLE_IO_EXACT_TOL = 1e-6

# Cross-engine y_pred tolerance for the bridge / cross-bundle comparisons. Equal
# to _conformance_helpers._DEFAULT_YPRED_TOL (the suite's measured cross-engine
# PLS+inverse noise ceiling); kept as a local literal so this new-file slice does
# not couple to a constant another lane may retune. Measured deltas here: bridge
# Δ ≤ 7.8e-4, cross-bundle Δ = 0.0 — both safely under 1e-3.
_CROSS_ENGINE_YPRED_TOL = 1e-3

# The two registry cases that declare the bundle round-trip contract. Both are
# fully-seeded single-model regression shapes (PLS is deterministic; Ridge and
# the splitter carry random_state=42), so the dag-ml export bridge refit is exact
# and the ``.n4a`` is reproducible. Explicit names pin exactly the shapes whose
# deltas were measured, mirroring test_conformance_export_roundtrip's _EXACT_CASES.
_ROUND_TRIP_CASES = (
    "round_trip_baseline_export_predict",
    "round_trip_with_y_processing_inverse",
)


def _holdout_test_x(dataset_key: str) -> tuple[list[int], np.ndarray]:
    """Held-out test sample ids + their raw 2D feature matrix at the dataset's native dtype.

    Native dtype (float32) is the dtype the dag-ml run predicts on and the bundle
    expects; feeding float64 would inject ~1e-6 noise that is not the bundle's
    fault. Mirrors ``test_conformance_export_roundtrip._test_x``.
    """
    base = DatasetConfigs(dataset_path(dataset_key)).get_dataset_at(0)
    ids = [int(s) for s in base.index_column("sample", {"partition": "test"})]
    return ids, np.asarray(base.x_rows(ids, layout="2d"))


def _bundle_predict_by_sample(result: object, output_path: Path, ids: list[int], x_test: np.ndarray) -> dict[int, np.ndarray]:
    """Export ``result`` to an ``.n4a``, reload via ``BundleLoader``, predict on ``x_test``, key by sample id.

    ``BundleLoader.predict`` returns predictions in the row order of ``x_test``;
    ``x_test`` rows correspond to ``ids`` (same order), so row ``i`` is sample
    ``ids[i]``.
    """
    bundle_path = result.export(str(output_path))  # type: ignore[attr-defined]
    assert Path(bundle_path).exists(), f"export wrote no bundle at {output_path}"
    preds = np.asarray(BundleLoader(bundle_path).predict(x_test), dtype=float).reshape(len(ids), -1)
    return {sid: preds[i] for i, sid in enumerate(ids)}


def _max_abs_delta(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> float:
    """Max per-sample ``|a - b|`` over the shared sample ids (asserts the id sets agree)."""
    a_ids, b_ids = set(a), set(b)
    assert a_ids == b_ids, f"sample-id sets diverge: a-only={sorted(a_ids - b_ids)[:5]} b-only={sorted(b_ids - a_ids)[:5]}"
    return float(max(np.max(np.abs(a[s].ravel() - b[s].ravel())) for s in a_ids))


@pytest.mark.parametrize("case_name", _ROUND_TRIP_CASES)
def test_n4a_bundle_roundtrip_cross_engine_parity(case_name: str, tmp_path: Path) -> None:
    """A dag-ml run's ``.n4a`` reload-predicts its native scores and equals the legacy ``.n4a``."""
    case: PipelineCase = get(case_name)
    dataset = H.make_dataset(case)

    duo = H.dual_engine_runner(case, dataset)
    legacy, dagml, native = duo["legacy"], duo["dag-ml"], duo["dagml_native"]
    if not native:
        pytest.skip(f"{case_name}: dag-ml ran legacy fallback on this build; cross-engine .n4a parity N/A")

    ids, x_test = _holdout_test_x(case.dataset_key)
    legacy_bundle = _bundle_predict_by_sample(legacy, tmp_path / "legacy.n4a", ids, x_test)
    dagml_bundle = _bundle_predict_by_sample(dagml, tmp_path / "dagml.n4a", ids, x_test)

    legacy_run = H._final_test_pred_by_sample(legacy)  # noqa: SLF001 -- canonical sample-keyed mapper, reused by the suite
    dagml_run = H._final_test_pred_by_sample(dagml)  # noqa: SLF001
    assert legacy_run and dagml_run, f"{case_name}: a run produced no per-sample final-(test) y_pred to compare"

    # B — same-engine bundle-IO fidelity: legacy .n4a == legacy run (measured Δ=0.0).
    legacy_fidelity = _max_abs_delta(legacy_bundle, legacy_run)
    assert legacy_fidelity <= _BUNDLE_IO_EXACT_TOL, (
        f"{case_name}: legacy .n4a reload-predict != legacy run final-(test) y_pred; "
        f"max Δ = {legacy_fidelity:.3e} > {_BUNDLE_IO_EXACT_TOL:.0e}"
    )

    # A — transitional bridge contract: dag-ml .n4a == dag-ml NATIVE run (cross-engine tol; measured Δ≤7.8e-4).
    bridge_delta = _max_abs_delta(dagml_bundle, dagml_run)
    assert bridge_delta <= _CROSS_ENGINE_YPRED_TOL, (
        f"{case_name}: dag-ml .n4a reload-predict != dag-ml NATIVE run final-(test) y_pred; "
        f"max Δ = {bridge_delta:.3e} > tol {_CROSS_ENGINE_YPRED_TOL:.0e}"
    )

    # C — cross-engine bundle equivalence: legacy .n4a == dag-ml .n4a (measured Δ=0.0).
    cross_engine_delta = _max_abs_delta(legacy_bundle, dagml_bundle)
    assert cross_engine_delta <= _CROSS_ENGINE_YPRED_TOL, (
        f"{case_name}: legacy .n4a and dag-ml .n4a predict differently on the same holdout X; "
        f"max Δ = {cross_engine_delta:.3e} > tol {_CROSS_ENGINE_YPRED_TOL:.0e}"
    )
