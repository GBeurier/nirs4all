"""CONFORMANCE: export_model → reload → predict round-trips on BOTH engines.

Pins the model-export contract for representative shapes. Two regimes, matching
how ``RunResult.export_model`` actually behaves:

* **NATIVE single-model dag-ml run** (regression single, classification, sweep,
  y_processing) — with native results enabled (``results_path=``) the dag-ml
  backend captures the fitted REFIT chain and ``export_model`` exports it
  DIRECTLY as a predict-capable ``_DagmlExportedModel`` (P3 Slice 2c-ii). The
  reloaded model's ``predict(X_test)`` reproduces the run's final-(test)
  ``y_pred`` EXACTLY (asserted within 1e-6; measured 0.0). The captured model
  embeds the preprocessing, so it predicts on RAW test X.

* **Multi-model / branch dag-ml run** (branch + merge) — the dag-ml path falls
  back to the legacy-refit bridge (≠1 captured artifact), and ``export_model``
  produces a model via that bridge. The merged feature space means predict on
  raw test X is not architecturally meaningful, so the contract here is the
  weaker "still round-trips": ``export_model`` writes a loadable artifact.

The legacy engine's ``export_model`` returns the BARE estimator at the model
step (no preprocessing wrapper), so its ``predict`` on raw X does not reproduce
the run's y_pred — the exactness claim is dag-ml-native only. For legacy we
assert the export round-trips (writes + loads).

Slow: each case runs twice. Gated by ``slow``.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

import nirs4all
from nirs4all.api.result import RunResult
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions

from . import _conformance_helpers as H
from ._datasets import dataset_path
from ._registry import PipelineCase, get

pytestmark = [pytest.mark.parity, pytest.mark.slow]


# Representative export cases: a regression single, a classification, a sweep,
# a branch/merge, and one with y_processing. ``exact`` flags the shapes whose
# native single-model export must reproduce final-(test) y_pred within 1e-6;
# the branch/merge shape only needs to round-trip (load).
_REGRESSION_SINGLE = "baseline_vertical_slice"
_CLASSIFICATION = "baseline_classification_rf_stratified"
_SWEEP = "generator_range_n_components"
_BRANCH_MERGE = "branch_dup_two_way_merge_features"
_Y_PROCESSING = "round_trip_with_y_processing_inverse"

_EXACT_CASES = (_REGRESSION_SINGLE, _CLASSIFICATION, _SWEEP, _Y_PROCESSING)


def _test_x(dataset_key: str) -> tuple[list[int], np.ndarray]:
    """The held-out test sample ids + their 2D feature matrix (raw, pre-preprocessing).

    X is returned at the dataset's NATIVE storage dtype (float32) — the same dtype the dag-ml RUN
    predicts on (the resolver/node_runner no longer widen X to float64). Feeding the reloaded export
    model float64 would diverge it from the run's final-(test) y_pred (the captured estimator is
    dtype-faithful; only the test input differs), so the round-trip oracle uses native X.
    """
    base = DatasetConfigs(dataset_path(dataset_key)).get_dataset_at(0)
    ids = [int(s) for s in base.index_column("sample", {"partition": "test"})]
    return ids, np.asarray(base.x_rows(ids, layout="2d"))


def _final_test_by_sample(result) -> dict[int, np.ndarray]:
    """Map sample-id → final-(test) y_pred vector for ``result`` (delegates to the helper)."""
    return H._final_test_pred_by_sample(result)  # noqa: SLF001


def _minimal_dagml_result(*, export_spec: dict[str, object] | None = None) -> RunResult:
    """Construct a lightweight dag-ml result for export-surface contract tests."""
    return RunResult(
        predictions=Predictions(),
        per_dataset={"toy": {"engine": "dag-ml"}},
        _dagml_export_spec=export_spec,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_fragment"),
    [
        ({"source": {"prediction_id": "p0"}}, "source=/chain_id"),
        ({"chain_id": "chain-0"}, "source=/chain_id"),
    ],
)
def test_dagml_n4a_export_rejects_workspace_selectors_before_legacy_refit(
    kwargs: dict[str, object],
    expected_fragment: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit workspace selectors on a dag-ml result fail fast and do not refit.

    ``source=`` and ``chain_id=`` name legacy workspace records. A dag-ml run has
    no such workspace, so the transitional export bridge must raise a catchable
    ``NotImplementedError`` before materializing the legacy delegate.
    """
    result = _minimal_dagml_result(export_spec={"pipeline": [], "dataset": object()})

    def _unexpected_delegate() -> object:
        raise AssertionError("dag-ml export delegate must not materialize for explicit workspace selectors")

    monkeypatch.setattr(result, "_dagml_export_delegate", _unexpected_delegate)

    with pytest.raises(NotImplementedError, match=expected_fragment):
        result.export(tmp_path / "model.n4a", **kwargs)

    assert result._dagml_legacy_result is None  # noqa: SLF001


def test_dagml_n4a_export_without_workspace_or_spec_is_catchable(tmp_path: Path) -> None:
    """A dag-ml result with no export spec raises ``NotImplementedError``, not a legacy misuse error."""
    result = _minimal_dagml_result()

    with pytest.raises(NotImplementedError, match="engine='dag-ml'.*no workspace artifacts"):
        result.export(tmp_path / "model.n4a", source={"prediction_id": "p0"})


@pytest.mark.parametrize("case_name", _EXACT_CASES)
def test_native_dagml_export_reproduces_final_test_pred(case_name: str, tmp_path: Path) -> None:
    """NATIVE dag-ml export reload-predicts the run's final-(test) y_pred within 1e-6.

    Skips (not fails) if the case does not run native dag-ml on this build — the
    exactness contract only applies to the native single-model export path.
    """
    case: PipelineCase = get(case_name)
    dataset = H.make_dataset(case)

    # results_path enables native results → the dag-ml run captures the fitted
    # REFIT model so export_model takes the direct native single-artifact path.
    # engine="dag-ml" is EXPLICIT (not None): resolve_engine honors $N4A_ENGINE, so
    # an engine=None leg under N4A_ENGINE=legacy would silently run legacy and SKIP
    # this whole test (the native path never triggers) — forcing dag-ml prevents that.
    result = nirs4all.run(
        pipeline=case.pipeline, dataset=dataset, verbose=0, engine="dag-ml", results_path=str(tmp_path / "res")
    )
    if not result._is_dagml_engine():  # noqa: SLF001
        pytest.skip(f"{case_name}: dag-ml ran legacy fallback on this build; native export N/A")
    if len(result._dagml_refit_artifacts) != 1:  # noqa: SLF001
        pytest.skip(f"{case_name}: not a single-artifact native run; covered by the bridge round-trip test")

    ids, x_test = _test_x(case.dataset_key)
    reloaded = H.round_trip_export_reload_predict(result, x_test, tmp_path / "model.joblib")

    by_sample = _final_test_by_sample(result)
    rows = [(i, ids[i]) for i in range(len(ids)) if ids[i] in by_sample]
    assert rows, f"{case_name}: no common final-(test) samples to compare"
    expected = np.array([by_sample[sid].ravel() for _, sid in rows]).ravel()
    actual = np.array([reloaded.reshape(len(ids), -1)[i] for i, _ in rows]).ravel()
    assert expected.shape == actual.shape
    max_delta = float(np.max(np.abs(expected - actual)))
    assert max_delta <= 1e-6, (
        f"{case_name}: reloaded native export predict != run final-(test) y_pred; max Δ = {max_delta:.3e}"
    )


def test_branch_merge_export_round_trips_via_bridge(tmp_path: Path) -> None:
    """A branch/merge run (multi-model → legacy-refit bridge) still round-trips.

    The branch+merge shape falls back to the legacy engine (≠1 native artifact),
    so ``export_model`` produces a model via the bridge. Its raw-X predict is not
    meaningful (the model expects the merged feature space), so the contract is
    that the export WRITES a loadable artifact on both engines.
    """
    case: PipelineCase = get(_BRANCH_MERGE)
    dataset = H.make_dataset(case)

    # Both engines EXPLICIT — "dag-ml" (not None) so $N4A_ENGINE=legacy cannot hijack
    # the dag-ml leg into running legacy; "legacy" forces the legacy orchestrator.
    for engine in ("legacy", "dag-ml"):
        result = nirs4all.run(
            pipeline=case.pipeline, dataset=dataset, verbose=0, engine=engine,
            results_path=str(tmp_path / f"res_{engine}"),
        )
        out = tmp_path / f"model_{engine}.joblib"
        path = result.export_model(out)
        assert Path(path).exists(), f"{_BRANCH_MERGE} (engine={engine!r}): export_model wrote no file"
        loaded = joblib.load(path)
        assert hasattr(loaded, "predict"), (
            f"{_BRANCH_MERGE} (engine={engine!r}): reloaded artifact is not predict-capable"
        )
