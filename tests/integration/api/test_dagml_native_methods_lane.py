"""Native Methods lane versus the host-callback lane under ``engine="dag-ml"``.

Pipelines of n4m role steps run through DAG-ML's callback-free Methods
estimator controllers. The host-callback lane (forced with
``N4A_DAGML_NATIVE_METHODS=0``) runs the same campaign with the same libn4m
through the n4m Python binding, so every CV score, OOF/train/refit prediction,
selected variant and exported model must agree.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

roles = pytest.importorskip("n4m.roles")
native = pytest.importorskip("dag_ml._dag_ml")
if not hasattr(native, "run_cv_refit_methods_in_process"):
    pytest.skip("the installed dag-ml lacks the native Methods CV/REFIT lane", allow_module_level=True)

import nirs4all  # noqa: E402

pytestmark = pytest.mark.methods

# Both lanes call the same libn4m kernels on the same float64 rows; any
# difference beyond this bound is a lowering defect, not rounding.
TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def spectra():
    rng = np.random.default_rng(7)
    scores = rng.normal(size=(60, 2))
    X = scores @ rng.normal(size=(2, 40)) + 1.0 + 0.05 * rng.normal(size=(60, 40))
    y = scores[:, 0] - 0.3 * scores[:, 1]
    classes = (scores[:, 0] > 0).astype(int) + (scores[:, 1] > 0.5).astype(int)
    return X, y, np.column_stack([y, scores[:, 1]]), classes


def run(pipeline, dataset, monkeypatch, *, native_lane: bool, **options):
    monkeypatch.setenv("N4A_DAGML_NATIVE_METHODS", "1" if native_lane else "0")
    options.setdefault("save_artifacts", False)
    return nirs4all.run(pipeline=pipeline, dataset=dataset, verbose=0, save_charts=False, engine="dag-ml", **options)


def rows(result) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        (row.get("partition"), str(row.get("fold_id")), row.get("config_name"), row.get("model_name")): row
        for row in result.predictions.filter_predictions(load_arrays=True)
    }


def assert_lanes_agree(native_result, callback_result) -> None:
    assert native_result.execution_lane == "native_methods"
    assert callback_result.execution_lane == "host_callback"
    assert math.isclose(native_result.cv_best_score, callback_result.cv_best_score, rel_tol=TOLERANCE, abs_tol=TOLERANCE)
    # Selected variant and its refit row.
    for selected in ("cv_best", "best"):
        for field in ("config_name", "model_name", "partition", "fold_id"):
            assert getattr(native_result, selected).get(field) == getattr(callback_result, selected).get(field)
    native_rows, callback_rows = rows(native_result), rows(callback_result)
    assert set(native_rows) == set(callback_rows)
    for key, callback_row in callback_rows.items():
        native_row = native_rows[key]
        for field in ("y_pred", "y_true", "y_proba"):
            if callback_row.get(field) is None:
                assert native_row.get(field) is None, (key, field)
                continue
            np.testing.assert_allclose(np.asarray(native_row[field], dtype=float), np.asarray(callback_row[field], dtype=float), rtol=TOLERANCE, atol=TOLERANCE, err_msg=f"{key} {field}")
        for field in ("val_score", "test_score", "train_score"):
            expected = callback_row.get(field)
            if expected is None:
                assert native_row.get(field) is None, (key, field)
            else:
                assert math.isclose(native_row[field], expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE), (key, field)
    # The native ScoreSet carries the same measurements report for report.
    native_reports = {_report_key(report): report for report in native_result._dagml_score_set["reports"]}
    callback_reports = {_report_key(report): report for report in callback_result._dagml_score_set["reports"]}
    assert set(native_reports) == set(callback_reports)
    for key, report in callback_reports.items():
        for metric, value in report["metrics"].items():
            assert math.isclose(native_reports[key]["metrics"][metric], value, rel_tol=TOLERANCE, abs_tol=TOLERANCE), (key, metric)


def _report_key(report: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(report.get(field) for field in ("producer_node", "variant_id", "partition", "fold_id", "level"))


@pytest.mark.parametrize(
    ("pipeline", "target", "options"),
    [
        pytest.param(lambda: [roles.SNV(), KFold(3), {"model": roles.CPPLS(n_components=3)}], "y", {}, id="snv-cppls"),
        pytest.param(
            lambda: [roles.SNV(), roles.VarianceFilter(top_k=20), ShuffleSplit(n_splits=3, test_size=0.25, random_state=1), {"model": roles.PLSRegression(n_components=4)}],
            "y", {}, id="selector-shufflesplit",
        ),
        pytest.param(lambda: [roles.SNV(), KFold(3), {"model": roles.PLSRegression(n_components=3)}], "multi", {}, id="multi-target"),
        pytest.param(lambda: [roles.SNV(), KFold(3), {"model": roles.BaggingPLS(n_estimators=5)}], "y", {}, id="seeded-default"),
        pytest.param(lambda: [roles.MSC(), KFold(3), {"model": roles.CPPLS(), "n_components": {"_range_": [1, 6, 2]}}], "y", {}, id="param-sweep"),
        pytest.param(lambda: [roles.SNV(), KFold(3), {"model": roles.CPPLS(), "n_components": {"_range_": [1, 6, 2]}}], "y", {"refit": {"top_k": 2}}, id="sweep-top-k"),
        pytest.param(lambda: [roles.SNV(), KFold(3), {"model": roles.CPPLS(n_components=3)}], "y", {"refit": False}, id="cv-only"),
        pytest.param(lambda: [{"exclude": roles.YOutlierFilter(threshold=1.0)}, roles.SNV(), KFold(3), {"model": roles.CPPLS(n_components=3)}], "y", {}, id="host-resolved-exclude"),
        pytest.param(lambda: [roles.SNV(), KFold(3), {"model": roles.PLSLogistic(n_components=2)}], "classes", {}, id="classifier-probabilities"),
        pytest.param(lambda: [KFold(4), {"model": roles.PLSLDA(n_components=3)}], "classes", {}, id="classifier"),
    ],
)
def test_native_lane_matches_host_callback_lane(spectra, monkeypatch, pipeline, target, options):
    X, y, multi, classes = spectra
    dataset = (X, {"y": y, "multi": multi, "classes": classes}[target])
    native_result = run(pipeline(), dataset, monkeypatch, native_lane=True, **options)
    callback_result = run(pipeline(), dataset, monkeypatch, native_lane=False, **options)
    assert_lanes_agree(native_result, callback_result)


def test_native_lane_runs_no_python_operator(spectra, monkeypatch):
    from nirs4all.pipeline.dagml import in_process_runner

    def forbidden(*args, **kwargs):
        raise AssertionError("the native Methods lane must not enter the host callback runner")

    monkeypatch.setattr(in_process_runner, "run_node", forbidden)
    monkeypatch.setattr(in_process_runner, "run_cv_refit_bundle", forbidden)
    X, y, _multi, _classes = spectra
    result = run([roles.SNV(), KFold(3), {"model": roles.CPPLS(n_components=3)}], (X, y), monkeypatch, native_lane=True)
    assert result.execution_lane == "native_methods"
    assert "execution_lane_reason" not in result.per_dataset["array_dataset"]
    artifact = result._dagml_refit_artifacts[0]
    assert artifact["controller_id"] == "controller:n4m.regressor"
    assert [type(step).__name__ for _, step in artifact["estimator"].steps] == ["SNV", "CPPLS"]


def test_ineligible_pipeline_records_why_it_kept_the_callback_lane(spectra, monkeypatch):
    X, y, _multi, _classes = spectra
    result = run([roles.SNV(), KFold(3), {"y_processing": MinMaxScaler()}, {"model": roles.CPPLS(n_components=3)}], (X, y), monkeypatch, native_lane=True)
    assert result.execution_lane == "host_callback"
    assert "is not an n4m transformer or selector" in result.per_dataset["array_dataset"]["execution_lane_reason"]


def test_exported_native_model_replays_like_the_callback_export(spectra, monkeypatch, tmp_path):
    X, y, _multi, _classes = spectra
    predictions = {}
    for native_lane in (True, False):
        workspace = tmp_path / f"workspace_{native_lane}"
        result = run([roles.SNV(), KFold(3), {"model": roles.CPPLS(n_components=3)}], (X[:50], y[:50]), monkeypatch, native_lane=native_lane, save_artifacts=True, workspace_path=workspace)
        archive = result.export(tmp_path / f"model_{native_lane}.n4a")
        predictions[native_lane] = np.asarray(nirs4all.predict(archive, X[50:]).y_pred, dtype=float)
        from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

        with WorkspaceStore(workspace) as store:
            summary = store.get_run(result.per_dataset["array_dataset"]["run_id"])["summary"]
        assert summary["execution_lane"] == ("native_methods" if native_lane else "host_callback")
    np.testing.assert_allclose(predictions[True], predictions[False], rtol=TOLERANCE, atol=TOLERANCE)
