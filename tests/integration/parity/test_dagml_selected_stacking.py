"""Public parity oracles for selected and weighted stacking inputs."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.models.meta import MetaModel, StackingConfig
from nirs4all.operators.models.meta import TestAggregation as FoldAggregation
from nirs4all.pipeline.dagml.detect import (
    _detect_proba_mean_stacking_branch,
    _detect_sequential_metamodel,
)

from ._datasets import dataset_path


def _data():
    return make_regression(n_samples=48, n_features=6, noise=0.1, random_state=42)


def test_sequential_metamodel_selects_named_source(tmp_path):
    meta_by_source = {}
    for source in ("PLSRegression", "Ridge"):
        pipeline = [
            KFold(3, shuffle=True, random_state=42),
            {"model": PLSRegression(n_components=2)},
            {"model": Ridge(alpha=10000)},
            {"model": MetaModel(Ridge(alpha=1), source_models=[source])},
        ]
        detected = _detect_sequential_metamodel(pipeline)
        assert detected is not None
        assert detected[2] == [{"model": "branch:0.node:0"}]
        legacy = nirs4all.run(pipeline, _data(), engine="legacy", refit=False,
                             save_artifacts=False, save_charts=False, verbose=0)
        assert np.isfinite(legacy.cv_best_score)
        native = nirs4all.run(pipeline, _data(), engine="dag-ml", refit=False,
                             allow_fallback=False, workspace_path=tmp_path / source,
                             save_artifacts=False, save_charts=False, verbose=0)
        try:
            assert native.execution_engine == "dag-ml"
            assert np.isfinite(native.cv_best_score)
            meta_by_source[source] = {
                block["fold_id"]: np.asarray(block["values"], dtype=float)
                for node in native._dagml_node_results
                for block in node.get("predictions", [])
                if block.get("producer_node") == "merge:stack"
                and block.get("partition") == "validation"
            }
        finally:
            native.close()
    assert meta_by_source["PLSRegression"].keys() == meta_by_source["Ridge"].keys()
    assert any(
        not np.allclose(meta_by_source["PLSRegression"][fold], meta_by_source["Ridge"][fold])
        for fold in meta_by_source["PLSRegression"]
    )


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_selected_source_archive_matches_native_final_test(tmp_path, monkeypatch, mechanism):
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    pipeline = [
        KFold(3, shuffle=True, random_state=42),
        {"model": PLSRegression(n_components=2)},
        {"model": Ridge(alpha=10000)},
        {"model": MetaModel(Ridge(alpha=1), source_models=["PLSRegression"])},
    ]
    path = dataset_path("regression")
    native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False,
                         workspace_path=tmp_path / "train", save_artifacts=False,
                         save_charts=False, verbose=0)
    try:
        archive = native.export(tmp_path / "selected.n4a")
        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        y_test = np.asarray(dataset.y({"partition": "test"})).ravel()
        replay = np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel()
        assert replay.shape == y_test.shape
        assert np.sqrt(np.mean((y_test - replay) ** 2)) == pytest.approx(native.best_rmse, rel=1e-6, abs=1e-6)
    finally:
        native.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("aggregation", [FoldAggregation.BEST_FOLD, FoldAggregation.WEIGHTED_MEAN])
def test_sequential_fold_test_features_and_archive_replay(tmp_path, monkeypatch, mechanism, aggregation):
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    pipeline = [
        KFold(3, shuffle=True, random_state=42),
        PLSRegression(n_components=2),
        Ridge(alpha=10000),
        {"model": MetaModel(Ridge(alpha=1), stacking_config=StackingConfig(
            test_aggregation=aggregation,
        ))},
    ]
    path = dataset_path("regression")
    legacy = nirs4all.run(pipeline, path, engine="legacy", refit=False,
                          save_artifacts=False, save_charts=False, verbose=0)
    assert np.isfinite(legacy.cv_best_score)
    native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False,
                         workspace_path=tmp_path / "train", save_artifacts=False,
                         save_charts=False, verbose=0)
    try:
        assert native.execution_engine == "dag-ml"
        assert np.isfinite(native.best_rmse)
        fold_models = [artifact["estimator"] for artifact in native._dagml_refit_artifacts
                       if hasattr(artifact["estimator"], "fold_estimators")]
        assert len(fold_models) == 2
        if aggregation == FoldAggregation.BEST_FOLD:
            assert all(estimator.selected_fold in estimator.fold_estimators for estimator in fold_models)
        else:
            assert all(estimator.weights is not None and sum(estimator.weights.values()) == pytest.approx(1.0)
                       for estimator in fold_models)
        archive = native.export(tmp_path / "best_fold.n4a")
        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        y_test = np.asarray(dataset.y({"partition": "test"})).ravel()
        if aggregation == FoldAggregation.WEIGHTED_MEAN:
            for estimator, model_name in zip(fold_models, ("PLSRegression", "Ridge"), strict=True):
                legacy_average = next(row for row in legacy.predictions._buffer
                                      if row.get("model_name") == model_name and row.get("partition") == "test"
                                      and row.get("fold_id") == "w_avg")
                np.testing.assert_allclose(np.asarray(estimator.predict(x_test)).ravel(),
                                           np.asarray(legacy_average["y_pred"]).ravel(), atol=1e-4)
        replay = np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel()
        assert replay.shape == y_test.shape
        assert np.sqrt(np.mean((y_test - replay) ** 2)) == pytest.approx(native.best_rmse, rel=1e-6, abs=1e-6)
    finally:
        native.close()


def test_legacy_weighted_test_aggregation_uses_inverse_rmse_average():
    pipeline = [
        KFold(3, shuffle=True, random_state=42),
        PLSRegression(n_components=2),
        Ridge(alpha=10000),
        {"model": MetaModel(Ridge(alpha=1), stacking_config=StackingConfig(
            test_aggregation=FoldAggregation.WEIGHTED_MEAN,
        ))},
    ]
    legacy = nirs4all.run(pipeline, dataset_path("regression"), engine="legacy", refit=False,
                          save_artifacts=False, save_charts=False, verbose=0)
    try:
        rows = [row for row in legacy.predictions._buffer
                if row.get("model_name") == "PLSRegression" and row.get("partition") == "test"]
        folds = sorted((row for row in rows if str(row.get("fold_id", "")).isdigit()),
                       key=lambda row: int(row["fold_id"]))
        aggregate = next(row for row in rows if row.get("fold_id") == "w_avg")
        assert len(folds) == 3
        errors = np.asarray([row["val_score"] for row in folds], dtype=float)
        predictions = np.asarray([row["y_pred"] for row in folds], dtype=float)
        np.testing.assert_allclose(np.asarray(aggregate["y_pred"]), np.average(predictions, axis=0, weights=1.0 / errors))
        assert not np.allclose(np.asarray(aggregate["y_pred"]),
                               np.average(predictions, axis=0, weights=errors))
    finally:
        legacy.close()


def test_branch_numeric_prediction_aggregation(tmp_path):
    meta_by_aggregate = {}
    for aggregate in ("mean", "weighted_mean"):
        pipeline = [
            KFold(3, shuffle=True, random_state=42),
            {"branch": [
                [{"model": PLSRegression(n_components=2)}, {"model": Ridge(alpha=10000)}],
                [{"model": Ridge(alpha=1)}, {"model": Ridge(alpha=1000)}],
            ]},
            {"merge": {"predictions": [
                {"branch": 0, "aggregate": aggregate},
                {"branch": 1, "aggregate": aggregate},
            ]}},
            {"model": Ridge(alpha=0.1)},
        ]
        detected = _detect_proba_mean_stacking_branch(pipeline)
        assert detected is not None
        assert [selector["aggregate"] for selector in detected[2]] == [aggregate, aggregate]
        legacy = nirs4all.run(pipeline, _data(), engine="legacy", refit=False,
                             save_artifacts=False, save_charts=False, verbose=0)
        assert np.isfinite(legacy.cv_best_score)
        native = nirs4all.run(pipeline, _data(), engine="dag-ml", refit=False,
                             allow_fallback=False, workspace_path=tmp_path / aggregate,
                             save_artifacts=False, save_charts=False, verbose=0)
        try:
            assert native.execution_engine == "dag-ml"
            assert np.isfinite(native.cv_best_score)
            meta_by_aggregate[aggregate] = {
                block["fold_id"]: np.asarray(block["values"], dtype=float)
                for node in native._dagml_node_results
                for block in node.get("predictions", [])
                if block.get("producer_node") == "merge:stack"
                and block.get("partition") == "validation"
            }
        finally:
            native.close()
    assert meta_by_aggregate["mean"].keys() == meta_by_aggregate["weighted_mean"].keys()
    assert any(
        not np.allclose(meta_by_aggregate["mean"][fold], meta_by_aggregate["weighted_mean"][fold])
        for fold in meta_by_aggregate["mean"]
    )


@pytest.mark.parametrize("selection", ["best", {"top_k": 2}])
def test_branch_model_selection_feeds_native_stacking(tmp_path, selection):
    pipeline = [
        KFold(3, shuffle=True, random_state=42),
        {"branch": [
            [{"model": PLSRegression(n_components=2)}, {"model": Ridge(alpha=10000)}],
            [{"model": Ridge(alpha=1)}, {"model": Ridge(alpha=1000)}],
        ]},
        {"merge": {"predictions": [
            {"branch": 0, "select": selection, "metric": "rmse"},
            {"branch": 1, "select": selection, "metric": "rmse"},
        ]}},
        {"model": Ridge(alpha=0.1)},
    ]
    detected = _detect_proba_mean_stacking_branch(pipeline)
    assert detected is not None
    assert [selector["select"] for selector in detected[2]] == [selection, selection]
    legacy = nirs4all.run(
        pipeline, _data(), engine="legacy", refit=False,
        save_artifacts=False, save_charts=False, verbose=0,
    )
    assert np.isfinite(legacy.cv_best_score)
    native = nirs4all.run(
        pipeline, _data(), engine="dag-ml", refit=False, allow_fallback=False,
        workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0,
    )
    try:
        assert native.execution_engine == "dag-ml"
        assert np.isfinite(native.cv_best_score)
    finally:
        native.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_best_branch_selection_archive_replays_fitted_meta_features(tmp_path, monkeypatch, mechanism):
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    pipeline = [
        KFold(3, shuffle=True, random_state=42),
        {"branch": [
            [{"model": PLSRegression(n_components=2)}, {"model": Ridge(alpha=10000)}],
            [{"model": Ridge(alpha=1)}, {"model": Ridge(alpha=1000)}],
        ]},
        {"merge": {"predictions": [
            {"branch": 0, "select": "best", "metric": "rmse"},
            {"branch": 1, "select": "best", "metric": "rmse"},
        ]}},
        {"model": Ridge(alpha=0.1)},
    ]
    path = dataset_path("regression")
    native = nirs4all.run(
        pipeline, path, engine="dag-ml", allow_fallback=False, refit=True,
        workspace_path=tmp_path / "native", save_artifacts=False, save_charts=False, verbose=0,
    )
    try:
        archive = native.export(tmp_path / "selected.n4a")
        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        y_test = np.asarray(dataset.y({"partition": "test"})).ravel()
        replay = np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel()
        assert replay.shape == y_test.shape
        assert np.sqrt(np.mean((y_test - replay) ** 2)) == pytest.approx(native.best_rmse, rel=1e-6, abs=1e-6)
    finally:
        native.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_weighted_branch_archive_matches_native_final_test(tmp_path, monkeypatch, mechanism):
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "subprocess" else "1")
    pipeline = [
        KFold(3, shuffle=True, random_state=42),
        {"branch": [
            [{"model": PLSRegression(n_components=2)}, {"model": Ridge(alpha=10000)}],
            [{"model": Ridge(alpha=1)}, {"model": Ridge(alpha=1000)}],
        ]},
        {"merge": {"predictions": [
            {"branch": 0, "aggregate": "weighted_mean"},
            {"branch": 1, "aggregate": "weighted_mean"},
        ]}},
        {"model": Ridge(alpha=0.1)},
    ]
    path = dataset_path("regression")
    native = nirs4all.run(pipeline, path, engine="dag-ml", allow_fallback=False,
                         workspace_path=tmp_path / "train", save_artifacts=False,
                         save_charts=False, verbose=0)
    try:
        archive = native.export(tmp_path / "weighted.n4a")
        dataset = DatasetConfigs(path).get_dataset_at(0)
        x_test = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
        y_test = np.asarray(dataset.y({"partition": "test"})).ravel()
        replay = np.asarray(nirs4all.predict(archive, x_test).y_pred).ravel()
        assert replay.shape == y_test.shape
        assert np.sqrt(np.mean((y_test - replay) ** 2)) == pytest.approx(native.best_rmse, rel=1e-6, abs=1e-6)
    finally:
        native.close()
