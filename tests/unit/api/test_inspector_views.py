"""Read-only Inspector diagnostics and portable exports from real stored predictions."""

from __future__ import annotations

import base64
import io
import json
import zipfile

import numpy as np
import polars as pl
import pytest

from nirs4all.api.inspector_views import read_inspector_view
from nirs4all.pipeline.storage import WorkspaceStore
from nirs4all.pipeline.storage.prediction_export import export_prediction_arrays


@pytest.fixture
def populated(tmp_path):
    path = tmp_path / "workspace"
    with WorkspaceStore(path) as store:
        run = store.begin_run("Inspector", config={}, datasets=[{"name": "Coffee labels"}])
        pipeline = store.begin_pipeline(
            run_id=run, name="RF", expanded_config=[{"class": "sklearn.ensemble.RandomForestClassifier", "params": {"n_estimators": 5}}], generator_choices=[], dataset_name="Coffee labels", dataset_hash="hash"
        )
        chain = store.save_chain(
            pipeline_id=pipeline,
            steps=[{"step_idx": 0, "operator_class": "sklearn.ensemble.RandomForestClassifier", "params": {"n_estimators": 5}, "artifact_id": None, "stateless": False}],
            model_step_idx=0,
            model_class="RandomForestClassifier",
            preprocessings="SNV",
            fold_strategy="per_fold",
            fold_artifacts={},
            shared_artifacts={},
            dataset_name="Coffee labels",
        )
        prediction_ids = []
        for index, partition in enumerate(("val", "val", "test")):
            prediction = store.save_prediction(
                pipeline_id=pipeline,
                chain_id=chain,
                dataset_name="Coffee labels",
                model_name="RF",
                model_class="RandomForestClassifier",
                fold_id=f"fold_{index}",
                partition=partition,
                val_score=0.8 + index * 0.05,
                test_score=0.85,
                train_score=0.9,
                metric="balanced_accuracy",
                task_type="multiclass_classification",
                n_samples=3,
                n_features=2,
                scores={partition: {"balanced_accuracy": 0.8}},
                best_params={"n_estimators": 5},
                branch_id=None,
                branch_name=None,
                exclusion_count=0,
                exclusion_rate=0,
            )
            prediction_ids.append(prediction)
            store.array_store.save_batch(
                [
                    {
                        "prediction_id": prediction,
                        "dataset_name": "Coffee labels",
                        "model_name": "RF",
                        "fold_id": f"fold_{index}",
                        "partition": partition,
                        "metric": "balanced_accuracy",
                        "val_score": 0.8,
                        "task_type": "multiclass_classification",
                        "y_true": np.array(["A", "B", "A"]),
                        "y_pred": np.array(["A", "A", "A"]),
                        "sample_indices": np.arange(3),
                    }
                ]
            )
        store.update_chain_summary(chain)
        store.complete_pipeline(pipeline, best_val=0.8, best_test=0.85, metric="balanced_accuracy", duration_ms=5)
        store.complete_run(run, summary={"total_pipelines": 1})
    return path, pipeline, chain, prediction_ids


def test_chain_facets_and_text_label_confusion_are_readonly(populated):
    path, _, chain, _ = populated
    before = (path / "store.sqlite").read_bytes()
    data = read_inspector_view("inspector.data", str(path), {})
    assert data["total"] == 1
    assert data["chains"][0]["chain_id"] == chain
    assert data["available_datasets"] == ["Coffee labels"]
    assert data["available_preprocessings"] == ["SNV"]
    confusion = read_inspector_view("inspector.confusion", str(path), {"chain_ids": [chain], "partition": "test"})
    assert confusion["labels"] == ["A", "B"]
    assert sum(cell["count"] for cell in confusion["cells"]) == 3
    assert (path / "store.sqlite").read_bytes() == before


@pytest.mark.parametrize(
    "operation,view_request,key",
    [
        ("histogram", {}, "bins"),
        ("rankings", {}, "rankings"),
        ("heatmap", {"x_variable": "model_class", "y_variable": "preprocessings"}, "cells"),
        ("candlestick", {}, "categories"),
        ("branch-comparison", {}, "branches"),
        ("preprocessing-impact", {}, "entries"),
        ("hyperparameter", {"param_name": "n_estimators"}, "points"),
    ],
)
def test_summary_views_use_actual_scores(populated, operation, view_request, key):
    path, _, chain, _ = populated
    result = read_inspector_view(f"inspector.{operation}", str(path), view_request)
    if operation in {"preprocessing-impact", "hyperparameter"}:
        assert result[key] == []
        if operation == "hyperparameter":
            assert "enough variation" in result["reason"]
        else:
            assert result["total_chains"] == 1
    else:
        assert result[key]
    if operation == "rankings":
        assert result[key][0]["chain_id"] == chain
        assert result["sort_ascending"] is False
    json.dumps(result, allow_nan=False)


def test_fold_and_topology_views(populated):
    path, pipeline, chain, _ = populated
    folds = read_inspector_view("inspector.fold-stability", str(path), {"chain_ids": [chain]})
    assert len(folds["entries"]) == 2
    assert folds["fold_ids"] == ["fold_0", "fold_1"]
    topology = read_inspector_view("inspector.branch-topology", str(path), {"pipeline_id": pipeline})
    assert topology["pipeline_id"] == pipeline
    assert topology["nodes"]


def test_export_safe_dataset_name_filters_and_roundtrips_labels(populated):
    path, _, _, prediction_ids = populated
    exported = export_prediction_arrays(str(path), {"format": "parquet", "dataset_names": ["Coffee labels"], "partition": "test"})
    assert exported["filename"] == "Coffee_labels.parquet"
    frame = pl.read_parquet(io.BytesIO(base64.b64decode(exported["content_base64"])))
    assert frame["prediction_id"].to_list() == [prediction_ids[2]]
    assert frame["y_true_labels"].to_list()
    zipped = export_prediction_arrays(str(path), {"format": "zip"})
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(zipped["content_base64"]))) as archive:
        assert archive.namelist() == ["Coffee_labels.parquet"]
        assert pl.read_parquet(io.BytesIO(archive.read(archive.namelist()[0]))).height == 3


@pytest.mark.parametrize(
    "operation, view_request",
    [
        ("data", {"workspace_path": "/foreign"}),
        ("histogram", {"n_bins": 0}),
        ("rankings", {"limit": 501}),
        ("scatter", {"chain_ids": ["a"] * 257, "partition": "test"}),
        ("confusion", {"chain_ids": [], "target_index": -1}),
    ],
)
def test_invalid_requests_reject_before_read(tmp_path, operation, view_request):
    with pytest.raises(ValueError):
        read_inspector_view(f"inspector.{operation}", str(tmp_path), view_request)


def test_export_missing_dataset_is_explicit(populated):
    path, _, _, _ = populated
    with pytest.raises(ValueError, match="not_found"):
        export_prediction_arrays(str(path), {"format": "parquet", "dataset_names": ["missing"]})


def test_export_omits_removed_metadata_arrays(populated):
    path, _, _, ids = populated
    with WorkspaceStore(path) as store:
        with store._conn as connection:
            connection.execute("DELETE FROM predictions WHERE prediction_id = ?", [ids[0]])
    exported = export_prediction_arrays(str(path), {"format": "parquet", "dataset_names": ["Coffee labels"]})
    frame = pl.read_parquet(io.BytesIO(base64.b64decode(exported["content_base64"])))
    assert ids[0] not in frame["prediction_id"].to_list()
    assert frame.height == 2


def test_structured_score_reference_selects_partition(populated):
    path, pipeline, chain, _ = populated
    result = read_inspector_view("inspector.fold-stability", str(path), {"chain_ids": [chain], "score_ref": {"protocol": "cross_validation", "partition": "test", "aggregation": "fold_mean"}})
    assert result["score_column"] == "cv_test_score"
    assert result["fold_ids"] == ["fold_2"]
    assert result["entries"][0]["score"] == 0.85
    topology = read_inspector_view("inspector.branch-topology", str(path), {"pipeline_id": pipeline, "score_ref": json.dumps({"protocol": "cross_validation", "partition": "validation", "aggregation": "fold_mean"})})
    assert topology["nodes"]


def test_selected_chain_without_numeric_arrays_explains_bias_variance(populated):
    path, _, chain, _ = populated
    result = read_inspector_view("inspector.bias-variance", str(path), {"chain_ids": [chain]})
    assert result["entries"] == []
    assert "needs repeated validation predictions" in result["reason"]


def test_fold_diagnostics_ignore_aggregate_rows(populated):
    path, pipeline, chain, _ = populated
    with WorkspaceStore(path) as store:
        store.save_prediction(
            pipeline_id=pipeline,
            chain_id=chain,
            dataset_name="Coffee labels",
            model_name="RF",
            model_class="RandomForestClassifier",
            fold_id="avg",
            partition="val",
            val_score=0.99,
            test_score=0.99,
            train_score=0.99,
            metric="balanced_accuracy",
            task_type="multiclass_classification",
            n_samples=3,
            n_features=2,
            scores={},
            best_params={},
            branch_id=None,
            branch_name=None,
            exclusion_count=0,
            exclusion_rate=0,
        )
    result = read_inspector_view("inspector.fold-stability", str(path), {"chain_ids": [chain]})
    assert result["fold_ids"] == ["fold_0", "fold_1"]
    assert all(entry["model_class"] == "RandomForestClassifier" for entry in result["entries"])
