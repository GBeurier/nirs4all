"""Regression coverage for the reviewed DT1 and VIZ prediction findings."""

import numpy as np
import pytest

from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


def _write_workspace(path, rows):
    """Persist realistic variant identities and distinguishable array evidence."""
    with WorkspaceStore(path) as store:
        run_id = store.begin_run("audit", config={}, datasets=[{"name": "data"}])
        for index, row in enumerate(rows):
            config = row.get("config", [{"model": "Ridge"}])
            pipeline_id = store.begin_pipeline(
                run_id=run_id, name="audit", expanded_config=config, generator_choices=[], dataset_name="data", dataset_hash="hash",
            )
            chain_id = store.save_chain(
                pipeline_id=pipeline_id, steps=[], model_step_idx=1, model_class="Ridge", preprocessings=row.get("preprocessings", ""),
                fold_strategy="per_fold", fold_artifacts={}, shared_artifacts={},
            )
            prediction_id = store.save_prediction(
                pipeline_id=pipeline_id, chain_id=chain_id, dataset_name="data", model_name=row.get("model_name", "Ridge"),
                model_class="Ridge", fold_id="0", partition="val", val_score=row["score"], metric=row.get("metric", "rmse"),
                preprocessings=row.get("preprocessings", ""), branch_id=row.get("branch_id"), n_samples=2, n_features=3,
                test_score=None, train_score=None, task_type="regression", scores={}, best_params={}, branch_name=None,
                exclusion_count=0, exclusion_rate=0.,
            )
            store.array_store.save_batch([{
                "prediction_id": prediction_id, "dataset_name": "data", "model_name": "Ridge", "fold_id": "0", "partition": "val",
                "y_true": np.array([1., 2.]), "y_pred": np.array([1., 2.]) + row["score"], "sample_indices": np.array([index * 2, index * 2 + 1]),
            }])


@pytest.mark.parametrize("policy", ["keep_best", "keep_existing", "overwrite"])
def test_merge_preserves_same_source_variants_and_arrays(tmp_path, policy):
    """DT1-03: same-named variants must not conflict with their own source."""
    source = tmp_path / "source"
    rows = [{"score": 0.3, "config": [{"model": "Ridge", "alpha": alpha}]} for alpha in (1, 2, 3)]
    _write_workspace(source, rows)
    report = Predictions.merge_stores([source], tmp_path / "target", on_conflict=policy)
    assert report.predictions_merged == 3
    assert report.conflicts_resolved == 0
    with Predictions(tmp_path / "target") as predictions:
        entries = predictions.iter_entries()
        assert len(entries) == 3
        assert sorted(tuple(entry["sample_indices"]) for entry in entries) == [(0, 1), (2, 3), (4, 5)]
        for entry in entries:
            np.testing.assert_allclose(entry["y_pred"], [1.3, 2.3])
    # The configuration survives merging, so re-merging identifies all variants.
    repeated = Predictions.merge_stores([source], tmp_path / "target", on_conflict="keep_existing")
    assert repeated.predictions_merged == 0
    assert repeated.conflicts_resolved == 3


@pytest.mark.parametrize("metric,scores,expected", [
    ("rmse", [0.1, 0.5], 0.1), ("rmse", [0.5, 0.1], 0.1),
    ("accuracy", [0.9, 0.5], 0.9), ("accuracy", [0.5, 0.9], 0.9),
    ("r2", [0.9, 0.5], 0.9), ("rmse", [0.1, 0.1], 0.1),
    ("rmse", [None, 0.1], 0.1), ("rmse", [0.1, None], 0.1),
])
def test_merge_keep_best_uses_metric_direction(tmp_path, metric, scores, expected):
    """DT1-07: skip worse sources; replace worse targets without stale FKs."""
    sources = [tmp_path / "first", tmp_path / "second"]
    for path, score in zip(sources, scores, strict=True):
        _write_workspace(path, [{"score": score if score is not None else 0.0, "metric": metric}])
        if score is None:
            with WorkspaceStore(path) as store:
                store._execute_with_retry("UPDATE predictions SET val_score = NULL", [])
    report = Predictions.merge_stores(sources, tmp_path / "target")
    assert report.conflicts_resolved == 1
    with WorkspaceStore(tmp_path / "target") as store:
        entries = store.query_predictions().to_dicts()
        assert len(entries) == 1
        assert entries[0]["val_score"] == expected
        assert store.get_chain(entries[0]["chain_id"]) is not None


@pytest.mark.parametrize("differing_field,first,second", [
    ("preprocessings", "SNV", "MSC"), ("branch_id", 0, 1),
    ("config", [{"alpha": 1}], [{"alpha": 2}]), ("metric", "rmse", "r2"),
])
def test_merge_distinct_specifications_do_not_conflict(tmp_path, differing_field, first, second):
    sources = [tmp_path / "first", tmp_path / "second"]
    for path, value in zip(sources, [first, second], strict=True):
        _write_workspace(path, [{"score": 0.1, differing_field: value}])
    report = Predictions.merge_stores(sources, tmp_path / "target")
    assert report.predictions_merged == 2
    assert report.conflicts_resolved == 0


@pytest.mark.parametrize("metric,scores,remaining", [
    ("rmse", [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3]),
    ("accuracy", [0.5, 0.6, 0.7, 0.8, 0.95], [0.7, 0.8, 0.95]),
    ("r2", [-0.5, 0.1, 0.5, 0.8, 0.9], [0.5, 0.8, 0.9]),
])
def test_remove_bottom_keeps_best_models(tmp_path, metric, scores, remaining):
    """DT1-02: verify survivors, not merely deletion counts."""
    path = tmp_path / "workspace"
    _write_workspace(path, [{"score": score, "metric": metric} for score in scores])
    with WorkspaceStore(path) as store:
        predictions = Predictions(store=store)
        result = predictions.remove_bottom(0.4)
        assert result["removed"] == 2
        assert result["remaining"] == 3
    with WorkspaceStore(path) as store:
        assert sorted(store.query_predictions()["val_score"].to_list()) == remaining


def test_remove_bottom_groups_incomparable_metrics(tmp_path):
    path = tmp_path / "workspace"
    _write_workspace(path, [{"score": value, "metric": metric} for metric, values in [("accuracy", [0.4, 0.9]), ("rmse", [10., 20.])] for value in values])
    with WorkspaceStore(path) as store:
        predictions = Predictions(store=store)
        result = predictions.remove_bottom(0.5)
        assert result == {"removed": 2, "remaining": 2, "threshold_score": None}
    with WorkspaceStore(path) as store:
        assert sorted(store.query_predictions()["val_score"].to_list()) == [0.9, 10.]


@pytest.mark.parametrize("fraction", [0., 0.01, 1.])
@pytest.mark.parametrize("dry_run", [True, False])
def test_remove_bottom_fraction_and_dry_run_counts(tmp_path, fraction, dry_run):
    """DT1-25: zero and rounding-to-zero cannot delete even one row."""
    path = tmp_path / "workspace"
    _write_workspace(path, [{"score": float(i)} for i in range(5)])
    with WorkspaceStore(path) as store:
        predictions = Predictions(store=store)
        result = predictions.remove_bottom(fraction, dry_run=dry_run)
    expected = 0 if dry_run else int(5 * fraction)
    assert result["removed"] == expected
    assert result["remaining"] == 5 - expected
    with WorkspaceStore(path) as store:
        assert store.query_predictions().height == 5 - expected


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), float("inf")])
def test_remove_bottom_rejects_invalid_fraction(tmp_path, fraction):
    path = tmp_path / "workspace"
    _write_workspace(path, [{"score": 0.1}])
    with WorkspaceStore(path) as store, pytest.raises(ValueError, match="fraction"):
        predictions = Predictions(store=store)
        predictions.remove_bottom(fraction)
    with WorkspaceStore(path) as store:
        assert store.query_predictions().height == 1


def test_top_preserves_validation_order_when_display_scores_disagree():
    """VIZ-01: downstream top-k slicing must retain validation selection."""
    predictions = Predictions()
    for name, val_score, test_score in [("CV winner", 0.1, 0.9), ("test winner", 0.2, 0.01)]:
        predictions.add_prediction(
            dataset_name="data", model_name=name, fold_id=0, partition="val", metric="rmse", val_score=val_score,
            scores={"val": {"rmse": val_score}, "test": {"rmse": test_score}},
        )
    results = predictions.top(1, group_by="model_name", score_scope="cv", display_metrics=["rmse"], aggregate_partitions=True)
    assert [row["model_name"] for row in results] == ["CV winner", "test winner"]
    assert results[0]["rank_score"] == 0.1
    assert results[0]["rmse"] == 0.9


@pytest.mark.parametrize("identity", ["dataset_name", "pipeline_uid", "branch_id", "chain_id"])
def test_final_deduplication_preserves_independent_outputs(identity):
    """DT1-04: independent finals survive while train/test twins collapse."""
    predictions = Predictions()
    for index, value in enumerate(["A", "B"] if identity != "branch_id" else [0, 1]):
        for partition in ["train", "test"]:
            entry = {"dataset_name": "data", "model_name": "Ridge", "fold_id": "final", "partition": partition, "metric": "rmse", "val_score": 0.1 + index}
            entry[identity] = value
            predictions.extend_from_list([entry])
    results = predictions.top(10, score_scope="refit")
    assert len(results) == 2
    assert [row[identity] for row in results] == ([0, 1] if identity == "branch_id" else ["A", "B"])
    assert all(row["partition"] == "test" for row in results)


@pytest.mark.parametrize("by_repetition", [None, "sample"])
def test_missing_display_partition_has_no_fabricated_metrics(by_repetition):
    """DT1-05: no test partition means no test display score."""
    predictions = Predictions()
    predictions.add_prediction(
        dataset_name="data", model_name="Ridge", fold_id=0, partition="val", metric="rmse", val_score=0.2,
        y_true=np.array([1., 2.]), y_pred=np.array([1.2, 2.2]), metadata={"sample": [1, 2]},
    )
    row = predictions.top(1, display_metrics=["rmse", "median_ae"], by_repetition=by_repetition)[0]
    assert row["rmse"] is None
    assert row["median_ae"] is None


def test_refit_metric_mismatch_rejected_without_reading_test_arrays():
    """DT1-10: changing a direction cannot reinterpret CV selection scores."""
    predictions = Predictions()
    for name, cv_score in [("good", 0.02), ("bad", 0.5)]:
        predictions.add_prediction(
            dataset_name="data", model_name=name, fold_id="final", partition="test", metric="rmse", val_score=cv_score,
            scores={"test": {"r2": 1000.}}, y_true=np.array([1., 2.]), y_pred=np.array([99., 99.]),
        )
    assert predictions.top(2, score_scope="refit")[0]["model_name"] == "good"
    with pytest.raises(ValueError, match="selection evidence uses 'rmse'"):
        predictions.top(2, score_scope="refit", rank_metric="r2")


def test_refit_default_uses_stored_classification_selection_metric():
    predictions = Predictions()
    for name, score in [("bad", 0.5), ("good", 0.9)]:
        predictions.add_prediction(dataset_name="data", model_name=name, fold_id="final", partition="test", metric="accuracy", task_type="classification", val_score=score)
    assert predictions.top(1, score_scope="refit")[0]["model_name"] == "good"


@pytest.mark.parametrize("identity", ["branch_id", "pipeline_uid", "preprocessings", "chain_id"])
def test_partition_lookup_keeps_matching_variant(identity):
    """DT1-11: enrichment cannot borrow a different branch's arrays."""
    predictions = Predictions()
    for index in [0, 1]:
        for partition in ["val", "train"]:
            row = {
                "dataset_name": "data", "config_name": "shared", "model_name": "Ridge", "fold_id": 0, "partition": partition,
                "metric": "rmse", "val_score": 0.1, "y_true": np.array([1., 2.]), "y_pred": np.array([1., 2.]) + index + 0.1,
            }
            row[identity] = index if identity == "branch_id" else str(index)
            predictions.extend_from_list([row])
    results = predictions.top(1, score_scope="cv", display_partition="train", display_metrics=["median_ae"], aggregate_partitions=True, **{identity: 1 if identity == "branch_id" else "1"})
    assert results[0]["median_ae"] == pytest.approx(1.1)
    assert results[0]["partitions"]["train"][identity] == (1 if identity == "branch_id" else "1")


@pytest.mark.parametrize("group_by", [None, "model_name"])
def test_group_by_fold_returns_top_n_per_fold(group_by):
    """DT1-24: fold grouping must affect selection and grouped output."""
    predictions = Predictions()
    for fold in [0, 1]:
        for score in [0.1, 0.2]:
            predictions.add_prediction(dataset_name="data", model_name="Ridge", fold_id=fold, partition="val", metric="rmse", val_score=score)
    assert len(predictions.top(1, score_scope="cv", group_by=group_by)) == 1
    grouped = predictions.top(1, score_scope="cv", group_by=group_by, group_by_fold=True, return_grouped=True)
    assert len(grouped) == 2
    assert {rows[0]["fold_id"] for rows in grouped.values()} == {"0", "1"}
    assert all(rows[0]["rank_score"] == 0.1 for rows in grouped.values())


@pytest.mark.parametrize("score_scope", ["cv", "folds"])
@pytest.mark.parametrize("group_by", [None, "model_name"])
@pytest.mark.parametrize("context_present", [False, True])
def test_cv_ranking_excludes_legacy_finals_without_refit_context(score_scope, group_by, context_present):
    """API-01/VIZ-01: a final fold is refit evidence even without optional context."""
    predictions = Predictions()
    final = {"dataset_name": "data", "model_name": "final winner", "fold_id": "final", "partition": "test", "metric": "rmse", "val_score": .01}
    if context_present:
        final["refit_context"] = None
    predictions.extend_from_list([final])
    predictions.add_prediction(dataset_name="data", model_name="CV winner", fold_id=0, partition="val", metric="rmse", val_score=.4)
    rows = predictions.top(10, score_scope=score_scope, group_by=group_by)
    assert [row["model_name"] for row in rows] == ["CV winner"]
    assert rows[0]["rank_score"] == .4
    assert predictions.top(1, score_scope="refit")[0]["model_name"] == "final winner"
