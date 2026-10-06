"""Shared final prediction identity and conflicting evidence regressions."""
from copy import deepcopy

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, ShuffleSplit

import nirs4all
from nirs4all.api.result import RunResult
from nirs4all.data.predictions import Predictions


def _row(**changes):
    row = {"id": "final-one", "pipeline_uid": "pipeline-one", "run_id": "run-one",
           "dataset_name": "spectra", "partition": "train", "fold_id": "final",
           "model_name": "PLSRegression", "model_classname": "PLSRegression",
           "metric": "rmse", "task_type": "regression", "n_samples": 3, "n_features": 1,
           "sample_indices": [0, 1, 2], "refit_context": "standalone",
           "y_true": np.array([[1.], [2.], [3.]]), "y_pred": np.array([[1.], [2.], [3.]]),
           "val_score": None, "test_score": None, "train_score": 0., "scores": {},
           "metadata": {"source": "spectra"}, "created_at": "buffer-time", "op_counter": 0}
    row.update(changes)
    return row


def _predictions(rows):
    predictions = Predictions()
    predictions.extend_from_list(rows)
    return predictions


def _result(global_rows, dataset_rows):
    return RunResult(_predictions(global_rows), {"spectra": {"run_predictions": _predictions(dataset_rows)}})


def test_shared_unscored_final_selected_without_mutation():
    row = _row()
    result = _result([row], [row])
    assert len(result._refit_predictions().filter_predictions(fold_id="final")) == 1
    assert result.final["id"] == row["id"]
    assert result.best["id"] == row["id"]
    assert result.predictions.filter_predictions(fold_id="final")[0] is row


def test_reloaded_projection_preserves_arrays_and_aliases():
    row = _row(weights=np.array([np.nan]))
    persisted = deepcopy(row)
    persisted.update(prediction_id=row["id"], pipeline_id=row["pipeline_uid"], model_class=row["model_classname"], created_at="store-time", op_counter=1)
    persisted["y_true"] = row["y_true"].ravel()
    persisted["y_pred"] = row["y_pred"].ravel()
    persisted["weights"] = []
    merged = _result([row], [persisted])._refit_predictions().filter_predictions(fold_id="final")[0]
    np.testing.assert_array_equal(merged["y_pred"], row["y_pred"])
    assert merged["prediction_id"] == row["id"]
    assert "prediction_id" not in row
    assert persisted["y_pred"].shape == (3,)
    metadata_only = deepcopy(persisted)
    metadata_only["y_pred"] = []
    merged = _result([metadata_only], [row])._refit_predictions().filter_predictions(fold_id="final")[0]
    assert merged["y_pred"] is row["y_pred"]


@pytest.mark.parametrize("field,value", [("id", "other"), ("pipeline_uid", "other"), ("dataset_name", "other"), ("partition", "test"), ("run_id", "other")])
def test_distinct_final_identity_remains_ambiguous(field, value):
    result = _result([_row()], [_row(**{field: value})])
    assert len(result._refit_predictions().filter_predictions(fold_id="final")) == 2
    assert result.final is None
    assert result.best == {}


@pytest.mark.parametrize("field,value", [("n_features", 2), ("refit_context", "other"), ("model_name", "other"), ("metric", "mae"), ("val_score", .1), ("sample_indices", [2, 1, 0]), ("metadata", {"source": "other"}), ("y_pred", np.array([4., 5., 6.])), ("weights", np.array([1., 2., 3.]))])
def test_conflicting_evidence_refused(field, value):
    first = _row(weights=np.ones(3))
    second = deepcopy(first)
    second[field] = value
    with pytest.raises(ValueError, match="Conflicting final prediction evidence"):
        _result([first], [second]).final
    np.testing.assert_array_equal(first["y_pred"], [[1.], [2.], [3.]])


def test_identity_alias_conflict_refused():
    with pytest.raises(ValueError, match="Conflicting final prediction identity"):
        _result([_row()], [_row(prediction_id="foreign")]).final


def test_real_legacy_refit_overlap_and_reopened_store(tmp_path):
    x = np.array([[31.], [30.1], [33.], [35.], [36.], [37.]])
    result = nirs4all.run(
        [ShuffleSplit(n_splits=1, test_size=.5, random_state=42), {"model": PLSRegression(n_components=1), "name": "PLSRegression"}],
        (x, x.ravel()), name="final-identity-regression", engine="legacy", verbose=0,
        save_artifacts=False, save_charts=False, random_state=42, refit=True, workspace_path=tmp_path)
    try:
        finals = result._refit_predictions().filter_predictions(fold_id="final")
        assert len(finals) == 1
        assert result.final["id"] == finals[0]["id"]
        assert result.best["id"] == finals[0]["id"]
        assert result.final_score is None
        with Predictions.from_workspace(tmp_path, load_arrays=True) as reopened:
            stored = reopened.filter_predictions(fold_id="final", load_arrays=True)
            assert len(stored) == 1
            merged = _result(finals, stored)._refit_predictions().filter_predictions(fold_id="final")
            assert len(merged) == 1
            np.testing.assert_allclose(merged[0]["y_pred"].ravel(), x.ravel(), atol=1e-6)
    finally:
        result.close()


def test_scored_finals_and_cv_scalar_selection_remain_stable():
    cv = _row(id="cv", fold_id="0", partition="val", refit_context=None, val_score=.2, scores={"val": {"rmse": .2}})
    winner = _row(selection_score=.2, val_score=.2, test_score=.3, scores={"test": {"rmse": .3}})
    loser = _row(id="final-two", pipeline_uid="pipeline-two", selection_score=.8, val_score=.8, test_score=.1)
    result = _result([cv, winner], [deepcopy(winner), loser])
    assert result.final["id"] == winner["id"]
    assert result.final_score == .3
    assert result.cv_best["id"] == cv["id"]
    assert result.cv_best_score == .2
    assert result.best["id"] == winner["id"]
    assert result.best_score == .3
    assert result.best_rmse == .3


@pytest.mark.parametrize("field", ["run_id", "pipeline_uid"])
def test_optional_context_recovered_only_from_unique_bound_view(field):
    row = _row()
    unbound = deepcopy(row)
    unbound.pop(field)
    result = _result([unbound], [row])
    assert len(result._refit_predictions().filter_predictions(fold_id="final")) == 1
    assert result.final["id"] == row["id"]
    assert field not in unbound
    other = _row(**{field: "foreign"})
    ambiguous = _result([unbound], [row, other])
    assert len(ambiguous._refit_predictions().filter_predictions(fold_id="final")) == 3
    assert ambiguous.final is None


def test_missing_prediction_id_never_invents_identity():
    unbound = _row()
    unbound.pop("id")
    assert _result([unbound], [deepcopy(unbound)]).final is None


@pytest.mark.parametrize("engine", [None, "dag-ml"])
@pytest.mark.parametrize("cv", [False, True])
def test_current_engine_full_training_and_cv_selection(tmp_path, engine, cv):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(24, 4))
    y = x @ np.arange(1., 5.)
    pipeline = ([KFold(3)] if cv else []) + [Ridge(alpha=.01)]
    options = {} if engine is None else {"engine": engine}
    result = nirs4all.run(pipeline, (x, y), verbose=0, save_artifacts=False, save_charts=False,
                          refit=True, random_state=42, workspace_path=tmp_path, **options)
    try:
        assert result.execution_engine == "dag-ml"
        finals = result._refit_predictions().filter_predictions(fold_id="final")
        assert len({row["id"] for row in finals}) == 1
        assert result.final["id"] == finals[0]["id"]
        assert result.best["id"] == finals[0]["id"]
        if cv:
            assert np.isfinite(result.cv_best_score)
        else:
            assert result.final_score is None
    finally:
        result.close()


def test_storage_named_metadata_and_nonempty_provenance_conflicts_refused():
    first = _row(metadata={"created_at": "first", "id": "source-first"}, model_artifact_id="artifact-first")
    for change in ({"metadata": {"created_at": "second", "id": "source-first"}}, {"metadata": {"created_at": "first", "id": "source-second"}}, {"model_artifact_id": "artifact-second"}):
        second = deepcopy(first)
        second.update(change)
        with pytest.raises(ValueError, match="Conflicting final prediction evidence"):
            _result([first], [second]).final
