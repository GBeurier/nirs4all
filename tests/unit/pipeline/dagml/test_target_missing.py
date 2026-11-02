"""Samples with a missing selected target are kept, never trained or scored, and predicted.

A training run excludes them through DAG-ML's ``FoldSet.target_missing_sample_ids``; the caller's
dataset keeps every sample, and the refit model predicts the excluded ones in a ``prediction`` row.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all
from nirs4all.data import SpectroDataset
from nirs4all.pipeline.dagml import in_process_runner
from nirs4all.pipeline.dagml.envelope import build_fold_set
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml.target_missing import find_target_missing, isolate_target_missing, missing_target_rows

MISSING = [3, 10, 25, 40]
OBSERVED = [index for index in range(60) if index not in MISSING]


def _features() -> np.ndarray:
    return np.random.default_rng(7).normal(size=(60, 24))


def _regression_targets() -> np.ndarray:
    features = _features()
    targets = features[:, :3].sum(axis=1) + 0.05 * features[:, 5]
    targets[MISSING] = np.nan
    return targets


def _dataset(features: np.ndarray, targets: np.ndarray, test: tuple[np.ndarray, np.ndarray] | None = None) -> SpectroDataset:
    dataset = SpectroDataset("target-missing")
    dataset.add_samples(features, indexes={"partition": "train"})
    dataset.add_targets(targets)
    if test is not None:
        dataset.add_samples(test[0], indexes={"partition": "test"})
        dataset.add_targets(test[1])
    return dataset


def _run(pipeline: list[Any], dataset: Any, **options: Any) -> Any:
    return nirs4all.run(pipeline, dataset, verbose=0, save_artifacts=False, **options)


def _prediction_rows(result: Any) -> list[dict[str, Any]]:
    return result.predictions.filter_predictions(partition="prediction")


@pytest.fixture
def declared_fold_sets(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture the fold set of every native CV/REFIT campaign the run executes."""
    captured: list[dict[str, Any]] = []
    original = in_process_runner.run_cv_refit_bundle

    def spy(**kwargs: Any) -> Any:
        captured.append(kwargs["dsl"]["split_invocation"]["fold_set"])
        return original(**kwargs)

    monkeypatch.setattr(in_process_runner, "run_cv_refit_bundle", spy)
    return captured


def test_missing_target_rows_detects_numeric_label_and_multi_target_gaps() -> None:
    assert missing_target_rows(np.array([1.0, np.nan, 2.0])).tolist() == [False, True, False]
    assert missing_target_rows(np.array(["a", None, "b"], dtype=object)).tolist() == [False, True, False]
    assert missing_target_rows(np.array([[1.0, np.nan], [2.0, 3.0]])).tolist() == [True, False]


def test_regression_run_matches_the_observed_cohort_and_predicts_missing_targets(declared_fold_sets: list[dict[str, Any]]) -> None:
    features, targets = _features(), _regression_targets()
    dataset = _dataset(features, targets)
    pipeline = [KFold(3), {"model": PLSRegression(3)}]

    result = _run(pipeline, dataset)
    observed_only = _run(pipeline, _dataset(features[OBSERVED], targets[OBSERVED]))

    # Training and scoring see exactly the target-observed cohort.
    assert result.cv_best_score == pytest.approx(observed_only.cv_best_score)
    for row in result.predictions.filter_predictions(partition="val"):
        assert not set(row["sample_indices"]) & set(MISSING)
    final_train = result.predictions.filter_predictions(partition="train", fold_id="final")[0]
    assert sorted(final_train["sample_indices"]) == OBSERVED
    assert result.final["partition"] == "train"

    # The fold set declares the excluded samples outside the CV universe.
    identity = mint_identity(dataset)
    declared = declared_fold_sets[0]
    assert declared["target_missing_sample_ids"] == [identity.to_wire(sample) for sample in MISSING]
    assert not set(declared["sample_ids"]) & set(declared["target_missing_sample_ids"])

    # The refit model predicts them; the caller's dataset is unchanged.
    (prediction,) = _prediction_rows(result)
    assert list(prediction["sample_indices"]) == MISSING
    assert prediction["fold_id"] == "final" and prediction["val_score"] is None and prediction["test_score"] is None
    reference = PLSRegression(3).fit(features[OBSERVED], targets[OBSERVED]).predict(features[MISSING]).ravel()
    np.testing.assert_allclose(prediction["y_pred"], reference, rtol=1e-5, atol=1e-5)
    assert result.per_dataset[dataset.name]["target_missing"] == {
        "policy": "excluded_from_training_and_scoring", "train_sample_indices": MISSING, "test_sample_indices": [],
        "count": len(MISSING), "predicted": True,
    }
    assert dataset.index_column("partition", {}) == ["train"] * 60
    assert np.isnan(dataset.y({"partition": "train"})).sum() == len(MISSING)


def test_missing_test_targets_are_predicted_not_scored() -> None:
    features, targets = _features(), _regression_targets()
    test_features = np.random.default_rng(8).normal(size=(12, 24))
    test_targets = test_features[:, :3].sum(axis=1)
    observed_test = [index for index in range(12) if index not in (2, 7)]
    with_missing = test_targets.copy()
    with_missing[[2, 7]] = np.nan
    pipeline = [MinMaxScaler(), KFold(3), {"model": PLSRegression(3)}]

    result = _run(pipeline, _dataset(features, targets, (test_features, with_missing)))
    observed_only = _run(pipeline, _dataset(features[OBSERVED], targets[OBSERVED], (test_features[observed_test], test_targets[observed_test])))

    assert result.final_score == pytest.approx(observed_only.final_score)
    (prediction,) = _prediction_rows(result)
    assert list(prediction["sample_indices"]) == [*MISSING, 62, 67]
    assert result.per_dataset["target-missing"]["target_missing"]["test_sample_indices"] == [62, 67]


@pytest.mark.parametrize("labels", [
    np.array([0.0, 1.0, 2.0] * 20),
    np.array(["low", "mid", "high"] * 20, dtype=object),
])
def test_classification_excludes_missing_labels_and_predicts_class_ids(labels: np.ndarray, declared_fold_sets: list[dict[str, Any]]) -> None:
    targets = labels.copy()
    targets[MISSING] = None if targets.dtype == object else np.nan
    result = _run([StandardScaler(), KFold(3), {"model": RandomForestClassifier(n_estimators=20, random_state=0)}], _dataset(_features(), targets))

    assert len(declared_fold_sets[0]["target_missing_sample_ids"]) == len(MISSING)
    (prediction,) = _prediction_rows(result)
    final_train = result.predictions.filter_predictions(partition="train", fold_id="final")[0]
    # Prediction rows share the run's class-id space, not the user's raw labels.
    assert set(np.asarray(prediction["y_pred"], dtype=float)) <= set(np.unique(final_train["y_true"]))


def test_sparse_multi_target_excludes_per_selected_targets() -> None:
    features = _features()
    targets = np.column_stack([features[:, :3].sum(axis=1), features[:, 3:6].sum(axis=1)])
    targets[[3, 10], 0] = np.nan
    targets[[25], 1] = np.nan

    both = find_target_missing(_dataset(features, targets))
    first_only = find_target_missing(_dataset(features, targets[:, 0]))

    assert both is not None and both.train == (3, 10, 25)
    assert first_only is not None and first_only.train == (3, 10)
    result = _run([KFold(3), {"model": PLSRegression(2)}], _dataset(features, targets))
    (prediction,) = _prediction_rows(result)
    assert np.asarray(prediction["y_pred"]).shape == (3, 2)


@pytest.mark.parametrize("pipeline", [
    pytest.param([KFold(3), {"model": PLSRegression(), "_range_": [1, 4, 1], "param": "n_components"}], id="native-sweep"),
    pytest.param([KFold(3), {"model": PLSRegression(), "finetune_params": {"n_trials": 3, "model_params": {"n_components": ("int", 1, 5)}}}], id="hpo"),
    pytest.param([KFold(3), {"branch": [[MinMaxScaler(), {"model": PLSRegression(3)}], [StandardScaler(), {"model": Ridge()}]]},
                  {"merge": "predictions"}, {"model": Ridge()}], id="nested-stacking"),
])
def test_sweeps_hpo_and_nested_stacking_declare_and_predict_missing_targets(pipeline: list[Any], declared_fold_sets: list[dict[str, Any]]) -> None:
    result = _run(pipeline, _dataset(_features(), _regression_targets()))

    assert declared_fold_sets
    assert all(len(fold_set["target_missing_sample_ids"]) == len(MISSING) for fold_set in declared_fold_sets)
    (prediction,) = _prediction_rows(result)
    assert list(prediction["sample_indices"]) == MISSING
    assert result.final is None or result.final["partition"] != "prediction"


def test_without_refit_the_exclusion_is_recorded_without_predictions() -> None:
    result = _run([KFold(3), {"model": PLSRegression(3)}], _dataset(_features(), _regression_targets()), refit=False)

    assert not _prediction_rows(result)
    record = result.per_dataset["target-missing"]["target_missing"]
    assert record["predicted"] is False and record["unpredicted_reason"] == "the run has no refit model"


def test_a_training_partition_without_observed_targets_is_refused() -> None:
    dataset = SpectroDataset("unlabelled-train")
    dataset.add_samples(_features(), indexes={"partition": "train"})
    dataset.add_samples(_features()[:6], indexes={"partition": "test"})
    dataset.add_targets(np.concatenate([np.full(60, np.nan), np.arange(6.0)]))
    with pytest.raises(ValueError, match="every training sample has a missing target"):
        _run([KFold(3), {"model": PLSRegression(3)}], dataset)


def test_build_fold_set_declares_only_held_out_samples() -> None:
    dataset = _dataset(_features(), _regression_targets())
    folds = [(OBSERVED[20:], OBSERVED[:20]), (OBSERVED[:20], OBSERVED[20:])]

    assert "target_missing_sample_ids" not in build_fold_set(mint_identity(dataset), folds)
    missing = find_target_missing(dataset)
    with isolate_target_missing(dataset, missing):
        identity = mint_identity(dataset)
        assert dataset.index_column("sample", {"partition": "train"}) == OBSERVED
        fold_set = build_fold_set(identity, folds)
    assert fold_set["target_missing_sample_ids"] == [identity.to_wire(sample) for sample in MISSING]
    assert dataset.index_column("partition", {}) == ["train"] * 60
    assert not hasattr(dataset, "_dagml_target_missing")


def test_metadata_branches_and_augmentation_predict_from_the_original_rows() -> None:
    dataset = _dataset(_features(), _regression_targets())
    dataset.add_metadata(pl.DataFrame({"site": ["a", "b"] * 30}))
    from nirs4all.operators.augmentation import GaussianAdditiveNoise

    for pipeline in (
        [KFold(3), {"branch": {"by_metadata": "site", "steps": [{"model": PLSRegression(3)}]}}, {"merge": "concat"}],
        [{"sample_augmentation": {"transformers": [GaussianAdditiveNoise(sigma=0.01)], "count": 1}}, KFold(3), {"model": PLSRegression(3)}],
    ):
        (prediction,) = _prediction_rows(_run(pipeline, dataset, random_state=0))
        assert list(prediction["sample_indices"]) == MISSING


def test_successive_models_share_folds_over_the_target_observed_samples(declared_fold_sets: list[dict[str, Any]]) -> None:
    features, targets = _features(), _regression_targets()
    pipeline = [MinMaxScaler(), KFold(3), {"model": PLSRegression(3)}, {"model": Ridge()}]

    result = _run(pipeline, _dataset(features, targets))
    reference = _run([MinMaxScaler(), KFold(3), {"model": PLSRegression(3)}], _dataset(features[OBSERVED], targets[OBSERVED]))

    # The shared FoldSet is KFold over the observed samples, not a filtered split of every sample.
    assert {len(fold["validation_sample_ids"]) for fold in declared_fold_sets[0]["folds"]} == {19, 18}
    pls = [row for row in result.predictions.filter_predictions(partition="val", fold_id="avg") if row["model_name"] == "PLSRegression"]
    assert pls[0]["val_score"] == pytest.approx(reference.cv_best_score)
    assert [list(row["sample_indices"]) for row in _prediction_rows(result)] == [MISSING, MISSING]


def test_independent_source_outputs_each_predict_missing_targets() -> None:
    rng = np.random.default_rng(11)
    first, second = rng.normal(size=(40, 10)), rng.normal(size=(40, 6))
    targets = first[:, 0] + second[:, 0]
    targets[[2, 9]] = np.nan
    dataset = SpectroDataset("sources")
    dataset.add_samples([first, second], indexes={"partition": "train"})
    dataset.add_targets(targets)
    names = ["source_0", "source_1"]
    pipeline = [KFold(2), {"branch": {"by_source": True, "steps": {name: [{"model": Ridge()}] for name in names}}}, {"merge": "auto"}]

    rows = {row["branch_name"]: row for row in _prediction_rows(_run(pipeline, dataset))}

    observed = [index for index in range(40) if index not in (2, 9)]
    for name, features in zip(names, (first, second), strict=True):
        reference = Ridge().fit(features[observed], targets[observed]).predict(features[[2, 9]])
        np.testing.assert_allclose(rows[name]["y_pred"], reference, rtol=1e-6, atol=1e-6)


def test_unlabelled_test_partition_is_not_a_target_gap() -> None:
    dataset = SpectroDataset("unlabelled-test")
    dataset.add_samples(_features(), indexes={"partition": "train"})
    dataset.add_targets(_regression_targets())
    dataset.add_samples(_features()[:5], indexes={"partition": "test"})

    missing = find_target_missing(dataset)
    assert missing is not None and missing.train == tuple(MISSING) and missing.test == ()


def test_fold_files_ignore_held_out_samples(tmp_path) -> None:
    fold_file = tmp_path / "folds.json"
    fold_file.write_text(json.dumps([{"train": list(range(30)), "val": list(range(30, 60))},
                                     {"train": list(range(30, 60)), "val": list(range(30))}]))
    result = _run([{"split": str(fold_file)}, {"model": PLSRegression(3)}], _dataset(_features(), _regression_targets()))

    (prediction,) = _prediction_rows(result)
    assert list(prediction["sample_indices"]) == MISSING
    for row in result.predictions.filter_predictions(partition="val"):
        assert not set(row["sample_indices"]) & set(MISSING)


def test_repetition_reshaping_with_missing_targets_is_refused() -> None:
    dataset = _dataset(_features(), _regression_targets())
    dataset.add_metadata(pl.DataFrame({"rep": [str(index // 2) for index in range(60)]}))
    with pytest.raises(Exception, match="rep_to_sources/rep_to_pp do not yet support"):
        _run([{"rep_to_pp": "rep"}, KFold(3), {"model": PLSRegression(3)}], dataset)


def test_native_results_keep_prediction_only_target_width() -> None:
    from nirs4all.data.predictions import Predictions
    from nirs4all.pipeline.dagml.native_results import _projection_rows

    predictions = Predictions()
    predictions.add_prediction(dataset_name="d", fold_id="final", partition="prediction", sample_indices=[0, 1],
                               y_pred=np.array([[1.0, 2.0], [3.0, 4.0]]), n_samples=2)
    (row,) = _projection_rows(predictions)
    assert row["target_width"] == 2 and json.loads(row["target_names"]) == ["y0", "y1"]
