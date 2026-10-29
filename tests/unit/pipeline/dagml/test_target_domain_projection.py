"""Keep native macrofold selection separate from sample prediction aggregates."""

from copy import deepcopy

import numpy as np
import pytest

from nirs4all.pipeline.dagml.result import _scores_to_run_result, _variant_cv_score


def test_projection_preserves_fold_domains_and_native_macro_without_pooled_arrays():
    domain = {"schema_version": 1, "state_fingerprint": "a" * 64, "state_json": "{}",
              "recipe_fingerprint": "b" * 64, "fit_sample_ids": ["train:a", "train:b"], "class_labels": [0., 1.]}
    reports = [
        {"partition": "validation", "fold_id": fold, "variant_id": "variant:base", "producer_node": "model:terminal",
         "metrics": {"balanced_accuracy": score}, "target_domain": None if fold == "macro" else domain}
        for fold, score in [("fold0", .9), ("fold1", .3), ("macro", .6)]
    ]
    reports.extend({"partition": partition, "fold_id": None, "variant_id": "variant:base", "producer_node": "model:terminal",
                    "metrics": {"balanced_accuracy": score}, "target_domain": domain}
                   for partition, score in [("final", .8), ("test", .7)])
    scores = {"reports": reports}
    original = deepcopy(scores)
    result = _scores_to_run_result(scores, "dataset", "Classifier", metric="balanced_accuracy", task_type="classification")
    assert result.cv_best_score == .6
    assert result.cv_best["fold_id"] == "macro"
    assert _variant_cv_score(scores, "balanced_accuracy") == .6
    rows = result.predictions.filter_predictions(load_arrays=True)
    assert {row["fold_id"] for row in rows} == {"0", "1", "macro", "final"}
    macro = next(row for row in rows if row["fold_id"] == "macro")
    assert np.asarray(macro["y_true"]).size == 0
    assert np.asarray(macro["y_pred"]).size == 0
    assert macro["result_metadata"]["dagml_projection"]["score_provenance"]["val"]["aggregation"] == "unweighted_mean_of_partition_metrics"
    fold = next(row for row in rows if row["fold_id"] == "0")
    assert fold["result_metadata"]["dagml_projection"]["target_domains"]["val"] == domain
    final = next(row for row in rows if row["fold_id"] == "final")
    provenance = final["result_metadata"]["dagml_projection"]["score_provenance"]
    assert provenance["val"]["fold_id"] == "macro"
    assert provenance["val"]["purpose"] == "model_selection"
    assert final["val_score"] == .6
    assert scores == original
    result.close()


def test_projection_refuses_native_macro_and_pooled_scores_for_same_domain_campaign():
    scores = {"reports": [{"partition": "validation", "fold_id": fold, "variant_id": "variant:base",
                            "metrics": {"balanced_accuracy": .5}}
                           for fold in ["fold0", "avg", "macro"]]}
    with pytest.raises(ValueError, match="cannot mix macrofold selection"):
        _scores_to_run_result(scores, "dataset", "Classifier", metric="balanced_accuracy", task_type="classification")


def _cv_result(score, *, macro, recipe="b" * 64):
    domain = {"schema_version": 1, "state_fingerprint": "a" * 64, "state_json": "{}",
              "recipe_fingerprint": recipe, "fit_sample_ids": ["a", "b"], "class_labels": [0., 1.]}
    reports = [{"partition": "validation", "fold_id": fold, "variant_id": "variant:base",
                "metrics": {"balanced_accuracy": score}, "target_domain": domain if fold.startswith("fold") else None}
               for fold in ["fold0", "fold1", "macro" if macro else "avg"]]
    return _scores_to_run_result({"reports": reports}, "dataset", "Classifier", metric="balanced_accuracy", task_type="classification")


@pytest.mark.parametrize("mixed_pooled", [True, False])
def test_combined_results_refuse_cross_protocol_best_instead_of_promoting_macro(mixed_pooled):
    from nirs4all.pipeline.dagml.public_batch import DagMLBatchResult

    first = _cv_result(.8, macro=True)
    second = _cv_result(.99, macro=not mixed_pooled, recipe="c" * 64)
    batch = DagMLBatchResult([first, second])
    for getter in [lambda: batch.cv_best, lambda: batch.best]:
        with pytest.raises(ValueError, match="different native target-domain protocols"):
            getter()
    assert batch.runs[0].cv_best_score == .8
    assert batch.runs[1].cv_best_score == .99
    batch.close()


def test_combined_results_compare_macro_scores_only_with_identical_target_contexts():
    from nirs4all.pipeline.dagml.public_batch import DagMLBatchResult

    batch = DagMLBatchResult([_cv_result(.8, macro=True), _cv_result(.99, macro=True)])
    assert batch.cv_best_score == .99
    batch.close()
