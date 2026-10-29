"""Real native learned-bin CV uses partition domains and macrofold selection."""

from copy import deepcopy

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import KBinsDiscretizer

import nirs4all
from nirs4all.operators.transforms.targets import IntegerKBinsDiscretizer, RangeDiscretizer
from nirs4all.pipeline.dagml.errors import DagMlUnsupported
from nirs4all.pipeline.dagml.target_capture import validate_classification_target_domains
from nirs4all.pipeline.dagml.target_domains import fold_local_domains_available


@pytest.mark.skipif(not fold_local_domains_available(), reason="native fold-local target domains schema v1 unavailable")
@pytest.mark.parametrize("mode", ["range", "uniform", "quantile"])
@pytest.mark.parametrize("n_bins", [2, 3])
def test_learned_cv_native_scores_domains_and_archive_replay(mode, n_bins, monkeypatch, tmp_path):
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    x = np.random.default_rng(29).normal(size=(48, 6))
    y = 3 * x[:, 0] + .1 * x[:, 1]
    source = y.copy()
    operator = RangeDiscretizer(n_bins) if mode == "range" else IntegerKBinsDiscretizer(n_bins=n_bins, strategy=mode, encode="ordinal")
    fitted = []
    original_fit = type(operator).fit

    def observe_fit(self, values, *args, **kwargs):
        result = original_fit(self, values, *args, **kwargs)
        fitted.append((np.asarray(values).reshape(-1).copy(), deepcopy(self)))
        return result

    monkeypatch.setattr(type(operator), "fit", observe_fit)
    splitter = KFold(3, shuffle=True, random_state=7)
    result = nirs4all.run([{"y_processing": operator}, splitter, LogisticRegression()],
                         {"x_train": x[:36], "y_train": y[:36], "x_test": x[36:], "y_test": y[36:]},
                         engine="dag-ml", save_artifacts=False, save_charts=False, verbose=0)
    assert sorted(len(values) for values, _ in fitted) == [24, 24, 24, 36]
    folds = list(splitter.split(x[:36]))
    transforms = {}
    for index, (train, _) in enumerate(folds):
        matches = [transform for values, transform in fitted if np.array_equal(values, y[train].astype(np.float32))]
        assert len(matches) == 1
        transforms[str(index)] = matches[0]
    transform = result._dagml_refit_artifacts[0]["y_transform"]
    transforms["final"] = transform
    domains = [frame["lineage"]["target_domain"] for frame in result._dagml_node_results
               if frame["lineage"]["phase"] == "FIT_CV" and frame["lineage"].get("target_domain") is not None]
    assert len(domains) == 3
    assert len({domain["state_fingerprint"] for domain in domains}) >= 2
    rows = result.predictions.filter_predictions(load_arrays=True)
    assert not {"avg", "w_avg"} & {row["fold_id"] for row in rows}
    validation_scores = []
    for row in rows:
        if row["fold_id"] == "macro":
            assert np.asarray(row["y_true"]).size == np.asarray(row["y_pred"]).size == 0
            continue
        indices = np.asarray(row["sample_indices"], dtype=int)
        truth = np.asarray(row["y_true"]).reshape(-1)
        prediction = np.asarray(row["y_pred"]).reshape(-1)
        expected = transforms[row["fold_id"]].transform(y[indices].astype(np.float32).reshape(-1, 1)).reshape(-1)
        np.testing.assert_array_equal(truth, expected)
        assert set(np.unique(prediction)) <= set(range(n_bins))
        score = balanced_accuracy_score(truth, prediction)
        assert row[f"{row['partition']}_score"] == pytest.approx(score)
        if row["partition"] == "val":
            validation_scores.append(score)
        domain = row["result_metadata"]["dagml_projection"]["target_domains"][row["partition"]]
        if row["fold_id"] != "final":
            train_row = next(item for item in rows if item["fold_id"] == row["fold_id"] and item["partition"] == "train")
            assert domain["fit_sample_ids"] == sorted(train_row["metadata"]["physical_sample_id"])
    assert result.cv_best["fold_id"] == "macro"
    assert result.cv_best_score == pytest.approx(np.mean(validation_scores))
    assert result.best["task_type"] == "classification"
    archive = tmp_path / f"{mode}.n4a"
    result.export(archive)
    monkeypatch.setattr(type(operator), "fit", lambda *args, **kwargs: pytest.fail("PREDICT refitted bins"))
    predicted = nirs4all.predict(archive, x[36:], verbose=0)
    np.testing.assert_array_equal(np.asarray(predicted.y_pred).reshape(-1), result._dagml_refit_artifacts[0]["estimator"].predict(x[36:]))
    assert predicted.metadata["target_domain"] == transform.target_domain_
    scoring = predicted.scoring_targets(y[36:])
    assert scoring["task_type"] == ("binary_classification" if n_bins == 2 else "multiclass_classification")
    assert scoring["validity_mask"].all()
    final_test = next(row for row in rows if row["fold_id"] == "final" and row["partition"] == "test")
    assert balanced_accuracy_score(scoring["values"], predicted.y_pred) == pytest.approx(final_test["test_score"])
    assert scoring["target_domain"] == transform.target_domain_
    np.testing.assert_array_equal(y, source)


@pytest.mark.parametrize("refit,top_k", [(False, 1), (True, 2)])
def test_learned_cv_refuses_unqualified_refit_strategy_before_fitting(refit, top_k, monkeypatch):
    monkeypatch.setattr("nirs4all.pipeline.dagml.target_domains.fold_local_domains_available", lambda: True)
    with pytest.raises(DagMlUnsupported, match="partition-scoped target domains"):
        validate_classification_target_domains(
            [{"y_processing": RangeDiscretizer(2)}, KFold(3), LogisticRegression()], refit=refit, refit_top_k=top_k)


@pytest.mark.skipif(not fold_local_domains_available(), reason="native fold-local target domains schema v1 unavailable")
@pytest.mark.parametrize("random_subsample", [False, True])
def test_native_parameter_grid_requires_identical_domains_between_candidate_classifiers(random_subsample, monkeypatch):
    import dag_ml

    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    x = np.random.default_rng(29).normal(size=(48, 6))
    y = 3 * x[:, 0] + .1 * x[:, 1]
    bins = KBinsDiscretizer(n_bins=2, strategy="quantile", encode="ordinal", subsample=11 if random_subsample else None, random_state=None)
    pipeline = [{"y_processing": bins}, KFold(3, shuffle=True, random_state=7), {"model": LogisticRegression(), "_grid_": {"C": [.1, 1.]}}]

    def run():
        return nirs4all.run(pipeline, {"x_train": x[:36], "y_train": y[:36], "x_test": x[36:], "y_test": y[36:]},
                            engine="dag-ml", random_state=19, save_artifacts=False, save_charts=False, verbose=0)

    if random_subsample:
        with pytest.raises(dag_ml.DagMlRuntimeError, match="candidate classifiers.*not comparable"):
            run()
    else:
        result = run()
        assert result.cv_best["fold_id"] == "macro"
        assert np.isfinite(result.cv_best_score)
