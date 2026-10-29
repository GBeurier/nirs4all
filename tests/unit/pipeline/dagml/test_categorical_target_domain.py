"""Classifiers score and replay in their fitted discretizer's class domain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import nirs4all
from nirs4all.operators.transforms.targets import IntegerKBinsDiscretizer, RangeDiscretizer
from nirs4all.pipeline.config.component_serialization import serialize_component
from nirs4all.pipeline.dagml.errors import DagMlUnsupported
from nirs4all.pipeline.dagml.rt import RtError
from nirs4all.pipeline.dagml.target_capture import (
    CategoricalTargetTransform,
    captured_target_transform,
    pipeline_task_type,
    validate_classification_target_domains,
)


@pytest.fixture
def data():
    x = np.random.default_rng(29).normal(size=(48, 6))
    y = 3 * x[:, 0] + .1 * x[:, 1]
    return x, y, {"x_train": x[:36], "y_train": y[:36], "x_test": x[36:], "y_test": y[36:]}


@pytest.mark.parametrize("mode", ["fixed", "fixed_multiclass", "uniform", "quantile"])
def test_native_class_scores_and_archive_replay_keep_class_ids(mode, data, monkeypatch, tmp_path: Path):
    x, y, dataset = data
    original = y.copy()
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    fixed = mode.startswith("fixed")
    operator = RangeDiscretizer([-1., 1.] if mode == "fixed_multiclass" else [0.]) if fixed else IntegerKBinsDiscretizer(n_bins=2, strategy=mode, encode="ordinal")
    classes = {0, 1, 2} if mode == "fixed_multiclass" else {0, 1}
    fit_inputs = []
    original_fit = type(operator).fit

    def observe_fit(self, values, *args, **kwargs):
        fit_inputs.append(np.asarray(values).reshape(-1).copy())
        return original_fit(self, values, *args, **kwargs)

    monkeypatch.setattr(type(operator), "fit", observe_fit)
    pipeline = [{"y_processing": operator}]
    if fixed:
        pipeline.append(KFold(3, shuffle=True, random_state=7))
    pipeline.append(LogisticRegression())
    result = nirs4all.run(pipeline, dataset, engine="dag-ml", save_artifacts=False, save_charts=False, verbose=0)
    assert result.best.get("metric") == "balanced_accuracy"
    assert result.best.get("task_type") == "classification"
    assert fit_inputs
    if fixed:
        assert sorted(len(values) for values in fit_inputs) == [24, 24, 24, 36]
        expected = [y[:36][train] for train, _ in KFold(3, shuffle=True, random_state=7).split(x[:36])]
        for values in expected:
            assert any(np.array_equal(actual, values.astype(np.float32)) for actual in fit_inputs)
    else:
        assert len(fit_inputs) == 1
        np.testing.assert_array_equal(fit_inputs[0], y[:36].astype(np.float32))

    transform = result._dagml_refit_artifacts[0]["y_transform"]
    assert isinstance(transform, CategoricalTargetTransform)
    for row in result.predictions.filter_predictions(load_arrays=True):
        truth = np.asarray(row["y_true"]).reshape(-1)
        prediction = np.asarray(row["y_pred"]).reshape(-1)
        assert set(np.unique(truth)) <= classes
        assert set(np.unique(prediction)) <= classes
        indices = np.asarray(row["sample_indices"], dtype=int)
        np.testing.assert_array_equal(truth, transform.transform(y[indices].astype(np.float32).reshape(-1, 1)).reshape(-1))
        score = balanced_accuracy_score(truth, prediction)
        native = row.get(f"{row['partition']}_score")
        if native is not None:
            assert native == pytest.approx(score)

    artifact = result._dagml_refit_artifacts[0]
    expected_prediction = artifact["estimator"].predict(x[36:])
    archive = tmp_path / f"{mode}.n4a"
    result.export(archive)
    # Replay must consume the captured transform; fitting after export is forbidden.
    monkeypatch.setattr(type(operator), "fit", lambda *args, **kwargs: pytest.fail("replay refitted targets"))
    replayed = nirs4all.predict(archive, x[36:], verbose=0)
    np.testing.assert_array_equal(np.asarray(replayed.y_pred).reshape(-1), expected_prediction)
    scoring = replayed.scoring_targets(y[36:])
    assert scoring["task_type"] == ("multiclass_classification" if mode == "fixed_multiclass" else "binary_classification")
    assert scoring["validity_mask"].all()
    np.testing.assert_array_equal(scoring["values"].reshape(-1), transform.transform(y[36:].astype(np.float32).reshape(-1, 1)).reshape(-1))
    np.testing.assert_array_equal(y, original)


@pytest.mark.parametrize("operator", [
    IntegerKBinsDiscretizer(n_bins=2, strategy="uniform", encode="ordinal"),
    IntegerKBinsDiscretizer(n_bins=2, strategy="quantile", encode="ordinal"),
    RangeDiscretizer(2),
    Pipeline([("scale", StandardScaler()), ("bins", RangeDiscretizer([0.]))]),
])
def test_learned_cv_domains_refused_before_fit(operator, data, monkeypatch):
    monkeypatch.setattr("nirs4all.pipeline.dagml.target_domains.fold_local_domains_available", lambda: False)
    monkeypatch.setattr(type(operator), "fit", lambda *args, **kwargs: pytest.fail("unsupported CV must fail before fit"))
    with pytest.raises(RtError, match="partition-scoped target domains"):
        nirs4all.run([{"y_processing": operator}, KFold(3), LogisticRegression()], data[2],
                     engine="dag-ml", save_artifacts=False, save_charts=False, verbose=0)


@pytest.mark.parametrize("body", [
    lambda bins: [{"branch": [[{"y_processing": bins}, {"model": LogisticRegression()}]]}],
    lambda bins: [[{"y_processing": bins}, {"model": LogisticRegression()}]],
    lambda bins: [{"y_processing": [serialize_component(bins)]}],
])
def test_nested_target_domains_cannot_bypass_cv_guard(body):
    pipeline = [KFold(3), *body(IntegerKBinsDiscretizer(n_bins=2, encode="ordinal"))]
    with pytest.raises(DagMlUnsupported, match="one top-level|partition-scoped target domains"):
        validate_classification_target_domains(pipeline)


def test_model_task_is_resolved_for_each_public_checkpoint(data, monkeypatch):
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    result = nirs4all.run([KFold(3), Ridge(), {"y_processing": RangeDiscretizer([0.])}, LogisticRegression()], data[2],
                         engine="dag-ml", save_artifacts=False, save_charts=False, verbose=0)
    rows = result.predictions.filter_predictions(load_arrays=True)
    assert {row["task_type"] for row in rows} == {"regression", "classification"}
    assert {row["metric"] for row in rows} == {"rmse", "balanced_accuracy"}


def test_fixed_bin_cv_without_refit_matches_original_studio_trigger(data, monkeypatch):
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    result = nirs4all.run([{"y_processing": RangeDiscretizer([0.])}, KFold(3), LogisticRegression()], data[2],
                         engine="dag-ml", refit=False, save_artifacts=False, save_charts=False, verbose=0)
    assert result._dagml_refit_artifacts == []
    assert result.best["task_type"] == "classification"
    assert result.best["metric"] == "balanced_accuracy"
    assert np.isfinite(result.cv_best_score)
    for row in result.predictions.filter_predictions(load_arrays=True):
        assert set(np.unique(row["y_true"])) <= {0, 1}
        assert set(np.unique(row["y_pred"])) <= {0, 1}


def test_mixed_tasks_inside_one_native_graph_are_refused():
    with pytest.raises(DagMlUnsupported, match="cannot mix classification and regression"):
        pipeline_task_type([{"branch": [[Ridge()], [LogisticRegression()]]}], "regression")


def test_derived_classes_do_not_reuse_original_dataset_label_decoder():
    transform = CategoricalTargetTransform(RangeDiscretizer([0.]).fit([-1., 1.]))
    assert captured_target_transform(transform, object(), LogisticRegression()) is transform
    np.testing.assert_array_equal(transform.inverse_transform([0, 1]), [0, 1])


@pytest.mark.parametrize("targets", [
    [{"y_processing": Pipeline([("bins", RangeDiscretizer([0.])), ("scale", StandardScaler())])}],
    [{"y_processing": StandardScaler()}, {"y_processing": RangeDiscretizer([0.])}],
])
def test_ambiguous_target_domains_are_refused_even_without_cv(targets):
    with pytest.raises(DagMlUnsupported, match="one top-level"):
        validate_classification_target_domains([*targets, LogisticRegression()])


def test_raw_classifier_model_choices_cannot_bypass_target_domain_admission(data, monkeypatch):
    models = [LogisticRegression(), DecisionTreeClassifier()]
    for model in models:
        monkeypatch.setattr(type(model), "fit", lambda *args, **kwargs: pytest.fail("unqualified model-choice CV fitted"))
    pipeline = [{"y_processing": IntegerKBinsDiscretizer(n_bins=2, encode="ordinal")}, KFold(3), {"_or_": models}]
    assert pipeline_task_type(pipeline, "regression") == "classification"
    with pytest.raises(RtError, match="partition-scoped target domains"):
        nirs4all.run(pipeline, data[2], engine="dag-ml", save_artifacts=False, save_charts=False, verbose=0)


def test_model_choice_parameters_are_not_executable_model_alternatives():
    assert pipeline_task_type([{"model": Ridge(), "unused_parameter": LogisticRegression()}], "regression") == "regression"
    with pytest.raises(DagMlUnsupported, match="cannot mix classification and regression"):
        pipeline_task_type([{"_or_": [Ridge(), LogisticRegression()]}], "regression")


def test_untyped_model_cannot_use_dataset_fallback_to_bypass_binned_target_admission():
    class UntypedModel:
        def fit(self, x, y):
            return self

        def predict(self, x):
            return np.zeros(len(x))

    with pytest.raises(DagMlUnsupported, match="recognized classifier or regressor"):
        pipeline_task_type([{"y_processing": RangeDiscretizer(2)}, UntypedModel()], "regression")
