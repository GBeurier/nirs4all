"""Studio preset estimators run on the DAG-ML engine: fold-aware AOM-DA, PLS-DA probabilities, non-finite parameters."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold

import nirs4all
from nirs4all.data import SpectroDataset
from nirs4all.operators.models.sklearn.oplsda import OPLSDA
from nirs4all.operators.models.sklearn.plsda import PLSDA
from nirs4all.pipeline.dagml.operator_parameters import decode_constructor_value, encode_constructor_value
from nirs4all.pipeline.dagml_bridge import pipeline_to_dsl


def _dataset(targets: np.ndarray) -> SpectroDataset:
    features = np.random.default_rng(3).normal(size=(len(targets), 30))
    dataset = SpectroDataset("presets")
    dataset.add_samples(features + targets.reshape(-1, 1) * 0.5, indexes={"partition": "train"})
    dataset.add_targets(targets)
    return dataset


def _run(pipeline: list, dataset: SpectroDataset):
    return nirs4all.run(pipeline, dataset, verbose=0, save_artifacts=False)


def test_aom_pls_classifier_trains_on_the_pipeline_folds():
    from nirs4all.operators.models import AOMPLSClassifier

    labels = np.array([0, 1] * 30)
    result = _run([StratifiedKFold(3), {"model": AOMPLSClassifier(n_components=4)}], _dataset(labels))
    assert np.isfinite(result.cv_best_score)


@pytest.mark.parametrize("model", [PLSDA(3), OPLSDA()], ids=["plsda", "oplsda"])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_pls_da_probabilities_are_valid_and_match_predict(model, n_classes):
    rng = np.random.default_rng(5)
    labels = np.arange(60) % n_classes
    features = rng.normal(size=(60, 12)) + labels.reshape(-1, 1) * 3.0
    model.fit(features, labels)
    probe = np.vstack([features, rng.normal(scale=40.0, size=(20, 12))])  # far outside the training responses

    proba = model.predict_proba(probe)
    assert proba.shape == (80, n_classes)
    assert np.isfinite(proba).all() and (proba >= 0).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(model.classes_[np.argmax(proba, axis=1)], model.predict(probe))


def test_pls_da_classification_runs_natively():
    result = _run([KFold(3), {"model": PLSDA(4)}], _dataset(np.array([0, 1, 2] * 20)))
    assert np.isfinite(result.cv_best_score)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), np.float64("nan")])
def test_non_finite_constructor_values_round_trip_through_strict_json(value):
    encoded = json.loads(json.dumps(encode_constructor_value({"missing": value}), allow_nan=False))
    decoded = decode_constructor_value(encoded)["missing"]
    assert type(decoded) is float and (np.isnan(decoded) if np.isnan(value) else decoded == value)


def test_nan_constructor_defaults_reach_the_native_dsl_and_train():
    # SimpleImputer(missing_values=nan) is the same contract as XGBoost's missing=nan.
    json.dumps(pipeline_to_dsl([SimpleImputer(), {"model": Ridge()}]), allow_nan=False)
    result = _run([SimpleImputer(), KFold(3), {"model": Ridge()}], _dataset(np.linspace(0, 1, 60)))
    assert np.isfinite(result.cv_best_score)


@pytest.mark.parametrize("criterion", ["cv", "hybrid"])
@pytest.mark.parametrize("splitter", [StratifiedKFold(3), KFold(4)], ids=["stratified", "kfold"])
def test_aom_da_classifiers_accept_label_aware_splitters(criterion, splitter):
    from sklearn.model_selection import LeaveOneOut

    from nirs4all.operators.models import AOMPLSClassifier, POPPLSClassifier

    rng = np.random.default_rng(9)
    labels = np.arange(30) % 3
    features = rng.normal(size=(30, 20)) + labels.reshape(-1, 1)
    for model in (AOMPLSClassifier(n_components=3, criterion=criterion, cv=splitter),
                  POPPLSClassifier(n_components=3, criterion=criterion, cv=splitter),
                  AOMPLSClassifier(n_components=3, criterion=criterion, cv=LeaveOneOut())):
        assert set(model.fit(features, labels).predict(features)) <= {0, 1, 2}


def test_model_params_siblings_and_finetune_metadata_keep_strict_json():
    dsl = pipeline_to_dsl([
        {"model": Ridge(), "model_params": {"alpha": float("inf")}},
        {"model": SimpleImputer(), "fill_value": float("nan")},
        {"model": Ridge(), "finetune_params": {"model_params": {"alpha": [float("nan"), 1.0]}}},
    ])
    text = json.dumps(dsl, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
