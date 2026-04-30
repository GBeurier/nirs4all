"""Tests for AOMPLSDAClassifier and POPPLSDAClassifier."""

from __future__ import annotations

import numpy as np
import pytest

from aompls.classification import AOMPLSDAClassifier, POPPLSDAClassifier
from aompls.metrics import (
    balanced_accuracy,
    expected_calibration_error,
    log_loss,
    macro_f1,
)
from aompls.synthetic import make_classification


@pytest.fixture
def binary_data():
    return make_classification(n_train=80, n_test=40, p=120, n_classes=2, random_state=3)


@pytest.fixture
def multiclass_data():
    return make_classification(n_train=120, n_test=60, p=140, n_classes=3, random_state=4)


def test_binary_predict_shapes(binary_data):
    ds = binary_data
    clf = AOMPLSDAClassifier(max_components=4, criterion="covariance")
    clf.fit(ds.X_train, ds.y_train)
    pred = clf.predict(ds.X_test)
    proba = clf.predict_proba(ds.X_test)
    assert pred.shape == (ds.X_test.shape[0],)
    assert proba.shape == (ds.X_test.shape[0], 2)


def test_proba_bounds_and_sum(binary_data):
    ds = binary_data
    clf = AOMPLSDAClassifier(max_components=3, criterion="covariance")
    clf.fit(ds.X_train, ds.y_train)
    proba = clf.predict_proba(ds.X_test)
    assert np.all(proba >= 0.0 - 1e-9) and np.all(proba <= 1.0 + 1e-9)
    sums = proba.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_multiclass_classification(multiclass_data):
    ds = multiclass_data
    clf = AOMPLSDAClassifier(max_components=4, criterion="covariance")
    clf.fit(ds.X_train, ds.y_train)
    pred = clf.predict(ds.X_test)
    acc = balanced_accuracy(ds.y_test, pred)
    assert acc > 0.5  # synthetic data is well separable


def test_pop_classifier(multiclass_data):
    ds = multiclass_data
    clf = POPPLSDAClassifier(max_components=3, criterion="covariance")
    clf.fit(ds.X_train, ds.y_train)
    pred = clf.predict(ds.X_test)
    assert pred.shape == (ds.X_test.shape[0],)
    seq = clf.get_selected_operators()
    assert len(seq) >= 1


def test_class_imbalance(binary_data):
    """Heavily imbalanced dataset must still produce normalised probabilities."""
    ds = binary_data
    rng = np.random.default_rng(11)
    keep_minor = rng.permutation(np.where(ds.y_train == 1)[0])[: max(2, len(np.where(ds.y_train == 1)[0]) // 8)]
    keep_major = np.where(ds.y_train == 0)[0]
    keep_idx = np.concatenate([keep_major, keep_minor])
    Xtr = ds.X_train[keep_idx]
    ytr = ds.y_train[keep_idx]
    clf = AOMPLSDAClassifier(max_components=3, criterion="covariance")
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(ds.X_test)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_calibration_metrics(multiclass_data):
    ds = multiclass_data
    clf = AOMPLSDAClassifier(max_components=4, criterion="covariance")
    clf.fit(ds.X_train, ds.y_train)
    proba = clf.predict_proba(ds.X_test)
    ll = log_loss(ds.y_test, proba, classes=clf.classes_)
    ece = expected_calibration_error(ds.y_test, proba, n_bins=5)
    assert ll >= 0.0
    assert 0.0 <= ece <= 1.0


def test_classifier_macro_f1(multiclass_data):
    ds = multiclass_data
    clf = AOMPLSDAClassifier(max_components=4, criterion="covariance")
    clf.fit(ds.X_train, ds.y_train)
    pred = clf.predict(ds.X_test)
    f1 = macro_f1(ds.y_test, pred)
    assert 0.0 <= f1 <= 1.0
