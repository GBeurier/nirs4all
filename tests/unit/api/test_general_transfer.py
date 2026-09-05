"""Scientific invariants for frozen preprocessing before API exposure."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.api.general_transfer import FrozenTransferTransform, fresh_training_estimator, transfer_training_steps


@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("with_target", [False, True])
def test_only_new_model_fits_source_preprocessing_remains_frozen(monkeypatch, replace, with_target):
    rng = np.random.default_rng(11)
    old_x = rng.normal(size=(30, 4))
    new_x = rng.normal(size=(21, 4)) + 5
    old_y = old_x[:, 0] + 50
    new_y = new_x[:, 1] - 80
    target = MinMaxScaler().fit(old_y[:, None]) if with_target else None
    source = make_pipeline(StandardScaler(), Ridge(alpha=0.1)).fit(old_x, old_y)
    before = pickle.dumps((source, target))
    model = Ridge(alpha=8) if replace else None
    expected_model = clone(model if model is not None else source[-1])
    expected_y = target.transform(new_y[:, None]).ravel() if target is not None else new_y
    expected_model.fit(source[:-1].transform(new_x), expected_y)
    expected = expected_model.predict(source[:-1].transform(new_x))
    if target is not None:
        expected = target.inverse_transform(expected[:, None]).ravel()

    def forbid_fit(*args, **kwargs):
        raise AssertionError("captured preprocessing was fitted on transfer data")

    monkeypatch.setattr(StandardScaler, "fit", forbid_fit)
    monkeypatch.setattr(MinMaxScaler, "fit", forbid_fit)
    steps = transfer_training_steps(source, target, model)
    estimator = clone(steps[-1]["model"])
    if target is not None:
        frozen_target = clone(steps[0]["y_processing"])
        fit_y = frozen_target.fit_transform(new_y[:, None]).ravel()
    else:
        fit_y = new_y
    estimator.fit(new_x, fit_y)
    actual = estimator.predict(new_x)
    if target is not None:
        actual = frozen_target.inverse_transform(actual[:, None]).ravel()
    np.testing.assert_array_equal(actual, expected)
    assert pickle.dumps((source, target)) == before
    assert not hasattr(steps[-1]["model"][-1], "coef_")
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(estimator)).predict(new_x), estimator.predict(new_x))


def test_transfer_retains_classifier_classes_and_real_probabilities():
    x = np.random.default_rng(2).normal(size=(40, 4))
    y = np.where(x[:, 0] > 0, 3, 7)
    source = make_pipeline(StandardScaler(), LogisticRegression()).fit(x, y)
    transfer = transfer_training_steps(source, None)[0]["model"]
    transfer.fit(x + 2, y)
    np.testing.assert_array_equal(transfer.classes_, [3, 7])
    np.testing.assert_allclose(transfer.predict_proba(x + 2).sum(axis=1), 1)


def test_full_retrain_removes_nested_transfer_freezes():
    x = np.arange(48, dtype=float).reshape(12, 4)
    source = make_pipeline(StandardScaler(), Ridge()).fit(x, x[:, 0])
    transfer = transfer_training_steps(source, None)[0]["model"].fit(x + 100, x[:, 1])
    # Repeated transfers can contain more than one frozen boundary.
    second = transfer_training_steps(transfer, None)[0]["model"].fit(x - 80, x[:, 2])
    fresh = fresh_training_estimator(second)
    for value in fresh.get_params(deep=True).values():
        assert not isinstance(value, FrozenTransferTransform)
        assert not hasattr(value, "mean_")
        assert not hasattr(value, "coef_")
    fresh.fit(x + 300, x[:, 0])
    assert np.isfinite(fresh.predict(x + 300)).all()
    target = MinMaxScaler().fit(x[:, :1])
    assert not hasattr(fresh_training_estimator(FrozenTransferTransform(target)), "data_min_")


def test_bare_model_can_transfer_without_a_preprocessing_pipeline():
    source = Ridge().fit([[1], [2], [3]], [2, 3, 4])
    model = transfer_training_steps(source, None)[0]["model"]
    assert isinstance(model, Ridge)
    assert not hasattr(model, "coef_")


@pytest.mark.parametrize("source", [object(), Pipeline([])])
def test_nontrainable_source_is_not_reinterpreted(source):
    with pytest.raises(ValueError, match="requires"):
        transfer_training_steps(source, None)


def test_frozen_clone_does_not_alias_source_state():
    fitted = StandardScaler().fit([[1], [2], [3]])
    frozen = FrozenTransferTransform(fitted)
    copied = clone(frozen)
    assert copied is not frozen
    assert copied.fitted is not fitted
    copied.fitted.mean_[0] = 99
    assert fitted.mean_[0] == 2


def test_full_training_keeps_legitimate_passthrough_steps():
    source = Pipeline([("noop", "passthrough"), ("optional", None), ("model", Ridge())])
    fresh = fresh_training_estimator(source)
    assert fresh.steps[:2] == source.steps[:2]
    assert fresh[-1] is not source[-1]
