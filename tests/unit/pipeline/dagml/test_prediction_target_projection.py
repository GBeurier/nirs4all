"""Prediction-time scoring consumes the frozen owner target domain."""

import json
from dataclasses import asdict

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer

from nirs4all.api.result import PredictResult
from nirs4all.data._targets.encoders import FlexibleLabelEncoder
from nirs4all.operators.transforms.targets import RangeDiscretizer
from nirs4all.pipeline.dagml.prediction_targets import attach_categorical_scoring_context
from nirs4all.pipeline.dagml.target_capture import CategoricalTargetTransform


def _prediction(transformer, count):
    return attach_categorical_scoring_context(PredictResult(y_pred=np.zeros(count)),
                                             {"estimator": LogisticRegression(), "y_transform": transformer})


def test_scoring_targets_preserve_missing_masks_original_values_and_sdk_input_precision(monkeypatch):
    cut = float(np.float32(.1))
    transformer = CategoricalTargetTransform(RangeDiscretizer([cut]).fit([-1., 1.]))
    result = _prediction(transformer, 5)
    raw = np.array([.1, np.nan, np.inf, 1.e100, -.5])
    original = raw.copy()
    monkeypatch.setattr(RangeDiscretizer, "fit", lambda *args, **kwargs: pytest.fail("score projection fitted bins"))
    monkeypatch.setattr(FunctionTransformer, "fit", lambda *args, **kwargs: pytest.fail("score projection fitted input encoder"))
    projected = result.scoring_targets(raw)
    assert projected["task_type"] == "binary_classification"
    assert projected["target_space"] == "categorical_class_ids"
    np.testing.assert_array_equal(projected["validity_mask"], [True, False, False, False, True])
    np.testing.assert_array_equal(projected["values"][[0, 4]], [1., 0.])
    assert np.isnan(projected["values"][[1, 2, 3]]).all()
    np.testing.assert_array_equal(raw, original)
    assert projected["fitted_on_prediction_cohort"] is False
    assert "_categorical_scoring_transform" not in asdict(result)
    json.dumps(result.metadata)


def test_scoring_context_is_frozen_and_respects_explicit_mask():
    transformer = CategoricalTargetTransform(RangeDiscretizer([0.]).fit([-1., 1.]))
    result = _prediction(transformer, 2)
    transformer.transformer._bins_array[:] = 99.
    projected = result.scoring_targets(np.array([[1.], [-1.]]), np.array([[True], [False]]))
    assert projected["values"][0, 0] == 1.
    assert np.isnan(projected["values"][1, 0])
    with pytest.raises(ValueError, match="align with every prediction"):
        result.scoring_targets([1.])
    with pytest.raises(ValueError, match="original target shape"):
        result.scoring_targets([[1.], [-1.]], [True, False])


def test_scoring_reuses_original_label_encoder_without_admitting_unseen_labels():
    transformer = CategoricalTargetTransform(RangeDiscretizer([.5]).fit([0., 1.]))
    transformer.input_transformer_ = FlexibleLabelEncoder().fit(np.array(["a", "b"]))
    result = _prediction(transformer, 3)
    projected = result.scoring_targets(np.array(["a", "new", "b"]))
    np.testing.assert_array_equal(projected["values"][[0, 2]], [0., 1.])
    np.testing.assert_array_equal(projected["validity_mask"], [True, False, True])
    np.testing.assert_array_equal(transformer.input_transformer_.classes_, ["a", "b"])


def test_prediction_record_without_fitted_context_cannot_fallback_to_raw_targets():
    result = PredictResult(y_pred=np.ones(2), metadata={"scoring_target_projection": {"target_space": "categorical_class_ids"}})
    with pytest.raises(ValueError, match="original captured predictor"):
        result.scoring_targets([1., 2.])
    assert PredictResult(y_pred=np.ones(2)).scoring_targets([1., 2.]) is None
