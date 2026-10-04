"""Declared series constraints survive cloning and prediction without fitting."""

import io

import joblib
import numpy as np
import pytest
from nirs4all_io.ragged import RaggedSeriesBatch
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from nirs4all.operators.transforms import SequenceSummary


def _batch(values, lengths):
    return RaggedSeriesBatch(np.asarray(values, dtype=float), np.r_[0, np.cumsum(lengths)])


def test_declared_domain_preserves_original_independent_summary_values():
    batch = _batch([[0, 10], [2, 12], [4, 14], [6, 16], [8, 18]], [2, 3])
    summary = SequenceSummary(statistics=("mean", "max"), min_observations=2,
                              channel_bounds=((0, 8), (10, 18))).fit(batch)
    np.testing.assert_array_equal(summary.transform(batch), [[1, 2, 11, 12, 2], [6, 8, 16, 18, 3]])
    assert summary.min_observations_ == 2
    assert summary.channel_bounds_ == ((0.0, 8.0), (10.0, 18.0))
    assert summary.get_feature_names_out().tolist() == [
        "channel_0_mean", "channel_0_max", "channel_1_mean", "channel_1_max", "length",
    ]


@pytest.mark.parametrize("operation", ["fit", "transform"])
def test_too_short_series_are_refused_without_padding_or_reordering(operation):
    encoder = SequenceSummary(min_observations=2)
    if operation == "transform":
        encoder.fit(_batch([[1], [2], [3], [4]], [2, 2]))
    batch = _batch([[1], [2], [3]], [2, 1])
    before = batch.values.copy(), batch.offsets.copy()
    with pytest.raises(ValueError, match="at least 2 observations"):
        getattr(encoder, operation)(batch)
    np.testing.assert_array_equal(batch.values, before[0])
    np.testing.assert_array_equal(batch.offsets, before[1])


@pytest.mark.parametrize("operation", ["fit", "transform"])
@pytest.mark.parametrize("violation", [-1.0, 11.0])
def test_each_observation_must_be_inside_the_declared_domain(operation, violation):
    encoder = SequenceSummary(channel_bounds=((0, 10),))
    if operation == "transform":
        encoder.fit(_batch([[4], [5]], [2]))
    # The average is in bounds: checking only the encoded mean is insufficient.
    batch = _batch([[violation], [5]], [2])
    with pytest.raises(ValueError, match="outside the declared channel_bounds"):
        getattr(encoder, operation)(batch)


@pytest.mark.parametrize("minimum", [True, np.bool_(True), 0, -1, 1.0, "2", None])
def test_invalid_length_contract_is_refused_before_fitted_state(minimum):
    encoder = SequenceSummary(min_observations=minimum)
    with pytest.raises(ValueError, match="positive integer"):
        encoder.fit(_batch([[1]], [1]))
    assert not hasattr(encoder, "n_features_in_")


@pytest.mark.parametrize("bounds", [
    ((1, 0),), ((0, float("inf")),), ((float("nan"), 1),),
    (("0", "1"),), ((False, True),), ((0, 1), (0, 1)), (0, 1),
])
def test_invalid_domain_contract_is_refused_before_fitted_state(bounds):
    encoder = SequenceSummary(channel_bounds=bounds)
    with pytest.raises(ValueError, match="finite, ordered numeric"):
        encoder.fit(_batch([[0.5]], [1]))
    assert not hasattr(encoder, "n_features_in_")


def test_clones_are_unfitted_and_fitted_constraints_do_not_follow_parameter_mutation():
    bounds = [[0.0, 10.0]]
    encoder = SequenceSummary(min_observations=np.int64(2), channel_bounds=bounds).fit(_batch([[2], [4]], [2]))
    fresh = clone(encoder)
    assert fresh.get_params() == encoder.get_params()
    assert not hasattr(fresh, "n_features_in_")
    bounds[0][1] = 1000
    encoder.set_params(min_observations=1)
    with pytest.raises(ValueError, match="at least 2 observations"):
        encoder.transform(_batch([[2]], [1]))
    with pytest.raises(ValueError, match="outside the declared"):
        encoder.transform(_batch([[11], [2]], [2]))


def test_reloaded_pipeline_preserves_refusals_before_numeric_prediction(monkeypatch):
    batch = _batch([[1], [3], [4], [6], [7], [9]], [2, 2, 2])
    pipeline = make_pipeline(SequenceSummary(statistics=("mean",), include_length=False,
                                             min_observations=2, channel_bounds=((0, 10),)), Ridge(alpha=0.1))
    pipeline.fit(batch, np.asarray([2.0, 5.0, 8.0]))
    expected = pipeline.predict(batch)
    saved = io.BytesIO()
    joblib.dump(pipeline, saved)
    saved.seek(0)
    monkeypatch.setattr(SequenceSummary, "fit", lambda *a, **k: pytest.fail("replay fitted the encoder"))
    monkeypatch.setattr(Ridge, "fit", lambda *a, **k: pytest.fail("replay fitted the model"))
    replay = joblib.load(saved)
    np.testing.assert_array_equal(replay.predict(batch), expected)
    monkeypatch.setattr(Ridge, "predict", lambda *a, **k: pytest.fail("invalid series reached the numeric model"))
    for bad, message in [(_batch([[1]], [1]), "at least 2"), (_batch([[11], [1]], [2]), "outside the declared")]:
        with pytest.raises(ValueError, match=message):
            replay.predict(bad)


def test_published_unconstrained_fitted_summary_retains_its_original_behavior():
    encoder = SequenceSummary(statistics=("mean",), include_length=False).fit(_batch([[1]], [1]))
    del encoder.min_observations_
    del encoder.channel_bounds_
    np.testing.assert_array_equal(encoder.transform(_batch([[-100], [100]], [1, 1])), [[-100], [100]])

