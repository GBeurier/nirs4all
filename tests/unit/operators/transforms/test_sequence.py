"""Real ragged IO inputs retain sample boundaries through fixed-width encoding."""

import io

import joblib
import numpy as np
import pytest
from nirs4all_io.ragged import RaggedSeriesBatch
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all.operators.transforms import SequenceSummary


def _batch(*, time_coordinates=None):
    return RaggedSeriesBatch(
        np.asarray([[1, 2], [3, 6], [5, 10], [10, -2]], dtype=float),
        np.asarray([0, 3, 4], dtype=np.int64), time_coordinates=time_coordinates,
    )


def test_summaries_match_variable_length_observations_and_output_names():
    batch = _batch()
    encoder = SequenceSummary(channel_names=["temperature", "pressure"])
    actual = encoder.fit_transform(batch, np.asarray([1.0, 2.0]))
    expected = [
        [3, np.sqrt(8 / 3), 1, 5, 6, np.sqrt(32 / 3), 2, 10, 3],
        [10, 0, 10, 10, -2, 0, -2, -2, 1],
    ]
    np.testing.assert_allclose(actual, expected)
    assert actual.dtype == np.float64
    assert encoder.n_features_in_ == 2
    assert encoder.n_features_out_ == 9
    assert encoder.get_feature_names_out().tolist() == [
        "temperature_mean", "temperature_std", "temperature_min", "temperature_max",
        "pressure_mean", "pressure_std", "pressure_min", "pressure_max", "length",
    ]
    np.testing.assert_array_equal(encoder.feature_names_in_, ["temperature", "pressure"])


def test_statistic_order_length_only_and_optional_length():
    actual = SequenceSummary(["max", "min"], include_length=False).fit_transform(_batch())
    np.testing.assert_array_equal(actual, [[5, 1, 10, 2], [10, 10, -2, -2]])
    lengths = SequenceSummary(statistics=()).fit(_batch())
    np.testing.assert_array_equal(lengths.transform(_batch()), [[3], [1]])
    assert lengths.get_feature_names_out().tolist() == ["length"]


def test_new_lengths_are_allowed_without_padding_or_global_array_conversion():
    fitted = SequenceSummary(statistics=("mean",), include_length=True).fit(_batch())
    values = np.column_stack([np.arange(9), np.arange(9) * 2])
    new = RaggedSeriesBatch(values, np.asarray([0, 2, 9]))
    with pytest.raises(TypeError):
        np.asarray(new)
    np.testing.assert_array_equal(fitted.transform(new), [[0.5, 1, 2], [5, 10, 7]])
    assert fitted.channel_names_ == ("channel_0", "channel_1")


def test_time_coordinates_have_uniform_observation_weights():
    regular = _batch(time_coordinates=np.asarray([0, 1, 2, 0], dtype=float))
    irregular = _batch(time_coordinates=np.asarray([0, 0.001, 1000, 5], dtype=float))
    encoder = SequenceSummary().fit(regular)
    np.testing.assert_array_equal(encoder.transform(regular), encoder.transform(irregular))


def test_encoder_retains_contract_only_and_does_not_mutate_source():
    batch = _batch()
    before_values, before_offsets = batch.values.copy(), batch.offsets.copy()
    encoder = SequenceSummary().fit(batch)
    encoder.transform(batch)
    assert not any(isinstance(value, (RaggedSeriesBatch, np.ndarray)) for value in vars(encoder).values())
    np.testing.assert_array_equal(batch.values, before_values)
    np.testing.assert_array_equal(batch.offsets, before_offsets)


def test_pipeline_clones_fit_only_selected_rows_and_reload_predicts_without_fit(monkeypatch):
    batch = RaggedSeriesBatch(
        np.asarray([[1], [3], [5], [7], [9], [1000], [2000], [3000]], dtype=float),
        np.asarray([0, 2, 3, 5, 8]),
    )
    y = np.asarray([2, 5, 8, -1000], dtype=float)
    train = batch.take_rows(np.asarray([0, 1, 2]))
    template = make_pipeline(SequenceSummary(statistics=("mean", "max")), StandardScaler(), Ridge(alpha=0.1))
    pipeline = clone(template).fit(train, y[:3])
    np.testing.assert_allclose(pipeline[1].mean_, [5, 17 / 3, 5 / 3])
    assert not hasattr(template[0], "n_features_in_")
    expected = pipeline.predict(batch)
    state = io.BytesIO()
    joblib.dump(pipeline, state)
    state.seek(0)
    monkeypatch.setattr(SequenceSummary, "fit", lambda *args, **kwargs: pytest.fail("reload attempted fitting"))
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("reload attempted model fitting"))
    np.testing.assert_array_equal(joblib.load(state).predict(batch), expected)


def test_clone_and_parameter_changes_do_not_copy_fitted_state():
    original = SequenceSummary(["mean", "max"], channel_names=["a", "b"]).fit(_batch())
    fresh = clone(original)
    assert fresh.get_params() == original.get_params()
    assert not hasattr(fresh, "n_features_in_")
    fresh.set_params(statistics=("min",), include_length=False)
    assert fresh.fit_transform(_batch()).shape == (2, 2)
    assert original.n_features_out_ == 5


def test_channel_contract_and_explicit_feature_names():
    encoder = SequenceSummary(channel_names=("a", "b")).fit(_batch())
    with pytest.raises(ValueError, match="expected 2 channels"):
        encoder.transform(RaggedSeriesBatch(np.ones((3, 3)), np.asarray([0, 3])))
    np.testing.assert_array_equal(encoder.get_feature_names_out(["a", "b"]), encoder.get_feature_names_out())
    with pytest.raises(ValueError, match="match the fitted"):
        encoder.get_feature_names_out(["b", "a"])
    with pytest.raises(ValueError, match="2 distinct"):
        encoder.get_feature_names_out(["a"])
    encoder.set_params(channel_names=None).fit(_batch())
    assert not hasattr(encoder, "feature_names_in_")
    assert encoder.get_feature_names_out(["left", "right"])[0] == "left_mean"


@pytest.mark.parametrize("params, message", [
    ({"statistics": "mean"}, "statistics"),
    ({"statistics": None}, "statistics"),
    ({"statistics": ("mean", "median")}, "statistics"),
    ({"statistics": ("mean", "mean")}, "duplicates"),
    ({"statistics": (), "include_length": False}, "at least one output"),
    ({"include_length": 1}, "boolean"),
    ({"channel_names": ["a"]}, "2 distinct"),
    ({"channel_names": ["a", "a"]}, "2 distinct"),
    ({"channel_names": ["a", ""]}, "2 distinct"),
    ({"channel_names": "ab"}, "per channel"),
])
def test_invalid_configuration_is_refused(params, message):
    with pytest.raises(ValueError, match=message):
        SequenceSummary(**params).fit(_batch())


@pytest.mark.parametrize("bad", [np.ones((2, 3, 2)), [np.ones((3, 2)), np.ones((1, 2))], np.ones((4, 2))])
def test_implicit_array_and_list_adapters_are_refused(bad):
    with pytest.raises(TypeError, match="requires a RaggedSeriesBatch"):
        SequenceSummary().fit(bad)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_observations_are_refused(value):
    batch = RaggedSeriesBatch(np.asarray([[value, 0.0]]), np.asarray([0, 1]))
    with pytest.raises(ValueError, match="finite real numeric"):
        SequenceSummary().fit(batch)
    with pytest.raises(ValueError, match="finite real numeric"):
        SequenceSummary().fit(_batch()).transform(batch)


def test_zero_samples_empty_series_and_target_count():
    empty = RaggedSeriesBatch(np.empty((0, 2)), np.asarray([0]))
    with pytest.raises(ValueError, match="at least one training"):
        SequenceSummary().fit(empty)
    assert SequenceSummary().fit(_batch()).transform(empty).shape == (0, 9)
    empty_series = RaggedSeriesBatch(np.empty((0, 2)), np.asarray([0, 0]))
    with pytest.raises(ValueError, match="empty series"):
        SequenceSummary().fit(empty_series)
    with pytest.raises(ValueError, match="empty series"):
        SequenceSummary().fit(_batch()).transform(empty_series)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        SequenceSummary().fit(_batch(), [1])


def test_numerical_overflow_is_refused_without_emitting_invalid_features():
    batch = RaggedSeriesBatch(np.asarray([[1e308, 1.0], [-1e308, 1.0]]), np.asarray([0, 2]))
    with pytest.raises(ValueError, match="overflowed"):
        SequenceSummary().fit_transform(batch)


def test_transform_requires_fitted_contract():
    with pytest.raises(NotFittedError):
        SequenceSummary().transform(_batch())
    with pytest.raises(NotFittedError):
        SequenceSummary().get_feature_names_out()
