"""Regression witnesses for DGC-01/02 and DGB-01/03 at helper and public boundaries."""

import json

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from nirs4all.data import SpectroDataset
from nirs4all.operators.augmentation.environmental import TemperatureAugmenter
from nirs4all.operators.filters.metadata import MetadataFilter
from nirs4all.operators.transforms import Resampler
from nirs4all.pipeline.dagml.envelope import _numeric_feature_axis
from nirs4all.pipeline.dagml.exclude import _excluded_from_pool, _resolve_exclude, _resolve_tags
from nirs4all.pipeline.dagml.native_results import _stacking_replay_manifest
from nirs4all.pipeline.dagml.node_runner import _coordinate_chain, _CoordinateTransform
from nirs4all.pipeline.dagml.operator_parameters import decode_constructor_value, encode_constructor_value
from nirs4all.pipeline.dagml.operator_routing import route_operator
from nirs4all.pipeline.dagml_bridge import _json_safe_params, _qualname


def _metadata_dataset():
    dataset = SpectroDataset("metadata-filter")
    dataset.add_samples(np.arange(24, dtype=float).reshape(6, 4), {"partition": "train"})
    dataset.add_samples(np.ones((2, 4)), {"partition": "test"})
    dataset.add_targets(np.arange(8, dtype=float))
    dataset.add_metadata(np.array(["good", "bad", "good", "bad", "good", "bad", "bad", "bad"])[:, None], headers=["batch"])
    return dataset


@pytest.mark.parametrize("pool", [[0, 1, 2, 3, 4, 5], [4, 1, 3, 0]])
def test_metadata_exclude_and_tag_align_masks_to_pool_order(pool):
    dataset = _metadata_dataset()
    filter_obj = MetadataFilter("batch", values_to_exclude=["bad"], tag_name="bad_batch")
    assert _excluded_from_pool({"exclude": filter_obj}, dataset, pool) == set(pool) & {1, 3, 5}
    remaining, tags = _resolve_tags([{"tag": filter_obj}], dataset, pool)
    assert remaining == []
    assert tags == {sample: ["bad_batch"] for sample in pool if sample in {1, 3, 5}}


def test_metadata_exclusion_is_sequential_and_fold_local():
    dataset = _metadata_dataset()
    steps = [
        {"exclude": MetadataFilter("batch", values_to_exclude=["bad"]), "keep_in_oof": True},
        {"exclude": MetadataFilter("batch", values_to_keep=["good"])},
    ]
    remaining, pool, excluded = _resolve_exclude(steps, dataset)
    assert remaining == []
    assert pool == list(range(6))
    assert set(excluded) == {1, 3, 5}
    assert excluded.apply(dataset, [([4, 1, 0], [2, 3, 5])]) == [([4, 0], [2, 3, 5])]


@pytest.mark.parametrize("keyword", ["exclude", "tag"])
def test_missing_metadata_column_still_refuses_filter(keyword):
    dataset = _metadata_dataset()
    step = {keyword: MetadataFilter("missing", values_to_keep=["good"])}
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match="missing"):
        if keyword == "exclude":
            _resolve_exclude([step], dataset)
        else:
            _resolve_tags([step], dataset, [4, 0, 1])


@pytest.mark.parametrize("mapping", [
    {0: 1.0, 1: 5.0},
    {"1": "string", 1: "integer", None: "null", 2.5: "float", False: "boolean"},
    {"nested": [{0: {1: "value"}}]},
])
def test_nonstring_constructor_keys_survive_json_transport(mapping):
    restored = decode_constructor_value(json.loads(json.dumps(encode_constructor_value(mapping))))
    assert restored == mapping
    assert list(restored) == list(mapping)


@pytest.mark.parametrize("nested", [False, True])
def test_class_weighted_classifier_preserves_integer_labels_and_predictions(nested):
    model = LogisticRegression(class_weight={0: 1.0, 1: 5.0}, random_state=42)
    if nested:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        model = Pipeline([("scale", StandardScaler()), ("model", model)])
    restored = route_operator("model", _qualname(model), _json_safe_params(model))
    classifier = restored[-1] if nested else restored
    assert classifier.class_weight == {0: 1.0, 1: 5.0}
    X = np.random.default_rng(5).normal(size=(40, 5))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    np.testing.assert_allclose(restored.fit(X, y).predict_proba(X), clone(model).fit(X, y).predict_proba(X))


@pytest.mark.parametrize("unit", ["nm", "cm-1"])
def test_temperature_transform_receives_nm_from_attested_cm1_axis(unit):
    wavelengths = np.linspace(1000, 2500, 150)
    X = np.random.default_rng(1).uniform(0.1, 1.0, (4, len(wavelengths)))
    dataset = SpectroDataset("temperature-axis")
    headers = wavelengths if unit == "nm" else 1e7 / wavelengths
    dataset.add_samples(X, {"partition": "train"}, headers=[str(value) for value in headers], header_unit=unit)
    coordinates = tuple(_numeric_feature_axis(dataset, 0))
    original_coordinates = coordinates
    operator = TemperatureAugmenter(temperature_delta=40.0, random_state=42)
    wrapped = _coordinate_chain([operator], [coordinates])[0]
    actual = wrapped.fit_transform(X)
    expected = clone(operator).fit(X, wavelengths=wavelengths).transform(X)
    assert np.max(np.abs(expected - X)) > 0.01
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert coordinates == original_coordinates


def test_coordinate_wrapper_keeps_resampler_in_cm1():
    coordinates = np.linspace(10000, 4000, 12)
    X = np.random.default_rng(3).normal(size=(4, 12))
    resampler = Resampler(target_wavelengths=np.linspace(9500, 4500, 8))
    wrapped = _CoordinateTransform(resampler, tuple(str(value) for value in coordinates))
    actual = wrapped.fit_transform(X)
    expected = clone(resampler).fit(X, wavelengths=coordinates).transform(X)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(resampler.original_wavelengths_, coordinates)


@pytest.mark.parametrize("unit", ["nm", "cm-1"])
def test_feature_concat_converts_coordinates_once_per_child(unit):
    from nirs4all.operators.transforms.concat import FeatureConcat

    wavelengths = np.linspace(1000, 2500, 150)
    X = np.random.default_rng(1).uniform(0.1, 1.0, (4, len(wavelengths)))
    dataset = SpectroDataset("mixed-coordinate-channels")
    headers = wavelengths if unit == "nm" else 1e7 / wavelengths
    dataset.add_samples(X, {"partition": "train"}, headers=[str(value) for value in headers], header_unit=unit)
    coordinates = tuple(_numeric_feature_axis(dataset, 0))
    resampler = Resampler(target_wavelengths=np.linspace(9000, 4500, 12))
    physical = TemperatureAugmenter(temperature_delta=40.0, random_state=42)
    concat = FeatureConcat([None, *({"class": _qualname(operator), "params": _json_safe_params(operator)} for operator in (resampler, physical))])
    actual = _coordinate_chain([concat], [coordinates])[0].fit_transform(X)
    expected = np.hstack([X, clone(resampler).fit(X, wavelengths=1e7 / wavelengths).transform(X),
                          clone(physical).fit(X, wavelengths=wavelengths).transform(X)])
    assert np.max(np.abs(expected[:, -X.shape[1]:] - X)) > 0.01
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_strict_wavelength_operator_still_refuses_index_only_data():
    dataset = SpectroDataset("index-only")
    dataset.add_samples(np.ones((4, 12)), {"partition": "train"})
    assert _numeric_feature_axis(dataset, 0) is None
    with pytest.raises(ValueError, match="requires coordinates"):
        _coordinate_chain([TemperatureAugmenter()], [None])


@pytest.mark.parametrize("metric,scores,expected", [
    ("f1", [0.9, 0.5], [0.9, 0.5]),
    ("accuracy", [0.9, 0.5], [0.9, 0.5]),
    ("balanced_accuracy", [0.9, 0.5], [0.9, 0.5]),
    ("r2", [0.9, 0.5], [0.9, 0.5]),
    ("rmse", [0.2, 0.4], [1 / (0.2 + 1e-10), 1 / (0.4 + 1e-10)]),
    ("f1", [float("nan"), 0.5], [0.0, 0.5]),
])
def test_stacking_replay_weights_follow_metric_direction(metric, scores, expected):
    producers = ["branch:0.model0", "branch:0.model1"]
    refs = [{"artifact_id": f"artifact:{node}:nirs4all:refit:0", "producer_node": node} for node in producers]
    refs.append({"artifact_id": "artifact:merge:stack:nirs4all:refit:0", "controller_id": "controller:nirs4all.meta_model"})
    reports = [{"producer_node": node, "partition": "validation", "fold_id": "fold0", "level": "sample", "metrics": {metric: score}}
               for node, score in zip(producers, scores, strict=True)]
    reports.append({"producer_node": "merge:stack", "partition": "final"})
    manifest = _stacking_replay_manifest({"reports": reports}, refs, [{"branch": 0, "aggregate": "weighted_mean", "metric": metric}])
    assert manifest is not None
    np.testing.assert_allclose(manifest["reduction_groups"][0]["weights"], expected)


@pytest.mark.parametrize("keyword", ["exclude", "tag"])
def test_public_dag_run_accepts_metadata_filters(keyword, tmp_path, monkeypatch):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    import nirs4all

    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    dataset = _metadata_dataset()
    pipeline = [{keyword: MetadataFilter("batch", values_to_exclude=["bad"])}, KFold(3), Ridge()]
    result = nirs4all.run(pipeline, dataset, engine="dag-ml", workspace_path=tmp_path,
                          save_charts=False, save_artifacts=False, verbose=0)
    try:
        assert result.execution_engine == "dag-ml"
        assert np.isfinite(result.cv_best_score)
        assert result._dagml_score_set is not None
    finally:
        result.close()


def test_public_dag_run_accepts_integer_class_weights(tmp_path, monkeypatch):
    from sklearn.model_selection import StratifiedKFold

    import nirs4all

    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    X = np.random.default_rng(5).normal(size=(40, 5))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    model = LogisticRegression(class_weight={0: 1.0, 1: 5.0}, random_state=42)
    result = nirs4all.run([StratifiedKFold(3), model], (X, y), engine="dag-ml", workspace_path=tmp_path,
                          save_charts=False, save_artifacts=False, verbose=0)
    try:
        assert result.execution_engine == "dag-ml"
        assert np.isfinite(result.cv_best_score)
        assert model.class_weight == {0: 1.0, 1: 5.0}
    finally:
        result.close()


def test_public_dag_temperature_matches_manual_nm_cv(tmp_path, monkeypatch):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    import nirs4all

    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    wavelengths = np.linspace(1000, 2500, 150)
    X = np.random.default_rng(1).uniform(0.1, 1.0, (18, len(wavelengths)))
    y = 2 * X[:, 42] - X[:, 75] + X[:, 95]
    dataset = SpectroDataset("temperature-cv")
    dataset.add_samples(X, {"partition": "train"}, headers=[str(value) for value in wavelengths], header_unit="nm")
    dataset.add_targets(y)
    operator = TemperatureAugmenter(temperature_delta=40.0, random_state=42)
    expected = np.empty_like(y)
    for train, validation in KFold(3).split(X):
        transform = clone(operator).fit(X[train], y[train], wavelengths=wavelengths)
        model = Ridge().fit(transform.transform(X[train]), y[train])
        expected[validation] = model.predict(transform.transform(X[validation]))
    expected_rmse = np.sqrt(np.mean((y - expected) ** 2))
    result = nirs4all.run([operator, KFold(3), Ridge()], dataset, engine="dag-ml", workspace_path=tmp_path,
                          save_charts=False, save_artifacts=False, verbose=0)
    try:
        assert result.cv_best_score == pytest.approx(expected_rmse, abs=1e-7)
    finally:
        result.close()
