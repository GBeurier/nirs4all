"""Signed availability boundaries and genuine target-specific encoder fits."""
from __future__ import annotations

import io
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.pipeline.dagml.multimodal_contracts import _late_class_labels, _late_learned_state_sha256
from nirs4all.pipeline.dagml.node_runner import _partial_meta_feature_matrix, _PerTargetLateEstimator
from nirs4all.pipeline.dagml.source_missing import (
    build_prediction_availability,
    checked_prediction_features,
    checked_target_validity,
    prediction_source_presence_metadata,
)
from nirs4all.pipeline.dagml.target_capture import captured_target_transform


def _dataset():
    ids = ["s0", "s1", "s2", "s3"]
    values = np.arange(8, dtype=float).reshape(4, 2)
    return MultimodalSpectroDataset(MultimodalDataset(
        {"a": TensorSource(values, ids, representation_id="signal_1d", presence_mask=[True, False, True, True]),
         "b": TensorSource(values + 1, ids, representation_id="signal_1d")},
        sample_ids=ids, y=[[1, 3], [2, 4], [3, 5], [4, 6]], target_names=["sugar", "protein"],
        target_mask=[[True, False], [False, True], [True, True], [True, True]],
        partitions=["train"] * 4, task_type="regression",
    ))


def test_io_availability_preserves_requested_order_and_masks():
    dataset = _dataset()
    identity = SimpleNamespace(to_wire=lambda row: dataset.sample_ids[row])
    result = build_prediction_availability(dataset, identity, [3, 1, 0],
                                          target_values=[[4, 6], [2, 4], [1, 3]], classification=False)
    assert result["sample_ids"] == ["s3", "s1", "s0"]
    assert result["source_presence"] == {"a": [True, False, True], "b": [True, True, True]}
    assert result["target_validity_masks"] == [[True, True], [False, True], [True, False]]
    metadata = prediction_source_presence_metadata(dataset, [3, 1])
    assert metadata == {"prediction_source_presence": {3: {"a": True, "b": True}, 1: {"a": False, "b": True}}}
    with pytest.raises(ValueError, match="unique"):
        prediction_source_presence_metadata(dataset, [0, 0])


@pytest.mark.parametrize("fault", ["numeric_mask", "shape", "names", "changed_mask", "foreign_fit"])
def test_target_validity_cannot_drift_after_native_admission(fault):
    ids = ["a", "b"]
    availability = {"sample_ids": ids, "target_names": ["t"], "target_validity_masks": [[True], [False]]}
    target = {"values": [[1], [0]], "target_names": ["t"], "validity_masks": [[True], [False]]}
    if fault == "numeric_mask":
        target["validity_masks"] = [[1], [0]]
    if fault == "shape":
        target["validity_masks"] = [True, False]
    if fault == "names":
        target["target_names"] = ["different"]
    if fault == "changed_mask":
        target["validity_masks"][1][0] = True
    if fault == "foreign_fit":
        ids = ["a", "foreign"]
    with pytest.raises(ValueError, match="validity"):
        checked_target_validity(target, ids, availability, fitting=True)


@pytest.mark.parametrize("fault", ["missing_masks", "numeric_mask", "absent_nonzero", "presence_drift", "partial_class_mask", "nan"])
def test_native_feature_placeholders_require_exact_separate_masks(fault):
    spec = {"prediction_width": 2, "sample_ids": ["s0", "s1"], "values": [[0.25, 0.75], [0, 0]], "source_presence": [True, False],
            "feature_validity_masks": [[True, True], [False, False]]}
    if fault == "missing_masks":
        del spec["feature_validity_masks"]
    if fault == "numeric_mask":
        spec["feature_validity_masks"] = [[1, 1], [0, 0]]
    if fault == "absent_nonzero":
        spec["values"][1][0] = 0.1
    if fault == "presence_drift":
        spec["source_presence"][1] = True
    if fault == "partial_class_mask":
        spec["feature_validity_masks"][0][1] = False
    if fault == "nan":
        spec["values"][0][0] = np.nan
    with pytest.raises(ValueError, match="availability"):
        checked_prediction_features(spec, np.array([True, False]))


def _native_meta():
    sources = [{"source_index": 0, "source_name": "a"}, {"source_index": 1, "source_name": "b"}]
    layout = {"sources": sources, "fingerprint": "signed", "missing_source_policy": "zero_with_indicator", "target_policy": "complete"}
    nodes = {"meta": {"metadata": {"nirs4all_source_stacking": layout}}}
    specs = []
    for index, name in enumerate(("a", "b")):
        nodes[name] = {"metadata": {"source_name": name, "source_index": index, "prediction_availability_source": name,
            "nirs4all_source_stacking": {"source": sources[index], "layout_fingerprint": "signed",
                                       "missing_source_policy": "zero_with_indicator", "target_policy": "complete"}}}
        specs.append({"prediction_width": 3, "producer_node": name, "source_port": "proba", "sample_ids": ["s0", "s1"], "target_names": ["0.0", "1.0", "2.0"],
                      "values": [[0.2, 0.3, 0.5], [0, 0, 0]], "source_presence": [True, False],
                      "feature_validity_masks": [[True] * 3, [False] * 3]})
    availability = {"sample_ids": ["s0"], "source_presence": {"a": [True], "b": [True]}, "class_labels": [0.0, 1.0, 2.0]}
    resolver = SimpleNamespace(resolve_source_presence=lambda ids, index, **kw: np.array([True, False]))
    return specs, nodes, availability, resolver


def test_full_probability_columns_and_all_absent_prediction_rows_are_features():
    specs, nodes, availability, resolver = _native_meta()
    ids, features = _partial_meta_feature_matrix(specs, "meta", resolver, nodes.__getitem__, availability, None)
    assert ids == ["s0", "s1"]
    np.testing.assert_array_equal(features, [[0.2, 0.3, 0.5, 1, 0.2, 0.3, 0.5, 1], [0] * 8])


@pytest.mark.parametrize("fault", ["reordered", "projected", "class_spelling", "fake_distribution", "source_binding"])
def test_partial_meta_refuses_contract_changes(fault):
    specs, nodes, availability, resolver = _native_meta()
    if fault == "reordered":
        specs.reverse()
    if fault == "projected":
        specs[0]["values"] = [[0.5], [0]]
        specs[0]["feature_validity_masks"] = [[True], [False]]
    if fault == "class_spelling":
        specs[0]["target_names"][0] = "0"
    if fault == "fake_distribution":
        specs[0]["values"][0] = [0, 0, 0]
    if fault == "source_binding":
        nodes["a"]["metadata"]["source_index"] = 1
    with pytest.raises(ValueError):
        _partial_meta_feature_matrix(specs, "meta", resolver, nodes.__getitem__, availability, None)


def test_per_target_fits_complete_independent_encoder_chains_only_on_observed_cells():
    X = np.arange(36, dtype=float).reshape(12, 3)
    y = np.column_stack([2 * X[:, 0], -3 * X[:, 1]])
    mask = np.column_stack([np.arange(12) % 2 == 0, np.arange(12) % 3 != 0])
    hidden = y.copy()
    hidden[~mask] = 1e30
    actual = _PerTargetLateEstimator(Ridge(alpha=0.3), "nir", [StandardScaler()]).fit(X, hidden, target_mask=mask)
    expected = [make_pipeline(StandardScaler(), Ridge(alpha=0.3)).fit(X[mask[:, i]], y[mask[:, i], i]) for i in range(2)]
    np.testing.assert_allclose(actual.predict(X), np.column_stack([model.predict(X) for model in expected]), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(actual.target_counts_, mask.sum(axis=0))
    clones = actual.model_.target_models_
    assert clones[0].transformers_["nir"] is not clones[1].transformers_["nir"]
    for index, fitted in enumerate(clones):
        np.testing.assert_array_equal(fitted.transformers_["nir"].named_steps["standardscaler"].mean_, X[mask[:, index]].mean(axis=0))
    again = _PerTargetLateEstimator(Ridge(alpha=0.3), "nir", [StandardScaler()]).fit(X, y, target_mask=mask)
    np.testing.assert_array_equal(actual.predict(X), again.predict(X))


@pytest.mark.parametrize("protocol,compression", [(4, 0), (5, 0), (5, 3)])
def test_partial_learned_state_survives_joblib_roundtrip_and_preserves_values(protocol, compression):
    rng = np.random.default_rng(18)
    X = rng.normal(size=(16, 5))
    model = make_pipeline(StandardScaler(), PCA(n_components=2), Ridge()).fit(X, X[:, 0] - X[:, 2])
    # PCA's fitted view may have noncompact strides; archive loading compacts it.
    fitted = model.named_steps["pca"]
    padded = np.zeros((2, 10))
    padded[:, ::2] = fitted.components_
    fitted.components_ = padded[:, ::2]
    model.multimodal_source_order = ("nir", "image")
    model.multimodal_target_names = ("sugar",)
    expected = _late_learned_state_sha256(model)
    stream = io.BytesIO()
    joblib.dump(model, stream, compress=compression, protocol=protocol)
    stream.seek(0)
    restored = joblib.load(stream)
    assert _late_learned_state_sha256(restored) == expected
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    restored.named_steps["ridge"].coef_[0] += 1
    assert _late_learned_state_sha256(restored) != expected


@pytest.mark.parametrize("change", ["dtype", "shape", "source_order", "target_names"])
def test_partial_learned_state_keeps_array_and_input_contract_changes_distinct(change):
    estimator = SimpleNamespace(coef_=np.arange(6, dtype=np.float64).reshape(2, 3),
                                source_order=("nir", "image"), target_names=("sugar", "protein"))
    expected = _late_learned_state_sha256(estimator)
    if change == "dtype":
        estimator.coef_ = estimator.coef_.astype(np.float32)
    elif change == "shape":
        estimator.coef_ = estimator.coef_.reshape(3, 2)
    elif change == "source_order":
        estimator.source_order = tuple(reversed(estimator.source_order))
    else:
        estimator.target_names = tuple(reversed(estimator.target_names))
    assert _late_learned_state_sha256(estimator) != expected


def test_partial_refit_public_vocabulary_is_identical_before_and_after_capture():
    estimator = LogisticRegression().fit(np.arange(8).reshape(4, 2), [0, 1, 0, 1])
    decoder = SimpleNamespace(column_transformers={0: SimpleNamespace(classes_=np.asarray(["healthy", "severe"]))})
    live = {"estimator": estimator, "target_decoder": decoder}
    captured = {"estimator": estimator, "y_transform": captured_target_transform(None, decoder, estimator)}
    assert _late_class_labels(live) == _late_class_labels(captured) == ["healthy", "severe"]
