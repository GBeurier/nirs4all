"""A user module receives fixed image and series axes without host flattening."""

import io

import joblib
import numpy as np
import pytest
import torch

from nirs4all.pipeline.dagml.named_torch_estimator import DagMLNamedTorchEstimator

pytestmark = pytest.mark.torch


def _inputs():
    rng = np.random.default_rng(214)
    return {"nir": rng.normal(size=(6, 4)).astype(np.float32),
            "image": rng.normal(size=(6, 2, 3, 3)).astype(np.float64),
            "series": rng.normal(size=(6, 5, 2)).astype(np.float32)}


def _estimator():
    return DagMLNamedTorchEstimator(factory_path="tests.fixtures.named_torch_nd.joint_factory", device="cpu",
        task_type="regression", epochs=2, batch_size=3, patience=2, lr=0.01)


def test_raw_shapes_reach_user_factory_and_forward_and_survive_joblib(monkeypatch):
    features = _inputs()
    model = _estimator().fit(features, features["nir"][:, 0])
    assert model.input_shapes_ == {"nir": (4,), "image": (2, 3, 3), "series": (5, 2)}
    assert model.n_features_in_ == 4 + 18 + 10
    assert model.model_.seen_shapes and all(value == model.input_shapes_ for value in model.model_.seen_shapes)
    for name in features:
        gradient = model.model_.encoders[name].weight.grad
        assert gradient is not None and torch.count_nonzero(gradient).item() > 0
    prediction = model.predict(features)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    monkeypatch.setattr(DagMLNamedTorchEstimator, "fit", lambda *a, **k: pytest.fail("replay fitted the model"))
    restored = joblib.load(buffer)
    np.testing.assert_array_equal(restored.predict(features), prediction)


@pytest.mark.parametrize("damage", ["axis_order", "rank", "dtype"])
def test_shape_or_dtype_changes_are_refused_before_forward(damage, monkeypatch):
    features = _inputs()
    model = _estimator().fit(features, features["nir"][:, 0])
    changed = dict(features)
    if damage == "axis_order":
        changed["image"] = features["image"].transpose(0, 2, 1, 3)
    elif damage == "rank":
        changed["image"] = features["image"].reshape(6, -1)
    else:
        changed["image"] = features["image"].astype(np.float32)
    monkeypatch.setattr(model.model_, "forward", lambda **k: pytest.fail("changed input reached numeric forward"))
    with pytest.raises(ValueError, match="changed fitted shape or dtype"):
        model.predict(changed)


@pytest.mark.parametrize("shape,dtype", [((6, 2, 2, 2, 2), np.float32), ((6, 0, 3), np.float32), ((6, 2, 3), np.uint8)])
def test_invalid_tensor_rank_size_or_dtype_is_refused_before_user_factory(shape, dtype, monkeypatch):
    features = _inputs()
    features["image"] = np.zeros(shape, dtype=dtype)
    model = _estimator()
    monkeypatch.setattr(model, "_new_named_model", lambda *a, **k: pytest.fail("invalid source constructed a user model"))
    with pytest.raises(ValueError, match="rank-2 to rank-4 float32/float64"):
        model.fit(features, features["nir"][:, 0])
