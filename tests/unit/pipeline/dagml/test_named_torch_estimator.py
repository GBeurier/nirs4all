"""Named tensors, joint gradients, independent Torch training and fitted replay."""
from __future__ import annotations

import base64
import copy
import pickle

import cloudpickle
import numpy as np
import pytest
from sklearn.base import clone

from nirs4all.operators.models.multimodal import MultimodalRegressor
from nirs4all.pipeline.dagml.named_torch import named_learned_state_sha256, named_torch_estimator, prepare_named_torch_pipeline
from nirs4all.pipeline.dagml.named_torch_estimator import DagMLNamedTorchEstimator

torch = pytest.importorskip("torch")


class JointEncoderFixture(torch.nn.Module):
    """Software fixture: two private encoders feeding a shared learned head."""

    def __init__(self):
        super().__init__()
        self.nir_encoder = torch.nn.Linear(3, 2)
        self.clinical_encoder = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(4, 1)

    def forward(self, *, nir, clinical):
        return self.head(torch.cat((torch.tanh(self.nir_encoder(nir)), torch.tanh(self.clinical_encoder(clinical))), dim=1))


def fixture():
    rng = np.random.default_rng(117)
    inputs = {"nir": rng.normal(size=(18, 3)), "clinical": rng.normal(size=(18, 2))}
    target = inputs["nir"][:, 0] - 0.4 * inputs["clinical"][:, 1]
    torch.manual_seed(117)
    template = JointEncoderFixture()
    adapter = DagMLNamedTorchEstimator(
        template_blob=base64.b64encode(cloudpickle.dumps(template)).decode("ascii"),
        task_type="regression", device="cpu", epochs=3, batch_size=6, patience=3, lr=0.01,
    )
    return inputs, target, template, adapter


def test_joint_fit_matches_independent_torch_and_changes_both_encoders():
    from torch.utils.data import DataLoader, TensorDataset

    inputs, target, template, adapter = fixture()
    before = {name: values.copy() for name, values in inputs.items()}
    torch.manual_seed(119)
    adapter.fit(inputs, target)
    expected = copy.deepcopy(template)
    torch.manual_seed(119)
    optimizer = torch.optim.Adam(expected.parameters(), lr=0.01)
    dataset = TensorDataset(torch.tensor(inputs["nir"], dtype=torch.float32),
                           torch.tensor(inputs["clinical"], dtype=torch.float32),
                           torch.tensor(target[:, None], dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=6, shuffle=True)
    for _ in range(3):
        for nir, clinical, y in loader:
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(expected(nir=nir, clinical=clinical), y)
            loss.backward()
            optimizer.step()
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(adapter.model_.state_dict()[name], value, rtol=0, atol=0)
    assert not torch.equal(adapter.model_.nir_encoder.weight, template.nir_encoder.weight)
    assert not torch.equal(adapter.model_.clinical_encoder.weight, template.clinical_encoder.weight)
    for name, values in inputs.items():
        np.testing.assert_array_equal(values, before[name])


def test_fitted_replay_keeps_names_and_weights_without_fit(monkeypatch):
    inputs, target, _, adapter = fixture()
    adapter.fit(inputs, target)
    expected = adapter.predict(inputs)
    replay = pickle.loads(pickle.dumps(adapter))
    monkeypatch.setattr(DagMLNamedTorchEstimator, "fit", lambda *a, **k: pytest.fail("Replay called fit"))
    np.testing.assert_array_equal(replay.predict(dict(reversed(list(inputs.items())))), expected)
    assert replay.get_params(deep=False) == adapter.get_params(deep=False)


def test_clone_contains_configuration_without_fitted_weights():
    inputs, target, _, adapter = fixture()
    adapter.fit(inputs, target)
    cloned = clone(adapter)
    assert cloned.get_params(deep=False) == adapter.get_params(deep=False)
    assert not hasattr(cloned, "model_")


@pytest.mark.parametrize("column_target", [False, True])
def test_public_intermediate_fit_preserves_named_buffers_and_target_rank(monkeypatch, column_target):
    inputs, target, _, adapter = fixture()
    target = target[:, None] if column_target else target
    wrapper = MultimodalRegressor(dict.fromkeys(inputs, "passthrough"), adapter, fusion="intermediate")
    fresh = clone(wrapper)
    torch.manual_seed(123)
    fresh.fit(list(inputs.values()), target)
    torch.manual_seed(123)
    expected = clone(adapter).fit(inputs, target).predict(inputs)
    actual = fresh.predict(list(inputs.values()))
    assert actual.shape == target.shape
    np.testing.assert_array_equal(actual.reshape(-1, 1), expected)
    assert fresh.model_.input_dtypes_ == {name: values.dtype.str for name, values in inputs.items()}
    assert not hasattr(wrapper, "model_")
    restored = pickle.loads(pickle.dumps(fresh))
    monkeypatch.setattr(DagMLNamedTorchEstimator, "fit", lambda *a, **k: pytest.fail("Public replay called fit"))
    np.testing.assert_array_equal(restored.predict(list(inputs.values())), actual)


def test_supplied_module_template_retains_private_initial_state_and_controls():
    _, _, template, _ = fixture()
    wrapper = MultimodalRegressor({"nir": None, "clinical": "passthrough"}, template, fusion="intermediate")
    adapter = named_torch_estimator(wrapper, train_params={"epochs": 3, "batch_size": 6, "lr": 0.01})
    module = adapter._new_named_model({"nir": (3,), "clinical": (2,)})
    assert module is not template
    for name, values in template.state_dict().items():
        torch.testing.assert_close(module.state_dict()[name], values, rtol=0, atol=0)
    assert adapter.epochs == 3 and adapter.batch_size == 6 and adapter.lr == 0.01


@pytest.mark.parametrize("changes", [
    {"transformers": {"nir": "passthrough"}},
    {"transformers": {"nïr": None, "clinical": None}},
    {"transformers": {"y": None, "clinical": None}},
    {"transformers": {"nir": object(), "clinical": None}},
    {"source_weights": {"nir": 1.0}},
    {"missing_source_policy": "zero_with_indicator"},
    {"target_policy": "per_target"},
])
def test_named_public_profile_refuses_unsupported_declarations_before_fit(monkeypatch, changes):
    inputs, target, _, adapter = fixture()
    wrapper = MultimodalRegressor(dict.fromkeys(inputs), adapter, fusion="intermediate").set_params(**changes)
    monkeypatch.setattr(DagMLNamedTorchEstimator, "fit", lambda *a, **k: pytest.fail("Invalid profile reached fit"))
    with pytest.raises(ValueError):
        wrapper.fit(list(inputs.values()), target)


@pytest.mark.parametrize("mutation", ["name", "shape", "dtype", "overflow"])
def test_prediction_refuses_changed_contract_before_forward(monkeypatch, mutation):
    inputs, target, _, adapter = fixture()
    adapter.fit(inputs, target)
    changed = dict(inputs)
    if mutation == "name":
        changed["other"] = changed.pop("clinical")
    elif mutation == "shape":
        changed["clinical"] = changed["clinical"][:, :1]
    elif mutation == "dtype":
        changed["clinical"] = changed["clinical"].astype(np.float32)
    else:
        changed["clinical"] = np.full_like(changed["clinical"], np.finfo(np.float64).max)
    monkeypatch.setattr(adapter.model_, "forward", lambda **k: pytest.fail("Invalid input reached forward"))
    with pytest.raises(ValueError):
        adapter.predict(changed)


def test_learned_identity_survives_pickle_and_detects_changed_weights_or_schema():
    inputs, target, _, adapter = fixture()
    adapter.fit(inputs, target)
    original = named_learned_state_sha256(adapter)
    restored = pickle.loads(pickle.dumps(adapter))
    assert named_learned_state_sha256(restored) == original
    with torch.no_grad():
        restored.model_.clinical_encoder.weight[0, 0] += 0.125
    assert named_learned_state_sha256(restored) != original
    restored = pickle.loads(pickle.dumps(adapter))
    restored.input_dtypes_["clinical"] = np.dtype("float32").str
    assert named_learned_state_sha256(restored) != original
    with pytest.raises(ValueError, match="original fitted"):
        named_learned_state_sha256(clone(adapter))


def test_float64_module_is_refused_before_optimizer(monkeypatch):
    from nirs4all.controllers.models.torch_model import PyTorchModelController

    inputs, target, template, adapter = fixture()
    adapter.template_blob = base64.b64encode(cloudpickle.dumps(template.double())).decode("ascii")
    monkeypatch.setattr(PyTorchModelController, "_train_model", lambda *a, **k: pytest.fail("Invalid dtype reached training"))
    with pytest.raises(ValueError, match="float32 model parameters"):
        adapter.fit(inputs, target)


def test_named_admission_keeps_each_io_source_identity_and_does_not_mutate_pipeline(monkeypatch):
    from nirs4all_io import MultimodalDataset, TensorSource
    from sklearn.model_selection import KFold

    from nirs4all.data.multimodal import MultimodalSpectroDataset
    from nirs4all.pipeline.dagml.steps import _apply_model_params

    inputs, target, _, adapter = fixture()
    ids = [f"sample_{index}" for index in range(len(target))]
    dataset = MultimodalSpectroDataset(MultimodalDataset(
        {name: TensorSource(values, ids, representation_id="tabular_numeric") for name, values in inputs.items()},
        sample_ids=ids, y=target, task_type="regression",
    ))
    # Wrapper declaration order differs from IO storage order deliberately.
    wrapper = MultimodalRegressor({"clinical": None, "nir": None}, adapter, fusion="intermediate")
    pipeline = [KFold(n_splits=3), {"model": wrapper, "train_params": {"epochs": 4}}]
    monkeypatch.setattr(DagMLNamedTorchEstimator, "fit", lambda *a, **k: pytest.fail("Admission called fit"))
    admitted = prepare_named_torch_pipeline(pipeline, dataset)
    assert pipeline[1]["model"] is wrapper and pipeline[1]["train_params"] == {"epochs": 4}
    assert "model_input" not in pipeline[1]
    assert admitted[0] is pipeline[0] and admitted[1]["model"].epochs == 4
    ports = admitted[1]["model_input"]["ports"]
    assert [(port["name"], port["metadata"]["source_id"], port["metadata"]["feature_shape"])
            for port in ports] == [("clinical", "src1", [2]), ("nir", "src0", [3])]
    assert [port["metadata"]["dtype"] for port in ports] == ["float64", "float64"]
    lowered = _apply_model_params([admitted[1]])
    assert lowered[0]["model_input"] == admitted[1]["model_input"]


def test_named_admission_refuses_effective_target_overflow_before_fit(monkeypatch):
    from nirs4all_io import MultimodalDataset, TensorSource

    from nirs4all.data.multimodal import MultimodalSpectroDataset

    inputs, target, _, adapter = fixture()
    ids = [f"sample_{index}" for index in range(len(target))]
    dataset = MultimodalSpectroDataset(MultimodalDataset(
        {name: TensorSource(values, ids, representation_id="tabular_numeric") for name, values in inputs.items()},
        sample_ids=ids, y=np.full_like(target, np.finfo(np.float64).max), task_type="regression",
    ))
    wrapper = MultimodalRegressor(dict.fromkeys(inputs), adapter, fusion="intermediate")
    monkeypatch.setattr(DagMLNamedTorchEstimator, "fit", lambda *a, **k: pytest.fail("Invalid target reached fit"))
    with pytest.raises(ValueError, match="effective float32"):
        prepare_named_torch_pipeline([wrapper], dataset)
