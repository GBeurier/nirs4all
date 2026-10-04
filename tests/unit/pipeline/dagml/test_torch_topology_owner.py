"""Closed CPU Torch owner scope; real fitting and adversarial OOF inputs."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from nirs4all.pipeline.dagml import torch_topology_replay as boundary
from nirs4all.pipeline.dagml.node_runner import run_node
from nirs4all.pipeline.dagml.torch_estimator import DagMLTorchEstimator


def _task(phase="FIT_CV", owner="controller:nirs4all.model"):
    return {"phase": phase, "variant_id": "variant:one", "fold_id": "fold0" if phase == "FIT_CV" else None,
            "node_plan": {"node_id": "raw", "kind": "model", "controller_id": owner, "params": {}},
            "resources": {"cpu_threads": 1, "gpu_devices": []}, "data_views": {}, "prediction_inputs": {}}


def _metadata():
    return {"python_torch_profile": {"schema_version": 1, "profile": "cpu_serial_regression_v1", "seed": 42,
            "source_order": ["a", "b", "c", "d"], "source_widths": dict.fromkeys("abcd", 2),
            "cpu_threads": 1, "gpu_devices": [], "target_names": ["y"], "training_policy": boundary.TRAINING_POLICY}}


def test_real_cpu_models_start_cold_and_task_rng_is_restored(monkeypatch):
    torch = pytest.importorskip("torch")
    from nirs4all.pipeline.dagml import node_runner

    X = np.arange(16, dtype=np.float32).reshape(8, 2) / 16
    y = X[:, :1] - X[:, 1:]
    task = _task()
    task["node_plan"]["params"] = DagMLTorchEstimator(factory_path=boundary.TORCH_FACTORY, factory_params={"hidden_units": 4}, force_layout="2d", task_type="regression", epochs=2, batch_size=3, device="cpu").get_params(deep=False)
    node = {"id": "raw", "operator": boundary.TORCH_CLASS, "metadata": {"source_selection": ["a"], "source_index": 0}}
    resolver = SimpleNamespace(_dataset=object(), partition_wire_ids=lambda _partition: [])
    monkeypatch.setattr(node_runner, "_train_predict_ids", lambda _task: ([], []))
    monkeypatch.setattr(boundary, "_validate_sources", lambda *_args: None)
    models = []

    def fit(*_args, **_kwargs):
        estimator = DagMLTorchEstimator(factory_path=boundary.TORCH_FACTORY, factory_params={"hidden_units": 4},
                                      force_layout="2d", task_type="regression", epochs=2, batch_size=3, device="cpu")
        estimator.fit(X, y)
        models.append(estimator)
        return {"values": estimator.predict(X)}

    monkeypatch.setattr(node_runner, "_run_node", fit)
    before = torch.random.get_rng_state().clone()
    first = run_node(copy.deepcopy(task), resolver, lambda _id: node, {}, graph_metadata=_metadata())
    assert torch.equal(before, torch.random.get_rng_state())
    second = run_node(copy.deepcopy(task), resolver, lambda _id: node, {}, graph_metadata=_metadata())
    assert torch.equal(before, torch.random.get_rng_state())
    assert models[0].model_ is not models[1].model_
    assert all(parameter.device.type == "cpu" for estimator in models for parameter in estimator.model_.parameters())
    np.testing.assert_array_equal(first["values"], second["values"])
    changed = copy.deepcopy(task)
    changed["variant_id"] = "variant:two"
    run_node(changed, resolver, lambda _id: node, {}, graph_metadata=_metadata())
    assert not torch.equal(next(models[0].model_.parameters()), next(models[2].model_.parameters()))


@pytest.mark.parametrize("fault", ["foreign", "train", "test", "outer", "missing", "reordered"])
def test_meta_refuses_wrong_producer_partition_outer_or_incomplete_oof_before_fit(fault):
    task = _task(owner="controller:nirs4all.meta_model")
    node = {"id": "raw", "operator": "sklearn.linear_model._ridge.Ridge", "metadata": {}}
    edges = [{"source": {"node_id": name, "port_name": "oof"}, "target": {"node_id": "raw", "port_name": "x"},
              "contract": {"requires_oof": True}} for name in ("left", "right")]
    for name in ("left", "right"):
        task["prediction_inputs"][name + ".oof"] = {"producer_node": name, "source_port": "oof", "target_port": "x",
            "sample_ids": ["s0", "s1"], "values": [[1.0], [2.0]], "partition": "validation", "fold_ids": ["fold0.inner.fold0", "fold0.inner.fold1"]}
    block = task["prediction_inputs"]["left.oof"]
    if fault == "foreign":
        block["producer_node"] = "foreign"
    if fault in {"train", "test"}:
        block["partition"] = fault
    if fault == "outer":
        block["fold_ids"] = ["fold0"]
    if fault == "missing":
        del task["prediction_inputs"]["right.oof"]
    if fault == "reordered":
        task["prediction_inputs"]["right.oof"]["sample_ids"].reverse()
    with pytest.raises(ValueError, match="Torch topology"):
        with boundary.torch_task_scope(task, SimpleNamespace(), lambda _id: node, edges, _metadata()):
            pytest.fail("invalid OOF reached an owner callback")


def test_known_factory_without_metadata_and_parallel_task_refuse_before_dispatch():
    task = _task()
    task["node_plan"]["params"]["factory_path"] = boundary.TORCH_FACTORY
    node = {"id": "raw", "operator": boundary.TORCH_CLASS, "metadata": {}}
    with pytest.raises(ValueError, match="signed native graph metadata"):
        run_node(task, SimpleNamespace(), lambda _id: node, {})
    task["resources"]["cpu_threads"] = 2
    with pytest.raises(ValueError, match="serial CPU"):
        run_node(task, SimpleNamespace(), lambda _id: node, {}, graph_metadata=_metadata())


def test_explicit_cpu_preserves_automatic_device_default_without_querying_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    from nirs4all.controllers.models.torch_model import PyTorchModelController

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: calls.append("auto") or False)
    controller = PyTorchModelController()
    X, y = torch.ones((4, 2)), torch.ones((4, 1))
    explicit = torch.nn.Linear(2, 1)
    controller._train_model(explicit, X, y, epochs=1, batch_size=2, device="cpu")
    assert next(explicit.parameters()).device.type == "cpu"
    explicit_queries = len(calls)
    automatic = torch.nn.Linear(2, 1)
    controller._train_model(automatic, X, y, epochs=1, batch_size=2)
    assert next(automatic.parameters()).device.type == "cpu"
    assert len(calls) > explicit_queries


def test_named_source_subset_order_selects_actual_columns_and_keeps_sample_order():
    blocks = [np.array([[i, i + 0.5], [i + 10, i + 10.5]], dtype=np.float32) for i in range(4)]
    requests = []
    resolver = SimpleNamespace(_dataset=SimpleNamespace(source_names=["a", "b", "c", "d"]))
    def resolve(ids, **options):
        requests.append((ids, options))
        return {"blocks": blocks}
    resolver.resolve_feature_blocks = resolve
    values = boundary.selected_torch_features(resolver, ["s1", "s0"], ["d", "b"], include_augmented=False)
    np.testing.assert_array_equal(values, np.hstack([blocks[3], blocks[1]]))
    assert requests[0][0] == ["s1", "s0"]
    with pytest.raises(ValueError, match="foreign or duplicate"):
        boundary.selected_torch_features(resolver, ["s1", "s0"], ["a", "a"], include_augmented=False)


@pytest.mark.parametrize("role", ["Torch", "Ridge"])
def test_emitted_refit_attestation_rejects_compatible_branch_swaps_and_foreign_fits(role):
    torch = pytest.importorskip("torch")
    from sklearn.linear_model import Ridge

    metadata = _metadata()
    metadata["source_schemas"] = {name: {"identity": name} for name in "abcd"}
    X = np.arange(24, dtype=np.float32).reshape(12, 2) / 24
    y = X[:, :1] - X[:, 1:]
    models, bundles, refs = [], [], []
    nodes = {"left": {"id": "left", "operator": boundary.TORCH_CLASS, "metadata": {"source_selection": ["a"]}},
             "right": {"id": "right", "operator": boundary.TORCH_CLASS, "metadata": {"source_selection": ["b"]}}}
    for index, identifier in enumerate(("left", "right")):
        if role == "Torch":
            model = DagMLTorchEstimator(factory_path=boundary.TORCH_FACTORY, factory_params={"hidden_units": 4},
                                         force_layout="2d", task_type="regression", epochs=1, batch_size=4, device="cpu")
            node = nodes[identifier]
            owner = "controller:nirs4all.model"
            inputs = {}
        else:
            model = Ridge(alpha=0.5, solver="svd")
            node = {"id": identifier, "operator": "sklearn.linear_model._ridge.Ridge",
                    "metadata": {"prediction_source_order": ["left", "right"]}}
            owner = "controller:nirs4all.meta_model"
            inputs = {name + ".oof": {"producer_node": name, "source_port": "oof", "target_port": "x",
                      "partition": "validation", "fold_id": None, "fold_ids": ["stacking.refit.inner.fold0", "stacking.refit.inner.fold1"],
                      "sample_ids": [f"s{i}" for i in range(12)], "values": X[:, col:col+1].tolist()}
                      for col, name in enumerate(("left", "right"))}
        task = {**_task("REFIT", owner), "run_id": "run:original", "seed": 7, "prediction_inputs": inputs}
        task["node_plan"]["node_id"] = identifier
        task["node_plan"]["params"] = model.get_params(deep=False)
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(boundary._task_seed(task, metadata["python_torch_profile"], identifier))
            model.fit(X + index, y + index)
        bundle = {"estimator": model, "y_transform": None}
        ref = {"id": "artifact:" + identifier, "controller_id": owner, "kind": "sklearn_estimator", "backend": "joblib"}
        boundary.emit_refit_attestation(task, node, bundle, ref, [f"s{i}" for i in range(12)], metadata, nodes.__getitem__, [])
        boundary.validate_refit_attestation(bundle, ref)
        with pytest.raises(ValueError, match="re-attest"):
            boundary.emit_refit_attestation(task, node, bundle, ref, [f"s{i}" for i in range(12)], metadata, nodes.__getitem__, [])
        models.append(model)
        bundles.append(bundle)
        refs.append(ref)
    assert models[0].get_params(deep=False) == models[1].get_params(deep=False)
    with pytest.raises(ValueError, match="foreign or missing REFIT origin"):
        boundary.validate_refit_attestation({**bundles[0], "estimator": models[1]}, refs[0])
    with pytest.raises(ValueError, match="attestation is missing or changed"):
        boundary.validate_refit_attestation(bundles[1], refs[0])
    foreign = copy.deepcopy(models[0])
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(999)
        foreign.fit(X + 3, y + 3)
    assert foreign.get_params(deep=False) == models[0].get_params(deep=False)
    with pytest.raises(ValueError, match="learned state"):
        boundary.validate_refit_attestation({**bundles[0], "estimator": foreign}, refs[0])


@pytest.mark.parametrize("role", ["raw source", "targets"])
def test_owner_float32_overflow_refuses_before_numerical_callback(role):
    with pytest.raises(ValueError, match="float32"):
        boundary._finite_float32(np.array([[1e300]], dtype=np.float64), role + " is not finite after float32 conversion")
    boundary._finite_float32(np.array([[np.nextafter(float(np.finfo(np.float32).max), 0.0)]], dtype=np.float64), role)
    boundary._finite_float32(np.array([[0.10000000000000001]], dtype=np.float64), role)


@pytest.mark.parametrize("values", [[1.0, 2.0], [[1.0], [2.0]]])
def test_owner_accepts_resolver_flat_or_column_mono_target(monkeypatch, values):
    pytest.importorskip("torch")
    from nirs4all.pipeline.dagml import node_runner

    task = _task()
    task["node_plan"]["params"] = DagMLTorchEstimator(factory_path=boundary.TORCH_FACTORY, factory_params={"hidden_units": 4},
        force_layout="2d", task_type="regression", epochs=1, batch_size=2, device="cpu").get_params(deep=False)
    node = {"id": "raw", "operator": boundary.TORCH_CLASS, "metadata": {"source_selection": ["a"], "source_index": 0}}
    resolver = SimpleNamespace(_identity=SimpleNamespace(to_wire=lambda sample: f"s{sample}"),
        _dataset=SimpleNamespace(index_column=lambda *_args: [0, 1]),
        resolve_targets=lambda _ids: {"values": values, "target_names": ["y"]}, partition_wire_ids=lambda _partition: [])
    monkeypatch.setattr(node_runner, "_train_predict_ids", lambda _task: (["s0", "s1"], []))
    monkeypatch.setattr(boundary, "_validate_sources", lambda *_args: None)
    with boundary.torch_task_scope(task, resolver, lambda _id: node, [], _metadata()):
        pass  # Genuine owner admission, without fitting a second test model.


@pytest.mark.parametrize("values,names", [
    ([[1.0, 2.0], [3.0, 4.0]], ["y"]),
    ([[[1.0]], [[2.0]]], ["y"]),
    ([1.0], ["y"]),
    ([1.0, float("nan")], ["y"]),
    ([1.0, float("inf")], ["y"]),
    ([1.0, 1e300], ["y"]),
    ([1.0, 2.0], ["other"]),
])
def test_owner_scalar_target_refuses_multi_target_wrong_rows_names_and_nonfinite_before_callback(monkeypatch, values, names):
    from nirs4all.pipeline.dagml import node_runner

    task = _task()
    task["node_plan"]["params"] = DagMLTorchEstimator(factory_path=boundary.TORCH_FACTORY, factory_params={"hidden_units": 4},
        force_layout="2d", task_type="regression", epochs=1, batch_size=2, device="cpu").get_params(deep=False)
    node = {"id": "raw", "operator": boundary.TORCH_CLASS, "metadata": {"source_selection": ["a"], "source_index": 0}}
    resolver = SimpleNamespace(_identity=SimpleNamespace(to_wire=lambda sample: f"s{sample}"),
        _dataset=SimpleNamespace(index_column=lambda *_args: [0, 1]),
        resolve_targets=lambda _ids: {"values": values, "target_names": names}, partition_wire_ids=lambda _partition: [])
    monkeypatch.setattr(node_runner, "_train_predict_ids", lambda _task: (["s0", "s1"], []))
    with pytest.raises(ValueError, match="Torch topology"):
        with boundary.torch_task_scope(task, resolver, lambda _id: node, [], _metadata()):
            pytest.fail("invalid scalar target reached the numerical owner")
