"""Real fixed IO reads admit CV Test companions without admitting Test FIT."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.model_selection import KFold

from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.operators.models.multimodal import MultimodalRegressor
from nirs4all.pipeline.dagml.cli_runner import data_bindings_for
from nirs4all.pipeline.dagml.envelope import build_envelope
from nirs4all.pipeline.dagml.fixed_cohort_views import FixedCohortViewStore
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml.named_torch import prepare_named_torch_pipeline
from nirs4all.pipeline.dagml.named_torch_estimator import DagMLNamedTorchEstimator
from nirs4all.pipeline.dagml.node_runner import resolve_named_model_features
from nirs4all.pipeline.dagml.resolver import MaterializationResolver
from nirs4all.pipeline.dagml_bridge import named_model_input_spec


def _fixture():
    ids = [f"row-{index}" for index in range(6)]
    X = np.arange(18, dtype=np.float32).reshape(6, 3)
    image = np.arange(72, dtype=np.float64).reshape(6, 2, 2, 3)
    cohort = MultimodalDataset({"nir": TensorSource(X, ids, representation_id="signal_1d"),
        "image": TensorSource(image, ids, representation_id="rgb_image")}, sample_ids=ids,
        y=X[:, 0], target_names=["y"], task_type="regression", partitions=["train"] * 4 + ["test"] * 2)
    dataset = MultimodalSpectroDataset(cohort)
    adapter = DagMLNamedTorchEstimator(factory_path="tests.fixtures.named_torch_nd.joint_factory",
        device="cpu", task_type="regression", epochs=1, batch_size=2)
    pipeline = prepare_named_torch_pipeline([KFold(2), {"model": MultimodalRegressor(
        dict.fromkeys(cohort.sources, "passthrough"), adapter, fusion="intermediate")}], dataset)
    specification = named_model_input_spec(pipeline)
    assert specification is not None
    identity = mint_identity(dataset)
    envelope = build_envelope(dataset, identity, sample_ints=[0, 1, 2, 3])
    resolver = MaterializationResolver(dataset, identity)
    store = FixedCohortViewStore(resolver, specification, envelope)
    bindings = data_bindings_for("model", envelope, model_input=specification)
    fit_ids = [identity.to_wire(0), identity.to_wire(1)]
    test_ids = [identity.to_wire(4), identity.to_wire(5)]
    task = {"run_id": "run:cv-companion", "phase": "FIT_CV", "variant_id": "variant:a", "fold_id": "fold:0", "seed": 123,
        "resources": {"cpu_threads": 1, "gpu_devices": []}, "node_plan": {"node_id": "model", "kind": "model",
        "controller_id": "controller:nirs4all.named_torch.test", "data_bindings": bindings},
        "data_views": {}, "input_handles": {}, "data_view_receipts": {}}
    calls = []
    for binding in bindings:
        name = binding["input_name"]
        for partition, sample_ids, suffix in (("fold_train", fit_ids, ""), ("predict", test_ids, ":test")):
            key = f"data:{name}{suffix}"
            view = {"partition": partition, "sample_ids": sample_ids, "fold_id": task["fold_id"], "source_ids": binding["source_ids"],
                "columns": None, "branch_view": None, "include_augmented": False, "include_excluded": False,
                "extra": {"feature_set_id": binding["feature_set_id"]}}
            handle = {"kind": "data_view", "handle": len(calls) + 1}
            request = {"run_id": task["run_id"], "node_id": "model", "phase": "FIT_CV", "fold_id": task["fold_id"],
                "variant_id": task["variant_id"], "input_name": name, "binding": binding, "view": view,
                "view_key": f"unit-cv:{name}:{partition}", "view_seed": 123}
            task["data_views"][key] = view
            task["input_handles"][key] = handle
            calls.append((key, {"request": request, "handle": handle}))
    return task, resolver, specification, store, calls, fit_ids, test_ids


def _materialized():
    task, resolver, specification, store, calls, fit_ids, test_ids = _fixture()
    for key, call in calls:
        task["data_view_receipts"][key] = store(call)
    return task, resolver, specification, store.bind_task(task), fit_ids, test_ids


def test_cv_companion_receipts_read_actual_nd_test_buffers_and_keep_fit_read_disjoint():
    task, resolver, specification, views, fit_ids, test_ids = _materialized()
    train = resolve_named_model_features(task, resolver, specification, fit_ids, "fold_train", views)
    test = resolve_named_model_features(task, resolver, specification, test_ids, "predict", views)
    assert set(fit_ids).isdisjoint(test_ids)
    np.testing.assert_array_equal(test["nir"], [[12, 13, 14], [15, 16, 17]])
    assert test["image"].shape == (2, 2, 2, 3) and test["image"].dtype == np.float64
    np.testing.assert_array_equal(test["image"], np.arange(48, 72, dtype=np.float64).reshape(2, 2, 2, 3))
    for name in test:
        views.record_model_call("fit", name, "fold_train", fit_ids, train[name], targets=np.asarray([[0], [3]]))
        views.record_model_call("predict", name, "predict", test_ids, test[name])
    evidence = views.consumed_data_views()
    for name in test:
        assert evidence[f"data:{name}:test"]["model_calls"][0]["operation"] == "predict"
        assert evidence[f"data:{name}"]["model_calls"][0]["sample_ids"] == fit_ids


def test_cv_companion_cannot_be_recorded_as_fit_even_after_a_valid_attested_read():
    task, resolver, specification, views, _fit_ids, test_ids = _materialized()
    test = resolve_named_model_features(task, resolver, specification, test_ids, "predict", views)
    with pytest.raises(ValueError, match="native training partition"):
        views.record_model_call("fit", "nir", "predict", test_ids, test["nir"], targets=np.ones((2, 1)))


@pytest.mark.parametrize("damage", ["augmentation", "columns", "branch", "source", "representation", "extra_policy", "partition", "duplicate_ids"])
def test_cv_companion_retains_all_source_selector_and_identity_refusals(damage):
    _task, _resolver, _specification, store, calls, _fit_ids, _test_ids = _fixture()
    call = deepcopy(next(call for key, call in calls if key.endswith(":test")))
    view = call["request"]["view"]
    if damage == "augmentation":
        view["include_augmented"] = True
    elif damage == "columns":
        view["columns"] = [0]
    elif damage == "branch":
        view["branch_view"] = {}
    elif damage == "source":
        view["source_ids"] = ["foreign"]
    elif damage == "representation":
        call["request"]["binding"]["output_representation"] = "foreign"
    elif damage == "extra_policy":
        view["extra"]["include_augmented_cv_train_predictions"] = True
    elif damage == "partition":
        view["partition"] = "all_observations"
    else:
        view["sample_ids"][1] = view["sample_ids"][0]
    with pytest.raises(ValueError):
        store(call)
