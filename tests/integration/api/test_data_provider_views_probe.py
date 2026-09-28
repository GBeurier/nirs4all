"""Real IO view buffers bound to DAG-ML's no-fit Python view probe."""

from __future__ import annotations

import copy
import inspect
import json
from typing import Any

import numpy as np
import pytest
from nirs4all_io import DataProvider, TensorSource

from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.pipeline.dagml.envelope import build_envelope
from nirs4all.pipeline.dagml.generated_views import GeneratedViewStore
from nirs4all.pipeline.dagml.identity import mint_identity
from tests.integration.api.test_multimodal_dagml import _cohort


def test_native_probe_binds_generated_io_buffers_to_exact_handle_and_ids() -> None:
    import dag_ml._dag_ml as native

    materialize_view = getattr(DataProvider, "materialize_view", None)
    if (not callable(getattr(native, "probe_data_view_in_process", None))
            or not callable(materialize_view)
            or "view_key" not in inspect.signature(materialize_view).parameters):
        pytest.skip("local DAG-ML probe and keyed nirs4all-io are not installed together")

    base = _cohort()
    scopes: list[dict[str, Any]] = []

    def generate(**_: Any) -> dict[str, Any]:
        return {"sample_ids": base.sample_ids, "sources": {"nir": base.sources["nir"]}}

    def generate_view(*, sample_ids: list[str], seed: int, context: dict[str, Any], **_: Any) -> dict[str, Any]:
        scopes.append(context["_dag_ml_view"])
        original = base.take(sample_ids).sources["nir"]
        return {"sample_ids": sample_ids, "sources": {"nir": TensorSource(
            np.asarray(original.values) + float(seed % 7), sample_ids,
            representation_id=original.representation_id,
            axis_units=original.axis_units,
            axis_coordinates=original.axis_coordinates,
        )}}

    provider = DataProvider(generate, generate_view=generate_view, provider_id="qualification.native-view-probe", base=base, replace_sources=["nir"])
    provider.materialize()
    store = GeneratedViewStore(provider)

    wrapped = MultimodalSpectroDataset(base)
    envelope = build_envelope(wrapped, mint_identity(wrapped), sample_ints=list(range(12)))
    selected = [base.sample_ids[1], base.sample_ids[0]]
    source_ids = [f"src{index}" for index in range(len(base.sources))]
    request = {
        "run_id": "run:provider.probe", "node_id": "model:provider.probe", "input_name": "x",
        "phase": "FIT_CV", "variant_id": None, "fold_id": "fold:0",
        "binding": {
            "node_id": "model:provider.probe", "input_name": "x", "request_id": envelope["plan"]["id"],
            "schema_fingerprint": envelope["schema_fingerprint"], "plan_fingerprint": envelope["plan_fingerprint"],
            "relation_fingerprint": envelope["relation_fingerprint"],
            "output_representation": envelope["plan"]["output_representation"],
            "feature_set_id": "x", "source_ids": source_ids, "require_relations": True, "metadata": {},
        },
        "data_handle": {"handle": 0, "kind": "data", "owner_controller": "controller:data.provider"},
        "view": {
            "sample_ids": selected, "partition": "fold_train", "fold_id": "fold:0",
            "source_ids": source_ids, "columns": None, "include_augmented": False,
            "include_excluded": False, "extra": {},
        },
        "view_key": "view:v1:" + "a" * 64, "view_seed": 19,
    }
    receipt = json.loads(native.probe_data_view_in_process(json.dumps(envelope), json.dumps(request), store))
    view = store.resolve(receipt["handle"], selected)
    assert view.sample_ids == tuple(selected)
    assert receipt["content_fingerprint"] == provider.view_state_dict(view)["fingerprint"]
    expected = np.asarray(base.take(selected).sources["nir"].values) + float(19 % 7)
    np.testing.assert_array_equal(view.sources["nir"].values, expected)
    assert scopes == [{"phase": "FIT_CV", "partition": "fold_train", "fold_id": "fold:0", "source_ids": source_ids}]

    with pytest.raises(ValueError, match="ordered IDs"):
        store.resolve(receipt["handle"], list(reversed(selected)))
    with pytest.raises(ValueError, match="Unknown generated"):
        store.resolve({**receipt["handle"], "handle": 999}, selected)

    for unsupported in (
        {"source_ids": source_ids[:1]},
        {"columns": ["first"]},
        {"include_augmented": True},
        {"include_excluded": True},
        {"extra": {"filter": "unexpected"}},
        {"branch_view": {"mode": "separation"}},
    ):
        changed = copy.deepcopy(request)
        changed["view"].update(unsupported)
        with pytest.raises(ValueError, match="does not support this native selector"):
            store({"request": changed, "handle": {**receipt["handle"], "handle": 100}})
    assert len(scopes) == 1

    validation_ids = [base.sample_ids[3], base.sample_ids[2]]
    validation_request = copy.deepcopy(request)
    validation_request["view"].update(sample_ids=validation_ids, partition="fold_validation")
    validation_request["view_key"] = "view:v1:" + "b" * 64
    validation_request["view_seed"] = 31
    validation_handle = {**receipt["handle"], "handle": receipt["handle"]["handle"] + 1000}
    validation_receipt = store({"request": validation_request, "handle": validation_handle})
    task = {
        "run_id": request["run_id"],
        "node_plan": {"node_id": request["node_id"]},
        "phase": request["phase"],
        "fold_id": request["fold_id"],
        "variant_id": request["variant_id"],
        "data_views": {
            "data:x": request["view"],
            "data:x:validation": validation_request["view"],
        },
        "input_handles": {
            "data:x": receipt["handle"],
            "data:x:validation": validation_receipt["handle"],
        },
        "data_view_receipts": {
            "data:x": receipt,
            "data:x:validation": validation_receipt,
        },
    }
    task_views = store.bind_task(task)
    train_row = task_views.take("x", "fold_train", [selected[1]])
    validation_row = task_views.take("x", "fold_validation", [validation_ids[0]])
    np.testing.assert_array_equal(
        train_row.sources["nir"].values,
        np.asarray(base.take([selected[1]]).sources["nir"].values) + float(19 % 7),
    )
    np.testing.assert_array_equal(
        validation_row.sources["nir"].values,
        np.asarray(base.take([validation_ids[0]]).sources["nir"].values) + float(31 % 7),
    )
    assert train_row.sample_ids == (selected[1],)
    assert validation_row.sample_ids == (validation_ids[0],)
    with pytest.raises(ValueError, match="within its native view"):
        task_views.take("x", "fold_train", [validation_ids[0]])
    with pytest.raises(ValueError, match="no native view"):
        task_views.take("x", "test", [selected[0]])
    with pytest.raises(ValueError, match="no native data-view handle"):
        store.bind_task({**task, "input_handles": {"data:x": receipt["handle"]}})
    with pytest.raises(ValueError, match="receipt for every data view"):
        store.bind_task({**task, "data_view_receipts": {"data:x": receipt}})
    altered_receipt = copy.deepcopy(task)
    altered_receipt["data_view_receipts"]["data:x"]["content_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="receipt does not match IO view"):
        store.bind_task(altered_receipt)
    with pytest.raises(ValueError, match="ambiguous native view scope"):
        store.bind_task({
            **{key: task[key] for key in ("run_id", "node_plan", "phase", "fold_id", "variant_id")},
            "data_views": {"data:x": validation_request["view"], "data:x:validation": validation_request["view"]},
            "input_handles": {"data:x": validation_handle, "data:x:validation": validation_handle},
            "data_view_receipts": {"data:x": validation_receipt, "data:x:validation": validation_receipt},
        })
    changed_scope = copy.deepcopy(task)
    changed_scope["data_views"]["data:x"]["partition"] = "fold_validation"
    with pytest.raises(ValueError, match="scope or selector changed"):
        store.bind_task(changed_scope)
    changed_fold = copy.deepcopy(task)
    changed_fold["fold_id"] = "fold:1"
    with pytest.raises(ValueError, match="scope or selector changed"):
        store.bind_task(changed_fold)
    colon_request = copy.deepcopy(request)
    colon_request["input_name"] = "aux:nir"
    colon_request["binding"]["input_name"] = "aux:nir"
    colon_request["view_key"] = "view:v1:" + "c" * 64
    colon_handle = {**receipt["handle"], "handle": receipt["handle"]["handle"] + 2000}
    colon_receipt = store({"request": colon_request, "handle": colon_handle})
    colon_task = {
        **{key: task[key] for key in ("run_id", "node_plan", "phase", "fold_id", "variant_id")},
        "data_views": {"data:aux:nir": colon_request["view"]},
        "input_handles": {"data:aux:nir": colon_handle},
        "data_view_receipts": {"data:aux:nir": colon_receipt},
    }
    assert store.bind_task(colon_task).take("aux:nir", "fold_train", [selected[0]]).sample_ids == (selected[0],)
