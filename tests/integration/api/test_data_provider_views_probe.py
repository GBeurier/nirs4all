"""Real IO view buffers bound to DAG-ML's no-fit Python view probe."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from nirs4all_io import DataProvider, TensorSource
from nirs4all_io.ragged import RaggedSeriesBatch

from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.dagml.envelope import build_envelope
from nirs4all.pipeline.dagml.generated_views import GeneratedViewStore, _model_value_fingerprint
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml.native_results import read_native_results, write_native_results
from tests.integration.api.test_multimodal_dagml import _cohort


def test_generated_hpo_trial_stores_serialize_shared_provider_callbacks() -> None:
    """Trial handle namespaces may differ, but IO view generation is shared state."""
    base = _cohort()
    active = 0
    peak_active = 0
    counter_lock = threading.Lock()
    start = threading.Barrier(2)

    def generate(**_: Any) -> dict[str, Any]:
        return {"sample_ids": base.sample_ids, "sources": {"nir": base.sources["nir"]}}

    def generate_view(*, sample_ids: list[str], seed: int, **_: Any) -> dict[str, Any]:
        nonlocal active, peak_active
        with counter_lock:
            active += 1
            peak_active = max(peak_active, active)
        try:
            time.sleep(0.1)  # Releases the GIL while the shared provider callback is active.
            original = base.take(sample_ids).sources["nir"]
            return {"sample_ids": sample_ids, "sources": {"nir": TensorSource(
                np.asarray(original.values) + float(seed % 7), sample_ids,
                representation_id=original.representation_id,
                axis_units=original.axis_units,
                axis_coordinates=original.axis_coordinates,
            )}}
        finally:
            with counter_lock:
                active -= 1

    provider = DataProvider(
        generate, generate_view=generate_view, provider_id="qualification.concurrent-trial-views",
        base=base, replace_sources=["nir"],
    )
    provider.materialize()
    root = GeneratedViewStore(provider)
    trials = [root.for_trial(), root.for_trial()]
    selected = [base.sample_ids[0], base.sample_ids[1]]

    def run_trial(index: int) -> tuple[dict[str, Any], Any]:
        request = {
            "input_name": "x", "phase": "FIT_CV",
            "binding": {"input_name": "x", "feature_set_id": "x", "source_ids": ["src0"], "metadata": {}},
            "view": {
                "sample_ids": selected, "partition": "fold_train", "fold_id": "fold:0",
                "source_ids": ["src0"], "columns": None, "branch_view": None,
                "include_augmented": False, "include_excluded": False, "extra": {},
            },
            "view_key": "view:v1:" + str(index + 1) * 64, "view_seed": 19 + index,
        }
        handle = {"kind": "data_view", "handle": 1, "owner_controller": "controller:data.provider"}
        start.wait(timeout=5)
        receipt = trials[index]({"request": request, "handle": handle})
        return receipt, trials[index].resolve(handle, selected)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_trial, range(2)))

    assert peak_active == 1
    assert [receipt["view_key"] for receipt, _ in results] == ["view:v1:" + digit * 64 for digit in ("1", "2")]
    for index, (_, view) in enumerate(results):
        expected = np.asarray(base.take(selected).sources["nir"].values) + float((19 + index) % 7)
        np.testing.assert_array_equal(view.sources["nir"].values, expected)


def test_native_probe_binds_generated_io_buffers_to_exact_handle_and_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(store)

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
    probe = json.loads(native.probe_data_view_in_process(json.dumps(envelope), json.dumps(request), store, True))
    generated_manifest = probe.pop("generated_view_manifest")
    receipt = probe
    view = store.resolve(receipt["handle"], selected)
    assert view.sample_ids == tuple(selected)
    assert receipt["content_fingerprint"] == provider.view_state_dict(view)["fingerprint"]
    expected = np.asarray(base.take(selected).sources["nir"].values) + float(19 % 7)
    np.testing.assert_array_equal(view.sources["nir"].values, expected)
    assert scopes == [{"partition": "fold_train", "fold_id": "fold:0", "source_ids": source_ids}]

    # The no-fit probe's genuine IO receipt survives the native results writer
    # and reader; tampered bytes or a rewritten SHA cannot bypass DAG-ML's TCV1.
    result = SimpleNamespace(predictions=Predictions(), _dagml_refit_artifacts=[], best={},
                             _dagml_generated_view_manifest=generated_manifest)
    run_dir = write_native_results(result, {"reports": []}, tmp_path)
    assert read_native_results(run_dir)["generated_view_manifest"] == generated_manifest
    assert json.loads((run_dir / "manifest.json").read_text())["schema_version"] == 4
    result._dagml_refit_artifacts = [{"estimator": object()}]
    result._dagml_initial_full_refit_package = {"unreplayable": True}
    filtered_dir = write_native_results(result, {"reports": []}, tmp_path)
    filtered = read_native_results(filtered_dir)
    assert filtered["artifacts"] == []
    assert filtered["manifest"]["capabilities"]["has_model_artifacts"] is False
    assert not (filtered_dir / "artifacts").exists()
    assert not (filtered_dir / "initial_full_refit_package.json").exists()
    result.per_dataset = {"synthetic": {"residual_replay": {}}}
    with pytest.raises(ValueError, match="cannot carry model replay metadata"):
        write_native_results(result, {"reports": []}, tmp_path)
    result.per_dataset = {}
    header_path = run_dir / "manifest.json"
    clean_header = json.loads(header_path.read_text())
    forged_header = copy.deepcopy(clean_header)
    forged_header["artifacts"] = [{"uri": "artifacts/forged.joblib"}]
    header_path.write_text(json.dumps(forged_header))
    with pytest.raises(ValueError, match="v4 cannot contain replayable model artifacts"):
        read_native_results(run_dir)
    header_path.write_text(json.dumps(clean_header))
    manifest_path = run_dir / "generated_view_manifest.json"
    original_bytes = manifest_path.read_bytes()
    manifest_path.write_bytes(original_bytes + b" ")
    with pytest.raises(ValueError, match="byte fingerprint mismatch"):
        read_native_results(run_dir)
    altered = copy.deepcopy(generated_manifest)
    altered["views"][0]["view"]["unexpected"] = 1
    altered_bytes = json.dumps(altered).encode("utf-8")
    manifest_path.write_bytes(altered_bytes)
    header = json.loads(header_path.read_text())
    header["generated_view_manifest_ref"]["sha256"] = hashlib.sha256(altered_bytes).hexdigest()
    header_path.write_text(json.dumps(header))
    with pytest.raises(ValueError, match="generated view manifest is invalid"):
        read_native_results(run_dir)

    altered = copy.deepcopy(generated_manifest)
    altered["views"][0]["content_fingerprint"] = "0" * 64
    altered_bytes = json.dumps(altered).encode("utf-8")
    manifest_path.write_bytes(altered_bytes)
    header["generated_view_manifest_ref"]["sha256"] = hashlib.sha256(altered_bytes).hexdigest()
    header_path.write_text(json.dumps(header))
    with pytest.raises(ValueError, match="generated view manifest is invalid"):
        read_native_results(run_dir)

    duplicate = original_bytes.replace(b'"views":', b'"views":[],"views":', 1)
    assert duplicate != original_bytes
    manifest_path.write_bytes(duplicate)
    header["generated_view_manifest_ref"]["sha256"] = hashlib.sha256(duplicate).hexdigest()
    header_path.write_text(json.dumps(header))
    with pytest.raises(ValueError, match="generated view manifest is invalid"):
        read_native_results(run_dir)

    header["schema_version"] = 3
    header_path.write_text(json.dumps(header))
    with pytest.raises(ValueError, match="requires schema v4 or v5"):
        read_native_results(run_dir)
    header["schema_version"] = 4
    manifest_path.unlink()
    header.pop("generated_view_manifest_ref")
    header_path.write_text(json.dumps(header))
    with pytest.raises(ValueError, match="generated results require a generated view manifest reference"):
        read_native_results(run_dir)

    with pytest.raises(ValueError, match="ordered IDs"):
        store.resolve(receipt["handle"], list(reversed(selected)))
    with pytest.raises(ValueError, match="Unknown generated"):
        store.resolve({**receipt["handle"], "handle": 999}, selected)

    for unsupported in (
        {"source_ids": source_ids[:1]},
        {"columns": ["first"]},
        {"partition": "fold_validation", "include_augmented": True},
        {"include_excluded": True},
        {"extra": {"filter": "unexpected"}},
        {"branch_view": {"mode": "separation"}},
    ):
        changed = copy.deepcopy(request)
        changed["view"].update(unsupported)
        with pytest.raises(ValueError, match="does not support this native selector"):
            store({"request": changed, "handle": {**receipt["handle"], "handle": 100}})
    mismatched_source_index = copy.deepcopy(request)
    mismatched_source_index["binding"]["metadata"] = {"source_index": {source: index for index, source in enumerate(source_ids)}}
    mismatched_source_index["view"]["extra"] = {"source_index": {source_ids[0]: 1}}
    with pytest.raises(ValueError, match="does not support this native selector"):
        store({"request": mismatched_source_index, "handle": {**receipt["handle"], "handle": 100}})
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
    task_views.validate_task(task)
    with pytest.raises(ValueError, match="different native task"):
        task_views.validate_task({**task, "fold_id": "fold:other"})

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import ShuffleSplit

    from nirs4all.pipeline.dagml.node_runner import run_node
    from nirs4all.pipeline.dagml.resolver import MaterializationResolver
    from nirs4all.pipeline.dagml_bridge import build_dagml_plan

    plan = build_dagml_plan([ShuffleSplit(n_splits=2, random_state=7), {"model": PLSRegression(n_components=1)}],
                            plan_id="provider-view", dsl_id="provider-view").to_dict()
    graph_model = next(node for node in plan["graph_plan"]["graph"]["nodes"] if node["kind"] == "model")
    graph_model = {**graph_model, "id": request["node_id"],
                   "metadata": {**(graph_model.get("metadata") or {}), "source_index": 0}}
    model_task = copy.deepcopy(task)
    model_task["node_plan"] = {**plan["node_plans"][next(key for key, node in plan["node_plans"].items() if node["kind"] == "model")],
                               "node_id": request["node_id"]}
    observed_fit: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
    observed_predict: list[tuple[np.ndarray, dict[str, Any]]] = []
    original_fit, original_predict = PLSRegression.fit, PLSRegression.predict

    def spy_fit(self: PLSRegression, X: Any, y: Any, **options: Any) -> PLSRegression:
        observed_fit.append((np.asarray(X).copy(), np.asarray(y).copy(), dict(options)))
        return original_fit(self, X, y, **options)

    def spy_predict(self: PLSRegression, X: Any, **options: Any) -> Any:
        observed_predict.append((np.asarray(X).copy(), dict(options)))
        return original_predict(self, X, **options)

    with monkeypatch.context() as patcher:
        patcher.setattr(PLSRegression, "fit", spy_fit)
        patcher.setattr(PLSRegression, "predict", spy_predict)
        model_result = run_node(model_task, MaterializationResolver(wrapped, mint_identity(wrapped)),
                                lambda _node_id: graph_model, {}, generated_views=store.bind_task(model_task))
    consumed = model_result["consumed_data_views"]
    assert set(consumed) == {"data:x", "data:x:validation"}
    assert consumed["data:x"]["receipt"] == receipt
    assert consumed["data:x:validation"]["receipt"] == validation_receipt
    assert selected in consumed["data:x"]["read_batches"]
    assert validation_ids in consumed["data:x:validation"]["read_batches"]
    train_calls = consumed["data:x"]["model_calls"]
    assert train_calls[0]["operation"] == "fit"
    assert train_calls[0]["sample_ids"] == selected
    assert len(observed_fit) == 1
    fit_x, fit_y, fit_options = observed_fit[0]
    assert train_calls[0]["input_fingerprint"] == _model_value_fingerprint({"features": fit_x, "options": fit_options})
    assert train_calls[0]["target_fingerprint"] == _model_value_fingerprint(fit_y)
    assert any(call["operation"] == "predict" for call in train_calls)
    validation_calls = [call for call in consumed["data:x:validation"]["model_calls"]
                        if call["operation"] == "predict" and call["sample_ids"] == validation_ids]
    assert validation_calls
    assert any(validation_calls[0]["input_fingerprint"] ==
               _model_value_fingerprint({"features": features, "options": options})
               for features, options in observed_predict)
    validation_prediction = next(block for block in model_result["predictions"] if block["partition"] == "validation")
    direct = PLSRegression(n_components=1)
    direct.fit(np.asarray(view.sources["nir"].values), np.asarray(base.take(selected).y))
    validation_values = np.asarray(store.resolve(validation_handle, validation_ids).sources["nir"].values)
    expected_prediction = np.asarray(direct.predict(validation_values))
    np.testing.assert_allclose(validation_prediction["values"], expected_prediction.reshape(len(validation_ids), -1), atol=1e-9)
    assert "train_pool" not in {block["partition"] for block in model_result["predictions"]}
    with pytest.raises(ValueError, match="without bound IO views"):
        run_node(model_task, MaterializationResolver(wrapped, mint_identity(wrapped)),
                 lambda _node_id: graph_model, {})

    train_row = task_views.take("x", "fold_train", [selected[1]])
    validation_row = task_views.take("x", "fold_validation", [validation_ids[0]])
    train_features = task_views.feature_blocks("x", "fold_train", [selected[1]], source_names=("nir",))
    assert train_features["source_names"] == ("nir",)
    np.testing.assert_array_equal(train_features["blocks"][0], train_row.sources["nir"].values)
    assert train_features["observation_ids"] == [selected[1]]
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
    with pytest.raises(ValueError, match="within its native view"):
        task_views.feature_blocks("x", "fold_train", [validation_ids[0]])
    task_views.take("x", "fold_train", list(reversed(selected)))
    with pytest.raises(ValueError, match="not read in order"):
        task_views.record_model_call("fit", "x", "fold_train", selected,
                                     train_features["blocks"][0], targets=np.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="features has 2 rows for 1"):
        task_views.record_model_call("fit", "x", "fold_train", [selected[1]],
                                     np.repeat(train_features["blocks"][0], 2, axis=0), targets=np.asarray([1.0]))
    with pytest.raises(ValueError, match="targets has 2 rows for 1"):
        task_views.record_model_call("fit", "x", "fold_train", [selected[1]],
                                     train_features["blocks"][0], targets=np.asarray([1.0, 2.0]))
    task_views.record_model_call("fit", "x", "fold_train", [selected[1]],
                                 train_features["blocks"][0], targets=np.asarray([1.0]))
    calls = task_views.consumed_data_views()["data:x"]["model_calls"]
    assert calls[0]["sample_ids"] == [selected[1]]
    assert calls[0]["target_fingerprint"] != calls[0]["input_fingerprint"]
    assert task_views.consumed_data_views() == {
        "data:x": {"receipt": receipt, "read_batches": [[selected[1]], [selected[1]], list(reversed(selected))],
                   "model_calls": calls},
        "data:x:validation": {"receipt": validation_receipt, "read_batches": [[validation_ids[0]]]},
    }
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


def test_generated_model_input_fingerprint_keeps_ragged_boundaries() -> None:
    values = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
    first = RaggedSeriesBatch(values, np.asarray([0, 1, 3], dtype=np.int64))
    second = RaggedSeriesBatch(values, np.asarray([0, 2, 3], dtype=np.int64))
    assert _model_value_fingerprint(first) != _model_value_fingerprint(second)
    assert _model_value_fingerprint(np.asarray(1.0)) != _model_value_fingerprint(np.asarray([1.0]))
    mixed = np.asarray([[1.0, "a"], [2.0, "b"]], dtype=object)
    changed = np.asarray([[1.0, "a"], [2.0, "c"]], dtype=object)
    assert _model_value_fingerprint(mixed) != _model_value_fingerprint(changed)
    with pytest.raises(TypeError, match="Unsupported generated model input type"):
        _model_value_fingerprint(np.asarray([[object()]], dtype=object))


def test_native_cv_requests_scheduler_selected_views_before_model_controller() -> None:
    """The live scheduler supplies selected fold views before the model callback."""
    import dag_ml._dag_ml as native
    from sklearn.cross_decomposition import PLSRegression

    from nirs4all.pipeline.dagml.cli_runner import assemble_cv_refit_dsl, controller_manifests

    if "view_callback" not in inspect.signature(native.run_cv_refit_in_process).parameters:
        pytest.skip("local DAG-ML binding does not expose scheduler-selected view callbacks")
    with pytest.raises(native.DagMlRuntimeError, match="view_callback must be callable"):
        native.run_cv_refit_in_process("{}", "{}", "[]", lambda _task: {}, "rmse", None, False, 1, 42)
    base = _cohort()
    wrapped = MultimodalSpectroDataset(base)
    identity = mint_identity(wrapped)
    envelope = build_envelope(wrapped, identity, sample_ints=list(range(12)))
    folds = [
        (list(range(6, 12)), list(range(6))),
        (list(range(6)), list(range(6, 12))),
    ]
    dsl = assemble_cv_refit_dsl([{"model": PLSRegression(n_components=1)}], identity, envelope,
                                folds, dsl_id="provider-live-folds", n_splits=2)
    calls: list[dict[str, Any]] = []
    operator_calls: list[dict[str, Any]] = []

    def view_callback(call: dict[str, Any]) -> dict[str, Any]:
        calls.append(call)
        request = call["request"]
        return {
            "handle": call["handle"], "view_key": request["view_key"],
            "sample_ids": request["view"]["sample_ids"],
            "schema_fingerprint": "a" * 64, "content_fingerprint": "b" * 64,
        }

    def stop_before_fit(task: dict[str, Any]) -> None:
        operator_calls.append(task)
        raise ValueError("model callback probe stopped before fit")

    with pytest.raises(native.DagMlRuntimeError, match="model callback probe stopped before fit"):
        native.run_cv_refit_in_process(
            json.dumps(dsl), json.dumps(envelope), json.dumps(controller_manifests()),
            stop_before_fit, "rmse", None, False, 1, view_callback,
        )
    assert len(operator_calls) == 1
    assert calls
    selected = set(base.sample_ids[:12])
    held_out = set(base.sample_ids[12:])
    first_fold = dsl["split_invocation"]["fold_set"]["folds"][0]
    for call in calls:
        request = call["request"]
        assert request["phase"] == "FIT_CV"
        assert request["view"]["fold_id"] == first_fold["fold_id"]
        assert request["view"]["partition"] in {"fold_train", "fold_validation", "predict"}
        if request["view"]["partition"] in {"fold_train", "fold_validation"}:
            fold_field = "train_sample_ids" if request["view"]["partition"] == "fold_train" else "validation_sample_ids"
            assert request["view"]["sample_ids"] == first_fold[fold_field]
        permitted = held_out if request["view"]["partition"] == "predict" else selected
        assert set(request["view"]["sample_ids"]).issubset(permitted)
    assert {call["request"]["view"]["partition"] for call in calls} >= {"fold_train", "fold_validation"}
