"""Native named-port contracts, source identities and refusal before model FIT."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.model_selection import KFold

from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.operators.models.multimodal import MultimodalRegressor
from nirs4all.pipeline.dagml.cli_runner import assemble_cv_refit_dsl, data_bindings_for
from nirs4all.pipeline.dagml.envelope import build_envelope
from nirs4all.pipeline.dagml.fixed_cohort_views import FixedCohortViewStore
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml.named_torch import named_task_seed, prepare_named_torch_pipeline
from nirs4all.pipeline.dagml.named_torch_estimator import DagMLNamedTorchEstimator
from nirs4all.pipeline.dagml.node_runner import resolve_named_model_features, run_node
from nirs4all.pipeline.dagml.resolver import MaterializationResolver
from nirs4all.pipeline.dagml.steps import _apply_model_params, _split_pipeline
from nirs4all.pipeline.dagml_bridge import compile_with_dagml, controller_manifests, named_model_input_spec, pipeline_to_dsl


def _fixture():
    ids = ["r3", "r1", "r2", "r4"]
    nir = np.asarray([[30, 31, 32], [10, 11, 12], [20, 21, 22], [40, 41, 42]], dtype=np.float64)
    # Physical source storage has a distinct order; IO aligns by real row IDs.
    clinical = np.asarray([[200, 201], [400, 401], [300, 301], [100, 101]], dtype=np.float32)
    cohort = MultimodalDataset({
        "nir": TensorSource(nir, ids, representation_id="tabular_numeric"),
        "clinical": TensorSource(clinical, ["r2", "r4", "r3", "r1"], representation_id="tabular_numeric"),
    }, sample_ids=ids, y=np.asarray([3., 1., 2., 4.]), target_names=["y"], task_type="regression")
    spectro = MultimodalSpectroDataset(cohort)
    adapter = DagMLNamedTorchEstimator(factory_path="user_models.joint_encoder", device="cpu", task_type="regression",
                                      force_layout="2d", epochs=1, batch_size=2, patience=1)
    public = [KFold(2), {"model": MultimodalRegressor({"clinical": None, "nir": "passthrough"}, adapter, fusion="intermediate")}]
    pipeline = prepare_named_torch_pipeline(public, spectro)
    identity = mint_identity(spectro)
    envelope = build_envelope(spectro, identity)
    specification = named_model_input_spec(pipeline)
    assert specification is not None
    return spectro, pipeline, identity, envelope, specification


def test_lowering_preserves_contract_outside_estimator_hyperparameters():
    _, pipeline, _, _, specification = _fixture()
    steps, _ = _split_pipeline(pipeline)
    normalized = _apply_model_params(steps)
    dsl = pipeline_to_dsl(normalized)
    step = dsl["pipeline"][0]
    assert step["model"].endswith(".DagMLNamedTorchEstimator")
    assert step["model_input"] == specification
    assert "model_input" not in step["params"]
    assert step["metadata"]["controller_id"] == normalized[0]["metadata"]["controller_id"]
    assert json.loads(json.dumps(dsl)) == dsl


def test_native_compile_has_real_named_ports_and_exact_contextual_manifest():
    _, pipeline, _, _, specification = _fixture()
    steps, _ = _split_pipeline(pipeline)
    graph = compile_with_dagml(steps).graph.to_dict()
    model = next(node for node in graph["nodes"] if node["kind"] == "model")
    assert [port["name"] for port in model["ports"]["inputs"]] == ["clinical", "nir", "y"]
    assert [port["kind"] for port in model["ports"]["inputs"]] == ["data", "data", "target"]
    assert [port["name"] for port in graph["interface"]["inputs"]] == ["clinical", "nir"]
    assert graph["edges"] == []
    assert model["metadata"]["dsl_model_input"] == specification
    from_pipeline = controller_manifests(pipeline)
    from_graph = controller_manifests(graph)
    assert from_pipeline == from_graph
    manifest = next(item for item in from_graph if item["controller_id"] == model["metadata"]["controller_id"])
    assert manifest["data_requirements"] == specification
    assert manifest["input_ports"] == model["ports"]["inputs"]
    assert "needs_python_gil" in manifest["capabilities"]
    assert not {"thread_safe", "process_safe"}.intersection(manifest["capabilities"])
    generic = next(item for item in from_graph if item["controller_id"] == "controller:nirs4all.model")
    assert {"thread_safe", "process_safe"}.issubset(generic["capabilities"])


def test_complete_cv_dsl_has_distinct_real_source_bindings_without_mutating_global_plan():
    _, pipeline, identity, envelope, specification = _fixture()
    original = copy.deepcopy(envelope)
    steps, _ = _split_pipeline(pipeline)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, [([0, 1], [2, 3]), ([2, 3], [0, 1])], n_splits=2)
    bindings = dsl["data_bindings"]
    assert [item["input_name"] for item in bindings] == ["clinical", "nir"]
    assert [item["source_ids"] for item in bindings] == [["src1"], ["src0"]]
    assert all(item["output_representation"] == "tabular_numeric" for item in bindings)
    assert all(item["view_policy"]["include_refit_test_view"] for item in bindings)
    assert all(item["plan_fingerprint"] == envelope["plan_fingerprint"] for item in bindings)
    assert envelope == original and envelope["plan"]["output_representation"] == "feature_block_set"
    assert named_model_input_spec(dsl) == specification


@pytest.mark.parametrize("change", ["reordered", "duplicate_name", "duplicate_native_id", "missing_name", "length"])
def test_binding_refuses_ambiguous_or_changed_native_source_mapping(change):
    _, _, _, envelope, specification = _fixture()
    layout = envelope["plan"]["source_layout"]
    if change == "reordered":
        layout["source_order"] = list(reversed(layout["source_order"]))
    elif change == "duplicate_name":
        layout["source_order"][1] = layout["source_order"][0]
    elif change == "duplicate_native_id":
        layout["source_ids"][1] = layout["source_ids"][0]
    elif change == "missing_name":
        layout["source_order"][1] = "foreign"
    else:
        layout["source_ids"].pop()
    with pytest.raises(ValueError):
        data_bindings_for("model", envelope, model_input=specification)


def _task_fixture(phase="FIT_CV", resources=None):
    spectro, _, identity, envelope, specification = _fixture()
    ids = [identity.to_wire(1), identity.to_wire(0)]
    bindings = data_bindings_for("model", envelope, model_input=specification)
    resolver = MaterializationResolver(spectro, identity)
    store = FixedCohortViewStore(resolver, specification, envelope)
    fold_id = "fold:0" if phase == "FIT_CV" else None
    partition = {"FIT_CV": "fold_train", "REFIT": "full_train", "PREDICT": "predict"}[phase]
    task = {"run_id": "run:fixed-unit", "phase": phase, "variant_id": "variant:a", "fold_id": fold_id, "seed": 123,
            "resources": resources if resources is not None else {"cpu_threads": 1, "gpu_devices": []},
            "node_plan": {"node_id": "model", "data_bindings": bindings, "kind": "model", "controller_id": "controller:nirs4all.named_torch.test"},
            "data_views": {}, "input_handles": {}, "data_view_receipts": {}}
    for index, binding in enumerate(bindings):
        name = binding["input_name"]
        key = f"data:{name}"
        view = {"partition": partition, "sample_ids": list(ids), "fold_id": fold_id, "source_ids": binding["source_ids"],
                "columns": None, "branch_view": None, "include_augmented": phase != "PREDICT", "include_excluded": False,
                "extra": {"feature_set_id": binding["feature_set_id"]}}
        handle = {"kind": "data_view", "handle": index + 1}
        # Unit callback-protocol fixture. Public integration witnesses use the
        # actual PyO3 scheduler; these receipts still read real IO buffers.
        request = {"run_id": task["run_id"], "node_id": "model", "phase": task["phase"], "fold_id": task["fold_id"],
                   "variant_id": task["variant_id"], "input_name": name, "binding": binding, "view": view,
                   "view_key": f"unit-view:{name}", "view_seed": 1}
        task["data_views"][key] = view
        task["input_handles"][key] = handle
        task["data_view_receipts"][key] = store({"request": request, "handle": handle})
    return task, resolver, specification, ids, store.bind_task(task)


def test_resolved_named_tables_keep_distinct_width_dtype_and_native_row_order():
    task, resolver, specification, ids, views = _task_fixture()
    actual = resolve_named_model_features(task, resolver, specification, ids, "fold_train", views)
    assert tuple(actual) == ("clinical", "nir")
    assert actual["clinical"].shape == (2, 2) and actual["clinical"].dtype == np.float32
    assert actual["nir"].shape == (2, 3) and actual["nir"].dtype == np.float64
    np.testing.assert_array_equal(actual["clinical"], [[100, 101], [300, 301]])
    np.testing.assert_array_equal(actual["nir"], [[10, 11, 12], [30, 31, 32]])
    assert set(views.consumed_data_views()) == {"data:clinical", "data:nir"}
    assert task["data_view_receipts"]["data:clinical"]["content_fingerprint"] != task["data_view_receipts"]["data:nir"]["content_fingerprint"]


@pytest.mark.parametrize("change", ["row_order", "source_swap", "source_concat", "missing_binding", "wrong_partition", "fake_receipt", "width", "dtype"])
def test_named_reads_refuse_false_port_or_identity_contracts_before_fit(change):
    task, resolver, specification, ids, views = _task_fixture()
    if change == "row_order":
        task["data_views"]["data:nir"]["sample_ids"] = list(reversed(ids))
    elif change == "source_swap":
        task["node_plan"]["data_bindings"][0]["source_ids"] = ["src0"]
    elif change == "source_concat":
        task["node_plan"]["data_bindings"][0]["source_ids"] = ["src0", "src1"]
    elif change == "missing_binding":
        task["node_plan"]["data_bindings"].pop()
    elif change == "wrong_partition":
        task["data_views"]["data:nir"]["partition"] = "fold_validation"
    elif change == "fake_receipt":
        task["data_view_receipts"] = {"data:nir": {"content_sha256": "a" * 64}}
    elif change == "width":
        specification["ports"][0]["metadata"]["feature_shape"] = [3]
    else:
        specification["ports"][0]["metadata"]["dtype"] = "float64"
    with pytest.raises(ValueError):
        resolve_named_model_features(task, resolver, specification, ids, "fold_train", views)


@pytest.mark.parametrize("change", ["absent", "null", "content", "handle", "reader"])
def test_named_reads_require_original_provider_receipts(change):
    task, resolver, specification, ids, views = _task_fixture()
    reader = views
    if change == "absent":
        task["data_view_receipts"] = {}
    elif change == "null":
        task["data_view_receipts"]["data:nir"] = None
    elif change == "content":
        task["data_view_receipts"]["data:nir"]["content_fingerprint"] = "0" * 64
    elif change == "handle":
        task["input_handles"]["data:nir"] = task["input_handles"]["data:clinical"]
    else:
        reader = None
    with pytest.raises(ValueError):
        resolve_named_model_features(task, resolver, specification, ids, "fold_train", reader)


def test_fixed_receipt_detects_changed_actual_buffer_before_numerical_use(monkeypatch):
    task, resolver, specification, ids, views = _task_fixture()
    original = resolver.resolve_source_block

    def changed(*args, **kwargs):
        resolved = original(*args, **kwargs)
        resolved["values"] = np.array(resolved["values"], copy=True)
        resolved["values"][0, 0] += 1
        return resolved

    monkeypatch.setattr(resolver, "resolve_source_block", changed)
    with pytest.raises(ValueError, match="actual buffers after receipt"):
        resolve_named_model_features(task, resolver, specification, ids, "fold_train", views)


def test_named_cli_refuses_before_starting_process_or_writing_inputs(monkeypatch, tmp_path):
    from nirs4all.pipeline.dagml import cli_runner

    _, pipeline, _, envelope, _ = _fixture()
    steps, _ = _split_pipeline(pipeline)
    dsl = pipeline_to_dsl(steps)

    def forbidden_process(*args, **kwargs):
        pytest.fail("ordinary CLI reached a named numerical process")

    monkeypatch.setattr(cli_runner.subprocess, "run", forbidden_process)
    with pytest.raises(NotImplementedError, match="attested fixed-cohort"):
        cli_runner.run_cv_refit_bundle(dsl=dsl, envelope=envelope, graph={}, dataset_path="unused", workdir=tmp_path / "cv",
                                      dagml_cli="unused", venv_python="unused")
    with pytest.raises(NotImplementedError, match="attested fixed-cohort"):
        cli_runner.run_refit_phase_cli(dsl=dsl, envelope=envelope, graph={}, dataset_path="unused", workdir=tmp_path / "refit",
                                      training_sample_ids=[], dagml_cli="unused", venv_python="unused")
    assert not (tmp_path / "cv").exists() and not (tmp_path / "refit").exists()


def test_absent_contract_keeps_existing_aggregate_x_binding():
    _, _, _, envelope, _ = _fixture()
    bindings = data_bindings_for("ordinary", envelope)
    assert len(bindings) == 1 and bindings[0]["input_name"] == "x"
    assert bindings[0]["source_ids"] == ["src0", "src1"]
    assert bindings[0]["output_representation"] == "feature_block_set"


@pytest.mark.torch
@pytest.mark.parametrize("phase", ["FIT_CV", "REFIT", "PREDICT"])
@pytest.mark.parametrize("fails", [False, True])
def test_named_dispatch_seeds_actual_callback_and_restores_rng(monkeypatch, phase, fails):
    torch = pytest.importorskip("torch")
    from nirs4all.pipeline.dagml import node_runner

    task, resolver, specification, _, views = _task_fixture(phase)
    node = {"metadata": {"dsl_model_input": specification, "named_torch_profile": "cpu_named_intermediate_regression_v1"}}
    before = torch.random.get_rng_state().clone()
    calls = []

    def callback(*args, **kwargs):
        calls.append(args[0])
        assert torch.random.initial_seed() == named_task_seed(task)
        torch.rand(5)
        if fails:
            raise RuntimeError("callback failure")
        return {"seed": named_task_seed(task)}

    monkeypatch.setattr(node_runner, "_run_node", callback)
    if fails:
        with pytest.raises(RuntimeError, match="callback failure"):
            run_node(task, resolver, lambda _: node, {}, generated_views=views)
    else:
        assert run_node(task, resolver, lambda _: node, {}, generated_views=views) == {"seed": named_task_seed(task)}
    assert calls == [task]
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.torch
@pytest.mark.parametrize("resources", [{"cpu_threads": 2, "gpu_devices": []}, {"cpu_threads": 1, "gpu_devices": ["cuda:0"]}])
def test_named_dispatch_refuses_parallel_or_gpu_before_callback(monkeypatch, resources):
    pytest.importorskip("torch")
    from nirs4all.pipeline.dagml import node_runner

    task, resolver, specification, _, views = _task_fixture(resources=resources)
    node = {"metadata": {"dsl_model_input": specification, "named_torch_profile": "cpu_named_intermediate_regression_v1"}}

    def unexpected_callback(*args, **kwargs):
        pytest.fail("invalid named resources reached the numerical callback")

    monkeypatch.setattr(node_runner, "_run_node", unexpected_callback)
    with pytest.raises(ValueError, match="serial CPU resources"):
        run_node(task, resolver, lambda _: node, {}, generated_views=views)
