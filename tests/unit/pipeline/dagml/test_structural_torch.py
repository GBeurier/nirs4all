"""Closed Torch topology declarations and refusals before native/model callbacks."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset, TensorSource

from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component
from nirs4all.pipeline.dagml.structural_torch import FACTORY, declared_torch_topologies, lower_torch_choices, torch_declaration
from nirs4all.pipeline.dagml.structural_tuning import _prepare_structure, validate_structural_profile

_PATH = Path(__file__).resolve().parents[4] / "examples/user/04_models/U24_structural_hpo_torch.py"
_SPEC = importlib.util.spec_from_file_location("torch_structural_unit_example", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


def _replace(cohort: Any, name: str, values: Any, *, presence: Any = None) -> Any:
    old = cohort.sources[name]
    sources = dict(cohort.sources)
    sources[name] = TensorSource(values, cohort.sample_ids, representation_id=old.representation_id, axes=old.axes,
        feature_names=[f"{name}:{i}" for i in range(np.shape(values)[1])],
        presence_mask=old.presence_mask if presence is None else presence)
    return MultimodalDataset(sources, sample_ids=cohort.sample_ids, y=cohort.y, groups=cohort.groups,
                            target_names=cohort.target_names, partitions=cohort.partitions, task_type="regression")


def test_public_roundtrip_preserves_selected_source_order_and_conditional_ids() -> None:
    pipeline = deserialize_component(json.loads(json.dumps(serialize_component(example.make_pipeline()))))
    steps, splitter = validate_structural_profile(pipeline)
    dsl, bindings, sinks = lower_torch_choices(steps, splitter)
    alternatives = dsl["pipeline"][0]["stages"][0]["branches"]
    ids = []
    for index, alternative in enumerate(alternatives):
        if index == 0:
            raw = alternative["steps"]
            assert raw[0]["metadata"]["source_selection"] == ["image", "nir"]
        else:
            branches, meta = alternative["steps"]
            order = list(pipeline[1]["_or_"][index][0]["branch"])
            assert [child["id"] for child in branches["branches"]] == order
            raw = [child["steps"][0] for child in branches["branches"]]
            assert [node["metadata"]["source_selection"] for node in raw] == [[name] for name in order]
            assert meta["sources"] == [node["id"] for node in raw]
            assert meta["params"]["solver"] == "svd"
            assert meta["metadata"]["stacking_oof_execution"] == "nested_oof_v1"
            assert meta["metadata"]["stacking_refit_oof"] == "partitioned_inner_v1"
            assert meta["inner_cv"] == {"kind": "group_kfold", "n_splits": 2}
            assert sinks[index] == meta["id"]
            ids.append(meta["id"])
        for node in raw:
            ids.append(node["id"])
            assert node["params"]["factory_path"] == FACTORY
            assert node["params"]["device"] == "cpu" and node["params"]["force_layout"] == "2d"
    assert len(ids) == len(set(ids)) == 16
    assert set(bindings) == {"early.lr", "late.nir.lr", "late.image.lr", "late.series.lr", "late.metadata.lr", "late.meta.alpha"}
    assert json.loads(json.dumps(dsl, sort_keys=True)) == dsl


@pytest.mark.parametrize("kind", ["factory", "template", "device", "layout", "classification", "optimizer", "loss", "hidden", "epochs", "batch", "patience", "lr", "learning_rate", "encoder", "weights"])
def test_closed_factory_controls_refuse_before_a_model_is_built(kind: str) -> None:
    model = example.make_model(("image", "nir"))
    changes = {"factory": {"factory_path": "torch.nn.Linear"}, "template": {"template_blob": "opaque"}, "device": {"device": "cuda:0"},
               "layout": {"force_layout": "3d"}, "classification": {"task_type": "classification"}, "optimizer": {"optimizer": "SGD"},
               "loss": {"loss": "L1Loss"}, "hidden": {"factory_params": {"hidden_units": 129}}, "epochs": {"epochs": 101},
               "batch": {"batch_size": 0}, "patience": {"patience": True}, "lr": {"lr": float("inf")}, "learning_rate": {"learning_rate": 0.01}}
    if kind == "encoder":
        from sklearn.preprocessing import StandardScaler
        model.transformers = {"image": StandardScaler(), "nir": None}
    elif kind == "weights":
        model.source_weights = {"image": 0.5}
    else:
        model.model.set_params(**changes[kind])
    with pytest.raises((ValueError, TypeError)):
        torch_declaration(model)


@pytest.mark.parametrize("kind", ["one_branch", "wrong_source", "features", "fit_controls", "bad_meta", "shuffle"])
def test_invalid_public_sequence_grammar_refuses(kind: str) -> None:
    pipeline = example.make_pipeline()
    sequence = pipeline[1]["_or_"][1]
    if kind == "one_branch":
        sequence[0]["branch"].pop("nir")
    elif kind == "wrong_source":
        sequence[0]["branch"]["nir"][0]["model"] = example.make_model(("image",))
    elif kind == "features":
        sequence[1] = {"merge": "features"}
    elif kind == "fit_controls":
        sequence[0]["branch"]["nir"][0]["train_params"] = {"device": "cuda:0"}
    elif kind == "bad_meta":
        sequence[2]["model"].positive = True
    else:
        from sklearn.model_selection import GroupKFold
        pipeline[0] = GroupKFold(3, shuffle=True, random_state=17)
    with pytest.raises((ValueError, TypeError)):
        declared_torch_topologies(pipeline[1:], pipeline[0])


@pytest.mark.parametrize("kind", ["parallel", "pruner", "sampler", "axis", "lr", "target", "missing", "nan", "source_float32_overflow", "target_float32_overflow", "parameter_cost", "work_cost"])
def test_preflight_refuses_without_native_or_optimizer_callbacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str) -> None:
    import dag_ml

    from nirs4all.pipeline.dagml.host_search_checkpoint import HostSearchOptimizer
    monkeypatch.setattr(dag_ml, "compile_pipeline_dsl_artifact_with_controllers", lambda *a, **k: pytest.fail("native compilation preceded preflight"))
    monkeypatch.setattr(HostSearchOptimizer, "__init__", lambda *a, **k: pytest.fail("optimizer constructed before preflight"))
    pipeline, cohort, tuning = example.make_pipeline(), example.make_dataset(), example.make_tuning(tmp_path / "study")
    if kind == "parallel":
        tuning["n_jobs"] = 2
    elif kind == "pruner":
        tuning["pruner"] = "median"
    elif kind == "sampler":
        tuning["sampler"] = "sobol"
    elif kind == "axis":
        tuning["space"]["early.epochs"] = [1, 2]
    elif kind == "lr":
        tuning["space"]["early.lr"] = [0.001, 1.0]
    elif kind == "target":
        cohort = MultimodalDataset(cohort.sources, sample_ids=cohort.sample_ids, y=np.column_stack((cohort.y, cohort.y)),
            target_names=("a", "b"), groups=cohort.groups, partitions=cohort.partitions, task_type="regression")
    elif kind == "missing":
        mask = np.ones(len(cohort.sample_ids), dtype=bool)
        mask[-1] = False
        cohort = _replace(cohort, "metadata", cohort.sources["metadata"].values, presence=mask)
    elif kind == "nan":
        values = cohort.sources["metadata"].values.copy()
        values[-1, 0] = np.nan
        cohort = _replace(cohort, "metadata", values)
    elif kind == "source_float32_overflow":
        values = cohort.sources["metadata"].values.copy()
        values[-1, 0] = 1e300
        cohort = _replace(cohort, "metadata", values)
    elif kind == "target_float32_overflow":
        targets = np.array(cohort.y, copy=True)
        targets[-1] = 1e300
        cohort = MultimodalDataset(cohort.sources, sample_ids=cohort.sample_ids, y=targets,
            target_names=cohort.target_names, groups=cohort.groups, partitions=cohort.partitions, task_type="regression")
    else:
        cohort = _replace(cohort, "image", np.zeros((len(cohort.sample_ids), 8000), dtype=np.float64))
        raw = pipeline[1]["_or_"][0][0]["model"].model
        raw.factory_params = {"hidden_units": 128 if kind == "parameter_cost" else 64}
        raw.epochs = 1 if kind == "parameter_cost" else 100
    with pytest.raises((ValueError, TypeError)):
        _prepare_structure(pipeline, cohort, tuning, {"random_state": 17, "refit": True})


@pytest.mark.parametrize("target_name", ["y", "concentration"])
def test_transport_signs_all_four_schemas_groups_and_fixed_training_policy(tmp_path: Path, target_name: str) -> None:
    prepared = _prepare_structure(example.make_pipeline(), example.make_dataset(target_name=target_name), example.make_tuning(tmp_path / "study"), {"random_state": 17})
    graph = prepared["graph"]
    profile = graph["metadata"]["python_torch_profile"]
    assert profile == {"schema_version": 1, "profile": "cpu_serial_regression_v1", "source_order": list(example.SOURCE_ORDER),
        "source_widths": {"nir": 4, "image": 3, "series": 2, "metadata": 2}, "seed": 17, "cpu_threads": 1, "gpu_devices": [],
        "target_names": [target_name],
        "training_policy": {"validation": "none", "shuffle": True, "early_stopping": False}}
    assert set(graph["metadata"]["source_schemas"]) == set(example.SOURCE_ORDER)
    for name, schema in graph["metadata"]["source_schemas"].items():
        assert json.loads(schema["identity"])["source_id"] == name
    assert set(prepared["dsl"]["split_invocation"]["fold_set"]["sample_groups"]) == {prepared["identity"].to_wire(row) for row in prepared["pool"]}
    assert all(len(binding["source_ids"]) == 4 for binding in prepared["dsl"]["data_bindings"])
    for node in graph["nodes"]:
        metadata = node.get("metadata") or {}
        if metadata.get("controller_id") == "controller:nirs4all.model":
            selection = metadata["source_selection"]
            if len(selection) == 1:
                assert metadata["source_index"] == list(example.SOURCE_ORDER).index(selection[0])
            else:
                assert "source_index" not in metadata
    assert prepared["catalogue"]["schema_version"] == 2
    descriptor = prepared["request"]["optimizer_descriptor"]
    assert descriptor["n_jobs"] == 1
    assert descriptor["sampler"] == "random"
    assert descriptor["pruner"] in {None, "none"}
    assert "parallel_execution" not in descriptor
    assert "generated_view_mode" not in descriptor
    # Transport admission is explicit without changing historical spec bytes.
    assert "n_jobs" not in prepared["spec"].to_dict()


@pytest.mark.parametrize("kind", ["source", "target"])
def test_float32_boundary_rounding_remains_admitted_before_compilation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str) -> None:
    import dag_ml

    class AdmissionReached(Exception):
        """Sentinel proving host preflight accepted without constructing a model."""

    def compile_reached(*args: Any, **kwargs: Any) -> Any:
        raise AdmissionReached

    monkeypatch.setattr(dag_ml, "compile_pipeline_dsl_artifact_with_controllers", compile_reached)
    cohort = example.make_dataset()
    # One float64 ULP above float32.max still rounds to a finite float32 value.
    # Checking the actual conversion preserves this legitimate boundary.
    boundary = np.nextafter(float(np.finfo(np.float32).max), np.inf)
    if kind == "source":
        values = cohort.sources["metadata"].values.copy()
        values[-1, 0] = boundary
        cohort = _replace(cohort, "metadata", values)
    else:
        targets = np.array(cohort.y, copy=True)
        targets[-1] = boundary
        cohort = MultimodalDataset(cohort.sources, sample_ids=cohort.sample_ids, y=targets,
            target_names=cohort.target_names, groups=cohort.groups, partitions=cohort.partitions, task_type="regression")
    with pytest.raises(AdmissionReached):
        _prepare_structure(example.make_pipeline(), cohort, example.make_tuning(tmp_path / "study"), {"random_state": 17})
