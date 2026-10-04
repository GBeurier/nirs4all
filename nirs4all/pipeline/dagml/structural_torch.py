"""Closed declarations for native structural early/learned-late Torch HPO.

Only declarations, immutable input buffers and scope identity cross this bridge.
DAG-ML owns proposals, grouped nested OOF, REFIT, scoring and selection. The
existing real Torch estimator and Ridge host owners perform numerical fitting.
"""
from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.operators.models.multimodal import MultimodalRegressor
from nirs4all.pipeline.dagml_bridge import controller_manifests

from .cli_runner import data_bindings_for_nodes, split_invocation_for
from .envelope import build_envelope
from .folds import _build_folds, _split_group_grain
from .identity import mint_identity
from .resources import current_execution_resources
from .torch_estimator import DagMLTorchEstimator
from .tuning_contracts import DagMLTuningSpec, tcv1_sha256

FACTORY = "nirs4all.operators.models.pytorch.mlp.structural_mlp"
ESTIMATOR = "nirs4all.pipeline.dagml.torch_estimator.DagMLTorchEstimator"
RAW_CONTROLLER = "controller:nirs4all.model"
META_CONTROLLER = "controller:nirs4all.meta_model"
INNER_SPLITS = 2
MAX_CELLS = 16_777_216
MAX_PARAMETERS = 1_000_000
MAX_WORK = 100_000_000


def _finite_float32(values: np.ndarray) -> bool:
    """Check the actual Torch conversion without rejecting finite rounding."""
    with np.errstate(over="ignore", invalid="ignore"):
        return bool(np.isfinite(np.asarray(values, dtype=np.float32)).all())


def is_torch_choice(steps: list[Any]) -> bool:
    """Recognize Torch declarations without instantiating any network."""
    def contains(value: Any) -> bool:
        if isinstance(value, MultimodalRegressor):
            return isinstance(value.model, DagMLTorchEstimator)
        if isinstance(value, Mapping):
            return any(contains(item) for item in value.values())
        return isinstance(value, (list, tuple)) and any(contains(item) for item in value)
    return len(steps) == 1 and isinstance(steps[0], dict) and "_or_" in steps[0] and contains(steps[0]["_or_"])


def _model_step(step: Any) -> Any:
    if not isinstance(step, dict) or set(step) != {"model"}:
        raise ValueError("Torch topology steps require exactly {'model': estimator}")
    return step["model"]


def torch_declaration(model: Any) -> tuple[list[str], dict[str, Any]]:
    """Close the public wrapper/factory/control surface before native callbacks."""
    if (type(model) is not MultimodalRegressor or model.backend != "sklearn" or model.fusion != "early"
            or model.missing_source_policy != "error" or model.source_weights is not None
            or not isinstance(model.transformers, Mapping) or not 1 <= len(model.transformers) <= 4
            or any(not isinstance(name, str) or not name for name in model.transformers)
            or any(transformer is not None and transformer != "passthrough" for transformer in model.transformers.values())
            or type(model.model) is not DagMLTorchEstimator):
        raise ValueError("Torch structural models require MultimodalRegressor with named passthrough sources and DagMLTorchEstimator")
    params = copy.deepcopy(model.model.get_params(deep=False))
    expected = {"factory_path", "template_blob", "factory_params", "force_layout", "task_type", "num_classes",
                "epochs", "batch_size", "patience", "optimizer", "lr", "learning_rate", "loss", "device"}
    if set(params) != expected or params["factory_path"] != FACTORY or params["template_blob"] is not None:
        raise ValueError("Torch structural profile admits only the public structural_mlp factory, never arbitrary templates")
    if (params["force_layout"] != "2d" or params["device"] != "cpu" or params["task_type"] != "regression"
            or params["num_classes"] is not None or params["learning_rate"] is not None
            or params["optimizer"] != "Adam" or params["loss"] != "MSELoss"):
        raise ValueError("Torch structural profile requires CPU flat mono-y regression with Adam/MSELoss and explicit lr")
    factory = params["factory_params"]
    if not isinstance(factory, dict) or set(factory) != {"hidden_units"} or type(factory["hidden_units"]) is not int or not 1 <= factory["hidden_units"] <= 128:
        raise ValueError("structural_mlp requires exactly hidden_units in [1,128]")
    for key, high in (("epochs", 100), ("batch_size", 1024), ("patience", 100)):
        if type(params[key]) is not int or not 1 <= params[key] <= high:
            raise ValueError(f"Torch structural {key} must be an integer in [1,{high}]")
    _bounded_number(params["lr"], 1e-6, 0.1, "lr")
    return list(model.transformers), params


def _bounded_number(value: Any, low: float, high: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Torch structural {name} must be finite in [{low},{high}]")


def _meta_params(model: Any) -> dict[str, Any]:
    if (type(model) is not Ridge or model.fit_intercept is not True or model.copy_X is not True
            or model.positive is not False or model.solver != "svd" or model.max_iter is not None
            or model.random_state is not None or model.tol != 1e-4):
        raise ValueError("Torch late fusion requires ordinary deterministic Ridge(alpha=...,fit_intercept=True)")
    _bounded_number(model.alpha, 0, 1e6, "late.meta.alpha")
    return copy.deepcopy(model.get_params(deep=False))


def declared_torch_topologies(steps: list[Any], splitter: Any) -> list[list[Any]]:
    """Validate the existing sequence _or_/branch/merge grammar once."""
    if type(splitter) is not GroupKFold or splitter.n_splits != 3 or getattr(splitter, "shuffle", False):
        raise ValueError("Torch structural HPO requires deterministic GroupKFold(3)")
    if len(steps) != 1 or not isinstance(steps[0], dict) or set(steps[0]) != {"_or_"}:
        raise ValueError("Torch structural HPO requires one _or_ of explicit pipeline sequences")
    alternatives = steps[0]["_or_"]
    if not isinstance(alternatives, list) or not alternatives:
        raise ValueError("Torch topology alternatives must be nonempty")
    for sequence in alternatives:
        if not isinstance(sequence, list):
            raise ValueError("Torch alternatives must be explicit pipeline sequences")
        if len(sequence) == 1:
            torch_declaration(_model_step(sequence[0]))
            continue
        if (len(sequence) != 3 or not isinstance(sequence[0], dict) or set(sequence[0]) != {"branch"}
                or sequence[1] != {"merge": "predictions"}):
            raise ValueError("late Torch requires named branches, merge='predictions', and Ridge")
        branches = sequence[0]["branch"]
        if not isinstance(branches, dict) or not 2 <= len(branches) <= 4:
            raise ValueError("late Torch requires two to four distinct named branches")
        for name, branch in branches.items():
            if not isinstance(branch, list) or len(branch) != 1 or torch_declaration(_model_step(branch[0]))[0] != [name]:
                raise ValueError("each late Torch predictor must select exactly its named source")
        _meta_params(_model_step(sequence[2]))
    return alternatives


def lower_torch_choices(steps: list[Any], splitter: Any) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Transport unique logical model IDs and numeric conditional destinations."""
    alternatives = declared_torch_topologies(steps, splitter)
    choices, sinks = [], []
    bindings: dict[str, list[dict[str, str]]] = {}
    def raw(model: Any, node_id: str, public_path: str) -> dict[str, Any]:
        selection, params = torch_declaration(model)
        bindings.setdefault(public_path, []).append({"node_id": node_id, "param_path": "lr"})
        return {"kind": "model", "id": node_id, "operator": {"class": ESTIMATOR}, "params": params,
                "metadata": {"controller_id": RAW_CONTROLLER, "source_selection": selection}}
    for index, sequence in enumerate(alternatives):
        if len(sequence) == 1:
            node = raw(_model_step(sequence[0]), f"a{index}:early", "early.lr")
            native_steps, sink = [node], node["id"]
        else:
            branches, sources = [], []
            for name, branch in sequence[0]["branch"].items():
                node = raw(_model_step(branch[0]), f"a{index}:source:{name}", f"late.{name}.lr")
                branches.append({"id": name, "steps": [node]})
                sources.append(node["id"])
            sink = f"a{index}:meta"
            bindings.setdefault("late.meta.alpha", []).append({"node_id": sink, "param_path": "alpha"})
            native_steps = [{"kind": "branch", "mode": "duplication", "branches": branches}, {
                "kind": "merge_model", "id": sink, "operator": {"class": "sklearn.linear_model._ridge.Ridge"},
                "params": _meta_params(_model_step(sequence[2])), "sources": sources,
                "merge_mode": "predictions", "include_original_data": False,
                "metadata": {"controller_id": META_CONTROLLER, "stacking_oof_execution": "nested_oof_v1",
                             "stacking_refit_oof": "partitioned_inner_v1"},
                "inner_cv": {"kind": "group_kfold", "n_splits": INNER_SPLITS},
            }]
        choices.append({"id": f"topology{index}", "steps": native_steps})
        sinks.append(sink)
    return {"id": "nirs4all-torch-early-late-hpo", "pipeline": [{"kind": "generator", "id": "generator:torch-topologies",
            "mode": "cartesian", "stages": [{"id": "topology", "branches": choices}]}]}, bindings, sinks


def validate_torch_space(spec: DagMLTuningSpec, bindings: Mapping[str, Any]) -> None:
    from .tuning_adapters import _categorical_choices, _categorical_codec_for_spec
    if set(spec.space) != set(bindings):
        raise ValueError(f"Torch structural HPO requires exactly declared conditional axes: {sorted(bindings)}")
    for path, declaration in spec.space.items():
        codec, choices = _categorical_codec_for_spec(declaration), _categorical_choices(declaration)
        if choices is not None:
            values = [codec.decode(value) for value in codec.choices] if codec is not None else choices
        elif isinstance(declaration, tuple):
            values = list(declaration[-2:])
        elif isinstance(declaration, dict):
            values = [declaration.get("low", declaration.get("min")), declaration.get("high", declaration.get("max"))]
            if declaration.get("step") is not None:
                _bounded_number(declaration["step"], 0, 1e6 if path.endswith("alpha") else 0.1, "step")
        else:
            raise ValueError("unsupported Torch conditional numeric space")
        if not values:
            raise ValueError("Torch conditional numeric choices must be nonempty")
        for value in values:
            _bounded_number(value, 0 if path.endswith("alpha") else 1e-6, 1e6 if path.endswith("alpha") else 0.1, path)


def prepare_torch_structure(steps: list[Any], splitter: Any, dataset: Any, spec: DagMLTuningSpec,
                            run_options: dict[str, Any], native: Any) -> dict[str, Any]:
    """Attest all raw inputs before native catalogue/model/proposal callbacks."""
    if (not isinstance(dataset, MultimodalSpectroDataset) or not dataset.is_regression
            or len(dataset.cohort.sources) != 4 or getattr(dataset, "_generated_view_store", None) is not None):
        raise ValueError("Torch structural HPO requires four complete named dense raw sources and mono-y regression")
    if spec.n_jobs != 1 or spec.pruner not in {None, "none"} or spec.sampler != "random":
        raise ValueError("Torch structural HPO is serial with sampler='random' and no pruning")
    resources = current_execution_resources()
    if resources.cpu_threads != 1 or resources.gpu_devices:
        raise ValueError("Torch structural HPO requires cpu_threads=1 and no GPU devices")
    cohort = dataset.cohort
    names = list(cohort.sources)
    target_names = list(cohort.target_names)
    if len(target_names) != 1 or not isinstance(target_names[0], str) or not target_names[0].strip():
        raise ValueError("Torch structural HPO requires one nonempty signed scalar target name")
    if (cohort.y is None or np.asarray(cohort.y).ndim not in (1, 2)
            or (np.asarray(cohort.y).ndim == 2 and np.asarray(cohort.y).shape[1] != 1)
            or not np.isfinite(np.asarray(cohort.y)).all()
            or not _finite_float32(np.asarray(cohort.y))
            or not np.asarray(cohort.target_mask).all() or set(cohort.partitions) - {"train", "test"}):
        raise ValueError("Torch structural HPO requires complete scalar targets finite in float32 and Train/Test cohorts")
    schemas, widths = {}, {}
    for name, descriptor in zip(names, cohort.schema_descriptors(), strict=True):
        values = np.asarray(cohort.sources[name].values)
        shape = descriptor["shape"]
        native_representation = descriptor["native_representation"]
        if (len(shape) != 2 or shape[0] is not None or type(shape[1]) is not int or shape[1] < 1
                or values.shape != (len(cohort.sample_ids), shape[1]) or values.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or native_representation["ragged"] or native_representation["sparse"]
                or not np.asarray(cohort.sources[name].presence_mask).all() or not np.isfinite(values).all()
                or not _finite_float32(values)
                or values.size > MAX_CELLS or descriptor["source_id"] != name):
            raise ValueError("Torch raw sources must be complete dense float32/64 two-dimensional blocks finite in float32 within the cell budget")
        identity_text = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        if len(identity_text.encode("utf-8")) > 1_048_576:
            raise ValueError("Torch raw source schema exceeds the identity byte budget")
        widths[name] = shape[1]
        schemas[name] = {"representation_id": descriptor["representation_id"], "input_shape": [shape[1]],
                         "dtype": descriptor["dtype"], "identity": identity_text}
    alternatives = declared_torch_topologies(steps, splitter)
    pool = dataset.index_column("sample", {"partition": "train"})
    for sequence in alternatives:
        raw_models = [_model_step(sequence[0])] if len(sequence) == 1 else [_model_step(branch[0]) for branch in sequence[0]["branch"].values()]
        for model in raw_models:
            selection, params = torch_declaration(model)
            if set(selection) - set(names):
                raise ValueError("Torch source selection names must belong to the four declared raw inputs")
            width = sum(widths[name] for name in selection)
            hidden = params["factory_params"]["hidden_units"]
            if ((width + 1) * hidden + hidden + 1 > MAX_PARAMETERS
                    or len(pool) * params["epochs"] * hidden * (width + 1) > MAX_WORK
                    or len(cohort.sample_ids) * width > MAX_CELLS):
                raise ValueError("Torch selected architecture/training request exceeds the closed CPU parameter/work/cell budgets")
    identity = mint_identity(dataset)
    if any(sample.augmented for sample in identity.identities):
        raise ValueError("Torch structural HPO refuses augmented sample views")
    folds = _build_folds(splitter, dataset, pool, set())
    groups = _split_group_grain(splitter, dataset, pool)
    if groups is None or len(folds) != 3 or any(not train or not validation for train, validation in folds):
        raise ValueError("Torch structural HPO requires three complete grouped outer folds")
    envelope = build_envelope(dataset, identity, sample_ints=pool, group_by_sample=groups)
    dsl, bindings, sinks = lower_torch_choices(steps, splitter)
    validate_torch_space(spec, bindings)
    seed = run_options.get("random_state")
    seed = (spec.seed if spec.seed is not None else 0) if seed is None else seed
    if type(seed) is not int or not 0 <= seed <= (1 << 64) - 1:
        raise ValueError("Torch structural seed must be an unsigned 64-bit integer")
    # Existing single-source owner materializes the exact global source index.
    # Multi-source early nodes retain their declared ordered concatenation.
    def stamp_source_indexes(value: Any) -> None:
        if isinstance(value, dict):
            metadata = value.get("metadata")
            if isinstance(metadata, dict) and metadata.get("controller_id") == RAW_CONTROLLER:
                selection = metadata["source_selection"]
                if len(selection) == 1:
                    metadata["source_index"] = names.index(selection[0])
            for child in value.values():
                stamp_source_indexes(child)
        elif isinstance(value, list):
            for child in value:
                stamp_source_indexes(child)

    stamp_source_indexes(dsl)
    dsl["root_seed"] = seed
    dsl["metadata"] = {"python_torch_profile": {"schema_version": 1, "profile": "cpu_serial_regression_v1",
        "source_order": names, "source_widths": widths, "seed": seed, "cpu_threads": 1, "gpu_devices": [],
        "target_names": target_names,
        "training_policy": {"validation": "none", "shuffle": True, "early_stopping": False}}, "source_schemas": schemas}
    dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=3, shuffle=False)
    dsl["split_invocation"]["fold_set"]["sample_groups"] = {identity.to_wire(sample): group for sample, group in groups.items()}
    manifests = controller_manifests()
    graph = native.compile_pipeline_dsl_artifact_with_controllers(dsl, manifests).graph.to_dict()
    raw_ids = [node["id"] for node in graph["nodes"] if (node.get("metadata") or {}).get("controller_id") == RAW_CONTROLLER]
    dsl["data_bindings"] = data_bindings_for_nodes(raw_ids, envelope)
    for binding in dsl["data_bindings"]:
        binding["view_policy"] = {"include_augmented_train": False, "include_refit_test_view": True}
    catalogue = native.prepare_host_hpo_topology_catalogue(dsl, envelope, manifests, bindings, sinks)
    descriptor = spec.to_dict()
    # The generic contract omits sequential n_jobs for historical checkpoint
    # identity; this closed native Torch profile requires an explicit CPU job.
    descriptor["n_jobs"] = spec.n_jobs
    for key in ("resume", "n_trials", "storage", "study_name"):
        descriptor.pop(key, None)
    descriptor["operator_rng"] = {"policy": "per_native_task_v1", "seed": seed}
    descriptor["training_content_fingerprint"] = tcv1_sha256({"raw_source_content": dataset.content_hash(),
        "targets": np.asarray(cohort.y).tolist(), "groups": [{"sample_id": identity.to_wire(sample), "group": groups[sample]} for sample in pool]})
    request = {"target_node": catalogue["entries"][0]["target_node"], "trial_budget": spec.n_trials,
        "metric": spec.metric, "direction": spec.direction, "optimizer_descriptor": descriptor,
        "fold_score_reduction": "mean", "structural_catalogue": catalogue}
    return {"spec": spec, "dataset": dataset, "identity": identity, "folds": folds, "envelope": envelope,
        "dsl": dsl, "manifests": manifests, "graph": graph, "catalogue": catalogue, "request": request,
        "operator_seed": seed, "splitter": splitter, "steps": steps, "pool": pool, "groups": groups, "torch_topology": True}
