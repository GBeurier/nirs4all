"""Serial native classification declarations, grouped scope preflight and capture."""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.operators.models.multimodal import MultimodalClassifier

from .cli_runner import data_bindings_for_nodes, split_invocation_for
from .envelope import build_envelope, source_ids
from .folds import _build_folds, _split_group_grain
from .identity import mint_identity
from .methods_classification import (
    HEAD,
    META_CONTROLLER,
    META_TYPE,
    RAW_CONTROLLER,
    RAW_TYPE,
    classification_vocabulary,
    classifier_head,
    classifier_recipe,
    encode_labels,
)
from .methods_multimodal import (
    SOURCE_ORDER,
    controller_sources,
    execute_methods_training,
    source_schemas_from_cohort,
)
from .tuning_contracts import DagMLTuningSpec, parse_tuning_spec, tcv1_sha256

INNER_SPLITS = 2


def is_classification_choice(steps: list[Any]) -> bool:
    """Recognize explicit native classifiers without consulting fitted state."""
    def contains(value: Any) -> bool:
        if isinstance(value, MultimodalClassifier):
            return getattr(value, "backend", "sklearn") == "methods"
        if isinstance(value, Mapping):
            return any(contains(child) for child in value.values())
        return isinstance(value, (list, tuple)) and any(contains(child) for child in value)
    return len(steps) == 1 and isinstance(steps[0], dict) and isinstance(steps[0].get("_or_"), list) and contains(steps[0])


def _model_step(step: Any) -> Any:
    if not isinstance(step, dict) or set(step) != {"model"}:
        raise ValueError("topology model steps require exactly {'model': estimator}")
    return step["model"]


def _meta_operator(model: Any, order: list[str], vocabulary: Mapping[str, Any]) -> dict[str, Any]:
    head = classifier_head(model)
    return {"type": META_TYPE, "steps": [{"methodId": HEAD, "params": head["params"]}],
            "source_order": list(order), "classification": copy.deepcopy(dict(vocabulary))}


def declared_classifier_topologies(steps: list[Any], splitter: Any) -> list[list[Any]]:
    """Validate public grammar without native state or recipe enumeration."""
    if type(splitter) is not GroupKFold or splitter.n_splits != 3 or getattr(splitter, "shuffle", False):
        raise ValueError("early/late structural tuning requires deterministic GroupKFold(3)")
    if len(steps) != 1 or not isinstance(steps[0], dict) or set(steps[0]) != {"_or_"}:
        raise ValueError("early/late structural tuning requires one _or_ of pipeline sequences")
    alternatives = steps[0]["_or_"]
    if not isinstance(alternatives, list) or not alternatives:
        raise ValueError("topology alternatives must be a nonempty list")
    for sequence in alternatives:
        if not isinstance(sequence, list):
            raise ValueError("each topology alternative must be an explicit pipeline sequence")
        if len(sequence) == 1:
            classifier_recipe(_model_step(sequence[0]), allow_source_selection=True)
            continue
        if (len(sequence) != 3 or not isinstance(sequence[0], dict) or set(sequence[0]) != {"branch"}
                or sequence[1] != {"merge": "predictions"}):
            raise ValueError("late topology requires named branches, merge='predictions', and one native PLS-logistic meta-model")
        branches = sequence[0]["branch"]
        if not isinstance(branches, dict) or not 2 <= len(branches) <= 4 or set(branches) - set(SOURCE_ORDER):
            raise ValueError("late fusion requires two to four distinct named raw-source branches")
        for name, branch in branches.items():
            if not isinstance(branch, list) or len(branch) != 1:
                raise ValueError("each late branch requires one complete Methods multimodal predictor")
            recipe = classifier_recipe(_model_step(branch[0]), allow_source_selection=True)
            if recipe["source_order"] != [name]:
                raise ValueError("late branch predictor must select exactly its named source")
        classifier_head(_model_step(sequence[2]))
    return alternatives


def _typed_node(model: Any, node_id: str, schemas: Mapping[str, Any], vocabulary: Mapping[str, Any]) -> dict[str, Any]:
    recipe = classifier_recipe(model, allow_source_selection=True)
    return {"kind": "model", "id": node_id, "prediction_output_ports": ["y_hat", "probabilities"],
            "operator": {"type": RAW_TYPE, "recipe": recipe, "source_schemas": copy.deepcopy(dict(schemas)), "classification": copy.deepcopy(dict(vocabulary))},
            "params": {"recipe": copy.deepcopy(recipe), "source_schemas": copy.deepcopy(dict(schemas)), "classification": copy.deepcopy(dict(vocabulary)),
                       "model__n_components": recipe["model"]["params"]["n_components"], "model__max_iter": recipe["model"]["params"]["max_iter"]},
            "metadata": {"controller_id": RAW_CONTROLLER}}


def lower_classifier_choices(steps: list[Any], splitter: Any, schemas: Mapping[str, Any], vocabulary: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Lower each public declaration once; native compilation owns expansion."""
    branches: list[dict[str, Any]] = []
    bindings: dict[str, list[dict[str, str]]] = {}
    sinks: list[str] = []

    def bind(path: str, node_id: str, param_path: str) -> None:
        bindings.setdefault(path, []).append({"node_id": node_id, "param_path": param_path})

    for index, sequence in enumerate(declared_classifier_topologies(steps, splitter)):
        prefix = f"a{index}"
        if len(sequence) == 1:
            node = _typed_node(_model_step(sequence[0]), f"{prefix}:early", schemas, vocabulary)
            native_steps = [node]
            bind("early.n_components", node["id"], "model__n_components")
            sinks.append(node["id"])
        else:
            raw_branches: list[dict[str, Any]] = []
            sources: list[str] = []
            order = list(sequence[0]["branch"])
            for name, branch in sequence[0]["branch"].items():
                node = _typed_node(_model_step(branch[0]), f"{prefix}:source:{name}", schemas, vocabulary)
                raw_branches.append({"id": name, "steps": [node]})
                sources.append(node["id"])
                bind(f"late.{name}.n_components", node["id"], "model__n_components")
            meta_id = f"{prefix}:meta"
            operator = _meta_operator(_model_step(sequence[2]), order, vocabulary)
            native_steps = [{"kind": "branch", "mode": "duplication", "branches": raw_branches}, {
                "kind": "merge_model", "id": meta_id, "operator": operator, "prediction_output_ports": ["y_hat", "probabilities"],
                "params": copy.deepcopy(operator["steps"][0]["params"]),
                "sources": sources, "source_ports": dict.fromkeys(sources, "probabilities"), "merge_mode": "predictions", "include_original_data": False,
                "metadata": {"controller_id": META_CONTROLLER, "stacking_oof_execution": "nested_oof_v1", "stacking_refit_oof": "partitioned_inner_v1"},
                "inner_cv": {"kind": "group_kfold", "n_splits": INNER_SPLITS},
            }]
            bind("late.meta.n_components", meta_id, "n_components")
            sinks.append(meta_id)
        branches.append({"id": f"topology{index}", "steps": native_steps})
    dsl = {"id": "nirs4all-classification-structural-hpo", "pipeline": [{
        "kind": "generator", "id": "generator:topologies", "mode": "cartesian",
        "stages": [{"id": "topology", "branches": branches}],
    }]}
    return dsl, bindings, sinks


def validate_count_space(declaration: Any) -> None:
    """Require an integer proposal domain; native checks actual fold capacities."""
    from . import tuning_adapters

    choices = tuning_adapters._categorical_choices(declaration)
    if choices is not None:
        values = choices
    elif isinstance(declaration, tuple) and len(declaration) in (2, 3):
        if len(declaration) == 3 and declaration[0] not in {"int", "int_log", "log_int"}:
            raise ValueError("classifier component axes require integer distributions")
        values = list(declaration[-2:])
    elif isinstance(declaration, dict) and str(declaration.get("type", "int")).lower() in {"int", "int_log", "log_int"}:
        values = [declaration.get("low", declaration.get("min")), declaration.get("high", declaration.get("max"))]
        if declaration.get("step") is not None and (type(declaration["step"]) is not int or declaration["step"] < 1):
            raise ValueError("classifier component steps must be positive integers")
    else:
        raise ValueError("classifier component axes require integer ranges or choices")
    if not values or any(type(value) is not int or not 1 <= value <= (1 << 31) - 1 for value in values):
        raise ValueError("classifier component proposals must be positive int32 integers")


def prepare_classification_structure(steps: list[Any], splitter: Any, dataset: Any, spec: DagMLTuningSpec,
                                     run_options: dict[str, Any], native: Any, *, require_space: bool = True) -> dict[str, Any]:
    """Sign Train-only label identities; native validates every actual train scope."""
    from dag_ml.multimodal_classification import ClassificationTopologyController

    if not isinstance(dataset, MultimodalSpectroDataset):
        raise TypeError("native classification requires the complete typed MultimodalDataset")
    if spec.n_jobs != 1 or spec.pruner not in {None, "none"}:
        raise ValueError("native multimodal classification is serial without pruning; parallel classification is unqualified")
    if spec.metric not in {"accuracy", "balanced_accuracy", "f1"} or spec.direction != "maximize":
        raise ValueError("native classification requires maximizing accuracy, balanced_accuracy or f1")
    if not callable(getattr(native, "prepare_host_hpo_topology_catalogue", None)):
        raise ImportError("installed DAG-ML lacks native classifier topology preflight")
    alternatives = declared_classifier_topologies(steps, splitter)
    cohort = dataset.cohort
    if (run_options.get("refit", True) is not True or cohort.y is None or tuple(cohort.target_names) != ("y",)
            or not np.asarray(cohort.target_mask).all() or cohort.groups is None
            or any(partition not in {"train", "test"} for partition in cohort.partitions)
            or getattr(cohort, "_generated_view_store", None) is not None):
        raise ValueError("native classification requires refit=True, complete y, grouped Train/Test and fixed raw buffers")
    schemas = source_schemas_from_cohort(cohort)
    pool = dataset.index_column("sample", {"partition": "train"})
    vocabulary = classification_vocabulary(cohort.y, pool)
    encoded = encode_labels(cohort.y, vocabulary)
    dsl, bindings, sinks = lower_classifier_choices(steps, splitter, schemas, vocabulary)
    if require_space and set(spec.space) != set(bindings):
        raise ValueError(f"classification requires exactly the declared node-qualified n_components axes: {sorted(bindings)}")
    for declaration in spec.space.values():
        validate_count_space(declaration)
    identity = mint_identity(dataset)
    folds = _build_folds(splitter, dataset, pool, set())
    groups = _split_group_grain(splitter, dataset, pool)
    if groups is None or len(folds) != 3 or any(not train or not heldout for train, heldout in folds):
        raise ValueError("native classification requires three complete nonempty grouped folds")
    envelope = build_envelope(dataset, identity, sample_ints=pool, group_by_sample=groups)
    seed = run_options.get("random_state")
    seed = (spec.seed if spec.seed is not None else 0) if seed is None else seed
    if type(seed) is not int or not 0 <= seed <= (1 << 64) - 1:
        raise ValueError("classifier operator seed must be an unsigned 64-bit integer")
    dsl["root_seed"] = seed
    dsl["metadata"] = {"classification_targets": {**copy.deepcopy(vocabulary),
        "sample_labels": {identity.to_wire(sample): int(encoded[sample]) for sample in pool}}}
    dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=3, shuffle=False)
    dsl["split_invocation"]["fold_set"]["sample_groups"] = {identity.to_wire(sample): group for sample, group in groups.items()}
    declarations: dict[str, Any] = {}
    for branch in dsl["pipeline"][0]["stages"][0]["branches"]:
        for node in branch["steps"]:
            if node["kind"] == "branch":
                for child in node["branches"]:
                    declarations.update({model["id"]: model["operator"] for model in child["steps"]})
            else:
                declarations[node["id"]] = node["operator"]
    controller = ClassificationTopologyController(operators=declarations, sources=controller_sources(cohort, schemas),
        targets=None, target_names=("y",), allow_fit=False, source_ids=tuple(source_ids(dataset)))
    try:
        manifests = copy.deepcopy(controller.manifests)
    finally:
        controller.close()
    graph = native.compile_pipeline_dsl_artifact_with_controllers(dsl, manifests).graph.to_dict()
    raw_ids = [node["id"] for node in graph["nodes"] if (node.get("operator") or {}).get("type") == RAW_TYPE]
    dsl["data_bindings"] = data_bindings_for_nodes(raw_ids, envelope)
    for binding in dsl["data_bindings"]:
        binding["view_policy"] = {"include_augmented_train": False, "include_refit_test_view": True}
    catalogue = native.prepare_host_hpo_topology_catalogue(dsl, envelope, manifests, bindings, sinks)
    descriptor = spec.to_dict()
    for key in ("resume", "n_trials", "storage", "study_name"):
        descriptor.pop(key, None)
    descriptor["operator_rng"] = {"policy": "per_native_task_v1", "seed": seed}
    descriptor["training_content_fingerprint"] = tcv1_sha256({"raw_source_content": dataset.content_hash(),
        "targets": np.asarray(cohort.y).tolist(), "classification": vocabulary,
        "groups": [{"sample_id": identity.to_wire(sample), "group": groups[sample]} for sample in pool]})
    request = {"target_node": catalogue["entries"][0]["target_node"], "trial_budget": spec.n_trials,
        "metric": spec.metric, "direction": spec.direction, "optimizer_descriptor": descriptor,
        "fold_score_reduction": "mean", "structural_catalogue": catalogue}
    return {"spec": spec, "dataset": dataset, "identity": identity, "folds": folds, "envelope": envelope,
        "dsl": dsl, "manifests": manifests, "graph": graph, "catalogue": catalogue, "request": request,
        "operator_seed": seed, "splitter": splitter, "steps": steps, "pool": pool, "groups": groups,
        "classification": vocabulary, "encoded_targets": encoded, "methods_typed": True, "methods_classification": True}


def classification_output(graph: Mapping[str, Any], node_id: str, vocabulary: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the label port; probability columns remain signed native auxiliaries."""
    from .raw_training_lowerer import _output_request_for_node

    output = _output_request_for_node(graph, node_id, target_names=["y"])
    output.update(port_name="y_hat", prediction_kind="class_label",
                  class_labels=[[str(label) for label in vocabulary["class_labels"]]])
    return output


def train_selected_classification(prepared: dict[str, Any], evidence: dict[str, Any] | None, run_options: dict[str, Any]) -> Any:
    """Resolve the exact native sink and capture all its portable predecessors."""
    from .raw_training_lowerer import _array_content_fingerprint, _core_relation_fingerprint, _data_contracts_from_campaign, _training_influence_manifest
    from .resources import current_execution_resources
    from .training_contracts import DagMLTrainingRequestSpec, assemble_training_request

    native = importlib.import_module("dag_ml")
    dataset, identity = prepared["dataset"], prepared["identity"]
    artifact = native.compile_pipeline_dsl_artifact_with_controllers(prepared["dsl"], prepared["manifests"])
    graph, campaign = artifact.graph.to_dict(), artifact.campaign_template.to_dict()
    envelope = copy.deepcopy(prepared["envelope"])
    envelope["relation_fingerprint"] = _core_relation_fingerprint(envelope["coordinator_relations"], native)
    envelope["data_content_fingerprint"] = dataset.content_hash(sample_rows=prepared["pool"])
    envelope["target_content_fingerprint"] = _array_content_fingerprint("y", prepared["encoded_targets"][prepared["pool"]])
    test = dataset.index_column("sample", {"partition": "test"})
    if test:
        test_envelope = build_envelope(dataset, identity, sample_ints=test)
        envelope.update(native.attach_predict_cohort_to_envelope(envelope, {
            "role": "external_test", "relations": test_envelope["coordinator_relations"], "target_names": ["y"],
            "data_content_fingerprint": dataset.content_hash(sample_rows=test),
            "target_content_fingerprint": _array_content_fingerprint("y", prepared["encoded_targets"][test]),
        }).to_dict())
    _envelopes, identities = _data_contracts_from_campaign(campaign, envelope)
    output = classification_output(graph, prepared["request"]["target_node"], prepared["classification"])
    resources = current_execution_resources()
    template = assemble_training_request(DagMLTrainingRequestSpec(
        request_id="training:nirs4all.classification", plan_id="plan:nirs4all.classification", graph=graph, campaign=campaign,
        controller_manifests=prepared["manifests"], data_identities=identities, output_requests=[output],
        seed=prepared["operator_seed"], cv_artifacts="discard", fitted_artifacts="portable_required", cpu_threads=resources.cpu_threads,
        gpu_devices=resources.gpu_devices, selection_metric=prepared["spec"].metric, selection_objective="maximize",
        selection_required_metric_level="sample", selection_evaluation_scope="oof",
    ))
    if evidence is None:
        # A singleton fixed declaration has no proposal/optimizer callback.
        request = template
        winner = prepared["catalogue"]["entries"][0]
    else:
        request = native.resolve_host_hpo_structural_winner(prepared["request"], evidence, template)
        winner = next(entry for entry in prepared["catalogue"]["entries"] if entry["recipe_id"] == evidence["selected_params"][prepared["catalogue"]["selector_path"]])
    output = classification_output(request["graph"], winner["target_node"], prepared["classification"])
    data_envelopes, _identities = _data_contracts_from_campaign(request["campaign"], envelope)
    influence = _training_influence_manifest(request["graph"], request["campaign"], prepared["folds"], identity, group_by_sample=prepared["groups"], selection_metric=prepared["spec"].metric)
    result = execute_methods_training(spectro=dataset, identity=identity, envelope=envelope, request=request,
                                      data_envelopes=data_envelopes, influence=influence, output=output,
                                      name=run_options.get("name", ""), binding_source_ids=source_ids(dataset))
    result._dagml_graph = copy.deepcopy(request["graph"])
    result.structural_tuning_training_request = copy.deepcopy(request)
    result.structural_tuning_training_outcome = copy.deepcopy(result._methods_multimodal_outcome.to_dict())
    for metadata in result.per_dataset.values():
        metadata["tuning_profile"] = "structural_classification_v2"
    return result


def run_fixed_classifier(pipeline: Any, dataset: Any, *, name: str, random_state: int | None, refit: bool) -> Any:
    """Preflight a fixed classifier before native CV/SELECT/REFIT, without HPO."""
    from .steps import _split_pipeline

    steps, splitter = _split_pipeline(pipeline)
    if len(steps) == 1 and isinstance(steps[0], dict) and not set(steps[0]) - {"model", "name"}:
        sequence = [{"model": _model_step({"model": steps[0].get("model")})}]
        classifier_recipe(sequence[0]["model"], allow_source_selection=True)
        space = {"early.n_components": [classifier_head(sequence[0]["model"].model)["params"]["n_components"]]}
    else:
        sequence = steps
        declared_classifier_topologies([{"_or_": [sequence]}], splitter)
        space = {f"late.{name}.n_components": [classifier_head(branch[0]["model"].model)["params"]["n_components"]]
                 for name, branch in sequence[0]["branch"].items()}
        space["late.meta.n_components"] = [classifier_head(sequence[2]["model"])["params"]["n_components"]]
    options = {"name": name, "random_state": random_state, "refit": refit}
    spec = parse_tuning_spec({"engine": "n4m", "space": space, "metric": "accuracy", "direction": "maximize", "n_trials": 1})
    prepared = prepare_classification_structure([{"_or_": [sequence]}], splitter, dataset, spec, options, importlib.import_module("dag_ml"))
    return train_selected_classification(prepared, None, options)
