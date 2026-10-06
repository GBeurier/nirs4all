"""Lower declared early/learned-late choices into native structural campaigns.

This bridge transports declarations and complete raw buffers. DAG-ML owns
recipe generation, grouped nested OOF, conditional patches and scoring; Methods
owns all learned encoders, branch predictors and the terminal Ridge.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from nirs4all.data.multimodal import MultimodalSpectroDataset

from .cli_runner import data_bindings_for_nodes, split_invocation_for
from .envelope import build_envelope, source_ids
from .folds import _build_folds, _split_group_grain
from .identity import mint_identity
from .methods_multimodal import (
    SOURCE_ORDER,
    _finite_nonnegative,
    controller_sources,
    execute_methods_training,
    methods_model_in_pipeline,
    recipe_from_estimator,
    source_schemas_from_cohort,
    validate_training_profile,
)
from .structural_multimodal import validate_alpha_space
from .tuning_contracts import DagMLTuningSpec, parse_tuning_spec, tcv1_sha256
from .typed_parallel import typed_parallel_execution

RAW_CONTROLLER = "controller:methods.python.multimodal"
META_CONTROLLER = "controller:methods.python.regression"
INNER_SPLITS = 2


def is_topology_choice(steps: list[Any]) -> bool:
    """Recognize existing sequence alternatives containing Methods declarations."""
    return (len(steps) == 1 and isinstance(steps[0], dict)
            and isinstance(steps[0].get("_or_"), list)
            and methods_model_in_pipeline(steps[0]["_or_"]) is not None)


def _model_step(step: Any) -> Any:
    if not isinstance(step, dict) or set(step) != {"model"}:
        raise ValueError("topology model steps require exactly {'model': estimator}")
    return step["model"]


def _meta_operator(model: Any, order: list[str]) -> dict[str, Any]:
    if (type(model) is not Ridge or model.fit_intercept is not True or model.positive is not False
            or model.solver != "auto" or model.max_iter is not None or model.random_state is not None
            or model.tol != 1e-4 or model.copy_X is not True):
        raise ValueError("learned late fusion requires ordinary Ridge(alpha=..., fit_intercept=True, solver='auto')")
    return {"type": "N4mRolePipeline", "steps": [{
        "methodId": "models.regularized.ridge",
        "params": {"alpha": _finite_nonnegative(model.alpha, "late.meta.alpha"), "center_x": True, "center_y": True, "scale_x": False},
    }], "source_order": list(order)}


def declared_topologies(steps: list[Any], splitter: Any) -> list[list[Any]]:
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
            recipe_from_estimator(_model_step(sequence[0]), allow_source_selection=True)
            continue
        if (len(sequence) != 3 or not isinstance(sequence[0], dict) or set(sequence[0]) != {"branch"}
                or sequence[1] != {"merge": "predictions"}):
            raise ValueError("late topology requires named branches, merge='predictions', and one Ridge meta-model")
        branches = sequence[0]["branch"]
        if not isinstance(branches, dict) or not 2 <= len(branches) <= 4 or set(branches) - set(SOURCE_ORDER):
            raise ValueError("late fusion requires two to four distinct named raw-source branches")
        for name, branch in branches.items():
            if not isinstance(branch, list) or len(branch) != 1:
                raise ValueError("each late branch requires one complete Methods multimodal predictor")
            recipe = recipe_from_estimator(_model_step(branch[0]), allow_source_selection=True)
            if recipe["source_order"] != [name]:
                raise ValueError("late branch predictor must select exactly its named source")
        _meta_operator(_model_step(sequence[2]), list(branches))
    return alternatives


def _typed_node(model: Any, node_id: str, schemas: Mapping[str, Any]) -> dict[str, Any]:
    recipe = recipe_from_estimator(model, allow_source_selection=True)
    return {"kind": "model", "id": node_id,
            "operator": {"type": "N4mMultimodalPipeline", "recipe": recipe, "source_schemas": copy.deepcopy(dict(schemas))},
            "params": {"recipe": copy.deepcopy(recipe), "source_schemas": copy.deepcopy(dict(schemas)), "model__alpha": recipe["model"]["params"]["alpha"]},
            "metadata": {"controller_id": RAW_CONTROLLER}}


def lower_topology_choices(steps: list[Any], splitter: Any, schemas: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Lower each public declaration once; native compilation owns expansion."""
    branches: list[dict[str, Any]] = []
    bindings: dict[str, list[dict[str, str]]] = {}
    sinks: list[str] = []

    def bind(path: str, node_id: str, param_path: str) -> None:
        bindings.setdefault(path, []).append({"node_id": node_id, "param_path": param_path})

    for index, sequence in enumerate(declared_topologies(steps, splitter)):
        prefix = f"a{index}"
        if len(sequence) == 1:
            node = _typed_node(_model_step(sequence[0]), f"{prefix}:early", schemas)
            native_steps = [node]
            bind("early.alpha", node["id"], "model__alpha")
            sinks.append(node["id"])
        else:
            raw_branches: list[dict[str, Any]] = []
            sources: list[str] = []
            order = list(sequence[0]["branch"])
            for name, branch in sequence[0]["branch"].items():
                node = _typed_node(_model_step(branch[0]), f"{prefix}:source:{name}", schemas)
                raw_branches.append({"id": name, "steps": [node]})
                sources.append(node["id"])
                bind(f"late.{name}.alpha", node["id"], "model__alpha")
            meta_id = f"{prefix}:meta"
            operator = _meta_operator(_model_step(sequence[2]), order)
            native_steps = [{"kind": "branch", "mode": "duplication", "branches": raw_branches}, {
                "kind": "merge_model", "id": meta_id, "operator": operator,
                "params": {"alpha": operator["steps"][0]["params"]["alpha"]},
                "sources": sources, "merge_mode": "predictions", "include_original_data": False,
                "metadata": {"controller_id": META_CONTROLLER, "stacking_oof_execution": "nested_oof_v1", "stacking_refit_oof": "partitioned_inner_v1"},
                "inner_cv": {"kind": "group_kfold", "n_splits": INNER_SPLITS},
            }]
            bind("late.meta.alpha", meta_id, "alpha")
            sinks.append(meta_id)
        branches.append({"id": f"topology{index}", "steps": native_steps})
    dsl = {"id": "nirs4all-early-late-structural-hpo", "input": {"name": "x", "representation": "feature_block_set"}, "pipeline": [{
        "kind": "generator", "id": "generator:topologies", "mode": "cartesian",
        "stages": [{"id": "topology", "branches": branches}],
    }]}
    return dsl, bindings, sinks


def _typed_models(alternatives: list[list[Any]]) -> list[Any]:
    models: list[Any] = []
    for sequence in alternatives:
        if len(sequence) == 1:
            models.append(_model_step(sequence[0]))
        else:
            models.extend(_model_step(branch[0]) for branch in sequence[0]["branch"].values())
    return models


def prepare_topology_structure(steps: list[Any], splitter: Any, dataset: Any, spec: DagMLTuningSpec, run_options: dict[str, Any], native: Any) -> dict[str, Any]:
    """Bind complete raw identity and ask native DAG-ML to validate every scope."""
    from dag_ml.multimodal_topology import MethodsTopologyController

    if not isinstance(dataset, MultimodalSpectroDataset):
        raise TypeError("early/late structural tuning requires a complete typed MultimodalDataset")
    if not dataset.is_regression:
        raise ValueError("early/late structural tuning requires a regression target")
    if spec.pruner not in {None, "none"}:
        raise ValueError("early/late structural tuning requires no pruning")
    parallel_execution = typed_parallel_execution(spec, run_options)
    if not callable(getattr(native, "prepare_host_hpo_topology_catalogue", None)):
        raise ImportError("installed DAG-ML lacks native topology catalogue preparation")
    alternatives = declared_topologies(steps, splitter)
    cohort = dataset.cohort
    for model in _typed_models(alternatives):
        validate_training_profile([splitter, {"model": model}], cohort, refit=run_options.get("refit", True) is True, allow_source_selection=True)
    schemas = source_schemas_from_cohort(cohort)
    dsl, bindings, sinks = lower_topology_choices(steps, splitter, schemas)
    if set(spec.space) != set(bindings):
        raise ValueError(f"early/late tuning requires exactly the declared node-qualified alpha axes: {sorted(bindings)}")
    for declaration in spec.space.values():
        validate_alpha_space(parse_tuning_spec({"engine": "n4m", "space": {"model.alpha": declaration}}))
    pool = dataset.index_column("sample", {"partition": "train"})
    identity = mint_identity(dataset)
    folds = _build_folds(splitter, dataset, pool, set())
    groups = _split_group_grain(splitter, dataset, pool)
    if groups is None or len(folds) != 3 or any(not train or not heldout for train, heldout in folds):
        raise ValueError("early/late tuning requires three complete nonempty grouped folds")
    envelope = build_envelope(dataset, identity, sample_ints=pool, group_by_sample=groups)
    seed = run_options.get("random_state")
    seed = (spec.seed if spec.seed is not None else 0) if seed is None else seed
    if type(seed) is not int or not 0 <= seed <= (1 << 64) - 1:
        raise ValueError("topology operator seed must be an unsigned 64-bit integer")
    dsl["root_seed"] = seed
    dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=3, shuffle=False)
    # The native nested planner consumes FoldSet group authority separately
    # from the coordinator relation groups already carried by the envelope.
    dsl["split_invocation"]["fold_set"]["sample_groups"] = {
        identity.to_wire(sample): group for sample, group in groups.items()
    }
    declarations: dict[str, Any] = {}
    for branch in dsl["pipeline"][0]["stages"][0]["branches"]:
        for node in branch["steps"]:
            if node["kind"] == "branch":
                for child in node["branches"]:
                    declarations.update({model["id"]: model["operator"] for model in child["steps"]})
            else:
                declarations[node["id"]] = node["operator"]
    controller = MethodsTopologyController(operators=declarations, sources=controller_sources(cohort, schemas), targets=None,
                                           target_names=("y",), allow_fit=False, source_ids=tuple(source_ids(dataset)))
    try:
        manifests = copy.deepcopy(controller.manifests)
    finally:
        controller.close()
    graph = native.compile_pipeline_dsl_artifact_with_controllers(dsl, manifests).graph.to_dict()
    raw_ids = [node["id"] for node in graph["nodes"] if (node.get("operator") or {}).get("type") == "N4mMultimodalPipeline"]
    dsl["data_bindings"] = data_bindings_for_nodes(raw_ids, envelope)
    for binding in dsl["data_bindings"]:
        binding["view_policy"] = {"include_augmented_train": False, "include_refit_test_view": True}
    # V2 native preflight validates PCA against actual grouped inner training
    # scopes for outer CV and full REFIT before any optimizer/model callback.
    catalogue = native.prepare_host_hpo_topology_catalogue(dsl, envelope, manifests, bindings, sinks)
    descriptor = spec.to_dict()
    if parallel_execution is not None:
        descriptor["parallel_execution"] = parallel_execution
    for key in ("resume", "n_trials", "storage", "study_name"):
        descriptor.pop(key, None)
    descriptor["operator_rng"] = {"policy": "per_native_task_v1", "seed": seed}
    descriptor["training_content_fingerprint"] = tcv1_sha256({
        "raw_source_content": dataset.content_hash(), "targets": np.asarray(cohort.y).tolist(),
        "groups": [{"sample_id": identity.to_wire(sample), "group": groups[sample]} for sample in pool],
    })
    request = {"target_node": catalogue["entries"][0]["target_node"], "trial_budget": spec.n_trials,
               "metric": spec.metric, "direction": spec.direction, "optimizer_descriptor": descriptor,
               "fold_score_reduction": "mean", "structural_catalogue": catalogue}
    return {"spec": spec, "dataset": dataset, "identity": identity, "folds": folds, "envelope": envelope,
            "dsl": dsl, "manifests": manifests, "graph": graph, "catalogue": catalogue, "request": request,
            "operator_seed": seed, "splitter": splitter, "steps": steps, "pool": pool, "groups": groups,
            "methods_typed": True, "methods_topology": True}


def train_selected_topology(prepared: dict[str, Any], evidence: dict[str, Any], run_options: dict[str, Any]) -> Any:
    """Resolve the exact native sink and capture all its portable predecessors."""
    from .raw_training_lowerer import _array_content_fingerprint, _core_relation_fingerprint, _data_contracts_from_campaign, _output_request_for_node, _training_influence_manifest
    from .resources import current_execution_resources
    from .training_contracts import DagMLTrainingRequestSpec, assemble_training_request

    native = importlib.import_module("dag_ml")
    dataset, identity = prepared["dataset"], prepared["identity"]
    artifact = native.compile_pipeline_dsl_artifact_with_controllers(prepared["dsl"], prepared["manifests"])
    graph, campaign = artifact.graph.to_dict(), artifact.campaign_template.to_dict()
    envelope = copy.deepcopy(prepared["envelope"])
    envelope["relation_fingerprint"] = _core_relation_fingerprint(envelope["coordinator_relations"], native)
    envelope["data_content_fingerprint"] = dataset.content_hash(sample_rows=prepared["pool"])
    envelope["target_content_fingerprint"] = _array_content_fingerprint("y", np.asarray(dataset.cohort.y)[prepared["pool"]])
    test = dataset.index_column("sample", {"partition": "test"})
    if test:
        test_envelope = build_envelope(dataset, identity, sample_ints=test)
        envelope.update(native.attach_predict_cohort_to_envelope(envelope, {
            "role": "external_test", "relations": test_envelope["coordinator_relations"], "target_names": ["y"],
            "data_content_fingerprint": dataset.content_hash(sample_rows=test),
            "target_content_fingerprint": _array_content_fingerprint("y", np.asarray(dataset.cohort.y)[test]),
        }).to_dict())
    _envelopes, identities = _data_contracts_from_campaign(campaign, envelope)
    output = _output_request_for_node(graph, prepared["request"]["target_node"], target_names=["y"])
    resources = current_execution_resources()
    template = assemble_training_request(DagMLTrainingRequestSpec(
        request_id="training:nirs4all.early-late", plan_id="plan:nirs4all.early-late", graph=graph, campaign=campaign,
        controller_manifests=prepared["manifests"], data_identities=identities, output_requests=[output],
        seed=prepared["operator_seed"], cv_artifacts="discard", fitted_artifacts="portable_required", cpu_threads=resources.cpu_threads,
        gpu_devices=resources.gpu_devices, selection_required_metric_level="sample", selection_evaluation_scope="oof",
    ))
    request = native.resolve_host_hpo_structural_winner(prepared["request"], evidence, template)
    winner = next(entry for entry in prepared["catalogue"]["entries"] if entry["recipe_id"] == evidence["selected_params"][prepared["catalogue"]["selector_path"]])
    output = _output_request_for_node(request["graph"], winner["target_node"], target_names=["y"])
    data_envelopes, _identities = _data_contracts_from_campaign(request["campaign"], envelope)
    influence = _training_influence_manifest(request["graph"], request["campaign"], prepared["folds"], identity, group_by_sample=prepared["groups"], selection_metric="rmse")
    result = execute_methods_training(spectro=dataset, identity=identity, envelope=envelope, request=request,
                                      data_envelopes=data_envelopes, influence=influence, output=output,
                                      name=run_options.get("name", ""), binding_source_ids=source_ids(dataset))
    result._dagml_graph = copy.deepcopy(request["graph"])
    result.structural_tuning_training_request = copy.deepcopy(request)
    result.structural_tuning_training_outcome = copy.deepcopy(result._methods_multimodal_outcome.to_dict())
    for metadata in result.per_dataset.values():
        metadata["tuning_profile"] = "structural_early_late_v2"
    return result
