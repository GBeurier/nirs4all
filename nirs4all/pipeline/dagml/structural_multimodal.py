"""Typed structural alternatives with one complete Methods early-fusion Ridge.

The SDK lowers declarations and supplies raw buffers. Native DAG-ML enumerates
recipes and Methods alone learns every selected encoder and predictor.
"""

from __future__ import annotations

import copy
import importlib
import math
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.operators.models.multimodal import MultimodalRegressor

from . import tuning_adapters
from .cli_runner import data_bindings_for_nodes, split_invocation_for
from .envelope import build_envelope, source_ids
from .folds import _build_folds, _split_group_grain
from .identity import mint_identity
from .methods_multimodal import controller_sources, execute_methods_training, recipe_from_estimator, source_schemas_from_cohort, validate_training_profile
from .tuning_contracts import DagMLTuningSpec, tcv1_sha256

PARAMETER_PATHS = {"model.alpha": "model__alpha"}


def is_methods_model_choice(steps: list[Any]) -> bool:
    """Identify the additive profile while retaining the old dense classifiers."""
    if len(steps) != 1 or not isinstance(steps[0], dict):
        return False
    choices = steps[0].get("model")
    alternatives = choices.get("_or_") if isinstance(choices, dict) else None
    return isinstance(alternatives, list) and any(isinstance(model, MultimodalRegressor) and model.backend == "methods" for model in alternatives)


def validate_typed_profile(steps: list[Any], splitter: Any) -> None:
    """Check concrete declared alternatives without opening native estimators."""
    if type(splitter) is not GroupKFold or splitter.n_splits != 3:
        raise ValueError("typed structural tuning requires GroupKFold(3) and cohort groups")
    if set(steps[0]) != {"model"} or set(steps[0]["model"]) != {"_or_"}:
        raise ValueError("typed structural tuning requires exactly model={'_or_': [Methods MultimodalRegressor, ...]}")
    for model in steps[0]["model"]["_or_"]:
        recipe_from_estimator(model, allow_source_selection=True)


def validate_alpha_space(spec: DagMLTuningSpec) -> None:
    """Require the sole numeric axis to remain finite nonnegative Ridge alpha."""
    if set(spec.space) != set(PARAMETER_PATHS):
        raise ValueError("typed structural tuning supports only model__alpha; modality weights are explicit recipe declarations")
    declaration = spec.space["model.alpha"]
    codec = tuning_adapters._categorical_codec_for_spec(declaration)
    choices = tuning_adapters._categorical_choices(declaration)
    if choices is not None:
        values = [codec.decode(value) for value in codec.choices] if codec is not None else choices
    elif isinstance(declaration, tuple):
        values = list(declaration[-2:])
    elif isinstance(declaration, dict):
        values = [declaration.get("low", declaration.get("min")), declaration.get("high", declaration.get("max"))]
        step = declaration.get("step")
        if step is not None:
            values.append(step)
    else:
        raise ValueError("unsupported typed structural alpha declaration")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("typed structural Ridge alpha bounds and choices must be finite nonnegative numbers")


def lower_typed_choices(steps: list[Any], schemas: dict[str, Any]) -> dict[str, Any]:
    """Lower one native Cartesian stage; recipe identities remain native-owned."""
    branches = []
    for index, model in enumerate(steps[0]["model"]["_or_"]):
        recipe = recipe_from_estimator(model, allow_source_selection=True)
        operator = {"type": "N4mMultimodalPipeline", "recipe": recipe, "source_schemas": copy.deepcopy(schemas)}
        params = {"recipe": copy.deepcopy(recipe), "source_schemas": copy.deepcopy(schemas), "model__alpha": recipe["model"]["params"]["alpha"]}
        branches.append(
            {
                "id": f"s0op{index}",
                "steps": [
                    {
                        "kind": "model",
                        "id": f"m:s0op{index}",
                        "operator": operator,
                        "params": params,
                        "metadata": {"controller_id": "controller:methods.python.multimodal"},
                    }
                ],
            }
        )
    return {
        "id": "nirs4all-typed-structural-hpo",
        "pipeline": [
            {
                "kind": "generator",
                "id": "generator:modalities",
                "mode": "cartesian",
                "stages": [{"id": "stage0", "branches": branches}],
            }
        ],
    }


def prepare_typed_structure(steps: list[Any], splitter: Any, dataset: Any, spec: DagMLTuningSpec, run_options: dict[str, Any], native: Any) -> dict[str, Any]:
    """Seal full raw contracts before catalogue preparation or optimizer FIT."""
    from dag_ml.multimodal_methods import MethodsMultimodalController

    if not isinstance(dataset, MultimodalSpectroDataset):
        raise TypeError("typed structural tuning requires an IO MultimodalDataset with the complete U07 raw input contract")
    if spec.n_jobs != 1:
        raise ValueError("typed structural Methods tuning currently requires n_jobs=1")
    validate_alpha_space(spec)
    cohort = dataset.cohort
    for model in steps[0]["model"]["_or_"]:
        validate_training_profile([splitter, {"model": model}], cohort, refit=run_options.get("refit", True) is True, allow_source_selection=True)
    schemas = source_schemas_from_cohort(cohort)
    pool = dataset.index_column("sample", {"partition": "train"})
    identity = mint_identity(dataset)
    folds = _build_folds(splitter, dataset, pool, set())
    if len(folds) != 3 or any(not train or not validation for train, validation in folds):
        raise ValueError("typed structural tuning requires three nonempty grouped training folds")
    for model in steps[0]["model"]["_or_"]:
        recipe = recipe_from_estimator(model, allow_source_selection=True)
        for name, encoder in recipe["encoders"].items():
            if encoder["kind"] == "tensor_pca" and encoder["n_components"] > min(math.prod(schemas[name]["input_shape"]), *(len(train) for train, _ in folds)):
                raise ValueError(f"typed structural {name} PCA components exceed its raw width or a fold training size")
    groups = _split_group_grain(splitter, dataset, pool)
    if groups is None:
        raise ValueError("typed structural tuning requires complete cohort groups")
    envelope = build_envelope(dataset, identity, sample_ints=pool, group_by_sample=groups)
    seed = run_options.get("random_state")
    seed = (spec.seed if spec.seed is not None else 0) if seed is None else seed
    if type(seed) is not int or not 0 <= seed <= (1 << 64) - 1:
        raise ValueError("typed structural operator seed must be an unsigned 64-bit integer")
    dsl = lower_typed_choices(steps, schemas)
    dsl["root_seed"] = seed
    dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=3, shuffle=bool(getattr(splitter, "shuffle", False)))
    declarations = {branch["steps"][0]["id"]: branch["steps"][0]["operator"] for branch in dsl["pipeline"][0]["stages"][0]["branches"]}
    declaration = MethodsMultimodalController(operators=declarations, sources=controller_sources(cohort, schemas), targets=None, target_names=("y",), allow_fit=False, source_ids=tuple(source_ids(dataset)))
    try:
        manifests = [copy.deepcopy(declaration.manifest)]
    finally:
        declaration.close()
    graph = native.compile_pipeline_dsl_artifact_with_controllers(dsl, manifests).graph.to_dict()
    dsl["data_bindings"] = data_bindings_for_nodes([node["id"] for node in graph["nodes"]], envelope)
    for binding in dsl["data_bindings"]:
        binding["view_policy"] = {"include_augmented_train": False, "include_refit_test_view": True}
    catalogue = native.prepare_host_hpo_structural_catalogue(dsl, envelope, manifests, PARAMETER_PATHS)
    descriptor = spec.to_dict()
    for key in ("resume", "n_trials", "storage", "study_name"):
        descriptor.pop(key, None)
    descriptor["operator_rng"] = {"policy": "per_native_task_v1", "seed": seed}
    descriptor["training_content_fingerprint"] = tcv1_sha256(
        {
            "raw_source_content": dataset.content_hash(),
            "targets": np.asarray(cohort.y).tolist(),
            "groups": [{"sample_id": identity.to_wire(sample), "group": groups[sample]} for sample in pool],
        }
    )
    request = {
        "target_node": catalogue["entries"][0]["target_node"],
        "trial_budget": spec.n_trials,
        "metric": spec.metric,
        "direction": spec.direction,
        "optimizer_descriptor": descriptor,
        "fold_score_reduction": "mean",
        "structural_catalogue": catalogue,
    }
    if spec.pruner not in {None, "none"}:
        request["progressive_pruning"] = True
    return {
        "spec": spec,
        "dataset": dataset,
        "identity": identity,
        "folds": folds,
        "envelope": envelope,
        "dsl": dsl,
        "manifests": manifests,
        "graph": graph,
        "catalogue": catalogue,
        "request": request,
        "operator_seed": seed,
        "splitter": splitter,
        "steps": steps,
        "pool": pool,
        "groups": groups,
        "methods_typed": True,
    }


def train_selected_typed_structure(prepared: dict[str, Any], evidence: dict[str, Any], run_options: dict[str, Any]) -> Any:
    """Resolve the native winner and reuse the complete Methods portable capture."""
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
        envelope.update(
            native.attach_predict_cohort_to_envelope(
                envelope,
                {
                    "role": "external_test",
                    "relations": test_envelope["coordinator_relations"],
                    "target_names": ["y"],
                    "data_content_fingerprint": dataset.content_hash(sample_rows=test),
                    "target_content_fingerprint": _array_content_fingerprint("y", np.asarray(dataset.cohort.y)[test]),
                },
            ).to_dict()
        )
    _envelopes, identities = _data_contracts_from_campaign(campaign, envelope)
    output = _output_request_for_node(graph, prepared["request"]["target_node"], target_names=["y"])
    resources = current_execution_resources()
    template = assemble_training_request(
        DagMLTrainingRequestSpec(
            request_id="training:nirs4all.typed-structural",
            plan_id="plan:nirs4all.typed-structural",
            graph=graph,
            campaign=campaign,
            controller_manifests=prepared["manifests"],
            data_identities=identities,
            output_requests=[output],
            seed=prepared["operator_seed"],
            cv_artifacts="discard",
            fitted_artifacts="portable_required",
            cpu_threads=resources.cpu_threads,
            gpu_devices=resources.gpu_devices,
            selection_required_metric_level="sample",
            selection_evaluation_scope="oof",
        )
    )
    request = native.resolve_host_hpo_structural_winner(prepared["request"], evidence, template)
    data_envelopes, _identities = _data_contracts_from_campaign(request["campaign"], envelope)
    influence = _training_influence_manifest(request["graph"], request["campaign"], prepared["folds"], identity, group_by_sample=prepared["groups"], selection_metric="rmse")
    selected_model = next(node for node in request["graph"]["nodes"] if node["kind"] == "model")
    output = _output_request_for_node(request["graph"], selected_model["id"], target_names=["y"])
    result = execute_methods_training(
        spectro=dataset, identity=identity, envelope=envelope, request=request, data_envelopes=data_envelopes, influence=influence, output=output, name=run_options.get("name", ""), binding_source_ids=source_ids(dataset)
    )
    result._dagml_graph = copy.deepcopy(request["graph"])
    result.structural_tuning_training_request = copy.deepcopy(request)
    result.structural_tuning_training_outcome = copy.deepcopy(result._methods_multimodal_outcome.to_dict())
    for metadata in result.per_dataset.values():
        metadata["tuning_profile"] = "structural_typed_modalities_v1"
    return result
