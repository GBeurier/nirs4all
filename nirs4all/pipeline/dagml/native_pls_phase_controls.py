"""Closed public declarations for the explicitly selected native PLS profile.

This adapter validates declarations and builds signed request inputs. Methods
owns preprocessing, fitting and saved-state inspection; DAG-ML owns phases,
folds, optimization and replay.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from sklearn.cross_decomposition import PLSRegression

from .steps import _split_pipeline

NATIVE_PLS_PHASE_PROFILE = "n4m.pls_role_pipeline.v1"
NATIVE_PLS_PHASE_CONTROLLER = "controller:methods.native.regression"


@dataclass(frozen=True)
class NativePlsPhaseControls:
    """Caller-independent declarations for one closed native campaign."""

    pipeline: list[Any]
    base_params: dict[str, Any]
    train_params: dict[str, Any]
    refit_params: dict[str, Any]
    hpo: Any
    search_axes: tuple[dict[str, Any], ...]
    hpo_scope: str = "campaign"


def _phase_values(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{NATIVE_PLS_PHASE_PROFILE} {label} must be a mapping")
    unknown = set(value) - {"n_components", "scale"}
    if unknown:
        raise ValueError(f"{NATIVE_PLS_PHASE_PROFILE} {label} has unsupported keys: {sorted(unknown, key=str)}")
    result = dict(value)
    if "n_components" in result and (type(result["n_components"]) is not int or not 1 <= result["n_components"] <= 2**31 - 1):
        raise ValueError(f"{label}.n_components must be a positive i32 integer")
    if "scale" in result and type(result["scale"]) is not bool:
        raise TypeError(f"{label}.scale must be a bool; it controls both native scale_x and scale_y")
    return result


def _scale_axis(value: Any) -> dict[str, Any]:
    """Accept the established public categorical spellings, with strict bools."""

    choices = value
    if isinstance(value, Mapping):
        if set(value) != {"type", "choices"} or value["type"] != "categorical":
            raise ValueError("native scale search requires a categorical bool space")
        choices = value["choices"]
    elif isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], str):
        if value[0] not in {"bool", "categorical"}:
            raise ValueError("native scale search requires a categorical bool space")
        choices = value[1]
    if not isinstance(choices, (list, tuple)) or len(choices) != 2 or any(type(item) is not bool for item in choices):
        raise ValueError("native scale search requires exactly the two bool choices False and True")
    if choices[0] == choices[1]:
        raise ValueError("native scale search choices must be distinct")
    # The closed native projection defines categorical index 0/1 as False/True.
    return {"kind": "categorical", "name": "scale", "values": [False, True]}


def normalize_native_pls_phase_controls(pipeline: Any, *, seed: int) -> NativePlsPhaseControls:
    """Validate the entire profile before data access or optimizer allocation."""

    from nirs4all.api.native_archive_training import _extract_portable_methods_hpo, _normalize_n_components_space

    from .raw_training_lowerer import _portable_methods_pls_params

    if not isinstance(pipeline, list):
        raise TypeError(f"{NATIVE_PLS_PHASE_PROFILE} requires a list pipeline")
    steps, splitter = _split_pipeline(pipeline)
    if splitter is None or not steps:
        raise ValueError(f"{NATIVE_PLS_PHASE_PROFILE} requires a splitter and terminal PLS model")
    terminal = steps[-1]
    if not isinstance(terminal, Mapping) or "model" not in terminal:
        raise ValueError(f"{NATIVE_PLS_PHASE_PROFILE} requires a terminal model declaration")
    unknown = set(terminal) - {"model", "train_params", "refit_params", "finetune_params"}
    if unknown:
        raise ValueError(f"{NATIVE_PLS_PHASE_PROFILE} model step has unsupported keys: {sorted(unknown, key=str)}")
    model = terminal["model"]
    if model is PLSRegression:
        model = PLSRegression()
    if type(model) is not PLSRegression:
        raise ValueError(f"{NATIVE_PLS_PHASE_PROFILE} requires an exact sklearn PLSRegression declaration")
    if type(model.scale) is not bool:
        raise TypeError("PLSRegression.scale must be a bool")
    _phase_values({"n_components": model.n_components}, "model")
    if model.copy is not True or type(model.max_iter) is not int or model.max_iter != 500:
        raise ValueError("native PLS phase controls require copy=True and max_iter=500")
    if isinstance(model.tol, bool) or not isinstance(model.tol, (int, float)) or model.tol != 1.0e-6:
        raise ValueError("native PLS phase controls require tol=1e-6")

    # Reuse the historical preprocessing validator. The new profile alone
    # executes scale through Methods; the historical profile keeps its defaults.
    canonical_model = PLSRegression(n_components=model.n_components)
    canonical_steps = [*steps[:-1], {"model": canonical_model}]
    base_params = _portable_methods_pls_params(canonical_steps)
    base_params.update(native_profile=NATIVE_PLS_PHASE_PROFILE, scale=model.scale)
    train = _phase_values(terminal.get("train_params", {}), "train_params")
    refit = _phase_values(terminal.get("refit_params", {}), "refit_params")

    hpo = None
    hpo_scope = "campaign"
    axes: list[dict[str, Any]] = []
    if "finetune_params" in terminal:
        options = terminal["finetune_params"]
        if not isinstance(options, Mapping):
            raise TypeError("native PLS finetune_params must be a mapping")
        hpo_scope = options.get("scope", "campaign")
        if not isinstance(hpo_scope, str) or hpo_scope not in ("campaign", "fold"):
            raise ValueError("native PLS finetune_params.scope must be 'campaign' or 'fold'")
        if hpo_scope == "fold":
            from .native_pls_fold_hpo import validate_fold_splitter

            validate_fold_splitter(splitter)
        space = options.get("model_params")
        if not isinstance(space, Mapping) or not space or set(space) - {"n_components", "scale"}:
            raise ValueError("native PLS search supports only n_components and/or scale")
        collision = set(space) & set(train)
        if collision:
            raise ValueError(f"native PLS train/search parameter ownership overlaps: {sorted(collision)}")
        for name in sorted(space):
            if name == "scale":
                axes.append(_scale_axis(space[name]))
            else:
                bounds = _normalize_n_components_space(space[name])
                if bounds != (1, 3, 1):
                    raise ValueError("native PLS n_components search requires exactly ['int', 1, 3, 1]")
                axes.append({"kind": "int", "name": "n_components", "low": 1, "high": 3, "step": 1, "log": False})
        # Optimizer options and complete-package resume retain the existing
        # native validator. Only the explicitly versioned space is different.
        validation_options = {**dict(options), "model_params": {"n_components": ["int", 1, 3, 1]}}
        validation_options.pop("scope", None)
        resume = validation_options.pop("resume_package", None) if hpo_scope == "fold" else None
        _, hpo = _extract_portable_methods_hpo(
            [splitter, *steps[:-1], {"model": canonical_model, "finetune_params": validation_options}], seed=seed,
        )
        if hpo_scope == "fold":
            from .native_pls_fold_hpo import normalize_fold_resume_package

            assert hpo is not None  # The validated terminal declares finetune_params.
            hpo = replace(hpo, resume_package_json=normalize_fold_resume_package(resume))
    return NativePlsPhaseControls(
        pipeline=[splitter, *canonical_steps], base_params=copy.deepcopy(base_params),
        train_params=train, refit_params=refit, hpo=hpo, search_axes=tuple(axes), hpo_scope=hpo_scope,
    )


def attach_native_pls_phase_controls(contracts: Any, profile: NativePlsPhaseControls, dag_ml: Any, *, fold_plan: Any = None) -> Any:
    """Bind the closed profile and phase patches before native request signing."""

    helper = getattr(dag_ml, "methods_pls_role_pipeline_contract", None)
    if not callable(helper):
        raise RuntimeError("installed dag-ml lacks the n4m.pls_role_pipeline.v1 native controller contract")
    native_contract = helper(copy.deepcopy(profile.base_params))
    if not isinstance(native_contract, Mapping) or not isinstance(native_contract.get("manifest"), Mapping):
        raise RuntimeError("native PLS controller returned an invalid contract")
    spec = contracts.request_spec
    graph = copy.deepcopy(dict(spec.graph))
    nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict) and node.get("kind") == "model"]
    if len(nodes) != 1:
        raise ValueError("native PLS phase controls require exactly one model node")
    target = nodes[0]
    target["operator"] = copy.deepcopy(native_contract["operator"])
    target["params"] = copy.deepcopy(profile.base_params)
    target.setdefault("metadata", {})["controller_id"] = NATIVE_PLS_PHASE_CONTROLLER
    manifest = copy.deepcopy(dict(native_contract["manifest"]))
    if manifest.get("controller_id") != NATIVE_PLS_PHASE_CONTROLLER:
        raise RuntimeError("native PLS controller identity differs from the selected profile")
    manifests = [manifest]
    campaign = copy.deepcopy(dict(spec.campaign))
    patches = [
        {"schema_version": 1, "node_id": target["id"], "namespace": "fit", "path": [name], "value": copy.deepcopy(values)}
        for name, values in (("refit_params", profile.refit_params), ("train_params", profile.train_params)) if values
    ]
    if profile.hpo is not None:
        hpo = profile.hpo
        tuner = copy.deepcopy(manifest)
        tuner.update(controller_id="controller:tuner.methods", operator_kind="tuner", input_ports=[], output_ports=[])
        manifests.append(tuner)
        operation = {
            "schema_version": 2, "native_profile": NATIVE_PLS_PHASE_PROFILE,
            "operation_id": "hpo:nirs4all.native.pls_phase",
            "study": {
                "controller_id": "controller:tuner.methods", "study_id": "study:nirs4all.native.pls_phase",
                "methods_abi": "n4m-abi-2.14", "search_space": {"parameters": copy.deepcopy(list(profile.search_axes))},
                "optimizer": {
                    "sampler": hpo.sampler, "pruner": hpo.pruner, "direction": "minimize", "metric": "rmse",
                    "seed": hpo.seed, "n_startup_trials": hpo.n_startup_trials, "max_resource": 0, "reduction_factor": 0,
                },
            },
            "trials": hpo.trials, "target_node_id": target["id"],
            "parameter_paths": {axis["name"]: axis["name"] for axis in profile.search_axes},
        }
        if profile.hpo_scope == "fold":
            if fold_plan is None:
                raise ValueError("native fold HPO requires identity-bound outer and inner folds")
            split = campaign.get("split_invocation")
            if not isinstance(split, dict) or split.get("fold_set") is None:
                raise ValueError("native fold HPO requires a materialized outer FoldSet")
            split["fold_set"] = copy.deepcopy(fold_plan.outer_fold_set)
            operation.update(
                schema_version=3, scope="fold", operation_id="hpo:nirs4all.native.pls_fold",
                inner_fold_sets=copy.deepcopy(fold_plan.inner_fold_sets),
                refit_inner_fold_set=copy.deepcopy(fold_plan.refit_inner_fold_set),
            )
            operation["study"]["study_id"] = "study:nirs4all.native.pls_fold"
        if hpo.resume_package_json is not None:
            operation["resume_package_json"] = hpo.resume_package_json
        campaign.setdefault("metadata", {})["methods_hpo_operation"] = operation
    return replace(
        contracts,
        request_spec=replace(
            spec, graph=graph, campaign=campaign, controller_manifests=manifests,
            parameter_patches=patches,
            patch_policies=[{"node_id": target["id"], "allowed_namespaces": ["operator", "fit"]}] if patches else [],
        ),
        diagnostics={**dict(contracts.diagnostics or {}), "nirs4all_native_profile": NATIVE_PLS_PHASE_PROFILE},
    )
