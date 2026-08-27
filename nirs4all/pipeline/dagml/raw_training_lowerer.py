"""Lower raw-array native Methods training into attested DAG-ML contracts.

The portable lane deliberately accepts only the Methods shapes that have an
equivalent controller-owned execution: a linear PLS model, or an exact
nested-OOF stack of two-or-more PLS base models followed by native Ridge.  In
particular, Ridge consumes scheduler-delivered OOF values only; the raw
provider matrix remains an identity/target attestation and never becomes a
meta-model feature matrix.  Every wider branch, transform, estimator, or
meta-model configuration fails before native execution.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.dagml_bridge import _model_controller_id, _model_data_requirements, controller_manifests

from .cli_runner import assemble_cv_refit_dsl, data_bindings_for_nodes, split_invocation_for
from .detect import _detect_stacking_branch
from .envelope import build_envelope, build_fold_set
from .errors import _reject_multi_model
from .estimator import DagMLPipelineEstimator, DagMLTrainingExecution
from .finetune_lowering import lower_deterministic_finetune_params_to_generators, reject_native_training_param_overrides
from .fit_identity import DagMLFitIdentityFrame, feature_content_fingerprint, target_content_fingerprint
from .folds import _build_folds
from .identity import IdentityMap, SampleIdentity
from .node_runner import run_node
from .resolver import MaterializationResolver
from .steps import _assert_supported_operators, _split_pipeline
from .training_compiler import DagMLTrainingRequestCompiler, DagMLTrainingRequestContracts
from .training_contracts import (
    DagMLTrainingRequestSpec,
    tcv1_fingerprint_without,
    training_data_identity_from_binding,
)


@dataclass(frozen=True)
class RawArrayDagMLTrainingCompiler:
    """Compile a minimal raw-array fit into native DAG-ML training contracts."""

    selection_metric: str = "rmse"
    selection_objective: str = "minimize"
    request_id: str = "training:nirs4all.raw_fit"
    plan_id: str = "plan:nirs4all.raw_fit"
    outcome_id: str = "outcome:nirs4all.raw_fit"
    run_id: str = "run:nirs4all.raw_fit"
    bundle_id: str = "bundle:nirs4all.raw_fit"
    seed: int = 12345
    dagml_module: str = "dag_ml"
    training_losses: tuple[Mapping[str, Any], ...] = ()
    local_implementations: Any = None
    methods_library_path: str | None = None
    methods_hpo_operation: Mapping[str, Any] | None = None
    additional_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def compile_fit(
        self,
        estimator: DagMLPipelineEstimator,
        X: Any,
        y: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
        identity_frame: DagMLFitIdentityFrame,
    ) -> DagMLTrainingExecution:
        """Compile one estimator fit into signed native execution inputs."""

        _ = (sample_ids, groups, metadata)
        contracts = lower_raw_array_training_contracts(
            estimator.pipeline,
            X,
            y,
            identity_frame=identity_frame,
            selection_metric=self.selection_metric,
            selection_objective=self.selection_objective,
            request_id=self.request_id,
            plan_id=self.plan_id,
            outcome_id=self.outcome_id,
            run_id=self.run_id,
            bundle_id=self.bundle_id,
            seed=self.seed,
            dagml_module=self.dagml_module,
            training_losses=self.training_losses,
            local_implementations=self.local_implementations,
            methods_library_path=self.methods_library_path,
            methods_hpo_operation=self.methods_hpo_operation,
        )
        compiler = DagMLTrainingRequestCompiler(
            contracts,
            additional_diagnostics={
                "nirs4all_lowerer": "raw_array_p3_r1b",
                **dict(self.additional_diagnostics),
            },
            dagml_module=self.dagml_module,
        )
        return compiler.compile_fit(
            estimator,
            X,
            y,
            sample_ids=identity_frame.sample_ids,
            groups=identity_frame.groups,
            metadata=identity_frame.metadata_by_sample_id(),
            identity_frame=identity_frame,
        )


def lower_raw_array_training_contracts(
    pipeline: Any,
    X: Any,
    y: Any,
    *,
    identity_frame: DagMLFitIdentityFrame,
    selection_metric: str = "rmse",
    selection_objective: str = "minimize",
    request_id: str = "training:nirs4all.raw_fit",
    plan_id: str = "plan:nirs4all.raw_fit",
    outcome_id: str = "outcome:nirs4all.raw_fit",
    run_id: str = "run:nirs4all.raw_fit",
    bundle_id: str = "bundle:nirs4all.raw_fit",
    seed: int = 12345,
    dagml_module: str = "dag_ml",
    training_losses: tuple[Mapping[str, Any], ...] = (),
    local_implementations: Any = None,
    methods_library_path: str | None = None,
    methods_hpo_operation: Mapping[str, Any] | None = None,
) -> DagMLTrainingRequestContracts:
    """Lower a linear raw-array pipeline into executable DAG-ML contracts."""

    steps, splitter, finetune_overrides = _supported_linear_steps(pipeline)
    portable_methods = methods_library_path is not None
    portable_methods_stacking = None
    if portable_methods:
        portable_methods_stacking = _portable_methods_stacking(steps)
        if portable_methods_stacking is None:
            _validate_portable_methods_pipeline(steps)
    selection_metric = finetune_overrides.get("selection_metric", selection_metric)
    selection_objective = finetune_overrides.get("selection_objective", selection_objective)
    dataset = raw_arrays_to_spectro_dataset(X, y, identity_frame=identity_frame)
    identity = identity_from_fit_frame(identity_frame)
    pool = dataset.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, dataset, pool, excluded=set())
    if portable_methods_stacking is not None:
        _require_partitioned_outer_stacking(identity, folds)
    envelope = build_envelope(
        dataset,
        identity,
        sample_ints=pool,
        metadata_by_sample=identity_frame.metadata_by_sample_int(),
        group_by_sample=identity_frame.group_by_sample_int(),
    )
    dag_ml = _import_dagml(dagml_module)
    envelope["relation_fingerprint"] = _core_relation_fingerprint(envelope["coordinator_relations"], dag_ml)
    envelope["data_content_fingerprint"] = feature_content_fingerprint(X)
    envelope["target_content_fingerprint"] = target_content_fingerprint(y)
    if portable_methods and portable_methods_stacking is not None:
        dsl = _portable_methods_stacking_dsl(
            portable_methods_stacking,
            identity=identity,
            envelope=envelope,
            folds=folds,
            splitter=splitter,
        )
        manifests = [_portable_methods_pls_manifest(), _portable_methods_ridge_manifest()]
    else:
        dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-raw-fit", n_splits=len(folds))
        manifests = controller_manifests()
        if portable_methods:
            _lower_portable_methods_pls_dsl(dsl)
            manifests = [_portable_methods_pls_manifest()]
    artifact = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, manifests)
    graph = artifact.graph.to_dict()
    if portable_methods:
        if portable_methods_stacking is None:
            _mark_portable_methods_pls_graph(graph)
        else:
            _mark_portable_methods_stacking_graph(graph, portable_methods_stacking)
    campaign = artifact.campaign_template.to_dict()
    output_requests = [
        _output_request_for_node(
            graph,
            portable_methods_stacking.meta_node_id if portable_methods_stacking is not None else None,
        )
    ]
    if methods_hpo_operation is not None:
        if not portable_methods:
            raise ValueError("Methods HPO requires the portable Methods execution lane")
        if portable_methods_stacking is not None:
            raise ValueError("Methods HPO does not yet tune a portable PLS-to-Ridge stacking topology")
        campaign.setdefault("metadata", {})["methods_hpo_operation"] = _bind_methods_hpo_target(
            methods_hpo_operation,
            output_requests[0]["node_id"],
        )
    if campaign.get("root_seed") is None:
        campaign["root_seed"] = seed
    data_envelopes, data_identities = _data_contracts_from_campaign(campaign, envelope)
    request_spec = DagMLTrainingRequestSpec(
        request_id=request_id,
        plan_id=plan_id,
        graph=graph,
        campaign=campaign,
        controller_manifests=manifests,
        data_identities=data_identities,
        training_losses=training_losses,
        selection_metric=selection_metric,
        selection_objective=selection_objective,
        output_requests=output_requests,
        selection_output_id=output_requests[0]["output_id"],
        seed=int(campaign.get("root_seed") if campaign.get("root_seed") is not None else seed),
        selection_required_metric_level="sample",
        selection_evaluation_scope="oof",
        cv_artifacts="discard",
        prediction_caches="retain",
        fitted_artifacts="portable_required" if portable_methods else "allow_host_sidecar",
    )
    training_influence = _training_influence_manifest(
        graph,
        campaign,
        folds,
        identity,
        group_by_sample=identity_frame.group_by_sample_int(),
        selection_metric=selection_metric,
    )
    return DagMLTrainingRequestContracts(
        request_spec=request_spec,
        data_envelopes=data_envelopes,
        relations=copy.deepcopy(envelope["coordinator_relations"]),
        training_influence=training_influence,
        op_callback=None if portable_methods else _op_callback(dataset, identity, graph, local_implementations),
        outcome_id=outcome_id,
        run_id=run_id,
        bundle_id=bundle_id,
        diagnostics={"nirs4all_raw_array_samples": identity_frame.n_samples},
        methods_inputs=(
            _methods_inputs_from_arrays(X, y, identity_frame, data_envelopes)
            if portable_methods
            else None
        ),
        methods_library_path=methods_library_path,
    )


def raw_arrays_to_spectro_dataset(
    X: Any,
    y: Any,
    *,
    identity_frame: DagMLFitIdentityFrame,
    name: str = "nirs4all_raw_fit",
) -> SpectroDataset:
    """Build the minimal ``SpectroDataset`` representation for raw estimator arrays."""

    features = np.asarray(X)
    targets = np.asarray(y)
    if features.ndim != 2:
        raise ValueError(f"native raw-array lowering requires 2D X, got {features.ndim}D")
    if len(features) != identity_frame.n_samples:
        raise ValueError("X row count does not match the normalized fit identity frame")
    dataset = SpectroDataset(name)
    headers = [f"f{index}" for index in range(features.shape[1])]
    dataset.add_samples(features, {"partition": "train"}, headers=headers, header_unit="index")
    dataset.add_targets(targets)
    return dataset


def identity_from_fit_frame(identity_frame: DagMLFitIdentityFrame) -> IdentityMap:
    """Create a DAG-ML identity map preserving explicit normalized sample ids."""

    identities = tuple(
        SampleIdentity(
            sample_int=index,
            origin_int=index,
            observation_id=sample_id,
            sample_id=sample_id,
            augmented=False,
        )
        for index, sample_id in enumerate(identity_frame.sample_ids)
    )
    return IdentityMap(
        fingerprint=identity_frame.fingerprint,
        identities=identities,
        _to_int={identity.observation_id: identity.sample_int for identity in identities},
        _to_wire={identity.sample_int: identity.observation_id for identity in identities},
    )


def _supported_linear_steps(pipeline: Any) -> tuple[list[Any], Any, dict[str, str]]:
    if not isinstance(pipeline, list):
        raise TypeError("RawArrayDagMLTrainingCompiler requires a list pipeline")
    steps, splitter = _split_pipeline(pipeline)
    steps, finetune_overrides = lower_deterministic_finetune_params_to_generators(
        steps,
        context="native raw-array",
    )
    model_steps = [step for step in steps if isinstance(step, dict) and "model" in step]
    allowed_keys = frozenset({"train_params"}) if len(model_steps) == 1 and _model_controller_id(model_steps[0]["model"]) is not None else frozenset()
    reject_native_training_param_overrides(
        steps,
        context="native raw-array",
        allowed_keys=allowed_keys,
    )
    if splitter is None:
        raise ValueError("RawArrayDagMLTrainingCompiler requires a splitter step")
    _reject_multi_model(steps)
    _assert_supported_operators(steps)
    return steps, splitter, finetune_overrides


def _validate_portable_methods_pipeline(steps: list[Any]) -> None:
    """Refuse every raw fit shape the Methods PLS lane cannot represent exactly."""

    if len(steps) != 1 or not isinstance(steps[0], Mapping) or set(steps[0]) - {"model"}:
        raise ValueError("portable Methods training currently supports exactly one PLSRegression model step after the splitter")
    model = steps[0].get("model")
    cls = model if isinstance(model, type) else type(model)
    if getattr(cls, "__name__", None) != "PLSRegression" or not str(getattr(cls, "__module__", "")).startswith("sklearn.cross_decomposition"):
        raise ValueError("portable Methods training currently supports sklearn.cross_decomposition.PLSRegression only")
    components = getattr(model, "n_components", None)
    if not isinstance(components, int) or isinstance(components, bool) or components < 1:
        raise ValueError("portable Methods PLS requires a positive integer n_components")


@dataclass(frozen=True)
class _PortableMethodsStacking:
    """Closed native stack lowering shape, derived before any scheduler work."""

    branches: tuple[int, ...]
    ridge_lambda: float
    meta_node_id: str = "merge:stack"


def _portable_methods_stacking(steps: list[Any]) -> _PortableMethodsStacking | None:
    """Recognize the exact PLS×N → nested-OOF → default Ridge native topology.

    The broad DAG-ML branch detector intentionally admits transform-bearing and
    Python-controller stacks.  This native lane narrows it further: each base
    branch is one PLS model, and Ridge keeps every sklearn parameter at its
    constructor default apart from finite non-negative ``alpha``.  That makes
    the ``alpha`` → native ``ridge_lambda`` mapping explicit rather than
    silently ignoring intercept/solver/copy policies that libn4m does not own.
    """

    detected = _detect_stacking_branch(steps)
    if detected is None:
        return None
    branches, ridge = detected
    if len(branches) < 2:
        raise ValueError("portable Methods stacking requires at least two PLS base branches")
    components: list[int] = []
    for index, branch in enumerate(branches):
        if len(branch) != 1 or not isinstance(branch[0], Mapping) or set(branch[0]) != {"model"}:
            raise ValueError(
                "portable Methods stacking requires every base branch to contain exactly one "
                "PLSRegression model (no transforms or policy keys)"
            )
        model = branch[0]["model"]
        cls = model if isinstance(model, type) else type(model)
        if (
            getattr(cls, "__name__", None) != "PLSRegression"
            or not str(getattr(cls, "__module__", "")).startswith("sklearn.cross_decomposition")
        ):
            raise ValueError(f"portable Methods stacking base branch {index} requires sklearn.cross_decomposition.PLSRegression")
        n_components = getattr(model, "n_components", None)
        if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
            raise ValueError(f"portable Methods stacking base branch {index} requires a positive integer PLS n_components")
        components.append(n_components)

    ridge_cls = ridge if isinstance(ridge, type) else type(ridge)
    if (
        getattr(ridge_cls, "__name__", None) != "Ridge"
        or not str(getattr(ridge_cls, "__module__", "")).startswith("sklearn.linear_model")
        or not callable(getattr(ridge, "get_params", None))
    ):
        raise ValueError("portable Methods stacking requires sklearn.linear_model.Ridge as its meta-model")
    alpha = getattr(ridge, "alpha", None)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("portable Methods Ridge requires finite non-negative scalar alpha")
    try:
        expected = ridge_cls(alpha=float(alpha)).get_params(deep=False)
        actual = ridge.get_params(deep=False)
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("portable Methods Ridge must expose a reconstructible sklearn parameter mapping") from error
    if actual != expected:
        raise ValueError(
            "portable Methods Ridge supports only default sklearn Ridge options plus alpha; "
            "other solver/intercept/copy policies are not silently lowered"
        )
    return _PortableMethodsStacking(branches=tuple(components), ridge_lambda=float(alpha))


def _require_partitioned_outer_stacking(
    identity: IdentityMap, folds: list[tuple[list[int], list[int]]]
) -> None:
    """Require exactly one outer OOF row per sample before lowering a stack."""

    fold_set = build_fold_set(identity, folds, set_id="folds.raw_methods.stacking.outer")
    if fold_set.get("partition_mode") == "resampled":
        raise ValueError(
            "portable Methods nested stacking requires an outer CV partition with exactly one validation "
            "prediction per sample; repeated or ShuffleSplit CV is not portable"
        )


def _portable_methods_stacking_dsl(
    stacking: _PortableMethodsStacking,
    *,
    identity: IdentityMap,
    envelope: Mapping[str, Any],
    folds: list[tuple[list[int], list[int]]],
    splitter: Any,
) -> dict[str, Any]:
    """Build the canonical nested-OOF DAG that the Ridge controller consumes."""

    base_node_ids = [f"branch:{index}.node:0" for index in range(len(stacking.branches))]
    pls_operator = {"class": "sklearn.cross_decomposition._pls.PLSRegression"}
    branches = [
        {
            "id": f"branch_{index}",
            "steps": [
                {
                    "kind": "model",
                    "id": node_id,
                    "operator": pls_operator,
                    "params": {"n_components": components},
                    "metadata": {"controller_id": "controller:methods.pls"},
                }
            ],
        }
        for index, (node_id, components) in enumerate(zip(base_node_ids, stacking.branches, strict=True))
    ]
    # The graph compiler owns construction of the OOF edges and the inner
    # fold sets.  The native ridge controller receives only those declared
    # prediction inputs and an x binding for identity/target attestation.
    return {
        "id": "nirs4all-raw-methods-stacking",
        "inner_cv": {"kind": "kfold", "n_splits": 2, "shuffle": False, "seed": None},
        "steps": [
            {"kind": "branch", "mode": "duplication", "branches": branches},
            {
                "kind": "merge_model",
                "id": stacking.meta_node_id,
                "operator": {"class": "sklearn.linear_model._ridge.Ridge"},
                "params": {"ridge_lambda": stacking.ridge_lambda},
                "metadata": {
                    "controller_id": "controller:methods.ridge",
                    "stacking_oof_execution": "nested_oof_v1",
                    "stacking_oof_refit_contract": {"policy": "require_full_coverage"},
                },
            },
        ],
        "data_bindings": _portable_methods_stacking_data_bindings(
            base_node_ids,
            stacking.meta_node_id,
            dict(envelope),
        ),
        "split_invocation": split_invocation_for(
            identity,
            folds,
            n_splits=len(folds),
            shuffle=bool(getattr(splitter, "shuffle", True)),
        ),
    }


def _portable_methods_stacking_data_bindings(
    base_node_ids: list[str], meta_node_id: str, envelope: dict[str, Any]
) -> list[dict[str, Any]]:
    """Bind base PLS nodes on ``x`` and Ridge attestation on canonical ``x_original``."""

    bindings = data_bindings_for_nodes(base_node_ids, envelope)
    meta_binding = data_bindings_for_nodes([meta_node_id], envelope)[0]
    meta_binding["input_name"] = "x_original"
    return [*bindings, meta_binding]


def _lower_portable_methods_pls_dsl(dsl: dict[str, Any]) -> None:
    """Bind the single model node to the typed Methods controller, never Python."""

    pipeline = dsl.get("pipeline")
    if not isinstance(pipeline, list) or len(pipeline) != 1 or not isinstance(pipeline[0], dict):
        raise ValueError("portable Methods DSL must contain exactly one model step")
    step = pipeline[0]
    params = step.get("params")
    if not isinstance(params, Mapping) or not isinstance(params.get("n_components"), int):
        raise ValueError("portable Methods DSL requires integer PLS n_components")
    # `model` stays a transparent declaration for graph provenance; the
    # controller id is explicit and scheduler-owned, so no Python import or
    # sklearn callback can become executable at runtime.
    step["metadata"] = {"controller_id": "controller:methods.pls"}


def _portable_methods_pls_manifest() -> dict[str, Any]:
    """The public controller declaration matching DAG-ML's Methods PLS runtime."""

    return {
        "controller_id": "controller:methods.pls",
        "controller_version": "n4m-abi-2.2",
        "operator_kind": "model",
        "priority": 100,
        "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
        "input_ports": [{"name": "x", "kind": "data", "representation": "tabular_numeric", "cardinality": "one"}],
        "output_ports": [
            {"name": "oof", "kind": "prediction", "representation": None, "cardinality": "one"},
            {"name": "model", "kind": "artifact", "representation": None, "cardinality": "one"},
        ],
        "data_requirements": _model_data_requirements(),
        "capabilities": [
            "deterministic",
            "thread_safe",
            "process_safe",
            "emits_predictions",
            "emits_artifacts",
            "stateful",
        ],
        "operator_selectors": [{"refs": ["sklearn.cross_decomposition._pls.PLSRegression"]}],
        "fit_scope": "fold_train",
        "rng_policy": "uses_core_seed",
        "artifact_policy": "serializable",
    }


def _portable_methods_ridge_manifest() -> dict[str, Any]:
    """Controller contract for a native Ridge meta-model over scheduler OOF blocks."""

    return {
        "controller_id": "controller:methods.ridge",
        "controller_version": "n4m-abi-2.3",
        "operator_kind": "model",
        "priority": 101,
        "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
        "input_ports": [
            {"name": "x_original", "kind": "data", "representation": "tabular_numeric", "cardinality": "one"},
            {"name": "oof", "kind": "prediction", "representation": None, "cardinality": "many"},
        ],
        "output_ports": [
            {"name": "oof", "kind": "prediction", "representation": None, "cardinality": "one"},
            {"name": "model", "kind": "artifact", "representation": None, "cardinality": "one"},
        ],
        "data_requirements": _model_data_requirements(),
        "capabilities": [
            "deterministic",
            "thread_safe",
            "process_safe",
            "emits_predictions",
            "emits_artifacts",
            "stateful",
            "consumes_oof_predictions",
        ],
        "operator_selectors": [{"refs": ["sklearn.linear_model._ridge.Ridge"]}],
        "fit_scope": "fold_train",
        "rng_policy": "uses_core_seed",
        "artifact_policy": "serializable",
    }


def _mark_portable_methods_pls_graph(graph: Mapping[str, Any]) -> None:
    """Make the model's executable operator the native portable ``pls`` one.

    Controller metadata alone is insufficient for Methods HPO: its scheduler
    validates both the controller and the target's transparent operator
    declaration.  This edit happens before request signing, so it remains
    attested provenance rather than a post-selection execution rewrite.
    """

    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("portable Methods graph must contain a node list")
    models = [node for node in nodes if isinstance(node, dict) and node.get("kind") == "model"]
    if len(models) != 1:
        raise ValueError("portable Methods graph must contain exactly one model node")
    models[0]["operator"] = "pls"


def _mark_portable_methods_stacking_graph(
    graph: Mapping[str, Any], stacking: _PortableMethodsStacking
) -> None:
    """Mark all executable stack nodes as their controller-owned native ops."""

    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("portable Methods stacking graph must contain a node list")
    models = {
        node.get("id"): node
        for node in nodes
        if isinstance(node, dict) and node.get("kind") == "model" and isinstance(node.get("id"), str)
    }
    base_ids = {f"branch:{index}.node:0" for index in range(len(stacking.branches))}
    expected_ids = base_ids | {stacking.meta_node_id}
    if set(models) != expected_ids:
        raise ValueError("portable Methods stacking graph does not contain the exact declared PLS and Ridge nodes")
    for node_id in base_ids:
        models[node_id]["operator"] = "pls"
        models[node_id]["metadata"] = {"controller_id": "controller:methods.pls"}
    meta = models[stacking.meta_node_id]
    meta["operator"] = "ridge"
    meta["metadata"] = {
        "controller_id": "controller:methods.ridge",
        "stacking_oof_execution": "nested_oof_v1",
        "stacking_oof_refit_contract": {"policy": "require_full_coverage"},
    }


def _bind_methods_hpo_target(operation: Mapping[str, Any], target_node_id: str) -> dict[str, Any]:
    """Attach the only graph-derived field of a native Methods HPO operation.

    The tuner is a scheduler operation, not a graph node.  Keeping the target
    binding here means the public native API never asks callers to guess an
    internal node identifier, while DAG-ML still signs and validates the full
    descriptor before it creates a native optimizer session.
    """

    if not isinstance(operation, Mapping):
        raise TypeError("methods_hpo_operation must be a mapping")
    bound = copy.deepcopy(dict(operation))
    if "target_node_id" in bound:
        raise ValueError("methods_hpo_operation must not supply target_node_id")
    bound["target_node_id"] = target_node_id
    return bound


def _methods_inputs_from_arrays(
    X: Any,
    y: Any,
    identity_frame: DagMLFitIdentityFrame,
    binding_keys: str | Mapping[str, Any],
) -> dict[str, Any]:
    """Create host-owned numeric inputs for every explicitly bound model node."""

    features = np.asarray(X, dtype=float)
    targets = np.asarray(y, dtype=float)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    if targets.ndim != 2 or targets.shape[1] != 1:
        raise ValueError("portable Methods PLS currently supports exactly one numeric target")
    keys = [binding_keys] if isinstance(binding_keys, str) else list(binding_keys)
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("portable Methods training requires one non-empty host input per data binding")
    template = {
        "sample_ids": list(identity_frame.sample_ids),
        "x": features.tolist(),
        "y": targets.tolist(),
        "target_names": ["y"],
    }
    return {key: copy.deepcopy(template) for key in keys}


def _data_contracts_from_campaign(
    campaign: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bindings = [binding for node_bindings in campaign.get("data_bindings", {}).values() for binding in node_bindings]
    data_envelopes = {f"{binding['node_id']}.{binding['input_name']}": dict(envelope) for binding in bindings}
    data_identities = [
        training_data_identity_from_binding(
            binding,
            data_content_fingerprint=envelope["data_content_fingerprint"],
            target_content_fingerprint=envelope["target_content_fingerprint"],
        )
        for binding in bindings
    ]
    return data_envelopes, data_identities


def _default_output_request(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Backward-compatible output selection for one-model native lowering."""

    return _output_request_for_node(graph)


def _output_request_for_node(graph: Mapping[str, Any], node_id: str | None = None) -> dict[str, Any]:
    """Return the sole requested prediction output, optionally for a declared meta node."""

    model_nodes = [node for node in graph.get("nodes", []) if node.get("kind") == "model"]
    if node_id is None:
        if len(model_nodes) != 1:
            raise ValueError("raw-array lowering requires exactly one model node")
        node_id = model_nodes[0]["id"]
    elif node_id not in {node.get("id") for node in model_nodes}:
        raise ValueError("raw-array lowering selected an output node absent from the compiled model graph")
    output: dict[str, Any] = {
        "output_id": "output:prediction",
        "node_id": node_id,
        "prediction_level": "sample",
        "unit_level": "physical_sample",
        "prediction_kind": "regression_point",
        "target_names": ["y"],
        "target_units": [None],
        "class_labels": [[]],
        "output_order": "target_order",
        "target_space": "raw",
    }
    return output


def _training_influence_manifest(
    graph: Mapping[str, Any],
    campaign: Mapping[str, Any],
    folds: list[tuple[list[int], list[int]]],
    identity: IdentityMap,
    *,
    group_by_sample: Mapping[int, str],
    selection_metric: str,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for node in graph.get("nodes", []):
        kind = node.get("kind")
        if kind not in {"transform", "model"}:
            continue
        influence_kind = "model_fit" if kind == "model" else "transform_fit"
        node_id = node["id"]
        for index, (train_ints, _validation_ints) in enumerate(folds):
            entries.append(_influence_entry(influence_kind, f"fit_cv:fold{index}", node_id, train_ints, identity, group_by_sample))
        entries.append(_influence_entry(influence_kind, "refit:full", node_id, sorted({sample for fold in folds for side in fold for sample in side}), identity, group_by_sample))
    entries.append(
        _influence_entry(
            "hpo_selection",
            f"select:selection:{selection_metric}",
            None,
            sorted({sample for fold in folds for side in fold for sample in side}),
            identity,
            group_by_sample,
        )
    )
    entries.sort(key=lambda entry: (_INFLUENCE_KIND_ORDER[entry["kind"]], entry["scope_id"], entry["node_id"] or ""))
    manifest = {
        "schema_version": 1,
        "relation_fingerprint": _first_relation_fingerprint(campaign),
        "entries": entries,
        "manifest_fingerprint": "0" * 64,
    }
    manifest["manifest_fingerprint"] = tcv1_fingerprint_without(manifest, "manifest_fingerprint")
    return manifest


def _core_relation_fingerprint(relations: Mapping[str, Any], dag_ml: Any) -> str:
    fingerprint = getattr(dag_ml, "sample_relation_set_fingerprint_json", None)
    if callable(fingerprint):
        return cast(str, fingerprint(json.dumps(relations, sort_keys=True, separators=(",", ":"))))
    raise RuntimeError("dag_ml.sample_relation_set_fingerprint_json is required for native raw-array training contracts")


_INFLUENCE_KIND_ORDER = {
    "transform_fit": 0,
    "model_fit": 1,
    "hpo_selection": 2,
    "early_stopping": 3,
    "weighting_resampling": 4,
    "trained_meta_aggregation": 5,
}


def _influence_entry(
    kind: str,
    scope_id: str,
    node_id: str | None,
    sample_ints: list[int],
    identity: IdentityMap,
    group_by_sample: Mapping[int, str],
) -> dict[str, Any]:
    return {
        "kind": kind,
        "scope_id": scope_id,
        "node_id": node_id,
        "physical_sample_ids": sorted(identity.to_wire(sample_int) for sample_int in sample_ints),
        "origin_sample_ids": [],
        "group_ids": sorted({group_by_sample[sample_int] for sample_int in sample_ints if sample_int in group_by_sample}),
    }


def _first_relation_fingerprint(campaign: Mapping[str, Any]) -> str:
    for node_bindings in campaign.get("data_bindings", {}).values():
        for binding in node_bindings:
            return cast(str, binding["relation_fingerprint"])
    raise ValueError("campaign contains no data binding relation fingerprint")


def _op_callback(
    dataset: SpectroDataset,
    identity: IdentityMap,
    graph: Mapping[str, Any],
    local_implementations: Any,
) -> Any:
    resolver = MaterializationResolver(dataset, identity)
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = graph.get("edges", [])
    y_transform_node = next((node for node in graph["nodes"] if node["kind"] == "y_transform"), None)
    store: dict[int, Any] = {}
    return lambda task: run_node(
        task,
        resolver,
        nodes.__getitem__,
        store,
        edges,
        y_transform_node,
        None,
        local_implementations,
    )


def _import_dagml(module_name: str) -> Any:
    import importlib

    return importlib.import_module(module_name)


__all__ = [
    "RawArrayDagMLTrainingCompiler",
    "identity_from_fit_frame",
    "lower_raw_array_training_contracts",
    "raw_arrays_to_spectro_dataset",
]
