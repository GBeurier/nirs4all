"""Lower a minimal raw-array estimator fit into native DAG-ML training contracts.

P3-R1b covers the first real nirs4all-native fit shape: raw ``X``/``y`` arrays,
a linear nirs4all pipeline with one splitter and one model, and the existing
DAG-ML host node runner.  It deliberately does not cover finetune_params,
branches, augmentation, repetition, conformal calibration or public routing yet.
Unsupported syntax fails before native execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.dagml_bridge import controller_manifests

from .cli_runner import assemble_cv_refit_dsl
from .envelope import build_envelope
from .errors import _reject_multi_model
from .estimator import DagMLPipelineEstimator, DagMLTrainingExecution
from .finetune_lowering import lower_deterministic_finetune_params_to_generators, reject_native_training_param_overrides
from .fit_identity import DagMLFitIdentityFrame
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
        )
        compiler = DagMLTrainingRequestCompiler(
            contracts,
            additional_diagnostics={"nirs4all_lowerer": "raw_array_p3_r1b"},
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
) -> DagMLTrainingRequestContracts:
    """Lower a linear raw-array pipeline into executable DAG-ML contracts."""

    steps, splitter, finetune_overrides = _supported_linear_steps(pipeline)
    selection_metric = finetune_overrides.get("selection_metric", selection_metric)
    selection_objective = finetune_overrides.get("selection_objective", selection_objective)
    dataset = raw_arrays_to_spectro_dataset(X, y, identity_frame=identity_frame)
    identity = identity_from_fit_frame(identity_frame)
    pool = dataset.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, dataset, pool, excluded=set())
    envelope = build_envelope(
        dataset,
        identity,
        sample_ints=pool,
        metadata_by_sample=identity_frame.metadata_by_sample_int(),
        group_by_sample=identity_frame.group_by_sample_int(),
    )
    dag_ml = _import_dagml(dagml_module)
    envelope["relation_fingerprint"] = _core_relation_fingerprint(envelope["coordinator_relations"], dag_ml)
    envelope["data_content_fingerprint"] = _array_content_fingerprint("X", X)
    envelope["target_content_fingerprint"] = _array_content_fingerprint("y", y)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-raw-fit", n_splits=len(folds))
    artifact = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests())
    graph = artifact.graph.to_dict()
    campaign = artifact.campaign_template.to_dict()
    if campaign.get("root_seed") is None:
        campaign["root_seed"] = seed
    data_envelopes, data_identities = _data_contracts_from_campaign(campaign, envelope)
    output_requests = [_default_output_request(graph)]
    request_spec = DagMLTrainingRequestSpec(
        request_id=request_id,
        plan_id=plan_id,
        graph=graph,
        campaign=campaign,
        controller_manifests=controller_manifests(),
        data_identities=data_identities,
        selection_metric=selection_metric,
        selection_objective=selection_objective,
        output_requests=output_requests,
        selection_output_id=output_requests[0]["output_id"],
        seed=int(campaign.get("root_seed") if campaign.get("root_seed") is not None else seed),
        selection_required_metric_level="sample",
        selection_evaluation_scope="oof",
        cv_artifacts="discard",
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
        op_callback=_op_callback(dataset, identity, graph),
        outcome_id=outcome_id,
        run_id=run_id,
        bundle_id=bundle_id,
        diagnostics={"nirs4all_raw_array_samples": identity_frame.n_samples},
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
    reject_native_training_param_overrides(steps, context="native raw-array")
    if splitter is None:
        raise ValueError("RawArrayDagMLTrainingCompiler requires a splitter step")
    _reject_multi_model(steps)
    _assert_supported_operators(steps)
    return steps, splitter, finetune_overrides


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
    model_nodes = [node for node in graph.get("nodes", []) if node.get("kind") == "model"]
    if len(model_nodes) != 1:
        raise ValueError("raw-array lowering requires exactly one model node")
    node_id = model_nodes[0]["id"]
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


def _array_content_fingerprint(label: str, value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    hasher = hashlib.sha256()
    hasher.update(label.encode("utf-8"))
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(array.tobytes())
    return hasher.hexdigest()


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


def _op_callback(dataset: SpectroDataset, identity: IdentityMap, graph: Mapping[str, Any]) -> Any:
    resolver = MaterializationResolver(dataset, identity)
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = graph.get("edges", [])
    y_transform_node = next((node for node in graph["nodes"] if node["kind"] == "y_transform"), None)
    store: dict[int, Any] = {}
    return lambda task: run_node(task, resolver, nodes.__getitem__, store, edges, y_transform_node, None)


def _import_dagml(module_name: str) -> Any:
    import importlib

    return importlib.import_module(module_name)


__all__ = [
    "RawArrayDagMLTrainingCompiler",
    "identity_from_fit_frame",
    "lower_raw_array_training_contracts",
    "raw_arrays_to_spectro_dataset",
]
