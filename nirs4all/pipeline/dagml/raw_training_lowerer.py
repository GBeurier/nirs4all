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
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.transforms.nirs import SavitzkyGolay
from nirs4all.operators.transforms.scalers import StandardNormalVariate
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
    portable_methods: bool = False,
    target_names: Sequence[str] | None = None,
) -> DagMLTrainingRequestContracts:
    """Lower a linear raw-array pipeline into executable DAG-ML contracts."""

    steps, splitter, finetune_overrides = _supported_linear_steps(pipeline)
    native_pls_params = _portable_methods_pls_params(steps) if portable_methods else {}
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
    envelope["data_content_fingerprint"] = (
        _portable_feature_fingerprint(X)
        if portable_methods
        else _array_content_fingerprint("X", X)
    )
    envelope["target_content_fingerprint"] = _array_content_fingerprint("y", y)
    dsl_steps = [steps[-1]] if portable_methods else steps
    dsl = assemble_cv_refit_dsl(dsl_steps, identity, envelope, folds, dsl_id="nirs4all-raw-fit", n_splits=len(folds))
    manifests = controller_manifests()
    if portable_methods:
        _lower_portable_methods_pls_dsl(dsl, native_pls_params)
        manifests = [_portable_methods_pls_manifest()]
    artifact = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, manifests)
    graph = artifact.graph.to_dict()
    campaign = artifact.campaign_template.to_dict()
    if campaign.get("root_seed") is None:
        campaign["root_seed"] = seed
    data_envelopes, data_identities = _data_contracts_from_campaign(campaign, envelope)
    output_requests = [_default_output_request(graph, target_names=target_names)]
    request_spec = DagMLTrainingRequestSpec(
        request_id=request_id,
        plan_id=plan_id,
        graph=graph,
        campaign=campaign,
        controller_manifests=manifests,
        data_identities=data_identities,
        selection_metric=selection_metric,
        selection_objective=selection_objective,
        output_requests=output_requests,
        selection_output_id=output_requests[0]["output_id"],
        seed=int(campaign.get("root_seed") if campaign.get("root_seed") is not None else seed),
        selection_required_metric_level="sample",
        selection_evaluation_scope="oof",
        cv_artifacts="discard",
        prediction_caches="retain",
        fitted_artifacts=(
            "portable_required" if portable_methods else "allow_host_sidecar"
        ),
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
        op_callback=None if portable_methods else _op_callback(dataset, identity, graph),
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


def _portable_methods_pls_params(steps: list[Any]) -> dict[str, Any]:
    """Normalize only raw PLS or canonical SNV -> SG smooth -> PLS."""

    pipeline: dict[str, Any] | None = None
    if len(steps) == 1:
        model_step = steps[0]
    elif len(steps) == 3:
        snv = _exact_public_operator(
            steps[0],
            StandardNormalVariate,
            "StandardNormalVariate",
        )
        savgol = _exact_public_operator(
            steps[1],
            SavitzkyGolay,
            "SavitzkyGolay",
        )
        if (
            type(snv.axis) is not int
            or snv.axis != 1
            or snv.with_mean is not True
            or snv.with_std is not True
            or type(snv.ddof) is not int
            or snv.ddof != 0
            or snv.copy is not True
        ):
            raise ValueError(
                "portable Methods StandardNormalVariate requires exactly "
                "axis=1, with_mean=True, with_std=True, ddof=0, copy=True"
            )
        window = savgol.window_length
        polyorder = savgol.polyorder
        if (
            type(window) is not int
            or not 3 <= window <= 501
            or window % 2 == 0
            or type(polyorder) is not int
            or not 0 <= polyorder < window
            or type(savgol.deriv) is not int
            or savgol.deriv != 0
            or isinstance(savgol.delta, bool)
            or not isinstance(savgol.delta, (int, float))
            or float(savgol.delta) != 1.0
            or savgol.copy is not True
        ):
            raise ValueError(
                "portable Methods SavitzkyGolay requires an odd window_length in 3..501, "
                "0 <= polyorder < window_length, deriv=0, delta=1.0, copy=True "
                "(public defaults are window_length=11, polyorder=3; native mode is interp)"
            )
        pipeline = {
            "schema_version": 1,
            "pipeline_type": "n4m.snv_savgol_smooth.v1",
            "savgol_window": window,
            "savgol_poly_degree": polyorder,
        }
        model_step = steps[2]
    else:
        raise ValueError(
            "portable Methods Archive V2 training supports only PLSRegression or the exact "
            "StandardNormalVariate -> SavitzkyGolay smooth -> PLSRegression order"
        )

    if not isinstance(model_step, Mapping) or set(model_step) != {"model"}:
        raise ValueError(
            "portable Methods Archive V2 training requires exactly one terminal model step"
        )
    model = model_step["model"]
    cls = model if isinstance(model, type) else type(model)
    if cls is not PLSRegression:
        raise ValueError(
            "portable Methods training supports sklearn.cross_decomposition.PLSRegression only"
        )
    if isinstance(model, type):
        model = model()
    default_model = PLSRegression()
    for parameter in ("scale", "max_iter", "tol", "copy"):
        value = getattr(model, parameter)
        default = getattr(default_model, parameter)
        if type(value) is not type(default) or value != default:
            raise ValueError(
                "portable Methods PLS supports only n_components; "
                f"{parameter} must retain its public default {default!r}"
            )
    components = getattr(model, "n_components", None)
    if type(components) is not int or components < 1:
        raise ValueError("portable Methods PLS requires a positive integer n_components")
    params: dict[str, Any] = {"n_components": components}
    if pipeline is not None:
        params["pipeline"] = pipeline
    return params


def _exact_public_operator(value: Any, expected: type[Any], label: str) -> Any:
    """Return one exact public operator instance, materializing its defaults."""

    if value is expected:
        return expected()
    if type(value) is expected:
        return value
    raise ValueError(
        "portable Methods Archive V2 training requires the exact "
        f"StandardNormalVariate -> SavitzkyGolay smooth -> PLSRegression order; got {label} mismatch"
    )


def _lower_portable_methods_pls_dsl(
    dsl: dict[str, Any],
    native_params: Mapping[str, Any],
) -> None:
    """Bind one raw-X model node to the dual-format Methods controller."""

    pipeline = dsl.get("pipeline")
    if not isinstance(pipeline, list) or len(pipeline) != 1 or not isinstance(pipeline[0], dict):
        raise ValueError("portable Methods DSL must contain exactly one model step")
    step = pipeline[0]
    step["params"] = copy.deepcopy(dict(native_params))
    step["metadata"] = {"controller_id": "controller:methods.pls"}


def _portable_methods_pls_manifest() -> dict[str, Any]:
    """Return the public manifest implemented by DAG-ML's Methods runtime."""

    data_requirements = {
        "schema_version": 1,
        "ports": [
            {
                "name": "x",
                "accepted_representations": ["tabular_numeric", "feature_block_set"],
                "accepted_types": ["table", "multi_block"],
                "rank": 2,
                "multi_source": True,
                "optional": False,
            }
        ],
        "default_fusion": {
            "mode": "concatenate_features",
            "alignment": "sample_id",
            "adapter_id": None,
            "params": {"namespace_columns": True},
        },
        "metadata": {"source": "nirs4all-dagml-bridge"},
    }
    return {
        "controller_id": "controller:methods.pls",
        "controller_version": "n4m-abi-2.5",
        "operator_kind": "model",
        "priority": 100,
        "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
        "input_ports": [
            {
                "name": "x",
                "kind": "data",
                "representation": "tabular_numeric",
                "cardinality": "one",
            }
        ],
        "output_ports": [
            {"name": "oof", "kind": "prediction", "representation": None, "cardinality": "one"},
            {"name": "model", "kind": "artifact", "representation": None, "cardinality": "one"},
        ],
        "data_requirements": data_requirements,
        "capabilities": [
            "deterministic",
            "thread_safe",
            "process_safe",
            "emits_predictions",
            "emits_artifacts",
            "stateful",
        ],
        "operator_selectors": [
            {"refs": ["sklearn.cross_decomposition._pls.PLSRegression"]}
        ],
        "fit_scope": "fold_train",
        "rng_policy": "uses_core_seed",
        "artifact_policy": "serializable",
    }


def _portable_feature_fingerprint(value: Any) -> str:
    """Return the Methods provider's canonical little-endian matrix digest."""

    matrix = np.ascontiguousarray(np.asarray(value, dtype=np.dtype("<f8")))
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("portable Methods training requires a non-empty rank-2 X matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("portable Methods training requires finite X values")
    hasher = hashlib.sha256()
    hasher.update(b"n4a-matrix-f64-le.v1\0")
    hasher.update(struct.pack("<QQ", matrix.shape[0], matrix.shape[1]))
    hasher.update(matrix.tobytes(order="C"))
    return hasher.hexdigest()


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


def _default_output_request(
    graph: Mapping[str, Any],
    *,
    target_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    model_nodes = [node for node in graph.get("nodes", []) if node.get("kind") == "model"]
    if len(model_nodes) != 1:
        raise ValueError("raw-array lowering requires exactly one model node")
    node_id = model_nodes[0]["id"]
    names = ["y"] if target_names is None else list(target_names)
    if not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError("raw-array lowering target_names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("raw-array lowering target_names must be unique")
    output: dict[str, Any] = {
        "output_id": "output:prediction",
        "node_id": node_id,
        "prediction_level": "sample",
        "unit_level": "physical_sample",
        "prediction_kind": "regression_point",
        "target_names": names,
        "target_units": [None] * len(names),
        "class_labels": [[] for _ in names],
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
