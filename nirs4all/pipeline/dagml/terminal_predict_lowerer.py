"""Closed raw-array lowerer for the strict Methods terminal prediction facade.

This module intentionally does not reuse ``RawArrayDagMLTrainingCompiler``.
The published DAG-ML terminal facade accepts a smaller contract than the
ordinary portable Methods lane: one raw numeric PLS model, an explicit
unshuffled KFold, internal ephemeral OOF scoring, one full refit, and an
X-only terminal cohort.  Keeping that lowering separate makes it impossible
for transforms, callbacks, HPO, calibration, group metadata, or retained OOF
state to leak into the terminal call by accident.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from .fit_identity import (
    DagMLFitIdentityFrame,
    DagMLPredictIdentityFrame,
    feature_content_fingerprint,
    normalize_fit_identity,
    normalize_predict_identity,
    target_content_fingerprint,
)
from .native_client import DagMLNativeCoverageError
from .training_contracts import (
    DagMLTrainingRequestSpec,
    assemble_training_request,
    tcv1_fingerprint_without,
    tcv1_sha256,
    training_data_identity_from_binding,
)

_NODE_ID = "model:terminal"
_TERMINAL_PORT = "oof"
_TARGET_NAMES = ["y"]
_OUTPUT_ID = "output:prediction"
_PROFILE = "nirs4all.strict_methods_terminal.v1"


@dataclass(frozen=True)
class StrictMethodsTerminalPredictionExecution:
    """Fully preflighted inputs for one callback-free terminal facade call.

    There is deliberately no ``op_callback`` field.  The executor transports
    only the returned contract bundle to
    ``dag_ml.execute_methods_cv_refit_terminal_predict``.

    The identity frames record the canonical input-row order for presentation
    by explicit sample id.  They are local ordering data only: neither they
    nor any mapping in this object attests a terminal prediction.  The native
    terminal result and its frozen receipt remain the sole authority.
    """

    request: Mapping[str, Any]
    data_envelopes: Mapping[str, Mapping[str, Any]]
    relations: Mapping[str, Any]
    training_influence: Mapping[str, Any]
    methods_inputs: Mapping[str, Any]
    predict_envelope: Mapping[str, Any]
    predict_input: Mapping[str, Any]
    outcome_id: str
    run_id: str
    bundle_id: str
    package_id: str
    terminal_node_id: str
    terminal_port: str
    fit_identity_frame: DagMLFitIdentityFrame
    predict_identity_frame: DagMLPredictIdentityFrame
    n_features: int


def lower_strict_methods_terminal_prediction(
    pipeline: list[Any],
    X: Any,
    y: Any,
    *,
    sample_ids: Sequence[Any],
    terminal_predict: Mapping[str, Any],
    seed: int,
    dagml_module: str = "dag_ml",
) -> StrictMethodsTerminalPredictionExecution:
    """Lower exactly one raw PLS/KFold terminal prediction request.

    Validation and canonical row ordering finish before a Methods runtime can
    be configured.  DAG-ML then remains the authority that validates the
    signed request, native data inputs, CV/refit, Package V2 and the closed
    terminal receipt.
    """

    if not isinstance(dagml_module, str) or not dagml_module:
        raise ValueError("strict terminal prediction requires a non-empty dagml_module")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TypeError("strict terminal prediction requires a non-negative integer seed")

    splitter, model = _strict_pipeline(pipeline)
    features, targets, fit_identity = _canonical_training_arrays(X, y, sample_ids=sample_ids)
    predict_features, predict_identity = _canonical_predict_arrays(terminal_predict)
    if predict_features.shape[1] != features.shape[1]:
        raise ValueError(
            "strict terminal prediction requires matching train and terminal_predict X feature widths"
        )

    n_components = int(model.n_components)
    folds = _kfolds(splitter, features, fit_identity.sample_ids)
    dag_ml = _import_dagml(dagml_module)
    relation_records = _raw_relation_records(fit_identity.sample_ids)
    relations: dict[str, Any] = {"records": relation_records}
    relation_fingerprint = _relation_fingerprint(dag_ml, relations)
    schema_fingerprint, plan_fingerprint = _schema_and_plan_fingerprints(features.shape[1])
    binding = _binding(schema_fingerprint, plan_fingerprint, relation_fingerprint)
    envelope = _training_envelope(
        schema_fingerprint,
        plan_fingerprint,
        relation_fingerprint,
        relations,
        features,
        targets,
    )
    request = _signed_request(
        dag_ml,
        n_components=n_components,
        seed=seed,
        folds=folds,
        binding=binding,
        envelope=envelope,
    )
    training_influence = _training_influence(folds, relation_fingerprint)
    methods_key = f"{_NODE_ID}.x"
    methods_inputs = {
        methods_key: {
            "sample_ids": list(fit_identity.sample_ids),
            "x": features.tolist(),
            "y": targets.tolist(),
            "target_names": list(_TARGET_NAMES),
        }
    }
    predict_envelope = _predict_envelope(
        dag_ml,
        schema_fingerprint=schema_fingerprint,
        plan_fingerprint=plan_fingerprint,
        relation_fingerprint=relation_fingerprint,
        training_relations=relations,
        features=predict_features,
        sample_ids=predict_identity.sample_ids,
    )
    predict_input = {
        "sample_ids": list(predict_identity.sample_ids),
        "x": predict_features.tolist(),
        "target_names": list(_TARGET_NAMES),
    }
    identity_suffix = tcv1_sha256(
        {
            "profile": _PROFILE,
            "fit_identity": fit_identity.fingerprint,
            "predict_identity": predict_identity.fingerprint,
            "n_components": n_components,
            "n_splits": len(folds),
        }
    )[:16]
    return StrictMethodsTerminalPredictionExecution(
        request=request,
        data_envelopes={methods_key: envelope},
        relations=relations,
        training_influence=training_influence,
        methods_inputs=methods_inputs,
        predict_envelope=predict_envelope,
        predict_input=predict_input,
        outcome_id=f"outcome:nirs4all.strict_terminal.{identity_suffix}",
        run_id=f"run:nirs4all.strict_terminal.{identity_suffix}",
        bundle_id=f"bundle:nirs4all.strict_terminal.{identity_suffix}",
        package_id=f"package:nirs4all.strict_terminal.{identity_suffix}",
        terminal_node_id=_NODE_ID,
        terminal_port=_TERMINAL_PORT,
        fit_identity_frame=fit_identity,
        predict_identity_frame=predict_identity,
        n_features=int(features.shape[1]),
    )


def _strict_pipeline(pipeline: list[Any]) -> tuple[KFold, PLSRegression]:
    """Accept only one exact unshuffled ``KFold`` and default-shape PLS."""

    if not isinstance(pipeline, list) or len(pipeline) != 2:
        raise ValueError(
            "strict terminal prediction requires exactly one KFold splitter and one PLSRegression model"
        )
    splitter = next((step for step in pipeline if type(step) is KFold), None)
    model_step = next((step for step in pipeline if isinstance(step, Mapping) and set(step) == {"model"}), None)
    if splitter is None or model_step is None:
        raise ValueError(
            "strict terminal prediction requires exactly one KFold splitter and one {'model': PLSRegression(...)} step"
        )
    model = model_step["model"]
    if type(model) is not PLSRegression:
        raise ValueError("strict terminal prediction supports sklearn.cross_decomposition.PLSRegression only")
    if splitter.shuffle is not False or splitter.random_state is not None:
        raise ValueError("strict terminal prediction requires KFold(shuffle=False, random_state=None)")
    if isinstance(splitter.n_splits, bool) or not isinstance(splitter.n_splits, int) or splitter.n_splits < 2:
        raise ValueError("strict terminal prediction requires KFold(n_splits >= 2)")
    components = model.n_components
    if isinstance(components, bool) or not isinstance(components, int) or components < 1:
        raise ValueError("strict terminal prediction requires a positive integer PLSRegression.n_components")
    expected = PLSRegression(n_components=components).get_params(deep=False)
    if model.get_params(deep=False) != expected:
        raise ValueError(
            "strict terminal prediction supports only default PLSRegression options plus n_components"
        )
    return splitter, model


def _canonical_training_arrays(
    X: Any,
    y: Any,
    *,
    sample_ids: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, DagMLFitIdentityFrame]:
    """Validate and order target-bound rows by their explicit sample identity."""

    features = _numeric_matrix(X, label="X")
    raw_targets = np.asarray(y)
    if raw_targets.ndim == 1:
        raw_targets = raw_targets.reshape(-1, 1)
    if raw_targets.ndim != 2 or raw_targets.shape[1] != 1:
        raise ValueError("strict terminal prediction requires exactly one numeric target column")
    if raw_targets.shape[0] != features.shape[0]:
        raise ValueError("strict terminal prediction requires aligned X and y rows")
    if not np.issubdtype(raw_targets.dtype, np.number):
        raise TypeError("strict terminal prediction requires numeric y")
    if np.issubdtype(raw_targets.dtype, np.complexfloating):
        raise TypeError("strict terminal prediction requires real numeric y; complex values are unsupported")
    targets = np.ascontiguousarray(raw_targets, dtype=np.float64)
    if not np.isfinite(targets).all():
        raise ValueError("strict terminal prediction requires finite y")
    identity = normalize_fit_identity(
        features,
        targets,
        sample_ids=sample_ids,
        require_explicit_sample_ids=True,
    )
    order = _canonical_order(identity.sample_ids)
    row_indices = np.asarray(order, dtype=int)
    ordered_features = np.ascontiguousarray(features[row_indices], dtype=np.float64)
    ordered_targets = np.ascontiguousarray(targets[row_indices], dtype=np.float64)
    ordered_ids = tuple(identity.sample_ids[index] for index in order)
    ordered_identity = normalize_fit_identity(
        ordered_features,
        ordered_targets,
        sample_ids=ordered_ids,
        require_explicit_sample_ids=True,
    )
    return ordered_features, ordered_targets, ordered_identity


def _canonical_predict_arrays(
    terminal_predict: Mapping[str, Any],
) -> tuple[np.ndarray, DagMLPredictIdentityFrame]:
    """Validate and order the separate target-free terminal cohort."""

    if not isinstance(terminal_predict, Mapping):
        raise TypeError("terminal_predict must be {'X': matrix, 'sample_ids': explicit_ids}")
    unknown = set(terminal_predict) - {"X", "sample_ids"}
    if unknown:
        raise ValueError(
            "terminal_predict supports only target-free X/sample_ids; unsupported keys: "
            f"{sorted(unknown)}"
        )
    missing = {"X", "sample_ids"} - set(terminal_predict)
    if missing:
        raise ValueError(f"terminal_predict is missing required keys: {sorted(missing)}")
    features = _numeric_matrix(terminal_predict["X"], label="terminal_predict.X")
    identity = normalize_predict_identity(
        features,
        sample_ids=terminal_predict["sample_ids"],
        require_explicit_sample_ids=True,
    )
    order = _canonical_order(identity.sample_ids)
    ordered_features = np.ascontiguousarray(features[np.asarray(order, dtype=int)], dtype=np.float64)
    ordered_ids = tuple(identity.sample_ids[index] for index in order)
    ordered_identity = normalize_predict_identity(
        ordered_features,
        sample_ids=ordered_ids,
        require_explicit_sample_ids=True,
    )
    return ordered_features, ordered_identity


def _numeric_matrix(value: Any, *, label: str) -> np.ndarray:
    """Return a finite, non-empty, raw numeric two-dimensional array."""

    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"strict terminal prediction requires non-empty 2-D {label}")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"strict terminal prediction requires numeric {label}")
    if np.issubdtype(matrix.dtype, np.complexfloating):
        raise TypeError(f"strict terminal prediction requires real numeric {label}; complex values are unsupported")
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"strict terminal prediction requires finite {label}")
    return cast(np.ndarray, matrix)


def _canonical_order(sample_ids: Sequence[str]) -> tuple[int, ...]:
    """Canonicalize a cohort as the explicit ``(sample_id, original_row)`` order.

    The returned original-row positions preserve a deterministic mapping from
    the caller's rows to the canonical identity-frame order.  This is only an
    input-ordering rule; it is not a prediction receipt or an attestation.
    """

    return tuple(index for index, _sample_id in sorted(enumerate(sample_ids), key=lambda pair: (pair[1], pair[0])))


def _kfolds(splitter: KFold, features: np.ndarray, sample_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Materialize the only admitted outer split before native execution."""

    try:
        partitions = list(splitter.split(features))
    except ValueError as error:
        raise ValueError("strict terminal prediction KFold cannot split the supplied training rows") from error
    folds: list[dict[str, Any]] = []
    for index, (train_indices, validation_indices) in enumerate(partitions):
        folds.append(
            {
                "fold_id": f"fold{index}",
                "train_sample_ids": [sample_ids[int(row)] for row in train_indices],
                "validation_sample_ids": [sample_ids[int(row)] for row in validation_indices],
                "metadata": {},
            }
        )
    return folds


def _raw_relation_records(sample_ids: Sequence[str]) -> list[dict[str, str]]:
    """Return the raw, group-free relation records the strict facade admits."""

    return [{"observation_id": sample_id, "sample_id": sample_id} for sample_id in sample_ids]


def _schema_and_plan_fingerprints(n_features: int) -> tuple[str, str]:
    """Derive explicit static raw-array schema/plan identities for one width."""

    schema = tcv1_sha256(
        {
            "profile": f"{_PROFILE}.schema",
            "feature_width": n_features,
            "target_names": _TARGET_NAMES,
        }
    )
    plan = tcv1_sha256(
        {
            "profile": f"{_PROFILE}.plan",
            "feature_width": n_features,
            "output_representation": "tabular_numeric",
        }
    )
    return schema, plan


def _relation_fingerprint(dag_ml: Any, relations: Mapping[str, Any]) -> str:
    """Use DAG-ML's public relation fingerprint implementation exactly once."""

    fingerprint = getattr(dag_ml, "sample_relation_set_fingerprint_json", None)
    if not callable(fingerprint):
        raise DagMLNativeCoverageError(
            "installed DAG-ML lacks sample_relation_set_fingerprint_json required by strict terminal prediction"
        )
    value = fingerprint(json.dumps(relations, sort_keys=True, separators=(",", ":")))
    if not isinstance(value, str) or len(value) != 64:
        raise DagMLNativeCoverageError("DAG-ML returned an invalid strict terminal relation fingerprint")
    return value


def _binding(schema_fingerprint: str, plan_fingerprint: str, relation_fingerprint: str) -> dict[str, Any]:
    return {
        "node_id": _NODE_ID,
        "input_name": "x",
        "request_id": "plan:nirs4all.strict_methods_terminal",
        "schema_fingerprint": schema_fingerprint,
        "plan_fingerprint": plan_fingerprint,
        "relation_fingerprint": relation_fingerprint,
        "output_representation": "tabular_numeric",
        "feature_set_id": "x",
        "source_ids": ["strict"],
        "require_relations": True,
        "metadata": {},
        "view_policy": {
            "fit_partition": "fold_train",
            "predict_partition": "fold_validation",
            "include_augmented_train": False,
            "include_augmented_validation": False,
            "include_excluded": False,
            "require_sample_ids": True,
        },
    }


def _training_envelope(
    schema_fingerprint: str,
    plan_fingerprint: str,
    relation_fingerprint: str,
    relations: Mapping[str, Any],
    features: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "schema_fingerprint": schema_fingerprint,
        "plan_fingerprint": plan_fingerprint,
        "relation_fingerprint": relation_fingerprint,
        "data_content_fingerprint": feature_content_fingerprint(features),
        "target_content_fingerprint": target_content_fingerprint(targets),
        "coordinator_relations": dict(relations),
    }


def _signed_request(
    dag_ml: Any,
    *,
    n_components: int,
    seed: int,
    folds: Sequence[Mapping[str, Any]],
    binding: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the closed single-node request and sign it with DAG-ML."""

    aggregation = {
        "aggregation_level": "sample",
        "method": "mean",
        "weights": "none",
        "emit_parallel_metrics": True,
        "selection_metric_level": "sample",
        "store_raw_predictions": True,
        "store_aggregated_predictions": True,
    }
    leakage = {
        "split_unit": "sample",
        "forbid_origin_cross_fold": True,
        "allow_observation_split_with_shared_target": False,
        "require_group_ids": False,
        "unsafe_flags": [],
    }
    shape_plan = {
        "node_id": _NODE_ID,
        "input_granularity": "sample",
        "target_granularity": "sample",
        "fit_rows": "fold_train",
        "predict_rows": "fold_validation",
        "feature_namespace": "model_input",
        "feature_schema_fingerprint": None,
        "target_space": "raw",
        "aggregation_policy": aggregation,
        "augmentation_policy": {
            "sample_scope": "none",
            "feature_scope": "none",
            "require_origin_id": True,
            "inherit_group": True,
            "inherit_target": True,
            "unsafe_flags": [],
        },
        "selection_policy": {
            "scope": "none",
            "store_masks": True,
            "allow_schema_mismatch_on_join": False,
        },
    }
    fold_set = {
        "id": "folds:nirs4all.strict_methods_terminal",
        "sample_ids": [record["observation_id"] for record in envelope["coordinator_relations"]["records"]],
        "folds": [dict(fold) for fold in folds],
        "sample_groups": {},
    }
    campaign = {
        "id": "campaign:nirs4all.strict_methods_terminal",
        "root_seed": seed,
        "leakage_policy": leakage,
        "aggregation_policy": aggregation,
        "split_invocation": {
            "id": "split:outer",
            "controller_id": None,
            "leakage_policy": leakage,
            "params": {"kind": "kfold", "n_splits": len(folds), "shuffle": False},
            "fold_set": fold_set,
        },
        "generation": {"strategy": "none", "dimensions": [], "max_variants": 1},
        "shape_plans": {_NODE_ID: shape_plan},
        "data_bindings": {_NODE_ID: [dict(binding)]},
        "metadata": {},
    }
    graph = {
        "id": "nirs4all.strict_methods_terminal",
        "interface": {
            "inputs": [
                {
                    "name": "x",
                    "kind": "data",
                    "representation": "tabular_numeric",
                    "cardinality": "one",
                    "description": "",
                }
            ],
            "outputs": [
                {
                    "name": "prediction",
                    "kind": "prediction",
                    "representation": None,
                    "cardinality": "one",
                    "description": "",
                }
            ],
        },
        "nodes": [
            {
                "id": _NODE_ID,
                "kind": "model",
                "operator": {"type": "PLSRegression"},
                "params": {"n_components": n_components},
                "ports": {
                    "inputs": [
                        {
                            "name": "x",
                            "kind": "data",
                            "representation": "tabular_numeric",
                            "cardinality": "one",
                            "description": "",
                        }
                    ],
                    "outputs": [
                        {
                            "name": _TERMINAL_PORT,
                            "kind": "prediction",
                            "representation": None,
                            "cardinality": "one",
                            "description": "",
                        }
                    ],
                },
                "metadata": {},
                "seed_label": None,
            }
        ],
        "edges": [],
        "search_space_fingerprint": None,
        "metadata": {},
    }
    output: dict[str, Any] = {
        "output_id": _OUTPUT_ID,
        "node_id": _NODE_ID,
        "prediction_level": "sample",
        "unit_level": "physical_sample",
        "prediction_kind": "regression_point",
        "target_names": list(_TARGET_NAMES),
        "target_units": [None],
        "class_labels": [[]],
        "output_order": "target_order",
        "target_space": "raw",
    }
    manifest = {
        "controller_id": "controller:methods.pls",
        "controller_version": "n4m-abi-2.3",
        "operator_kind": "model",
        "priority": 100,
        "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
        "input_ports": [
            {
                "name": "x",
                "kind": "data",
                "representation": "tabular_numeric",
                "cardinality": "one",
                "description": "",
            }
        ],
        "output_ports": [
            {
                "name": _TERMINAL_PORT,
                "kind": "prediction",
                "representation": None,
                "cardinality": "one",
                "description": "",
            }
        ],
        "data_requirements": None,
        "capabilities": [
            "deterministic",
            "thread_safe",
            "process_safe",
            "uses_core_rng",
            "emits_predictions",
            "emits_artifacts",
            "stateful",
        ],
        "fit_scope": "fold_train",
        "rng_policy": "uses_core_seed",
        "artifact_policy": "serializable",
    }
    identity = training_data_identity_from_binding(
        binding,
        data_content_fingerprint=str(envelope["data_content_fingerprint"]),
        target_content_fingerprint=str(envelope["target_content_fingerprint"]),
    )
    request = assemble_training_request(
        DagMLTrainingRequestSpec(
            request_id="training:nirs4all.strict_methods_terminal",
            plan_id="plan:nirs4all.strict_methods_terminal",
            graph=graph,
            campaign=campaign,
            controller_manifests=[manifest],
            data_identities=[identity],
            selection_metric="rmse",
            selection_objective="minimize",
            selection_output_id=_OUTPUT_ID,
            output_requests=[output],
            seed=seed,
            selection_required_metric_level="sample",
            selection_evaluation_scope="oof",
            cv_artifacts="discard",
            prediction_caches="discard",
            fitted_artifacts="portable_required",
        )
    )
    signer = getattr(dag_ml, "sign_training_request", None)
    if not callable(signer):
        raise DagMLNativeCoverageError(
            "installed DAG-ML lacks sign_training_request required by strict terminal prediction"
        )
    signed = signer(request)
    document = signed.to_dict() if hasattr(signed, "to_dict") else signed
    if not isinstance(document, Mapping):
        raise DagMLNativeCoverageError("DAG-ML returned an invalid signed strict terminal request")
    return dict(document)


def _training_influence(
    folds: Sequence[Mapping[str, Any]],
    relation_fingerprint: str,
) -> dict[str, Any]:
    """Create the complete signed influence manifest for CV, SELECT and REFIT."""

    entries: list[dict[str, Any]] = []
    all_sample_ids: set[str] = set()
    for fold in folds:
        train_ids = sorted(str(sample_id) for sample_id in fold["train_sample_ids"])
        validation_ids = [str(sample_id) for sample_id in fold["validation_sample_ids"]]
        all_sample_ids.update(train_ids)
        all_sample_ids.update(validation_ids)
        entries.append(
            {
                "kind": "model_fit",
                "scope_id": f"fit_cv:{fold['fold_id']}",
                "node_id": _NODE_ID,
                "physical_sample_ids": train_ids,
                "origin_sample_ids": [],
                "group_ids": [],
            }
        )
    full_ids = sorted(all_sample_ids)
    entries.extend(
        [
            {
                "kind": "model_fit",
                "scope_id": "refit:full",
                "node_id": _NODE_ID,
                "physical_sample_ids": full_ids,
                "origin_sample_ids": [],
                "group_ids": [],
            },
            {
                "kind": "hpo_selection",
                "scope_id": "select:selection:rmse",
                "node_id": None,
                "physical_sample_ids": full_ids,
                "origin_sample_ids": [],
                "group_ids": [],
            },
        ]
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "relation_fingerprint": relation_fingerprint,
        "entries": entries,
        "manifest_fingerprint": "0" * 64,
    }
    manifest["manifest_fingerprint"] = tcv1_fingerprint_without(manifest, "manifest_fingerprint")
    return manifest


def _predict_envelope(
    dag_ml: Any,
    *,
    schema_fingerprint: str,
    plan_fingerprint: str,
    relation_fingerprint: str,
    training_relations: Mapping[str, Any],
    features: np.ndarray,
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    """Attach the target-free V2 cohort through DAG-ML's official helper."""

    attach = getattr(dag_ml, "attach_predict_cohort_to_envelope", None)
    if not callable(attach):
        raise DagMLNativeCoverageError(
            "installed DAG-ML lacks attach_predict_cohort_to_envelope required by strict terminal prediction"
        )
    fingerprint = feature_content_fingerprint(features)
    base_envelope = {
        "schema_version": 1,
        "schema_fingerprint": schema_fingerprint,
        "plan_fingerprint": plan_fingerprint,
        "relation_fingerprint": relation_fingerprint,
        "data_content_fingerprint": fingerprint,
        "target_content_fingerprint": None,
        "coordinator_relations": dict(training_relations),
    }
    cohort_request = {
        "role": "inference",
        "relations": {"records": _raw_relation_records(sample_ids)},
        "target_names": list(_TARGET_NAMES),
        "data_content_fingerprint": fingerprint,
    }
    attached = attach(base_envelope, cohort_request)
    document = attached.to_dict() if hasattr(attached, "to_dict") else attached
    if not isinstance(document, Mapping):
        raise DagMLNativeCoverageError("DAG-ML returned an invalid V2 strict terminal predict cohort")
    return dict(document)


def _import_dagml(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:  # pragma: no cover - environment-specific dependency failure
        raise DagMLNativeCoverageError(
            f"strict terminal prediction cannot import the configured DAG-ML facade '{module_name}'"
        ) from error


__all__ = ["StrictMethodsTerminalPredictionExecution", "lower_strict_methods_terminal_prediction"]
