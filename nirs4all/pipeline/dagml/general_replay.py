"""DAG-owned inference from an already captured, trusted REFIT artifact.

This module accepts live objects only. Archive callers must verify the original
artifact bytes before deserialization and retain that archive's provenance.
No training outcome or portable predictor package is invented for host models.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

import numpy as np

from nirs4all.pipeline.dagml_bridge import controller_manifests, named_model_input_spec, pipeline_to_dsl

from .cli_runner import data_bindings_for
from .envelope import build_envelope
from .errors import DagMlUnavailable, DagMlUnsupported
from .identity import mint_identity
from .node_runner import (
    _build_result,
    _MultiBlockEstimator,
    _source_index,
    _SourceConcatEstimator,
    _train_predict_ids,
    resolve_named_model_features,
    validate_named_refit_origin,
)
from .public_normalization import normalize_model_steps
from .raw_training_lowerer import _array_content_fingerprint
from .resolver import MaterializationResolver
from .steps import _apply_model_params, _split_pipeline


def predict_captured_artifact(
    artifact: dict[str, Any], spectro: Any, *, pipeline: list[Any], target_names: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict every input row through a native PREDICT-only execution plan.

    ``artifact`` contains the fitted ``estimator`` (including its X-chain) and
    optional ``y_transform`` captured at REFIT. ``pipeline`` is the corresponding
    concrete training topology, including an optional training-only splitter.
    Target names come from training metadata, defaulting to ``['y']`` for one
    target. Input labels are never required or used: this is inference, not a
    new validation experiment. Returned arrays follow input storage order;
    evidence retains the unmodified native identity-keyed execution results.
    """
    import dag_ml

    estimator, y_transform = artifact["estimator"], artifact.get("y_transform")
    from nirs4all.api.result import _DagmlExportedModel, _DagmlNativeStackingModel

    # General ``.n4a`` archives store the public wrapper as their sole joblib
    # member. Unwrap it before entering the numeric-only DAG callback; public
    # label decoding is applied after the native result has been validated.
    if isinstance(estimator, _DagmlExportedModel) and y_transform is None:
        estimator, y_transform = estimator.estimator, estimator.y_transform
    from .named_torch_estimator import DagMLNamedTorchEstimator

    named_origin = None
    named_bundle = {**artifact, "estimator": estimator, "y_transform": y_transform}
    if isinstance(estimator, DagMLNamedTorchEstimator):
        named_origin = validate_named_refit_origin(named_bundle, artifact)
    from .multimodal_contracts import validate_input_contract, validate_late_partial_stack

    late_contract = validate_late_partial_stack(estimator, artifact.get("late_partial_refit_contract"))
    if late_contract is not None and artifact.get("late_partial_refit_contract") is not None:
        from .tuning_contracts import tcv1_sha256

        if (artifact.get("late_partial_refit_fingerprint") != tcv1_sha256(late_contract)
                or artifact.get("content_fingerprint") != artifact["late_partial_refit_fingerprint"]):
            raise ValueError("late partial replay changed its original archive REFIT closure fingerprint")

    validate_input_contract(estimator, spectro)
    from .torch_topology_replay import CapturedTorchTopology, replay_torch_topology

    if isinstance(estimator, CapturedTorchTopology):
        if y_transform is not None or (target_names is not None and target_names != estimator.package["effective_plan"]["graph_plan"]["graph"]["metadata"]["python_torch_profile"]["target_names"]):
            raise ValueError("Torch topology replay requires its signed mono-y target contract")
        return replay_torch_topology(estimator, spectro)

    from .target_capture import CapturedTargetTransform

    public_target_transform = y_transform if isinstance(y_transform, CapturedTargetTransform) else None
    runtime_target_transform = public_target_transform.transformer if public_target_transform is not None else y_transform
    if isinstance(estimator, _DagmlNativeStackingModel):
        # Members inverse their own numeric target transforms while forming
        # meta-features; decode only the final output after native validation.
        meta_transform = estimator.meta_member.y_transform
        if isinstance(meta_transform, CapturedTargetTransform):
            public_target_transform = meta_transform
        runtime_target_transform = None
    if not callable(getattr(estimator, "predict", None)):
        raise ValueError("captured artifact must contain a fitted prediction estimator")
    names = ["y"] if target_names is None else list(target_names)
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("prediction target names must be nonempty and unique")
    if late_contract is not None and (names != late_contract["target_names"]
                                     or list(spectro.source_names) != late_contract["source_order"]):
        raise ValueError("late partial replay requires its original target names and ordered named sources")
    execute = getattr(importlib.import_module("dag_ml._dag_ml"), "execute_phase_in_process", None)
    cohort_builder = getattr(dag_ml, "attach_predict_cohort_to_envelope", None)
    if not callable(execute) or not callable(cohort_builder):
        raise DagMlUnavailable("captured-artifact replay requires the qualified DAG-ML PREDICT phase and cohort APIs")
    if named_origin is not None:
        # General archives carry a cosmetic exported wrapper as their pipeline.
        # Compile the preserved native owner declaration, never a replacement
        # factory inferred from the replay cohort or current wrapper defaults.
        original_node = named_origin["graph_node"]
        original_metadata = {key: value for key, value in original_node.get("metadata", {}).items() if key != "dsl_model_input"}
        dsl = {
            "schema_version": 1, "id": "nirs4all-captured-replay",
            "root_seed": named_origin["effective_seed"],
            "steps": [{"kind": "model", "id": named_origin["node_id"], "operator": original_node["operator"],
                       "params": named_origin["params"], "model_input": named_origin["model_input"], "metadata": original_metadata}],
        }
    else:
        steps, _ = _split_pipeline(normalize_model_steps(pipeline))
        dsl = pipeline_to_dsl(_apply_model_params(steps), "nirs4all-captured-replay")
    named_input = named_model_input_spec(dsl)
    if named_origin is not None and named_input != named_origin["model_input"]:
        raise ValueError("named Torch replay requires its original captured input contract")
    manifests = controller_manifests(dsl) if named_input is not None else controller_manifests()
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, manifests).graph.to_dict()
    models = [node for node in graph["nodes"] if node["kind"] == "model"]
    if len(models) != 1 or any(node["kind"] not in {"model", "transform", "y_transform"} for node in graph["nodes"]):
        raise DagMlUnsupported("one captured artifact requires its concrete single-model replay topology")
    model_node = models[0]
    model_id = model_node["id"]
    if named_origin is not None and (model_node != named_origin["graph_node"] or names != named_origin["target_names"]):
        raise ValueError("named Torch replay graph or targets disagree with its completed REFIT owner")
    identity = mint_identity(spectro)
    storage_ids = identity.observation_ids()
    if not storage_ids:
        raise ValueError("prediction input must contain at least one row")
    metadata_key = getattr(estimator, "metadata_key", None)
    metadata_by_id = None
    if isinstance(metadata_key, str):
        try:
            metadata_values = spectro.metadata_column(metadata_key, {})
        except (KeyError, ValueError) as exc:
            raise ValueError(f"captured predictor requires metadata column {metadata_key!r}") from exc
        if len(metadata_values) != len(storage_ids):
            raise ValueError(f"captured predictor requires metadata column {metadata_key!r} for every input row")
        metadata_by_id = dict(zip(storage_ids, metadata_values, strict=True))
    source_metadata = None
    if late_contract is not None:
        from .source_missing import prediction_source_presence_metadata

        source_metadata = prediction_source_presence_metadata(spectro, list(range(len(storage_ids))))
    envelope = build_envelope(spectro, identity, metadata_by_sample=source_metadata)
    envelope.update(cohort_builder(envelope, {
        "role": "inference", "relations": envelope["coordinator_relations"], "target_names": names,
        "data_content_fingerprint": _prediction_content_fingerprint(spectro),
        "target_content_fingerprint": None,
    }).to_dict())
    dsl["data_bindings"] = data_bindings_for(model_id, envelope, model_input=named_input)
    resolver = MaterializationResolver(spectro, identity)
    fixed_views = None
    if named_input is not None:
        from .fixed_cohort_views import FixedCohortViewStore

        fixed_views = FixedCohortViewStore(resolver, named_input, envelope)

    def callback(task: dict[str, Any]) -> dict[str, Any]:
        if task["phase"] != "PREDICT":
            raise ValueError("captured-artifact replay cannot execute a training phase")
        if late_contract is not None:
            validate_late_partial_stack(estimator, late_contract)
        if task["node_plan"]["node_id"] != model_id:
            # The captured sklearn estimator already owns its fitted X-chain.
            # As in node_runner, transform nodes only transmit native handles.
            return _build_result(task, [], [], {})
        _, ids = _train_predict_ids(task)
        source_index = _source_index(model_node)
        options: dict[str, Any] = {}
        bound_views = fixed_views.bind_task(task) if fixed_views is not None else None
        if named_input is not None:
            if named_origin is None:
                raise ValueError("named model replay requires its original completed REFIT artifact")
            validate_named_refit_origin(named_bundle, artifact)
            x = resolve_named_model_features(task, resolver, named_input, ids, "predict", bound_views)
        elif isinstance(estimator, _MultiBlockEstimator) or (isinstance(estimator, _DagmlNativeStackingModel) and estimator.source_names is not None) or (
            isinstance(estimator, _SourceConcatEstimator) and resolver.is_multi_source()
        ):
            resolved = resolver.resolve_feature_blocks(
                ids, include_augmented=False, source_names=getattr(estimator, "source_names", None),
            )
            x = resolved["blocks"]
            if "source_masks" in resolved:
                if not isinstance(estimator, (_MultiBlockEstimator, _DagmlNativeStackingModel)):
                    raise ValueError("partial modalities require a multimodal model with an explicit missing_source_policy")
                options["source_masks"] = resolved["source_masks"]
        elif source_index is not None:
            x = resolver.resolve_source_block(ids, source_index, include_augmented=False)["values"]
        else:
            x = resolver.resolve_features(ids, include_augmented=False)["values"]
        metadata_predict = getattr(estimator, "predict_with_metadata", None)
        if named_input is not None:
            from .named_torch import named_torch_task_scope

            if bound_views is None:
                raise ValueError("named Torch replay requires its attested fixed-cohort task views")
            with named_torch_task_scope(task, model_node):
                for name, values in x.items():
                    bound_views.record_model_call("predict", name, "predict", ids, values)
                prediction = estimator.predict(x)
        elif callable(metadata_predict) and metadata_by_id is not None:
            prediction = metadata_predict(x, {metadata_key: [metadata_by_id[sample_id] for sample_id in ids]})
        else:
            prediction = estimator.predict_numeric(x, **options) if isinstance(estimator, _DagmlNativeStackingModel) else estimator.predict(x, **options)
        values = np.asarray(prediction, dtype=float).reshape(len(ids), -1)
        if named_origin is not None:
            validate_named_refit_origin(named_bundle, artifact)
        if late_contract is not None:
            validate_late_partial_stack(estimator, late_contract)
        if runtime_target_transform is not None:
            values = np.asarray(runtime_target_transform.inverse_transform(values), dtype=float)
        if values.shape != (len(ids), len(names)):
            raise ValueError("captured prediction width disagrees with training target names")
        block = {
            "prediction_id": f"pred:{model_id}:captured:PREDICT", "producer_node": model_id,
            "partition": "final", "fold_id": None, "sample_ids": ids,
            "values": values.tolist(), "target_names": names,
        }
        result = _build_result(task, [block], [], {})
        if bound_views is not None:
            result["consumed_data_views"] = bound_views.consumed_data_views()
        return result

    execute_options = ({"view_callback": fixed_views,
                        "resource_limits_json": json.dumps({"cpu_threads": 1, "gpu_devices": []})}
                       if fixed_views is not None else {})
    outcome = json.loads(execute(json.dumps(dsl), json.dumps(envelope), json.dumps(manifests), callback, "PREDICT", **execute_options))
    if outcome["phase"] != "PREDICT" or (outcome["scores"] is not None and outcome["scores"]["reports"]):
        raise ValueError("inference unexpectedly returned training or score-bearing evidence")
    blocks = [block for result in outcome["node_results"] for block in result["predictions"] if block["producer_node"] == model_id]
    if len(blocks) != 1:
        raise ValueError("captured replay must produce exactly one native prediction block")
    block = blocks[0]
    ids = block["sample_ids"]
    if len(ids) != len(set(ids)) or set(ids) != set(storage_ids) or block["target_names"] != names:
        raise ValueError("native replay output identities or targets disagree with its input cohort")
    position = {sample_id: index for index, sample_id in enumerate(ids)}
    values = np.asarray(block["values"], dtype=float)[[position[sample_id] for sample_id in storage_ids]]
    evidence = {
        "engine": "dag-ml", "execution_profile": "captured_artifact_replay", "phase": outcome["phase"],
        "effective_plan": outcome["effective_plan"], "node_results": outcome["node_results"], "scores": outcome["scores"],
        "predict_cohort": envelope["predict_cohort"], "sample_ids": storage_ids, "target_names": names,
        "source_artifact_id": artifact.get("artifact_id"), "source_content_fingerprint": artifact.get("content_fingerprint"),
        "cross_validation": False, "training_performed": False,
    }
    if public_target_transform is not None:
        values = np.asarray(public_target_transform.decode(values))
    if named_origin is not None:
        evidence["named_refit_origin"] = json.loads(json.dumps(named_origin))
        evidence["named_refit_fingerprint"] = artifact["named_refit_fingerprint"]
    if late_contract is not None:
        evidence["late_partial_refit_contract"] = json.loads(json.dumps(late_contract))
        evidence["late_partial_refit_fingerprint"] = artifact.get("late_partial_refit_fingerprint")
    return (values.ravel() if len(names) == 1 else values), evidence


def _prediction_content_fingerprint(spectro: Any) -> str:
    from nirs4all.data.multimodal import MultimodalSpectroDataset

    if isinstance(spectro, MultimodalSpectroDataset):
        return spectro.content_hash()
    return _array_content_fingerprint("X", spectro.x({}, layout="2d"))
