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

from nirs4all.pipeline.dagml_bridge import controller_manifests, pipeline_to_dsl

from .cli_runner import data_bindings_for
from .envelope import build_envelope
from .errors import DagMlUnavailable, DagMlUnsupported
from .identity import mint_identity
from .node_runner import _build_result, _MultiBlockEstimator, _source_index, _SourceConcatEstimator, _train_predict_ids
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
    from nirs4all.api.result import _DagmlExportedModel

    # General ``.n4a`` archives store the public wrapper as their sole joblib
    # member. Unwrap it before entering the numeric-only DAG callback; public
    # label decoding is applied after the native result has been validated.
    if isinstance(estimator, _DagmlExportedModel) and y_transform is None:
        estimator, y_transform = estimator.estimator, estimator.y_transform
    from .target_capture import CapturedTargetTransform

    public_target_transform = y_transform if isinstance(y_transform, CapturedTargetTransform) else None
    runtime_target_transform = public_target_transform.transformer if public_target_transform is not None else y_transform
    if not callable(getattr(estimator, "predict", None)):
        raise ValueError("captured artifact must contain a fitted prediction estimator")
    names = ["y"] if target_names is None else list(target_names)
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("prediction target names must be nonempty and unique")
    execute = getattr(importlib.import_module("dag_ml._dag_ml"), "execute_phase_in_process", None)
    cohort_builder = getattr(dag_ml, "attach_predict_cohort_to_envelope", None)
    if not callable(execute) or not callable(cohort_builder):
        raise DagMlUnavailable("captured-artifact replay requires the qualified DAG-ML PREDICT phase and cohort APIs")
    steps, _ = _split_pipeline(normalize_model_steps(pipeline))
    dsl = pipeline_to_dsl(_apply_model_params(steps), "nirs4all-captured-replay")
    manifests = controller_manifests()
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, manifests).graph.to_dict()
    models = [node for node in graph["nodes"] if node["kind"] == "model"]
    if len(models) != 1 or any(node["kind"] not in {"model", "transform", "y_transform"} for node in graph["nodes"]):
        raise DagMlUnsupported("one captured artifact requires its concrete single-model replay topology")
    model_node = models[0]
    model_id = model_node["id"]
    identity = mint_identity(spectro)
    storage_ids = identity.observation_ids()
    if not storage_ids:
        raise ValueError("prediction input must contain at least one row")
    envelope = build_envelope(spectro, identity)
    envelope.update(cohort_builder(envelope, {
        "role": "inference", "relations": envelope["coordinator_relations"], "target_names": names,
        "data_content_fingerprint": _array_content_fingerprint("X", spectro.x({}, layout="2d")),
        "target_content_fingerprint": None,
    }).to_dict())
    dsl["data_bindings"] = data_bindings_for(model_id, envelope)
    resolver = MaterializationResolver(spectro, identity)

    def callback(task: dict[str, Any]) -> dict[str, Any]:
        if task["phase"] != "PREDICT":
            raise ValueError("captured-artifact replay cannot execute a training phase")
        if task["node_plan"]["node_id"] != model_id:
            # The captured sklearn estimator already owns its fitted X-chain.
            # As in node_runner, transform nodes only transmit native handles.
            return _build_result(task, [], [], {})
        _, ids = _train_predict_ids(task)
        source_index = _source_index(model_node)
        if isinstance(estimator, _MultiBlockEstimator) or (
            isinstance(estimator, _SourceConcatEstimator) and resolver.is_multi_source()
        ):
            x = resolver.resolve_feature_blocks(ids, include_augmented=False)["blocks"]
        elif source_index is not None:
            x = resolver.resolve_source_block(ids, source_index, include_augmented=False)["values"]
        else:
            x = resolver.resolve_features(ids, include_augmented=False)["values"]
        values = np.asarray(estimator.predict(x), dtype=float).reshape(len(ids), -1)
        if runtime_target_transform is not None:
            values = np.asarray(runtime_target_transform.inverse_transform(values), dtype=float)
        if values.shape != (len(ids), len(names)):
            raise ValueError("captured prediction width disagrees with training target names")
        block = {
            "prediction_id": f"pred:{model_id}:captured:PREDICT", "producer_node": model_id,
            "partition": "final", "fold_id": None, "sample_ids": ids,
            "values": values.tolist(), "target_names": names,
        }
        return _build_result(task, [block], [], {})

    outcome = json.loads(execute(json.dumps(dsl), json.dumps(envelope), json.dumps(manifests), callback, "PREDICT"))
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
    return (values.ravel() if len(names) == 1 else values), evidence
