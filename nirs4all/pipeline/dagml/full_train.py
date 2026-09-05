"""DAG-owned single full-training execution, with no invented cross-validation.

The host assembles the plan and supplies operator callbacks, as for the CV
profile. Rust schedules one REFIT phase and owns its NodeResults and ScoreSet.
The historical test-as-validation view is explicitly labeled; it never becomes
a native OOF report or an independently validated model-selection claim.
"""

from __future__ import annotations

import importlib
import json
import warnings
from typing import Any

import numpy as np

from nirs4all.api.result import RunResult
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.dagml_bridge import controller_manifests, pipeline_to_dsl

from .cli_runner import data_bindings_for
from .envelope import build_envelope, target_names
from .errors import DagMlUnavailable, DagMlUnsupported, _reject_multi_model
from .identity import IdentityMap, mint_identity
from .in_process_runner import _capture_refit_artifacts, in_process_enabled
from .node_runner import run_node
from .public_normalization import normalize_model_steps
from .raw_training_lowerer import _array_content_fingerprint
from .resolver import MaterializationResolver
from .result import _index_sample_blocks
from .steps import _apply_model_params, _assert_supported_operators, _model_name, _split_pipeline


class NoSplitEvaluationWarning(UserWarning):
    """A full-training run has no independent cross-validation evidence."""


def run_full_train(
    pipeline: list[Any], spectro: Any, *, metric: str = "rmse",
    task_type: str = "regression", config_name: str = "",
) -> RunResult:
    """Fit one concrete pipeline once on all train rows using the DAG scheduler.

    No splitter, selection loop or legacy runner is introduced. Test rows are
    never fitted. A test partition retains the historical ``val`` alias with
    explicit provenance; without test, only training predictions are exposed.
    ``cv_best_score`` stays NaN in both cases because no CV occurred.
    """
    steps, splitter = _split_pipeline(normalize_model_steps(pipeline))
    if splitter is not None:
        raise DagMlUnsupported("full-training execution must not receive a splitter")
    _reject_multi_model(steps)
    _assert_supported_operators(steps)
    steps = _apply_model_params(steps)
    if not in_process_enabled():
        raise DagMlUnavailable("full-training execution requires the in-process DAG phase API; no legacy retry is performed")
    extension = importlib.import_module("dag_ml._dag_ml")
    execute = getattr(extension, "execute_phase_in_process", None)
    if not callable(execute):
        raise DagMlUnavailable("the installed DAG-ML runtime lacks execute_phase_in_process; install the qualified V1 corrective runtime")

    import dag_ml

    identity = mint_identity(spectro)
    train = spectro.index_column("sample", {"partition": "train"})
    test = spectro.index_column("sample", {"partition": "test"})
    envelope = build_envelope(spectro, identity, sample_ints=train)
    if test:
        cohort_builder = getattr(dag_ml, "attach_predict_cohort_to_envelope", None)
        if not callable(cohort_builder):
            raise DagMlUnavailable("the installed DAG-ML runtime lacks the native test-cohort constructor")
        test_envelope = build_envelope(spectro, identity, sample_ints=test)
        # Preserve the host's DataPlan (needed to construct bindings); the
        # execution-core envelope projection carries its fingerprint only.
        envelope.update(cohort_builder(envelope, {
            "role": "external_test", "relations": test_envelope["coordinator_relations"],
            "target_names": target_names(spectro),
            "data_content_fingerprint": _array_content_fingerprint("X", spectro.x({"partition": "test"}, layout="2d")),
            "target_content_fingerprint": _array_content_fingerprint("y", spectro.y({"partition": "test"})),
        }).to_dict())
    dsl = pipeline_to_dsl(steps, "nirs4all-full-train")
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    models = [node for node in graph["nodes"] if node["kind"] == "model"]
    if len(models) != 1:
        raise DagMlUnsupported("full-training execution needs one concrete model; expand independent public model requests before dispatch")
    model_id = models[0]["id"]
    dsl["data_bindings"] = data_bindings_for(model_id, envelope)
    resolver = MaterializationResolver(spectro, identity)
    nodes = {node["id"]: node for node in graph["nodes"]}
    target_transform = next((node for node in graph["nodes"] if node["kind"] == "y_transform"), None)
    store: dict[int, Any] = {}

    def callback(task: dict[str, Any]) -> dict[str, Any]:
        return run_node(task, resolver, nodes.__getitem__, store, graph.get("edges", []), target_transform)

    message = (
        "No splitter provided: fitting all training rows once; the test set is also used as validation. "
        "There is no cross-validation or independent model-selection holdout."
        if test else
        "No splitter or test set provided: fitting all training rows once; scores are training resubstitution only, not independent validation."
    )
    warnings.warn(message, NoSplitEvaluationWarning, stacklevel=2)
    outcome = json.loads(execute(
        json.dumps(dsl), json.dumps(envelope), json.dumps(controller_manifests()), callback, "REFIT",
        training_sample_ids=[identity.to_wire(sample) for sample in train],
    ))
    if outcome["phase"] != "REFIT":
        raise ValueError("full-training runtime returned an unexpected phase")
    return _project_full_train(
        outcome, identity, dataset_name=spectro.name, model_id=model_id,
        model_name=_model_name(steps), metric=metric, task_type=task_type,
        config_name=config_name, artifacts=_capture_refit_artifacts(outcome["node_results"], store),
    )


def _project_full_train(
    outcome: dict[str, Any], identity: IdentityMap, *, dataset_name: str,
    model_id: str, model_name: str, metric: str, task_type: str,
    config_name: str, artifacts: list[dict[str, Any]],
) -> RunResult:
    """Expose actual full-training reports without manufacturing CV evidence."""
    scores = outcome["scores"]
    reports = [report for report in scores["reports"] if report["producer_node"] == model_id and report["level"] == "sample"]
    if any(report["partition"] not in {"final", "test"} or report.get("fold_id") is not None for report in reports):
        raise ValueError("full-training reports unexpectedly contain cross-validation evidence")
    blocks = {report["partition"]: dict(report["metrics"]) for report in reports}
    if "final" not in blocks or len(blocks) != len(reports):
        raise ValueError("full-training requires one unambiguous native training report")
    has_test = "test" in blocks
    evaluation = {
        "profile": "full_train", "cross_validation": False,
        "training_scope": "resubstitution",
        "validation_source": "test" if has_test else None,
        "test_used_for_validation": has_test,
        "independent_model_selection_holdout": False,
    }
    partition_scores = {"train": blocks["final"]}
    if has_test:
        partition_scores.update(val=blocks["test"], test=blocks["test"])
    indexed = _index_sample_blocks(outcome["node_results"])
    predictions = Predictions()
    for partition, native_partition in [("train", "final"), *([("val", "test"), ("test", "test")] if has_test else [])]:
        block, target = indexed[(model_id, native_partition, None)]
        if target is None:
            raise ValueError("full-training prediction has no native target evidence")
        ids = block["sample_ids"]
        target_ids = [unit["id"] for unit in target["unit_ids"]]
        if ids != target_ids:
            raise ValueError("full-training prediction/target identities disagree")
        y_pred, y_true = np.asarray(block["values"], dtype=float), np.asarray(target["values"], dtype=float)
        predictions.add_prediction(
            dataset_name=dataset_name, config_name=config_name, model_name=model_name,
            fold_id="final", refit_context="full_train", partition=partition,
            metric=metric, task_type=task_type, scores=partition_scores,
            train_score=blocks["final"].get(metric), val_score=blocks.get("test", {}).get(metric),
            test_score=blocks.get("test", {}).get(metric),
            sample_indices=[identity.to_int(sample_id) for sample_id in ids],
            metadata={"physical_sample_id": list(ids)},
            result_metadata={"evaluation": dict(evaluation), "native_partition": native_partition},
            y_pred=y_pred.ravel() if y_pred.shape[1] == 1 else y_pred,
            y_true=y_true.ravel() if y_true.shape[1] == 1 else y_true,
            n_samples=len(ids),
        )
    predictions.flush()
    result = RunResult(predictions=predictions, per_dataset={dataset_name: {
        "engine": "dag-ml", "execution_profile": "full_train", "evaluation": evaluation,
    }})
    result._dagml_score_set = scores  # noqa: SLF001 -- untouched native authority
    result._dagml_node_results = outcome["node_results"]  # noqa: SLF001
    result._dagml_refit_artifacts = artifacts  # noqa: SLF001
    return result
