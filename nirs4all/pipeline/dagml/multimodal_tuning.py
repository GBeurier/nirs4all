"""Native grouped search over raw multimodal sources with durable optimizer state."""

from __future__ import annotations

import copy
import importlib
import json
from dataclasses import replace
from typing import Any, cast

import numpy as np
from sklearn.base import clone

from nirs4all.core.metrics import is_higher_better
from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.operators.models.multimodal import MultimodalClassifier, MultimodalRegressor
from nirs4all.pipeline.dagml_bridge import controller_manifests

from .cli_runner import assemble_cv_refit_dsl
from .envelope import build_envelope
from .folds import _build_folds, _split_group_grain
from .host_search_checkpoint import HostSearchOptimizer
from .identity import mint_identity
from .late_tuning import prepare_late_tuning
from .node_runner import run_node
from .public_normalization import normalize_model_steps
from .resolver import MaterializationResolver
from .steps import _split_pipeline
from .tuning_contracts import SUPPORTED_TUNING_KEYS, TrialResult, TuningResult, parse_tuning_spec, tcv1_sha256


class MultimodalTuningStopped(RuntimeError):
    """A cancelled native search, resumable from its paired checkpoint."""

    def __init__(self, evidence: dict[str, Any]) -> None:
        super().__init__("multimodal tuning cancelled between trials; resume=True continues the saved search")
        self.evidence = evidence


def run_multimodal_tuning(pipeline: Any, cohort: Any, tuning: Any, *, run_options: dict[str, Any]) -> Any:
    """Use DAG for folds, trial execution, scoring and winner selection; N4M for proposals."""
    from .cancellation import DagRunCancelled

    should_stop = run_options.get("should_stop")
    if should_stop is not None and not callable(should_stop):
        raise TypeError("should_stop must be a zero-argument cancellation callback")
    if should_stop is not None and should_stop():
        raise DagRunCancelled("DAG multimodal search cancelled by caller")
    if not isinstance(tuning, dict):
        raise TypeError("multimodal tuning must be a mapping")
    unknown = tuning.keys() - SUPPORTED_TUNING_KEYS - {"progress_callback"}
    if unknown:
        raise ValueError(f"unsupported multimodal tuning controls: {sorted(unknown)}")
    if not isinstance(pipeline, list):
        raise TypeError("multimodal tuning requires a pipeline list")
    pipeline = normalize_model_steps(pipeline)
    steps, splitter = _split_pipeline(pipeline)
    if splitter is None:
        raise ValueError("multimodal tuning requires an explicit outer splitter")
    model = (steps[0]["model"] if len(steps) == 1 and isinstance(steps[0], dict) and set(steps[0]) == {"model"}
             and isinstance(steps[0]["model"], (MultimodalRegressor, MultimodalClassifier)) else None)
    # Only training rows enter either the native search contract or callbacks.
    train_ids = [sample for sample, partition in zip(cohort.sample_ids, cohort.partitions, strict=True) if partition == "train"]
    dataset = MultimodalSpectroDataset(cohort.take(train_ids))
    if dataset.cohort.y is None:
        raise ValueError("multimodal tuning requires training targets")
    classification = isinstance(model, MultimodalClassifier) if model is not None else dataset.is_classification
    controls = {key: value for key, value in tuning.items() if key in SUPPORTED_TUNING_KEYS}
    if classification:
        controls.setdefault("metric", "balanced_accuracy")
    spec = parse_tuning_spec(controls)
    if "direction" not in controls:
        spec = replace(spec, direction="maximize" if is_higher_better(spec.metric) else "minimize")
    if spec.engine != "n4m" or spec.sampler not in {None, "random"} or spec.pruner not in {None, "none"}:
        raise ValueError("durable multimodal tuning currently requires engine='n4m', sampler='random', without pruning")
    supported_metrics = {"accuracy", "balanced_accuracy"} if classification else {"rmse", "mse", "mae", "r2"}
    if spec.metric not in supported_metrics:
        raise ValueError(f"multimodal {'classification' if classification else 'regression'} tuning requires one of {sorted(supported_metrics)}")
    if spec.force_params is not None:
        raise ValueError("durable multimodal tuning does not yet support queued force_params")
    from nirs4all.pipeline.runner import init_global_random_state

    operator_seed = run_options.get("random_state")
    if operator_seed is None:
        operator_seed = spec.seed if spec.seed is not None else 0
    init_global_random_state(operator_seed)
    progress = tuning.get("progress_callback")
    if progress is not None and not callable(progress):
        raise TypeError("tuning.progress_callback must be callable")
    recipe = prepare_late_tuning(pipeline, dataset, spec.space) if model is None else None
    if recipe is not None and (not cohort.target_mask.all() or any(not mask.all() for mask in cohort.source_presence().values())):
        raise ValueError("late-fusion tuning requires complete targets and complete sources")
    pool = list(range(dataset.num_samples))
    identity = mint_identity(dataset)
    folds = _build_folds(splitter, dataset, pool, set())
    groups = _split_group_grain(splitter, dataset, pool)
    envelope = build_envelope(dataset, identity, sample_ints=pool, group_by_sample=groups)
    native = importlib.import_module("dag_ml")
    if recipe is None:
        dsl = assemble_cv_refit_dsl([{"model": clone(model)}], identity, envelope, folds, dsl_id="multimodal-hpo", n_splits=len(folds))
        graph = json.loads(native.compile_pipeline_dsl_graph_json(json.dumps(dsl)))
        target = next(node["id"] for node in graph["nodes"] if node["kind"] == "model")
    else:
        from .run_paths import _assemble_stacking_dsl

        dsl, graph, _base_ids = _assemble_stacking_dsl(
            recipe.pipeline, recipe.branches, recipe.meta, dataset, identity, pool, folds, envelope,
            task_type="classification" if classification else "regression", random_state=operator_seed, group_by_sample=groups, source_layout=recipe.layout,
        )
        target = "merge:stack"
    nodes = {node["id"]: node for node in graph["nodes"]}
    resolver = MaterializationResolver(dataset, identity)
    store: dict[Any, Any] = {}
    descriptor = spec.to_dict()
    for key in ("resume", "n_trials", "storage", "study_name"):
        descriptor.pop(key, None)
    descriptor["operator_rng"] = {"policy": "per_native_task_v1", "seed": operator_seed}
    provider_evidence = getattr(cohort, "_data_provider_evidence", None)
    if provider_evidence is not None:
        descriptor["data_provider_execution"] = provider_evidence["execution"]["execution_fingerprint"]
    fingerprint_targets = dataset.cohort.y
    if not dataset.cohort.target_mask.all():
        fingerprint_targets = np.where(dataset.cohort.target_mask, fingerprint_targets, 0)
    descriptor["training_content_fingerprint"] = tcv1_sha256({
        "buffers": dataset.content_hash(),
        "targets": fingerprint_targets.tolist(),
        "target_names": list(dataset.cohort.target_names),
        "target_mask": dataset.cohort.target_mask.tolist(),
        "task_type": dataset.cohort.task_type,
        "schema": dataset.cohort.schema_descriptors(),
    })
    request = {"target_node": target, "trial_budget": spec.n_trials, "metric": spec.metric,
               "direction": spec.direction, "optimizer_descriptor": descriptor, "fold_score_reduction": "mean"}
    if recipe is not None:
        request["parameter_bindings"] = recipe.bindings

    def evaluate(task: dict[str, Any]) -> dict[str, Any]:
        # Starting from the same native task identity also reproduces unseeded
        # sklearn operators after resume, independently of earlier callbacks.
        task_seed = int(tcv1_sha256({
            "seed": operator_seed, "variant": task.get("variant_id"), "fold": task.get("fold_id"),
            "node": task["node_plan"]["node_id"], "phase": task["phase"],
        })[:8], 16)
        init_global_random_state(task_seed)
        # The portable tuning contract uses dotted paths; sklearn uses '__'.
        host_task = copy.deepcopy(task)
        for choice in (host_task.get("variant") or {}).get("choices", {}).values():
            for override in choice.get("param_overrides", []):
                override["params"] = {key.replace(".", "__"): value for key, value in override.get("params", {}).items()}
        return run_node(host_task, resolver, nodes.__getitem__, store, graph.get("edges", []), None)

    optimizer = HostSearchOptimizer(spec)
    stop_requested = False

    def checkpoint(event: dict[str, Any]) -> Any:
        nonlocal stop_requested
        if event["operation"] == "prepare_terminal":
            # The optimizer is still RUNNING here; publish only after tell/fail.
            return True
        optimizer.checkpoint(event)
        response = progress(copy.deepcopy(event)) if progress is not None else True
        if should_stop is not None and should_stop():
            stop_requested = True
            return False
        return response

    try:
        evidence = native.run_host_hpo_search_in_process(
            dsl, envelope, controller_manifests(), request, evaluate, optimizer,
            resume_checkpoint=optimizer.resume_checkpoint, progress_callback=checkpoint,
        )
    finally:
        optimizer.close()
    if evidence["status"] == "cancelled":
        if stop_requested:
            raise DagRunCancelled("DAG multimodal search cancelled by caller; checkpoint saved for resume=True")
        raise MultimodalTuningStopped(evidence)
    if "selected_params" not in evidence:
        raise RuntimeError("multimodal search produced no successful candidate")
    # Reuse the existing native CV/refit execution path for the selected recipe.
    # Selection has already completed; held-out test targets never reach HPO.
    from nirs4all.api.result import RunResult
    from nirs4all.api.run import run

    if recipe is None:
        selected = clone(model).set_params(**{key.replace(".", "__"): value for key, value in evidence["selected_params"].items()})
        selected_pipeline = [splitter, {"model": selected}]
    else:
        selected_pipeline = recipe.selected_pipeline(evidence["selected_params"])
    # A completed-checkpoint replay performs no trial callbacks. Give final
    # training the same RNG start regardless of how many trials ran this time.
    init_global_random_state(operator_seed)
    result = cast(RunResult, run(selected_pipeline, cohort, engine="dag-ml", **run_options))
    if recipe is not None:
        from .named_stacking import NamedStackingResult

        # The objective selects an ensemble. A base producer's better score must
        # never silently change which fitted predictor export()/best refer to.
        result = next(view for view in cast(NamedStackingResult, result).runs if any(item.get("producer_node") == target for item in view.per_dataset.values()))
    records = evidence["checkpoint"]["trials"]
    trials = []
    for record in records:
        complete = record["state"] == "complete"
        item = record.get("evidence", record)
        trials.append(TrialResult(number=item["trial_index"], params=item["params"],
                                  value=item["score"] if complete else None, state="COMPLETE" if complete else "FAIL",
                                  diagnostics={"engine": "dag-ml", "test_used": False}))
    winner = next(item for item in evidence["trials"] if item["trial_index"] == evidence["selected_trial_index"])
    result._tuning_result = TuningResult(tuning=spec, best_params=evidence["selected_params"],
                                        best_value=winner["score"], trials=tuple(trials), optimizer="n4m")
    for artifact in result._dagml_refit_artifacts:
        artifact["estimator"].multimodal_tuning_evidence = evidence
    return result
