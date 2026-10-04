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
from .host_finetune import attach_host_finetune_splitter
from .host_search_checkpoint import HostSearchOptimizer
from .identity import mint_identity
from .late_tuning import prepare_late_tuning, validate_nested_local_finetune
from .node_runner import clear_cv_weight_transfers, run_node
from .public_normalization import normalize_model_steps
from .resolver import MaterializationResolver
from .steps import _split_pipeline
from .training_controls import validate_cv_weight_transfer_graph, validate_training_control_declarations
from .tuning_contracts import SUPPORTED_TUNING_KEYS, TrialResult, TuningResult, normalize_parameter_path, parse_tuning_spec, tcv1_sha256


class MultimodalTuningStopped(RuntimeError):
    """A cancelled native search, resumable from its paired checkpoint."""

    def __init__(self, evidence: dict[str, Any]) -> None:
        super().__init__("multimodal tuning cancelled between trials; resume=True continues the saved search")
        self.evidence = evidence


def _evaluate_host_task(
    task: dict[str, Any], *, resolver: Any, nodes: dict[str, Any], graph: dict[str, Any],
    model_store: dict[Any, Any], view_store: Any, operator_seed: int,
) -> dict[str, Any]:
    """Apply the same per-task seed and native task mapping in either host process."""
    from nirs4all.pipeline.runner import init_global_random_state

    try:
        task_seed = int(tcv1_sha256({
            "seed": operator_seed, "variant": task.get("variant_id"), "fold": task.get("fold_id"),
            "node": task["node_plan"]["node_id"], "phase": task["phase"],
        })[:8], 16)
        if not (graph.get("metadata") or {}).get("python_torch_profile"):
            init_global_random_state(task_seed)
        host_task = copy.deepcopy(task)
        for choice in (host_task.get("variant") or {}).get("choices", {}).values():
            for override in choice.get("param_overrides", []):
                override["params"] = {key.replace(".", "__"): value for key, value in override.get("params", {}).items()}
        validate_cv_weight_transfer_graph(graph, resolver)
        generated_views = view_store.bind_task(host_task) if view_store is not None else None
        return run_node(host_task, resolver, nodes.__getitem__, model_store, graph.get("edges", []), None,
                        generated_views=generated_views, graph_metadata=graph.get("metadata", {}))
    finally:
        # Search callbacks are CV-only. Candidate snapshots cannot initialize a
        # later trial or the selected run, which captures its own native CV fold.
        clear_cv_weight_transfers(model_store)


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
    validate_training_control_declarations(pipeline)
    steps, splitter = _split_pipeline(pipeline)
    if splitter is None:
        raise ValueError("multimodal tuning requires an explicit outer splitter")
    model_step = (steps[0] if len(steps) == 1 and isinstance(steps[0], dict)
                  and isinstance(steps[0].get("model"), (MultimodalRegressor, MultimodalClassifier)) else None)
    if model_step is not None and set(model_step) - {"model", "train_params", "refit_params", "finetune_params", "name"}:
        raise ValueError("multimodal tuning model steps accept only model, train_params, refit_params, finetune_params and name")
    model = model_step["model"] if model_step is not None else None
    model_controls = {key: value for key, value in (model_step or {}).items() if key != "model"}
    if isinstance(model, MultimodalClassifier) and getattr(model, "backend", "sklearn") == "methods":
        raise ValueError("native classifier HPO requires explicit _or_ topology sequences with node-qualified n_components axes")
    methods_backend = isinstance(model, MultimodalRegressor) and model.backend == "methods"
    if methods_backend:
        from .methods_multimodal import TUNABLE_KEYS, validate_training_profile

        validate_training_profile(pipeline, cohort, refit=run_options.get("refit", True) is True)
        allowed_paths = {normalize_parameter_path(key)[0] for key in TUNABLE_KEYS}
        if (set(model_controls) - {"name"}
                or any(normalize_parameter_path(key)[0] not in allowed_paths for key in tuning.get("space", {}))):
            raise ValueError("Methods multimodal global tuning accepts only alpha, image weight and image components without fit controls")
        if tuning.get("n_jobs", 1) != 1:
            raise ValueError("Methods multimodal global tuning currently requires n_jobs=1")
        if run_options.get("results_path") is not None or run_options.get("session") is not None:
            raise ValueError("Methods multimodal uses public result.export('.n4a') and does not support native results directories or sessions")
        if not callable(getattr(importlib.import_module("n4m"), "MultimodalPipeline", None)):
            raise ImportError("installed nirs4all-methods lacks MultimodalPipeline; install the matching native encoder build")
    generated_store = getattr(cohort, "_generated_view_store", None)
    if methods_backend and generated_store is not None:
        raise ValueError("Methods multimodal tuning requires fixed complete sources")
    if generated_store is not None and model is None:
        raise NotImplementedError("generated-view tuning requires one concrete multimodal model")
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
    if spec.engine != "n4m":
        raise ValueError("durable multimodal tuning requires engine='n4m'")
    sampler = spec.sampler or "tpe"
    if sampler not in {"random", "sobol", "lhs", "ternary", "ga", "pso", "cmaes", "tpe", "gp_ei"}:
        raise ValueError(f"durable multimodal tuning does not support sampler={sampler!r}")
    if spec.pruner not in {None, "none", "median", "successive_halving", "asha", "hyperband", "racing"}:
        raise ValueError(f"durable multimodal tuning does not support pruner={spec.pruner!r}")
    if spec.n_jobs != 1 and (spec.sampler not in {None, "random"} or spec.pruner not in {None, "none"}):
        raise ValueError("durable multimodal tuning requires n_jobs=1 for nonrandom samplers or pruning")
    supported_metrics = {"accuracy", "balanced_accuracy", "f1"} if classification else {"rmse", "mse", "mae", "r2"}
    if spec.metric not in supported_metrics:
        raise ValueError(f"multimodal {'classification' if classification else 'regression'} tuning requires one of {sorted(supported_metrics)}")
    if spec.force_params is not None:
        raise ValueError("durable multimodal tuning does not yet support queued force_params")
    local_params = validate_nested_local_finetune(model_step, spec.space) if model_step is not None else None
    if local_params is not None:
        assert model is not None
        if generated_store is not None:
            raise ValueError("generated data views do not support nested local finetune_params")
        if (not cohort.target_mask.all() or any(not mask.all() for mask in cohort.source_presence().values())
                or model.missing_source_policy != "error"):
            raise ValueError("nested local HPO requires complete targets and sources with missing_source_policy='error'")
        model_controls["finetune_params"] = local_params
    parallel_candidates = spec.n_jobs != 1
    from nirs4all.pipeline.runner import init_global_random_state

    operator_seed = run_options.get("random_state")
    if operator_seed is None:
        operator_seed = spec.seed if spec.seed is not None else 0
    init_global_random_state(operator_seed)
    progress = tuning.get("progress_callback")
    if progress is not None and not callable(progress):
        raise TypeError("tuning.progress_callback must be callable")
    recipe = prepare_late_tuning(pipeline, dataset, spec.space) if model is None else None
    if recipe is not None:
        if recipe.layout.get("schema") == "nirs4all.source-stacking-layout.v4" and spec.n_jobs != 1:
            raise ValueError("incomplete late-fusion tuning requires serial n_jobs=1")
        if recipe.layout.get("missing_source_policy", "error") == "error" and any(not mask.all() for mask in cohort.source_presence().values()):
            raise ValueError("late-fusion tuning requires complete sources unless missing_source_policy='zero_with_indicator' is explicit")
    pool = list(range(dataset.num_samples))
    identity = mint_identity(dataset)
    folds = _build_folds(splitter, dataset, pool, set())
    groups = _split_group_grain(splitter, dataset, pool)
    envelope = build_envelope(dataset, identity, sample_ints=pool, group_by_sample=groups)
    native = importlib.import_module("dag_ml")
    if recipe is None:
        search_step = {"model": clone(model), **copy.deepcopy(model_controls)}
        search_step = attach_host_finetune_splitter([splitter, search_step])[1]
        dsl = assemble_cv_refit_dsl([search_step], identity, envelope, folds, dsl_id="multimodal-hpo", n_splits=len(folds))
        if generated_store is not None:
            dsl["root_seed"] = operator_seed
        if methods_backend:
            from .methods_multimodal import bind_methods_dsl

            if not isinstance(model, MultimodalRegressor):
                raise ValueError("Methods multimodal tuning requires a MultimodalRegressor")
            declaration = bind_methods_dsl(dsl, model, dataset.cohort)
            dsl = declaration["dsl"]
            manifests = [declaration["manifest"]]
            graph = native.compile_pipeline_dsl_artifact_with_controllers(dsl, manifests).graph.to_dict()
        else:
            graph = json.loads(native.compile_pipeline_dsl_graph_json(json.dumps(dsl)))
            manifests = controller_manifests(dsl)
        target = next(node["id"] for node in graph["nodes"] if node["kind"] == "model")
    else:
        from .run_paths import _assemble_stacking_dsl

        dsl, graph, _base_ids = _assemble_stacking_dsl(
            recipe.pipeline, recipe.branches, recipe.meta, dataset, identity, pool, folds, envelope,
            task_type="classification" if classification else "regression", random_state=operator_seed, group_by_sample=groups, source_layout=recipe.layout,
        )
        target = "merge:stack"
        manifests = controller_manifests(dsl)
    nodes = {node["id"]: node for node in graph["nodes"]}
    resolver = MaterializationResolver(dataset, identity)
    store: dict[Any, Any] = {}
    descriptor = spec.to_dict()
    for key in ("resume", "n_trials", "storage", "study_name"):
        descriptor.pop(key, None)
    descriptor["operator_rng"] = {"policy": "per_native_task_v1", "seed": operator_seed}
    if local_params is not None or (recipe is not None and recipe.has_local_finetune):
        descriptor["nested_local_hpo"] = {
            "profile": "raw_source_recipe_inner_cv_v1", "graph_fingerprint": tcv1_sha256(graph),
        }
    if spec.pruner not in {None, "none"}:
        descriptor["n4m_options"] = {
            "n_startup_trials": 10, "max_resource": len(folds) if spec.pruner == "hyperband" else 0,
            "reduction_factor": 0,
        }
    if generated_store is not None:
        descriptor["generated_view_mode"] = "checkpoint_manifest_v1"
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
    if spec.pruner not in {None, "none"}:
        request["progressive_pruning"] = True
    if recipe is not None:
        request["parameter_bindings"] = recipe.bindings
    elif methods_backend:
        # Keep optimizer/checkpoint paths in their public canonical dotted form.
        # Native HPO maps them to the exact closed operator keys before PLAN/FIT.
        request["parameter_bindings"] = {
            path: {"node_id": target, "param_path": path.replace(".", "__")}
            for path in spec.space
        }

    def evaluate(task: dict[str, Any], *, model_store: dict[Any, Any], view_store: Any = None) -> dict[str, Any]:
        # Native task identity reproduces unseeded operators across resume.
        return _evaluate_host_task(
            task, resolver=resolver, nodes=nodes, graph=graph, model_store=model_store,
            view_store=view_store, operator_seed=operator_seed,
        )

    trial_view_stores: dict[int, Any] = {}
    candidate_processes: dict[int, Any] = {}
    methods_controller: Any = None

    def candidate_process(index: int) -> Any:
        worker = candidate_processes.get(index)
        if worker is None:
            from .host_hpo_candidate import HostHpoCandidate

            provider = generated_store.provider_for_worker() if generated_store is not None else None
            worker = HostHpoCandidate(
                index, provider=provider, dataset=dataset, identity=identity,
                graph=graph, operator_seed=operator_seed,
            )
            candidate_processes[index] = worker
        return worker

    def view_callback_factory(index: int) -> Any:
        if generated_store is None:
            raise ValueError("static HPO cannot create generated view callbacks")
        if parallel_candidates:
            return lambda call: candidate_process(index).call("view", call)
        if index in trial_view_stores:
            raise ValueError("generated HPO candidate reused its trial index")
        view_store = generated_store.for_trial()
        trial_view_stores[index] = view_store
        return view_store

    def candidate_callback_factory(index: int) -> Any:
        if parallel_candidates:
            return lambda task: candidate_process(index).call("operator", task)
        view_store = trial_view_stores[index]
        model_store: dict[Any, Any] = {}

        def evaluate_candidate(task: dict[str, Any]) -> dict[str, Any]:
            if not task.get("data_view_receipts"):
                raise ValueError("generated HPO candidate has no native data-view receipts")
            return evaluate(task, model_store=model_store, view_store=view_store)

        return evaluate_candidate

    def fallback_evaluate(task: dict[str, Any]) -> dict[str, Any]:
        nonlocal methods_controller
        if generated_store is not None:
            raise ValueError("generated HPO used the static fallback operator")
        if methods_backend:
            from .methods_multimodal import controller_for_graph

            if methods_controller is None:
                methods_controller = controller_for_graph(graph, dataset.cohort, allow_fit=True,
                                                           binding_source_ids=dsl["data_bindings"][0]["source_ids"])
            return cast(dict[str, Any], methods_controller.operator(task))
        return evaluate(task, model_store=store)

    optimizer = HostSearchOptimizer(spec, n_folds=len(folds))
    stop_requested = False

    def checkpoint(event: dict[str, Any]) -> Any:
        nonlocal stop_requested, methods_controller
        if event["operation"] == "prepare_terminal":
            # The optimizer is still RUNNING here; publish only after tell/fail.
            return True
        optimizer.checkpoint(event)
        if methods_controller is not None:
            methods_controller.close()
            methods_controller = None
        terminal_count = len(event["checkpoint"]["trials"])
        for index in list(candidate_processes):
            if index < terminal_count:
                candidate_processes.pop(index).close()
        response = progress(copy.deepcopy(event)) if progress is not None else True
        if should_stop is not None and should_stop():
            stop_requested = True
            return False
        return response

    host_hpo_kwargs: dict[str, Any] = {
        "resume_checkpoint": optimizer.resume_checkpoint,
        "progress_callback": checkpoint,
        "candidate_callback_factory": candidate_callback_factory if generated_store is not None or parallel_candidates else None,
    }
    if generated_store is not None:
        # DAG-ML 0.3.30 has no generated-view arguments. Static tuning must
        # continue to work with the published minimum dependency.
        host_hpo_kwargs["view_callback_factory"] = view_callback_factory
        host_hpo_kwargs["resume_view_validator"] = (
            generated_store.recheck_record if optimizer.resume_checkpoint is not None else None
        )
    try:
        evidence = native.run_host_hpo_search_in_process(
            dsl, envelope, manifests, request,
            fallback_evaluate, optimizer,
            **host_hpo_kwargs,
        )
    finally:
        try:
            for worker in candidate_processes.values():
                worker.close()
        finally:
            try:
                if methods_controller is not None:
                    methods_controller.close()
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
        selected_pipeline = [splitter, {"model": selected, **copy.deepcopy(model_controls)}]
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
        state = {"complete": "COMPLETE", "pruned": "PRUNED", "failed": "FAIL"}[record["state"]]
        complete = state == "COMPLETE"
        item = record.get("evidence", record)
        trials.append(TrialResult(number=item["trial_index"], params=item["params"],
                                  value=item["score"] if complete else None, state=state,
                                  diagnostics={"engine": "dag-ml", "test_used": False}))
    winner = next(item for item in evidence["trials"] if item["trial_index"] == evidence["selected_trial_index"])
    result._tuning_result = TuningResult(tuning=spec, best_params=evidence["selected_params"],
                                        best_value=winner["score"], trials=tuple(trials), optimizer="n4m")
    if methods_backend:
        from .methods_multimodal import MethodsMultimodalRunResult

        if not isinstance(result, MethodsMultimodalRunResult):
            raise RuntimeError("Methods multimodal selected training returned a different result profile")
        result.methods_multimodal_tuning_evidence = copy.deepcopy(evidence)
        result.methods_multimodal_search_request = copy.deepcopy({"dsl": dsl, "envelope": envelope, "controller_manifests": manifests, "request": request})
    for artifact in result._dagml_refit_artifacts:
        # Native REFIT already sealed the estimator's learned state. Search
        # evidence belongs to its transport envelope, not to that signed state.
        artifact["multimodal_tuning_evidence"] = copy.deepcopy(evidence)
    return result
