"""Closed dense structural HPO through native DAG generation and N4M."""

from __future__ import annotations

import copy
import importlib
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from nirs4all.operators.transforms.nirs import SavitzkyGolay
from nirs4all.operators.transforms.scalers import StandardNormalVariate
from nirs4all.pipeline.dagml_bridge import controller_manifests, lower_structural_hpo_pipeline

from . import tuning_adapters
from .cancellation import DagRunCancelled
from .cli_runner import data_bindings_for_nodes, split_invocation_for
from .dataset import _materialize_dataset
from .envelope import build_envelope
from .folds import _build_folds, _pool_features, _pool_targets, _split_group_grain, _splitter_operator
from .host_search_checkpoint import HostSearchOptimizer
from .identity import mint_identity
from .multimodal_tuning import _evaluate_host_task
from .node_runner import clear_cv_weight_transfers
from .resolver import MaterializationResolver
from .steps import DagMlSplitStep, _split_pipeline
from .structural_sources import validate_source_selection_alternatives
from .tuning_contracts import SUPPORTED_TUNING_KEYS, DagMLTuningSpec, TrialResult, TuningResult, parse_tuning_spec, tcv1_sha256

_PARAMETER_PATHS = {"model.alpha": "alpha", "model.n_components": "n_components"}


def is_structural_tuning_pipeline(pipeline: Any) -> bool:
    """Route declared structural generators without changing fixed estimator HPO."""
    if not isinstance(pipeline, list):
        return False
    return any(isinstance(step, dict) and (
        "_or_" in step or isinstance(step.get("model"), dict) and "_or_" in step["model"]
    ) for step in pipeline)


def _validate_preprocessing_alternatives(alternatives: Any, *, n_features: int | None = None) -> None:
    """Check declared chains without expanding native recipes or fitting operators."""
    if not isinstance(alternatives, list) or len(alternatives) < 2 or sum(item is None for item in alternatives) != 1:
        raise ValueError("structural preprocessing alternatives require exactly one None and at least one nonempty operator branch")
    for alternative in alternatives:
        if alternative is None:
            continue
        chain = alternative if isinstance(alternative, list) else [alternative]
        if not chain:
            raise ValueError("structural preprocessing chains must be nonempty flat lists of supported operators")
        for operator in chain:
            if type(operator) is StandardScaler:
                if any(type(operator.get_params()[key]) is not bool for key in ("copy", "with_mean", "with_std")):
                    raise ValueError("StandardScaler constructor controls must be booleans")
            elif type(operator) is StandardNormalVariate:
                if (type(operator.axis) is not int or operator.axis != 1
                        or type(operator.ddof) is not int or operator.ddof < 0
                        or any(type(operator.get_params()[key]) is not bool for key in ("copy", "with_mean", "with_std"))
                        or operator.copy is not True):
                    raise ValueError("structural StandardNormalVariate requires axis=1, nonnegative integer ddof, boolean controls and copy=True")
                if n_features is not None and operator.with_std and operator.ddof >= n_features:
                    raise ValueError("structural StandardNormalVariate ddof must be smaller than the feature width when with_std=True")
            elif type(operator) is SavitzkyGolay:
                if (type(operator.window_length) is not int or operator.window_length < 1
                        or type(operator.polyorder) is not int or not 0 <= operator.polyorder < operator.window_length
                        or type(operator.deriv) is not int or operator.deriv < 0
                        or isinstance(operator.delta, bool) or not isinstance(operator.delta, (int, float))
                        or not math.isfinite(operator.delta) or operator.delta == 0
                        or operator.copy is not True):
                    raise ValueError("structural SavitzkyGolay requires a positive integer window_length, integer polyorder in [0, window_length), "
                                     "nonnegative integer deriv, finite nonzero delta and copy=True")
                if n_features is not None and operator.window_length > n_features:
                    raise ValueError("structural SavitzkyGolay window_length must not exceed the training feature width")
            else:
                raise ValueError("structural preprocessing supports only StandardScaler, StandardNormalVariate and SavitzkyGolay instances in flat chains")


def validate_structural_profile(pipeline: Any) -> tuple[list[Any], Any]:
    """Validate the existing DSL shape before any native handles or callbacks."""
    if not isinstance(pipeline, list):
        raise TypeError("structural tuning requires a pipeline list")
    steps, splitter = _split_pipeline(pipeline)
    from .structural_multimodal import is_methods_model_choice, validate_typed_profile
    from .structural_topology import declared_topologies, is_topology_choice

    if is_topology_choice(steps):
        declared_topologies(steps, splitter)
        return steps, splitter
    if is_methods_model_choice(steps):
        validate_typed_profile(steps, splitter)
        return steps, splitter
    if (not isinstance(splitter, DagMlSplitStep) or type(_splitter_operator(splitter)) is not GroupKFold
            or not isinstance(splitter.group_by, str) or not splitter.group_by):
        raise ValueError("structural tuning requires an explicit GroupKFold split with group_by metadata")
    source_selections = len(steps) == 3
    body = steps[1:] if source_selections else steps
    if len(body) != 2 or not isinstance(body[0], dict) or set(body[0]) != {"_or_"}:
        raise ValueError("structural tuning requires preprocessing alternatives followed by model choices")
    if source_selections:
        if not isinstance(steps[0], dict) or set(steps[0]) != {"_or_"}:
            raise ValueError("structural source selection requires a first _or_ site of explicit source-concat merges")
        validate_source_selection_alternatives(steps[0]["_or_"])
    _validate_preprocessing_alternatives(body[0]["_or_"])
    if not isinstance(body[1], dict) or set(body[1]) != {"model"}:
        raise ValueError("structural tuning does not support model fit controls or additional steps")
    choices = body[1]["model"]
    models = choices.get("_or_") if isinstance(choices, dict) and set(choices) == {"_or_"} else None
    if (not isinstance(models, list) or len(models) != 2
            or sum(type(item) is Ridge for item in models) != 1 or sum(type(item) is PLSRegression for item in models) != 1):
        raise ValueError("structural model alternatives must be exactly Ridge and PLSRegression")
    pls = next(item for item in models if type(item) is PLSRegression)
    if pls.scale is not False or pls.copy is not True:
        raise ValueError("structural PLS requires scale=False and copy=True")
    ridge = next(item for item in models if type(item) is Ridge)
    if ridge.copy_X is not True or ridge.positive is not False or ridge.solver not in {"auto", "svd", "cholesky"}:
        raise ValueError("structural Ridge requires copy_X=True, positive=False and a deterministic dense solver")
    # Strict serialization checks constructor values, including finite numbers,
    # using the same lowering as ordinary native operator execution.
    # Source columns are dataset-dependent; validate their declaration now and
    # strictly lower the unchanged operator body before materialization.
    lower_structural_hpo_pipeline(body)
    return steps, splitter


def _validate_numeric_space(spec: DagMLTuningSpec, max_components: int) -> None:
    if set(spec.space) != set(_PARAMETER_PATHS):
        raise ValueError("structural tuning space requires only model.alpha and model.n_components")
    for path, declaration in spec.space.items():
        codec = tuning_adapters._categorical_codec_for_spec(declaration)
        choices = tuning_adapters._categorical_choices(declaration)
        if choices is not None:
            values = [codec.decode(value) for value in codec.choices] if codec is not None else choices
        elif isinstance(declaration, tuple):
            if path == "model.n_components" and len(declaration) == 3 and declaration[0] not in {"int", "int_log", "log_int"}:
                raise ValueError("structural n_components must use an integer search space")
            values = list(declaration[-2:])
        elif isinstance(declaration, dict):
            values = [declaration.get("low", declaration.get("min")), declaration.get("high", declaration.get("max"))]
            kind = str(declaration.get("type", "")).lower()
            if path == "model.n_components" and kind not in {"", "int", "int_log", "log_int"}:
                raise ValueError("structural n_components must use an integer search space")
            step = declaration.get("step")
            if step is not None:
                if path == "model.n_components" and (type(step) is not int or step < 1):
                    raise ValueError("structural n_components step must be a positive integer")
                if path == "model.alpha" and (isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step) or step < 0):
                    raise ValueError("structural alpha step must be finite and nonnegative")
        else:
            raise ValueError(f"unsupported structural tuning declaration for {path}")
        for value in values:
            if path == "model.n_components":
                if type(value) is not int or not 1 <= value <= max_components:
                    raise ValueError("structural n_components must be positive integers bounded by features and every fold train size")
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("structural Ridge alpha must be finite and nonnegative")


def _prepare_structure(pipeline: Any, dataset_input: Any, tuning: Any, run_options: dict[str, Any]) -> dict[str, Any]:
    steps, splitter = validate_structural_profile(pipeline)
    if not isinstance(tuning, dict):
        raise TypeError("structural tuning must be a mapping")
    unknown = tuning.keys() - SUPPORTED_TUNING_KEYS - {"progress_callback"}
    if unknown:
        raise ValueError(f"unsupported structural tuning controls: {sorted(unknown)}")
    spec = parse_tuning_spec({key: value for key, value in tuning.items() if key in SUPPORTED_TUNING_KEYS})
    if spec.engine != "n4m" or spec.metric != "rmse" or spec.direction != "minimize":
        raise ValueError("structural tuning requires n4m with minimizing native RMSE")
    if spec.seed is not None and not 0 <= spec.seed <= (1 << 64) - 1:
        raise ValueError("structural tuning seed must be an unsigned 64-bit integer")
    if spec.force_params is not None:
        raise ValueError("structural tuning does not support force_params; recipe selectors are internal native identities")
    if spec.n_jobs != 1 and (spec.sampler not in {"random"} or spec.pruner not in {None, "none"}):
        raise ValueError("parallel structural tuning requires sampler='random' without pruning")
    if spec.sampler not in {None, "auto", "random", "sobol", "lhs", "ternary", "ga", "pso", "cmaes", "tpe", "gp_ei"}:
        raise ValueError("unsupported structural sampler")
    if run_options.get("refit", True) is not True or run_options.get("session") is not None or run_options.get("results_path") is not None:
        raise ValueError("structural tuning requires standalone winner refit and public export('.n4a')")
    if run_options.get("cache") is not None:
        raise ValueError("structural tuning owns candidate-local caches; run(cache=...) is not supported")
    allowed_options = {"name", "verbose", "save_artifacts", "save_charts", "plots_visible", "random_state", "refit", "cache",
                       "project", "report_naming", "results_path", "session", "workspace_path", "store_run_id", "should_stop", "cpu_threads", "gpu_devices"}
    if unknown_options := run_options.keys() - allowed_options:
        raise ValueError(f"unsupported structural run controls: {sorted(unknown_options)}")
    if run_options.get("report_naming", "nirs") not in {"nirs", "ml", "auto"}:
        raise ValueError("report_naming must be 'nirs', 'ml', or 'auto'")
    if type(run_options.get("verbose", 0)) is not int or run_options.get("verbose", 0) not in range(4):
        raise ValueError("verbose must be an integer from zero through three")
    if run_options.get("save_artifacts") is not None and type(run_options["save_artifacts"]) is not bool:
        raise TypeError("save_artifacts must be a boolean")
    for key in ("should_stop", "progress_callback"):
        callback = run_options.get(key) if key == "should_stop" else tuning.get(key)
        if callback is not None and not callable(callback):
            raise TypeError(f"{key} must be callable")
    native = importlib.import_module("dag_ml")
    if not callable(getattr(native, "prepare_host_hpo_structural_catalogue", None)):
        raise ImportError("installed DAG-ML lacks native structural HPO preparation; install the matching structural build")
    dataset = _materialize_dataset(dataset_input)
    from .structural_multimodal import is_methods_model_choice, prepare_typed_structure
    from .structural_topology import is_topology_choice, prepare_topology_structure

    if is_topology_choice(steps):
        return prepare_topology_structure(steps, splitter, dataset, spec, run_options, native)
    if is_methods_model_choice(steps):
        return prepare_typed_structure(steps, splitter, dataset, spec, run_options, native)
    source_selections = len(steps) == 3
    n_sources = dataset.features_sources()
    if source_selections:
        from nirs4all.data.multimodal import MultimodalSpectroDataset

        if isinstance(dataset, MultimodalSpectroDataset) or getattr(dataset, "_generated_view_store", None) is not None:
            raise ValueError("structural source selection supports dense raw SpectroDataset blocks only; typed multimodal sources and generated views are unsupported")
        if (n_sources < 2 or any(dataset.features_processings(index) != ["raw"] for index in range(n_sources))
                or not dataset.is_regression):
            raise ValueError("structural source selection requires at least two dense raw sources and regression targets")
    elif n_sources != 1 or dataset.features_processings(0) != ["raw"] or not dataset.is_regression:
        raise ValueError("structural tuning requires one dense raw source and regression targets")
    identity = mint_identity(dataset)
    if any(sample.augmented for sample in identity.identities) or getattr(dataset, "_generated_view_store", None) is not None:
        raise ValueError("structural tuning does not support augmented or generated views")
    pool = dataset.index_column("sample", {"partition": "train"})
    x = _pool_features(dataset, pool)
    y = _pool_targets(dataset, pool)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or y.ndim != 1 or len(y) != len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("structural tuning requires finite dense training X and one complete finite target")
    min_features = x.shape[1]
    if source_selections:
        blocks = dataset.x_rows(pool, layout="2d", concat_source=False)
        if (not isinstance(blocks, list) or len(blocks) != n_sources
                or any(np.ndim(block) != 2 or np.shape(block)[0] != len(pool) or not np.shape(block)[1] for block in blocks)):
            raise ValueError("structural source selection requires aligned, nonempty two-dimensional dense source blocks")
        widths = [int(np.shape(block)[1]) for block in blocks]
        if sum(widths) != x.shape[1]:
            raise ValueError("structural source widths disagree with the complete concatenated input")
        selections = validate_source_selection_alternatives(steps[0]["_or_"], source_widths=widths)
        min_features = min(sum(widths[index] for index in selection) for selection in selections)
    preproc = steps[1] if source_selections else steps[0]
    _validate_preprocessing_alternatives(preproc["_or_"], n_features=min_features)
    folds = _build_folds(splitter, dataset, pool, set())
    if not folds or any(not train or not validation for train, validation in folds):
        raise ValueError("structural tuning requires nonempty train and validation rows in every fold")
    _validate_numeric_space(spec, min(min_features, *(len(train) for train, _validation in folds)))
    groups = _split_group_grain(splitter, dataset, pool)
    if groups is None:
        raise ValueError("structural tuning requires complete declared sample groups")
    envelope = build_envelope(dataset, identity, sample_ints=pool, group_by_sample=groups)
    operator_seed = run_options.get("random_state")
    if operator_seed is None:
        operator_seed = spec.seed if spec.seed is not None else 0
    if type(operator_seed) is not int or not 0 <= operator_seed <= (1 << 64) - 1:
        raise ValueError("structural operator seed must be an unsigned 64-bit integer")
    dsl = lower_structural_hpo_pipeline(steps, source_layout=envelope["plan"].get("source_layout") if source_selections else None)
    dsl["root_seed"] = operator_seed
    dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=len(folds), shuffle=False)
    manifests = controller_manifests()
    graph = native.compile_pipeline_dsl_artifact_with_controllers(dsl, manifests).graph.to_dict()
    dsl["data_bindings"] = data_bindings_for_nodes([node["id"] for node in graph["nodes"] if node["kind"] in {"model", "transform"}], envelope)
    for binding in dsl["data_bindings"]:
        binding["view_policy"] = {"include_augmented_train": False, "include_refit_test_view": True}
    catalogue = native.prepare_host_hpo_structural_catalogue(dsl, envelope, manifests, _PARAMETER_PATHS)
    descriptor = spec.to_dict()
    for key in ("resume", "n_trials", "storage", "study_name"):
        descriptor.pop(key, None)
    descriptor["operator_rng"] = {"policy": "per_native_task_v1", "seed": operator_seed}
    from .raw_training_lowerer import _array_content_fingerprint

    descriptor["training_content_fingerprint"] = tcv1_sha256({
        "buffers": _array_content_fingerprint("X", x), "targets": y.tolist(),
        "groups": [{"sample_id": identity.to_wire(sample), "group": groups[sample]} for sample in pool],
    })
    request = {"target_node": catalogue["entries"][0]["target_node"], "trial_budget": spec.n_trials,
               "metric": spec.metric, "direction": spec.direction, "optimizer_descriptor": descriptor,
               "fold_score_reduction": "mean", "structural_catalogue": catalogue}
    if spec.pruner not in {None, "none"}:
        request["progressive_pruning"] = True
    prepared = {"spec": spec, "dataset": dataset, "identity": identity, "folds": folds, "envelope": envelope,
                "dsl": dsl, "manifests": manifests, "graph": graph, "catalogue": catalogue, "request": request,
                "operator_seed": operator_seed, "splitter": splitter, "steps": steps, "pool": pool, "groups": groups}
    if source_selections:
        prepared["source_layout"] = copy.deepcopy(envelope["plan"]["source_layout"])
    return prepared


def _run_structural_tuning(pipeline: Any, dataset_input: Any, tuning: Any, *, run_options: dict[str, Any]) -> Any:
    """Run native conditional proposals and train only the selected native recipe."""
    from .resources import bind_execution_resources, current_execution_resources, reset_execution_resources

    prepared = _prepare_structure(pipeline, dataset_input, tuning, run_options)
    native = importlib.import_module("dag_ml")
    if not callable(getattr(native, "resolve_host_hpo_structural_winner", None)):
        raise ImportError("installed DAG-ML lacks native structural winner resolution")
    spec = prepared["spec"]
    dataset = prepared["dataset"]
    resolver = MaterializationResolver(dataset, prepared["identity"])
    stores: dict[int, dict[Any, Any]] = {}
    workers: dict[int, Any] = {}
    methods_controllers: dict[int, Any] = {}
    proposals: dict[int, dict[str, Any]] = {}
    resources = current_execution_resources()
    optimizer = HostSearchOptimizer(spec, n_folds=len(prepared["folds"]), structural_catalogue=prepared["catalogue"])
    progress = tuning.get("progress_callback")
    should_stop = run_options.get("should_stop")
    stop_requested = False
    checkpoint_fingerprint = (optimizer.resume_checkpoint["fingerprint"]
                              if optimizer.resume_checkpoint is not None else None)
    optimizer_progress = False

    def release_candidate(index: int) -> None:
        store = stores.pop(index, None)
        if store is not None:
            clear_cv_weight_transfers(store)
            store.clear()
        worker = workers.pop(index, None)
        if worker is not None:
            worker.close()
        controller = methods_controllers.pop(index, None)
        if controller is not None:
            controller.close()
        proposals.pop(index, None)

    def propose(event: dict[str, Any]) -> Any:
        nonlocal optimizer_progress
        response = optimizer(event)
        optimizer_progress = True
        if event["operation"] == "ask":
            proposals[event["trial_index"]] = copy.deepcopy(response)
        return response

    def candidate_callback_factory(index: int) -> Any:
        if index in stores or index in workers or index in methods_controllers:
            raise ValueError("structural HPO reused a candidate callback namespace")
        proposal = proposals[index]
        selector = prepared["catalogue"]["selector_path"]
        entry = next(item for item in prepared["catalogue"]["entries"] if item["recipe_id"] == proposal[selector])
        graph = entry["graph"]
        nodes = {node["id"]: node for node in graph["nodes"]}

        def check_task(task: dict[str, Any]) -> None:
            variant = task.get("variant") or {}
            choices = variant.get("choices") or {}
            expected = {"trial_index": index, "recipe_id": entry["recipe_id"],
                        "catalogue_fingerprint": prepared["catalogue"]["catalogue_fingerprint"]}
            if (not task.get("variant_id") or task["variant_id"] != variant.get("variant_id")
                    or (choices.get("host_hpo") or {}).get("value") != expected
                    or {key: value for key, value in choices.items() if key != "host_hpo"} != entry["variant"]["choices"]
                    or task["node_plan"]["node_id"] not in nodes):
                raise ValueError("candidate task native trial, recipe or node disagrees with the structural catalogue")

        if prepared.get("methods_typed"):
            from .envelope import source_ids
            from .methods_multimodal import controller_for_graph

            controller = controller_for_graph(graph, dataset.cohort, allow_fit=True, binding_source_ids=source_ids(dataset))
            methods_controllers[index] = controller

            def typed(task: dict[str, Any]) -> dict[str, Any]:
                check_task(task)
                token = bind_execution_resources(resources)
                try:
                    return cast(dict[str, Any], controller.operator(task))
                finally:
                    reset_execution_resources(token)

            return typed
        if spec.n_jobs != 1:
            from .host_hpo_candidate import HostHpoCandidate

            token = bind_execution_resources(resources)
            try:
                worker = HostHpoCandidate(index, provider=None, dataset=dataset, identity=prepared["identity"],
                                          graph=graph, operator_seed=prepared["operator_seed"])
            finally:
                reset_execution_resources(token)
            workers[index] = worker

            def isolated(task: dict[str, Any]) -> dict[str, Any]:
                check_task(task)
                return cast(dict[str, Any], worker.call("operator", task))

            return isolated
        store: dict[Any, Any] = {}
        stores[index] = store

        def sequential(task: dict[str, Any]) -> dict[str, Any]:
            check_task(task)
            token = bind_execution_resources(resources)
            try:
                return _evaluate_host_task(task, resolver=resolver, nodes=nodes, graph=graph,
                                           model_store=store, view_store=None, operator_seed=prepared["operator_seed"])
            finally:
                reset_execution_resources(token)

        return sequential

    def checkpoint(event: dict[str, Any]) -> Any:
        nonlocal stop_requested, checkpoint_fingerprint, optimizer_progress
        if event["operation"] == "prepare_terminal":
            return True
        # Preserve an attested resume frontier when only the optimizer clock
        # would change. Proposals and feedback still require paired saves,
        # even before the native terminal frontier advances.
        fingerprint = event["checkpoint"]["fingerprint"]
        if fingerprint != checkpoint_fingerprint or optimizer_progress:
            optimizer.checkpoint(event)
            checkpoint_fingerprint = fingerprint
            optimizer_progress = False
        count = len(event["checkpoint"]["trials"])
        for index in list(stores.keys() | workers.keys() | methods_controllers.keys()):
            if index < count:
                release_candidate(index)
        response = progress(copy.deepcopy(event)) if progress is not None else True
        if should_stop is not None and should_stop():
            stop_requested = True
            return False
        return response

    def fallback(_task: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("structural HPO requires a candidate-local native callback namespace")

    try:
        if should_stop is not None and should_stop():
            raise DagRunCancelled("structural search cancelled before trial execution")
        evidence = native.run_host_hpo_search_in_process(
            prepared["dsl"], prepared["envelope"], prepared["manifests"], prepared["request"], fallback, propose,
            resume_checkpoint=optimizer.resume_checkpoint, progress_callback=checkpoint,
            candidate_callback_factory=candidate_callback_factory,
        )
    finally:
        try:
            with ExitStack() as cleanup:
                for index in list(stores.keys() | workers.keys() | methods_controllers.keys()):
                    cleanup.callback(release_candidate, index)
        finally:
            optimizer.close()
    if evidence["status"] == "cancelled":
        reason = "caller" if stop_requested else "progress callback"
        raise DagRunCancelled(f"structural search cancelled by {reason}; paired checkpoint saved for resume=True")
    if evidence.get("selected_params") is None:
        raise RuntimeError("structural search produced no successful candidate")
    result = _train_selected_structure(prepared, evidence, pipeline, dataset_input, run_options)
    selector = prepared["catalogue"]["selector_path"]
    trials = []
    for record in evidence["checkpoint"]["trials"]:
        item = record.get("evidence", record)
        state = {"complete": "COMPLETE", "pruned": "PRUNED", "failed": "FAIL"}[record["state"]]
        params = {path: value for path, value in item["params"].items() if path != selector}
        trials.append(TrialResult(number=item["trial_index"], params=params, state=state,
                                  value=item["score"] if state == "COMPLETE" else None,
                                  diagnostics={"engine": "dag-ml", "test_used": False, "recipe_id": item["params"][selector]}))
    winner = next(item for item in evidence["trials"] if item["trial_index"] == evidence["selected_trial_index"])
    result._tuning_result = TuningResult(tuning=spec, best_params={path: value for path, value in evidence["selected_params"].items() if path != selector},
                                        best_value=winner["score"], trials=tuple(trials), optimizer="n4m")
    result.structural_tuning_evidence = copy.deepcopy(evidence)
    result.structural_tuning_search_request = copy.deepcopy({key: prepared[key] for key in ("dsl", "envelope", "request")})
    result.structural_tuning_search_request["controller_manifests"] = copy.deepcopy(prepared["manifests"])
    if not prepared.get("methods_typed"):
        for artifact in result._dagml_refit_artifacts:
            artifact["estimator"].structural_tuning_evidence = copy.deepcopy(evidence)
    return result


def _train_selected_structure(
    prepared: dict[str, Any], evidence: dict[str, Any], pipeline: Any,
    dataset_input: Any, run_options: dict[str, Any],
) -> Any:
    """Execute the signed native winner with the ordinary host artifact capture."""
    if prepared.get("methods_topology"):
        from .structural_topology import train_selected_topology

        return train_selected_topology(prepared, evidence, run_options)
    if prepared.get("methods_typed"):
        from .structural_multimodal import train_selected_typed_structure

        return train_selected_typed_structure(prepared, evidence, run_options)
    from .in_process_runner import _capture_refit_artifacts
    from .native_results import write_native_results
    from .raw_training_lowerer import _array_content_fingerprint, _core_relation_fingerprint, _data_contracts_from_campaign, _output_request_for_node, _training_influence_manifest
    from .resources import current_execution_resources
    from .result import _scores_to_run_result
    from .run_backend import _attach_export_spec
    from .training_contracts import DagMLTrainingRequestSpec, assemble_training_request

    native = importlib.import_module("dag_ml")
    dataset, identity = prepared["dataset"], prepared["identity"]
    artifact = native.compile_pipeline_dsl_artifact_with_controllers(prepared["dsl"], prepared["manifests"])
    graph = artifact.graph.to_dict()
    campaign = artifact.campaign_template.to_dict()
    envelope = copy.deepcopy(prepared["envelope"])
    envelope["relation_fingerprint"] = _core_relation_fingerprint(envelope["coordinator_relations"], native)
    envelope["data_content_fingerprint"] = _array_content_fingerprint("X", _pool_features(dataset, prepared["pool"]))
    envelope["target_content_fingerprint"] = _array_content_fingerprint("y", _pool_targets(dataset, prepared["pool"]))
    resolver = MaterializationResolver(dataset, identity)
    pool_ids = [identity.to_wire(sample) for sample in prepared["pool"]]
    target_payload = resolver.resolve_targets(resolver.target_sample_ids(pool_ids))
    names = target_payload.get("target_names", ["y"])
    test = dataset.index_column("sample", {"partition": "test"})
    if test:
        test_envelope = build_envelope(dataset, identity, sample_ints=test)
        envelope.update(native.attach_predict_cohort_to_envelope(envelope, {
            "role": "external_test", "relations": test_envelope["coordinator_relations"], "target_names": names,
            "data_content_fingerprint": _array_content_fingerprint("X", _pool_features(dataset, test)),
            "target_content_fingerprint": _array_content_fingerprint("y", _pool_targets(dataset, test)),
        }).to_dict())
    _union_envelopes, identities = _data_contracts_from_campaign(campaign, envelope)
    output = _output_request_for_node(graph, prepared["request"]["target_node"], target_names=names)
    resources = current_execution_resources()
    template = assemble_training_request(DagMLTrainingRequestSpec(
        request_id="training:nirs4all.structural", plan_id="plan:nirs4all.structural", graph=graph, campaign=campaign,
        controller_manifests=prepared["manifests"], data_identities=identities, output_requests=[output],
        seed=prepared["operator_seed"], cv_artifacts="discard", fitted_artifacts="allow_host_sidecar",
        cpu_threads=resources.cpu_threads, gpu_devices=resources.gpu_devices,
        selection_required_metric_level="sample", selection_evaluation_scope="oof",
    ))
    request = native.resolve_host_hpo_structural_winner(prepared["request"], evidence, template)
    graph, campaign = request["graph"], request["campaign"]
    data_envelopes, _identities = _data_contracts_from_campaign(campaign, envelope)
    influence = _training_influence_manifest(graph, campaign, prepared["folds"], identity,
                                             group_by_sample=prepared["groups"], selection_metric="rmse")
    nodes = {node["id"]: node for node in graph["nodes"]}
    store: dict[Any, Any] = {}
    frames: list[dict[str, Any]] = []
    training = None

    def callback(task: dict[str, Any]) -> dict[str, Any]:
        response = _evaluate_host_task(task, resolver=resolver, nodes=nodes, graph=graph, model_store=store,
                                       view_store=None, operator_seed=prepared["operator_seed"])
        frames.append({**response, "variant_id": task.get("variant_id")})
        return response

    try:
        training = native.execute_training(request, data_envelopes, envelope["coordinator_relations"], influence, callback,
                                           outcome_id="outcome:nirs4all.structural", run_id="run:nirs4all.structural", bundle_id="bundle:nirs4all.structural")
        document = training.outcome.to_dict()
        captures = _capture_refit_artifacts(frames, store)
        if len(captures) != 1:
            raise RuntimeError("structural winner must produce exactly one fitted REFIT predictor")
        if "source_layout" in prepared:
            from .multimodal_contracts import bind_dense_concat_input_contract

            bind_dense_concat_input_contract(captures[0]["estimator"], prepared["source_layout"])
        captures[0]["estimator"].structural_tuning_evidence = copy.deepcopy(evidence)
        for average in document.get("oof_averages", []):
            frames.append({"aggregated_predictions": [average["predictions"]], "regression_targets": [average["y_true"]]})
        by_variant: dict[Any, list[dict[str, Any]]] = {}
        for frame in frames:
            by_variant.setdefault(frame.get("variant_id", document["selected_variant_id"]), []).append(frame)
        model = next(node for node in graph["nodes"] if node["kind"] == "model")
        result = _scores_to_run_result(document["score_set"], dataset.name, str(model["operator"]).rsplit(".", 1)[-1],
                                      producer=model["id"], config_name=run_options.get("name", ""),
                                      results_by_variant=by_variant, identity=identity, refit_artifacts=captures)
        for attribute, value in (("_dagml_graph", graph), ("structural_tuning_training_request", request),
                                 ("structural_tuning_training_outcome", document)):
            setattr(result, attribute, copy.deepcopy(value))
        result._dagml_target_names = names
        for metadata in result.per_dataset.values():
            metadata.update(engine="dag-ml", refit_enabled=True, tuning_profile="structural_ridge_pls_v1")
        _attach_export_spec(result, pipeline, dataset_input, run_options.get("name", ""), prepared["operator_seed"])
        if run_options.get("save_artifacts") or run_options.get("project") is not None or "workspace_path" in run_options:
            from nirs4all.pipeline.runner import _get_default_workspace_path

            from .workspace_projection import publish_workspace_result

            workspace = Path(run_options.get("workspace_path") or _get_default_workspace_path())
            if run_options.get("save_artifacts"):
                result._dagml_results_dir = write_native_results(result, result._dagml_score_set, workspace / "native_results")
            publish_workspace_result(result, pipeline, dataset, workspace, name=run_options.get("name", ""),
                                     project=run_options.get("project"), report_naming=run_options.get("report_naming", "nirs"),
                                     store_run_id=run_options.get("store_run_id"))
        return result
    finally:
        try:
            if training is not None:
                training.detach()
        finally:
            clear_cv_weight_transfers(store)
            store.clear()


def run_structural_tuning(pipeline: Any, dataset_input: Any, tuning: Any, *, run_options: dict[str, Any]) -> Any:
    """Bind existing execution resources for search, workers and winner training."""
    from .cancellation import SHOULD_STOP
    from .resources import bind_execution_resources, normalize_execution_resources, reset_execution_resources

    resources = normalize_execution_resources(run_options.get("cpu_threads", 1), run_options.get("gpu_devices", ()))
    resource_token = bind_execution_resources(resources)
    cancellation_token = SHOULD_STOP.set(run_options.get("should_stop"))
    try:
        return _run_structural_tuning(pipeline, dataset_input, tuning, run_options=run_options)
    finally:
        SHOULD_STOP.reset(cancellation_token)
        reset_execution_resources(resource_token)
