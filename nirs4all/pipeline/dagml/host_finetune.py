"""Scoped Optuna proposals evaluated by native DAG, not a Python CV loop."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

_HOST_KEYS = {"n_trials", "sampler", "sample", "verbose", "seed", "storage", "phases", "pruner", "n_jobs"}


def is_host_finetune(config: dict[str, Any]) -> bool:
    """Choose the host optimizer before execution, never after native failure."""
    engine = str(config.get("engine", "")).lower()
    return engine == "optuna" or (not engine and (bool(_HOST_KEYS & config.keys()) or config.get("approach") == "single"))


def validate_host_finetune(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize the first restored host profile without ignoring controls."""
    from nirs4all.optimization.optuna import OptunaManager

    manager = OptunaManager()
    if not manager.is_available:
        raise ImportError("General finetuning requires the optuna installation extra")
    params = manager._validate_and_normalize_finetune_params(dict(config))  # noqa: SLF001 -- optimizer owns its grammar
    allowed = {"model_params", "n_trials", "sampler", "verbose", "seed", "approach", "metric", "direction", "eval_mode", "engine", "pruner", "n_jobs"}
    unknown = params.keys() - allowed
    if unknown:
        raise NotImplementedError(f"DAG host finetuning controls not wired yet: {sorted(unknown)}")
    if params.get("approach", "grouped") != "single":
        raise NotImplementedError("DAG host finetuning currently restores approach='single'; grouped/individual optimizer profiles remain unqualified")
    if params.get("pruner", "none") != "none" or params.get("n_jobs", 1) != 1:
        raise NotImplementedError("DAG host single-holdout search has no progressive pruning or parallel-trial contract yet")
    budget = params.get("n_trials", 50)
    if type(budget) is not int or not 0 < budget <= 2**32 - 1:
        raise ValueError("finetune_params.n_trials must be a positive u32 integer")
    if not isinstance(params.get("model_params"), dict) or not params["model_params"]:
        raise ValueError("finetune_params.model_params must be a nonempty mapping")
    params["n_trials"] = budget
    params["engine"] = "optuna"
    return params


def run_scoped_finetune(
    model: Any, upstream: list[Any], x: np.ndarray, y: np.ndarray,
    config: dict[str, Any], *, scope: dict[str, Any], task_type: Any,
    y_transform: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune raw rows from one outer-training scope; no external targets enter.

    The optimizer owns sampling only. DAG owns the bounded ask/evaluate/tell
    loop, every candidate's fold-train preprocessing, scores and selection.
    The outer caller subsequently fits its chosen estimator on its own train.
    """
    import dag_ml
    from sklearn.base import clone
    from sklearn.model_selection import ShuffleSplit

    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.optimization.optuna import OptunaManager
    from nirs4all.pipeline.dagml_bridge import controller_manifests

    from .cli_runner import assemble_cv_refit_dsl
    from .envelope import build_envelope
    from .identity import mint_identity
    from .node_runner import run_node
    from .resolver import MaterializationResolver

    params = validate_host_finetune(config)
    manager = OptunaManager()
    seed = params.setdefault("seed", 42)
    metric = params.get("metric", "balanced_accuracy" if "classif" in str(task_type) else "rmse")
    if metric not in {"rmse", "mse", "mae", "r2", "accuracy", "balanced_accuracy"}:
        raise ValueError(f"DAG host finetuning metric {metric!r} is not supported by native scoring")
    direction = params.setdefault("direction", "maximize" if metric in {"r2", "accuracy", "balanced_accuracy"} else "minimize")
    dataset = SpectroDataset(name="host_hpo_inner")
    if task_type is not None:
        dataset.set_task_type(task_type)
    dataset.add_samples(np.asarray(x), indexes={"partition": "train"})
    dataset.add_targets(np.asarray(y))
    identity = mint_identity(dataset)
    pool = dataset.index_column("sample", {"partition": "train"})
    # Historical single search uses a deterministic 80/20 training-only holdout.
    splitter = ShuffleSplit(1, test_size=0.2, random_state=seed)
    folds = [(train.tolist(), val.tolist()) for train, val in splitter.split(x, y)]
    pipeline = [*upstream]
    if y_transform is not None:
        pipeline.append({"y_processing": clone(y_transform)})
    pipeline.append({"model": clone(model)})
    envelope = build_envelope(dataset, identity, sample_ints=pool)
    dsl = assemble_cv_refit_dsl(pipeline, identity, envelope, folds, dsl_id="nirs4all-host-hpo", n_splits=1)
    graph = json.loads(dag_ml.compile_pipeline_dsl_graph_json(json.dumps(dsl)))
    nodes = {node["id"]: node for node in graph["nodes"]}
    target = next(node["id"] for node in graph["nodes"] if node["kind"] == "model")
    target_transform = next((node for node in graph["nodes"] if node["kind"] == "y_transform"), None)
    resolver = MaterializationResolver(dataset, identity)
    store: dict[Any, Any] = {}
    manager._configure_logging(params.get("verbose", 0))  # noqa: SLF001
    study = manager._create_study(params)  # noqa: SLF001 -- reuse optimizer-owned sampler grammar
    pending: dict[int, Any] = {}
    stopped = False

    def stop_search() -> None:
        # GridSampler.after_trial signals exhaustion via Study.stop(), whose
        # built-in implementation assumes Study.optimize owns the loop. Here
        # native DAG owns it, so translate that signal into the next ask=None.
        nonlocal stopped
        stopped = True

    study.stop = stop_search

    def optimizer_callback(request: dict[str, Any]) -> Any:
        index = request["trial_index"]
        if request["operation"] == "ask":
            if stopped or (hasattr(study.sampler, "is_exhausted") and study.sampler.is_exhausted(study)):
                return None
            trial = study.ask()
            pending[index] = trial
            values, train_params = manager.sample_hyperparameters(trial, params)
            if train_params:
                raise ValueError("Host HPO cannot silently discard sampled train_params")
            # Canonical JSON restoration preserves tuple/type-token grammars in
            # configuration; concrete proposed parameters are ordinary JSON.
            return json.loads(json.dumps(values))
        if request["operation"] != "tell" or index not in pending:
            raise ValueError("Unexpected native optimizer transition")
        study.tell(pending.pop(index), request["score"])
        return None

    def op_callback(task: dict[str, Any]) -> dict[str, Any]:
        return run_node(task, resolver, nodes.__getitem__, store, graph.get("edges", []), target_transform)

    import importlib

    # The source facade is additive; installed dependency stubs may predate it.
    native = importlib.import_module("dag_ml")
    evidence: dict[str, Any] = native.run_host_hpo_search_in_process(
        dsl, envelope, controller_manifests(),
        {"target_node": target, "trial_budget": params["n_trials"], "metric": metric,
         "direction": direction, "optimizer_descriptor": json.loads(json.dumps(params))},
        op_callback, optimizer_callback,
    )
    evidence["scope"] = scope
    evidence["evaluation"] = {"role": "inner_parameter_selection", "outer_validation_used": False, "test_used": False}
    evidence["optimizer"] = {"name": "optuna", "sampler_class": type(study.sampler).__name__, "best_trial_number": study.best_trial.number}
    if evidence["selected_trial_index"] != study.best_trial.number:
        raise RuntimeError("Native selection and optimizer incumbent disagree")
    return dict(evidence["selected_params"]), evidence
