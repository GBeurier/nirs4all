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

from .cli_runner import data_bindings_for_fitted_x_chain
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
    task_type: str = "regression", config_name: str = "", augmented_train: bool = False,
    cli: str | None = None, venv_python: str | None = None,
    dataset_path: str | None = None, dataset_pickle: str | None = None,
    workdir: Any = None, random_state: int | None = None,
    train_sample_ids: list[int] | None = None,
) -> RunResult:
    """Fit one concrete pipeline once on selected train rows using the DAG scheduler.

    No splitter, selection loop or legacy runner is introduced. Test rows are
    fitted only by an explicit ``fit_on_all=True`` transform. A test partition
    retains the historical ``val`` alias with explicit provenance; without test,
    only training predictions are exposed.
    ``cv_best_score`` stays NaN in both cases because no CV occurred.
    """
    steps, splitter = _split_pipeline(normalize_model_steps(pipeline))
    if splitter is not None:
        raise DagMlUnsupported("full-training execution must not receive a splitter")
    from .detect import _detect_duplication_branch, _detect_separation_branch, _is_exclude_step

    train_pool: set[int] | None = None
    if any(_is_exclude_step(step) for step in steps):
        from .exclude import _resolve_exclude

        steps, allowed, marked = _resolve_exclude(steps, spectro)
        train_pool = set(allowed) - marked

    duplication = _detect_duplication_branch(steps)
    if duplication is not None:
        branches, merge_mode = duplication
        if merge_mode in ("features", "all"):
            from .run_paths import _branch_merge_transformer_step

            model_step = next(step for step in steps if isinstance(step, dict) and "model" in step)
            steps = [_branch_merge_transformer_step(branches, merge_mode), model_step]
    separation = _detect_separation_branch(steps)
    if separation is None:
        _reject_multi_model(steps)
        _assert_supported_operators(steps)
        steps = _apply_model_params(steps)
    execute = None
    if in_process_enabled():
        extension = importlib.import_module("dag_ml._dag_ml")
        execute = getattr(extension, "execute_phase_in_process", None)
        if not callable(execute):
            raise DagMlUnavailable("the installed DAG-ML runtime lacks execute_phase_in_process; install the qualified V1 corrective runtime")

    import dag_ml

    identity = mint_identity(spectro)
    metadata_by_sample = None
    sample_metadata = None
    if separation is not None:
        from .run_paths import _branch_metadata

        branch_step, _branch_body = separation
        criterion = branch_step["branch"]
        mode, key = ("by_metadata", criterion["by_metadata"]) if "by_metadata" in criterion else ("by_tag", criterion["by_tag"])
        metadata_by_sample, sample_metadata = _branch_metadata(spectro, identity, mode, key)
    partition_train_all = spectro.index_column("sample", {"partition": "train"})
    train_all = partition_train_all
    if train_sample_ids is not None:
        if not train_sample_ids or len(train_sample_ids) != len(set(train_sample_ids)) or not set(train_sample_ids) <= set(train_all):
            raise ValueError("full-training sample IDs must be a non-empty unique subset of the train partition")
        train_all = list(train_sample_ids)
    train = train_all
    if train_pool is not None:
        train = [sample for sample in train_all if int(sample) in train_pool]
    augmentation_by_sample = None
    envelope_train = train if train_pool is not None else train_all
    if augmented_train:
        from .folds import _split_base_samples

        origins = spectro.index_column("origin", {"partition": "train"})
        allowed_origins = set(_split_base_samples(spectro))
        if train_sample_ids is not None:
            allowed_origins &= set(train_sample_ids)
        if train_pool is not None:
            allowed_origins &= train_pool
        train = [sample for sample, origin in zip(partition_train_all, origins, strict=True) if sample == origin and sample in allowed_origins]
        envelope_train = [sample for sample, origin in zip(partition_train_all, origins, strict=True) if origin in allowed_origins]
        augmentation_by_sample = {
            sample: "sample_augmentation"
            for sample, origin in zip(partition_train_all, origins, strict=True) if sample != origin and origin in allowed_origins
        }
    test = spectro.index_column("sample", {"partition": "test"})
    envelope = build_envelope(spectro, identity, sample_ints=envelope_train, augmentation_by_sample=augmentation_by_sample, metadata_by_sample=metadata_by_sample)
    if test:
        cohort_builder = getattr(dag_ml, "attach_predict_cohort_to_envelope", None)
        if not callable(cohort_builder):
            raise DagMlUnavailable("the installed DAG-ML runtime lacks the native test-cohort constructor")
        test_envelope = build_envelope(spectro, identity, sample_ints=test, metadata_by_sample=metadata_by_sample)
        # Preserve the host's DataPlan (needed to construct bindings); the
        # execution-core envelope projection carries its fingerprint only.
        envelope.update(cohort_builder(envelope, {
            "role": "external_test", "relations": test_envelope["coordinator_relations"],
            "target_names": target_names(spectro),
            "data_content_fingerprint": _array_content_fingerprint("X", spectro.x({"partition": "test"}, layout="2d")),
            "target_content_fingerprint": _array_content_fingerprint("y", spectro.y({"partition": "test"})),
        }).to_dict())
    if separation is not None:
        assert sample_metadata is not None
        branch_step, branch_body = separation
        return _run_full_train_separation(
            branch_step, branch_body, spectro, identity, envelope, train,
            sample_metadata, execute, metric=metric, task_type=task_type,
            config_name=config_name, cli=cli, venv_python=venv_python,
            dataset_path=dataset_path, dataset_pickle=dataset_pickle,
            workdir=workdir, random_state=random_state,
        )
    dsl = pipeline_to_dsl(steps, "nirs4all-full-train")
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    models = [node for node in graph["nodes"] if node["kind"] == "model"]
    if len(models) != 1:
        raise DagMlUnsupported("full-training execution needs one concrete model; expand independent public model requests before dispatch")
    model_id = models[0]["id"]
    from .cli_runner import needs_dynamic_feature_axis

    dsl["data_bindings"] = data_bindings_for_fitted_x_chain(
        graph, model_id, envelope, force=needs_dynamic_feature_axis(steps),
    )
    if train_sample_ids is not None:
        message = "Single-fold file: fitting its train IDs once and evaluating its validation IDs as a test holdout; no cross-validation occurred."
    elif test:
        message = (
            "No splitter provided: fitting all training rows once; the test set is also used as validation. "
            "There is no cross-validation or independent model-selection holdout."
        )
    else:
        message = "No splitter or test set provided: fitting all training rows once; scores are training resubstitution only, not independent validation."
    if execute is None:
        if cli is None or venv_python is None or dataset_path is None or workdir is None:
            raise DagMlUnavailable("CLI full training requires a DAG-ML CLI, Python adapter, and reloadable dataset")
        from pathlib import Path

        from .cli_runner import run_refit_phase_cli
        from .errors import _raise_run_failure
        from .in_process_runner import _load_subprocess_refit_artifacts

        cli_run = run_refit_phase_cli(
            dsl=dsl, envelope=envelope, graph=graph,
            training_sample_ids=[identity.to_wire(sample) for sample in train],
            dataset_path=dataset_path, dataset_pickle=dataset_pickle,
            workdir=Path(workdir), dagml_cli=cli, venv_python=venv_python,
            random_state=random_state,
        )
        if cli_run["returncode"]:
            _raise_run_failure(cli_run, "full-training CLI phase failed")
        outcome = json.loads(cli_run["phase_output"].read_text())
        if outcome["phase"] != "REFIT":
            raise ValueError("full-training CLI returned an unexpected phase")
        artifacts = _load_subprocess_refit_artifacts(outcome["node_results"], cli_run["artifact_dir"])
        warnings.warn(message, NoSplitEvaluationWarning, stacklevel=2)
        return _project_full_train(
            outcome, identity, dataset_name=spectro.name, model_id=model_id,
            model_name=_model_name(steps), metric=metric, task_type=task_type,
            config_name=config_name, artifacts=artifacts,
        )
    resolver = MaterializationResolver(spectro, identity)
    nodes = {node["id"]: node for node in graph["nodes"]}
    from nirs4all.api.general_transfer import bind_transfer_operators

    bind_transfer_operators(nodes, steps)
    target_transform = next((node for node in graph["nodes"] if node["kind"] == "y_transform"), None)
    store: dict[int, Any] = {}

    def callback(task: dict[str, Any]) -> dict[str, Any]:
        return run_node(task, resolver, nodes.__getitem__, store, graph.get("edges", []), target_transform)

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


def _run_full_train_separation(
    branch_step: dict[str, Any], branch_body: list[Any], spectro: Any,
    identity: IdentityMap, envelope: dict[str, Any], train: list[int],
    sample_metadata: dict[str, dict[str, Any]], execute: Any | None, *,
    metric: str, task_type: str, config_name: str,
    cli: str | None, venv_python: str | None, dataset_path: str | None,
    dataset_pickle: str | None, workdir: Any, random_state: int | None,
) -> RunResult:
    """Fan out metadata partitions and reassemble their REFIT predictions."""
    import dag_ml

    from .cli_runner import data_bindings_for_nodes
    from .run_paths import _MERGE_NODE_ID, _branch_compat_step
    from .steps import _supported_body_steps

    criterion = branch_step["branch"]
    key = criterion.get("by_metadata", criterion.get("by_tag"))
    body_steps = _supported_body_steps(branch_body)
    template = {"id": "per_partition", "steps": [_branch_compat_step(step) for step in body_steps]}
    compat_dsl = {
        "id": "nirs4all-full-train-separation",
        "pipeline": [
            {"branch": {"branches": [template]}, "mode": "by_metadata", "selector": {"metadata_key": key}, "metadata": {"auto_separate": True}},
            {"merge": "concat", "output_as": "predictions", "id": _MERGE_NODE_ID},
        ],
    }
    dsl = dag_ml.fan_out_data_aware_branches(compat_dsl, envelope).to_dict()
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    model_ids = [node["id"] for node in graph["nodes"] if node["kind"] == "model"]
    if not model_ids:
        raise DagMlUnsupported("full-training separation fan-out produced no model nodes")
    dsl["data_bindings"] = data_bindings_for_nodes(model_ids, envelope)
    resolver = MaterializationResolver(spectro, identity)
    nodes = {node["id"]: node for node in graph["nodes"]}
    target_transform = next((node for node in graph["nodes"] if node["kind"] == "y_transform"), None)
    store: dict[int, Any] = {}

    def callback(task: dict[str, Any]) -> dict[str, Any]:
        return run_node(task, resolver, nodes.__getitem__, store, graph.get("edges", []), target_transform, sample_metadata)

    warnings.warn(
        "No splitter provided: fitting each metadata partition on all its training rows once; "
        "scores are training resubstitution or held-out test evaluation, not cross-validation.",
        NoSplitEvaluationWarning, stacklevel=2,
    )
    training_ids = [identity.to_wire(sample) for sample in train]
    if execute is None:
        if cli is None or venv_python is None or dataset_path is None or workdir is None:
            raise DagMlUnavailable("CLI separation refit requires a DAG-ML CLI, Python adapter, and reloadable dataset")
        from pathlib import Path

        from .cli_runner import run_refit_phase_cli
        from .errors import _raise_run_failure
        from .in_process_runner import _load_subprocess_refit_artifacts

        cli_run = run_refit_phase_cli(
            dsl=dsl, envelope=envelope, graph=graph, training_sample_ids=training_ids,
            dataset_path=dataset_path, dataset_pickle=dataset_pickle,
            sample_metadata=sample_metadata, workdir=Path(workdir),
            dagml_cli=cli, venv_python=venv_python, random_state=random_state,
        )
        if cli_run["returncode"]:
            _raise_run_failure(cli_run, "full-training separation CLI phase failed")
        outcome = json.loads(cli_run["phase_output"].read_text())
        artifacts = _load_subprocess_refit_artifacts(outcome["node_results"], cli_run["artifact_dir"])
    else:
        outcome = json.loads(execute(
            json.dumps(dsl), json.dumps(envelope), json.dumps(controller_manifests()), callback, "REFIT",
            training_sample_ids=training_ids,
        ))
        artifacts = _capture_refit_artifacts(outcome["node_results"], store)
    if outcome["phase"] != "REFIT":
        raise ValueError("full-training separation runtime returned an unexpected phase")
    result = _project_full_train_separation(
        outcome, identity, train=train, model_ids=model_ids,
        dataset_name=spectro.name, model_name=_model_name(body_steps),
        metric=metric, task_type=task_type, config_name=config_name,
        artifacts=artifacts,
    )
    from .native_results import separation_replay_manifest

    replay = separation_replay_manifest(graph, str(key), result._dagml_refit_artifacts)
    if replay is not None:
        result.per_dataset[spectro.name]["separation_replay"] = replay
    return result


def _project_full_train_separation(
    outcome: dict[str, Any], identity: IdentityMap, *, train: list[int],
    model_ids: list[str], dataset_name: str, model_name: str, metric: str,
    task_type: str, config_name: str, artifacts: list[dict[str, Any]],
) -> RunResult:
    """Join fanned REFIT train blocks and use the native concat test report."""
    from nirs4all.core.metrics import eval_list

    from .run_paths import _MERGE_NODE_ID

    indexed = _index_sample_blocks(outcome["node_results"])
    training_rows: dict[str, tuple[list[float], list[float]]] = {}
    for model_id in model_ids:
        block, target = indexed[(model_id, "final", None)]
        if target is None:
            raise ValueError(f"separation model {model_id!r} has no native training target evidence")
        target_ids = [unit["id"] for unit in target["unit_ids"]]
        if block["sample_ids"] != target_ids:
            raise ValueError("separation training prediction/target identities disagree")
        for sample_id, prediction, truth in zip(block["sample_ids"], block["values"], target["values"], strict=True):
            if sample_id in training_rows:
                raise ValueError(f"separation training sample {sample_id!r} appears in multiple partitions")
            training_rows[sample_id] = (prediction, truth)
    train_ids = [identity.to_wire(sample) for sample in train]
    if set(training_rows) != set(train_ids):
        raise ValueError("separation model partitions do not cover the full training universe")
    y_pred_train = np.asarray([training_rows[sample_id][0] for sample_id in train_ids], dtype=float)
    y_true_train = np.asarray([training_rows[sample_id][1] for sample_id in train_ids], dtype=float)
    train_score = eval_list(y_true_train, y_pred_train, [metric])[0]
    if train_score is None:
        raise ValueError(f"separation training metric {metric!r} could not be evaluated")
    train_metrics = {metric: float(train_score)}

    test_reports = [report for report in outcome["scores"]["reports"] if report["producer_node"] == _MERGE_NODE_ID and report["partition"] == "test" and report["level"] == "sample"]
    if len(test_reports) > 1:
        raise ValueError("separation concat produced ambiguous native test reports")
    test_metrics = dict(test_reports[0]["metrics"]) if test_reports else None
    evaluation = {
        "profile": "full_train", "cross_validation": False,
        "training_scope": "resubstitution",
        "validation_source": "test" if test_metrics is not None else None,
        "test_used_for_validation": test_metrics is not None,
        "independent_model_selection_holdout": False,
    }
    predictions = Predictions()
    rows: list[tuple[str, list[str], np.ndarray, np.ndarray, str]] = [
        ("train", train_ids, y_pred_train, y_true_train, "final"),
    ]
    if test_metrics is not None:
        block, target = indexed[(_MERGE_NODE_ID, "test", None)]
        if target is None:
            raise ValueError("separation concat has no native test target evidence")
        test_ids = block["sample_ids"]
        if test_ids != [unit["id"] for unit in target["unit_ids"]]:
            raise ValueError("separation test prediction/target identities disagree")
        y_pred_test = np.asarray(block["values"], dtype=float)
        y_true_test = np.asarray(target["values"], dtype=float)
        rows.extend([
            ("val", test_ids, y_pred_test, y_true_test, "test"),
            ("test", test_ids, y_pred_test, y_true_test, "test"),
        ])
    scores = {"train": train_metrics}
    if test_metrics is not None:
        scores.update(val=test_metrics, test=test_metrics)
    for partition, ids, y_pred, y_true, native_partition in rows:
        predictions.add_prediction(
            dataset_name=dataset_name, config_name=config_name, model_name=model_name,
            fold_id="final", refit_context="full_train", partition=partition,
            metric=metric, task_type=task_type, scores=scores,
            train_score=float(train_score),
            val_score=test_metrics.get(metric) if test_metrics is not None else None,
            test_score=test_metrics.get(metric) if test_metrics is not None else None,
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
    result._dagml_score_set = outcome["scores"]  # noqa: SLF001
    result._dagml_node_results = outcome["node_results"]  # noqa: SLF001
    result._dagml_refit_artifacts = artifacts  # noqa: SLF001
    return result


def run_by_source_auto_full_train(
    source_bodies: dict[str, list[Any]], y_steps: list[Any], spectro: Any, *,
    metric: str = "rmse", task_type: str = "regression", config_name: str = "",
    cli: str | None = None, venv_python: str | None = None,
    dataset_path: str | None = None, dataset_pickle: str | None = None,
    workdir: Any = None, random_state: int | None = None,
    train_sample_ids: list[int] | None = None,
) -> RunResult:
    """Fit source-local models once each for a by_source auto merge without CV."""
    execute = None
    if in_process_enabled():
        extension = importlib.import_module("dag_ml._dag_ml")
        execute = getattr(extension, "execute_phase_in_process", None)
        if not callable(execute):
            raise DagMlUnavailable("the installed DAG-ML runtime lacks execute_phase_in_process")

    import dag_ml

    from .cli_runner import data_bindings_for_nodes
    from .run_paths import _canonical_source_branch, _source_names

    n_sources = spectro.features_sources()
    names = _source_names(spectro, n_sources)
    if set(source_bodies) != set(names):
        raise DagMlUnsupported(f"by_source model names {list(source_bodies)!r} must match source names {names!r}")
    branches = [_canonical_source_branch([*y_steps, *source_bodies[name]], index) for index, name in enumerate(names)]
    dsl: dict[str, Any] = {
        "id": "nirs4all-by-source-auto-full-train",
        "steps": [{"kind": "branch", "mode": "duplication", "branches": branches}],
    }
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    model_ids = [next(node["id"] for node in branch["steps"] if node["kind"] == "model") for branch in branches]
    compiled_models = {node["id"] for node in graph["nodes"] if node["kind"] == "model"}
    if compiled_models != set(model_ids):
        raise DagMlUnsupported(f"by_source full training compiled models {compiled_models!r}, expected {model_ids!r}")

    identity = mint_identity(spectro)
    train = spectro.index_column("sample", {"partition": "train"})
    if train_sample_ids is not None:
        if not train_sample_ids or len(train_sample_ids) != len(set(train_sample_ids)) or not set(train_sample_ids) <= set(train):
            raise ValueError("full-training sample IDs must be a non-empty unique subset of the train partition")
        train = list(train_sample_ids)
    test = spectro.index_column("sample", {"partition": "test"})
    envelope = build_envelope(spectro, identity, sample_ints=train)
    if test:
        cohort_builder = getattr(dag_ml, "attach_predict_cohort_to_envelope", None)
        if not callable(cohort_builder):
            raise DagMlUnavailable("the installed DAG-ML runtime lacks the native test-cohort constructor")
        test_envelope = build_envelope(spectro, identity, sample_ints=test)
        envelope.update(cohort_builder(envelope, {
            "role": "external_test", "relations": test_envelope["coordinator_relations"],
            "target_names": target_names(spectro),
            "data_content_fingerprint": _array_content_fingerprint("X", spectro.x({"partition": "test"}, layout="2d")),
            "target_content_fingerprint": _array_content_fingerprint("y", spectro.y({"partition": "test"})),
        }).to_dict())
    dsl["data_bindings"] = data_bindings_for_nodes(model_ids, envelope)
    resolver = MaterializationResolver(spectro, identity)
    nodes = {node["id"]: node for node in graph["nodes"]}
    target_transform = next((node for node in graph["nodes"] if node["kind"] == "y_transform"), None)
    store: dict[int, Any] = {}

    def callback(task: dict[str, Any]) -> dict[str, Any]:
        return run_node(task, resolver, nodes.__getitem__, store, graph.get("edges", []), target_transform)

    warnings.warn(
        "Single-fold file: fitting each source model on its declared train IDs once; validation IDs are a test holdout, not cross-validation."
        if train_sample_ids is not None else
        "No splitter provided: fitting each source model on all training rows once; "
        "scores are training resubstitution or held-out test evaluation, not cross-validation.",
        NoSplitEvaluationWarning, stacklevel=2,
    )
    training_ids = [identity.to_wire(sample) for sample in train]
    if execute is None:
        if cli is None or venv_python is None or dataset_path is None or workdir is None:
            raise DagMlUnavailable("CLI by_source refit requires a DAG-ML CLI, Python adapter, and reloadable dataset")
        from pathlib import Path

        from .cli_runner import run_refit_phase_cli
        from .errors import _raise_run_failure
        from .in_process_runner import _load_subprocess_refit_artifacts

        cli_run = run_refit_phase_cli(
            dsl=dsl, envelope=envelope, graph=graph, training_sample_ids=training_ids,
            dataset_path=dataset_path, dataset_pickle=dataset_pickle,
            workdir=Path(workdir), dagml_cli=cli, venv_python=venv_python,
            random_state=random_state,
        )
        if cli_run["returncode"]:
            _raise_run_failure(cli_run, "full-training by_source CLI phase failed")
        outcome = json.loads(cli_run["phase_output"].read_text())
        artifacts = _load_subprocess_refit_artifacts(outcome["node_results"], cli_run["artifact_dir"])
    else:
        outcome = json.loads(execute(
            json.dumps(dsl), json.dumps(envelope), json.dumps(controller_manifests()), callback, "REFIT",
            training_sample_ids=training_ids,
        ))
        artifacts = _capture_refit_artifacts(outcome["node_results"], store)
    if outcome["phase"] != "REFIT":
        raise ValueError("by_source full-training runtime returned an unexpected phase")
    predictions = Predictions()
    evaluation: dict[str, Any] | None = None
    for index, (name, model_id) in enumerate(zip(names, model_ids, strict=True)):
        local = _project_full_train(
            outcome, identity, dataset_name=spectro.name, model_id=model_id,
            model_name=_model_name(source_bodies[name]), metric=metric,
            task_type=task_type, config_name=config_name, artifacts=artifacts,
        )
        evaluation = local.per_dataset[spectro.name]["evaluation"]
        for row in local.predictions.filter_predictions(load_arrays=True):
            row["branch_id"] = index
            row["branch_name"] = name
            predictions.extend_from_list([row])
    predictions.flush()
    result = RunResult(predictions=predictions, per_dataset={spectro.name: {
        "engine": "dag-ml", "execution_profile": "full_train", "evaluation": evaluation,
        "output_topology": "independent_by_source",
    }})
    result._dagml_score_set = outcome["scores"]  # noqa: SLF001
    result._dagml_node_results = outcome["node_results"]  # noqa: SLF001
    result._dagml_refit_artifacts = artifacts  # noqa: SLF001
    from .envelope import _numeric_feature_axis

    result._dagml_source_feature_axes = tuple(_numeric_feature_axis(spectro, index) for index in range(n_sources))  # noqa: SLF001
    return result


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
