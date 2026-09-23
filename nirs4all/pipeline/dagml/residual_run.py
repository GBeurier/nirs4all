"""Lower a residual model into base, learner, and native fusion graph nodes."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from nirs4all.api.result import RunResult
from nirs4all.operators.models.residual import ResidualModel
from nirs4all.pipeline.dagml_bridge import (
    _RESIDUAL_LEARNER_CONTROLLER_ID,
    _RESIDUAL_LEARNER_REF,
    _json_safe_params,
    _qualname,
    controller_manifests,
)

from .cli_runner import data_bindings_for_nodes, split_invocation_for
from .envelope import build_envelope
from .errors import DagMlUnsupported, _raise_run_failure
from .folds import _build_folds, _split_group_grain
from .identity import mint_identity
from .in_process_runner import run_cv_refit_bundle_router as run_cv_refit_bundle
from .result import _scores_to_run_result
from .run_paths import _canonical_branch, _canonical_branch_step, _supported_body_steps
from .steps import _is_split_step, _split_pipeline
from .training_controls import encode_training_controls


class ResidualImplicitCvWarning(UserWarning):
    """A residual model inferred a training-only CV for its OOF targets."""


def residual_operator(pipeline: list[Any]) -> ResidualModel | None:
    """Recognize the public single residual-model pipeline form."""
    residuals = []
    for step in pipeline:
        if not isinstance(step, dict):
            continue
        operator = step.get("model")
        if isinstance(operator, ResidualModel):
            residuals.append(operator)
        elif isinstance(step.get("residual"), dict):
            residuals.append(ResidualModel(**step["residual"]))
    if not residuals:
        return None
    if len(residuals) != 1 or not isinstance(pipeline[-1], dict) or not (
        isinstance(pipeline[-1].get("model"), ResidualModel) or isinstance(pipeline[-1].get("residual"), dict)
    ):
        raise DagMlUnsupported("residual model requires one terminal residual step")
    return residuals[0]


def run_residual_model(
    pipeline: list[Any], operator: ResidualModel, spectro: Any,
    dataset_arg: str, cli: str, venv_python: str, run_dir: Path,
    metric: str, task_type: str, *, dataset_pickle: str | None,
    config_name: str, random_state: int | None, refit: bool,
) -> RunResult:
    """Execute the native residual graph; host nodes only fit base/learner estimators."""
    if task_type != "regression":
        raise DagMlUnsupported("ResidualModel requires a regression target")
    _, splitter = _split_pipeline(pipeline)
    implicit_cv = splitter is None
    if splitter is None:
        from sklearn.model_selection import KFold

        splitter = KFold(n_splits=2, shuffle=True, random_state=0 if random_state is None else random_state)
        warnings.warn(
            "ResidualModel without a splitter uses an internal two-fold training-only CV to form residual targets; "
            "the held-out test set is evaluated only after full-training refit. Legacy used test rows as validation, "
            "so its validation score is not comparable.",
            ResidualImplicitCvWarning,
            stacklevel=2,
        )
    prefix = [step for step in pipeline[:-1] if not _is_split_step(step)]
    branch_positions = [index for index, step in enumerate(prefix) if isinstance(step, dict) and "branch" in step]
    source_concat = False
    distinct_source_steps: dict[str, list[Any]] | None = None
    if branch_positions:
        from .detect import _duplication_branch_bodies, _selected_duplication_feature_branches, _simple_duplication_merge_mode
        from .run_paths import _branch_merge_transformer_step

        branch = prefix[branch_positions[0]]["branch"]
        if isinstance(branch, dict) and branch.get("by_source") in (True, "auto"):
            from .detect import _is_source_concat_merge_step

            if len(prefix) != 2 or branch_positions != [0] or not _is_source_concat_merge_step(prefix[1]) or spectro.features_sources() < 2 or set(branch) != {"by_source", "steps"}:
                raise DagMlUnsupported("residual by_source prefix requires preprocessing and concat on multiple sources")
            if isinstance(branch["steps"], list) and branch["steps"]:
                prefix = branch["steps"]
                source_concat = True
            elif isinstance(branch["steps"], dict) and len(branch["steps"]) == spectro.features_sources():
                distinct_source_steps = branch["steps"]
                prefix = []
            else:
                raise DagMlUnsupported("residual by_source prefix needs shared or source-named preprocessing")
        else:
            if len(branch_positions) != 1 or branch_positions[0] + 1 >= len(prefix):
                raise DagMlUnsupported("residual branch prefix requires one duplication branch and feature merge")
            branch_index = branch_positions[0]
            merge_step = prefix[branch_index + 1]
            merge_mode = _simple_duplication_merge_mode(merge_step)
            if merge_mode not in {"features", "all"}:
                raise DagMlUnsupported("residual branch prefix requires merge='features' or merge='all'")
            branches = _duplication_branch_bodies(prefix[branch_index])
            if branches is None:
                raise DagMlUnsupported("residual feature merge requires duplication branch bodies")
            if merge_mode == "features":
                branches = _selected_duplication_feature_branches(branches, merge_step)
                if branches is None:
                    raise DagMlUnsupported("residual feature merge has an invalid branch selection")
            prefix = [*prefix[:branch_index], _branch_merge_transformer_step(branches, merge_mode), *prefix[branch_index + 2:]]
    prefix = _supported_body_steps(prefix)
    prefix_steps = [_canonical_branch_step(step, f"residual.prefix:{index}") for index, step in enumerate(prefix)]
    if any(step["kind"] not in {"transform", "y_transform"} for step in prefix_steps):
        raise DagMlUnsupported("residual prefix requires X or target preprocessing steps")
    target_steps = [(index, step) for index, step in enumerate(prefix_steps) if step["kind"] == "y_transform"]
    if len(target_steps) > 1:
        from sklearn.pipeline import Pipeline

        from .operator_routing import route_graph_node

        chain = Pipeline([(f"target_{index}", route_graph_node(step)) for index, step in target_steps])
        prefix_steps = [step for step in prefix_steps if step["kind"] != "y_transform"]
        first_index = target_steps[0][0]
        prefix_steps.insert(first_index, _canonical_branch_step({"y_processing": chain}, f"residual.prefix:{first_index}"))
    learner_finetune: dict[str, Any] = {}
    if operator.finetune_space:
        from .host_finetune import validate_host_finetune

        learner_finetune = validate_host_finetune(operator.finetune_space)

    import dag_ml

    identity = mint_identity(spectro)
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, set())
    groups = _split_group_grain(splitter, spectro, pool)
    envelope = build_envelope(spectro, identity, sample_ints=pool, group_by_sample=groups)
    source_preprocessing = None
    if distinct_source_steps is not None:
        from .run_paths import _source_preprocessing_metadata

        source_preprocessing = _source_preprocessing_metadata(distinct_source_steps, (envelope.get("plan") or {}).get("source_layout"))
    base_id = "branch:0.node:0"
    learner_id = "model:residual.learner"
    fusion_id = f"{learner_id}.residual_fusion"
    dsl = {
        "id": "nirs4all-residual-model",
        "inner_cv": {"kind": "kfold", "n_splits": 2, "shuffle": False, "seed": random_state},
        "steps": [
            *prefix_steps,
            {"kind": "branch", "mode": "duplication", "branches": [_canonical_branch([{"model": operator.base}], 0)]},
            {
                "kind": "merge_model", "id": learner_id,
                "operator": {"class": _qualname(operator.learner), "ref": _RESIDUAL_LEARNER_REF},
                "params": _json_safe_params(operator.learner),
                "include_original_data": True,
                "metadata": {
                    "controller_id": _RESIDUAL_LEARNER_CONTROLLER_ID,
                    "residual_target_execution": "nested_oof_v1",
                    "stacking_refit_oof": "partitioned_inner_v1",
                    "residual_lambda": operator.lam,
                    "residual_gate": operator.gate,
                    "residual_rli_threshold": operator.rli_threshold,
                    **({"nirs4all_train_params": encode_training_controls(operator.train_params, name="train_params")} if operator.train_params else {}),
                    **({
                        "nirs4all_finetune_params": learner_finetune,
                        "nirs4all_finetune_model_param_order": list(learner_finetune["model_params"]),
                    } if learner_finetune else {}),
                },
                **({"train_params": operator.train_params} if operator.train_params else {}),
            },
        ],
    }
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    if source_concat or source_preprocessing is not None:
        for node in graph["nodes"]:
            if node["kind"] == "model":
                if source_concat:
                    node["metadata"]["source_concat_x_chain"] = True
                else:
                    node["metadata"]["source_concat_preprocessing"] = source_preprocessing
    model_ids = {node["id"] for node in graph["nodes"] if node["kind"] == "model"}
    if model_ids != {base_id, learner_id} or fusion_id not in {node["id"] for node in graph["nodes"]}:
        raise ValueError("residual graph did not compile to its declared base, learner and fusion nodes")
    bindings = data_bindings_for_nodes([base_id, learner_id], envelope)
    bindings[1]["input_name"] = "x_original"
    dsl["data_bindings"] = bindings
    dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=len(folds))
    if groups:
        dsl["split_invocation"]["fold_set"]["sample_groups"] = {
            identity.to_wire(int(sample)): group for sample, group in groups.items()
        }
    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg,
        workdir=run_dir, dagml_cli=cli, venv_python=venv_python,
        selection_metric=metric, dataset_pickle=dataset_pickle, dataset=spectro,
        random_state=random_state, refit=refit or implicit_cv,
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml residual model run failed")
    result = _scores_to_run_result(
        outcome["scores"], spectro.name, operator.name, metric, task_type,
        producer=fusion_id, config_name=config_name, results=outcome["results"],
        identity=identity, refit_artifacts=outcome["refit_artifacts"],
    )
    if operator.gate == "auto":
        gate_records = outcome.get("residual_gates") or []
        if not isinstance(gate_records, list) or not gate_records or any(
            not isinstance(record, dict) or not isinstance(record.get("gate"), (int, float))
            for record in gate_records
        ):
            raise ValueError("native residual run omitted automatic gate calibration evidence")
        final_gates = [record["gate"] for record in gate_records if record.get("fold_id") is None]
        if (refit or implicit_cv) and len(final_gates) != 1:
            raise ValueError("native residual refit needs exactly one full-training automatic gate")
        gate = float(final_gates[0]) if final_gates else None
    else:
        gate = float(1.0 if operator.gate is False else operator.gate)
    result.per_dataset[spectro.name]["residual_replay"] = {
        "schema_version": 1,
        "producer_node": fusion_id,
        "base_producer_node": base_id,
        "learner_producer_node": learner_id,
        "lambda": float(operator.lam),
        "gate": gate,
        **({"gate_records": gate_records} if operator.gate == "auto" else {}),
        **({"implicit_training_cv": True} if implicit_cv else {}),
    }
    return result
