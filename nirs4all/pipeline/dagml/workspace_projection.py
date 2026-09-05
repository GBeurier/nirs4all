"""Publish already-executed DAG results into the library workspace.

This is a storage projection, not a second executor. It never fits operators,
recalculates scores, or invents a native ScoreSet for host-only execution paths.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from nirs4all.api.result import RunResult
from nirs4all.data import Predictions
from nirs4all.pipeline.config.component_serialization import serialize_component
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


def _projection_group(row: dict[str, Any], dataset_name: str) -> tuple[str, str, str, str]:
    """Resolve native source identity without interpreting user-facing names."""
    metadata = row.get("result_metadata") or {}
    source = metadata.get("dagml_projection") if isinstance(metadata, dict) else None
    if isinstance(source, dict) and source.get("schema") == "nirs4all.dagml-result-projection.v1":
        return (str(row.get("dataset_name", dataset_name)), str(source["config_name"]),
                str(source.get("variant_id") or ""), str(source.get("producer_node") or ""))
    return str(row.get("dataset_name", dataset_name)), str(row.get("config_name", "")), "", ""


def publish_workspace_result(
    result: RunResult,
    pipeline: Any,
    spectro: Any,
    workspace_path: Path,
    *,
    name: str,
    project: str | None,
    report_naming: str,
    store_run_id: str | None = None,
) -> str:
    """Persist exact prediction rows and durable provenance, returning the run ID.

    Native fitted artifacts remain in the separately verified native results
    directory. Chains without captured fold artifacts are explicitly score-only;
    this projection must not claim they can replay a CV model.
    """
    template = serialize_component(pipeline)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in result.predictions.filter_predictions(load_arrays=True):
        groups[_projection_group(row, spectro.name)].append(row)

    with WorkspaceStore(workspace_path) as store:
        owns_run = store_run_id is None
        run_id = store_run_id if store_run_id is not None else store.begin_run(
            name=name,
            config={"engine": result.execution_engine, "pipeline": template, "report_naming": report_naming},
            datasets=[{"name": spectro.name, "hash": spectro.content_hash()}],
        )
        try:
            with store.transaction():
                captured_step: dict[str, Any] | None = None
                captured = None
                selected_row = result.best
                selected_group = _projection_group(selected_row, spectro.name)
                selected_model = tuple(str(selected_row.get(field, "")) for field in ("model_name", "branch_id", "branch_name"))
                if result._dagml_results_dir is not None and len(result._dagml_refit_artifacts) == 1:
                    from .native_results import read_native_results

                    artifacts = read_native_results(result._dagml_results_dir)["artifacts"]
                    if len(artifacts) == 1:
                        captured = artifacts[0]
                        estimator_type = type(captured["estimator"])
                        class_name = f"{estimator_type.__module__}.{estimator_type.__qualname__}"
                        fitted_steps = getattr(captured["estimator"], "steps", None)
                        model_operator = fitted_steps[-1][1] if fitted_steps else captured["estimator"]
                        model_type = type(model_operator)
                        model_class = f"{model_type.__module__}.{model_type.__qualname__}"
                        preprocessings = ">".join(type(operator).__name__ for _, operator in fitted_steps[:-1]) if fitted_steps else ""
                        captured_step = {
                            "step_idx": 0, "operator_class": class_name, "params": {}, "stateless": False,
                            "dagml_host_replay": {
                                "schema": "nirs4all.dagml-workspace-refit.v1",
                                "native_artifact_id": captured["artifact_id"], "target_names": result._dagml_target_names,
                                "model_class": model_class, "preprocessings": preprocessings,
                                "scope": "full_training_refit", "cv_artifacts_available": False,
                            },
                        }
                        for key in ("relation_replay_manifest", "relation_materialization_manifest"):
                            recorded = [metadata[key] for metadata in result.per_dataset.values() if isinstance(metadata.get(key), dict)]
                            if recorded and all(value == recorded[0] for value in recorded):
                                captured_step["dagml_host_replay"][key] = recorded[0]
                if project is not None:
                    store.set_run_project(run_id, store.get_or_create_project(project))
                for group, rows in groups.items():
                    dataset_name, config_name, _, _ = group
                    pipeline_id = store.begin_pipeline(
                        run_id, config_name or name, template, [], dataset_name,
                        spectro.content_hash(), original_template=template,
                    )
                    # Keep each model/branch identity distinct; a row-only chain
                    # cannot accidentally select another branch's captured model.
                    chain_ids: dict[tuple[str, str, str], str] = {}

                    def chain_for(
                        row: dict[str, Any], *, chain_ids: dict[tuple[str, str, str], str] = chain_ids,
                        pipeline_id: str = pipeline_id, dataset_name: str = dataset_name,
                        group: tuple[str, str, str, str] = group,
                    ) -> str:
                        key = (str(row.get("model_name", "")), str(row.get("branch_id", "")), str(row.get("branch_name", "")))
                        if key not in chain_ids:
                            replay_step: dict[str, Any] = {}
                            if captured_step is not None and group == selected_group and key == selected_model:
                                # CV and REFIT rows share one native source chain;
                                # other models/branches never borrow its artifact.
                                stored_id = store.save_artifact(captured, captured_step["operator_class"], "model", "joblib")
                                fingerprint = "sha256:" + hashlib.sha256(store.get_artifact_path(stored_id).read_bytes()).hexdigest()
                                replay_step = {
                                    **captured_step, "artifact_id": stored_id,
                                    "dagml_host_replay": {**captured_step["dagml_host_replay"], "artifact_fingerprint": fingerprint},
                                }
                                row["model_classname"] = row.get("model_classname") or captured_step["dagml_host_replay"]["model_class"]
                                row["preprocessings"] = row.get("preprocessings") or captured_step["dagml_host_replay"]["preprocessings"]
                            chain_ids[key] = store.save_chain(
                                pipeline_id, [replay_step] if replay_step else [], 0 if replay_step else -1,
                                str(row.get("model_classname") or replay_step.get("operator_class", "")),
                                str(row.get("preprocessings", "")), "refit_only" if replay_step else "score_only",
                                {"final": replay_step["artifact_id"]} if replay_step else {}, {},
                                branch_path=row.get("branch_path"), dataset_name=dataset_name,
                            )
                        return chain_ids[key]

                    projection = Predictions()
                    projection._buffer = rows  # noqa: SLF001 -- exact existing rows, no score/array reconstruction
                    projection.flush(pipeline_id=pipeline_id, store=store, chain_id_resolver=chain_for)
                    by_id = {row["id"]: row for row in rows}
                    for original_row in result.predictions._buffer:
                        if original_row.get("id") in by_id:
                            original_row["chain_id"] = by_id[original_row["id"]]["chain_id"]
                            original_row["workspace_path"] = str(workspace_path)
                            original_row["model_classname"] = by_id[original_row["id"]].get("model_classname", "")
                            original_row["preprocessings"] = by_id[original_row["id"]].get("preprocessings", "")
                    # Workspace readers consume the canonical denormalized chain
                    # summary, not the raw prediction rows. The native runner
                    # has no legacy executor to perform this publication step.
                    store.bulk_update_chain_summaries(list(chain_ids.values()))
                    best = projection.top(1)
                    assert not isinstance(best, dict)
                    selected: Any = best[0] if best else rows[0]
                    store.complete_pipeline(
                        pipeline_id, selected.get("val_score"), selected.get("test_score"),
                        str(selected.get("metric", "")), 0,
                    )
                summary = {
                    "execution_engine": result.execution_engine,
                    "num_predictions": result.num_predictions,
                    "native_score_set_available": result._dagml_score_set is not None,  # noqa: SLF001
                    "native_results_dir": str(result._dagml_results_dir) if result._dagml_results_dir else None,  # noqa: SLF001
                    "cv_best_score": result.cv_best_score if math.isfinite(result.cv_best_score) else None,
                    "evaluation": {dataset: metadata["evaluation"] for dataset, metadata in result.per_dataset.items() if "evaluation" in metadata},
                }
                if owns_run:
                    store.complete_run(run_id, summary)
        except BaseException as error:
            if owns_run:
                store.fail_run(run_id, str(error))
            raise
    result._workspace_path = workspace_path  # noqa: SLF001 -- detached result retains its durable workspace
    for metadata in result.per_dataset.values():
        if isinstance(metadata, dict):
            metadata["run_id"] = run_id
            metadata["workspace_path"] = str(workspace_path)
    return run_id
