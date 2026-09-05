"""Publish already-executed DAG results into the library workspace.

This is a storage projection, not a second executor. It never fits operators,
recalculates scores, or invents a native ScoreSet for host-only execution paths.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from nirs4all.api.result import RunResult
from nirs4all.data import Predictions
from nirs4all.pipeline.config.component_serialization import serialize_component
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


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
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in result.predictions.filter_predictions(load_arrays=True):
        groups[(str(row.get("dataset_name", spectro.name)), str(row.get("config_name", "")))].append(row)

    with WorkspaceStore(workspace_path) as store:
        owns_run = store_run_id is None
        run_id = store_run_id if store_run_id is not None else store.begin_run(
            name=name,
            config={"engine": result.execution_engine, "pipeline": template, "report_naming": report_naming},
            datasets=[{"name": spectro.name, "hash": spectro.content_hash()}],
        )
        try:
            with store.transaction():
                if project is not None:
                    store.set_run_project(run_id, store.get_or_create_project(project))
                for (dataset_name, config_name), rows in groups.items():
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
                    ) -> str:
                        key = (str(row.get("model_name", "")), str(row.get("branch_id", "")), str(row.get("branch_name", "")))
                        if key not in chain_ids:
                            chain_ids[key] = store.save_chain(
                                pipeline_id, [], -1, str(row.get("model_classname", "")),
                                str(row.get("preprocessings", "")), "score_only", {}, {},
                                branch_path=row.get("branch_path"), dataset_name=dataset_name,
                            )
                        return chain_ids[key]

                    projection = Predictions()
                    projection._buffer = rows  # noqa: SLF001 -- exact existing rows, no score/array reconstruction
                    projection.flush(pipeline_id=pipeline_id, store=store, chain_id_resolver=chain_for)
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
