"""Named model views of one native nested-stacking execution outcome."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .public_batch import DagMLBatchResult


class NamedStackingResult(DagMLBatchResult):
    """Preserve named base/meta predictions and select their own fitted artifacts.

    The children are producer views of ONE DAG execution, not independently
    trained pipelines. Their canonical ScoreSet is the exact shared outcome.
    """

    def export(
        self, output_path: str | Path, format: str = "n4a", source: dict[str, Any] | None = None,
        chain_id: str | None = None, *, compatibility: str | None = None,
    ) -> Path:
        """Resolve a producer identity, then export that view's captured REFIT."""
        return self._source_run(source, chain_id).export(output_path, format, compatibility=compatibility)

    def export_model(
        self, output_path: str | Path, source: dict[str, Any] | None = None,
        format: str | None = None, fold: int | None = None, *, compatibility: str | None = None,
    ) -> Path:
        """Choose the actual producer before delegating model-format validation."""
        return self._source_run(source).export_model(output_path, format=format, fold=fold, compatibility=compatibility)


def project_named_stacking(
    outcome: dict[str, Any], *, branches: list[list[Any]], branch_names: list[str],
    base_model_ids: list[str], meta_node_id: str, meta_learner: Any,
    spectro: Any, identity: Any, metric: str, task_type: str, config_name: str,
    pipeline: list[Any], random_state: int | None,
) -> NamedStackingResult:
    """Project real producer scores/arrays and bind only that producer's capture."""
    from .envelope import target_names
    from .native_results import _producer_node_from_artifact_id
    from .result import _scores_to_run_result
    from .run_backend import _attach_export_spec
    from .steps import _model_name, _split_pipeline

    captures = outcome["refit_artifacts"]
    _, splitter = _split_pipeline(pipeline)
    children = []
    outer_fold_ids = {
        str(report["fold_id"]) for report in outcome["scores"].get("reports", [])
        if report.get("producer_node") == meta_node_id and report.get("fold_id") is not None
    }
    views: list[tuple[str, str, int | None, str | None, list[Any]]] = []
    for index, branch in enumerate(branches):
        # Graph order is lexical (branch:10 precedes branch:2), whereas names
        # and branch bodies retain the user's insertion order.
        producers = [producer for producer in base_model_ids if producer.startswith(f"branch:{index}.node:")]
        if len(producers) != 1:
            raise ValueError(f"Named stacking branch {index} must identify exactly one model producer")
        views.append((producers[0], _model_name(branch), index, branch_names[index], [splitter, *branch]))
    views.append((meta_node_id, type(meta_learner).__name__, None, None, pipeline))
    for producer, label, branch_id, branch_name, training_pipeline in views:
        own_captures = captures if producer == meta_node_id else [
            artifact for artifact in captures if _producer_node_from_artifact_id(artifact.get("artifact_id")) == producer
        ]
        if producer != meta_node_id and len(own_captures) != 1:
            raise ValueError(f"Named stacking producer {producer!r} does not identify one REFIT artifact")
        child = _scores_to_run_result(
            outcome["scores"], spectro.name, label, metric, task_type,
            producer=producer, config_name=config_name, results=outcome["results"],
            identity=identity, refit_artifacts=own_captures, report_fold_ids=outer_fold_ids,
        )
        child._dagml_target_names = target_names(spectro)
        _attach_export_spec(child, training_pipeline, spectro, config_name, random_state)
        for row in child.predictions._buffer:
            stacking_role = "meta" if producer == meta_node_id else "base"
            row["branch_id"] = branch_id
            row["branch_name"] = branch_name
            row["stacking_role"] = stacking_role
            row["result_metadata"] = {**(row.get("result_metadata") or {}), "dagml_producer_node": producer, "stacking_role": stacking_role}
        for metadata in child.per_dataset.values():
            metadata["producer_node"] = producer
            metadata["stacking_oof_execution"] = "nested_oof_v1"
        children.append(child)
    combined = NamedStackingResult(children)
    combined._dagml_score_set = outcome["scores"]
    combined._dagml_refit_artifacts = captures
    combined._dagml_node_results = outcome["results"]
    return combined
