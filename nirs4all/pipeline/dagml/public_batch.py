"""Compose independent public DAG requests while retaining their fitted results."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from nirs4all.api.result import RunResult
from nirs4all.data.predictions import Predictions

from .errors import DagMlExportRefusal
from .workspace_projection import _projection_group


def _projection_identity(row: dict[str, Any]) -> tuple[str, str, str, str, str, str, str] | None:
    """Return the artifact-owning DAG identity of one projected prediction."""
    dataset_name = str(row.get("dataset_name") or "")
    model_name = str(row.get("model_name") or "")
    branch_id = str(row.get("branch_id") or "")
    branch_name = str(row.get("branch_name") or "")
    identity = (
        *_projection_group(row, dataset_name),
        model_name,
        branch_id,
        branch_name,
    )
    return identity if any(identity) else None


def _validate_leaf_export_selection(
    result: RunResult,
    source: dict[str, Any] | None,
    chain_id: str | None,
) -> None:
    """Refuse a selector that does not identify the leaf's captured refit artifact."""
    if source is None and chain_id is None:
        return

    canonical = result.best
    canonical_identity = _projection_identity(canonical)
    if canonical_identity is None:
        raise DagMlExportRefusal(
            "batch export",
            "the selected leaf has no unambiguous native artifact identity",
            mitigation="Select an exportable refit row with its prediction id",
        )

    requested_identities: list[tuple[str, str, str, str, str, str, str]] = []
    if source is not None:
        identifier = source.get("id") or source.get("prediction_id")
        resolved = result.predictions.get_prediction_by_id(identifier, load_arrays=False) if identifier else None
        identity = _projection_identity(resolved) if resolved is not None else None
        if identity is None:
            raise DagMlExportRefusal(
                "batch export",
                "source does not resolve to one native prediction in the selected leaf",
                mitigation="Pass a prediction row returned by this batch result",
            )
        requested_identities.append(identity)
    if chain_id is not None:
        chain_rows = result.predictions.filter_predictions(chain_id=chain_id, load_arrays=False)
        identities = {_projection_identity(row) for row in chain_rows}
        if not chain_rows or None in identities or len(identities) != 1:
            raise DagMlExportRefusal(
                "batch export",
                "chain_id does not resolve to one native artifact identity in the selected leaf",
                mitigation="Pass the prediction id of the exportable refit row",
            )
        requested_identities.extend(identity for identity in identities if identity is not None)

    if any(identity != canonical_identity for identity in requested_identities):
        raise DagMlExportRefusal(
            "batch export",
            "the requested prediction does not own the leaf's captured refit artifact",
            mitigation="Export the selected refit winner or retrain the requested variant explicitly",
        )

    artifact_variants = {
        str(artifact.get("artifact_id")).split(":nirs4all:refit:", 1)[1]
        for artifact in result._dagml_refit_artifacts
        if ":nirs4all:refit:" in str(artifact.get("artifact_id"))
    }
    if canonical_identity[2] and artifact_variants and artifact_variants != {canonical_identity[2]}:
        raise DagMlExportRefusal(
            "batch export",
            "the canonical prediction identity does not match the captured refit artifact",
            mitigation="Rerun training to produce a consistent native result",
        )


class DagMLBatchResult(RunResult):
    """A normal result view whose child runs retain their own artifact identities."""

    def __init__(self, results: list[RunResult]) -> None:
        self._batch_results = tuple(results)
        predictions = Predictions()
        per_dataset: dict[str, Any] = {}
        for result in results:
            predictions.merge_predictions(result.predictions)
            for name, info in result.per_dataset.items():
                if name not in per_dataset:
                    per_dataset[name] = {**info, "runs": [info]}
                    if isinstance(info.get("run_predictions"), Predictions):
                        per_dataset[name]["run_predictions"] = Predictions()
                        per_dataset[name]["run_predictions"].merge_predictions(info["run_predictions"])
                else:
                    aggregate = per_dataset[name]
                    aggregate["runs"].append(info)
                    if aggregate.get("engine") != info.get("engine"):
                        aggregate["engine"] = "mixed"
                    if isinstance(info.get("run_predictions"), Predictions) and isinstance(aggregate.get("run_predictions"), Predictions):
                        aggregate["run_predictions"].merge_predictions(info["run_predictions"])
        super().__init__(predictions=predictions, per_dataset=per_dataset)

    @property
    def runs(self) -> tuple[RunResult, ...]:
        """Individual results in pipeline-major, dataset-minor execution order."""
        return self._batch_results

    def _source_run(self, source: dict[str, Any] | None, chain_id: str | None = None) -> RunResult:
        selected = self.best if source is None else source
        identifier = selected.get("id") or selected.get("prediction_id")
        if chain_id is not None:
            matches = [result for result in self.runs if result.predictions.filter_predictions(chain_id=chain_id)]
        elif identifier:
            matches = [result for result in self.runs if result.predictions.get_prediction_by_id(identifier, load_arrays=False) is not None]
        else:
            raise ValueError("Cannot identify the source run for this batch export")
        if len(matches) != 1:
            raise ValueError("Batch export must identify exactly one source run")
        return matches[0]

    def export(
        self, output_path: str | Path, format: str = "n4a", source: dict[str, Any] | None = None,
        chain_id: str | None = None, *, compatibility: str | None = None,
    ) -> Path:
        """Export from the selected child's actual artifacts, never refit a batch."""
        selected = self._source_run(source, chain_id)
        if isinstance(selected, DagMLBatchResult):
            return selected.export(
                output_path,
                format,
                source,
                chain_id,
                compatibility=compatibility,
            )
        _validate_leaf_export_selection(selected, source, chain_id)
        return selected.export(output_path, format, compatibility=compatibility)

    def export_model(
        self, output_path: str | Path, source: dict[str, Any] | None = None,
        format: str | None = None, fold: int | None = None, *, compatibility: str | None = None,
    ) -> Path:
        """Delegate lightweight export to the uniquely selected child result."""
        selected = self._source_run(source)
        if isinstance(selected, DagMLBatchResult):
            return selected.export_model(
                output_path,
                source,
                format,
                fold,
                compatibility=compatibility,
            )
        _validate_leaf_export_selection(selected, source, None)
        return selected.export_model(output_path, format=format, fold=fold, compatibility=compatibility)

    def close(self) -> None:
        """Release every child resource; session-owned resources remain shared."""
        for result in self._batch_results:
            result.close()


def run_dagml_public(pipeline: Any, dataset: Any, **options: Any) -> RunResult:
    """Apply the public cartesian-product semantics using only DAG child runs.

    Reuse the public distinction between a nested step and a pipeline batch.
    No exception is caught and retried, no estimator is fitted here, and no
    synthetic ScoreSet combines independently identified DAG runs.
    """
    from nirs4all.api.run import _is_single_dataset, _is_single_pipeline, _normalize_to_list
    from nirs4all.data.config import DatasetConfigs

    from .run_backend import run_via_dagml

    def run_child(single_pipeline: Any, single_dataset: Any, child_options: dict[str, Any]) -> RunResult:
        from nirs4all.pipeline.config.component_serialization import deserialize_component

        from .dataset import _materialize_dataset
        from .sequential_models import sequential_model_pipelines, share_model_folds

        successive = sequential_model_pipelines(deserialize_component(single_pipeline))
        if successive is None:
            return run_via_dagml(single_pipeline, single_dataset, **child_options)
        spectro = _materialize_dataset(single_dataset)
        # A request-local FoldSet is resolved once before any child executes.
        # Recursion only expands the now single-model requests, never retries.
        return run_dagml_public(share_model_folds(successive, spectro), spectro, **child_options)

    pipelines = _normalize_to_list(pipeline, _is_single_pipeline)
    datasets = _normalize_to_list(dataset, _is_single_dataset)
    if isinstance(dataset, DatasetConfigs) and len(dataset.configs) > 1:
        datasets = dataset.get_datasets()
    from nirs4all.pipeline.config.component_serialization import deserialize_component

    from .full_train_variants import expand_full_train_variants
    from .steps import _is_split_step

    pipeline_entries: list[tuple[Any, str, str | None]] = []
    for index, single_pipeline in enumerate(pipelines):
        base_name = options.get("name", "")
        pipeline_name = f"{base_name}_p{index}" if base_name else f"pipeline_{index}" if len(pipelines) > 1 else base_name
        runtime_steps = deserialize_component(single_pipeline)
        variants = []
        if isinstance(runtime_steps, list) and not any(_is_split_step(step) for step in runtime_steps):
            variants = expand_full_train_variants(runtime_steps, name=pipeline_name)
        if len(variants) > 1:
            pipeline_entries.extend((steps, pipeline_name, variant_name) for steps, variant_name in variants)
        else:
            pipeline_entries.append((single_pipeline, pipeline_name, None))
    if len(pipeline_entries) == len(datasets) == 1:
        return run_child(pipelines[0], datasets[0], options)
    if not pipelines or not datasets:
        raise ValueError("A batch requires at least one pipeline and one dataset")
    results: list[RunResult] = []
    scratch_root = Path(options["workdir"]) / f"batch-{uuid4().hex}" if options.get("workdir") is not None else None
    try:
        for index, (single_pipeline, pipeline_name, resolved_config_name) in enumerate(pipeline_entries):
            for dataset_index, single_dataset in enumerate(datasets):
                child_options = {**options, "name": pipeline_name}
                if resolved_config_name is not None:
                    child_options["resolved_config_name"] = resolved_config_name
                if scratch_root is not None:
                    child_options["workdir"] = scratch_root / f"pipeline-{index}-dataset-{dataset_index}"
                results.append(run_child(single_pipeline, single_dataset, child_options))
        aggregate = DagMLBatchResult(results)
        if options.get("session") is not None:
            # Keep the logical batch as the Session's result, not whichever
            # child happened to execute last. Individual histories remain exact.
            options["session"]._last_result = aggregate
        return aggregate
    except BaseException:
        for result in results:
            result.close()
        raise
