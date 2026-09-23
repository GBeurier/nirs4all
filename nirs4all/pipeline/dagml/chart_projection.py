"""Presentation-only charts from scored DAG runs and their captured refit transforms.

No estimator is fitted here. A processed spectrum is explicitly a full-training
REFIT view, not an out-of-fold observation or an independent validation result.
Every image has an adjacent HTML text alternative and downloadable numeric data.
"""

import copy
import csv
import html
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import numpy as np


def validate_chart_projection(pipeline: list[Any], spectro: Any) -> None:
    """Check that requested chart stages have an unambiguous captured prefix."""
    from .detect import _is_augmentation_step
    from .run_backend import _is_chart_step
    from .steps import _is_split_step

    uncertain_stage = False
    transformed = False
    augmentation_seen = False
    has_augmentation = any(_is_augmentation_step(step) for step in pipeline)
    for index, step in enumerate(pipeline):
        if _is_chart_step(step):
            materialized_between_augmentations = augmentation_seen and any(
                _is_augmentation_step(later) for later in pipeline[index + 1:]
            )
            if (uncertain_stage or (transformed and spectro.is_multi_source() and not materialized_between_augmentations)
                    or (has_augmentation and transformed and not augmentation_seen)):
                raise NotImplementedError("This chart stage needs a captured branch/source snapshot; a raw-data substitute would be misleading.")
        elif _is_split_step(step) or step is None:
            continue
        elif _is_augmentation_step(step):
            # Every augmentation captures its own full-train stage during the
            # actual materialization/refit pass, including fold-local runs.
            transformed = False
            augmentation_seen = True
        elif isinstance(step, dict):
            if set(step) == {"preprocessing"}:
                transformed = True
            elif "exclude" in step:
                # Exclusion is resolved once by the scored run and captured as
                # sample IDs for presentation at this precise pipeline stage.
                continue
            elif not ("model" in step or "y_processing" in step):
                uncertain_stage = True
        elif hasattr(step, "predict"):
            continue
        elif hasattr(step, "transform"):
            transformed = True
        else:
            uncertain_stage = True


def _folds_from_scores(result: Any) -> list[tuple[list[int], list[int]]]:
    """Use scored row membership, never invoke a random splitter a second time."""
    groups: dict[str, dict[str, list[int]]] = {}
    selected_config = None
    for row in result.predictions.filter_predictions(load_arrays=True):
        fold = str(row.get("fold_id", ""))
        if fold in {"", "final", "avg", "w_avg"} or row.get("partition") not in {"train", "val"}:
            continue
        config = (row.get("config_name"), row.get("model_name"))
        if selected_config is None:
            selected_config = config
        if config == selected_config:
            groups.setdefault(fold, {})[row["partition"]] = list(row.get("sample_indices", []))
    return [(group["train"], group["val"]) for group in groups.values() if "train" in group and "val" in group]


def _write_alternative(directory: Path, stem: str, snapshot: Any, context: Any, summary: str, image_name: str, *,
                       include_excluded: bool, source_index: int | None = None, color_column: str | None = None,
                       plotted_groups: dict[str, list[int]] | None = None, include_augmented: bool = True,
                       processing_indices: list[int] | None = None) -> None:
    """Expose exact plotted inputs without requiring interpretation of colors."""
    sample_indices = snapshot._indexer.x_indices(context.selector, include_augmented=include_augmented, include_excluded=include_excluded)
    arrays = snapshot.x(context.selector, "3d", False, include_augmented=include_augmented, include_excluded=include_excluded)
    arrays = arrays if isinstance(arrays, list) else [arrays]
    targets = np.asarray(snapshot.y(context, include_augmented=include_augmented, include_excluded=include_excluded)).reshape(len(sample_indices), -1)
    origins = [snapshot._indexer.get_origin_for_sample(int(sample_id)) for sample_id in sample_indices]
    partitions = {int(row["sample"]): str(row["partition"]) for row in snapshot._indexer.df.select("sample", "partition").iter_rows(named=True)}
    colors = None
    if color_column is not None:
        colors = np.asarray(snapshot.metadata_column(color_column, context.selector, include_augmented=include_augmented)).reshape(len(sample_indices), -1)
    data_name = f"{stem}.csv"
    excluded = {
        int(row["sample"]): str(row.get("exclusion_reason") or "")
        for row in snapshot._indexer.get_excluded_samples(context.selector).to_dicts()
    }
    plotted_in: dict[int, list[str]] = {}
    if plotted_groups is not None:
        for group, ids in plotted_groups.items():
            for sample_id in ids:
                plotted_in.setdefault(int(sample_id), []).append(group)
    with (directory / data_name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", "origin_sample_index", "partition", "synthetic", "excluded", "exclusion_reason", "source", "processing", "feature_index", "value",
                         *[f"target_{i}" for i in range(targets.shape[1])], *([f"color_{color_column}_{i}" for i in range(colors.shape[1])] if colors is not None else []),
                         *(["plotted_in"] if plotted_groups is not None else [])])
        for source, array in enumerate(arrays):
            if source_index is not None and source != source_index:
                continue
            for sample, sample_data in enumerate(array):
                sample_id = int(sample_indices[sample])
                if plotted_groups is not None and sample_id not in plotted_in:
                    continue
                for processing, values in enumerate(sample_data):
                    if processing_indices is not None and processing not in processing_indices:
                        continue
                    for feature, value in enumerate(values):
                        origin = origins[sample]
                        writer.writerow([sample_id, origin, partitions.get(sample_id, ""), origin is not None and origin != sample_id,
                                         sample_id in excluded, excluded.get(sample_id, ""), source, processing, feature,
                                         float(value), *targets[sample].tolist(), *(colors[sample].tolist() if colors is not None else []),
                                         *(["; ".join(plotted_in[sample_id])] if plotted_groups is not None else [])])
    (directory / f"{stem}.json").write_text(json.dumps({"summary": summary, "folds": snapshot.folds,
                                                        **({"plotted_groups": plotted_groups} if plotted_groups is not None else {})},
                                                       indent=2), encoding="utf-8")
    excluded_table = ""
    if excluded:
        excluded_rows = "".join(
            f"<tr><th scope=\"row\">{sample_id}</th><td>{html.escape(reason)}</td></tr>"
            for sample_id, reason in sorted(excluded.items())
        )
        excluded_table = (
            "<table><caption>Excluded sample IDs and reasons</caption>"
            "<thead><tr><th scope=\"col\">Sample ID</th><th scope=\"col\">Reason</th></tr></thead>"
            f"<tbody>{excluded_rows}</tbody></table>"
        )
    (directory / f"{stem}.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>DAG chart</title>'
        f'<main><h1>DAG run chart</h1><p>{html.escape(summary)}</p>'
        f'<p><a href="{html.escape(data_name)}">Download exact numeric inputs (CSV)</a></p>'
        f'<p><a href="{html.escape(stem)}.json">Read scored fold memberships and methodology (JSON)</a></p>'
        f'{excluded_table}'
        f'<img src="{html.escape(image_name)}" alt="{html.escape(summary)}"></main></html>',
        encoding="utf-8",
    )


def render_run_charts(result: Any, pipeline: list[Any], spectro: Any, *, original_spectro: Any | None = None,
                      pre_holdout_spectro: Any | None = None, file_holdout_lowered: bool = False, workspace_path: Path | None,
                      save_charts: bool, plots_visible: bool, verbose: int) -> list[str]:
    """Reuse library chart presenters on immutable snapshots of fitted DAG state."""
    from nirs4all.controllers.registry import CONTROLLER_REGISTRY
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext, StepMetadata
    from nirs4all.pipeline.steps.parser import StepParser

    from .detect import _is_augmentation_step
    from .run_backend import _is_chart_step
    from .steps import _is_split_step

    if not (save_charts or plots_visible) or not any(_is_chart_step(step) for step in pipeline):
        return []
    artifacts = result._dagml_refit_artifacts
    artifact = artifacts[0] if artifacts else None
    estimator = artifact["estimator"] if artifact else None
    fitted_steps = list(getattr(estimator, "steps", [])[:-1])
    directory = workspace_path / "charts" / uuid4().hex if save_charts and workspace_path is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
    runtime = RuntimeContext(step_runner=SimpleNamespace(verbose=verbose, plots_visible=plots_visible, _figure_refs=[]))
    prefix = 0
    after_split = False
    processed_target = False
    augmentation_count = 0
    prefix_at_last_augmentation = 0
    augmentation_snapshots = getattr(result, "_dagml_chart_aug_snapshots", None)
    transform_snapshots = getattr(result, "_dagml_chart_transform_snapshots", None) or {}
    exclusion_stages = getattr(spectro, "_dagml_exclusion_chart_stages", None) or []
    expected_augmentations = sum(_is_augmentation_step(step) for step in pipeline)
    if expected_augmentations and (augmentation_snapshots is None or len(augmentation_snapshots) != expected_augmentations):
        raise RuntimeError("Chart augmentation stages are missing from the scored full-training pass.")
    output_paths: list[str] = []
    exclusion_count = 0
    for index, step in enumerate(pipeline):
        if not _is_chart_step(step):
            if _is_split_step(step):
                after_split = True
            elif isinstance(step, dict) and "exclude" in step:
                exclusion_count += 1
            elif _is_augmentation_step(step):
                augmentation_count += 1
                prefix_at_last_augmentation = prefix
            elif isinstance(step, dict) and "y_processing" in step:
                processed_target = True
            elif (isinstance(step, dict) and set(step) == {"preprocessing"}) or (not isinstance(step, dict) and hasattr(step, "transform") and not hasattr(step, "predict")):
                prefix += 1
            continue
        materialized_stage = transform_snapshots.get((augmentation_count, prefix)) if augmentation_count else None
        snapshot_source = materialized_stage if materialized_stage is not None else (
            cast(list[Any], augmentation_snapshots)[augmentation_count - 1] if augmentation_count else (
                pre_holdout_spectro if pre_holdout_spectro is not None and not after_split else (original_spectro or spectro)
            )
        )
        snapshot = copy.deepcopy(snapshot_source)
        if exclusion_count:
            if len(exclusion_stages) < exclusion_count:
                raise RuntimeError("Chart exclusion stages are missing from the scored training pass.")
            for sample_ids, reason, cascade in exclusion_stages[:exclusion_count]:
                if sample_ids:
                    snapshot._indexer.mark_excluded(sample_ids, reason=reason, cascade_to_augmented=cascade)  # noqa: SLF001
        snapshot.set_folds(_folds_from_scores(result) if after_split else [])
        pending_prefix = 0 if materialized_stage is not None else (prefix - prefix_at_last_augmentation if augmentation_count else prefix)
        if pending_prefix:
            if pending_prefix > len(fitted_steps):
                raise RuntimeError("Chart transform prefix is missing from the scored refit artifact.")
            values = np.asarray(snapshot.x({}, layout="2d"))
            for _, transformer in fitted_steps[:pending_prefix]:
                values = transformer.transform(values)
            snapshot.add_merged_features(values, processing_name=f"refit_stage_{pending_prefix}")
        parsed = StepParser().parse(step)
        context = ExecutionContext(metadata=StepMetadata(keyword=parsed.keyword, step_id=str(index)))
        exclusion_chart = parsed.keyword in {"exclusion_chart", "chart_exclusion"}
        config = step.get(parsed.keyword, {}) if isinstance(step, dict) else {}
        chart_partition = config.get("partition", "train") if exclusion_chart and isinstance(config, dict) else None
        context = context.with_partition(chart_partition if exclusion_chart else ("train" if augmentation_count else None))
        context = context.with_processing([snapshot.features_processings(source) for source in range(snapshot.features_sources())])
        if processed_target and artifact and artifact["y_transform"] is not None:
            target = np.asarray(snapshot.y({})).reshape(snapshot.num_samples, -1)
            snapshot.add_processed_targets("chart_refit", artifact["y_transform"].transform(target))
            context = context.with_y("chart_refit")
        controller = next(cls for cls in CONTROLLER_REGISTRY if cls.__module__.startswith("nirs4all.controllers.charts.") and cls.matches(step, parsed.operator, parsed.keyword))
        _, output = controller().execute(parsed, snapshot, context, runtime)
        color_column = parsed.keyword[5:] if parsed.keyword.startswith("fold_") and parsed.keyword != "fold_chart" else None
        scope = ("full-training REFIT augmentation view; not out-of-fold features; observed and synthetic augmentation features"
                 if augmentation_count else ("captured full-training REFIT transforms; not out-of-fold features" if prefix else "original observed features"))
        if file_holdout_lowered and augmentation_count:
            scope += "; the single-file test holdout was applied before augmentation, so its rows and synthetic children were excluded from fitting"
        target_scope = "captured REFIT target transform" if processed_target else "original numeric targets"
        include_augmented = output.metadata.get("include_augmented", True)
        plotted_count = len(snapshot._indexer.x_indices(context.selector, include_augmented=include_augmented, include_excluded=exclusion_chart))
        if exclusion_chart:
            excluded_count = len(snapshot._indexer.get_excluded_samples(context.selector))
            chart_subject = f"{parsed.keyword}: {plotted_count - excluded_count} included and {excluded_count} excluded samples in {chart_partition or 'all'} partition"
        else:
            chart_subject = f"{parsed.keyword}: {plotted_count} samples"
        color_scope = f" Color coding uses metadata column {color_column!r}." if color_column is not None else ""
        if not include_augmented and augmentation_count:
            scope = ("full-training REFIT augmentation stage; not out-of-fold features; "
                     "only observed samples are included in this chart's envelopes")
        summary = f"{chart_subject}; {scope}; {target_scope}. {len(snapshot.folds)} scored cross-validation folds.{color_scope} Numeric inputs and fold memberships are supplied alongside the image."
        if exclusion_chart and isinstance(config, dict) and config.get("title"):
            summary = f"{config['title']}. {summary}"
        if directory is None:
            print(summary)
        else:
            include_excluded = exclusion_chart or (bool(config.get("include_excluded", False)) if isinstance(config, dict) else False)
            for number, (data, _, extension) in enumerate(output.outputs):
                stem = f"step_{index:03d}_{number:02d}"
                image_path = directory / f"{stem}.{extension}"
                image_path.write_bytes(data)
                source_index = number if controller.use_multi_source() and len(output.outputs) == snapshot.features_sources() else None
                groups_by_source = output.metadata.get("plotted_groups")
                plotted_groups = groups_by_source[number] if groups_by_source is not None else None
                processings_by_source = output.metadata.get("plotted_processings")
                processing_indices = processings_by_source[number] if processings_by_source is not None else None
                report_summary = (f"{len({sample for ids in plotted_groups.values() for sample in ids})} spectra actually plotted "
                                  f"(max_samples={config.get('max_samples', 50)} per group); {summary}") if plotted_groups is not None else summary
                if processing_indices is not None:
                    report_summary += f" Processing indices shown: {processing_indices}."
                _write_alternative(directory, stem, snapshot, context, report_summary, image_path.name,
                                   include_excluded=include_excluded, source_index=source_index, color_column=color_column,
                                   plotted_groups=plotted_groups, include_augmented=include_augmented,
                                   processing_indices=processing_indices)
                output_paths.append(str(directory / f"{stem}.html"))
    if plots_visible:
        import matplotlib.pyplot as plt

        plt.show(block=False)
    return output_paths
