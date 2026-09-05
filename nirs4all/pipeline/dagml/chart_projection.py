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
from typing import Any
from uuid import uuid4

import numpy as np


def validate_chart_projection(pipeline: list[Any], spectro: Any) -> None:
    """Check that requested chart stages have an unambiguous captured prefix."""
    from .run_backend import _is_chart_step
    from .steps import _is_split_step

    uncertain_stage = False
    transformed = False
    for step in pipeline:
        if _is_chart_step(step):
            if uncertain_stage or (transformed and spectro.is_multi_source()):
                raise NotImplementedError("This chart stage needs a captured branch/source snapshot; a raw-data substitute would be misleading.")
        elif _is_split_step(step) or step is None:
            continue
        elif isinstance(step, dict):
            if set(step) == {"preprocessing"}:
                transformed = True
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


def _write_alternative(directory: Path, stem: str, snapshot: Any, context: Any, summary: str, image_name: str, *, include_excluded: bool) -> None:
    """Expose exact plotted inputs without requiring interpretation of colors."""
    sample_indices = snapshot._indexer.x_indices(context.selector, include_augmented=True, include_excluded=include_excluded)
    arrays = snapshot.x(context.selector, "3d", False, include_excluded=include_excluded)
    arrays = arrays if isinstance(arrays, list) else [arrays]
    targets = np.asarray(snapshot.y(context, include_excluded=include_excluded)).reshape(len(sample_indices), -1)
    data_name = f"{stem}.csv"
    with (directory / data_name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", "source", "processing", "feature_index", "value", *[f"target_{i}" for i in range(targets.shape[1])]])
        for source, array in enumerate(arrays):
            for sample, sample_data in enumerate(array):
                for processing, values in enumerate(sample_data):
                    for feature, value in enumerate(values):
                        writer.writerow([sample_indices[sample], source, processing, feature, float(value), *targets[sample].tolist()])
    (directory / f"{stem}.json").write_text(json.dumps({"summary": summary, "folds": snapshot.folds}, indent=2), encoding="utf-8")
    (directory / f"{stem}.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>DAG chart</title>'
        f'<main><h1>DAG run chart</h1><p>{html.escape(summary)}</p>'
        f'<p><a href="{html.escape(data_name)}">Download exact numeric inputs (CSV)</a></p>'
        f'<p><a href="{html.escape(stem)}.json">Read scored fold memberships and methodology (JSON)</a></p>'
        f'<img src="{html.escape(image_name)}" alt="{html.escape(summary)}"></main></html>',
        encoding="utf-8",
    )


def render_run_charts(result: Any, pipeline: list[Any], spectro: Any, *, workspace_path: Path | None,
                      save_charts: bool, plots_visible: bool, verbose: int) -> list[str]:
    """Reuse library chart presenters on immutable snapshots of fitted DAG state."""
    from nirs4all.controllers.registry import CONTROLLER_REGISTRY
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext, StepMetadata
    from nirs4all.pipeline.steps.parser import StepParser

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
    output_paths: list[str] = []
    for index, step in enumerate(pipeline):
        if not _is_chart_step(step):
            if _is_split_step(step):
                after_split = True
            elif isinstance(step, dict) and "y_processing" in step:
                processed_target = True
            elif (isinstance(step, dict) and set(step) == {"preprocessing"}) or (not isinstance(step, dict) and hasattr(step, "transform") and not hasattr(step, "predict")):
                prefix += 1
            continue
        snapshot = copy.deepcopy(spectro)
        snapshot.set_folds(_folds_from_scores(result) if after_split else [])
        if prefix:
            if prefix > len(fitted_steps):
                raise RuntimeError("Chart transform prefix is missing from the scored refit artifact.")
            values = np.asarray(snapshot.x({}, layout="2d"))
            for _, transformer in fitted_steps[:prefix]:
                values = transformer.transform(values)
            snapshot.add_merged_features(values, processing_name=f"refit_stage_{prefix}")
        parsed = StepParser().parse(step)
        context = ExecutionContext(metadata=StepMetadata(keyword=parsed.keyword, step_id=str(index)))
        context = context.with_partition(None)
        context = context.with_processing([snapshot.features_processings(source) for source in range(snapshot.features_sources())])
        if processed_target and artifact and artifact["y_transform"] is not None:
            target = np.asarray(snapshot.y()).reshape(snapshot.num_samples, -1)
            snapshot.add_processed_targets("chart_refit", artifact["y_transform"].transform(target))
            context = context.with_y("chart_refit")
        controller = next(cls for cls in CONTROLLER_REGISTRY if cls.__module__.startswith("nirs4all.controllers.charts.") and cls.matches(step, parsed.operator, parsed.keyword))
        _, output = controller().execute(parsed, snapshot, context, runtime)
        scope = "captured full-training REFIT transforms; not out-of-fold features" if prefix else "original observed features"
        summary = f"{parsed.keyword}: {snapshot.num_samples} samples; {scope}. {len(snapshot.folds)} scored cross-validation folds. Numeric inputs and fold memberships are supplied alongside the image."
        if directory is None:
            print(summary)
        else:
            config = step.get(parsed.keyword, {}) if isinstance(step, dict) else {}
            include_excluded = bool(config.get("include_excluded", False)) if isinstance(config, dict) else False
            for number, (data, _, extension) in enumerate(output.outputs):
                stem = f"step_{index:03d}_{number:02d}"
                image_path = directory / f"{stem}.{extension}"
                image_path.write_bytes(data)
                _write_alternative(directory, stem, snapshot, context, summary, image_path.name, include_excluded=include_excluded)
                output_paths.append(str(directory / f"{stem}.html"))
    if plots_visible:
        import matplotlib.pyplot as plt

        plt.show(block=False)
    return output_paths
