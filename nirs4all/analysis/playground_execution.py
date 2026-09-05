"""Stateless scientific Playground preview; no jobs, HTTP or training runner.

Each step delegates to owner operators. The response is an exploration result,
not a fitted predictive pipeline or an independent evaluation score.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from nirs4all.analysis.playground_metrics import compute_descriptors
from nirs4all.analysis.playground_partition import filter_batch, select_sample_indices, split_batch
from nirs4all.analysis.playground_prepare import (
    PreviewStep,
    establish_test_partition,
    partition_summary,
    prepare_batch,
    remap_folds,
)
from nirs4all.analysis.playground_projections import (
    display_indices,
    pca_projection,
    repetition_projection,
    spectral_payload,
    umap_projection,
)
from nirs4all.analysis.playground_steps import augment_batch, transform_batch
from nirs4all.analysis.playground_types import PreviewBatch, PreviewLimits, positive_count


def _project(function: Callable[[], Any]) -> Any:
    """Preserve per-chart error reporting without replacing scientific output."""
    try:
        return function()
    except Exception as error:
        return {"error": str(error)}


def _step(batch: PreviewBatch, step: PreviewStep, options: dict[str, Any], limits: PreviewLimits,
          split_count: int) -> tuple[PreviewBatch, dict[str, Any]]:
    params = dict(step.params)
    if step.type == "preprocessing":
        return transform_batch(batch, step.operator, limits=limits), {}
    if step.type == "augmentation":
        copies = positive_count(params.get("n_augmented_copies", 1), "n_augmented_copies", allow_zero=True)
        result, info = augment_batch(batch, step.operator, copies=copies, limits=limits)
        return result, {"augmentation": info, "copies": copies, "parents": np.tile(np.arange(len(batch.x)), copies + 1)}
    if step.type == "filter":
        mode = params.get("filter_mode", "remove")
        if step.name == "SampleIndexFilter":
            result, info = select_sample_indices(batch, params.get("indices", []), selection=params.get("mode", "keep"), mode=mode)
        else:
            result, info = filter_batch(batch, step.operator, mode=mode)
        return result, {"filter": info, "parents": np.arange(len(batch.x)) if mode == "tag" else np.flatnonzero(info["mask"])}
    if step.type == "splitting":
        has_test = batch.partitions is not None and bool(np.any(batch.partitions == "test"))
        kind = "cv_folds" if has_test or split_count else "test_split"
        allowed = {key: params[key] for key in ("group_by", "ignore_repetition", "aggregation", "y_aggregation") if key in params}
        info, warnings = split_batch(batch, step.operator, kind=kind, legacy_group=params.get("group"),
                                     repetition=options.get("dataset_repetition") or options.get("bio_sample_column"),
                                     split_index=options.get("split_index"), **allowed)
        result = establish_test_partition(batch, info) if kind == "test_split" else batch
        return result, {"folds": info, "warnings": warnings}
    raise ValueError(f"Unknown preview step type: {step.type}")


def execute_preview(batch: PreviewBatch, steps: list[PreviewStep] | None = None, *,
                    sampling: dict[str, Any] | None = None, options: dict[str, Any] | None = None,
                    limits: PreviewLimits | None = None,
                    metric_compute: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return true transformed arrays and inline analysis, preserving row lineage.

    ``metric_compute`` optionally replaces the owner descriptor callable (not a
    server callback). The default computes the historical descriptor set.
    Request/response byte limits remain the application's transport duty.
    ``use_cache`` is an optional optimization hint; this callable is stateless.
    """
    start = time.perf_counter()
    options = dict(options or {})
    steps = list(steps or [])
    budget = limits or PreviewLimits()
    if len(steps) > positive_count(options.get("max_steps", 50), "max_steps"):
        raise ValueError("Preview exceeds configured step count")
    original, subset, warnings = prepare_batch(batch, sampling, options)
    assert original.origins is not None  # prepare_batch establishes row lineage.
    budget.admit(*original.x.shape)
    current = original
    folds = None
    filter_info: dict[str, Any] | None = None
    augmentation: dict[str, Any] | None = None
    split_count = 0
    traces, errors = [], []
    for step in steps:
        if not step.enabled:
            continue
        step_start = time.perf_counter()
        try:
            next_batch, info = _step(current, step, options, budget, split_count)
            if "parents" in info:
                folds = remap_folds(folds, info["parents"], next_batch)
            current = next_batch
            assert current.origins is not None  # All owner steps retain lineage.
            if "folds" in info:
                folds = info["folds"]
                split_count += 1
            warnings.extend(info.get("warnings", []))
            if "augmentation" in info:
                item = info["augmentation"]
                if augmentation is None:
                    augmentation = {"original_count": item["original_count"], "steps": []}
                augmentation["steps"].append({"name": step.name, "copies": info["copies"], "samples_added": item["augmented_count"]})
                augmentation["total_count"] = item["total_count"]
            if "filter" in info:
                item = info["filter"]
                if filter_info is None:
                    filter_info = {"filters_applied": [], "total_removed": 0, "tagged_samples": {},
                                   "tag_mask": [False] * len(original.x), "mask_scope": "original_rows"}
                mode = item["filter_mode"]
                if mode == "tag":
                    tagged = np.flatnonzero(np.isin(original.origins, item["sample_origins"])).tolist()
                    filter_info["tagged_samples"][step.name] = tagged
                    for index in tagged:
                        filter_info["tag_mask"][index] = True
                else:
                    filter_info["total_removed"] += item["removed"]
                filter_info["filters_applied"].append({"name": step.name, "removed_count": item["removed"],
                                                        "reason": item["reason"], "mode": mode})
                filter_info["final_mask"] = np.isin(original.origins, current.origins).tolist()
            traces.append({"step_id": step.id, "name": step.name, "duration_ms": (time.perf_counter() - step_start) * 1000,
                           "success": True, "error": None, "output_shape": None if step.type == "splitting" else list(current.x.shape)})
        except Exception as error:
            traces.append({"step_id": step.id, "name": step.name, "duration_ms": (time.perf_counter() - step_start) * 1000,
                           "success": False, "error": str(error), "output_shape": None})
            errors.append({"step": step.id, "name": step.name, "error": str(error)})
    response_cells = original.x.size + current.x.size + len(current.x) * 20
    if response_cells > positive_count(options.get("max_response_cells", 200_000_000), "max_response_cells"):
        raise ValueError("Preview presentation exceeds host cardinality budget before projection construction")
    pca = _project(lambda: pca_projection(current, folds)) if options.get("compute_pca", True) else None
    umap = _project(lambda: umap_projection(current, folds, **options.get("umap_params", {}))) if options.get("compute_umap", False) else None
    repetitions = _project(lambda: repetition_projection(current, pca=pca, umap=umap, options=options)) if options.get("compute_repetitions", True) else None
    metrics = None
    if options.get("compute_metrics", False):
        metric_owner = metric_compute or compute_descriptors
        metrics = _project(lambda: metric_owner(current, pca=pca, options=options))
    maximum = options.get("max_wavelengths_returned")
    # Identical axes share the processed-mean LTTB selection, as in 0.9.1.
    common = display_indices(current.wavelengths, current.x, maximum) if len(current.x) and np.array_equal(original.wavelengths, current.wavelengths) else None
    compute_stats = options.get("compute_statistics", True)
    original_payload = _project(lambda: spectral_payload(original, compute_statistics=compute_stats, max_wavelengths=maximum, indices=common))
    processed_payload = _project(lambda: spectral_payload(current, compute_statistics=compute_stats, max_wavelengths=maximum, indices=common))
    return {"success": not errors, "execution_time_ms": (time.perf_counter() - start) * 1000,
            "original": original_payload, "processed": processed_payload,
            "pca": pca, "umap": umap, "folds": folds, "filter_info": filter_info, "augmentation_info": augmentation,
            "repetitions": repetitions, "metrics": metrics, "subset_info": subset, "execution_trace": traces,
            "step_errors": errors, "warnings": warnings, "is_raw_data": not any(step.enabled for step in steps),
            "source_partitions": partition_summary(original), "processed_partitions": partition_summary(current),
            "evaluation_scope": "exploratory_preview", "cache": {"used": False, "scope": "stateless_callable"}}
