"""Read-only Inspector result views owned by nirs4all, independent of transports.

The view algorithms were moved from Studio so the native product can request
scientific diagnostics through its bounded library host without an HTTP backend.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Hashable
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .inspector_json import sanitize_dict, sanitize_float
from .inspector_normalization import (
    _build_available_targets,
    _extract_model_params_from_expanded_config,
    _flatten_numeric_params,
    _is_classification_task,
    _load_pipeline_metadata_map,
    _matches_task_type_filter,
    _merge_variant_params,
    _normalize_chain_record,
    _normalize_chain_records,
    _parse_json_like,
    _split_preprocessing_steps,
)

_workspace: ContextVar[str] = ContextVar("inspector_workspace")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InspectorViewError(ValueError):
    def __init__(self, *, status_code: int, detail: str) -> None:
        self.status_code = status_code
        super().__init__(f"not_found: {detail}" if status_code == 404 else detail)


def _get_store() -> Any:
    from nirs4all.pipeline.storage import WorkspaceStore

    path = Path(_workspace.get())
    if not (path / "store.sqlite").is_file():
        raise InspectorViewError(status_code=404, detail="No results store found. Run a pipeline first.")
    return WorkspaceStore.open_readonly(path)


_SCORE_REF_DESCRIPTORS: dict[str, tuple[str, str, str]] = {
    "cv_val_score": ("cross_validation", "validation", "fold_mean"),
    "cv_test_score": ("cross_validation", "test", "fold_mean"),
    "cv_train_score": ("cross_validation", "train", "fold_mean"),
    "final_test_score": ("final", "test", "final_model"),
    "final_train_score": ("final", "train", "final_model"),
}


class ScoreRef(_StrictModel):
    """Structured score identifier accepted alongside legacy score_column."""

    model_config = ConfigDict(extra="allow")
    key: str | None = None
    metric: str | None = None
    protocol: str | None = None
    partition: str | None = None
    aggregation: str | None = None
    legacyScoreColumn: str | None = None
    legacy_score_column: str | None = None


_SCORE_REF_PROTOCOL_ALIASES = {"cv": "cross_validation", "cross-validation": "cross_validation", "cross_validation": "cross_validation", "final": "final"}
_SCORE_REF_PARTITION_ALIASES = {"val": "validation", "valid": "validation", "validation": "validation", "test": "test", "train": "train"}
_SCORE_REF_AGGREGATION_ALIASES = {"mean": "fold_mean", "avg": "fold_mean", "fold_mean": "fold_mean", "final": "final_model", "final_model": "final_model"}


class InspectorDataResponse(_StrictModel):
    """Response for /inspector/data."""

    chains: list[dict[str, Any]]
    total: int
    available_metrics: list[str]
    available_models: list[str]
    available_datasets: list[str]
    available_runs: list[str]
    available_preprocessings: list[str]
    available_targets: list[dict[str, Any]] = Field(default_factory=list)
    generated_at: str


class ScatterRequest(_StrictModel):
    """Request body for /inspector/scatter."""

    chain_ids: list[str]
    partition: str = "val"
    target_index: int = Field(default=0, ge=0)


class ScatterPoint(_StrictModel):
    """A single chain's scatter data."""

    chain_id: str
    model_class: str
    model_name: str | None = None
    preprocessings: str | None = None
    y_true: list[float]
    y_pred: list[float]
    sample_indices: list[int] | None = None
    fold_id: str | None = None
    score: float | None = None


class ScatterResponse(_StrictModel):
    """Response for /inspector/scatter."""

    points: list[ScatterPoint]
    partition: str
    total_samples: int


class HistogramBin(_StrictModel):
    """A single histogram bin."""

    bin_start: float
    bin_end: float
    count: int
    chain_ids: list[str]


class HistogramResponse(_StrictModel):
    """Response for /inspector/histogram."""

    bins: list[HistogramBin]
    score_column: str
    total_chains: int
    min_score: float | None = None
    max_score: float | None = None
    mean_score: float | None = None


class RankingsResponse(_StrictModel):
    """Response for /inspector/rankings."""

    rankings: list[dict[str, Any]]
    total: int
    score_column: str
    sort_ascending: bool


class HeatmapRequest(_StrictModel):
    """Request body for /inspector/heatmap."""

    run_id: list[str] | None = None
    dataset_name: list[str] | None = None
    x_variable: str = "model_class"
    y_variable: str = "preprocessings"
    score_column: str = "cv_val_score"
    aggregate: str = "best"


class HeatmapCell(_StrictModel):
    """A single heatmap cell."""

    x_label: str
    y_label: str
    value: float | None = None
    count: int
    chain_ids: list[str]


class HeatmapResponse(_StrictModel):
    """Response for /inspector/heatmap."""

    cells: list[dict[str, Any]]
    x_labels: list[str]
    y_labels: list[str]
    x_variable: str
    y_variable: str
    score_column: str
    min_value: float | None = None
    max_value: float | None = None


class CandlestickRequest(_StrictModel):
    """Request body for /inspector/candlestick."""

    run_id: list[str] | None = None
    dataset_name: list[str] | None = None
    category_variable: str = "model_class"
    score_column: str = "cv_val_score"


class CandlestickCategory(_StrictModel):
    """Box-plot statistics for a single category."""

    label: str
    min: float
    q25: float
    median: float
    q75: float
    max: float
    mean: float
    count: int
    outlier_values: list[float]
    chain_ids: list[str]


class CandlestickResponse(_StrictModel):
    """Response for /inspector/candlestick."""

    categories: list[dict[str, Any]]
    category_variable: str
    score_column: str


def _get_arrays(store: Any, prediction_id: str) -> dict[str, Any] | None:
    """Get prediction arrays with fallback for stores without get_prediction_arrays."""
    get_arrays = getattr(store, "get_prediction_arrays", None)
    if callable(get_arrays):
        return cast(dict[str, Any] | None, get_arrays(prediction_id))
    prediction = store.get_prediction(prediction_id, load_arrays=True)
    return cast(dict[str, Any] | None, prediction)


def _normalize_score_ref_part(value: str | None, aliases: dict[str, str]) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    return aliases.get(normalized, normalized)


def _score_ref_descriptor(score_ref: ScoreRef | None) -> tuple[str, str, str] | None:
    if score_ref is None:
        return None
    protocol = _normalize_score_ref_part(score_ref.protocol, _SCORE_REF_PROTOCOL_ALIASES)
    partition = _normalize_score_ref_part(score_ref.partition, _SCORE_REF_PARTITION_ALIASES)
    aggregation = _normalize_score_ref_part(score_ref.aggregation, _SCORE_REF_AGGREGATION_ALIASES)
    if protocol is None or partition is None or aggregation is None:
        return None
    return (protocol, partition, aggregation)


def _score_ref_legacy_column(score_ref: ScoreRef | None) -> str | None:
    if score_ref is None:
        return None
    legacy = score_ref.legacyScoreColumn or score_ref.legacy_score_column
    if legacy in _SCORE_REF_DESCRIPTORS:
        return legacy
    return None


def _resolve_score_column_from_score_ref(score_column: str, score_ref: ScoreRef | None) -> str:
    """Resolve a structured score_ref to a legacy score column when possible."""
    if score_ref is None:
        return score_column
    descriptor = _score_ref_descriptor(score_ref)
    legacy_column = _score_ref_legacy_column(score_ref)
    if legacy_column is not None and (descriptor is None or _SCORE_REF_DESCRIPTORS[legacy_column] == descriptor):
        return legacy_column
    if descriptor is not None:
        for candidate, candidate_descriptor in _SCORE_REF_DESCRIPTORS.items():
            if candidate_descriptor == descriptor:
                return candidate
    return score_column


def _parse_score_ref_query_param(score_ref: str | None) -> ScoreRef | None:
    """Parse a JSON-encoded score_ref query parameter."""
    if score_ref is None or not score_ref.strip():
        return None
    try:
        return ScoreRef.model_validate_json(score_ref)
    except ValidationError as exc:
        raise InspectorViewError(status_code=422, detail="Invalid score_ref query parameter") from exc


def _has_resolved_score_ref(score_column: str, score_ref: ScoreRef | None) -> bool:
    if score_ref is None:
        return False
    descriptor = _score_ref_descriptor(score_ref)
    legacy_column = _score_ref_legacy_column(score_ref)
    if legacy_column is not None and (descriptor is None or _SCORE_REF_DESCRIPTORS[legacy_column] == descriptor):
        return True
    if descriptor is not None:
        return descriptor in _SCORE_REF_DESCRIPTORS.values()
    return False


def _prediction_partition_from_score_column(score_column: str, fallback: str = "val") -> str:
    return {"cv_val_score": "val", "cv_test_score": "test", "cv_train_score": "train", "final_test_score": "test", "final_train_score": "train"}.get(score_column, fallback)


def _prediction_score_field_from_partition(partition: str) -> str:
    return {"val": "val_score", "test": "test_score", "train": "train_score"}.get(partition, "val_score")


def _prediction_score_field_from_score_column(score_column: str, fallback_partition: str = "val") -> str:
    partition = _prediction_partition_from_score_column(score_column, fallback_partition)
    return _prediction_score_field_from_partition(partition)


def _coerce_vector(values: Any) -> list[Any] | None:
    """Coerce a scalar, list, or ndarray into a flat Python list."""
    if values is None:
        return None
    try:
        import numpy as np

        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            return [array.item()]
        return array.reshape(-1).tolist()
    except Exception:
        if isinstance(values, (list, tuple)):
            return list(values)
        return None


def _coerce_numeric_vector(values: Any) -> list[float] | None:
    """Coerce values into a flat list of floats."""
    vector = _coerce_vector(values)
    if vector is None:
        return None
    numeric: list[float] = []
    for value in vector:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            return None
    return numeric


def _coerce_target_vector(values: Any, target_index: int = 0) -> list[Any] | None:
    """Coerce values into a flat list after selecting a target column."""
    if values is None or target_index < 0:
        return None
    try:
        import numpy as np

        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            return [array.item()]
        if array.ndim > 1:
            if target_index >= array.shape[1]:
                return None
            array = array[:, target_index]
        return array.reshape(-1).tolist()
    except Exception:
        if isinstance(values, (list, tuple)):
            if not values:
                return []
            if any(isinstance(item, (list, tuple)) for item in values):
                selected: list[Any] = []
                for row in values:
                    if not isinstance(row, (list, tuple)) or target_index >= len(row):
                        return None
                    selected.append(row[target_index])
                return selected
            return list(values)
        return None


def _coerce_numeric_target_vector(values: Any, target_index: int = 0) -> list[float] | None:
    """Coerce a selected target column into a flat list of floats."""
    vector = _coerce_target_vector(values, target_index=target_index)
    if vector is None:
        return None
    numeric: list[float] = []
    for value in vector:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            return None
    return numeric


def _coerce_index_vector(values: Any) -> list[int] | None:
    """Coerce values into a flat list of integer sample indices."""
    vector = _coerce_vector(values)
    if vector is None:
        return None
    indices: list[int] = []
    for value in vector:
        try:
            indices.append(int(value))
        except (TypeError, ValueError):
            return None
    return indices


def _is_lower_better(metric: str | None) -> bool:
    """Return True when a lower score is better for *metric*.

    Single source of truth: ``nirs4all.pipeline.run.get_metric_info`` â€” the
    same metric-direction table the store uses to rank chains. Unknown
    metrics inherit the library default (higher-is-better).
    """
    from nirs4all.pipeline.run import get_metric_info

    return not bool(get_metric_info(metric).get("higher_is_better", True))


def get_inspector_data(
    run_id: list[str] | None = None, dataset_name: list[str] | None = None, model_class: list[str] | None = None, preprocessings: list[str] | None = None, task_type: str | None = None, metric: str | None = None
):
    """Load chain summaries and metadata for the Inspector.

    Returns all matching chains plus lists of unique values for
    populating filter bar dropdowns (metrics, models, datasets, runs, preprocessings).
    Supports multi-value filters via repeated query params.
    """
    store = _get_store()
    try:
        _run_id = run_id if run_id and len(run_id) > 1 else run_id[0] if run_id else None
        _dataset_name = dataset_name if dataset_name and len(dataset_name) > 1 else dataset_name[0] if dataset_name else None
        _model_class = model_class if model_class and len(model_class) > 1 else model_class[0] if model_class else None
        facet_records = _normalize_chain_records(store, store.query_chain_summaries())
        facet_by_id = {r.get("chain_id"): r for r in facet_records}
        filtered_df = store.query_chain_summaries(run_id=_run_id, dataset_name=_dataset_name, model_class=_model_class, metric=metric)
        filtered_ids = [str(row.get("chain_id") or "") for row in filtered_df.iter_rows(named=True)]
        records = [facet_by_id[cid] for cid in filtered_ids if cid in facet_by_id]
        if task_type:
            records = [record for record in records if _matches_task_type_filter(record.get("task_type"), task_type)]
        if preprocessings:
            filtered = []
            for r in records:
                preps = r.get("preprocessings") or ""
                steps = {str(step) for step in r.get("preprocessing_steps") or []}
                if any(p in preps or p in steps for p in preprocessings):
                    filtered.append(r)
            records = filtered
        metrics_set = sorted({cast(str, r["metric"]) for r in facet_records if r.get("metric")})
        models = sorted({cast(str, r["model_class"]) for r in facet_records if r.get("model_class")})
        datasets = sorted({cast(str, r["dataset_name"]) for r in facet_records if r.get("dataset_name")})
        runs = sorted({cast(str, r["run_id"]) for r in facet_records if r.get("run_id")})
        prep_steps: set[str] = set()
        for r in facet_records:
            for step in r.get("preprocessing_steps") or []:
                prep_steps.add(str(step))
        return InspectorDataResponse(
            chains=records,
            total=len(records),
            available_metrics=metrics_set,
            available_models=models,
            available_datasets=datasets,
            available_runs=runs,
            available_preprocessings=sorted(prep_steps),
            available_targets=_build_available_targets(records),
            generated_at=datetime.now(UTC).isoformat(),
        )
    finally:
        store.close()


def get_scatter_data(request: ScatterRequest):
    """Get y_true/y_pred arrays for scatter visualization.

    For each chain_id, loads the fold-level predictions matching the
    requested partition and concatenates the arrays.
    """
    if not request.chain_ids:
        return ScatterResponse(points=[], partition=request.partition, total_samples=0)
    store = _get_store()
    try:
        points: list[ScatterPoint] = []
        total_samples = 0
        score_field = {"val": "val_score", "test": "test_score", "train": "train_score"}.get(request.partition, "val_score")
        for chain_id in request.chain_ids:
            pred_df = store.get_chain_predictions(chain_id=chain_id, partition=request.partition)
            if len(pred_df) == 0:
                continue
            first_row = {**dict(pred_df.row(0, named=True)), **(store.get_chain(chain_id) or {})}
            all_y_true: list[float] = []
            all_y_pred: list[float] = []
            all_indices: list[int] = []
            score = sanitize_float(first_row.get(score_field))
            for row in pred_df.iter_rows(named=True):
                row_dict = dict(row)
                prediction_id = row_dict.get("prediction_id")
                if not prediction_id:
                    continue
                arrays = _get_arrays(store, prediction_id)
                if arrays is None:
                    continue
                y_true_values = _coerce_numeric_target_vector(arrays.get("y_true"), request.target_index)
                y_pred_values = _coerce_numeric_target_vector(arrays.get("y_pred"), request.target_index)
                if not y_true_values or not y_pred_values:
                    continue
                sample_indices = _coerce_index_vector(arrays.get("sample_indices"))
                pair_count = min(len(y_true_values), len(y_pred_values))
                for index in range(pair_count):
                    yt = y_true_values[index]
                    yp = y_pred_values[index]
                    if math.isnan(yt) or math.isnan(yp) or math.isinf(yt) or math.isinf(yp):
                        continue
                    all_y_true.append(yt)
                    all_y_pred.append(yp)
                    if sample_indices is not None and index < len(sample_indices):
                        all_indices.append(sample_indices[index])
            if all_y_true and all_y_pred:
                total_samples += len(all_y_true)
                points.append(
                    ScatterPoint(
                        chain_id=chain_id,
                        model_class=first_row.get("model_class", ""),
                        model_name=first_row.get("model_name"),
                        preprocessings=first_row.get("preprocessings"),
                        y_true=all_y_true,
                        y_pred=all_y_pred,
                        sample_indices=all_indices if all_indices else None,
                        fold_id=None,
                        score=score,
                    )
                )
        return ScatterResponse(points=points, partition=request.partition, total_samples=total_samples)
    finally:
        store.close()


def get_histogram_data(run_id: list[str] | None = None, dataset_name: list[str] | None = None, score_column: str = "cv_val_score", n_bins: int = 20):
    """Score distribution histogram.

    Computes histogram bins from chain summary scores and includes
    chain_ids per bin for click-to-select functionality.
    """
    import numpy as np

    store = _get_store()
    try:
        df = store.query_chain_summaries(run_id=run_id or None, dataset_name=dataset_name or None)
        records = _normalize_chain_records(store, df)
        scores: list[float] = []
        chain_ids_for_scores: list[str] = []
        for r in records:
            val = r.get(score_column)
            if val is not None and isinstance(val, (int, float)):
                scores.append(float(val))
                chain_ids_for_scores.append(r["chain_id"])
        if not scores:
            return HistogramResponse(bins=[], score_column=score_column, total_chains=0)
        scores_arr = np.array(scores)
        counts, bin_edges = np.histogram(scores_arr, bins=n_bins)
        bins: list[HistogramBin] = []
        for i in range(len(counts)):
            bin_start = float(bin_edges[i])
            bin_end = float(bin_edges[i + 1])
            bin_chain_ids = [cid for cid, s in zip(chain_ids_for_scores, scores, strict=False) if bin_start <= s < bin_end or (i == len(counts) - 1 and s == bin_end)]
            bins.append(HistogramBin(bin_start=round(bin_start, 6), bin_end=round(bin_end, 6), count=int(counts[i]), chain_ids=bin_chain_ids))
        return HistogramResponse(
            bins=bins, score_column=score_column, total_chains=len(scores), min_score=round(float(scores_arr.min()), 6), max_score=round(float(scores_arr.max()), 6), mean_score=round(float(scores_arr.mean()), 6)
        )
    finally:
        store.close()


def get_rankings_data(run_id: list[str] | None = None, dataset_name: list[str] | None = None, score_column: str = "cv_val_score", sort_ascending: bool | None = None, limit: int = 50, offset: int = 0):
    """Ranked chain summaries.

    Returns chains sorted by the chosen score column, with rank numbers.
    Sort direction is auto-detected from the metric (lower-better for RMSE, etc.).
    """
    store = _get_store()
    try:
        if sort_ascending is None:
            peek = store.query_top_chains(n=1, score_column=score_column, ascending=True, run_id=run_id or None, dataset_name=dataset_name or None)
            first_metric = None
            if len(peek) > 0:
                first_metric = peek.row(0, named=True).get("metric")
            sort_ascending = _is_lower_better(first_metric)
        df = store.query_top_chains(n=limit, offset=offset, score_column=score_column, ascending=sort_ascending, run_id=run_id or None, dataset_name=dataset_name or None)
        records = _normalize_chain_records(store, df)
        total = store.count_chain_summaries(run_id=run_id or None, dataset_name=dataset_name or None)
        rankings: list[dict] = []
        for i, r in enumerate(records):
            rankings.append(
                {
                    "rank": offset + i + 1,
                    "chain_id": r.get("chain_id", ""),
                    "model_class": r.get("model_class", ""),
                    "model_name": r.get("model_name"),
                    "preprocessings": r.get("preprocessings"),
                    "cv_val_score": r.get("cv_val_score"),
                    "cv_test_score": r.get("cv_test_score"),
                    "cv_train_score": r.get("cv_train_score"),
                    "final_test_score": r.get("final_test_score"),
                    "final_train_score": r.get("final_train_score"),
                    "cv_fold_count": r.get("cv_fold_count", 0),
                    "dataset_name": r.get("dataset_name"),
                    "best_params": r.get("best_params"),
                }
            )
        return RankingsResponse(rankings=rankings, total=total, score_column=score_column, sort_ascending=sort_ascending)
    finally:
        store.close()


def get_heatmap_data(request: HeatmapRequest):
    """Performance heatmap: aggregated score at intersection of two variables.

    Groups chains by (x_variable, y_variable) and aggregates score_column
    using the requested aggregation method (best/mean/median/worst).
    """
    import numpy as np

    store = _get_store()
    try:
        df = store.query_chain_summaries(run_id=request.run_id, dataset_name=request.dataset_name)
        records = _normalize_chain_records(store, df)
        grid: dict[tuple[str, str], list[dict]] = {}
        for r in records:
            x_val = str(r.get(request.x_variable) or "(empty)")
            y_val = str(r.get(request.y_variable) or "(empty)")
            grid.setdefault((x_val, y_val), []).append(r)
        cells: list[dict] = []
        all_values: list[float] = []
        x_labels_set: set[str] = set()
        y_labels_set: set[str] = set()
        for (x_label, y_label), chains_in_cell in grid.items():
            x_labels_set.add(x_label)
            y_labels_set.add(y_label)
            scores = [float(c[request.score_column]) for c in chains_in_cell if c.get(request.score_column) is not None]
            chain_ids = [c["chain_id"] for c in chains_in_cell]
            if scores:
                first_metric = next((c.get("metric") for c in chains_in_cell if c.get("metric")), None)
                lower_better = _is_lower_better(first_metric)
                if request.aggregate == "best":
                    value = min(scores) if lower_better else max(scores)
                elif request.aggregate == "worst":
                    value = max(scores) if lower_better else min(scores)
                elif request.aggregate == "median":
                    value = float(np.median(scores))
                else:
                    value = float(np.mean(scores))
                all_values.append(value)
            else:
                value = None
            cells.append(HeatmapCell(x_label=x_label, y_label=y_label, value=round(value, 6) if value is not None else None, count=len(chains_in_cell), chain_ids=chain_ids).model_dump())
        return HeatmapResponse(
            cells=cells,
            x_labels=sorted(x_labels_set),
            y_labels=sorted(y_labels_set),
            x_variable=request.x_variable,
            y_variable=request.y_variable,
            score_column=request.score_column,
            min_value=round(min(all_values), 6) if all_values else None,
            max_value=round(max(all_values), 6) if all_values else None,
        )
    finally:
        store.close()


def get_candlestick_data(request: CandlestickRequest):
    """Box-plot statistics per category.

    Groups chains by category_variable and computes min, Q25, median,
    Q75, max, mean, and IQR-based outliers for the chosen score column.
    """
    import numpy as np

    store = _get_store()
    try:
        df = store.query_chain_summaries(run_id=request.run_id, dataset_name=request.dataset_name)
        records = _normalize_chain_records(store, df)
        buckets: dict[str, list[dict]] = {}
        for r in records:
            label = str(r.get(request.category_variable) or "(empty)")
            buckets.setdefault(label, []).append(r)
        categories: list[dict] = []
        for label, chains_in_cat in buckets.items():
            scores = [float(c[request.score_column]) for c in chains_in_cat if c.get(request.score_column) is not None]
            chain_ids = [c["chain_id"] for c in chains_in_cat]
            if not scores:
                continue
            arr = np.array(scores)
            q25 = float(np.percentile(arr, 25))
            q75 = float(np.percentile(arr, 75))
            iqr = q75 - q25
            lower_fence = q25 - 1.5 * iqr
            upper_fence = q75 + 1.5 * iqr
            outlier_values = [float(s) for s in scores if s < lower_fence or s > upper_fence]
            categories.append(
                CandlestickCategory(
                    label=label,
                    min=round(float(arr.min()), 6),
                    q25=round(q25, 6),
                    median=round(float(np.median(arr)), 6),
                    q75=round(q75, 6),
                    max=round(float(arr.max()), 6),
                    mean=round(float(arr.mean()), 6),
                    count=len(scores),
                    outlier_values=[round(v, 6) for v in outlier_values],
                    chain_ids=chain_ids,
                ).model_dump()
            )
        categories.sort(key=lambda c: c.get("median", 0), reverse=True)
        return CandlestickResponse(categories=categories, category_variable=request.category_variable, score_column=request.score_column)
    finally:
        store.close()


class BranchComparisonRequest(_StrictModel):
    """Request body for /inspector/branch-comparison."""

    run_id: list[str] | None = None
    dataset_name: list[str] | None = None
    score_column: str = "cv_val_score"


class BranchComparisonEntry(_StrictModel):
    """A single branch comparison entry."""

    branch_path: str
    label: str
    mean: float
    std: float
    min: float
    max: float
    ci_lower: float
    ci_upper: float
    count: int
    chain_ids: list[str]


class BranchComparisonResponse(_StrictModel):
    """Response for /inspector/branch-comparison."""

    branches: list[dict[str, Any]]
    score_column: str
    total_chains: int


class TopologyNode(_StrictModel):
    """A single node in the pipeline topology DAG."""

    id: str
    label: str
    type: str
    depth: int
    branch_path: list[int]
    metrics: dict[str, Any] | None = None
    children: list[dict[str, Any]] | None = None
    chain_ids: list[str] | None = None


class BranchTopologyResponse(_StrictModel):
    """Response for /inspector/branch-topology."""

    nodes: list[dict[str, Any]]
    pipeline_id: str
    pipeline_name: str
    has_stacking: bool
    has_branches: bool
    max_depth: int


class FoldScoreEntry(_StrictModel):
    """A single fold score entry."""

    chain_id: str
    model_class: str
    preprocessings: str | None = None
    fold_id: str
    fold_index: int
    score: float


class FoldStabilityRequest(_StrictModel):
    """Request body for /inspector/fold-stability."""

    chain_ids: list[str]
    score_column: str = "cv_val_score"
    score_ref: ScoreRef | None = None
    partition: str = "val"


class FoldStabilityResponse(_StrictModel):
    """Response for /inspector/fold-stability."""

    entries: list[dict[str, Any]]
    fold_ids: list[str]
    score_column: str
    total_chains: int


def _stringify_branch_path(branch_path: Any) -> str:
    """Convert a branch_path to a human-readable string for grouping."""
    if branch_path is None:
        return "(no branch)"
    if isinstance(branch_path, (list, tuple)):
        if len(branch_path) == 0:
            return "(no branch)"
        return " > ".join(str(p) for p in branch_path)
    return str(branch_path)


def get_branch_comparison(request: BranchComparisonRequest):
    """Branch comparison: mean score with CI per branch.

    Groups chains by branch_path and computes descriptive statistics
    including 95% confidence intervals for each branch.
    """
    import numpy as np

    store = _get_store()
    try:
        df = store.query_chain_summaries(run_id=request.run_id, dataset_name=request.dataset_name)
        records = _normalize_chain_records(store, df)
        buckets: dict[str, list[dict]] = {}
        for r in records:
            key = _stringify_branch_path(r.get("branch_path"))
            buckets.setdefault(key, []).append(r)
        branches: list[dict] = []
        total_chains = 0
        for branch_label, chains_in_branch in buckets.items():
            scores = [float(c[request.score_column]) for c in chains_in_branch if c.get(request.score_column) is not None]
            chain_ids = [c["chain_id"] for c in chains_in_branch]
            if not scores:
                continue
            arr = np.array(scores)
            count = len(scores)
            mean_val = float(arr.mean())
            std_val = float(arr.std(ddof=1)) if count > 1 else 0.0
            ci_half = 1.96 * std_val / math.sqrt(count) if count > 1 else 0.0
            total_chains += count
            branches.append(
                BranchComparisonEntry(
                    branch_path=branch_label,
                    label=branch_label,
                    mean=round(mean_val, 6),
                    std=round(std_val, 6),
                    min=round(float(arr.min()), 6),
                    max=round(float(arr.max()), 6),
                    ci_lower=round(mean_val - ci_half, 6),
                    ci_upper=round(mean_val + ci_half, 6),
                    count=count,
                    chain_ids=chain_ids,
                ).model_dump()
            )
        first_metric = next((r.get("metric") for r in records if r.get("metric")), None)
        lower_better = _is_lower_better(first_metric)
        branches.sort(key=lambda b: b.get("mean", 0), reverse=not lower_better)
        return BranchComparisonResponse(branches=branches, score_column=request.score_column, total_chains=total_chains)
    finally:
        store.close()


def get_branch_topology(pipeline_id: str, score_column: str = "cv_val_score", score_ref: str | None = None):
    """Pipeline topology: DAG structure with metrics overlay.

    Uses nirs4all.pipeline.analysis.topology.analyze_topology() to parse
    the expanded pipeline config into a tree of nodes.
    """
    import numpy as np

    parsed_score_ref = _parse_score_ref_query_param(score_ref)
    effective_score_column = _resolve_score_column_from_score_ref(score_column, parsed_score_ref)
    store = _get_store()
    try:
        pipeline = store.get_pipeline(pipeline_id)
        if not pipeline:
            raise InspectorViewError(status_code=404, detail=f"Pipeline {pipeline_id} not found")
        expanded_config = pipeline.get("expanded_config")
        pipeline_name = pipeline.get("name") or pipeline_id
        has_branches = False
        has_stacking = False
        max_depth = 0
        nodes: list[dict] = []
        if expanded_config and isinstance(expanded_config, list):
            try:
                from nirs4all.pipeline.analysis.topology import analyze_topology

                topology = analyze_topology(expanded_config)
                has_branches = len(topology.model_nodes) > 1
                has_stacking = topology.has_stacking
                max_depth = topology.max_stacking_depth
                df = store.query_chain_summaries(pipeline_id=pipeline_id)
                chain_records = _normalize_chain_records(store, df)
                for i, mn in enumerate(topology.model_nodes):
                    bp = list(mn.branch_path) if mn.branch_path else []
                    bp_str = _stringify_branch_path(bp)
                    matching_chains = [c for c in chain_records if _stringify_branch_path(c.get("branch_path")) == bp_str]
                    matching_scores = [float(c[effective_score_column]) for c in matching_chains if c.get(effective_score_column) is not None]
                    mean_score = round(float(np.mean(matching_scores)), 6) if matching_scores else None
                    nodes.append(
                        TopologyNode(
                            id=f"model_{i}",
                            label=mn.model_class,
                            type="model",
                            depth=mn.branch_depth,
                            branch_path=bp,
                            metrics={"mean_score": mean_score, "chain_count": len(matching_chains)},
                            chain_ids=[c["chain_id"] for c in matching_chains],
                        ).model_dump()
                    )
            except ImportError:
                pass
        if not nodes:
            df = store.query_chain_summaries(pipeline_id=pipeline_id)
            chain_records = _normalize_chain_records(store, df)
            model_groups: dict[str, list[dict]] = {}
            for r in chain_records:
                mc = r.get("model_class", "unknown")
                model_groups.setdefault(mc, []).append(r)
            for i, (mc, chains_for_model) in enumerate(model_groups.items()):
                scores = [float(c[effective_score_column]) for c in chains_for_model if c.get(effective_score_column) is not None]
                nodes.append(
                    TopologyNode(
                        id=f"model_{i}",
                        label=mc,
                        type="model",
                        depth=0,
                        branch_path=[],
                        metrics={"mean_score": round(float(np.mean(scores)), 6) if scores else None, "chain_count": len(chains_for_model)},
                        chain_ids=[c["chain_id"] for c in chains_for_model],
                    ).model_dump()
                )
        return BranchTopologyResponse(nodes=nodes, pipeline_id=pipeline_id, pipeline_name=pipeline_name, has_stacking=has_stacking, has_branches=has_branches, max_depth=max_depth)
    finally:
        store.close()


def _is_fold_prediction(row: dict[str, Any], score_column: str) -> bool:
    """Keep aggregate rows out of fold diagnostics and honor CV/final protocols."""
    fold = str(row.get("fold_id") or "")
    final = row.get("refit_context") is not None or fold == "final"
    if score_column.startswith("final_"):
        return final and not fold.endswith("_agg")
    return not final and fold not in {"avg", "w_avg"} and not fold.endswith("_agg")


def get_fold_stability(request: FoldStabilityRequest):
    """Per-fold score stability for selected chains.

    For each chain, retrieves fold-level predictions and extracts
    the score for the requested partition.
    """
    effective_score_column = _resolve_score_column_from_score_ref(request.score_column, request.score_ref)
    resolved_score_ref = _has_resolved_score_ref(request.score_column, request.score_ref)
    effective_partition = _prediction_partition_from_score_column(effective_score_column, request.partition) if resolved_score_ref else request.partition
    if not request.chain_ids:
        return FoldStabilityResponse(entries=[], fold_ids=[], score_column=effective_score_column, total_chains=0)
    store = _get_store()
    try:
        entries: list[dict] = []
        all_fold_ids: set[str] = set()
        score_field = _prediction_score_field_from_score_column(effective_score_column, request.partition) if resolved_score_ref else _prediction_score_field_from_partition(request.partition)
        for chain_id in request.chain_ids:
            pred_df = store.get_chain_predictions(chain_id=chain_id, partition=effective_partition)
            if len(pred_df) == 0:
                continue
            first_row = {**dict(pred_df.row(0, named=True)), **(store.get_chain(chain_id) or {})}
            model_class = first_row.get("model_class", "")
            preprocessings = first_row.get("preprocessings")
            for fold_idx, row in enumerate(pred_df.iter_rows(named=True)):
                row_dict = dict(row)
                if not _is_fold_prediction(row_dict, effective_score_column):
                    continue
                fold_id = str(row_dict.get("fold_id", fold_idx))
                score = row_dict.get(score_field)
                if score is None:
                    continue
                score = sanitize_float(float(score))
                if score is None:
                    continue
                all_fold_ids.add(fold_id)
                entries.append(FoldScoreEntry(chain_id=chain_id, model_class=model_class, preprocessings=preprocessings, fold_id=fold_id, fold_index=fold_idx, score=round(score, 6)).model_dump())
        entries.sort(key=lambda e: (e.get("chain_id", ""), e.get("fold_index", 0)))
        return FoldStabilityResponse(entries=entries, fold_ids=sorted(all_fold_ids), score_column=effective_score_column, total_chains=len({e.get("chain_id") for e in entries}))
    finally:
        store.close()


class ConfusionMatrixRequest(_StrictModel):
    """Request body for /inspector/confusion."""

    chain_ids: list[str]
    partition: str = "val"
    normalize: str = "none"
    target_index: int = Field(default=0, ge=0)


class ConfusionMatrixCell(_StrictModel):
    """A single cell in the confusion matrix."""

    true_label: str
    pred_label: str
    count: int
    normalized: float | None = None


class ConfusionMatrixResponse(_StrictModel):
    """Response for /inspector/confusion."""

    cells: list[dict[str, Any]]
    labels: list[str]
    total_samples: int
    partition: str
    normalize: str
    reason: str | None = None


def get_confusion_matrix(request: ConfusionMatrixRequest):
    """Confusion matrix for classification chains.

    Aggregates y_true/y_pred from fold-level predictions across the
    selected chains and computes the confusion matrix with optional
    normalization (by row=recall, column=precision, or all).
    """
    if not request.chain_ids:
        return ConfusionMatrixResponse(cells=[], labels=[], total_samples=0, partition=request.partition, normalize=request.normalize)
    store = _get_store()
    try:
        df = store.query_chain_summaries()
        chain_records = {record["chain_id"]: record for record in _normalize_chain_records(store, df)}
        eligible_chain_ids = [chain_id for chain_id in request.chain_ids if _is_classification_task(chain_records.get(chain_id, {}).get("task_type"))]
        if not eligible_chain_ids:
            return ConfusionMatrixResponse(cells=[], labels=[], total_samples=0, partition=request.partition, normalize=request.normalize, reason="Confusion matrix is only available for classification chains.")
        all_y_true: list[Any] = []
        all_y_pred: list[Any] = []
        for chain_id in eligible_chain_ids:
            pred_df = store.get_chain_predictions(chain_id=chain_id, partition=request.partition)
            if len(pred_df) == 0:
                continue
            for row in pred_df.iter_rows(named=True):
                row_dict = dict(row)
                prediction_id = row_dict.get("prediction_id")
                if not prediction_id:
                    continue
                arrays = _get_arrays(store, prediction_id)
                if arrays is None:
                    continue
                y_true_values = _coerce_target_vector(arrays.get("y_true"), request.target_index)
                y_pred_values = _coerce_target_vector(arrays.get("y_pred"), request.target_index)
                if not y_true_values or not y_pred_values:
                    continue
                for yt, yp in zip(y_true_values, y_pred_values, strict=False):
                    if isinstance(yt, float) and (math.isnan(yt) or math.isinf(yt)):
                        continue
                    if isinstance(yp, float) and (math.isnan(yp) or math.isinf(yp)):
                        continue
                    all_y_true.append(yt)
                    all_y_pred.append(yp)
        if not all_y_true or not all_y_pred:
            return ConfusionMatrixResponse(
                cells=[], labels=[], total_samples=0, partition=request.partition, normalize=request.normalize, reason="Selected classification chains do not have prediction arrays for this partition."
            )
        y_true_labels = [str(v) for v in all_y_true]
        y_pred_labels = [str(v) for v in all_y_pred]
        labels = sorted(set(y_true_labels) | set(y_pred_labels))
        if len(labels) > 24:
            return ConfusionMatrixResponse(
                cells=[], labels=[], total_samples=0, partition=request.partition, normalize=request.normalize, reason=f"Too many class labels ({len(labels)}) to render a meaningful confusion matrix."
            )
        counts: dict[tuple[str, str], int] = {}
        for yt, yp in zip(y_true_labels, y_pred_labels, strict=False):
            counts[yt, yp] = counts.get((yt, yp), 0) + 1
        total_samples = len(y_true_labels)
        row_sums: dict[str, int] = {}
        col_sums: dict[str, int] = {}
        for tl in labels:
            row_sums[tl] = sum(counts.get((tl, pl), 0) for pl in labels)
            col_sums[tl] = sum(counts.get((tl2, tl), 0) for tl2 in labels)
        cells: list[dict] = []
        for tl in labels:
            for pl in labels:
                count = counts.get((tl, pl), 0)
                normalized: float | None = None
                if request.normalize == "row" and row_sums[tl] > 0:
                    normalized = round(count / row_sums[tl], 4)
                elif request.normalize == "column" and col_sums[pl] > 0:
                    normalized = round(count / col_sums[pl], 4)
                elif request.normalize == "all" and total_samples > 0:
                    normalized = round(count / total_samples, 4)
                cells.append(ConfusionMatrixCell(true_label=tl, pred_label=pl, count=count, normalized=normalized).model_dump())
        return ConfusionMatrixResponse(cells=cells, labels=labels, total_samples=total_samples, partition=request.partition, normalize=request.normalize, reason=None)
    finally:
        store.close()


class PreprocessingImpactRequest(_StrictModel):
    """Request body for /inspector/preprocessing-impact."""

    run_id: list[str] | None = None
    dataset_name: list[str] | None = None
    score_column: str = "cv_val_score"


class PreprocessingImpactEntry(_StrictModel):
    """One preprocessing step's impact on score."""

    step_name: str
    impact: float | None = None
    mean_with: float | None = None
    mean_without: float | None = None
    count_with: int = 0
    count_without: int = 0


class PreprocessingImpactResponse(_StrictModel):
    """Response for /inspector/preprocessing-impact."""

    entries: list[dict[str, Any]]
    score_column: str
    total_chains: int


class HyperparameterRequest(_StrictModel):
    """Request body for /inspector/hyperparameter."""

    run_id: list[str] | None = None
    dataset_name: list[str] | None = None
    param_name: str
    score_column: str = "cv_val_score"


class HyperparameterPoint(_StrictModel):
    """One chain's hyperparameter value + score."""

    chain_id: str
    param_value: float
    score: float
    model_class: str


class HyperparameterResponse(_StrictModel):
    """Response for /inspector/hyperparameter."""

    points: list[dict[str, Any]]
    param_name: str
    score_column: str
    available_params: list[str]
    reason: str | None = None


class BiasVarianceRequest(_StrictModel):
    """Request body for /inspector/bias-variance."""

    chain_ids: list[str]
    score_column: str = "cv_val_score"
    score_ref: ScoreRef | None = None
    group_by: str = "model_class"


class BiasVarianceEntry(_StrictModel):
    """One group's bias-variance decomposition."""

    group_label: str
    bias_squared: float | None = None
    variance: float | None = None
    total_error: float | None = None
    n_chains: int = 0
    n_folds: int = 0
    n_samples: int = 0
    chain_ids: list[str] = []


class BiasVarianceResponse(_StrictModel):
    """Response for /inspector/bias-variance."""

    entries: list[dict[str, Any]]
    score_column: str
    group_by: str
    reason: str | None = None


def get_preprocessing_impact(request: PreprocessingImpactRequest):
    """Analyse the impact of each preprocessing step on the score.

    For each unique preprocessing step found across chains, computes
    the mean score of chains with vs without that step, and the impact
    (difference). Sign is flipped for lower-is-better metrics.
    """
    import numpy as np

    store = _get_store()
    try:
        df = store.query_chain_summaries(run_id=request.run_id, dataset_name=request.dataset_name)
        records = _normalize_chain_records(store, df)
        if not records:
            return PreprocessingImpactResponse(entries=[], score_column=request.score_column, total_chains=0)
        step_chains: dict[str, list[int]] = {}
        for i, r in enumerate(records):
            for step in r.get("preprocessing_steps") or []:
                step_chains.setdefault(str(step), []).append(i)
        first_metric = next((r.get("metric") for r in records if r.get("metric")), None)
        lower_better = _is_lower_better(first_metric)
        entries: list[dict] = []
        all_indices = set(range(len(records)))
        for step_name, indices_with in step_chains.items():
            with_set = set(indices_with)
            without_set = all_indices - with_set
            scores_with = [float(records[i][request.score_column]) for i in with_set if records[i].get(request.score_column) is not None]
            scores_without = [float(records[i][request.score_column]) for i in without_set if records[i].get(request.score_column) is not None]
            if not scores_with or not scores_without:
                continue
            mean_w = float(np.mean(scores_with))
            mean_wo = float(np.mean(scores_without))
            impact = mean_w - mean_wo
            if lower_better:
                impact = -impact
            entries.append(
                PreprocessingImpactEntry(
                    step_name=step_name,
                    impact=sanitize_float(round(impact, 6)),
                    mean_with=sanitize_float(round(mean_w, 6)),
                    mean_without=sanitize_float(round(mean_wo, 6)),
                    count_with=len(scores_with),
                    count_without=len(scores_without),
                ).model_dump()
            )
        entries.sort(key=lambda e: abs(e.get("impact") or 0), reverse=True)
        return PreprocessingImpactResponse(entries=entries, score_column=request.score_column, total_chains=len(records))
    finally:
        store.close()


def get_hyperparameter_data(request: HyperparameterRequest):
    """Get hyperparameter value vs score scatter data.

    Extracts numeric values of the requested param_name from each chain's
    variant params, paired with the chain's score.
    """
    store = _get_store()
    try:
        df = store.query_chain_summaries(run_id=request.run_id, dataset_name=request.dataset_name)
        records = _normalize_chain_records(store, df)
        if not records:
            return HyperparameterResponse(points=[], param_name=request.param_name, score_column=request.score_column, available_params=[], reason="No chains match the current filters.")
        param_counts: dict[str, int] = {}
        param_values: dict[str, set[float]] = {}
        for r in records:
            params = _flatten_numeric_params(r.get("variant_params") or r.get("best_params"))
            for key, value in params.items():
                param_counts[key] = param_counts.get(key, 0) + 1
                param_values.setdefault(key, set()).add(value)
        available_params = sorted((key for key, count in param_counts.items() if count >= 2 and len(param_values.get(key, set())) >= 2))
        if request.param_name not in available_params:
            reason = (
                "No numeric model parameters with enough variation are available for the current filter set." if not available_params else f"Parameter '{request.param_name}' is not available for the current filter set."
            )
            return HyperparameterResponse(points=[], param_name=request.param_name, score_column=request.score_column, available_params=available_params, reason=reason)
        points: list[dict] = []
        for r in records:
            params = _flatten_numeric_params(r.get("variant_params") or r.get("best_params"))
            val = params.get(request.param_name)
            score = r.get(request.score_column)
            if val is not None and score is not None:
                points.append(HyperparameterPoint(chain_id=r["chain_id"], param_value=val, score=float(score), model_class=r.get("model_class", "Unknown")).model_dump())
        return HyperparameterResponse(
            points=points, param_name=request.param_name, score_column=request.score_column, available_params=available_params, reason=None if points else f"No chains expose '{request.param_name}' with a valid score."
        )
    finally:
        store.close()


def get_bias_variance(request: BiasVarianceRequest):
    """Bias-variance decomposition grouped by a chain field.

    For each group, collects fold-level predictions across chains.
    Per sample appearing in 2+ folds:
      biasÂ² = (mean_pred - y_true)Â²
      variance = Var(y_pred across folds)
    Aggregated per group: mean_biasÂ², mean_variance, total_error.
    """
    from nirs4all.pipeline.analysis.model_diagnostics import bias_variance_decomposition

    effective_score_column = _resolve_score_column_from_score_ref(request.score_column, request.score_ref)
    effective_partition = _prediction_partition_from_score_column(effective_score_column, "val") if _has_resolved_score_ref(request.score_column, request.score_ref) else "val"
    if not request.chain_ids:
        return BiasVarianceResponse(entries=[], score_column=effective_score_column, group_by=request.group_by, reason="Select at least one chain to compare bias and variance.")
    store = _get_store()
    try:
        df = store.query_chain_summaries()
        records = {record["chain_id"]: record for record in _normalize_chain_records(store, df)}
        groups: dict[str, list[str]] = {}
        for cid in request.chain_ids:
            r = records.get(cid)
            if not r:
                continue
            label = str(r.get(request.group_by, "Unknown") or "Unknown")
            groups.setdefault(label, []).append(cid)
        if not groups:
            return BiasVarianceResponse(entries=[], score_column=effective_score_column, group_by=request.group_by, reason="No eligible chains were found for the requested comparison.")
        entries: list[dict] = []
        for label, chain_ids in groups.items():
            sample_preds: dict[Hashable, list[tuple[float, float]]] = {}
            total_folds = 0
            for cid in chain_ids:
                record = records.get(cid, {})
                dataset_name = str(record.get("dataset_name") or "unknown")
                pred_df = store.get_chain_predictions(chain_id=cid, partition=effective_partition)
                if len(pred_df) == 0:
                    continue
                for row in pred_df.iter_rows(named=True):
                    row_dict = dict(row)
                    if not _is_fold_prediction(row_dict, effective_score_column):
                        continue
                    pid = row_dict.get("prediction_id")
                    if not pid:
                        continue
                    total_folds += 1
                    arrays = _get_arrays(store, pid)
                    if arrays is None:
                        continue
                    y_true_values = _coerce_numeric_vector(arrays.get("y_true"))
                    y_pred_values = _coerce_numeric_vector(arrays.get("y_pred"))
                    if not y_true_values or not y_pred_values:
                        continue
                    indices = _coerce_index_vector(arrays.get("sample_indices"))
                    pair_count = min(len(y_true_values), len(y_pred_values))
                    for j in range(pair_count):
                        yt = y_true_values[j]
                        yp = y_pred_values[j]
                        if math.isnan(yt) or math.isnan(yp) or math.isinf(yt) or math.isinf(yp):
                            continue
                        sample_idx = indices[j] if indices is not None and j < len(indices) else j
                        sample_key = (dataset_name, int(sample_idx))
                        sample_preds.setdefault(sample_key, []).append((yt, yp))
            decomposition = bias_variance_decomposition(sample_preds)
            if decomposition is not None:
                entries.append(
                    BiasVarianceEntry(
                        group_label=label,
                        bias_squared=sanitize_float(round(decomposition.bias_squared, 6)),
                        variance=sanitize_float(round(decomposition.variance, 6)),
                        total_error=sanitize_float(round(decomposition.total_error, 6)),
                        n_chains=len(chain_ids),
                        n_folds=total_folds,
                        n_samples=decomposition.n_samples,
                        chain_ids=chain_ids,
                    ).model_dump()
                )
        entries.sort(key=lambda e: e.get("total_error") or 0, reverse=True)
        return BiasVarianceResponse(
            entries=entries,
            score_column=effective_score_column,
            group_by=request.group_by,
            reason=None if entries else "Bias-variance needs repeated validation predictions for the same samples across at least two comparable chains.",
        )
    finally:
        store.close()


_VIEWS = {
    "data": "get_inspector_data",
    "scatter": "get_scatter_data",
    "histogram": "get_histogram_data",
    "rankings": "get_rankings_data",
    "heatmap": "get_heatmap_data",
    "candlestick": "get_candlestick_data",
    "branch-comparison": "get_branch_comparison",
    "branch-topology": "get_branch_topology",
    "fold-stability": "get_fold_stability",
    "confusion": "get_confusion_matrix",
    "preprocessing-impact": "get_preprocessing_impact",
    "hyperparameter": "get_hyperparameter_data",
    "bias-variance": "get_bias_variance",
}


def read_inspector_view(operation: str, workspace_path: str, request: dict[str, Any]) -> dict[str, Any]:
    """Read one validated view from an explicit read-only workspace snapshot."""
    if not isinstance(workspace_path, str) or not workspace_path:
        raise ValueError("Inspector requires an explicit workspace path")
    if not isinstance(request, dict):
        raise ValueError("Inspector request must be an object")
    name = _VIEWS.get(operation.removeprefix("inspector."))
    if name is None:
        raise ValueError("Unknown Inspector view")
    function = globals()[name]
    signature = inspect.signature(function)
    hints = get_type_hints(function)
    kwargs: dict[str, Any]
    if "request" in signature.parameters:
        kwargs = {"request": hints["request"].model_validate(request)}
    else:
        if set(request) - set(signature.parameters):
            raise ValueError("Unknown Inspector query field")
        kwargs = dict(request)
        for key, value in request.items():
            if key in {"run_id", "dataset_name", "model_class", "preprocessings"} and (not isinstance(value, list) or len(value) > 256 or any(not isinstance(item, str) or not item or len(item) > 256 for item in value)):
                raise ValueError("Inspector filters require at most 256 nonempty strings")
        for key, low, high in (("n_bins", 5, 100), ("limit", 1, 500), ("offset", 0, 1000000)):
            if key in kwargs and (isinstance(kwargs[key], bool) or not isinstance(kwargs[key], int) or not low <= kwargs[key] <= high):
                raise ValueError(f"Invalid Inspector {key}")
    ids = request.get("chain_ids")
    if ids is not None and (not isinstance(ids, list) or len(ids) > 256 or any(not isinstance(item, str) or not item or len(item) > 256 for item in ids)):
        raise ValueError("Inspector chain selection exceeds its bounds")
    token = _workspace.set(workspace_path)
    try:
        result = function(**kwargs)
        return sanitize_dict(result.model_dump(mode="json") if isinstance(result, BaseModel) else result)
    finally:
        _workspace.reset(token)
