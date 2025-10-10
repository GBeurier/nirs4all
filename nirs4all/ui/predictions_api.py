from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
from datetime import datetime

from nirs4all.ui import utils

try:
    from nirs4all.dataset.predictions import Predictions
except Exception:  # pragma: no cover - best-effort import
    Predictions = None

router = APIRouter()


def _load_predictions_instance(dataset: Optional[str]) -> Tuple[Any, Path, Optional[float]]:
    """Load a Predictions instance for the given dataset and return (preds, results_path, preds_file_mtime).

    This centralizes the logic that decides whether the dataset argument is a folder, a name or omitted.
    """
    results_path = utils.resolve_results_path()
    preds_file_mtime = None

    if Predictions is None:
        raise HTTPException(status_code=500, detail="Predictions backend is not available in this environment")

    try:
        if not dataset:
            preds = Predictions.load(path=str(results_path))
        else:
            ds_path = Path(dataset)
            if ds_path.exists() and ds_path.is_dir():
                preds_file = ds_path / 'predictions.json'
                if preds_file.exists():
                    preds = Predictions.load_from_file_cls(str(preds_file))
                else:
                    preds = Predictions()
            else:
                preds = Predictions.load(dataset_name=dataset, path=str(results_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load predictions: {e}") from e

    if dataset:
        ds_p = Path(dataset)
        if not (ds_p.exists() and ds_p.is_dir()):
            ds_p = results_path / dataset
        preds_file = ds_p / 'predictions.json'
        if preds_file.exists():
            try:
                preds_file_mtime = preds_file.stat().st_mtime
            except Exception:
                preds_file_mtime = None

    return preds, results_path, preds_file_mtime


def _serialize_all_rows(preds: Any, preds_file_mtime: Optional[float], include_arrays: bool) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Serialize all prediction rows from a Predictions instance.

    Returns (serialized_rows, columns_set)
    """
    try:
        rows = preds.filter_predictions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query predictions: {e}") from e

    serialized: List[Dict[str, Any]] = []
    columns_set: List[str] = []

    for r in rows:
        ser, added = _serialize_prediction_row(r, preds_file_mtime, include_arrays)
        serialized.append(ser)
        for k in added:
            if k not in columns_set:
                columns_set.append(k)

    return serialized, columns_set


def _apply_sort_and_paginate(filtered: List[Dict[str, Any]], sort_by: Optional[str], sort_dir: str,
                             page: int, page_size: int, cursor: Optional[int]):
    """Sort and paginate a list of prediction rows (in-memory). Returns (page_rows, total, page, page_size, start, next_cursor, prev_cursor)."""
    total = len(filtered)

    if sort_by:
        reverse = (sort_dir.lower() != 'asc')
        try:
            filtered.sort(key=lambda r: (r.get(sort_by) is None, r.get(sort_by)), reverse=reverse)
        except Exception:
            filtered.sort(key=lambda r: str(r.get(sort_by, '')), reverse=reverse)

    # paginate
    if cursor is not None:
        try:
            start = int(cursor)
            if start < 0:
                start = 0
        except Exception:
            start = 0
    else:
        if page < 1:
            page = 1
        start = (page - 1) * page_size
    end = start + page_size
    page_rows = filtered[start:end]

    next_cursor = end if end < total else None
    prev_cursor = max(0, start - page_size) if start > 0 else None
    return page_rows, total, page, page_size, start, next_cursor, prev_cursor


def _serialize_prediction_row(r: Dict[str, Any], preds_file_mtime: Optional[float], include_arrays: bool) -> Tuple[Dict[str, Any], List[str]]:
    """Serialize a single prediction dict to JSON-serializable types and compute column keys added.

    Returns (serialized_row, added_keys_list).
    """
    ser: Dict[str, Any] = {}
    added: List[str] = []
    for k, v in r.items():
        ser[k] = utils.to_json_basic(v)
        added.append(k)

    # attach a best-effort date
    date_val = None
    meta = r.get('metadata') if isinstance(r.get('metadata'), dict) else None
    if meta:
        date_val = meta.get('created_at') or meta.get('saved_at') or meta.get('timestamp')
    if not date_val and preds_file_mtime:
        try:
            date_val = datetime.utcfromtimestamp(preds_file_mtime).isoformat()
        except Exception:
            date_val = None
    ser['_date'] = date_val

    # strip heavy arrays unless requested
    if not include_arrays:
        ser = utils.strip_heavy_arrays(ser, include_arrays=include_arrays)

    return ser, added


def _compute_ordered_columns(columns_set: List[str]) -> List[str]:
    """Return a sensible ordered list of unique column names, preferring common fields."""
    preferred = [
        'id', 'dataset_name', 'dataset_path', 'config_name', 'config_path',
        'model_name', 'model_classname', 'metric', 'test_score', 'val_score', 'train_score',
        'fold_id', 'op_counter', 'n_samples', 'n_features'
    ]
    ordered: List[str] = []
    for p in preferred:
        if p in columns_set and p not in ordered:
            ordered.append(p)
    for c in columns_set:
        if c not in ordered:
            ordered.append(c)
    return ordered


@router.get("/api/predictions/counts")
def predictions_counts(path: str = "results") -> Dict[str, Any]:
    """
    Return counts of stored predictions per dataset.
    """
    if Predictions is None:
        raise HTTPException(status_code=500, detail="Predictions backend is not available in this environment")

    try:
        preds = Predictions.load(path=path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading predictions from '{path}': {e}") from e

    try:
        datasets = preds.get_datasets()
        counts = {}
        for ds in datasets:
            rows = preds.filter_predictions(dataset_name=ds)
            counts[ds] = len(rows)

        return {"counts": counts, "total": len(preds)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/predictions/search")
def predictions_search(dataset: Optional[str] = None,
                       page: int = 1,
                       page_size: int = 50,
                       cursor: Optional[int] = None,
                       sort_by: Optional[str] = None,
                       sort_dir: str = 'desc',
                       model_name: Optional[str] = None,
                       config_name: Optional[str] = None,
                       partition: Optional[str] = None,
                       min_score: Optional[float] = None,
                       max_score: Optional[float] = None,
                       score_field: str = 'test_score',
                       q: Optional[str] = None,
                       date_from: Optional[str] = None,
                       date_to: Optional[str] = None,
                       include_arrays: bool = False) -> Dict[str, Any]:
    """Server-side paginated, sortable and filterable predictions endpoint."""
    results = predictions_list(dataset, include_arrays=include_arrays) if dataset is not None else predictions_all(include_arrays=include_arrays)
    rows = results.get('predictions', [])

    filtered = utils.apply_filters_to_rows(
        rows,
        model_name=model_name,
        config_name=config_name,
        partition=partition,
        min_score=min_score,
        max_score=max_score,
        score_field=score_field,
        text_query=q,
        date_from=date_from,
        date_to=date_to,
    )

    total = len(filtered)

    if sort_by:
        reverse = (sort_dir.lower() != 'asc')
        try:
            filtered.sort(key=lambda r: (r.get(sort_by) is None, r.get(sort_by)), reverse=reverse)
        except Exception:
            filtered.sort(key=lambda r: str(r.get(sort_by, '')), reverse=reverse)

    if cursor is not None:
        try:
            start = int(cursor)
            if start < 0:
                start = 0
        except Exception:
            start = 0
    else:
        if page < 1:
            page = 1
        start = (page - 1) * page_size
    end = start + page_size
    page_rows = filtered[start:end]

    cols = results.get('columns') or []
    if sort_by and sort_by not in cols:
        cols = [sort_by] + cols

    next_cursor = end if end < total else None
    prev_cursor = max(0, start - page_size) if start > 0 else None
    return {"predictions": page_rows, "columns": cols, "total": total, "page": page, "page_size": page_size, "cursor": start, "next_cursor": next_cursor, "prev_cursor": prev_cursor}


@router.get("/api/predictions/datasets")
def predictions_datasets() -> Dict[str, Any]:
    """List direct dataset subfolders inside the configured results folder."""
    current_ws = utils.get_current_workspace()
    if current_ws is None:
        return {"datasets": [], "results_path": None}

    results_path = utils.resolve_results_path()
    if not results_path.exists() or not results_path.is_dir():
        return {"datasets": [], "results_path": str(results_path)}

    datasets = []
    for entry in sorted(results_path.iterdir(), key=lambda e: e.name.lower()):
        if not entry.is_dir():
            continue
        preds_file = entry / 'predictions.json'
        count = 0
        file_mtime = None
        if preds_file.exists():
            try:
                if Predictions is not None:
                    p = Predictions.load_from_file_cls(str(preds_file))
                    count = p.num_predictions
                else:
                    count = 0
                try:
                    file_mtime = preds_file.stat().st_mtime
                except Exception:
                    file_mtime = None
            except Exception:
                count = 0

        datasets.append({
            "name": entry.name,
            "path": str(entry),
            "predictions_file": str(preds_file) if preds_file.exists() else None,
            "predictions_count": count,
            "predictions_mtime": file_mtime,
        })

    return {"datasets": datasets, "results_path": str(results_path)}


@router.get("/api/predictions/list")
def predictions_list(dataset: Optional[str] = None, include_arrays: bool = False) -> Dict[str, Any]:
    """Return predictions for a dataset (or all datasets when dataset is omitted)."""
    results_path = utils.resolve_results_path()
    if not results_path.exists() or not results_path.is_dir():
        return {"predictions": [], "columns": [], "count": 0, "results_path": str(results_path)}

    if Predictions is None:
        raise HTTPException(status_code=500, detail="Predictions backend is not available in this environment")

    try:
        if not dataset:
            preds = Predictions.load(path=str(results_path))
        else:
            ds_path = Path(dataset)
            if ds_path.exists() and ds_path.is_dir():
                preds_file = ds_path / 'predictions.json'
                if preds_file.exists():
                    preds = Predictions.load_from_file_cls(str(preds_file))
                else:
                    preds = Predictions()
            else:
                preds = Predictions.load(dataset_name=dataset, path=str(results_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load predictions: {e}") from e

    try:
        rows = preds.filter_predictions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query predictions: {e}") from e

    serialized = []
    columns_set = []
    preds_file_mtime = None
    if dataset:
        ds_p = Path(dataset)
        if not (ds_p.exists() and ds_p.is_dir()):
            ds_p = results_path / dataset
        preds_file = ds_p / 'predictions.json'
        if preds_file.exists():
            try:
                preds_file_mtime = preds_file.stat().st_mtime
            except Exception:
                preds_file_mtime = None

    for r in rows:
        ser = {}
        for k, v in r.items():
            ser[k] = utils.to_json_basic(v)
            if k not in columns_set:
                columns_set.append(k)
        date_val = None
        meta = r.get('metadata') if isinstance(r.get('metadata'), dict) else None
        if meta:
            date_val = meta.get('created_at') or meta.get('saved_at') or meta.get('timestamp')
        if not date_val and preds_file_mtime:
            try:
                date_val = datetime.utcfromtimestamp(preds_file_mtime).isoformat()
            except Exception:
                date_val = None
        ser['_date'] = date_val
        # Strip heavy arrays when requested to reduce payload
        if not include_arrays:
            ser = utils.strip_heavy_arrays(ser, include_arrays=include_arrays)
        serialized.append(ser)

    preferred = [
        'id', 'dataset_name', 'dataset_path', 'config_name', 'config_path',
        'model_name', 'model_classname', 'metric', 'test_score', 'val_score', 'train_score',
        'fold_id', 'op_counter', 'n_samples', 'n_features'
    ]
    ordered = []
    for p in preferred:
        if p in columns_set and p not in ordered:
            ordered.append(p)
    # When arrays are stripped, remove heavy fields from the column list so UIs don't show empty columns
    if not include_arrays:
        columns_set = [c for c in columns_set if c not in utils.HEAVY_ARRAY_FIELDS]

    for c in columns_set:
        if c not in ordered:
            ordered.append(c)

    return {"predictions": serialized, "columns": ordered, "count": len(serialized), "results_path": str(results_path)}


@router.get("/api/predictions/all")
def predictions_all(include_arrays: bool = False) -> Dict[str, Any]:
    """Return all predictions across datasets as a list of lightweight records."""
    results_path = utils.resolve_results_path()
    if Predictions is None:
        raise HTTPException(status_code=500, detail="Predictions backend not available")
    try:
        preds = Predictions.load(path=str(results_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load predictions: {e}") from e

    rows = preds.filter_predictions()
    out = []
    for r in rows:
        rec = {k: utils.to_json_basic(v) for k, v in r.items()}
        if not include_arrays:
            rec = utils.strip_heavy_arrays(rec, include_arrays=include_arrays)
        out.append(rec)
    return {"predictions": out, "count": len(out), "results_path": str(results_path)}


@router.get("/api/predictions/meta")
def predictions_meta(dataset: Optional[str] = None) -> Dict[str, Any]:
    results_path = utils.resolve_results_path()
    if Predictions is None:
        raise HTTPException(status_code=500, detail="Predictions backend not available")

    try:
        if dataset:
            preds = Predictions.load(dataset_name=dataset, path=str(results_path))
        else:
            preds = Predictions.load(path=str(results_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load predictions for meta: {e}") from e

    try:
        models = preds.get_models()
    except Exception:
        models = []
    try:
        configs = preds.get_configs()
    except Exception:
        configs = []
    try:
        parts = preds.get_partitions()
    except Exception:
        parts = []

    return {"models": models, "configs": configs, "partitions": parts, "count": preds.num_predictions}


@router.get("/api/pipeline/from-prediction")
def pipeline_from_prediction(prediction_id: str) -> Dict[str, Any]:
    if Predictions is None:
        raise HTTPException(status_code=500, detail="Predictions backend not available")

    results_path = utils.resolve_results_path()
    try:
        preds = Predictions.load(path=str(results_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load predictions: {e}") from e

    try:
        matches = preds.filter_predictions(id=prediction_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching prediction id: {e}") from e

    if not matches:
        raise HTTPException(status_code=404, detail=f"Prediction id not found: {prediction_id}")

    rec = matches[0]
    cfg_path = rec.get('config_path') or rec.get('config') or ''
    if not cfg_path:
        raise HTTPException(status_code=404, detail="Prediction does not contain a config_path")

    cfg_p = Path(cfg_path)
    if not cfg_p.is_absolute():
        cfg_p = results_path / cfg_p

    pipeline_file = cfg_p / 'pipeline.json'
    if not pipeline_file.exists():
        raise HTTPException(status_code=404, detail=f"pipeline.json not found at {pipeline_file}")

    try:
        with pipeline_file.open('r', encoding='utf-8') as fh:
            data = json.load(fh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read pipeline file: {e}") from e

    return {
        'pipeline': data,
        'prediction': {
            'id': rec.get('id'),
            'dataset_name': rec.get('dataset_name'),
            'model_name': rec.get('model_name'),
            'config_path': str(cfg_p),
        }
    }
