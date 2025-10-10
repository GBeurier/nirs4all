from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import hashlib
# stat import not needed
import os
import json
from datetime import datetime
import sys
import subprocess
import numpy as np

try:
    # Import predictions manager from nirs4all core
    from nirs4all.dataset.predictions import Predictions
except Exception:  # pragma: no cover - best-effort import
    Predictions = None

try:
    from nirs4all.dataset.dataset_config_parser import folder_to_name
except Exception:  # pragma: no cover - fall back to simple namer
    def folder_to_name(folder_path: str) -> str:  # type: ignore[override]
        p = Path(folder_path)
        return p.name or 'dataset'


app = FastAPI(title="nirs4all UI (minimal)")

# Allow simple CORS for local development so the UI can POST/OPTIONS without
# cross-origin issues when served from different origins during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Workspace persistence configuration ---------------------------------
# Location to store a pointer to the last workspace (per-user)
GLOBAL_CONFIG_DIR = Path.home() / ".nirs4all"
GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LAST_WORKSPACE_FILE = GLOBAL_CONFIG_DIR / "last_workspace.json"
WORKSPACE_CONFIG_FILENAME = ".nirs4all_workspace.json"

# In-memory cache for the currently selected workspace and its config
CURRENT_WORKSPACE_PATH: Optional[Path] = None
CURRENT_WORKSPACE_CONFIG: Optional[Dict[str, Any]] = None


def _save_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _workspace_config_path(workspace_path: Path) -> Path:
    return workspace_path / WORKSPACE_CONFIG_FILENAME


def load_workspace_config(workspace_path: Path) -> Dict[str, Any]:
    cfg_path = _workspace_config_path(workspace_path)
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            # If corrupt, return a safe default
            return {"datasets": [], "pipeline_repo": None, "created": None}
    # default config
    return {"datasets": [], "pipeline_repo": None, "created": datetime.utcnow().isoformat()}


def save_workspace_config(workspace_path: Path, config: Dict[str, Any]) -> None:
    cfg_path = _workspace_config_path(workspace_path)
    _save_json_atomic(cfg_path, config)


def save_last_workspace_pointer(path: Path) -> None:
    LAST_WORKSPACE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"path": str(path), "updated": datetime.utcnow().isoformat()}
    _save_json_atomic(LAST_WORKSPACE_FILE, data)


def load_last_workspace_pointer() -> Optional[Path]:
    env_path = os.environ.get("NIRS4ALL_WORKSPACE")
    if env_path:
        try:
            p = Path(env_path).expanduser().resolve()
            return p
        except Exception:
            return None
    if LAST_WORKSPACE_FILE.exists():
        try:
            with LAST_WORKSPACE_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                path_val = data.get("path")
                if path_val:
                    return Path(path_val).expanduser().resolve()
                return None
        except Exception:
            return None
    return None


def _ensure_workspace_loaded_on_startup() -> None:
    global CURRENT_WORKSPACE_PATH, CURRENT_WORKSPACE_CONFIG
    p = load_last_workspace_pointer()
    if p and p.exists():
        CURRENT_WORKSPACE_PATH = p
        CURRENT_WORKSPACE_CONFIG = load_workspace_config(p)
        # also set in-process env so other code sees it during this run
        os.environ["NIRS4ALL_WORKSPACE"] = str(p)


# Load a workspace pointer at module import time if available
_ensure_workspace_loaded_on_startup()

# Serve the static UI from the top-level `webapp/` folder. For backwards
# compatibility we fallback to the docs/webapp_mockup path if the new folder
# does not exist yet.
HERE = Path(__file__).resolve()
WEBAPP_DIR = HERE.parents[2] / "webapp"
if WEBAPP_DIR.exists():
    # Serve static UI under /webapp to avoid intercepting API POST calls
    app.mount("/webapp", StaticFiles(directory=str(WEBAPP_DIR), html=True), name="webapp")

    # Provide a simple root redirect to the SPA index so users can open '/'
    @app.get("/")
    def _root_redirect():
        return RedirectResponse("/webapp/index.html")
else:
    # If the webapp folder is missing we intentionally do not mount a static
    # site so that only the API is exposed. This helps keep behavior
    # deterministic in CI/packaging and prevents accidentally serving the
    # large mockup from docs/.
    print(f"ℹ️ webapp directory not found at {WEBAPP_DIR}; static UI not mounted")


@app.get("/api/ping")
def ping() -> Dict[str, str]:
    return {"status": "ok", "service": "nirs4all-ui-minimal"}


@app.get("/api/predictions/counts")
def predictions_counts(path: str = "results") -> Dict[str, Any]:
    """
    Return counts of stored predictions per dataset.

    Query params:
    - path: location where predictions files are stored (default: 'results')
    """
    if Predictions is None:
        raise HTTPException(status_code=500, detail="Predictions backend is not available in this environment")

    try:
        preds = Predictions.load(path=path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading predictions from '{path}': {e}") from e

    # Build counts per dataset using existing API
    try:
        datasets = preds.get_datasets()
        counts = {}
        for ds in datasets:
            # filter_predictions returns list-like; use length
            rows = preds.filter_predictions(dataset_name=ds)
            counts[ds] = len(rows)

        return {"counts": counts, "total": len(preds)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _resolve_results_path() -> Path:
    """Resolve the results folder to an absolute Path using the current workspace config.

    Priority:
    - If workspace config contains an absolute path in 'results_folder' use it
    - If workspace config contains a relative path, interpret it relative to the workspace
    - Otherwise fallback to a 'results' folder relative to the workspace or current working dir
    """
    cfg = CURRENT_WORKSPACE_CONFIG or (load_workspace_config(CURRENT_WORKSPACE_PATH) if CURRENT_WORKSPACE_PATH else {})
    rf = cfg.get('results_folder') if cfg else None
    if rf:
        p = Path(rf)
        if p.is_absolute():
            return p.expanduser().resolve()
        # relative to workspace when possible
        if CURRENT_WORKSPACE_PATH is not None:
            return (CURRENT_WORKSPACE_PATH / p).expanduser().resolve()
        return p.expanduser().resolve()
    # fallback
    if CURRENT_WORKSPACE_PATH is not None:
        return (CURRENT_WORKSPACE_PATH / 'results').expanduser().resolve()
    return Path('results').expanduser().resolve()


def _to_json_basic(obj: Any):
    """Convert numpy and Path types to JSON-serializable basic Python types.

    This function recursively handles common container types as well.
    """
    # numpy array -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # numpy scalar -> python scalar
    if hasattr(obj, 'dtype') and getattr(obj, 'dtype', None) is not None:
        try:
            return obj.item()
        except Exception:
            pass
    # numpy generic (e.g., np.int64, np.float64)
    if isinstance(obj, (np.generic,)):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_json_basic(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_basic(v) for v in obj]
    # default: let json serializer handle or convert to string as last resort
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _parse_iso_date(v: Optional[str]) -> Optional[datetime]:
    if not v:
        return None
    try:
        return datetime.fromisoformat(v)
    except Exception:
        # try common fallback
        try:
            return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None


def _apply_filters_to_rows(rows: List[Dict[str, Any]],
                           model_name: Optional[str] = None,
                           config_name: Optional[str] = None,
                           partition: Optional[str] = None,
                           min_score: Optional[float] = None,
                           max_score: Optional[float] = None,
                           score_field: str = 'test_score',
                           text_query: Optional[str] = None,
                           date_from: Optional[str] = None,
                           date_to: Optional[str] = None) -> List[Dict[str, Any]]:
    """Filter a list of prediction rows (dictionaries) in memory.

    Supports equality filters, numeric ranges and a simple substring text search.
    Date filters operate on '_date' field when present or on metadata.created_at.
    """
    out = []
    q = text_query.lower() if text_query else None
    df = _parse_iso_date(date_from) if date_from else None
    dt = _parse_iso_date(date_to) if date_to else None

    for r in rows:
        # equality filters
        if model_name and str(r.get('model_name', '')).lower() != model_name.lower():
            continue
        if config_name and str(r.get('config_name', '')).lower() != config_name.lower():
            continue
        if partition and str(r.get('partition', '')).lower() != partition.lower():
            continue

        # score range
        if min_score is not None or max_score is not None:
            val = r.get(score_field)
            try:
                num = float(val) if val is not None and val != '' else None
            except Exception:
                num = None
            if num is None:
                # if no numeric score, skip when any numeric constraint exists
                if min_score is not None or max_score is not None:
                    continue
            else:
                if min_score is not None and num < min_score:
                    continue
                if max_score is not None and num > max_score:
                    continue

        # date filter
        if df or dt:
            meta = r.get('metadata') if isinstance(r.get('metadata'), dict) else None
            date_val = r.get('_date') or (meta and meta.get('created_at'))
            parsed = None
            if date_val:
                parsed = _parse_iso_date(str(date_val))
            # if no date, exclude when filters specified
            if parsed is None:
                continue
            if df and parsed < df:
                continue
            if dt and parsed > dt:
                continue

        # text query: search in a concatenation of main fields
        if q:
            hay = ' '.join([str(r.get('dataset_name', '')), str(r.get('config_name', '')), str(r.get('model_name', '')), str(r.get('id', ''))]).lower()
            if q not in hay:
                # also try stringifying values
                if q not in json.dumps(r).lower():
                    continue

        out.append(r)

    return out


@app.get("/api/predictions/search")
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
                       date_to: Optional[str] = None) -> Dict[str, Any]:
    """Server-side paginated, sortable and filterable predictions endpoint.

    - dataset: dataset name (subfolder) or path
    - page/page_size: pagination (page starts at 1)
    - sort_by: column name to sort on
    - sort_dir: 'asc' or 'desc'
    - model_name/config_name/partition: equality filters
    - min_score/max_score: numeric range filtering on given score_field
    - q: free-text search
    - date_from/date_to: ISO dates to filter by prediction creation date
    """
    # load predictions for dataset or all
    results = predictions_list(dataset) if dataset is not None else predictions_all()
    rows = results.get('predictions', [])

    # apply filters
    filtered = _apply_filters_to_rows(rows, model_name=model_name, config_name=config_name, partition=partition,
                                      min_score=min_score, max_score=max_score, score_field=score_field,
                                      text_query=q, date_from=date_from, date_to=date_to)

    total = len(filtered)

    # sort
    if sort_by:
        reverse = (sort_dir.lower() != 'asc')
        try:
            filtered.sort(key=lambda r: (r.get(sort_by) is None, r.get(sort_by)), reverse=reverse)
        except Exception:
            # fallback to string sort
            filtered.sort(key=lambda r: str(r.get(sort_by, '')), reverse=reverse)

    # paginate: support cursor-based pagination for large datasets
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

    # columns ordering
    cols = results.get('columns') or []
    if sort_by and sort_by not in cols:
        cols = [sort_by] + cols

    next_cursor = end if end < total else None
    prev_cursor = max(0, start - page_size) if start > 0 else None
    return {"predictions": page_rows, "columns": cols, "total": total, "page": page, "page_size": page_size, "cursor": start, "next_cursor": next_cursor, "prev_cursor": prev_cursor}


@app.get("/api/predictions/datasets")
def predictions_datasets() -> Dict[str, Any]:
    """List direct dataset subfolders inside the configured results folder.

    Returns dataset entries with name, path, predictions_file (if present) and predictions_count.
    """
    if CURRENT_WORKSPACE_PATH is None:
        return {"datasets": [], "results_path": None}

    results_path = _resolve_results_path()
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


@app.get("/api/predictions/list")
def predictions_list(dataset: Optional[str] = None) -> Dict[str, Any]:
    """Return predictions for a dataset (or all datasets when dataset is omitted).

    The 'dataset' argument may be either a dataset name (direct subfolder under results)
    or an absolute path to a dataset folder. The endpoint returns a 'columns' list
    and list of serialized 'predictions' rows where numpy arrays are converted to lists.
    """
    results_path = _resolve_results_path()
    if not results_path.exists() or not results_path.is_dir():
        return {"predictions": [], "columns": [], "count": 0, "results_path": str(results_path)}

    if Predictions is None:
        raise HTTPException(status_code=500, detail="Predictions backend is not available in this environment")

    try:
        if not dataset:
            preds = Predictions.load(path=str(results_path))
        else:
            # If dataset looks like an existing path, use it; otherwise treat as name
            ds_path = Path(dataset)
            if ds_path.exists() and ds_path.is_dir():
                preds_file = ds_path / 'predictions.json'
                if preds_file.exists():
                    preds = Predictions.load_from_file_cls(str(preds_file))
                else:
                    preds = Predictions()
            else:
                # treat as dataset name under results_path
                preds = Predictions.load(dataset_name=dataset, path=str(results_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load predictions: {e}") from e

    # Retrieve raw prediction rows
    try:
        rows = preds.filter_predictions()  # returns list of dicts with numpy arrays etc.
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query predictions: {e}") from e

    serialized = []
    columns_set = []
    # Try to find the dataset predictions file mtime to use as fallback for date
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
            ser[k] = _to_json_basic(v)
            if k not in columns_set:
                columns_set.append(k)
        # attach a best-effort date: attempt metadata.created_at or metadata.saved_at, else file mtime
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
        serialized.append(ser)

    # Prefer a sensible column ordering when possible
    preferred = [
        'id', 'dataset_name', 'dataset_path', 'config_name', 'config_path',
        'model_name', 'model_classname', 'metric', 'test_score', 'val_score', 'train_score',
        'fold_id', 'op_counter', 'n_samples', 'n_features'
    ]
    # produce ordered unique columns
    ordered = []
    for p in preferred:
        if p in columns_set and p not in ordered:
            ordered.append(p)
    for c in columns_set:
        if c not in ordered:
            ordered.append(c)

    return {"predictions": serialized, "columns": ordered, "count": len(serialized), "results_path": str(results_path)}


@app.get("/api/predictions/all")
def predictions_all() -> Dict[str, Any]:
    """Return all predictions across datasets as a list of lightweight records.

    Uses Predictions.load(path=results_path) and returns a flattened list where
    arrays are converted to basic Python types.
    """
    results_path = _resolve_results_path()
    if Predictions is None:
        raise HTTPException(status_code=500, detail="Predictions backend not available")
    try:
        preds = Predictions.load(path=str(results_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load predictions: {e}") from e

    rows = preds.filter_predictions()
    out = []
    for r in rows:
        rec = {k: _to_json_basic(v) for k, v in r.items()}
        out.append(rec)
    return {"predictions": out, "count": len(out), "results_path": str(results_path)}


@app.get("/api/predictions/meta")
def predictions_meta(dataset: Optional[str] = None) -> Dict[str, Any]:
    """Return unique models, configs, partitions and a total count for a dataset or for all datasets."""
    results_path = _resolve_results_path()
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


@app.get("/api/pipeline/from-prediction")
def pipeline_from_prediction(prediction_id: str) -> Dict[str, Any]:
    """Return the pipeline.json associated with a prediction ID.

    The prediction record must include a 'config_path' field which is interpreted
    relative to the results folder when not absolute.
    """
    if Predictions is None:
        raise HTTPException(status_code=500, detail="Predictions backend not available")

    results_path = _resolve_results_path()
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

    # Return pipeline and basic prediction metadata
    return {
        'pipeline': data,
        'prediction': {
            'id': rec.get('id'),
            'dataset_name': rec.get('dataset_name'),
            'model_name': rec.get('model_name'),
            'config_path': str(cfg_p),
        }
    }

# Workspace management endpoints ------------------------------------------------
@app.get("/api/workspace")
def get_workspace() -> Dict[str, Any]:
    """Return the currently selected workspace and its persisted config (if any)."""
    if CURRENT_WORKSPACE_PATH is None:
        return {"workspace": None, "config": None}

    return {
        "workspace": str(CURRENT_WORKSPACE_PATH),
        "config": CURRENT_WORKSPACE_CONFIG,
        "workspace_config_path": str(_workspace_config_path(CURRENT_WORKSPACE_PATH)),
    }


class SelectWorkspaceRequest(BaseModel):
    path: str
    create: bool = True
    persist_global: bool = True
    persist_env: bool = False


def _compute_default_pipeline_repo_for_workspace(p: Path) -> Optional[Path]:
    # pipeline repo expected to be at same level as workspace
    candidate = p.parent / "pipelines"
    if candidate.exists():
        return candidate
    return None


@app.post("/api/workspace/select")
def select_workspace(req: SelectWorkspaceRequest) -> Dict[str, Any]:
    """Select or create a workspace folder where pipeline outputs and a workspace
    config will be stored.

    Query/body params:
    - path: path to folder to use as workspace
    - create: whether to create it if it doesn't exist (default: true)
    - persist_global: whether to persist pointer to this workspace in the user's home directory (default: true)
    - persist_env: whether to attempt to persist NIRS4ALL_WORKSPACE in the user environment (Windows only; uses setx) (default: false)
    """
    global CURRENT_WORKSPACE_PATH, CURRENT_WORKSPACE_CONFIG
    path = req.path
    create = req.create
    persist_global = req.persist_global
    persist_env = req.persist_env
    try:
        p = Path(path).expanduser().resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {e}") from e

    if not p.exists():
        if create:
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Cannot create workspace path: {e}") from e
        else:
            raise HTTPException(status_code=404, detail="Workspace path does not exist")

    # Load or initialize workspace config
    cfg = load_workspace_config(p)
    # ensure fields exist
    cfg.setdefault("datasets", [])
    cfg.setdefault("pipeline_repo", None)
    cfg.setdefault("created", cfg.get("created") or datetime.utcnow().isoformat())
    # If the pipeline repo is not set, attempt a sane default at the same
    # level as the workspace (parent/pipelines).
    if cfg.get("pipeline_repo") is None:
        default = _compute_default_pipeline_repo_for_workspace(p)
        if default is not None:
            cfg["pipeline_repo"] = str(default)
    try:
        save_workspace_config(p, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot write workspace config: {e}") from e

    # Persist last-workspace pointer if requested
    if persist_global:
        try:
            save_last_workspace_pointer(p)
        except Exception as e:
            # do not fail the request for this, just warn
            return {"ok": False, "error": f"Failed to persist last-workspace pointer: {e}"}

    # Optionally persist environment variable (Windows setx)
    if persist_env and sys.platform.startswith("win"):
        try:
            # setx expects str arguments; this will persist for the current user
            subprocess.run(["setx", "NIRS4ALL_WORKSPACE", str(p)], check=True)
        except Exception as e:
            # warn but don't fail
            return {"ok": False, "warning": f"Workspace selected but failed to persist env var: {e}"}

    # Update process env and in-memory cache
    os.environ["NIRS4ALL_WORKSPACE"] = str(p)
    CURRENT_WORKSPACE_PATH = p
    CURRENT_WORKSPACE_CONFIG = cfg

    return {"ok": True, "workspace": str(p), "config": cfg}


class PathRequest(BaseModel):
    path: str


@app.post("/api/workspace/link-dataset")
def link_dataset(req: PathRequest) -> Dict[str, Any]:
    """Add a dataset path to the current workspace config (persisted in workspace)."""
    global CURRENT_WORKSPACE_PATH, CURRENT_WORKSPACE_CONFIG
    path = req.path
    if CURRENT_WORKSPACE_PATH is None:
        raise HTTPException(status_code=400, detail="No workspace selected")

    try:
        ds_path = Path(path).expanduser().resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid dataset path: {e}") from e

    if not ds_path.exists():
        raise HTTPException(status_code=404, detail="Dataset path does not exist")

    cfg = CURRENT_WORKSPACE_CONFIG or load_workspace_config(CURRENT_WORKSPACE_PATH)
    ds_list = cfg.setdefault("datasets", [])
    if str(ds_path) in ds_list:
        return {"ok": True, "message": "dataset already linked", "datasets": ds_list}

    ds_list.append(str(ds_path))
    try:
        save_workspace_config(CURRENT_WORKSPACE_PATH, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot save workspace config: {e}") from e

    CURRENT_WORKSPACE_CONFIG = cfg
    return {"ok": True, "datasets": ds_list}


@app.post("/api/workspace/unlink-dataset")
def unlink_dataset(req: PathRequest) -> Dict[str, Any]:
    """Remove a dataset path from the current workspace config."""
    path = req.path
    global CURRENT_WORKSPACE_PATH, CURRENT_WORKSPACE_CONFIG
    if CURRENT_WORKSPACE_PATH is None:
        raise HTTPException(status_code=400, detail="No workspace selected")

    cfg = CURRENT_WORKSPACE_CONFIG or load_workspace_config(CURRENT_WORKSPACE_PATH)
    ds_list = cfg.setdefault("datasets", [])
    try:
        ds_list.remove(str(Path(path).expanduser().resolve()))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found in workspace config") from exc
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        save_workspace_config(CURRENT_WORKSPACE_PATH, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot save workspace config: {e}") from e

    CURRENT_WORKSPACE_CONFIG = cfg
    return {"ok": True, "datasets": ds_list}


class SetPipelineRepoRequest(BaseModel):
    path: str
    allow_force: bool = False


@app.post("/api/workspace/set-pipeline-repo")
def set_pipeline_repo(req: SetPipelineRepoRequest) -> Dict[str, Any]:
    """Set a pipelines repository folder inside the current workspace configuration.

    If `allow_force` is False the endpoint validates that the chosen repository
    is located on the same parent directory level as the workspace. If it is not
    an error is returned explaining the mismatch; the UI may re-call the
    endpoint with `allow_force=true` to bypass the check.
    """
    path = req.path
    allow_force = req.allow_force
    global CURRENT_WORKSPACE_PATH, CURRENT_WORKSPACE_CONFIG
    if CURRENT_WORKSPACE_PATH is None:
        raise HTTPException(status_code=400, detail="No workspace selected")

    try:
        repo_path = Path(path).expanduser().resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid pipeline repo path: {e}") from e

    if not repo_path.exists():
        raise HTTPException(status_code=404, detail="Pipeline repo path does not exist")

    # Enforce sibling rule unless forced
    if not allow_force:
        if repo_path.parent != CURRENT_WORKSPACE_PATH.parent:
            raise HTTPException(status_code=400, detail=(
                "Pipeline repository must be located at the same parent directory as the workspace. "
                f"Workspace parent: {CURRENT_WORKSPACE_PATH.parent}, pipeline parent: {repo_path.parent}"
            ))

    cfg = CURRENT_WORKSPACE_CONFIG or load_workspace_config(CURRENT_WORKSPACE_PATH)
    cfg["pipeline_repo"] = str(repo_path)
    try:
        save_workspace_config(CURRENT_WORKSPACE_PATH, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot save workspace config: {e}") from e

    CURRENT_WORKSPACE_CONFIG = cfg
    return {"ok": True, "pipeline_repo": cfg.get("pipeline_repo")}


@app.get("/api/pipelines/list")
def pipelines_list() -> Dict[str, Any]:
    """List pipeline files (yaml/json) in the configured pipeline repository."""
    # read-only access to global workspace variables; no global declaration necessary
    if CURRENT_WORKSPACE_PATH is None:
        raise HTTPException(status_code=400, detail="No workspace selected")
    cfg = CURRENT_WORKSPACE_CONFIG or load_workspace_config(CURRENT_WORKSPACE_PATH)
    repo = cfg.get("pipeline_repo")
    if not repo:
        return {"repo": None, "files": []}
    repo_path = Path(repo)
    if not repo_path.exists() or not repo_path.is_dir():
        raise HTTPException(status_code=404, detail="Pipeline repo does not exist or is not a directory")
    files = []
    for entry in sorted(repo_path.iterdir()):
        if entry.is_file() and entry.suffix.lower() in {".yml", ".yaml", ".json"}:
            try:
                st = entry.stat()
            except Exception:
                st = None
            files.append({
                "name": entry.name,
                "path": str(entry),
                "size": st.st_size if st else None,
                "mtime": st.st_mtime if st else None,
            })
    return {"repo": str(repo_path), "files": files}


@app.post("/api/workspace/set-results-folder")
def set_results_folder(req: PathRequest) -> Dict[str, Any]:
    """Set the results folder used to store run outputs and predictions."""
    global CURRENT_WORKSPACE_PATH, CURRENT_WORKSPACE_CONFIG
    if CURRENT_WORKSPACE_PATH is None:
        raise HTTPException(status_code=400, detail="No workspace selected")
    path = req.path
    try:
        res_path = Path(path).expanduser().resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid results path: {e}") from e
    if not res_path.exists():
        try:
            res_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cannot create results path: {e}") from e

    cfg = CURRENT_WORKSPACE_CONFIG or load_workspace_config(CURRENT_WORKSPACE_PATH)
    cfg['results_folder'] = str(res_path)
    try:
        save_workspace_config(CURRENT_WORKSPACE_PATH, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot save workspace config: {e}") from e

    CURRENT_WORKSPACE_CONFIG = cfg
    return {"ok": True, "results_folder": cfg.get('results_folder')}


class SaveDatasetRequest(BaseModel):
    name: Optional[str]
    config: Dict[str, Any]
    path: Optional[str] = None


@app.post("/api/workspace/datasets/save")
def save_dataset_config(req: SaveDatasetRequest) -> Dict[str, Any]:
    """Save a dataset configuration JSON inside the workspace datasets folder.

    The frontend should post a JSON object with optional 'name' and mandatory
    'config' (a dict compatible with nirs4all dataset loader). The server will
    create workspace/datasets/<name>.json and add it to the workspace config
    list so it appears in the UI.
    """
    global CURRENT_WORKSPACE_PATH, CURRENT_WORKSPACE_CONFIG
    if CURRENT_WORKSPACE_PATH is None:
        raise HTTPException(status_code=400, detail="No workspace selected")

    cfg = CURRENT_WORKSPACE_CONFIG or load_workspace_config(CURRENT_WORKSPACE_PATH)
    datasets_dir = CURRENT_WORKSPACE_PATH / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Determine a name for the config file
    name = req.name
    if not name:
        # Try to infer a name from config: folder or first train_x/test_x
        folder = req.config.get('folder')
        if folder:
            name = folder_to_name(folder)
        else:
            first = None
            for k in ('train_x', 'test_x'):
                v = req.config.get(k)
                if v:
                    if isinstance(v, list):
                        first = v[0]
                    else:
                        first = v
                    break
            if first:
                try:
                    name = folder_to_name(str(Path(first).parent))
                except Exception:
                    name = 'dataset'
            else:
                name = 'dataset'

    # If a path was provided, try to update existing file (must be under workspace datasets dir)
    provided = req.path
    if provided:
        try:
            provided_p = Path(provided).expanduser().resolve()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid provided path: {e}") from e
        # Ensure provided path is inside datasets_dir
        try:
            provided_p.relative_to(datasets_dir)
        except Exception:
            raise HTTPException(status_code=400, detail="Provided path is not inside workspace datasets folder")
        dest = provided_p
        try:
            _save_json_atomic(dest, req.config)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write dataset config: {e}") from e
        ds_list = cfg.setdefault('datasets', [])
        if str(dest) not in ds_list:
            ds_list.append(str(dest))
    else:
        safe = folder_to_name(name)
        dest = datasets_dir / f"{safe}.json"
        i = 1
        while dest.exists():
            dest = datasets_dir / f"{safe}-{i}.json"
            i += 1
        try:
            _save_json_atomic(dest, req.config)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write dataset config: {e}") from e
        # Ensure config path is saved in workspace config
        ds_list = cfg.setdefault('datasets', [])
        if str(dest) not in ds_list:
            ds_list.append(str(dest))
    try:
        save_workspace_config(CURRENT_WORKSPACE_PATH, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist workspace config: {e}") from e

    CURRENT_WORKSPACE_CONFIG = cfg
    return {"ok": True, "path": str(dest)}


@app.get("/api/workspace/datasets/config")
def get_dataset_config(path: str) -> Dict[str, Any]:
    """Return the contents of a saved dataset config file."""
    try:
        p = Path(path).expanduser().resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {e}") from e
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Config file not found")
    try:
        with p.open('r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot read config file: {e}") from e


@app.delete("/api/workspace/datasets")
def delete_dataset_config(path: str) -> Dict[str, Any]:
    """Delete a saved dataset config file and remove it from the workspace config."""
    global CURRENT_WORKSPACE_PATH, CURRENT_WORKSPACE_CONFIG
    if CURRENT_WORKSPACE_PATH is None:
        raise HTTPException(status_code=400, detail="No workspace selected")
    try:
        p = Path(path).expanduser().resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {e}") from e
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Config file not found")
    try:
        p.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}") from e

    cfg = CURRENT_WORKSPACE_CONFIG or load_workspace_config(CURRENT_WORKSPACE_PATH)
    ds_list = cfg.setdefault('datasets', [])
    try:
        ds_list.remove(str(p))
    except ValueError:
        pass
    try:
        save_workspace_config(CURRENT_WORKSPACE_PATH, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist workspace config: {e}") from e

    CURRENT_WORKSPACE_CONFIG = cfg
    return {"ok": True}


# --- File browsing API (server-side file picker used by the UI) ----------
@app.get("/api/files/roots")
def files_roots() -> Dict[str, List[str]]:
    """Return filesystem roots (drives on Windows, '/' on POSIX)."""
    roots: List[str] = []
    if sys.platform.startswith("win"):
        from string import ascii_uppercase
        for letter in ascii_uppercase:
            candidate = Path(f"{letter}:/")
            if candidate.exists():
                roots.append(str(candidate))
    else:
        roots.append("/")
    return {"roots": roots}


def _safe_list_dir(p: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        for entry in sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
            try:
                st = entry.stat()
            except Exception:
                st = None
            items.append({
                "name": entry.name,
                "path": str(entry),
                "is_dir": entry.is_dir(),
                "size": st.st_size if st and not entry.is_dir() else None,
                "mtime": st.st_mtime if st else None,
            })
    except PermissionError:
        # return empty list when not allowed
        pass
    return items


@app.get("/api/files/list")
def files_list(path: str) -> Dict[str, Any]:
    """List files/directories for a given path.

    Query param:
    - path: absolute path to list
    """
    try:
        p = Path(path).expanduser().resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {e}") from e
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=404, detail="Path does not exist or is not a directory")
    return {"path": str(p), "items": _safe_list_dir(p)}


@app.get("/api/files/parent")
def files_parent(path: str) -> Dict[str, Any]:
    """Return the parent directory for a given path. Useful for file browser up action."""
    try:
        p = Path(path).expanduser().resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {e}") from e
    parent = p.parent
    # On Windows, parent of a root like C:\ is itself; ensure we return something sensible
    return {"parent": str(parent)}


# --- Dataset metadata (datatable) ---------------------------------------
def _human_readable_size(n: int | float) -> str:
    # simple human-readable formatting
    value = float(n)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if value < 1024:
            return f"{value:.1f} {unit}" if unit != 'B' else f"{int(value)} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def _compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.sha256()
    with file_path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def _compute_dir_hash(path: Path, max_files: Optional[int] = None) -> str:
    h = hashlib.sha256()
    count = 0
    for root, _dirs, files in os.walk(path):
        for fname in sorted(files):
            if max_files is not None and count >= max_files:
                break
            fpath = Path(root) / fname
            try:
                st = fpath.stat()
                h.update(str(Path(root).relative_to(path)).encode('utf-8'))
                h.update(fname.encode('utf-8'))
                h.update(str(st.st_mtime).encode('utf-8'))
                h.update(str(st.st_size).encode('utf-8'))
            except Exception:
                continue
            count += 1
        if max_files is not None and count >= max_files:
            break
    return h.hexdigest()


@app.get("/api/workspace/datasets")
def workspace_datasets(predictions_path: str = "results") -> Dict[str, Any]:
    """Return metadata for linked datasets: name, path, size, hash, predictions_count."""
    # read-only access only
    if CURRENT_WORKSPACE_PATH is None:
        return {"datasets": []}

    cfg = CURRENT_WORKSPACE_CONFIG or load_workspace_config(CURRENT_WORKSPACE_PATH)
    ds_paths = cfg.get("datasets", [])

    # load predictions counts once
    preds_counts: Dict[str, int] = {}
    if Predictions is not None:
        try:
            preds = Predictions.load(path=predictions_path)
            for ds in preds.get_datasets():
                preds_counts[ds] = len(preds.filter_predictions(dataset_name=ds))
        except Exception:
            preds_counts = {}

    results = []
    for ds in ds_paths:
        try:
            p = Path(ds).expanduser().resolve()
        except Exception:
            continue
        if not p.exists():
            continue
        # compute size & hash
        if p.is_file() and p.suffix.lower() == '.json':
            # This is a saved dataset config file. Load config and attempt to
            # compute summary stats from referenced data paths.
            try:
                with p.open('r', encoding='utf-8') as fh:
                    cfg_json = json.load(fh)
            except Exception:
                cfg_json = None

            if cfg_json:
                # Attempt to compute total size from train_x/test_x entries
                size = 0
                referenced_paths = []
                for key in ('train_x', 'test_x'):
                    val = cfg_json.get(key)
                    if val:
                        if isinstance(val, list):
                            referenced_paths.extend(val)
                        else:
                            referenced_paths.append(val)

                for ref in referenced_paths:
                    try:
                        refp = Path(ref).expanduser().resolve()
                        if refp.exists():
                            if refp.is_file():
                                size += refp.stat().st_size
                            else:
                                for _root, _dirs, files in os.walk(refp):
                                    for fname in files:
                                        try:
                                            size += (Path(_root) / fname).stat().st_size
                                        except Exception:
                                            continue
                    except Exception:
                        continue

                # Use config file hash (fast) as identifier for the dataset config
                try:
                    h = _compute_file_hash(p) if p.stat().st_size < 200 * 1024 * 1024 else None
                except Exception:
                    h = None
            else:
                # fallback: config unreadable
                try:
                    size = p.stat().st_size
                except Exception:
                    size = 0
                h = _compute_file_hash(p) if size < 200 * 1024 * 1024 else None
        else:
            if p.is_file():
                try:
                    size = p.stat().st_size
                except Exception:
                    size = 0
                h = _compute_file_hash(p) if size < 200 * 1024 * 1024 else None
            else:
                # directory
                size = 0
                for _root, _dirs, files in os.walk(p):
                    for fname in files:
                        try:
                            size += (Path(_root) / fname).stat().st_size
                        except Exception:
                            continue
                # compute a directory hash based on filenames and mtimes (fast, non-content)
                h = _compute_dir_hash(p, max_files=2000)

        # attempt to match predictions by config-derived name, base name or full path
        name = p.name
        # if JSON config, try to extract a more meaningful dataset name
        if p.is_file() and p.suffix.lower() == '.json' and 'cfg_json' in locals() and cfg_json:
            # Prefer explicit name in config if present
            conf_name = cfg_json.get('name') or cfg_json.get('dataset_name')
            if conf_name:
                name = conf_name
            else:
                # try to derive from first referenced path
                if referenced_paths:
                    try:
                        name = folder_to_name(str(Path(referenced_paths[0]).parent))
                    except Exception:
                        name = p.stem

        pred_count = preds_counts.get(name) or preds_counts.get(p.stem) or preds_counts.get(str(p)) or 0

        results.append({
            "name": name,
            "path": str(p),
            "is_dir": p.is_dir(),
            "size_bytes": size,
            "size_human": _human_readable_size(size),
            "hash": h,
            "predictions_count": pred_count,
        })

    return {"datasets": results}
