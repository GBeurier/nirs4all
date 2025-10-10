from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import os
from datetime import datetime
import hashlib
import sys
import subprocess
import numpy as np

# Workspace persistence configuration
GLOBAL_CONFIG_DIR = Path.home() / ".nirs4all"
GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LAST_WORKSPACE_FILE = GLOBAL_CONFIG_DIR / "last_workspace.json"
WORKSPACE_CONFIG_FILENAME = ".nirs4all_workspace.json"

# In-memory cache for the currently selected workspace and its config
CURRENT_WORKSPACE_PATH: Optional[Path] = None
CURRENT_WORKSPACE_CONFIG: Optional[Dict[str, Any]] = None

# Fields that often contain large arrays and should be omitted from list/search responses
HEAVY_ARRAY_FIELDS = [
    "y_true",
    "y_pred",
    "y_proba",
    "sample_indices",
    "sample_indices_unrolled",
    "sample_weights",
]


def save_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def workspace_config_path(workspace_path: Path) -> Path:
    return workspace_path / WORKSPACE_CONFIG_FILENAME


def load_workspace_config(workspace_path: Path) -> Dict[str, Any]:
    cfg_path = workspace_config_path(workspace_path)
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
    cfg_path = workspace_config_path(workspace_path)
    save_json_atomic(cfg_path, config)


def save_last_workspace_pointer(path: Path) -> None:
    LAST_WORKSPACE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"path": str(path), "updated": datetime.utcnow().isoformat()}
    save_json_atomic(LAST_WORKSPACE_FILE, data)


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


def ensure_workspace_loaded_on_startup() -> None:
    """Populate CURRENT_WORKSPACE_* from environment or last pointer if available."""
    global CURRENT_WORKSPACE_PATH, CURRENT_WORKSPACE_CONFIG
    p = load_last_workspace_pointer()
    if p and p.exists():
        CURRENT_WORKSPACE_PATH = p
        CURRENT_WORKSPACE_CONFIG = load_workspace_config(p)
        os.environ["NIRS4ALL_WORKSPACE"] = str(p)


def get_current_workspace() -> Optional[Path]:
    return CURRENT_WORKSPACE_PATH


def get_current_workspace_config() -> Optional[Dict[str, Any]]:
    return CURRENT_WORKSPACE_CONFIG


def set_current_workspace(p: Path, cfg: Dict[str, Any], persist_global: bool = True) -> None:
    global CURRENT_WORKSPACE_PATH, CURRENT_WORKSPACE_CONFIG
    CURRENT_WORKSPACE_PATH = p
    CURRENT_WORKSPACE_CONFIG = cfg
    os.environ["NIRS4ALL_WORKSPACE"] = str(p)
    if persist_global:
        save_last_workspace_pointer(p)


def resolve_results_path() -> Path:
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


def to_json_basic(obj: Any):
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
        return {str(k): to_json_basic(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_basic(v) for v in obj]
    # default: let json serializer handle or convert to string as last resort
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def strip_heavy_arrays(record: Dict[str, Any], include_arrays: bool = False, fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return a copy of record with heavy array fields removed unless include_arrays is True.

    Adds a marker '_arrays_stripped': True when fields were removed.
    """
    if include_arrays:
        return record
    if fields is None:
        fields = HEAVY_ARRAY_FIELDS
    out = {k: v for k, v in record.items() if k not in fields}
    if any(k in record for k in fields):
        out["_arrays_stripped"] = True
    return out


def parse_iso_date(v: Optional[str]) -> Optional[datetime]:
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


def apply_filters_to_rows(
    rows: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    config_name: Optional[str] = None,
    partition: Optional[str] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    score_field: str = 'test_score',
    text_query: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter a list of prediction rows (dictionaries) in memory.

    Supports equality filters, numeric ranges and a simple substring text search.
    Date filters operate on '_date' field when present or on metadata.created_at.
    """
    out = []
    q = text_query.lower() if text_query else None
    df = parse_iso_date(date_from) if date_from else None
    dt = parse_iso_date(date_to) if date_to else None

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
                parsed = parse_iso_date(str(date_val))
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


def safe_list_dir(p: Path) -> List[Dict[str, Any]]:
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


def human_readable_size(n: int | float) -> str:
    # simple human-readable formatting
    value = float(n)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if value < 1024:
            return f"{value:.1f} {unit}" if unit != 'B' else f"{int(value)} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.sha256()
    with file_path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def compute_dir_hash(path: Path, max_files: Optional[int] = None) -> str:
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


# Populate workspace state on import when possible
ensure_workspace_loaded_on_startup()
