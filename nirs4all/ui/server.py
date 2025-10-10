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

try:
    # Import predictions manager from nirs4all core
    from nirs4all.dataset.predictions import Predictions
except Exception:  # pragma: no cover - best-effort import
    Predictions = None


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
        # compute size
        if p.is_file():
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            h = _compute_file_hash(p) if size < 200 * 1024 * 1024 else None
        else:
            # directory
            size = 0
            for root, _dirs, files in os.walk(p):
                for fname in files:
                    try:
                        size += (Path(root) / fname).stat().st_size
                    except Exception:
                        continue
            # compute a directory hash based on filenames and mtimes (fast, non-content)
            h = _compute_dir_hash(p, max_files=2000)

        # attempt to match predictions by base name or full path
        name = p.name
        pred_count = preds_counts.get(name) or preds_counts.get(str(p)) or 0

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
