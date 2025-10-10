from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import sys
import json
import os
from datetime import datetime

from nirs4all.ui import utils
from nirs4all.dataset.dataset_config import DatasetConfigs
from fastapi.concurrency import run_in_threadpool

try:
    from nirs4all.dataset.dataset_config_parser import folder_to_name
except Exception:  # pragma: no cover - fall back to simple namer
    def folder_to_name(folder_path: str) -> str:  # type: ignore[override]
        p = Path(folder_path)
        return p.name or 'dataset'


router = APIRouter()


@router.get("/api/workspace")
def get_workspace() -> Dict[str, Any]:
    current = utils.get_current_workspace()
    if current is None:
        return {"workspace": None, "config": None}
    return {
        "workspace": str(current),
        "config": utils.get_current_workspace_config(),
        "workspace_config_path": str(utils.workspace_config_path(current)),
    }


class SelectWorkspaceRequest(BaseModel):
    path: str
    create: bool = True
    persist_global: bool = True
    persist_env: bool = False


def _compute_default_pipeline_repo_for_workspace(p: Path) -> Optional[Path]:
    candidate = p.parent / "pipelines"
    if candidate.exists():
        return candidate
    return None


@router.post("/api/workspace/select")
def select_workspace(req: SelectWorkspaceRequest) -> Dict[str, Any]:
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

    cfg = utils.load_workspace_config(p)
    cfg.setdefault("datasets", [])
    cfg.setdefault("pipeline_repo", None)
    cfg.setdefault("created", cfg.get("created") or datetime.utcnow().isoformat())
    if cfg.get("pipeline_repo") is None:
        default = _compute_default_pipeline_repo_for_workspace(p)
        if default is not None:
            cfg["pipeline_repo"] = str(default)
    try:
        utils.save_workspace_config(p, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot write workspace config: {e}") from e

    if persist_global:
        try:
            utils.save_last_workspace_pointer(p)
        except Exception as e:
            return {"ok": False, "error": f"Failed to persist last-workspace pointer: {e}"}

    if persist_env and sys.platform.startswith("win"):
        try:
            subprocess.run(["setx", "NIRS4ALL_WORKSPACE", str(p)], check=True)
        except Exception as e:
            return {"ok": False, "warning": f"Workspace selected but failed to persist env var: {e}"}

    utils.set_current_workspace(p, cfg, persist_global=persist_global)
    return {"ok": True, "workspace": str(p), "config": cfg}


class PathRequest(BaseModel):
    path: str


@router.post("/api/workspace/link-dataset")
def link_dataset(req: PathRequest) -> Dict[str, Any]:
    path = req.path
    current = utils.get_current_workspace()
    if current is None:
        raise HTTPException(status_code=400, detail="No workspace selected")

    try:
        ds_path = Path(path).expanduser().resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid dataset path: {e}") from e

    if not ds_path.exists():
        raise HTTPException(status_code=404, detail="Dataset path does not exist")

    cfg = utils.get_current_workspace_config() or (utils.load_workspace_config(current) if current is not None else {})
    ds_list = cfg.setdefault("datasets", [])
    if str(ds_path) in ds_list:
        return {"ok": True, "message": "dataset already linked", "datasets": ds_list}

    ds_list.append(str(ds_path))
    try:
        utils.save_workspace_config(current, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot save workspace config: {e}") from e

    utils.set_current_workspace(current, cfg)
    return {"ok": True, "datasets": ds_list}


@router.post("/api/workspace/unlink-dataset")
def unlink_dataset(req: PathRequest) -> Dict[str, Any]:
    path = req.path
    current = utils.get_current_workspace()
    if current is None:
        raise HTTPException(status_code=400, detail="No workspace selected")

    cfg = utils.get_current_workspace_config() or (utils.load_workspace_config(current) if current is not None else {})
    ds_list = cfg.setdefault("datasets", [])
    try:
        ds_list.remove(str(Path(path).expanduser().resolve()))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found in workspace config") from exc
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        utils.save_workspace_config(current, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot save workspace config: {e}") from e

    utils.set_current_workspace(current, cfg)
    return {"ok": True, "datasets": ds_list}


class SetPipelineRepoRequest(BaseModel):
    path: str
    allow_force: bool = False


@router.post("/api/workspace/set-pipeline-repo")
def set_pipeline_repo(req: SetPipelineRepoRequest) -> Dict[str, Any]:
    path = req.path
    allow_force = req.allow_force
    current = utils.get_current_workspace()
    if current is None:
        raise HTTPException(status_code=400, detail="No workspace selected")

    try:
        repo_path = Path(path).expanduser().resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid pipeline repo path: {e}") from e

    if not repo_path.exists():
        raise HTTPException(status_code=404, detail="Pipeline repo path does not exist")

    if not allow_force:
        if repo_path.parent != current.parent:
            raise HTTPException(status_code=400, detail=(
                "Pipeline repository must be located at the same parent directory as the workspace. "
                f"Workspace parent: {current.parent}, pipeline parent: {repo_path.parent}"
            ))

    cfg = utils.get_current_workspace_config() or utils.load_workspace_config(current)
    cfg["pipeline_repo"] = str(repo_path)
    try:
        utils.save_workspace_config(current, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot save workspace config: {e}") from e

    utils.set_current_workspace(current, cfg)
    return {"ok": True, "pipeline_repo": cfg.get("pipeline_repo")}


@router.get("/api/pipelines/list")
def pipelines_list() -> Dict[str, Any]:
    if utils.get_current_workspace() is None:
        raise HTTPException(status_code=400, detail="No workspace selected")
    current_ws = utils.get_current_workspace()
    cfg = utils.get_current_workspace_config() or (utils.load_workspace_config(current_ws) if current_ws is not None else {})
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


@router.post("/api/workspace/set-results-folder")
def set_results_folder(req: PathRequest) -> Dict[str, Any]:
    current = utils.get_current_workspace()
    if current is None:
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

    cfg = utils.get_current_workspace_config() or utils.load_workspace_config(current)
    cfg['results_folder'] = str(res_path)
    try:
        utils.save_workspace_config(current, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot save workspace config: {e}") from e

    utils.set_current_workspace(current, cfg)
    return {"ok": True, "results_folder": cfg.get('results_folder')}


class SaveDatasetRequest(BaseModel):
    name: Optional[str]
    config: Dict[str, Any]
    path: Optional[str] = None


@router.post("/api/workspace/datasets/save")
def save_dataset_config(req: SaveDatasetRequest) -> Dict[str, Any]:
    current = utils.get_current_workspace()
    if current is None:
        raise HTTPException(status_code=400, detail="No workspace selected")

    cfg = utils.get_current_workspace_config() or utils.load_workspace_config(current)
    datasets_dir = current / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    name = req.name
    if not name:
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

    provided = req.path
    if provided:
        try:
            provided_p = Path(provided).expanduser().resolve()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid provided path: {e}") from e
        try:
            provided_p.relative_to(datasets_dir)
        except Exception:
            raise HTTPException(status_code=400, detail="Provided path is not inside workspace datasets folder")
        dest = provided_p
        try:
            utils.save_json_atomic(dest, req.config)
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
            utils.save_json_atomic(dest, req.config)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write dataset config: {e}") from e
        ds_list = cfg.setdefault('datasets', [])
        if str(dest) not in ds_list:
            ds_list.append(str(dest))
    try:
        utils.save_workspace_config(current, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist workspace config: {e}") from e

    utils.set_current_workspace(current, cfg)
    return {"ok": True, "path": str(dest)}


@router.get("/api/workspace/datasets/config")
def get_dataset_config(path: str) -> Dict[str, Any]:
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


@router.delete("/api/workspace/datasets")
def delete_dataset_config(path: str) -> Dict[str, Any]:
    current = utils.get_current_workspace()
    if current is None:
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

    cfg = utils.get_current_workspace_config() or utils.load_workspace_config(current)
    ds_list = cfg.setdefault('datasets', [])
    try:
        ds_list.remove(str(p))
    except ValueError:
        pass
    try:
        utils.save_workspace_config(current, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist workspace config: {e}") from e

    utils.set_current_workspace(current, cfg)
    return {"ok": True}


@router.get("/api/workspace/datasets")
def workspace_datasets(predictions_path: str = "results") -> Dict[str, Any]:
    if utils.get_current_workspace() is None:
        return {"datasets": []}

    current_ws = utils.get_current_workspace()
    cfg = utils.get_current_workspace_config() or (utils.load_workspace_config(current_ws) if current_ws is not None else {})
    ds_paths = cfg.get("datasets", [])

    preds_counts: Dict[str, int] = {}
    try:
        from nirs4all.dataset.predictions import Predictions
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
        if p.is_file() and p.suffix.lower() == '.json':
            try:
                with p.open('r', encoding='utf-8') as fh:
                    cfg_json = json.load(fh)
            except Exception:
                cfg_json = None

            if cfg_json:
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

                try:
                    h = utils.compute_file_hash(p) if p.stat().st_size < 200 * 1024 * 1024 else None
                except Exception:
                    h = None
            else:
                try:
                    size = p.stat().st_size
                except Exception:
                    size = 0
                h = utils.compute_file_hash(p) if size < 200 * 1024 * 1024 else None
        else:
            if p.is_file():
                try:
                    size = p.stat().st_size
                except Exception:
                    size = 0
                h = utils.compute_file_hash(p) if size < 200 * 1024 * 1024 else None
            else:
                size = 0
                for _root, _dirs, files in os.walk(p):
                    for fname in files:
                        try:
                            size += (Path(_root) / fname).stat().st_size
                        except Exception:
                            continue
                h = utils.compute_dir_hash(p, max_files=2000)

        name = p.name
        if p.is_file() and p.suffix.lower() == '.json' and 'cfg_json' in locals() and cfg_json:
            conf_name = cfg_json.get('name') or cfg_json.get('dataset_name')
            if conf_name:
                name = conf_name
            else:
                if referenced_paths:
                    try:
                        name = folder_to_name(str(Path(referenced_paths[0]).parent))
                    except Exception:
                        name = p.stem

        pred_count = preds_counts.get(name) or preds_counts.get(p.stem) or preds_counts.get(str(p)) or 0

        results.append(
            {
                "name": name,
                "path": str(p),
                "is_dir": p.is_dir(),
                "size_bytes": size,
                "size_human": utils.human_readable_size(size),
                "hash": h,
                "predictions_count": pred_count,
            }
        )

    return {"datasets": results}


@router.get('/api/workspace/dataset/props')
async def dataset_props(path: str) -> Dict[str, Any]:
    """Compute basic dataset properties (n_samples, num_features, n_sources) for a dataset.

    The path can be either a dataset config JSON file inside the workspace datasets folder
    or a folder path. The computation is performed in a thread to avoid blocking the
    FastAPI event loop.
    """
    current = utils.get_current_workspace()
    if current is None:
        raise HTTPException(status_code=400, detail="No workspace selected")

    try:
        p = Path(path).expanduser().resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {e}") from e

    def _compute_props():
        # If it's a JSON config file, load it and pass the dict; otherwise pass the path
        if p.exists() and p.is_file() and p.suffix.lower() == '.json':
            try:
                with p.open('r', encoding='utf-8') as fh:
                    cfg = json.load(fh)
            except Exception as e:
                raise RuntimeError(f"Failed to read config file: {e}")
            dc = DatasetConfigs(cfg)
        else:
            # pass folder path or generic string to DatasetConfigs
            dc = DatasetConfigs(str(p))

        datasets = dc.get_datasets()
        if not datasets:
            return {"n_samples": 0, "num_features": 0, "n_sources": 0}
        ds = datasets[0]
        nf = ds.num_features
        return {
            "n_samples": int(ds.num_samples or 0),
            "num_features": nf if isinstance(nf, (int, list)) else int(nf or 0),
            "n_sources": int(ds.n_sources or 0),
            "task_type": ds.task_type,
        }

    try:
        props = await run_in_threadpool(_compute_props)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute dataset properties: {e}") from e

    return {"ok": True, "props": props}
