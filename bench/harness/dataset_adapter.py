"""bench/harness/dataset_adapter.py.

Canonical dataset loader for the benchmark harness. Implements the
contract proposed under D-C-011 (DECISION_PENDING_CODEX_REVIEW).

Convention (provisional):

    bench/_datasets/<collection>/<dataset_name>/
        Xtrain.csv    semicolon-separated, header row of wavelengths
        Xtest.csv     same shape
        Ytrain.csv    one column (or `;`-separated when multi-target);
                      header row gives the target name(s)
        Ytest.csv     same
        Mtrain.csv    optional metadata, semicolon-separated; columns
                      include sample IDs, group keys, etc.
        Mtest.csv     same

`<collection>` is currently one of `hiba`, `redox` (see
`bench/_datasets/`). Future collections can be added without changing
this file as long as the per-dataset directory structure holds.

Public API:

    DatasetBundle              — frozen dataclass with arrays + meta.
    DatasetNotFoundError       — raised when no candidate path matches.
    load_dataset(name, *, root=None, cache=True) -> DatasetBundle.
    discover_dataset(name, *, root=None) -> Path | None.

The harness `ModelDispatcher.dispatch` will call `load_dataset(name)`
and surface failures as `ResultRow.status="failed"` with a clear
`error_message`. Production fit / predict NEVER touches the file
system directly.

DECISION_PENDING_CODEX_REVIEW (D-C-011).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("numpy is required by the harness dataset adapter") from exc

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pandas is required by the harness dataset adapter") from exc


BENCH = Path(__file__).resolve().parents[1]
# Search roots in priority order. First hit wins. Each root is scanned at
# depth 0 (direct match), depth 1 (`<root>/<collection>/<name>/`), and
# depth 2 (`<root>/<collection>/<group>/<name>/`) — covers both the
# `bench/_datasets/{hiba,redox}/<name>/` convention and the canonical
# `bench/tabpfn_paper/data/regression/<GROUP>/<dataset>/` layout that
# Agent A flagged in D-A-Q5.
DEFAULT_ROOTS: tuple[Path, ...] = (
    BENCH / "_datasets",
    BENCH / "tabpfn_paper" / "data" / "regression",
    BENCH / "tabpfn_paper" / "data" / "classification",
)
DEFAULT_ROOT = DEFAULT_ROOTS[0]  # legacy alias; kept for backward callers.
MAX_SEARCH_DEPTH = 2
REQUIRED_FILES = ("Xtrain.csv", "Ytrain.csv", "Xtest.csv", "Ytest.csv")
OPTIONAL_FILES = ("Mtrain.csv", "Mtest.csv")
DELIMITER = ";"


class DatasetNotFoundError(LookupError):
    """Raised when no `bench/_datasets/<*>/<name>/` directory has the
    minimal {Xtrain, Ytrain, Xtest, Ytest}.csv files.
    """


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    path: Path
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta_train: pd.DataFrame | None
    meta_test: pd.DataFrame | None
    n_train: int
    n_test: int
    n_features: int


_LOAD_LOCK = threading.Lock()
_CACHE: dict[tuple[str, str], DatasetBundle] = {}


def discover_dataset(name: str, *, root: Path | None = None) -> Path | None:
    """Return the directory that contains `name` under one of the search
    roots, or None.

    Search semantics:
      * if `root` is given, only that root is scanned;
      * otherwise the module-level `DEFAULT_ROOTS` tuple is scanned in
        priority order;
      * each root is scanned at depth 0 (`<root>/<name>/`), depth 1
        (`<root>/<collection>/<name>/`), and depth 2
        (`<root>/<collection>/<group>/<name>/`) — first hit wins.
    """
    bases: list[Path] = [root] if root is not None else list(DEFAULT_ROOTS)
    for base in bases:
        if not base.is_dir():
            continue
        # Depth 0
        candidate = base / name
        if _has_required_files(candidate):
            return candidate
        # Depth 1: <root>/*/<name>/
        for collection in sorted(base.iterdir()):
            if not collection.is_dir():
                continue
            level1 = collection / name
            if _has_required_files(level1):
                return level1
            # Depth 2: <root>/*/*/<name>/
            for group in sorted(collection.iterdir()):
                if not group.is_dir():
                    continue
                level2 = group / name
                if _has_required_files(level2):
                    return level2
    return None


def _has_required_files(path: Path) -> bool:
    return path.is_dir() and all((path / f).is_file() for f in REQUIRED_FILES)


def load_dataset(name: str, *, root: Path | None = None, cache: bool = True) -> DatasetBundle:
    cache_key = (str(root.resolve()) if root is not None else "<defaults>", name)
    if cache:
        existing = _CACHE.get(cache_key)
        if existing is not None:
            return existing
    path = discover_dataset(name, root=root)
    if path is None:
        roots_text = (
            str(root)
            if root is not None
            else " | ".join(str(r) for r in DEFAULT_ROOTS)
        )
        raise DatasetNotFoundError(
            f"dataset {name!r} not found under {roots_text}; "
            f"expected `<root>/<name>/`, `<root>/<collection>/<name>/`, or "
            f"`<root>/<collection>/<group>/<name>/` with files "
            f"{REQUIRED_FILES + OPTIONAL_FILES}"
        )
    bundle = _read_bundle(name, path)
    if cache:
        with _LOAD_LOCK:
            _CACHE[cache_key] = bundle
    return bundle


def _read_bundle(name: str, path: Path) -> DatasetBundle:
    X_train = _read_array(path / "Xtrain.csv")
    X_test = _read_array(path / "Xtest.csv")
    y_train = _read_array(path / "Ytrain.csv", squeeze=True)
    y_test = _read_array(path / "Ytest.csv", squeeze=True)

    if X_train.ndim != 2:
        raise ValueError(f"{path}/Xtrain.csv must be 2D, got shape {X_train.shape}")
    if X_test.ndim != 2:
        raise ValueError(f"{path}/Xtest.csv must be 2D, got shape {X_test.shape}")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"{path}: feature counts differ between train and test "
            f"({X_train.shape[1]} vs {X_test.shape[1]})"
        )
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"{path}: X_train / y_train length mismatch")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"{path}: X_test / y_test length mismatch")

    meta_train = _read_meta(path / "Mtrain.csv")
    meta_test = _read_meta(path / "Mtest.csv")

    return DatasetBundle(
        name=name,
        path=path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta_train=meta_train,
        meta_test=meta_test,
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        n_features=int(X_train.shape[1]),
    )


def _read_array(path: Path, *, squeeze: bool = False) -> np.ndarray:
    df = pd.read_csv(path, sep=DELIMITER)
    arr: np.ndarray = df.to_numpy(dtype=float, copy=False)
    if squeeze and arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    return arr


def _read_meta(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_csv(path, sep=DELIMITER)


def list_available_datasets(*, root: Path | None = None) -> list[str]:
    """Return the sorted list of dataset names with the 4 required files
    discoverable under one of the search roots. Useful for diagnostic CLI
    tools. Mirrors `discover_dataset`'s depth-0/1/2 walk across either
    the given `root` or `DEFAULT_ROOTS`.
    """
    bases: list[Path] = [root] if root is not None else list(DEFAULT_ROOTS)
    out: set[str] = set()
    for base in bases:
        if not base.is_dir():
            continue
        for direct in base.iterdir():
            if direct.is_dir() and _has_required_files(direct):
                out.add(direct.name)
        for collection in base.iterdir():
            if not collection.is_dir():
                continue
            for level1 in collection.iterdir():
                if level1.is_dir() and _has_required_files(level1):
                    out.add(level1.name)
                if not level1.is_dir():
                    continue
                for level2 in level1.iterdir():
                    if level2.is_dir() and _has_required_files(level2):
                        out.add(level2.name)
    return sorted(out)


def summarise_bundle(bundle: DatasetBundle) -> dict[str, Any]:
    """Compact dict summary; safe to drop into a JSON record."""
    return {
        "name": bundle.name,
        "path": str(bundle.path),
        "n_train": bundle.n_train,
        "n_test": bundle.n_test,
        "n_features": bundle.n_features,
        "y_train_finite": int(np.isfinite(bundle.y_train).sum()),
        "y_test_finite": int(np.isfinite(bundle.y_test).sum()),
        "has_meta_train": bundle.meta_train is not None,
        "has_meta_test": bundle.meta_test is not None,
    }
