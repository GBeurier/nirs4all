"""Keep generated workers on the parent's scientific package closure."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def scientific_worker_environment(interpreter: str | None = None) -> dict[str, str]:
    """Prefer the parent's installed core packages while retaining user modules.

    ``-P -s`` removes the implicit current-directory and user-site imports in
    the child. The explicit path keeps the parent's three scientific packages
    ahead of project modules and inherited ``PYTHONPATH`` entries, so a stale
    user installation cannot change the class loaded by cloudpickle.
    """
    environment = dict(os.environ)
    roots: list[str] = []
    # An explicitly different interpreter may use another Python ABI and its
    # own wheel set. Do not put this process's native packages on that path.
    if interpreter is None or os.path.abspath(interpreter) == os.path.abspath(sys.executable):
        for name in ("nirs4all", "nirs4all_io", "dag_ml"):
            module_path = getattr(importlib.import_module(name), "__file__", None)
            if module_path is None:
                raise RuntimeError(f"{name} has no importable package location")
            roots.append(str(Path(module_path).resolve().parent.parent))
    roots.append(os.getcwd())
    roots.extend(part for part in environment.get("PYTHONPATH", "").split(os.pathsep) if part)
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(roots))
    environment["PYTHONNOUSERSITE"] = "1"
    return environment
