"""Resolve the installed portable Methods runtime without a worktree path."""

from __future__ import annotations

import importlib
from os import fspath
from pathlib import Path
from typing import Any

from .native_client import DagMLNativeCoverageError


def resolve_methods_library_path(path: str | Path | None = None) -> str:
    """Return one absolute regular ``libn4m`` path for native DAG-ML calls.

    A caller may explicitly select a library, which is useful for an audited
    deployment.  Otherwise the resolver uses the bundled library exposed by
    the official ``nirs4all-methods`` distribution (import package ``n4m``).
    The resolved path is passed explicitly to DAG-ML; no sibling checkout or
    loader search path becomes part of the portable training contract.
    """

    if path is not None:
        return _validated_library_path(path, source="methods_library_path")

    try:
        module = importlib.import_module("n4m")
    except Exception as error:  # pragma: no cover - exact import errors vary by platform
        raise DagMLNativeCoverageError(
            "native Methods execution requires the bundled n4m runtime; install "
            "nirs4all[native] or provide an explicit absolute methods_library_path"
        ) from error

    library_path = getattr(module, "library_path", None)
    if not callable(library_path):
        raise DagMLNativeCoverageError(
            "installed n4m runtime does not expose library_path(); upgrade nirs4all-methods "
            "or provide an explicit absolute methods_library_path"
        )
    _require_optimizer_abi(module)
    try:
        return _validated_library_path(library_path(), source="nirs4all-methods")
    except (OSError, TypeError, ValueError) as error:
        raise DagMLNativeCoverageError(
            "installed n4m runtime did not resolve a usable bundled libn4m; provide an "
            "explicit absolute methods_library_path"
        ) from error


def _require_optimizer_abi(module: Any) -> None:
    abi_version = getattr(module, "abi_version", None)
    if not callable(abi_version):
        raise DagMLNativeCoverageError(
            "installed n4m runtime does not expose ABI metadata; upgrade nirs4all-methods "
            "or provide an explicit absolute methods_library_path"
        )
    try:
        abi = tuple(abi_version())
    except Exception as error:  # pragma: no cover - runtime-specific loader failure
        raise DagMLNativeCoverageError(
            "installed n4m runtime could not report its ABI; upgrade nirs4all-methods"
        ) from error
    if len(abi) != 3 or abi[0:2] != (2, 2):
        raise DagMLNativeCoverageError(
            "installed n4m runtime ABI is incompatible with native DAG-ML Methods execution; "
            "install nirs4all-methods>=1.0.10 or provide a compatible explicit "
            "methods_library_path"
        )


def _validated_library_path(value: str | Path | Any, *, source: str) -> str:
    try:
        candidate = Path(fspath(value))
    except TypeError as error:
        raise TypeError(f"{source} must identify an absolute libn4m file") from error
    if not candidate.is_absolute():
        raise ValueError(f"{source} must identify an absolute libn4m file")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{source} does not identify an existing libn4m file") from error
    if not resolved.is_file():
        raise ValueError(f"{source} must identify a regular libn4m file")
    return str(resolved)


__all__ = ["resolve_methods_library_path"]
