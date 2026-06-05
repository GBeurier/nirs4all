"""Lazy module proxy for deferring heavy imports to first use.

Some modules (e.g. chart controllers) must be importable eagerly — the controller
registry imports them at ``import nirs4all`` so routing knows about them — but only
need a heavy dependency (matplotlib) when they actually render. Binding the
dependency to a module-level :class:`LazyModule` keeps the familiar
``plt = lazy_module("matplotlib.pyplot")`` name while deferring the real import
until an attribute is first accessed, so importing the package stays cheap.

A module-level ``__getattr__`` (PEP 562) cannot serve this purpose: it is only
consulted for *external* attribute access on the package, not for *internal* global
name lookups like ``plt.subplots(...)`` inside the module's own functions. The proxy
object is bound as a real global, so internal usage resolves normally and triggers
the lazy import on the first attribute access.
"""
from __future__ import annotations

import importlib
from typing import Any


class LazyModule:
    """Proxy that imports the wrapped module on first attribute access.

    The wrapped module is imported once and cached; every attribute access
    thereafter is a plain ``getattr`` on the real module.
    """

    __slots__ = ("_lazy_name", "_lazy_module")

    def __init__(self, name: str) -> None:
        self._lazy_name = name
        self._lazy_module: Any = None

    def __getattr__(self, attr: str) -> Any:
        # __getattr__ runs only for names not found normally; the two slots are
        # found normally, so accessing them here does not recurse.
        module = self._lazy_module
        if module is None:
            module = importlib.import_module(self._lazy_name)
            self._lazy_module = module
        return getattr(module, attr)

    def __repr__(self) -> str:
        loaded = self._lazy_module is not None
        return f"<LazyModule {self._lazy_name!r} loaded={loaded}>"


def lazy_module(name: str) -> LazyModule:
    """Return a :class:`LazyModule` proxy for the importable module ``name``."""
    return LazyModule(name)
