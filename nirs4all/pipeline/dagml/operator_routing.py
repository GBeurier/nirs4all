"""Operator-routing registry: dag-ml graph node → real nirs4all/sklearn operator.

Phase 2a lowers a live pipeline to compat DSL; the compiler turns each step into a graph
node carrying an operator ref + params. This module inverts that for execution: given a
node it instantiates the concrete operator a host controller fits/transforms/predicts
with. Mechanism-independent — both the CLI process-adapter and an in-process callback
route the same way.

The operator ref shape differs by node kind (verified against the compiled vertical
slice):

* ``transform``   — ``{"class": "<FQN>"}``; params live on the node.
* ``y_transform`` — ``{"class": "<FQN>", "params": {...}}``; params live on the operator.
* ``model``       — a bare short name string (``"PLSRegression"``); params on the node.

Transforms/y_transforms carry a fully-qualified class resolved by import; models carry a
short name resolved through an explicit allow-table (short names are ambiguous across
modules, so silent fuzzy resolution is refused). The transform-vs-target distinction is
the node *kind*, never the class — a bare scaler is an X transform, a ``y_processing``
scaler is a target transform.
"""

from __future__ import annotations

import importlib
from typing import Any

# Short model name → fully-qualified class. Covers the parity baseline models; extend as
# cases land. No fuzzy lookup: an unknown name fails loudly rather than mis-resolving.
_MODEL_TABLE: dict[str, str] = {
    "PLSRegression": "sklearn.cross_decomposition.PLSRegression",
    "Ridge": "sklearn.linear_model.Ridge",
    "RandomForestRegressor": "sklearn.ensemble.RandomForestRegressor",
    "GradientBoostingRegressor": "sklearn.ensemble.GradientBoostingRegressor",
}


def _import_class(fqn: str) -> type:
    """Import a fully-qualified ``module.QualName`` and return the class object."""
    module_name, _, qualname = fqn.rpartition(".")
    if not module_name:
        raise ValueError(f"not a fully-qualified class name: {fqn!r}")
    obj: Any = importlib.import_module(module_name)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    if not isinstance(obj, type):
        raise TypeError(f"{fqn!r} resolved to a {type(obj).__name__}, not a class")
    return obj


def route_operator(
    operator_kind: str,
    operator_ref: str,
    params: dict[str, Any] | None = None,
    *,
    variant_overrides: dict[str, Any] | None = None,
) -> object:
    """Instantiate the operator for one node.

    Args:
        operator_kind: ``transform`` | ``y_transform`` | ``model``.
        operator_ref: a fully-qualified class (transform/y_transform) or a short model
            name (model).
        params: operator constructor params.
        variant_overrides: per-variant params that win over ``params`` (generator sweeps).
    """
    merged = {**(params or {}), **(variant_overrides or {})}
    if operator_kind == "model":
        fqn = _MODEL_TABLE.get(operator_ref)
        if fqn is None:
            raise KeyError(f"no model registered for {operator_ref!r}; known: {sorted(_MODEL_TABLE)}")
    elif operator_kind in ("transform", "y_transform"):
        fqn = operator_ref
    else:
        raise ValueError(f"unsupported operator_kind {operator_kind!r}")
    return _import_class(fqn)(**merged)


def route_graph_node(node: dict[str, Any], *, variant_overrides: dict[str, Any] | None = None) -> object:
    """Instantiate the operator for a compiled dag-ml graph node.

    Normalizes the per-kind operator-ref shape into ``route_operator`` inputs.
    """
    kind = node["kind"]
    operator = node["operator"]
    if isinstance(operator, str):
        operator_ref, params = operator, node.get("params") or {}
    else:
        operator_ref = operator["class"]
        params = operator.get("params") or node.get("params") or {}
    return route_operator(kind, operator_ref, params, variant_overrides=variant_overrides)
