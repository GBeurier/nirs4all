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
import os
from pathlib import Path
from typing import Any

# Short model name → fully-qualified class. Covers the parity baseline models; extend as
# cases land. No fuzzy lookup: an unknown name fails loudly rather than mis-resolving.
_MODEL_TABLE: dict[str, str] = {
    "PLSRegression": "sklearn.cross_decomposition.PLSRegression",
    "Ridge": "sklearn.linear_model.Ridge",
    "RandomForestRegressor": "sklearn.ensemble.RandomForestRegressor",
    "GradientBoostingRegressor": "sklearn.ensemble.GradientBoostingRegressor",
}

_METHODS_SNV_ENV = "N4A_DAGML_METHODS_SNV"
_STANDARD_SNV_FQNS = frozenset({
    "nirs4all.operators.transforms.scalers.StandardNormalVariate",
    "nirs4all.operators.transforms.StandardNormalVariate",
})
_METHODS_SNV_FQNS = frozenset({
    "nirs4all.operators.methods.n4m_ops.MethodsSNV",
    "nirs4all.operators.methods.MethodsSNV",
})


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


def _methods_snv_enabled() -> bool:
    """Whether the dag-ml router may replace Python SNV with the n4m MethodsSNV."""
    value = os.environ.get(_METHODS_SNV_ENV, "")
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _integral_param(name: str, value: Any) -> int:
    """Return ``value`` as an int only when JSON/native generation kept it integral."""
    if isinstance(value, bool):
        raise ValueError(f"dag-ml n4m SNV route requires integer `{name}`, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(f"dag-ml n4m SNV route requires integer `{name}`, got {value!r}")


def _bool_param(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"dag-ml n4m SNV route requires boolean `{name}`, got {value!r}")
    return value


def _methods_snv_params(params: dict[str, Any]) -> dict[str, Any]:
    """Validate the Python SNV params that are numerically identical to MethodsSNV."""
    supported = {"axis", "with_mean", "with_std", "ddof", "copy"}
    unknown = sorted(set(params) - supported)
    if unknown:
        raise ValueError(f"dag-ml n4m SNV route does not support StandardNormalVariate param(s) {unknown}")

    axis = _integral_param("axis", params.get("axis", 1))
    if axis != 1:
        raise ValueError(f"dag-ml n4m SNV route supports only row-wise StandardNormalVariate(axis=1), got axis={axis}")

    copy = _bool_param("copy", params.get("copy", True))
    if not copy:
        raise ValueError("dag-ml n4m SNV route supports only StandardNormalVariate(copy=True)")

    ddof = _integral_param("ddof", params.get("ddof", 0))
    if ddof < 0:
        raise ValueError(f"dag-ml n4m SNV route requires ddof >= 0, got {ddof}")

    return {
        "with_mean": _bool_param("with_mean", params.get("with_mean", True)),
        "with_std": _bool_param("with_std", params.get("with_std", True)),
        "ddof": ddof,
    }


def _assert_methods_snv_available() -> None:
    """Fail closed before execution if the n4m SNV binding or ABI probe is unavailable."""
    from nirs4all.operators.methods import n4m_ops

    if getattr(n4m_ops, "_N4MSNV", None) is None:
        raise ImportError(
            "dag-ml n4m SNV route requires nirs4all-methods with the SNV binding; "
            "unset N4A_DAGML_METHODS_SNV or install a compatible nirs4all-methods wheel"
        )
    try:
        n4m = importlib.import_module("n4m")
        abi_version = n4m.abi_version()
        library_path = Path(n4m.library_path())
    except Exception as exc:  # noqa: BLE001 - any ABI/library probe failure makes the route unavailable
        raise ImportError(f"dag-ml n4m SNV route could not validate the n4m ABI/library: {exc}") from exc
    if not abi_version:
        raise ImportError("dag-ml n4m SNV route could not validate the n4m ABI: empty abi_version()")
    if not library_path.exists():
        raise ImportError(f"dag-ml n4m SNV route could not validate the n4m library path: {library_path}")


def _route_methods_snv(params: dict[str, Any]) -> object:
    """Instantiate MethodsSNV for the proven-safe StandardNormalVariate subset."""
    methods_params = _methods_snv_params(params)
    _assert_methods_snv_available()
    from nirs4all.operators.methods import MethodsSNV

    return MethodsSNV(**methods_params)


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
        # Short aliases stay supported; otherwise the model id IS a fully-qualified class (the bridge
        # now emits FQNs), so any sklearn-style estimator — regressor or classifier — is imported.
        fqn = _MODEL_TABLE.get(operator_ref, operator_ref)
    elif operator_kind in ("transform", "y_transform"):
        fqn = operator_ref
    else:
        raise ValueError(f"unsupported operator_kind {operator_kind!r}")
    if operator_kind == "transform" and fqn in _STANDARD_SNV_FQNS and _methods_snv_enabled():
        return _route_methods_snv(merged)
    if fqn in _METHODS_SNV_FQNS:
        _assert_methods_snv_available()
    cls = _import_class(fqn)
    return cls(**_coerce_json_params(cls, merged))


def _coerce_one(value: Any, default: Any) -> Any:
    """Coerce one JSON-decoded param value back to the type its operator default implies.

    JSON loses Python's int-vs-float and tuple-vs-list distinctions, which some sklearn operators
    validate strictly at fit. Two default-driven coercions (only when the default is informative):

    * default is a ``tuple`` and value is a ``list`` → tuple (e.g. ``MinMaxScaler(feature_range``).
    * default is an ``int`` and value is an integral ``float`` → int (e.g. dag-ml's native range
      generator emits ``n_components`` as ``3.0``; ``PLSRegression`` rejects a float). ``bool`` is an
      ``int`` subclass but is left untouched; non-integral floats (e.g. ``alpha=0.001``) pass through.
    """
    if isinstance(value, list) and isinstance(default, tuple):
        return tuple(value)
    if (
        isinstance(value, float)
        and not isinstance(value, bool)
        and isinstance(default, int)
        and not isinstance(default, bool)
        and value.is_integer()
    ):
        return int(value)
    return value


def _coerce_json_params(cls: type, params: dict[str, Any]) -> dict[str, Any]:
    """Restore param types lost to JSON, driven by the operator's own defaults (see ``_coerce_one``)."""
    try:
        defaults = cls().get_params()
    except (TypeError, AttributeError):  # operator needs ctor args / is not an estimator
        return params
    return {key: _coerce_one(value, defaults.get(key)) for key, value in params.items()}


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
