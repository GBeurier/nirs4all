"""Per-method adaptations for shared n4m JSON/YAML recipes."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PortableMethodSpec:
    """Python binding and parameter contract for one shared Methods recipe."""

    class_path: str
    defaults: dict[str, Any] = field(default_factory=dict)
    required: frozenset[str] = frozenset()
    validate: Callable[[dict[str, Any]], None] | None = None


def _validate_group_sparse(params: dict[str, Any]) -> None:
    components = params["n_components"]
    groups = params.get("group_assignment")
    group_lambda = params["group_lambda"]
    if type(components) is not int or not 1 <= components <= 2**31 - 1:
        raise ValueError("portable n4m n_components must be a positive integer")
    if (not isinstance(groups, list) or len(groups) < 2
            or any(type(group) is not int or not 0 <= group <= 2**31 - 1
                   for group in groups)):
        raise ValueError("portable n4m group_assignment must contain at least two nonnegative int32 group IDs")
    try:
        valid_lambda = (type(group_lambda) in (int, float)
                        and math.isfinite(group_lambda) and group_lambda >= 0)
    except OverflowError:
        valid_lambda = False
    if not valid_lambda:
        raise ValueError("portable n4m group_lambda must be finite and nonnegative")


PORTABLE_METHOD_SPECS: dict[str, PortableMethodSpec] = {
    "n4m.GroupSparsePLS": PortableMethodSpec(
        class_path="n4m.estimators.regression.sparse.GroupSparsePLS",
        defaults={"n_components": 2, "group_lambda": 0.05},
        required=frozenset({"group_assignment"}),
        validate=_validate_group_sparse,
    ),
}
