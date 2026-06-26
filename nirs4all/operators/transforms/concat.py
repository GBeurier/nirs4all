"""``FeatureConcat`` — a JSON-spec'd horizontal feature concatenation transformer.

This is the single-source, single-processing ("replace mode") lowering of nirs4all's
``concat_transform`` step onto the dag-ml engine (backlog #27). nirs4all's
``ConcatAugmentationController`` ``np.hstack``-es the outputs of several sub-transformers
(each fit on the training rows) into one wider 2D feature matrix — exactly the semantics of
``sklearn.pipeline.FeatureUnion``. Expressing it as ONE sklearn-compatible transformer lets the
dag-ml engine run it as an ordinary X-chain transform node: the model node reconstructs
``make_pipeline(FeatureConcat(...), model)`` and fits it on fold-train only (leakage-safe), so
the column-count change propagates through fold-val / test apply just like the supervised /
column-changing X-transforms (OSC/EPO/CARS/MC-UVE, backlog #24).

The sub-transformer list is carried as a JSON-serializable ``operations`` spec (a list of
``{"class": "<FQN>", "params": {...}}`` entries, where an entry may itself be a *chain* —
a list of such dicts applied sequentially) so the whole node round-trips through the dag-ml
compat DSL as a single ``{"class", "params"}`` node. The transformers are instantiated lazily at
``fit`` time, so an unfitted ``FeatureConcat`` stays JSON-cloneable.

Scope (host-only, single-source replace mode): the multi-processing 3D ``signal_with_processings``
and the ``feature_augmentation``-nested *add* mode (which grow the processing axis) are NOT covered
here — they need the data-plane multi-block representation (backlog #29/#31).
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import FeatureUnion


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


class _Chain(BaseEstimator, TransformerMixin):
    """A sequential transformer chain (``[A, B, C] → C(B(A(X)))``) for use inside a ``FeatureUnion``.

    A plain ``make_pipeline`` is unusable here: ``Pipeline.transform`` runs ``check_is_fitted`` on
    itself, which raises ``NotFittedError`` when *every* step is a nirs4all stateless transformer
    (those set no fitted attribute), even right after ``fit``. This wrapper fits/transforms the steps
    directly, so a chain of stateless preprocessings concatenates cleanly.
    """

    def __init__(self, steps: list[Any]):
        self.steps = steps

    def fit(self, X: Any, y: Any = None) -> _Chain:
        self.fitted_: list[Any] = []
        current = X
        for step in self.steps:
            fitted = clone(step)
            current = fitted.fit_transform(current, y)
            self.fitted_.append(fitted)
        return self

    def transform(self, X: Any) -> Any:
        current = X
        for fitted in self.fitted_:
            current = fitted.transform(current)
        return current


def _build_operation(operation: Any) -> Any:
    """Instantiate one ``operations`` entry: a single ``{class, params}`` or a chain list thereof."""
    if isinstance(operation, list):
        # Chain [A, B, C] → C(B(A(X))), applied sequentially before concatenation.
        return _Chain([_build_operation(item) for item in operation])
    cls = _import_class(operation["class"])
    return cls(**operation.get("params", {}))


class FeatureConcat(BaseEstimator, TransformerMixin):
    """Fit several sub-transformers and horizontally concatenate their transform outputs.

    Mirrors nirs4all's top-level ``concat_transform`` (replace mode) on a single 2D feature matrix:
    every sub-transformer fits on the same training rows and the per-sample outputs are stacked
    column-wise. Equivalent to ``sklearn.pipeline.FeatureUnion`` over the supplied operations, but
    parameterised by a JSON-serializable ``operations`` spec so it serializes as one DSL node.

    Args:
        operations: A list whose entries are each either a single transformer spec
            (``{"class": "<FQN>", "params": {...}}``) or a *chain* (a list of such specs applied
            sequentially). ``None`` entries are rejected — a pass-through "raw" channel belongs in
            the 3D processing-axis path, not this flat single-matrix lowering.
    """

    _stateless = False

    def __init__(self, operations: list[Any] | None = None):
        self.operations = operations

    def _make_union(self) -> FeatureUnion:
        if not self.operations:
            raise ValueError("FeatureConcat requires a non-empty `operations` spec")
        return FeatureUnion(
            [(f"op{index}", _build_operation(operation)) for index, operation in enumerate(self.operations)]
        )

    def fit(self, X: Any, y: Any = None) -> FeatureConcat:
        self.union_ = self._make_union()
        self.union_.fit(np.asarray(X, dtype=float), y)
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.asarray(self.union_.transform(np.asarray(X, dtype=float)), dtype=float)
