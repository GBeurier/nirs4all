"""``FeatureConcat`` — a JSON-spec'd horizontal feature concatenation transformer.

This is the single-source, single-processing horizontal-concat lowering of nirs4all's
``concat_transform`` (backlog #27) and ``feature_augmentation`` (backlog #31 / S6) steps onto the
dag-ml engine. Both grow the dataset's *processing* axis (parallel preprocessing layers on the same
samples); for a 2D model nirs4all materializes that axis with the ``FLAT_2D`` layout, which
``reshape``-flattens ``(samples, processings, features)`` → ``(samples, processings × features)`` —
i.e. an ``np.hstack`` of the per-processing layers in processing order (``layout_transformer.py``).
So a 2D model consuming the augmented/concatenated layers sees exactly the same matrix as a
``FeatureUnion`` over the layer transformers. Expressing it as ONE sklearn-compatible transformer
lets the dag-ml engine run it as an ordinary X-chain transform node: the model node reconstructs
``make_pipeline(FeatureConcat(...), model)`` and fits it on fold-train only (leakage-safe), so the
column-count change propagates through fold-val / test apply just like the supervised /
column-changing X-transforms (OSC/EPO/CARS/MC-UVE, backlog #24).

The sub-transformer list is carried as a JSON-serializable ``operations`` spec (a list of
``{"class": "<FQN>", "params": {...}}`` entries, where an entry may itself be a *chain* —
a list of such dicts applied sequentially, or ``None`` for a pass-through "raw" channel) so the
whole node round-trips through the dag-ml compat DSL as a single ``{"class", "params"}`` node. The
transformers are instantiated lazily at ``fit`` time, so an unfitted ``FeatureConcat`` stays
JSON-cloneable.

Pass-through (raw) channel: a ``None`` entry keeps the un-transformed base layer alongside the new
ones — this is the ``feature_augmentation`` *extend*/*add* modes' raw layer (``[raw, op1(raw),
op2(raw), …]``), lowered as ``FeatureConcat([None, op1, op2, …])``. *replace* mode drops the raw
layer (``FeatureConcat([op1, op2, …])``, identical to ``concat_transform``). The raw channel maps to
``sklearn``'s native ``"passthrough"`` ``FeatureUnion`` member (raw columns first, in spec order).

Scope (host-only, single-source 2D model): the multi-processing 3D ``signal_with_processings``
delivered as parallel channels to a 3D/CNN model, multi-source feature_augmentation, and the nested
(concat-of-concat) shapes are NOT covered here — those need the 3D data-plane / a DL operator
(backlog #29/#31).
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
    """Instantiate one ``operations`` entry: ``None`` (raw pass-through), a single ``{class, params}``, or a chain list."""
    if operation is None:
        # The feature_augmentation extend/add raw layer: keep the base matrix unchanged beside the
        # transformed layers. sklearn's native FeatureUnion member "passthrough" emits X as-is.
        return "passthrough"
    if isinstance(operation, list):
        # Chain [A, B, C] → C(B(A(X))), applied sequentially before concatenation.
        return _Chain([_build_operation(item) for item in operation])
    cls = _import_class(operation["class"])
    return cls(**operation.get("params", {}))


class FeatureConcat(BaseEstimator, TransformerMixin):
    """Fit several sub-transformers and horizontally concatenate their transform outputs.

    Mirrors nirs4all's ``concat_transform`` (replace mode) and ``feature_augmentation`` (extend/add/
    replace) on a single 2D feature matrix: every sub-transformer fits on the same training rows and
    the per-sample outputs are stacked column-wise. Equivalent to ``sklearn.pipeline.FeatureUnion``
    over the supplied operations, but parameterised by a JSON-serializable ``operations`` spec so it
    serializes as one DSL node.

    Args:
        operations: A list whose entries are each either a single transformer spec
            (``{"class": "<FQN>", "params": {...}}``), a *chain* (a list of such specs applied
            sequentially), or ``None`` for a pass-through "raw" channel (the un-transformed base
            layer — the feature_augmentation extend/add raw layer). The raw channel emits its columns
            first, in spec order.
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
