"""Utilities for forwarding pipeline folds to split-aware estimators."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

Fold = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class PrecomputedFoldSplitter:
    """Picklable sklearn-compatible splitter backed by explicit fold indices."""

    folds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    n_samples: int | None = None
    label: str = "pipeline"

    @classmethod
    def from_folds(
        cls,
        folds: Iterable[tuple[Sequence[int], Sequence[int]]],
        *,
        n_samples: int | None = None,
        label: str = "pipeline",
    ) -> PrecomputedFoldSplitter:
        normalised: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for train_idx, val_idx in folds:
            train_tuple = tuple(int(i) for i in train_idx)
            val_tuple = tuple(int(i) for i in val_idx)
            if train_tuple and val_tuple:
                normalised.append((train_tuple, val_tuple))
        return cls(tuple(normalised), n_samples=n_samples, label=label)

    def split(self, X, y=None, groups=None):  # noqa: ANN001, ARG002 - sklearn protocol
        if self.n_samples is not None and len(X) != int(self.n_samples):
            raise ValueError(
                f"{self.label} splitter expected {self.n_samples} rows, got {len(X)}"
            )
        for train_idx, val_idx in self.folds:
            yield np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: ANN001, ARG002 - sklearn protocol
        return len(self.folds)

    @property
    def validation_folds(self) -> list[list[int]]:
        """Return validation indices in the external-folds format used by AOM_lib."""

        return [list(val_idx) for _, val_idx in self.folds]

    def for_training_subset(
        self,
        train_indices: Sequence[int],
        *,
        label: str | None = None,
    ) -> PrecomputedFoldSplitter | None:
        """Restrict this splitter to rows selected by ``train_indices``.

        ``train_indices`` are row positions in the current splitter's coordinate
        system. The returned splitter uses positions in ``X[train_indices]``.
        """

        active = np.asarray(train_indices, dtype=int)
        if active.ndim != 1 or active.size == 0:
            return None
        position_by_parent = {int(parent_pos): int(local_pos) for local_pos, parent_pos in enumerate(active)}
        local_folds: list[tuple[list[int], list[int]]] = []
        for parent_train_idx, parent_val_idx in self.folds:
            local_val = [position_by_parent[int(i)] for i in parent_val_idx if int(i) in position_by_parent]
            if not local_val:
                continue
            local_val_set = set(local_val)
            local_train = [
                position_by_parent[int(i)]
                for i in parent_train_idx
                if int(i) in position_by_parent and position_by_parent[int(i)] not in local_val_set
            ]
            if not local_train:
                continue
            local_folds.append((local_train, local_val))
        if not local_folds:
            return None
        return PrecomputedFoldSplitter.from_folds(
            local_folds,
            n_samples=int(active.size),
            label=label or f"{self.label}:subset",
        )


def make_pipeline_fold_splitter(
    pipeline_folds: Sequence[tuple[Sequence[int], Sequence[int]]] | None,
    *,
    n_samples: int,
    train_indices: Sequence[int] | None = None,
    label: str = "pipeline",
) -> PrecomputedFoldSplitter | None:
    """Build a splitter in the coordinate system of the current training matrix."""

    if not pipeline_folds:
        return None
    candidate: PrecomputedFoldSplitter | None
    if train_indices is None:
        candidate = PrecomputedFoldSplitter.from_folds(
            pipeline_folds,
            n_samples=int(n_samples),
            label=label,
        )
    else:
        parent = PrecomputedFoldSplitter.from_folds(
            pipeline_folds,
            label=f"{label}:parent",
        )
        candidate = parent.for_training_subset(train_indices, label=label)
    if candidate is None or candidate.get_n_splits() < 2:
        return None
    return candidate


def is_aom_estimator(estimator: Any) -> bool:
    """Return whether an estimator belongs to the vendored AOM family."""

    cls = estimator.__class__
    module = str(getattr(cls, "__module__", "")).lower()
    name = str(getattr(cls, "__name__", "")).lower()
    return (
        "aom" in name
        or "._aom_nirs." in module
        or ".sklearn.aom_" in module
        or module.endswith(".aom_pls_aomlib")
    )


def pipeline_cv_policy_enabled(policy: Any) -> bool:
    """Interpret train_params.use_pipeline_folds_for_aom."""

    if isinstance(policy, str):
        return policy.strip().lower() not in {"0", "false", "no", "off", "none", "disabled"}
    return bool(policy)


def pipeline_cv_policy_required(policy: Any) -> bool:
    """Return whether missing pipeline folds should fail the fit."""

    return isinstance(policy, str) and policy.strip().lower() in {"required", "require", "strict"}


def apply_pipeline_folds_to_aom_estimator(
    estimator: Any,
    splitter: PrecomputedFoldSplitter | None,
    *,
    policy: Any = "auto",
    unavailable_reason: str | None = None,
) -> bool:
    """Inject pipeline folds into supported AOM estimator parameters.

    Returns ``True`` when at least one estimator parameter was changed.
    """

    if not is_aom_estimator(estimator) or not pipeline_cv_policy_enabled(policy):
        return False
    if splitter is None:
        if pipeline_cv_policy_required(policy):
            reason = unavailable_reason or "no pipeline fold splitter was provided"
            raise ValueError(f"Pipeline folds are required for {estimator.__class__.__name__}, but {reason}.")
        return False
    params = estimator.get_params(deep=False) if hasattr(estimator, "get_params") else {}
    updates: dict[str, Any] = {}
    if "external_folds" in params:
        updates["external_folds"] = splitter.validation_folds
        if "selection" in params:
            updates["selection"] = "external"
        if "cv" in params:
            updates["cv"] = splitter.get_n_splits()
    if "cv_splitter" in params:
        repeats = params.get("repeats", 1)
        if repeats not in (None, 1, "1"):
            if pipeline_cv_policy_required(policy):
                raise ValueError(
                    f"Pipeline folds cannot be used with {estimator.__class__.__name__} "
                    "when repeats > 1; set repeats=1."
                )
            return bool(updates and _set_estimator_params(estimator, updates))
        updates["cv_splitter"] = splitter
        if "cv" in params:
            updates["cv"] = splitter.get_n_splits()
    if "outer_cv" in params:
        updates["outer_cv"] = splitter
        if "inner_cv" in params:
            updates["inner_cv"] = splitter
    elif "cv" in params and "external_folds" not in params and "cv_splitter" not in params:
        updates["cv"] = splitter
    if not updates:
        if pipeline_cv_policy_required(policy):
            raise ValueError(
                f"{estimator.__class__.__name__} does not expose a supported pipeline-fold parameter "
                "(expected one of cv, cv_splitter, outer_cv, external_folds)."
            )
        return False
    return _set_estimator_params(estimator, updates)


def _set_estimator_params(estimator: Any, updates: dict[str, Any]) -> bool:
    if hasattr(estimator, "set_params"):
        estimator.set_params(**updates)
    else:
        for key, value in updates.items():
            setattr(estimator, key, value)
    return True
