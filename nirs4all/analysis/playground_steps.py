"""Exploration steps delegated to existing library/sklearn operators.

The caller supplies an already resolved operator, not an import path. No
dynamic registry traversal, parameter dropping or retry is performed here.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any

import numpy as np
from sklearn.base import clone

from nirs4all.analysis.playground_types import PreviewBatch, PreviewLimits, positive_count
from nirs4all.data.selection.sampling import kmeans_sample, random_sample, stratified_sample


def sample_batch(batch: PreviewBatch, *, method: str = "all", n_samples: int = 100,
                 seed: int = 42) -> tuple[PreviewBatch, list[str]]:
    """Use owner sampling, retaining all aligned columns and original indices."""
    count = positive_count(n_samples, "n_samples")
    positive_count(seed, "seed", allow_zero=True)
    messages: list[str] = []
    if method == "all":
        return batch, messages
    if method == "random" or method == "stratified" and batch.y is None:
        if method == "stratified":
            messages.append("Stratified sampling has no observed y; historical random sampling is used")
        indices = random_sample(len(batch.x), count, seed)
    elif method == "stratified":
        assert batch.y is not None  # Missing-target policy was handled above.
        indices = stratified_sample(batch.x, batch.y, count, seed)
    elif method == "kmeans":
        indices = kmeans_sample(batch.x, count, seed)
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    return batch.take(indices), messages


def _fit_kwargs(operator: Any, batch: PreviewBatch) -> dict[str, Any]:
    signature = inspect.signature(operator.fit_transform)
    kwargs: dict[str, Any] = {}
    if "y" in signature.parameters and batch.y is not None:
        kwargs["y"] = batch.y.copy()
    fit_signature = inspect.signature(operator.fit)
    if getattr(operator, "_requires_wavelengths", False) or "wavelengths" in signature.parameters or "wavelengths" in fit_signature.parameters:
        # Historical inputs without measured headers use relative feature
        # positions. Keep that capability, explicitly labelled feature_index.
        kwargs["wavelengths"] = batch.wavelengths.copy()
    return kwargs


def _output_axis(operator: Any, batch: PreviewBatch, width: int) -> tuple[np.ndarray, str, str | None]:
    from nirs4all.operators.transforms.features import CropTransformer, ResampleTransformer
    from nirs4all.operators.transforms.resampler import Resampler

    if isinstance(operator, Resampler):
        # get_feature_names_out rounds to two decimals; fitted owner grid is exact.
        return np.asarray(operator.interpolator_params_["target_wavelengths"], dtype=float), batch.axis_kind, batch.header_unit
    if isinstance(operator, CropTransformer):
        return np.asarray(operator.transform(batch.wavelengths[None])[0]), batch.axis_kind, batch.header_unit
    if isinstance(operator, ResampleTransformer):
        return np.asarray(operator.transform(batch.wavelengths[None])[0]), batch.axis_kind, batch.header_unit
    if callable(getattr(operator, "get_support", None)):
        return batch.wavelengths[operator.get_support(indices=True)], batch.axis_kind, batch.header_unit
    if callable(getattr(operator, "get_feature_names_out", None)):
        names = operator.get_feature_names_out(batch.wavelengths.astype(str))
        try:
            return np.asarray(names, dtype=float), batch.axis_kind, batch.header_unit
        except (TypeError, ValueError):
            return np.arange(width, dtype=float), "feature_index", None
    if width == len(batch.wavelengths):
        return batch.wavelengths.copy(), batch.axis_kind, batch.header_unit
    return np.arange(width, dtype=float), "feature_index", None


def transform_batch(batch: PreviewBatch, operator: Any, *, limits: PreviewLimits | None = None) -> PreviewBatch:
    """Clone and fit-transform once, tracking the true output feature axis."""
    budget = limits or PreviewLimits()
    budget.admit(*batch.x.shape)
    # Known resampling expansion is admitted before clone/fit/allocation.
    parameters = operator.get_params(deep=False)
    requested = parameters.get("target_wavelengths")
    width = len(requested) if requested is not None else parameters.get("num_samples")
    if width is not None:
        budget.admit(len(batch.x), positive_count(width, "output features"))
    fitted = clone(operator)
    transformed = fitted.fit_transform(batch.x.copy(), **_fit_kwargs(fitted, batch))
    if hasattr(transformed, "toarray"):
        budget.admit(*transformed.shape)
        transformed = transformed.toarray()
    result = np.asarray(transformed, dtype=float)
    if result.ndim != 2 or len(result) != len(batch.x):
        raise ValueError("Preprocessing must preserve sample rows")
    budget.admit(*result.shape)
    axis, kind, unit = _output_axis(fitted, batch, result.shape[1])
    return replace(batch, x=result, wavelengths=axis, axis_kind=kind, header_unit=unit)


def augment_batch(batch: PreviewBatch, operator: Any, *, copies: int = 1,
                  limits: PreviewLimits | None = None) -> tuple[PreviewBatch, dict[str, int]]:
    """Append operator-generated copies, aligning observed labels and metadata.

    Bounds are checked before cloning, fitting, copying or concatenation.
    Reusing one cloned operator across copies preserves its random-state policy.
    """
    count = positive_count(copies, "copies", allow_zero=True)
    budget = limits or PreviewLimits()
    total = len(batch.x) * (count + 1)
    budget.admit(total, batch.x.shape[1])
    fitted = clone(operator)
    matrices = [batch.x]
    for _ in range(count):
        augmented = np.asarray(fitted.fit_transform(batch.x.copy(), **_fit_kwargs(fitted, batch)), dtype=float)
        if augmented.shape != batch.x.shape:
            raise ValueError("Augmentation must preserve each copied matrix shape")
        matrices.append(augmented)
    repeated = batch.take(np.tile(np.arange(len(batch.x)), count + 1))
    return replace(repeated, x=np.concatenate(matrices, axis=0)), {
        "original_count": len(batch.x), "augmented_count": total - len(batch.x), "total_count": total,
    }
