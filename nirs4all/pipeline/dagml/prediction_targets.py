"""Project observed targets into a captured classifier's frozen output domain."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.preprocessing import FunctionTransformer

from nirs4all.data._targets.converters import NumericConverter

from .target_capture import CategoricalTargetTransform
from .target_domains import validate_captured_bin_domain

if TYPE_CHECKING:
    from nirs4all.api.result import PredictResult


def _metric_task_type(transformer: CategoricalTargetTransform) -> str:
    from sklearn.preprocessing import KBinsDiscretizer

    from nirs4all.operators.transforms.targets import IntegerKBinsDiscretizer, RangeDiscretizer

    operator = transformer.transformer
    while isinstance(getattr(operator, "steps", None), (list, tuple)) and operator.steps:
        operator = operator.steps[-1][1]
    if isinstance(operator, RangeDiscretizer):
        count = operator.n_bins
    else:
        member = operator.discretizer if isinstance(operator, IntegerKBinsDiscretizer) else operator
        if not isinstance(member, KBinsDiscretizer) or not hasattr(member, "n_bins_"):
            raise ValueError("categorical scoring requires a captured fitted class vocabulary")
        count = int(np.max(member.n_bins_))
    return "binary_classification" if count <= 2 else "multiclass_classification"


def attach_categorical_scoring_context(result: PredictResult, artifact: dict[str, Any]) -> PredictResult:
    """Attach a private fitted context; public metadata remains plain JSON."""
    from nirs4all.api.result import _DagmlExportedModel

    transformer = artifact.get("y_transform")
    estimator = artifact["estimator"]
    if transformer is None and isinstance(estimator, _DagmlExportedModel):
        transformer = estimator.y_transform
    if not isinstance(transformer, CategoricalTargetTransform):
        return result
    validate_captured_bin_domain(transformer)
    result.__dict__["_categorical_scoring_transform"] = deepcopy(transformer)
    result.metadata["scoring_target_projection"] = {
        "schema_version": 1, "task_type": _metric_task_type(transformer), "target_space": "categorical_class_ids",
        "fitted_on_prediction_cohort": False,
    }
    return result


def categorical_scoring_targets(result: PredictResult, raw_targets: Any, validity_mask: Any = None) -> dict[str, Any] | None:
    """Apply only the captured input converter and fitted target transform.

    NumericConverter preserves the SDK's training target precision. Missing or
    unrepresentable values remain masked; none are turned into a class label.
    This function never fits an encoder or a target transform.
    """
    transformer = getattr(result, "_categorical_scoring_transform", None)
    if transformer is None:
        if result.metadata.get("scoring_target_projection", {}).get("target_space") == "categorical_class_ids":
            raise ValueError("scoring requires the original captured predictor; serialized prediction metadata cannot reconstruct its fitted target map")
        return None
    if not isinstance(transformer, CategoricalTargetTransform):
        raise ValueError("prediction result lost its categorical scoring context")
    captured = validate_captured_bin_domain(transformer)
    raw = np.asarray(raw_targets)
    if raw.ndim not in (1, 2) or raw.ndim == 2 and raw.shape[1] != 1:
        raise ValueError("categorical scoring requires one target column")
    if len(raw) != len(result.y_pred):
        raise ValueError("scoring targets must align with every prediction row")
    mask = np.ones(raw.shape, dtype=bool) if validity_mask is None else np.asarray(validity_mask, dtype=bool).copy()
    if mask.shape != raw.shape:
        raise ValueError("target validity mask must have the original target shape")
    if np.issubdtype(raw.dtype, np.number):
        mask &= np.isfinite(raw)
    else:
        mask &= np.asarray([value is not None and not (isinstance(value, (float, np.floating)) and not np.isfinite(value))
                            for value in raw.reshape(-1)], dtype=bool).reshape(raw.shape)
    output = np.full(raw.shape, np.nan, dtype=float)
    indices = np.flatnonzero(mask.reshape(-1))
    if indices.size:
        converter = getattr(transformer, "input_transformer_", None)
        if converter is None:
            converter = FunctionTransformer(validate=False)
        with np.errstate(over="ignore", invalid="ignore"):
            numeric, _ = NumericConverter.convert(raw.reshape(-1)[indices].reshape(-1, 1), existing_transformer=converter)
        numeric = np.asarray(numeric, dtype=float).reshape(-1, 1)
        representable = np.isfinite(numeric[:, 0])
        source_classes = getattr(converter, "classes_", None)
        columns = getattr(converter, "column_transformers", None)
        if source_classes is None and isinstance(columns, dict) and len(columns) == 1:
            source_classes = getattr(next(iter(columns.values())), "classes_", None)
        if source_classes is not None:
            # FlexibleLabelEncoder can assign temporary indices to unseen
            # labels. They are outside this frozen input axis, never new bins.
            representable &= (numeric[:, 0] >= 0) & (numeric[:, 0] < len(source_classes))
        mask.reshape(-1)[indices[~representable]] = False
        if representable.any():
            projected = np.asarray(transformer.transform(numeric[representable]), dtype=float).reshape(-1)
            if projected.shape != (int(representable.sum()),) or not np.isfinite(projected).all():
                raise ValueError("captured categorical transform returned invalid score targets")
            output.reshape(-1)[indices[representable]] = projected
    return {
        "schema_version": 1, "values": output, "validity_mask": mask,
        "task_type": _metric_task_type(transformer), "target_space": "categorical_class_ids",
        "target_domain": deepcopy(captured[1]) if captured is not None else None,
        "fitted_on_prediction_cohort": False,
    }
