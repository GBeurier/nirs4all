"""Fold-local preprocessing for legacy model hyperparameter evaluation."""

from typing import Any

import numpy as np


def finetune_inputs(dataset: Any, context: Any, controller: Any, X: Any) -> tuple[Any, Any]:
    """Capture sample IDs before splitting raw input for a tuning study."""
    custom = getattr(context, "custom", None)
    plan = custom.get("cv_preprocessing") if isinstance(custom, dict) else None
    if not isinstance(custom, dict) or plan is None or not plan.active or custom.get("meta_operator"):
        return X, context
    ids, _ = controller._get_partition_sample_indices(dataset, context, "finetune", include_augmented=True)
    if len(ids) != len(X):
        ids, _ = controller._get_partition_sample_indices(dataset, context, "finetune", include_augmented=False)
    if len(ids) != len(X):
        raise ValueError("Finetuning preprocessing requires aligned training sample IDs")
    local = context.copy()
    local.custom["finetune_sample_ids"] = np.asarray(ids)
    local.custom["finetune_preprocessing_cache"] = {}
    return plan.raw_features(ids, context.selector.layout or "2d"), local


def finetune_fold(dataset: Any, context: Any, X: Any, train_indices: Any, val_indices: Any) -> tuple[Any, Any]:
    """Fit preprocessing on training IDs and replay on held-out features."""
    X_train, X_val = X[train_indices], X[val_indices]
    custom = getattr(context, "custom", None)
    if not isinstance(custom, dict) or "finetune_sample_ids" not in custom or custom.get("meta_operator"):
        return X_train, X_val
    plan = custom["cv_preprocessing"]
    ids = custom["finetune_sample_ids"][train_indices]
    key = tuple(int(i) for i in ids)
    cache = custom["finetune_preprocessing_cache"]
    if key not in cache:
        cache[key] = plan.prepare_fold(dataset, context, ids)
    replay = cache[key]
    return replay.transform(X_train), replay.transform(X_val)
