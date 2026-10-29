"""Owner-side bin-domain provenance for native fold-local classification.

This module extracts the fitted target operator's state. Scoring, selection and
the admission of incompatible domains remain native DAG-ML responsibilities.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from numbers import Integral
from typing import Any

import numpy as np
import sklearn
from sklearn.preprocessing import KBinsDiscretizer

from nirs4all.operators.transforms.targets import IntegerKBinsDiscretizer, RangeDiscretizer
from nirs4all.pipeline.config.component_serialization import serialize_component

from .errors import DagMlUnsupported
from .target_capture import CategoricalTargetTransform


def _strict_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False)


def _unwrapped(transformer: Any) -> Any:
    return transformer.transformer if isinstance(transformer, CategoricalTargetTransform) else transformer


def fold_local_domains_available() -> bool:
    """Require an explicit native capability; package versions are insufficient."""
    try:
        native = importlib.import_module("dag_ml._dag_ml")
    except ImportError:
        return False
    version = getattr(native, "fold_local_target_domains_schema_version", None)
    return callable(version) and version() == 1


def learned_bin_transform(pipeline: list[Any]) -> Any | None:
    """Find the bounded flat, single-model target recipe eligible for native CV."""
    from nirs4all.pipeline.config.component_serialization import deserialize_component

    from .detect import _generation_kind
    from .public_normalization import normalize_model_steps

    # Keep branch/local chains outside this first qualified profile.
    if any(isinstance(step, list) or isinstance(step, dict) and any(key in step for key in ("branch", "merge", "residual")) for step in pipeline):
        return None
    steps = normalize_model_steps(pipeline)
    if _generation_kind(steps) == "operator":
        return None
    if any(isinstance(step, dict) and not any(key in step for key in ("model", "y_processing", "preprocessing", "split")) for step in steps):
        return None
    if sum(isinstance(step, dict) and "model" in step for step in steps) != 1:
        return None
    transforms = [step["y_processing"] for step in steps if isinstance(step, dict) and "y_processing" in step]
    if len(transforms) != 1:
        return None
    operator = transforms[0]
    if isinstance(operator, (dict, str)):
        operator = deserialize_component(operator)
    if isinstance(operator, type):
        operator = operator()
    try:
        bin_domain_recipe(operator)
    except DagMlUnsupported:
        return None
    return operator


def bin_domain_recipe(transformer: Any) -> dict[str, Any]:
    """Describe one supported mono-target bin recipe without fitting it."""
    transformer = _unwrapped(transformer)
    from nirs4all import __version__
    if isinstance(transformer, RangeDiscretizer):
        if not isinstance(transformer.bins, Integral) or isinstance(transformer.bins, bool) or transformer.bins < 2:
            raise DagMlUnsupported("fold-local learned RangeDiscretizer requires an integer bin count >= 2")
        count = int(transformer.bins)
    elif isinstance(transformer, (IntegerKBinsDiscretizer, KBinsDiscretizer)):
        if (transformer.encode != "ordinal" or transformer.strategy not in {"uniform", "quantile"}
                or not isinstance(transformer.n_bins, Integral) or isinstance(transformer.n_bins, bool)
                or transformer.n_bins < 2):
            raise DagMlUnsupported("fold-local target binning requires one ordinal uniform/quantile recipe with >= 2 bins")
        count = int(transformer.n_bins)
    else:
        raise DagMlUnsupported("fold-local targets require one direct RangeDiscretizer or ordinal KBinsDiscretizer")
    recipe = {
        "schema_version": 1,
        "operator": serialize_component(transformer),
        "sklearn_version": sklearn.__version__,
        "owner_version": __version__,
        "class_labels": [float(label) for label in range(count)],
    }
    # Validate recipe determinism and strict JSON before native execution.
    _strict_json(recipe)
    return recipe


def bin_domain_policy(transformer: Any, node_id: str) -> dict[str, Any]:
    """Build a native graph policy; constructing it does not enable execution."""
    recipe = bin_domain_recipe(transformer)
    return {
        "schema_version": 1,
        "mode": "fold_local_macro",
        "node_id": node_id,
        "recipe_fingerprint": hashlib.sha256(_strict_json(recipe).encode("utf-8")).hexdigest(),
        "class_labels": recipe["class_labels"],
    }


def fitted_bin_domain(transformer: Any, fit_sample_ids: list[str], policy: dict[str, Any]) -> dict[str, Any]:
    """Capture exact fitted boundaries and the training cohort for native lineage."""
    transformer = _unwrapped(transformer)
    recipe = bin_domain_recipe(transformer)
    expected = bin_domain_policy(transformer, policy["node_id"])
    if policy != expected:
        raise ValueError("fitted target recipe disagrees with the native target-domain policy")
    if not fit_sample_ids or len(set(fit_sample_ids)) != len(fit_sample_ids) or not all(isinstance(key, str) and key for key in fit_sample_ids):
        raise ValueError("target-domain fit cohort must contain nonempty unique sample IDs")
    if isinstance(transformer, RangeDiscretizer):
        if not hasattr(transformer, "_centers_"):
            raise ValueError("target-domain provenance requires a fitted RangeDiscretizer")
        boundaries = np.asarray(transformer._bins_array, dtype=float)
        actual_bins = transformer.n_bins
        state = {"cut_points": boundaries.tolist(), "intervals": "left_closed_right_open; unbounded outer intervals"}
    else:
        member = transformer.discretizer if isinstance(transformer, IntegerKBinsDiscretizer) else transformer
        if not hasattr(member, "bin_edges_") or len(member.bin_edges_) != 1:
            raise ValueError("target-domain provenance requires one fitted target feature")
        boundaries = np.asarray(member.bin_edges_[0], dtype=float)
        actual_bins = int(member.n_bins_[0])
        state = {"bin_edges": boundaries.tolist(), "intervals": "sklearn ordinal; clipped to outer bins"}
    # Collapsed/degenerate bins need their own native vocabulary contract. Never
    # claim that an absent learned interval has the declared recipe's meaning.
    if actual_bins != len(recipe["class_labels"]) or not np.isfinite(boundaries).all():
        raise DagMlUnsupported("fold-local target bins collapsed or have non-finite boundaries; an explicit fixed class domain is required")
    state_json = _strict_json({"schema_version": 1, "recipe": recipe, "fitted_state": state})
    return {
        "schema_version": 1,
        "recipe_fingerprint": policy["recipe_fingerprint"],
        "state_json": state_json,
        "state_fingerprint": hashlib.sha256(state_json.encode("utf-8")).hexdigest(),
        "fit_sample_ids": sorted(fit_sample_ids),
        "class_labels": recipe["class_labels"],
    }


def validate_captured_bin_domain(transformer: Any) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Verify the captured domain against its actual fitted operator state."""
    policy = getattr(transformer, "target_domain_policy_", None)
    domain = getattr(transformer, "target_domain_", None)
    if policy is None and domain is None:
        return None
    if not isinstance(policy, dict) or not isinstance(domain, dict):
        raise ValueError("captured target domain lost its native REFIT provenance")
    actual = fitted_bin_domain(transformer, domain.get("fit_sample_ids", []), policy)
    if domain != actual:
        raise ValueError("captured target domain disagrees with its fitted REFIT state")
    return policy, domain
