"""Target-output transforms attached only to captured host artifacts.

Training and native scoring operate in the model's declared numeric target space.
Exported predictors additionally need to restore the labels supplied by the
user. Keeping that decoder outside the runtime ``y_processing`` transform
prevents decoded labels from leaking back into DAG score blocks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CapturedTargetTransform(TransformerMixin, BaseEstimator):
    """Compose an optional model-target transform with a fitted label decoder."""

    def __init__(self, transformer: Any, decoder: Any) -> None:
        self.transformer = transformer
        self.decoder = decoder

    def fit(self, y: Any, *_args: Any, **_kwargs: Any) -> CapturedTargetTransform:
        if self.transformer is not None:
            self.transformer.fit(y)
        return self

    def fit_transform(self, y: Any, *_args: Any, **_kwargs: Any) -> Any:
        if self.transformer is None:
            return np.asarray(y)
        return self.transformer.fit_transform(y)

    def transform(self, y: Any) -> Any:
        if self.transformer is None:
            return np.asarray(y)
        return self.transformer.transform(y)

    def inverse_numeric(self, y: Any) -> Any:
        """Restore numeric target space without decoding labels."""
        if self.transformer is None:
            return np.asarray(y)
        return self.transformer.inverse_transform(y)

    def decode(self, y: Any) -> Any:
        """Decode numeric targets to the labels supplied by the user."""
        return self.decoder.inverse_transform(y)

    def inverse_transform(self, y: Any) -> Any:
        return self.decode(self.inverse_numeric(y))

    @property
    def classes_(self) -> Any:
        """Expose the captured public class axis when the decoder has one."""
        classes = getattr(self.decoder, "classes_", None)
        if classes is not None:
            return classes
        columns = getattr(self.decoder, "column_transformers", None)
        if isinstance(columns, dict) and len(columns) == 1:
            member = next(iter(columns.values()))
            classes = getattr(member, "classes_", None)
            if classes is not None:
                return classes
        raise AttributeError("captured target decoder has no single class axis")


def _predicts_encoded_classes(estimator: Any) -> bool:
    """Whether a fitted estimator's ``predict`` output is a class index."""
    from sklearn.base import is_classifier

    candidates = [estimator]
    seen: set[int] = set()
    while candidates:
        candidate = candidates.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        try:
            if is_classifier(candidate):
                return True
        except (AttributeError, TypeError):
            # SDK wrappers need not implement sklearn tags; inspect their model.
            pass
        candidates.extend(getattr(candidate, name, None) for name in ("model", "_model"))
    return False


def captured_target_transform(transformer: Any, decoder: Any, estimator: Any) -> Any:
    """Attach public label decoding only to estimators that emit class indices."""
    if (decoder is None or isinstance(transformer, (CapturedTargetTransform, CategoricalTargetTransform))
            or not _predicts_encoded_classes(estimator)):
        return transformer
    return CapturedTargetTransform(transformer, decoder)


class CategoricalTargetTransform(TransformerMixin, BaseEstimator):
    """A fitted target discretizer whose prediction domain is its class IDs.

    The learned transform remains available for fold-local score targets and
    replay, while class predictions never inverse-transform to bin centers.
    """

    def __init__(self, transformer: Any) -> None:
        self.transformer = transformer
        self.target_domain_policy_: dict[str, Any] | None = None
        self.target_domain_: dict[str, Any] | None = None
        self.input_transformer_: Any = None

    def fit(self, y: Any, *_args: Any, **_kwargs: Any) -> CategoricalTargetTransform:
        self.transformer.fit(y)
        return self

    def fit_transform(self, y: Any, *_args: Any, **_kwargs: Any) -> Any:
        return self.transformer.fit_transform(y)

    def transform(self, y: Any) -> Any:
        return self.transformer.transform(y)

    def inverse_transform(self, y: Any) -> np.ndarray:
        """Keep predictions in the categorical output domain."""
        return np.asarray(y)


def has_categorical_target_output(transformer: Any) -> bool:
    """Recognize owner-defined ordinal binning, including composed target chains."""
    from sklearn.preprocessing import KBinsDiscretizer

    from nirs4all.operators.transforms.targets import IntegerKBinsDiscretizer, RangeDiscretizer

    if isinstance(transformer, CategoricalTargetTransform):
        return True
    if isinstance(transformer, RangeDiscretizer):
        return True
    if isinstance(transformer, (KBinsDiscretizer, IntegerKBinsDiscretizer)):
        return bool(transformer.encode == "ordinal")
    steps = getattr(transformer, "steps", None)
    if isinstance(steps, (list, tuple)) and steps:
        final = steps[-1]
        return has_categorical_target_output(final[1] if isinstance(final, tuple) else final)
    return False


def _contains_target_binning(operator: Any) -> bool:
    from nirs4all.pipeline.config.component_serialization import deserialize_component

    if isinstance(operator, (dict, str)):
        operator = deserialize_component(operator)
    if isinstance(operator, type):
        operator = operator()
    if isinstance(operator, list):
        return any(_contains_target_binning(member) for member in operator)
    steps = getattr(operator, "steps", None)
    if isinstance(steps, (list, tuple)):
        return any(_contains_target_binning(member[1] if isinstance(member, tuple) else member) for member in steps)
    return has_categorical_target_output(operator)


def _executable_steps(value: Any) -> Any:
    """Visit executable branch/list bodies without descending into operator parameters."""
    if isinstance(value, list):
        for member in value:
            yield from _executable_steps(member)
    elif isinstance(value, dict):
        if any(key in value for key in ("model", "y_processing", "preprocessing", "branch", "merge")):
            yield value
            if "branch" in value:
                yield from _executable_steps(value["branch"])
        elif "class" in value:
            from nirs4all.pipeline.config.component_serialization import deserialize_component

            component = deserialize_component(value)
            if callable(getattr(component, "fit", None)) and callable(getattr(component, "predict", None)):
                yield {"model": component}
        else:
            for member in value.values():
                yield from _executable_steps(member)
    elif callable(getattr(value, "fit", None)) and callable(getattr(value, "predict", None)):
        yield {"model": value}


def pipeline_task_type(pipeline: list[Any], fallback: str) -> str:
    """Resolve a homogeneous model task without mutating dataset targets.

    Public successive checkpoints are dispatched separately before this point.
    A single native graph has one task and cannot silently mix model tasks.
    """
    from sklearn.base import is_regressor

    from nirs4all.pipeline.config.component_serialization import deserialize_component

    from .errors import DagMlUnsupported
    from .public_normalization import normalize_model_steps

    if "classif" in fallback.lower():
        fallback = "classification"
    elif "regression" in fallback.lower():
        fallback = "regression"
    tasks = set()
    unknown_tasks = False
    for step in _executable_steps(normalize_model_steps(pipeline)):
        if "model" not in step:
            continue
        model = step["model"]
        if isinstance(model, (str, dict)):
            model = deserialize_component(model)
        if isinstance(model, type):
            try:
                model = model()
            except TypeError:
                tasks.add(fallback)
                unknown_tasks = True
                continue
        if _predicts_encoded_classes(model):
            tasks.add("classification")
        else:
            candidates, seen = [model], set()
            resolved = fallback
            identified = False
            while candidates:
                candidate = candidates.pop()
                if candidate is None or id(candidate) in seen:
                    continue
                seen.add(id(candidate))
                try:
                    if is_regressor(candidate):
                        resolved = "regression"
                        identified = True
                        break
                except (AttributeError, TypeError):
                    pass
                candidates.extend(getattr(candidate, name, None) for name in ("model", "_model"))
            tasks.add(resolved)
            unknown_tasks |= not identified
    if (unknown_tasks or not tasks) and any(
        _contains_target_binning(step["y_processing"])
        for step in _executable_steps(pipeline) if "y_processing" in step
    ):
        raise DagMlUnsupported("Declare a recognized classifier or regressor task before target binning; dataset task fallback cannot define the model's class domain")
    if len(tasks) > 1:
        raise DagMlUnsupported(
            "One native graph cannot mix classification and regression target domains. "
            "Run these models as separate pipelines; top-level successive checkpoints are dispatched separately."
        )
    return next(iter(tasks), fallback)


def validate_classification_target_domains(pipeline: list[Any], *, refit: bool = True, refit_top_k: int = 1) -> None:
    """Refuse pooled CV when learned bin maps can change class meaning by fold.

    One full-training fit has one domain and remains supported. Fixed explicit
    cut points also define one domain across CV folds. Learned fold maps require
    a native target-domain-aware selection/aggregation contract.
    """
    from numbers import Integral

    from nirs4all.operators.transforms.targets import RangeDiscretizer
    from nirs4all.pipeline.config.component_serialization import deserialize_component

    from .errors import DagMlUnsupported
    from .steps import _split_pipeline

    _, splitter = _split_pipeline(pipeline)
    target_steps = [step["y_processing"] for step in _executable_steps(pipeline) if "y_processing" in step]

    for configured in target_steps:
        transformer = configured
        if isinstance(transformer, (dict, str)):
            transformer = deserialize_component(transformer)
        if isinstance(transformer, type):
            transformer = transformer()
        if not _contains_target_binning(transformer):
            continue
        top_level = any(isinstance(step, dict) and step.get("y_processing") is configured for step in pipeline)
        if len(target_steps) != 1 or not top_level or not has_categorical_target_output(transformer):
            raise DagMlUnsupported(
                "Classification target binning requires one top-level y_processing transform with categorical output; "
                "branch-local, repeated and post-binning continuous target transforms do not declare one class domain."
            )
        fixed_domain = (len(target_steps) == 1 and isinstance(transformer, RangeDiscretizer)
                and not isinstance(transformer.bins, Integral)
                and top_level)
        if fixed_domain:
            continue
        if splitter is None:
            from .detect import _generation_kind

            if _generation_kind(pipeline) != "none":
                raise DagMlUnsupported("learned target bins without CV require one classifier fit; model comparisons require an explicitly fixed class domain")
            continue
        from sklearn.model_selection import KFold

        from .target_domains import fold_local_domains_available, learned_bin_transform

        split_operator = getattr(splitter, "splitter", splitter)
        if (learned_bin_transform(pipeline) is not None and isinstance(split_operator, KFold)
                and refit and refit_top_k == 1 and fold_local_domains_available()):
            continue
        raise DagMlUnsupported(
            "Classification cross-validation with learned target-bin boundaries requires "
            "partition-scoped target domains: fitting bins independently on each training fold "
            "changes class meaning, so pooled/ensemble scores cannot share one ground truth. "
            "Use explicitly declared fixed RangeDiscretizer cut points, or a single full-training "
            "fit without cross-validation. No target bins have been fitted on validation or test rows."
        )
