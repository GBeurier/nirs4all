"""Admission and configuration for joint Torch models with named input ports."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, cast

import numpy as np
from sklearn.base import clone

from .named_torch_estimator import DagMLNamedTorchEstimator
from .torch_estimator import DagMLTorchEstimator, torch_model_params


def named_task_seed(task: dict[str, Any]) -> int:
    """Derive the CPU callback seed from its native task identity."""
    from .tuning_contracts import tcv1_sha256

    node = task["node_plan"]
    return int(tcv1_sha256({"native_seed": task.get("seed"), "node_id": node["node_id"],
                           "variant_id": task.get("variant_id"), "fold_id": task.get("fold_id"),
                           "phase": task["phase"]})[:8], 16)


@contextmanager
def named_torch_task_scope(task: dict[str, Any], node: dict[str, Any]) -> Iterator[None]:
    """Seed only admitted named CPU callbacks, restoring the caller's RNG."""
    if (node.get("metadata") or {}).get("named_torch_profile") != "cpu_named_intermediate_regression_v1":
        yield
        return
    import torch

    if task["phase"] not in {"FIT_CV", "REFIT", "PREDICT"}:
        raise ValueError("Named Torch does not support this native callback phase.")
    resources = task.get("resources") or {}
    if resources.get("gpu_devices") or resources.get("cpu_threads") != 1:
        raise ValueError("Named Torch callbacks require serial CPU resources.")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(named_task_seed(task))
        with torch.device("cpu"):
            yield


def named_learned_state_sha256(estimator: Any) -> str:
    """Bind fitted tensor bytes and named input schemas to their REFIT origin.

    This computes an identity only. Capture and export must compare it with the
    identity recorded by the completed native REFIT, never replace that origin.
    """
    import torch

    from .torch_topology_replay import _array_identity
    from .tuning_contracts import tcv1_sha256

    if (type(estimator) is not DagMLNamedTorchEstimator or not isinstance(getattr(estimator, "model_", None), torch.nn.Module)
            or not all(hasattr(estimator, name) for name in ("source_names_", "input_shapes_", "input_dtypes_"))):
        raise ValueError("Named Torch requires the original fitted named estimator.")
    names = tuple(estimator.source_names_)
    if not 2 <= len(names) <= 4 or set(names) != set(estimator.input_shapes_) or set(names) != set(estimator.input_dtypes_):
        raise ValueError("Named Torch fitted input schemas are incomplete or changed.")
    state = {}
    for name, tensor in sorted(estimator.model_.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Named Torch learned state must contain only finite numeric tensors.")
        values = tensor.detach().cpu().numpy()
        # Preserve Boolean buffer types while using the canonical numeric bytes
        # helper shared with the existing fitted-state integrity contract.
        if values.dtype.kind == "b":
            values = values.astype(np.uint8)
        state[name] = {"torch_dtype": str(tensor.dtype), **_array_identity(values)}
    return tcv1_sha256({
        "schema_version": 1, "profile": "cpu_named_intermediate_regression_v1",
        "source_names": list(names),
        "input_shapes": {name: list(estimator.input_shapes_[name]) for name in names},
        "input_dtypes": {name: estimator.input_dtypes_[name] for name in names},
        "multimodal_input_schema": getattr(estimator, "multimodal_input_schema", None),
        "n_features_in": int(estimator.n_features_in_), "state": state,
    })


def is_named_torch_model(model: Any) -> bool:
    """Recognize intermediate Torch declarations without encoding a template."""
    if getattr(model, "fusion", None) != "intermediate" or getattr(model, "backend", None) != "sklearn":
        return False
    predictor = model.model
    return (isinstance(predictor, DagMLTorchEstimator) or getattr(predictor, "framework", None) == "pytorch"
            or isinstance(predictor, dict) and predictor.get("framework") == "pytorch"
            or any(base.__module__.startswith("torch.") for base in type(predictor).__mro__))


def named_torch_estimator(model: Any, *, train_params: Mapping[str, Any] | None = None) -> DagMLNamedTorchEstimator | None:
    """Configure the existing public intermediate wrapper for named Torch FIT.

    Return ``None`` for other model families. A Torch factory retains its public
    import path; a trusted module retains its initial parameters via the existing
    template encoding. Ordinary list-input intermediate models keep their API.
    """
    if not is_named_torch_model(model):
        return None
    predictor = model.model
    if isinstance(predictor, DagMLTorchEstimator):
        params = predictor.get_params(deep=False)
    else:
        configured = torch_model_params(predictor)
        if configured is None:
            return None
        params = configured
    transformers = model.transformers
    if not isinstance(transformers, Mapping) or not 2 <= len(transformers) <= 4:
        raise ValueError("Named Torch intermediate fusion requires two to four named sources.")
    if any(not isinstance(name, str) or not name.isascii() or not name.isidentifier() or name == "y" for name in transformers):
        raise ValueError("Named Torch source names must be ASCII Python identifiers other than y.")
    if any(value is not None and not (isinstance(value, str) and value == "passthrough") for value in transformers.values()):
        raise ValueError("Named Torch intermediate fusion currently requires passthrough source transformers.")
    if model.source_weights is not None:
        raise ValueError("Named Torch intermediate fusion currently requires source_weights=None.")
    if model.missing_source_policy != "error" or getattr(model, "target_policy", None) != "complete":
        raise ValueError("Named Torch intermediate fusion requires complete sources and one complete target.")
    controls = dict(train_params or {})
    defaults = {"task_type": "regression", "device": "cpu", "force_layout": "2d"}
    # Keep explicitly configured incompatible controls visible to the boundary
    # validator instead of replacing them with a supported value.
    for key, value in defaults.items():
        if params.get(key) is None:
            params[key] = value
    allowed = set(DagMLNamedTorchEstimator().get_params(deep=False))
    if set(controls) - allowed:
        raise ValueError(f"Unsupported named Torch training controls: {sorted(set(controls) - allowed)}.")
    params.update(controls)
    estimator = cast(DagMLNamedTorchEstimator, clone(DagMLNamedTorchEstimator(**params)))
    estimator.validate_configuration()
    return estimator


def prepare_named_torch_pipeline(pipeline: Any, dataset: Any) -> Any:
    """Bind one public intermediate model to its complete IO source identities.

    The returned step declares native named ports. Generic pipelines retain
    their original objects; no Torch model or optimizer is constructed here.
    The first named profile accepts a model plus an optional index/group fold
    splitter. Preprocessing belongs to the user's joint encoders.
    """
    from nirs4all.data.multimodal import MultimodalSpectroDataset
    from nirs4all.operators.models.multimodal import MultimodalRegressor

    from .envelope import source_ids
    from .steps import _is_split_step
    from .tuning_contracts import tcv1_sha256

    if not isinstance(pipeline, list):
        return pipeline
    declarations = []
    for index, step in enumerate(pipeline):
        operator = step.get("model") if isinstance(step, dict) else step
        if isinstance(operator, MultimodalRegressor) and is_named_torch_model(operator):
            declarations.append((index, step, operator))
    if not declarations:
        return pipeline
    if len(declarations) != 1 or any(index != declarations[0][0] and not _is_split_step(step) for index, step in enumerate(pipeline)):
        raise ValueError("Named Torch currently requires one intermediate model and an optional index/group splitter.")
    if not isinstance(dataset, MultimodalSpectroDataset):
        raise ValueError("Named Torch intermediate fusion requires an IO MultimodalDataset with named sources.")
    cohort = dataset.cohort
    if (cohort.y is None or cohort.task_type not in (None, "regression")
            or np.asarray(cohort.y).shape not in ((len(cohort.sample_ids),), (len(cohort.sample_ids), 1))
            or not np.asarray(cohort.target_mask).all()):
        raise ValueError("Named Torch native training requires one complete regression target.")
    index, raw, operator = declarations[0]
    step = dict(raw) if isinstance(raw, dict) else {"model": raw}
    estimator = named_torch_estimator(operator, train_params=step.get("train_params"))
    assert estimator is not None
    names = tuple(operator.transformers)
    if set(names) != set(dataset.source_names):
        raise ValueError("Named Torch ports must cover exactly the cohort source names.")
    buffers = {}
    native_ids = dict(zip(dataset.source_names, source_ids(dataset), strict=True))
    for name in names:
        source = cohort.sources[name]
        if (source.representation_id not in {"tabular_numeric", "signal_1d", "gray_image", "rgb_image", "mc_image", "multispectral_image", "series_mv"}
                or not np.asarray(source.presence_mask).all()):
            raise ValueError(f"Named Torch source {name!r} requires a complete fixed numeric table, signal, image or series tensor.")
        buffers[name] = source.values
    estimator._named_features(buffers, fitting=True)
    targets = np.asarray(cohort.y)
    if targets.dtype.kind not in "fiu":
        raise ValueError("Named Torch native training requires a numeric regression target.")
    with np.errstate(over="ignore", invalid="ignore"):
        effective_targets = targets.astype(np.float32)
    if not np.isfinite(effective_targets).all():
        raise ValueError("Named Torch targets must remain finite in the effective float32 training dtype.")
    specification = {
        "schema_version": 1,
        "ports": [{"name": name, "accepted_representations": [cohort.sources[name].representation_id],
                   "accepted_types": [cohort.sources[name].descriptor(name)["type_id"]],
                   "rank": np.asarray(buffers[name]).ndim, "multi_source": False, "optional": False,
                   "metadata": {"source_id": native_ids[name], "dtype": str(np.asarray(buffers[name]).dtype),
                                "feature_shape": list(np.asarray(buffers[name]).shape[1:])}} for name in names],
        "default_fusion": None,
        "metadata": {},
    }
    step["model"] = estimator
    step["model_input"] = specification
    step["metadata"] = {**(step.get("metadata") or {}),
                        "controller_id": "controller:nirs4all.named_torch." + tcv1_sha256(specification)[:16],
                        "named_torch_profile": "cpu_named_intermediate_regression_v1"}
    # Controls are already encoded on the clone. Keeping a second carrier would
    # let later generic normalization reinterpret or override the admitted model.
    step.pop("train_params", None)
    result = list(pipeline)
    result[index] = step
    return result
