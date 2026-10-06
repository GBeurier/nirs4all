"""Declarations and label transport for native Methods multimodal classifiers.

All encoders, PLS-logistic fitting, probability calculation, grouped folds and
metrics remain in Methods/DAG-ML. This module only signs and decodes identities.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from nirs4all.operators.models.multimodal import MultimodalClassifier

PROFILE = "dagml.methods.multimodal.classification.v1"
RAW_TYPE = "N4mMultimodalClassifierPipeline"
META_TYPE = "N4mRoleClassifierPipeline"
RAW_CONTROLLER = "controller:methods.python.multimodal.classification"
META_CONTROLLER = "controller:methods.python.classification"
HEAD = "models.classification.pls_logistic"
MAX_CLASSES = 65536
MAX_LABEL_UTF8_BYTES = 1 << 20


def classifier_head(model: Any) -> dict[str, Any]:
    """Read one genuine native PLS-logistic declaration without fitting it."""
    from n4m.roles import PLSLogistic, RolePipeline

    if type(model) is PLSLogistic:
        params = model.get_params(deep=False)
    elif type(model) is RolePipeline and isinstance(model.steps, (list, tuple)) and len(model.steps) == 1:
        step = model.steps[0]
        if isinstance(step, tuple) and len(step) == 2:
            method, params = step
        elif isinstance(step, Mapping) and set(step) == {"class", "params"}:
            method, params = step["class"], step["params"]
        else:
            raise ValueError("native classifier RolePipeline requires one explicitly parameterized PLS-logistic step")
        if not isinstance(method, str) or method.removeprefix("n4m:") != HEAD:
            raise ValueError("native classification requires the genuine Methods PLS-logistic head")
    else:
        raise ValueError("native classification requires n4m.roles.PLSLogistic or its singleton RolePipeline")
    if not isinstance(params, Mapping) or set(params) != {"n_components", "max_iter"}:
        raise ValueError("PLS-logistic requires exactly n_components and max_iter")
    if any(type(value) is not int or not 1 <= value <= (1 << 31) - 1 for value in params.values()):
        raise ValueError("PLS-logistic counts must be positive int32 integers")
    return {"method_id": HEAD, "params": dict(params)}


def classifier_recipe(model: Any, *, allow_source_selection: bool = False) -> dict[str, Any]:
    """Reuse raw source declarations with the distinct native classifier head."""
    from .methods_multimodal import _finite_nonnegative, _source_encoder_recipe

    if type(model) is not MultimodalClassifier or model.backend != "methods":
        raise ValueError("native classification requires MultimodalClassifier(backend='methods')")
    selected, encoders = _source_encoder_recipe(model, allow_source_selection=allow_source_selection)
    head = classifier_head(model.model)
    weights = {} if model.source_weights is None else model.source_weights
    if not isinstance(weights, Mapping) or set(weights) - set(selected):
        raise ValueError("source_weights must name only selected sources")
    return {"schema_version": 1, "fusion": "early", "source_order": list(selected), "encoders": encoders,
            "source_weights": {name: _finite_nonnegative(weights.get(name, 1.0), f"source_weights.{name}") for name in selected},
            "model": head}


def _typed_labels(values: Any) -> list[str] | list[int]:
    array = np.asarray(values, dtype=object)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or not len(array):
        raise ValueError("Methods classification requires one complete nonempty label column")
    labels = array.tolist()
    if all(isinstance(value, str) for value in labels):
        if any(len(value.encode("utf-8")) > MAX_LABEL_UTF8_BYTES for value in labels):
            raise ValueError("classification label exceeds the closed native UTF-8 byte budget")
        return cast(list[str], labels)
    if all(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
           and -(1 << 63) <= int(value) < (1 << 63) for value in labels):
        return [int(value) for value in labels]
    raise ValueError("Methods classification labels must be homogeneous strings or int64; mixed, float and missing labels are unsupported")


def classification_vocabulary(values: Any, train_rows: list[int]) -> dict[str, Any]:
    """Declare a sorted typed vocabulary using training rows alone."""
    labels = _typed_labels(values)
    if not train_rows or len(set(train_rows)) != len(train_rows) or any(type(row) is not int or not 0 <= row < len(labels) for row in train_rows):
        raise ValueError("classification vocabulary requires distinct valid training rows")
    names = sorted({labels[row] for row in train_rows})
    if not 2 <= len(names) <= MAX_CLASSES:
        raise ValueError("Methods classification requires two to 65536 training classes within its native state budget")
    return {"schema_version": 1, "class_labels": list(range(len(names))), "label_names": names}


def checked_vocabulary(value: Any) -> dict[str, Any]:
    """Attest the exact signed contiguous-ID/original-label correspondence."""
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "class_labels", "label_names"} or type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("invalid signed classification vocabulary")
    names = _typed_labels(value["label_names"])
    ids = value["class_labels"]
    if (not isinstance(ids, list) or any(type(label) is not int for label in ids)
            or ids != list(range(len(names))) or not 2 <= len(names) <= MAX_CLASSES or names != sorted(set(cast(list[Any], names)))):
        raise ValueError("classification vocabulary must preserve sorted unique typed names and contiguous IDs")
    return {"schema_version": 1, "class_labels": list(ids), "label_names": names}


def encode_labels(values: Any, vocabulary: Any) -> np.ndarray:
    """Encode target identities; a heldout-only or differently typed label fails."""
    signed = checked_vocabulary(vocabulary)
    names = signed["label_names"]
    labels = _typed_labels(values)
    if isinstance(labels[0], str) != isinstance(names[0], str):
        raise ValueError("target label type differs from the signed training vocabulary")
    positions = {name: index for index, name in enumerate(names)}
    if any(label not in positions for label in labels):
        raise ValueError("target label is absent from the training-only classification vocabulary")
    return np.asarray([positions[label] for label in labels], dtype=np.int64)


def decode_labels(values: Any, vocabulary: Any) -> np.ndarray:
    """Decode only exact native class IDs, preserving original scalar types."""
    signed = checked_vocabulary(vocabulary)
    ids = np.asarray(values)
    if ids.dtype.kind not in "fiu" or not np.isfinite(ids).all() or np.any(ids != np.floor(ids)) or np.any(ids < 0) or np.any(ids >= len(signed["class_labels"])):
        raise ValueError("native classification predictions contain invalid or out-of-vocabulary IDs")
    names = np.asarray(signed["label_names"], dtype=object if isinstance(signed["label_names"][0], str) else np.int64)
    return names[ids.astype(np.int64)]


def graph_vocabulary(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Require one common signed vocabulary on all classifier producers."""
    models = [node for node in graph["nodes"] if node["kind"] == "model"]
    if not models or any((node.get("operator") or {}).get("type") not in {RAW_TYPE, META_TYPE} for node in models):
        raise ValueError("Methods classification graph requires only its exact native classifier types")
    vocabulary = checked_vocabulary(models[0]["operator"].get("classification"))
    if any(checked_vocabulary(node["operator"].get("classification")) != vocabulary for node in models):
        raise ValueError("native classifier producers disagree on the signed class vocabulary")
    return vocabulary


def classifier_controller_for_graph(graph: Mapping[str, Any], cohort: Any, *, allow_fit: bool,
                                    node_params: Any = None, binding_source_ids: Any = None,
                                    binding_controller_id: str | None = None) -> Any:
    """Transport encoded identities to native owners with the signed graph edges."""
    from dag_ml.multimodal_classification import ClassificationTopologyController

    from nirs4all.data.multimodal import MultimodalSpectroDataset

    from .envelope import source_ids
    from .methods_multimodal import controller_sources, source_schemas_from_cohort

    vocabulary = graph_vocabulary(graph)
    schemas = source_schemas_from_cohort(cohort)
    models = [node for node in graph["nodes"] if node["kind"] == "model"]
    raw = [node for node in models if node["operator"]["type"] == RAW_TYPE]
    if not raw or any(node["operator"].get("source_schemas") != schemas for node in raw):
        raise ValueError("classification raw source schemas differ from the signed declarations")
    current_ids = tuple(source_ids(MultimodalSpectroDataset(cohort)))
    if binding_source_ids is not None and tuple(binding_source_ids) != current_ids:
        raise ValueError("classification source order differs from the signed raw binding")
    if binding_controller_id is not None and binding_controller_id != RAW_CONTROLLER:
        raise ValueError("classification replay requires the exact Python raw producer owner")
    if any(node.get("metadata", {}).get("controller_id") != (RAW_CONTROLLER if node in raw else META_CONTROLLER) for node in models):
        raise ValueError("classification producer owner differs from its signed declaration")
    return ClassificationTopologyController(
        operators={node["id"]: node["operator"] for node in models}, sources=controller_sources(cohort, schemas),
        targets={"sample_ids": list(cohort.sample_ids), "values": encode_labels(cohort.y, vocabulary)} if allow_fit else None,
        target_names=("y",), allow_fit=allow_fit, source_ids=current_ids, node_params=node_params, edges=graph["edges"],
    )


def fit_declared_methods_classifier(model: Any, blocks: list[Any], y: Any, *, source_schemas: Any) -> None:
    """Fit genuine native state; a failed replacement leaves the old state intact."""
    from .methods_multimodal import methods_input_blocks

    recipe = classifier_recipe(model, allow_source_selection=True)
    if not isinstance(source_schemas, Mapping):
        raise ValueError("Methods classifier fit requires explicit IO-derived source_schemas")
    names = tuple(recipe["source_order"])
    values = model._validate_blocks(blocks, names)
    targets = _typed_labels(y)
    if len(targets) != len(values[0]):
        raise ValueError("classifier raw rows and target labels must align")
    vocabulary = classification_vocabulary(targets, list(range(len(targets))))
    pipeline_type = getattr(importlib.import_module("n4m"), "MultimodalClassifierPipeline", None)
    if not callable(pipeline_type):
        raise ImportError("installed Methods lacks MultimodalClassifierPipeline (ABI 2.17 required)")
    schemas = {name: copy.deepcopy(source_schemas[name]) for name in names}
    native = pipeline_type(recipe, schemas)
    previous = model.__dict__.get("native_pipeline_")
    try:
        native.fit(methods_input_blocks(dict(zip(names, values, strict=True)), schemas), np.asarray(targets))
        if list(native.classes_) != vocabulary["label_names"]:
            raise ValueError("native classifier class order differs from the declared typed training vocabulary")
        model.__dict__ = {**model.__dict__, "native_pipeline_": native, "source_schemas_": schemas,
                          "source_names_": names, "input_shapes_": {name: tuple(np.shape(block)[1:]) for name, block in zip(names, values, strict=True)},
                          "classes_": np.asarray(vocabulary["label_names"]), "classification_": vocabulary}
    except BaseException:
        native.close()
        raise
    if previous is not None:
        previous.close()
