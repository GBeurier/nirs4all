"""Input and dependency contracts for captured multimodal Python predictors."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
from typing import Any, cast

from nirs4all.data.multimodal import MultimodalSpectroDataset


def bind_input_contract(estimator: Any, dataset: Any, source_index: int | None = None) -> None:
    """Record semantic source contracts alongside fitted host state."""
    if isinstance(dataset, MultimodalSpectroDataset):
        estimator.multimodal_input_schema = {
            descriptor["source_id"]: descriptor for descriptor in dataset.cohort.schema_descriptors()
        }
        evidence = getattr(dataset, "_data_provider_evidence", None)
        if evidence is not None:
            estimator.data_provider_evidence = copy.deepcopy(evidence)
        if source_index is not None:
            estimator.multimodal_source_name = dataset.source_names[source_index]


def bind_dense_concat_input_contract(estimator: Any, source_layout: dict[str, Any]) -> None:
    """Retain the original dense source layout with a fitted source projection."""
    estimator.dense_concat_input_layout = copy.deepcopy(source_layout)


def _validate_dense_concat_input_contract(estimator: Any, dataset: Any) -> None:
    """Accept either the declared raw blocks or their full ordered flat buffer."""
    layout = getattr(estimator, "dense_concat_input_layout", None)
    if layout is None:
        return
    if not isinstance(layout, dict) or layout.get("kind") != "by_source_concat":
        raise ValueError("invalid captured dense source input layout")
    blocks = layout.get("blocks")
    names = layout.get("source_order")
    if not isinstance(blocks, list) or not blocks or not isinstance(names, list) or len(names) != len(blocks):
        raise ValueError("invalid captured dense source input layout")
    raw_widths = [block.get("column_count") if isinstance(block, dict) else None for block in blocks]
    if any(type(width) is not int or width < 1 for width in raw_widths):
        raise ValueError("invalid captured dense source input widths")
    widths = cast(list[int], raw_widths)
    count = dataset.features_sources()
    actual_widths = dataset.num_features
    actual_widths = actual_widths if isinstance(actual_widths, list) else [actual_widths]
    # A flat matrix cannot express separate source boundaries; its columns must
    # already follow the complete original order retained in the archive.
    expected_widths = [sum(widths)] if count == 1 else widths
    if actual_widths != expected_widths or count != len(expected_widths):
        raise ValueError("dense source input layout mismatch: source counts or widths changed")
    if any(dataset.features_processings(index) != ["raw"] for index in range(count)):
        raise ValueError("dense source input layout requires raw features only")
    if count > 1:
        actual_names = [dataset.source_name(index) if hasattr(dataset, "source_name") else f"source_{index}" for index in range(count)]
        if actual_names != names:
            raise ValueError("dense source input layout mismatch: source names or order changed")


def validate_input_contract(estimator: Any, dataset: Any) -> None:
    """Reject shape, axis, unit or feature-schema drift before prediction."""
    _validate_dense_concat_input_contract(estimator, dataset)
    expected = getattr(estimator, "multimodal_input_schema", None)
    if expected is None:
        return
    if not isinstance(dataset, MultimodalSpectroDataset):
        raise ValueError("this multimodal archive requires a MultimodalDataset with named raw sources")
    actual = {descriptor["source_id"]: descriptor for descriptor in dataset.cohort.schema_descriptors()}
    if set(actual) != set(expected):
        raise ValueError(f"multimodal source names mismatch: required {sorted(expected)}, received {sorted(actual)}")
    for name in expected:
        if actual[name] != expected[name]:
            raise ValueError(f"multimodal input schema mismatch for source {name!r}: shape, axes, units, coordinates, dtype or feature names changed")


def generated_prediction_contract(estimator: Any) -> dict[str, Any] | None:
    """Qualify a captured generated-view model for explicit-cohort prediction."""
    from sklearn.base import is_regressor

    from nirs4all.operators.models.multimodal import MultimodalClassifier, MultimodalRegressor

    from .named_torch_estimator import DagMLNamedTorchEstimator
    from .node_runner import _SourceConcatEstimator

    named = isinstance(estimator, DagMLNamedTorchEstimator)
    model = getattr(estimator, "_model", estimator)
    schema = getattr(estimator, "multimodal_input_schema", None)
    names = getattr(estimator, "source_names_", None) if named else getattr(estimator, "source_names", None)
    source_concat = isinstance(estimator, _SourceConcatEstimator)
    if source_concat and (
        not isinstance(names, (list, tuple))
        or len(estimator._source_chains) != len(names)
        or len(getattr(estimator, "_source_widths", ())) != len(names)
        or not is_regressor(model)
        or not callable(getattr(model, "predict", None))
    ):
        return None
    if (not (isinstance(model, (MultimodalRegressor, MultimodalClassifier)) or source_concat or named)
            or not isinstance(schema, dict) or not schema
            or not isinstance(names, (list, tuple)) or set(names) != set(schema)
            or len(names) != len(set(names))
            or not all(isinstance(name, str) and name for name in names)):
        return None
    if source_concat and list(names) != list(schema):
        return None
    try:
        schema_sha256 = generated_input_schema_sha256(schema)
    except (TypeError, ValueError):
        return None
    return {
        "schema_version": 1,
        "mode": "explicit_cohort_predict_only",
        "source_order": list(names),
        "input_schema": copy.deepcopy(schema),
        "input_schema_sha256": schema_sha256,
    }


def generated_input_schema_sha256(schema: dict[str, Any]) -> str:
    """Hash the explicit prediction input schema before any model load."""
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _late_learned_state_sha256(estimator: Any) -> str:
    """Fingerprint fitted values, preserving identity across archive round trips."""
    from joblib.hashing import NumpyHasher

    class LearnedStateHasher(NumpyHasher):
        def save(self, obj: Any) -> None:
            # Joblib reloads sliced/PCA arrays with compact strides. Layout and
            # pickle memo references are storage details, not learned values.
            # Keep array type, dtype, shape and every value in the SHA-256.
            if isinstance(obj, self.np.ndarray) and not obj.dtype.hasobject:
                obj = obj.copy(order="C")
            super().save(obj)

    state = {key: value for key, value in vars(estimator).items()
             if key not in {"_nirs4all_late_partial_refit_origin", "_nirs4all_late_partial_refit_fingerprint"}}
    return cast(str, LearnedStateHasher(hash_name="sha256").hash({
        "type": f"{type(estimator).__module__}.{type(estimator).__qualname__}", "state": state,
    }))


def _late_class_labels(bundle: dict[str, Any]) -> list[Any] | None:
    """Read the retained public decoder vocabulary without fitting it."""
    from .target_capture import captured_target_transform

    decoder = captured_target_transform(bundle.get("y_transform"), bundle.get("target_decoder"), bundle["estimator"])
    labels = getattr(decoder, "classes_", None) if decoder is not None else None
    if labels is None:
        labels = getattr(bundle["estimator"], "classes_", None)
    if labels is None:
        return None
    return labels.tolist() if hasattr(labels, "tolist") else list(labels)


def emit_late_partial_refit_origin(
    task: dict[str, Any], estimator: Any, *, graph_node: dict[str, Any], dataset: Any,
    fit_sample_ids: list[str], target_names: list[str], source_name: str | None,
    availability: dict[str, Any], target_decoder: Any = None,
) -> dict[str, Any]:
    """Seal one completed native REFIT component, including its exact validity."""
    from .tuning_contracts import tcv1_sha256

    node = task["node_plan"]
    if (task["phase"] != "REFIT" or task.get("fold_id") is not None
            or not isinstance(dataset, MultimodalSpectroDataset)
            or hasattr(estimator, "_nirs4all_late_partial_refit_origin")
            or graph_node.get("id") != node["node_id"]
            or not fit_sample_ids or len(fit_sample_ids) != len(set(fit_sample_ids))):
        raise ValueError("late partial provenance requires a completed genuine native REFIT")
    source_order = list(dataset.source_names)
    schema = getattr(estimator, "multimodal_input_schema", None)
    ids = availability.get("sample_ids")
    masks = availability.get("target_validity_masks")
    presence = availability.get("source_presence")
    actual_schema = {descriptor["source_id"]: descriptor for descriptor in dataset.cohort.schema_descriptors()}
    if (not isinstance(schema, dict) or schema != actual_schema or set(schema) != set(source_order)
            or availability.get("schema_version") != 1 or availability.get("target_names") != target_names
            or not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
            or not isinstance(presence, dict) or set(presence) != set(source_order)
            or any(not isinstance(values, list) or len(values) != len(ids) or any(type(value) is not bool for value in values) for values in presence.values())
            or not isinstance(masks, list) or len(masks) != len(ids)
            or any(not isinstance(row, list) or len(row) != len(target_names) or any(type(value) is not bool for value in row) for row in masks)
            or not set(fit_sample_ids).issubset(ids)
            or source_name is not None and source_name not in source_order):
        raise ValueError("late partial REFIT disagrees with its complete IO schema or native availability")
    missing_policy = getattr(estimator, "multimodal_missing_source_policy", None)
    if (list(getattr(estimator, "multimodal_source_order", ())) != source_order
            or list(getattr(estimator, "multimodal_target_names", ())) != target_names
            or missing_policy not in {"error", "zero_with_indicator"}
            or missing_policy == "error" and any(not all(values) for values in presence.values())
            or getattr(estimator, "multimodal_source_name", None) != source_name):
        raise ValueError("late partial fitted component changed its source, target or missing-source policy")
    positions = {sample_id: index for index, sample_id in enumerate(ids)}
    labels = availability.get("class_labels")
    classification = labels is not None
    target_policy = getattr(estimator, "multimodal_target_policy", None)
    target_fit_ids: dict[str, object]
    if classification:
        actual_classes = getattr(estimator, "classes_", None)
        actual_classes = actual_classes.tolist() if actual_classes is not None and hasattr(actual_classes, "tolist") else actual_classes
        if (len(target_names) != 1 or target_policy != "complete" or actual_classes != labels
                or any(not row[0] for row in masks)):
            raise ValueError("late partial classification REFIT requires its complete mono-y vocabulary")
        target_fit_ids = {target_names[0]: list(fit_sample_ids)}
    else:
        declared_target_fit_ids = getattr(estimator, "multimodal_target_fit_sample_ids", None)
        if (target_policy not in {"complete", "per_target"}
                or target_policy == "complete" and any(not all(row) for row in masks)
                or not isinstance(declared_target_fit_ids, dict) or set(declared_target_fit_ids) != set(target_names)):
            raise ValueError("late partial regression REFIT requires its declared target policy and fitted sample IDs")
        target_fit_ids = declared_target_fit_ids
    for index, name in enumerate(target_names):
        expected = [sample_id for sample_id in fit_sample_ids if masks[positions[sample_id]][index]
                    and (source_name is None or presence[source_name][positions[sample_id]])]
        if not expected or target_fit_ids[name] != expected:
            raise ValueError("late partial REFIT target scope differs from source-presence and observed-label intersections")
    origin = {
        "schema_version": 1, "profile": "late_partial_v1", "phase": "REFIT", "fold_id": None,
        "artifact_id": f"artifact:{node['node_id']}:nirs4all:refit:{task.get('variant_id') or 'base'}",
        "run_id": task["run_id"], "node_id": node["node_id"], "controller_id": node["controller_id"],
        "variant_id": task.get("variant_id"), "native_seed": task.get("seed"), "native_params": copy.deepcopy(node["params"]),
        "graph_node": copy.deepcopy(graph_node), "source_order": source_order, "source_name": source_name,
        "target_names": list(target_names), "target_policy": target_policy, "missing_source_policy": missing_policy,
        "multimodal_input_schema": copy.deepcopy(schema), "availability": copy.deepcopy(availability),
        "fit_sample_ids": list(fit_sample_ids), "target_fit_sample_ids": copy.deepcopy(target_fit_ids),
        "class_labels": copy.deepcopy(labels),
        "public_class_labels": _late_class_labels({"estimator": estimator, "target_decoder": target_decoder}) if classification else None,
        "learned_state_sha256": _late_learned_state_sha256(estimator),
    }
    estimator._nirs4all_late_partial_refit_origin = copy.deepcopy(origin)
    estimator._nirs4all_late_partial_refit_fingerprint = tcv1_sha256(origin)
    return {"id": origin["artifact_id"], "kind": "sklearn_estimator", "controller_id": origin["controller_id"],
            "backend": "joblib", "content_fingerprint": estimator._nirs4all_late_partial_refit_fingerprint}


def validate_late_partial_refit_origin(bundle: dict[str, Any], artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    """Check the original component state and anchor; never reattest it."""
    from .tuning_contracts import tcv1_sha256

    estimator = bundle["estimator"]
    origin, fingerprint = bundle.get("late_partial_refit_origin"), bundle.get("late_partial_refit_fingerprint")
    if (not isinstance(origin, dict) or origin.get("schema_version") != 1 or origin.get("profile") != "late_partial_v1"
            or origin.get("phase") != "REFIT" or origin.get("fold_id") is not None
            or fingerprint != tcv1_sha256(origin)
            or getattr(estimator, "_nirs4all_late_partial_refit_origin", None) != origin
            or getattr(estimator, "_nirs4all_late_partial_refit_fingerprint", None) != fingerprint
            or origin.get("multimodal_input_schema") != getattr(estimator, "multimodal_input_schema", None)
            or origin.get("source_order") != list(getattr(estimator, "multimodal_source_order", ()))
            or origin.get("source_name") != getattr(estimator, "multimodal_source_name", None)
            or origin.get("target_names") != list(getattr(estimator, "multimodal_target_names", ()))
            or origin.get("target_policy") != getattr(estimator, "multimodal_target_policy", None)
            or origin.get("missing_source_policy") != getattr(estimator, "multimodal_missing_source_policy", None)
            or origin.get("learned_state_sha256") != _late_learned_state_sha256(estimator)
            or origin.get("class_labels") is not None and origin.get("public_class_labels") != _late_class_labels(bundle)
            or origin.get("class_labels") is None and origin.get("target_fit_sample_ids") != getattr(estimator, "multimodal_target_fit_sample_ids", None)):
        raise ValueError("late partial learned state, input schema, policies or original REFIT provenance changed")
    if artifact is not None and (
        artifact.get("artifact_id", artifact.get("id")) != origin["artifact_id"]
        or artifact.get("controller_id") != origin["controller_id"] or artifact.get("content_fingerprint") != fingerprint
    ):
        raise ValueError("late partial native ArtifactRef disagrees with its original REFIT fingerprint")
    return origin


def late_partial_stack_contract(estimator: Any) -> dict[str, Any] | None:
    """Compose a closure from verified original component anchors only."""
    if not hasattr(estimator, "base_members") or not hasattr(estimator, "meta_member"):
        return None
    members = [*estimator.base_members, estimator.meta_member]
    anchors = [getattr(member, "late_partial_refit_artifact", None) for member in members]
    if not any(anchor is not None for anchor in anchors):
        return None
    if any(not isinstance(anchor, dict) for anchor in anchors):
        raise ValueError("late partial stack requires every original base and meta REFIT anchor")
    checked_anchors = [cast(dict[str, object], anchor) for anchor in anchors]
    origins = [validate_late_partial_refit_origin({**anchor, "estimator": member.estimator, "y_transform": member.y_transform}, anchor)
               for member, anchor in zip(members, checked_anchors, strict=True)]
    meta = origins[-1]
    shared = ("run_id", "variant_id", "source_order", "target_names", "target_policy", "missing_source_policy", "multimodal_input_schema", "availability", "class_labels", "public_class_labels")
    if (list(estimator.source_names or ()) != meta["source_order"]
            or [origin["source_name"] for origin in origins[:-1]] != meta["source_order"] or meta["source_name"] is not None
            or any(any(origin[key] != meta[key] for key in shared) for origin in origins[:-1])):
        raise ValueError("late partial stack mixed native REFIT runs, source order, vocabularies or validity contracts")
    probability_sources = list(estimator.probability_sources)
    if (probability_sources != [meta["class_labels"] is not None] * len(origins[:-1])
            or any(estimator.selected_probability_sources) or estimator.reduction_groups is not None):
        raise ValueError("late partial stack requires each complete prediction block in its original source order")
    return {
        "schema_version": 1, "profile": "late_partial_v1", "source_order": copy.deepcopy(meta["source_order"]),
        "target_names": copy.deepcopy(meta["target_names"]), "target_policy": meta["target_policy"],
        "missing_source_policy": meta["missing_source_policy"], "class_labels": copy.deepcopy(meta["class_labels"]),
        "public_class_labels": copy.deepcopy(meta["public_class_labels"]), "input_schema": copy.deepcopy(meta["multimodal_input_schema"]),
        "components": [copy.deepcopy(anchor) for anchor in anchors],
        "probability_sources": probability_sources,
        "source_presence": stacking_source_presence_contract(estimator),
    }


def validate_late_partial_stack(estimator: Any, expected: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Compare current closure state with its independently retained anchor."""
    actual = late_partial_stack_contract(estimator)
    captured = getattr(estimator, "late_partial_refit_contract", None)
    if actual is None and captured is None and expected is None:
        return None
    if (actual is None or captured != actual or expected is not None and actual != expected
            or getattr(estimator, "multimodal_input_schema", None) != actual["input_schema"]):
        raise ValueError("late partial stack changed its original REFIT closure")
    return actual


def late_partial_archive_contract(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Validate the independent closure and input declaration before joblib."""
    record = manifest.get("late_partial_refit")
    if record is None:
        return None
    from .tuning_contracts import tcv1_sha256

    if not isinstance(record, dict) or set(record) != {"schema_version", "closure", "fingerprint"} or record["schema_version"] != 1:
        raise ValueError("late partial archive requires its closed original REFIT declaration")
    closure = record["closure"]
    host = manifest.get("multimodal_host")
    if (not isinstance(closure, dict) or closure.get("schema_version") != 1 or closure.get("profile") != "late_partial_v1"
            or record["fingerprint"] != tcv1_sha256(closure)
            or not isinstance(host, dict) or host.get("input_schema") != closure.get("input_schema")
            or host.get("source_presence") != closure.get("source_presence")
            or not isinstance(closure.get("components"), list) or len(closure["components"]) != len(closure.get("source_order", ())) + 1):
        raise ValueError("late partial archive changed its REFIT closure, schema or source-presence layout")
    expected_keys = {"artifact_id", "controller_id", "content_fingerprint", "late_partial_refit_origin", "late_partial_refit_fingerprint"}
    origins = []
    for anchor in closure["components"]:
        if not isinstance(anchor, dict) or set(anchor) != expected_keys:
            raise ValueError("late partial archive lost a native component anchor")
        origin = anchor["late_partial_refit_origin"]
        if (not isinstance(origin, dict) or origin.get("profile") != "late_partial_v1" or origin.get("phase") != "REFIT"
                or origin.get("fold_id") is not None or origin.get("artifact_id") != anchor["artifact_id"]
                or origin.get("controller_id") != anchor["controller_id"]
                or anchor["late_partial_refit_fingerprint"] != tcv1_sha256(origin)
                or anchor["content_fingerprint"] != anchor["late_partial_refit_fingerprint"]
                or origin.get("multimodal_input_schema") != closure["input_schema"]
                or origin.get("source_order") != closure["source_order"] or origin.get("target_names") != closure["target_names"]
                or origin.get("target_policy") != closure["target_policy"] or origin.get("missing_source_policy") != closure["missing_source_policy"]
                or origin.get("class_labels") != closure["class_labels"] or origin.get("public_class_labels") != closure["public_class_labels"]):
            raise ValueError("late partial archive component disagrees with its original native REFIT")
        origins.append(origin)
    meta = origins[-1]
    if ([origin.get("source_name") for origin in origins[:-1]] != closure["source_order"] or meta.get("source_name") is not None
            or any(any(origin.get(key) != meta.get(key) for key in ("run_id", "variant_id", "availability")) for origin in origins[:-1])):
        raise ValueError("late partial archive mixed native REFIT scopes or reordered its sources")
    return copy.deepcopy(closure)


def validate_late_partial_archive_contract(manifest: dict[str, Any], estimator: Any) -> dict[str, Any] | None:
    """Link the verified archive declaration to every loaded fitted component."""
    expected = late_partial_archive_contract(manifest)
    actual = validate_late_partial_stack(estimator, expected)
    if actual is not None and expected is None:
        raise ValueError("late partial archive lost its independent original REFIT declaration")
    return actual


def stacking_source_presence_contract(estimator: Any) -> dict[str, Any] | None:
    """Validate fitted source presence separately from authentic predictions."""
    if not hasattr(estimator, "base_members") or not hasattr(estimator, "meta_member"):
        return None
    bases = [getattr(member, "estimator", None) for member in estimator.base_members]
    meta = getattr(estimator.meta_member, "estimator", None)
    policies = [getattr(model, "multimodal_missing_source_policy", None) for model in [*bases, meta]]
    if all(policy in (None, "error") for policy in policies):
        return None
    if any(policy != "zero_with_indicator" for policy in policies):
        raise ValueError("native stacking base and meta missing-source policies disagree or are unsupported")
    if meta is None or any(model is None for model in bases):
        raise ValueError("native stacking presence layout requires direct fitted base and meta models")
    if getattr(estimator, "reduction_groups", None) is not None or any(getattr(estimator, "selected_probability_sources", ())):
        raise ValueError("native stacking presence layout does not support reductions or selected-class projection")
    from sklearn.base import is_classifier

    classification = is_classifier(meta)
    if any(is_classifier(model) != classification for model in bases):
        raise ValueError("native stacking presence layout cannot mix classifiers and regressors")
    probability_sources = getattr(estimator, "probability_sources", ())
    if (classification and (len(probability_sources) != len(bases) or not all(probability_sources))
            or not classification and any(probability_sources)):
        raise ValueError("native stacking classification presence requires every full probability block")
    class_labels = None
    if classification:
        class_labels = meta.classes_.tolist()
        if len(class_labels) < 2:
            raise ValueError("native stacking presence classifiers changed their common class vocabulary")
        for model in bases:
            if model is None or model.classes_.tolist() != class_labels:
                raise ValueError("native stacking presence classifiers changed their common class vocabulary")
    names = getattr(estimator, "source_names", None)
    if names is None or len(names) != len(bases) or len(names) != len(set(names)) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("native stacking presence layout requires distinct named sources")
    if [getattr(model, "multimodal_source_name", None) for model in bases] != list(names):
        raise ValueError("native stacking base source bindings disagree with the presence layout")
    meta_names = getattr(meta, "multimodal_source_names", None)
    if not isinstance(meta_names, (list, tuple)) or list(meta_names) != list(names):
        raise ValueError("native stacking meta source order disagrees with the base artifact order")
    widths: list[int] = []
    for model in bases:
        width = getattr(model, "multimodal_prediction_width", None)
        if type(width) is not int or width < 1:
            raise ValueError("native stacking presence layout requires positive integer prediction widths")
        if class_labels is not None and width != len(class_labels):
            raise ValueError("native stacking presence probability width disagrees with the complete vocabulary")
        widths.append(width)
    total_width = sum(width + 1 for width in widths)
    if getattr(meta, "n_features_in_", total_width) != total_width:
        raise ValueError("native stacking meta feature width disagrees with its presence layout")
    return {
        "schema_version": 1, "missing_source_policy": "zero_with_indicator",
        "source_names": list(names), "prediction_widths": widths,
        "meta_feature_layout": "per_source_predictions_then_presence", "meta_feature_width": total_width,
        **({"column_block": "class_probability_values", "class_labels": class_labels} if classification else {}),
    }


def validate_stacking_archive_contract(manifest: dict[str, Any], estimator: Any) -> None:
    """Bind the declared source-presence layout to the verified fitted payload."""
    actual = stacking_source_presence_contract(estimator)
    host = manifest.get("multimodal_host") or {}
    declared = host.get("source_presence")
    if actual != declared:
        raise ValueError("native stacking archive source-presence contract disagrees with fitted state")
    if actual is not None:
        recipe = host.get("selected_model") or {}
        if (recipe.get("fusion") != "late_oof"
                or recipe.get("source_names") != actual["source_names"]
                or recipe.get("missing_source_policy") != actual["missing_source_policy"]):
            raise ValueError("native stacking archive recipe disagrees with its source-presence layout")


def named_archive_artifact(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Read the original native named REFIT anchor, never create a new one."""
    record = manifest.get("named_torch_refit")
    if record is None:
        return None
    from .tuning_contracts import tcv1_sha256

    if not isinstance(record, dict) or set(record) != {"artifact_id", "controller_id", "named_refit_origin", "named_refit_fingerprint"}:
        raise ValueError("named Torch archive requires its closed original REFIT contract")
    origin = record["named_refit_origin"]
    if (not isinstance(origin, dict) or origin.get("schema_version") != 1
            or origin.get("phase") != "REFIT" or origin.get("fold_id") is not None
            or record["artifact_id"] != origin.get("artifact_id") or record["controller_id"] != origin.get("controller_id")
            or not isinstance(record["artifact_id"], str) or not record["artifact_id"]
            or not isinstance(record["controller_id"], str) or not record["controller_id"]
            or record["named_refit_fingerprint"] != tcv1_sha256(origin)):
        raise ValueError("named Torch archive original REFIT fingerprint or identity changed")
    schema = origin.get("multimodal_input_schema")
    host = manifest.get("multimodal_host")
    if (not isinstance(schema, dict) or not schema
            or not isinstance(host, dict) or host.get("input_schema") != schema):
        raise ValueError("named Torch archive input schema disagrees with its original REFIT schema")
    return {**copy.deepcopy(record), "content_fingerprint": record["named_refit_fingerprint"]}


def validate_named_archive_contract(manifest: dict[str, Any], model: Any) -> dict[str, Any] | None:
    """Bind an independent archive anchor to the fitted object after load."""
    from nirs4all.api.result import _DagmlExportedModel

    from .named_torch_estimator import DagMLNamedTorchEstimator
    from .node_runner import validate_named_refit_origin

    estimator = model.estimator if isinstance(model, _DagmlExportedModel) else model
    artifact = named_archive_artifact(manifest)
    if artifact is None and not isinstance(estimator, DagMLNamedTorchEstimator):
        return None
    if artifact is None or not isinstance(estimator, DagMLNamedTorchEstimator):
        raise ValueError("named Torch archive is missing its original REFIT anchor or named estimator")
    validate_named_refit_origin({**artifact, "estimator": estimator, "y_transform": getattr(model, "y_transform", None)}, artifact)
    if isinstance(model, _DagmlExportedModel) and getattr(model, "named_refit_artifact", None) != artifact:
        raise ValueError("named Torch exported wrapper disagrees with the archive REFIT anchor")
    return artifact


def archive_metadata(estimator: Any, *, artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    """Expose retained input contracts and the original verified REFIT anchor."""
    from .named_torch_estimator import DagMLNamedTorchEstimator

    named_metadata: dict[str, Any] = {}
    if isinstance(estimator, DagMLNamedTorchEstimator):
        from .node_runner import validate_named_refit_origin

        if artifact is None:
            raise ValueError("named Torch export requires the original captured native ArtifactRef")
        validate_named_refit_origin({**artifact, "estimator": estimator}, artifact)
        named_metadata["named_torch_refit"] = {
            key: copy.deepcopy(artifact[key]) for key in ("artifact_id", "controller_id", "named_refit_origin", "named_refit_fingerprint")
        }
    closure = validate_late_partial_stack(estimator)
    if closure is not None:
        from .tuning_contracts import tcv1_sha256

        named_metadata["late_partial_refit"] = {"schema_version": 1, "closure": copy.deepcopy(closure), "fingerprint": tcv1_sha256(closure)}
    schema = getattr(estimator, "multimodal_input_schema", None)
    if "named_torch_refit" in named_metadata:
        # Export the schema emitted by the genuine REFIT owner, never a new
        # declaration inferred from mutable fitted attributes at export time.
        schema = copy.deepcopy(named_metadata["named_torch_refit"]["named_refit_origin"]["multimodal_input_schema"])
    if closure is not None:
        schema = copy.deepcopy(closure["input_schema"])
    if schema is None:
        return named_metadata
    from nirs4all.pipeline.config.component_serialization import serialize_component

    packages: tuple[str, ...] = ("nirs4all", "nirs4all-io", "nirs4all-methods", "dag-ml", "dag-ml-data", "numpy", "scipy", "scikit-learn", "joblib")
    if "named_torch_refit" in named_metadata:
        packages = (*packages, "torch", "cloudpickle")
    presence_contract = stacking_source_presence_contract(estimator)
    if getattr(estimator, "source_names", None) is not None and hasattr(estimator, "base_members"):
        recipe = {"fusion": "late_oof", "source_names": list(estimator.source_names),
                  "base_models": [serialize_component(member.estimator) for member in estimator.base_members],
                  "meta_model": serialize_component(estimator.meta_member.estimator)}
        if presence_contract is not None:
            recipe["missing_source_policy"] = presence_contract["missing_source_policy"]
    else:
        recipe = serialize_component(getattr(estimator, "_model", estimator))
    provider_evidence = getattr(estimator, "data_provider_evidence", None)
    if getattr(estimator, "base_members", None):
        witnesses = [getattr(member.estimator, "data_provider_evidence", None) for member in estimator.base_members]
        if any(witness != witnesses[0] for witness in witnesses):
            raise ValueError("stacking base models disagree on data-provider provenance")
        provider_evidence = witnesses[0]
    return {**named_metadata, "multimodal_host": {
        "schema_version": 1,
        "input_schema": copy.deepcopy(schema),
        "selected_model": recipe,
        "upstream_transforms": [serialize_component(step) for step in getattr(estimator, "_chain_template", [])],
        "python": platform.python_version(),
        "dependencies": {package: importlib.metadata.version(package) for package in packages},
        **({"source_presence": presence_contract} if presence_contract is not None else {}),
        **({"data_provider": copy.deepcopy(provider_evidence)} if provider_evidence is not None else {}),
    }}


def validate_dependencies(manifest: dict[str, Any]) -> None:
    """Diagnose incompatible declared host packages before deserializing state."""
    contract = manifest.get("multimodal_host")
    if contract is None:
        return
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise ValueError("unsupported multimodal host archive contract")
    if str(contract.get("python", "")).split(".")[:2] != platform.python_version().split(".")[:2]:
        raise ImportError(f"multimodal archive requires Python {contract.get('python')}; current interpreter is {platform.python_version()}")
    dependencies = contract.get("dependencies")
    if not isinstance(dependencies, dict) or not dependencies:
        raise ValueError("multimodal archive is missing its declared dependencies")
    for package, version in dependencies.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ImportError(f"multimodal archive requires {package}=={version}; the package is missing") from exc
        if installed != version:
            raise ImportError(f"multimodal archive requires {package}=={version}; installed version is {installed}")
