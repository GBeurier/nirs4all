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

    from .node_runner import _SourceConcatEstimator

    model = getattr(estimator, "_model", estimator)
    schema = getattr(estimator, "multimodal_input_schema", None)
    names = getattr(estimator, "source_names", None)
    source_concat = isinstance(estimator, _SourceConcatEstimator)
    if source_concat and (
        not isinstance(names, (list, tuple))
        or len(estimator._source_chains) != len(names)
        or len(getattr(estimator, "_source_widths", ())) != len(names)
        or not is_regressor(model)
        or not callable(getattr(model, "predict", None))
    ):
        return None
    if (not (isinstance(model, (MultimodalRegressor, MultimodalClassifier)) or source_concat)
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


def stacking_source_presence_contract(estimator: Any) -> dict[str, Any] | None:
    """Validate the fitted regression stack's missing-source feature layout."""
    if not hasattr(estimator, "base_members") or not hasattr(estimator, "meta_member"):
        return None
    bases = [getattr(member, "estimator", None) for member in estimator.base_members]
    meta = getattr(estimator.meta_member, "estimator", None)
    policies = [getattr(model, "multimodal_missing_source_policy", None) for model in [*bases, meta]]
    if all(policy in (None, "error") for policy in policies):
        return None
    if any(policy != "zero_with_indicator" for policy in policies):
        raise ValueError("native stacking base and meta missing-source policies disagree or are unsupported")
    if any(model is None for model in [*bases, meta]):
        raise ValueError("native stacking presence layout requires direct fitted base and meta models")
    if (getattr(estimator, "reduction_groups", None) is not None
            or any(getattr(estimator, "probability_sources", ()))
            or any(getattr(estimator, "selected_probability_sources", ()))):
        raise ValueError("native stacking presence layout does not support reductions or probability columns")
    from sklearn.base import is_classifier

    if any(is_classifier(model) for model in [*bases, meta]):
        raise ValueError("native stacking zero_with_indicator supports regression only")
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
        widths.append(width)
    total_width = sum(width + 1 for width in widths)
    if getattr(meta, "n_features_in_", total_width) != total_width:
        raise ValueError("native stacking meta feature width disagrees with its presence layout")
    return {
        "schema_version": 1, "missing_source_policy": "zero_with_indicator",
        "source_names": list(names), "prediction_widths": widths,
        "meta_feature_layout": "per_source_predictions_then_presence", "meta_feature_width": total_width,
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


def archive_metadata(estimator: Any) -> dict[str, Any]:
    """Expose the retained input contract, selected recipe and exact environment."""
    schema = getattr(estimator, "multimodal_input_schema", None)
    if schema is None:
        return {}
    from nirs4all.pipeline.config.component_serialization import serialize_component

    packages = ("nirs4all", "nirs4all-io", "nirs4all-methods", "dag-ml", "dag-ml-data", "numpy", "scipy", "scikit-learn", "joblib")
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
    return {"multimodal_host": {
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
