"""Callback-free DAG-ML lane for pipelines of generic n4m role steps.

A generator-free or model-parameter-sweep pipeline whose steps are all
``n4m.roles`` estimators runs through DAG-ML's native Methods estimator
controllers (``dag_ml._dag_ml.run_cv_refit_methods_in_process``). DAG-ML
compiles and schedules exactly the campaign of the host-callback lane — same
fold set, envelope views, variant SELECT, FIT_CV, REFIT and scoring — while
every node fits, transforms and predicts inside libn4m. Python only lowers the
steps to ``n4m:<method_id>`` nodes, supplies the identity-keyed numeric rows
once, and rebuilds the captured REFIT pipeline from the N4ME states for the
existing joblib artifact contract.

:func:`native_methods_refusal` is the complete eligibility rule. A pipeline
it refuses keeps the host-callback lane unchanged; the chosen lane is recorded
on the result as ``per_dataset[...]["execution_lane"]``.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

NATIVE_METHODS_LANE = "native_methods"
HOST_CALLBACK_LANE = "host_callback"
MIXED_LANES = "mixed"
NATIVE_METHODS_ENV_VAR = "N4A_DAGML_NATIVE_METHODS"

_METHOD_OPERATOR_PREFIX = "n4m:"
_METHOD_PARAM = "method_id"
_OPERATOR_KEYS = frozenset({"class", "function", "ref", "type", "model"})
# ``n4m_fit_input_t`` order (n4m/estimator.h): y, labels, then inputs a DAG
# node cannot supply from its single data binding.
_UNSUPPLIED_FIT_INPUTS = ("sample_weight", "groups", "feature_groups", "blocks", "axis", "X_target", "fold_ids")
_REQUIRED = 2
_CAP_RETAINS_TRAINING_ROWS = 1 << 9
_MODEL_STEP_KEYS = frozenset({"model", "name"})


def native_methods_refusal(steps: list[Any], spectro: Any, *, identity: Any, excluded: set[int] | None = None) -> str | None:
    """Return why ``steps`` cannot run on the native Methods lane, or ``None``.

    ``steps`` are the splitter-free, model-parameter-applied steps of one DAG
    campaign. The native lane takes them when:

    * ``N4A_DAGML_NATIVE_METHODS`` does not disable it, the in-process
      DAG-ML mechanism is selected and the installed extension exposes the
      native Methods CV/REFIT lane;
    * the dataset is one single-source 2D block with no augmented rows, no
      repetition grouping, no held-out test partition and no sample excluded
      in the OOF universe;
    * every step before the model is an ``n4m.roles`` transformer or selector
      and the last step is ``{"model": <n4m.roles regressor or classifier>}``
      (optionally with ``name``) — a classifier exactly for a classification
      task, which then has one integral class column;
    * no method requires a fit input a DAG node cannot supply (sample weights,
      groups, feature groups, blocks, spectral axis, target domain, fold ids)
      or retains training rows in its fitted state;
    * every parameter has an exact native value (integral ``int``, finite
      ``double``), and a parameter sweep only produces exact native values: a
      ``_range_`` over an ``int`` parameter has integral bounds and step, and a
      ``_log_range_`` sweeps a ``double`` parameter.

    Anything else keeps the host-callback lane.
    """

    if os.environ.get(NATIVE_METHODS_ENV_VAR, "").strip().lower() in {"0", "false", "off"}:
        return f"{NATIVE_METHODS_ENV_VAR} disables the native Methods lane"
    import importlib

    from .in_process_runner import in_process_enabled

    if not in_process_enabled():
        return "N4A_DAGML_INPROCESS selects the subprocess mechanism"
    try:
        native = importlib.import_module("dag_ml._dag_ml")
    except ImportError:
        return "the dag_ml extension is not importable"
    if not callable(getattr(native, "run_cv_refit_methods_in_process", None)):
        return "the installed dag-ml lacks the native Methods CV/REFIT lane"
    if not steps:
        return "the pipeline has no model"
    try:
        from n4m import roles
    except ImportError:
        return "the n4m roles binding is not importable"

    *transforms, model_step = steps
    if not isinstance(model_step, dict) or "model" not in model_step:
        return "the last step is not a model step"
    model = model_step["model"]
    if not isinstance(model, (roles.NativeRegressor, roles.NativeClassifier)):
        return "the model is not an n4m regressor or classifier"
    for index, step in enumerate(transforms):
        if not isinstance(step, (roles.NativeTransformer, roles.NativeSelector)):
            described = f"{{{', '.join(map(repr, step))}}} " if isinstance(step, dict) else ""
            return f"step {index} {described}is not an n4m transformer or selector"
    for key, value in model_step.items():
        if key not in _MODEL_STEP_KEYS and (reason := _sweep_refusal(model, key, value)) is not None:
            return reason
    for estimator in [*transforms, model]:
        if (reason := _method_refusal(estimator)) is not None:
            return reason

    if (reason := _dataset_refusal(spectro, identity, excluded)) is not None:
        return reason
    from .envelope import num_targets

    classifier = isinstance(model, roles.NativeClassifier)
    if classifier != bool(spectro.is_classification):
        return "the model role does not match the dataset task type"
    if classifier and num_targets(spectro) != 1:
        return "an n4m classifier needs exactly one class column"
    return None


def _method_refusal(estimator: Any) -> str | None:
    from n4m import roles

    info = roles.method_info(estimator._method_id)
    method_id = estimator._method_id
    # Class labels come from the target column of a classifier only.
    if info.inputs[1] == _REQUIRED and not isinstance(estimator, roles.NativeClassifier):
        return f"n4m method {method_id!r} requires class labels outside a classifier model"
    if any(info.inputs[2 + offset] == _REQUIRED for offset in range(len(_UNSUPPLIED_FIT_INPUTS))):
        return f"n4m method {method_id!r} requires a fit input a DAG node cannot supply"
    if info.capabilities & _CAP_RETAINS_TRAINING_ROWS:
        return f"n4m method {method_id!r} retains training rows in its fitted state"
    try:
        native_params(estimator)
    except ValueError as error:
        return str(error)
    return None


def _sweep_refusal(model: Any, key: str, value: Any) -> str | None:
    from nirs4all.pipeline.dagml_bridge import is_param_generator_spec

    if not is_param_generator_spec(value):
        return f"model step key {key!r} is not a native parameter sweep"
    kind = type(model)._param_types.get(key)
    exact = kind == "double" or (
        kind == "int" and "_range_" in value and all(float(bound).is_integer() for bound in value["_range_"])
    )
    if not exact:
        return f"parameter sweep {value!r} over {key!r} does not produce exact native {kind} values"
    return None


def _dataset_refusal(spectro: Any, identity: Any, excluded: set[int] | None) -> str | None:
    from nirs4all.data.multimodal import MultimodalSpectroDataset

    from .folds import _is_repetition_dataset

    if isinstance(spectro, MultimodalSpectroDataset) or spectro.features_sources() != 1:
        return "the dataset is not one single-source feature block"
    if any(sample.augmented for sample in identity.identities):
        return "the dataset holds augmented rows"
    if _is_repetition_dataset(spectro):
        return "the dataset groups repeated measurements"
    if len(spectro.index_column("sample", {"partition": "test"})):
        return "the dataset has a held-out test partition"
    if excluded:
        return "excluded samples are kept in the OOF universe"
    return None


def native_params(estimator: Any) -> dict[str, Any]:
    """Typed native parameter values of an n4m role estimator.

    ``None`` leaves the native default, as the n4m Python binding does. A
    value without an exact native representation raises ``ValueError``.
    """

    params: dict[str, Any] = {}
    for name, kind in type(estimator)._param_types.items():
        value = getattr(estimator, name)
        if value is None:
            continue
        params[name] = _native_value(estimator._method_id, name, kind, value)
    return params


def _native_value(method_id: str, name: str, kind: str, value: Any) -> Any:
    def inexact() -> ValueError:
        return ValueError(f"n4m parameter {method_id}.{name}={value!r} has no exact native {kind} value")

    def integral(item: Any) -> int:
        if isinstance(item, bool) or not isinstance(item, (int, float, np.integer, np.floating)) or not float(item).is_integer():
            raise inexact()
        return int(item)

    def finite(item: Any) -> float:
        if isinstance(item, bool) or not isinstance(item, (int, float, np.integer, np.floating)) or not math.isfinite(float(item)):
            raise inexact()
        return float(item)

    if kind == "int":
        return integral(value)
    if kind == "double":
        return finite(value)
    if kind == "bool":
        if not isinstance(value, (bool, np.bool_)):
            raise inexact()
        return bool(value)
    if kind == "enum":
        return str(value)
    items = np.asarray(value).reshape(-1).tolist()
    if kind == "int_array":
        return [integral(item) for item in items]
    if kind == "double_array":
        return [finite(item) for item in items]
    raise inexact()


def lane_record(lane: str, reason: str | None = None) -> dict[str, Any]:
    """The provenance record of the lane one campaign ran on."""

    return {"lane": lane, **({"reason": reason} if reason is not None else {})}


def merge_lane_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """One record for a result assembled from several campaigns."""

    lanes = {record["lane"] for record in records}
    if len(lanes) == 1:
        reasons = {record.get("reason") for record in records}
        return lane_record(lanes.pop(), reasons.pop() if len(reasons) == 1 else None)
    return lane_record(MIXED_LANES)


def record_execution_lane(result: Any, record: Mapping[str, Any]) -> None:
    """Record the lane on every dataset entry of ``result``."""

    for metadata in result.per_dataset.values():
        metadata["execution_lane"] = record["lane"]
        if "reason" in record:
            metadata["execution_lane_reason"] = record["reason"]


def result_lane_record(result: Any) -> dict[str, Any]:
    """The lane record of a single-campaign-family result.

    DAG runs outside the lowered concrete and parameter-sweep paths execute
    their operators through the host callback, so an unrecorded entry is a
    host-callback run.
    """

    return merge_lane_records([
        lane_record(metadata.get("execution_lane", HOST_CALLBACK_LANE), metadata.get("execution_lane_reason"))
        for metadata in result.per_dataset.values()
    ])


def run_cv_refit_lane(
    *,
    steps: list[Any],
    dsl: dict[str, Any],
    envelope: dict[str, Any],
    spectro: Any,
    identity: Any,
    excluded: set[int] | None,
    selection_metric: str,
    refit: bool,
    refit_top_k: int,
    run_callback: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Run one CV+REFIT campaign on the native Methods lane when eligible.

    ``run_callback`` runs the unchanged host-callback campaign. Both lanes
    return the same outcome shape plus an ``execution_lane`` record; neither
    retries the other.
    """

    reason = native_methods_refusal(steps, spectro, identity=identity, excluded=excluded)
    if reason is not None:
        outcome = run_callback()
        outcome["execution_lane"] = lane_record(HOST_CALLBACK_LANE, reason)
        return outcome
    outcome = _run_native_methods(
        steps=steps, dsl=dsl, envelope=envelope, spectro=spectro, identity=identity,
        selection_metric=selection_metric, refit=refit, refit_top_k=refit_top_k,
    )
    outcome["execution_lane"] = lane_record(NATIVE_METHODS_LANE)
    return outcome


def _run_native_methods(
    *,
    steps: list[Any],
    dsl: dict[str, Any],
    envelope: dict[str, Any],
    spectro: Any,
    identity: Any,
    selection_metric: str,
    refit: bool,
    refit_top_k: int,
) -> dict[str, Any]:
    import importlib

    from .methods_runtime import resolve_methods_library_path
    from .resolver import MaterializationResolver
    from .resources import current_execution_resources

    native_dsl, graph, head = _native_dsl(dsl, steps, envelope)
    resolver = MaterializationResolver(spectro, identity)
    sample_ids = list(dict.fromkeys(record["sample_id"] for record in envelope["coordinator_relations"]["records"]))
    x = np.asarray(resolver.resolve_features(sample_ids)["values"], dtype=np.float64)
    target_block = resolver.resolve_targets(resolver.target_sample_ids(sample_ids))
    y = np.asarray(target_block["values"], dtype=np.float64).reshape(len(sample_ids), -1)
    # The same target names the host model controller emits.
    names = target_block.get("target_names", [f"y{index}" for index in range(y.shape[1])] if y.shape[1] > 1 else ["y"])

    from .raw_training_lowerer import _array_content_fingerprint

    native_envelope = {
        **envelope,
        "data_content_fingerprint": _array_content_fingerprint("X", x),
        "target_content_fingerprint": _array_content_fingerprint("y", y),
    }
    inputs = {f"{head}.x": {"sample_ids": sample_ids, "x": x.tolist(), "y": y.tolist(), "target_names": list(names)}}
    native = importlib.import_module("dag_ml._dag_ml")
    payload_json, payloads = native.run_cv_refit_methods_in_process(
        json.dumps(native_dsl),
        json.dumps(native_envelope),
        json.dumps(inputs),
        resolve_methods_library_path(),
        selection_metric,
        json.dumps(current_execution_resources().to_contract()),
        refit,
        refit_top_k,
    )
    payload = json.loads(payload_json)
    node_results = payload.get("node_results", [])
    return {
        "returncode": 0,
        "stdout": "",
        "results": node_results,
        "scores": payload.get("scores"),
        "residual_gates": payload.get("residual_gates", []),
        "variant_catalog": payload.get("variant_catalog", []),
        "selected_refit_variant_ids": payload.get("selected_refit_variant_ids", []),
        "classification_evidence": [],
        "refit_artifacts": _host_refit_artifacts(node_results, payloads, graph, resolver),
    }


def _native_dsl(dsl: dict[str, Any], steps: list[Any], envelope: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Lower the host DSL's steps to ``n4m:<method_id>`` nodes bound at the chain head.

    Returns the native DSL, its compiled graph and the chain head node id.
    """

    import dag_ml
    from n4m import roles

    from .cli_runner import data_bindings_for

    native = copy.deepcopy(dsl)
    pipeline = native["pipeline"]
    if len(pipeline) != len(steps):
        raise ValueError("native Methods lowering requires one DSL step per pipeline step")
    for position, step in enumerate(steps):
        estimator = step["model"] if isinstance(step, dict) else step
        declared = pipeline[position]
        # The operator is a `ref` so the manifest-derived `n4m:<method_id>`
        # selectors resolve it; every other declared key is preserved.
        lowered = {key: value for key, value in (declared.items() if isinstance(declared, dict) else ()) if key not in _OPERATOR_KEYS}
        reference = f"{_METHOD_OPERATOR_PREFIX}{estimator._method_id}"
        if isinstance(step, dict):
            lowered["model"] = reference
        else:
            lowered["ref"] = reference
        lowered["params"] = {_METHOD_PARAM: estimator._method_id, **native_params(estimator)}
        pipeline[position] = lowered
    native.pop("data_bindings", None)
    manifests = dag_ml.derive_controller_manifests(dag_ml.n4m_host_controller_specs(roles.manifest())).to_dict()
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(native, manifests).graph.to_dict()
    head = _head_node_id(graph)
    native["data_bindings"] = data_bindings_for(head, envelope)
    return native, graph, head


def _head_node_id(graph: Mapping[str, Any]) -> str:
    targets = {edge["target"]["node_id"] for edge in graph.get("edges", [])}
    heads = [node["id"] for node in graph["nodes"] if node["id"] not in targets]
    if len(heads) != 1:
        raise ValueError("native Methods lowering requires one linear chain")
    return str(heads[0])


def _x_chain(graph: Mapping[str, Any], node_id: str) -> list[str]:
    """Upstream node ids of ``node_id``, furthest first."""

    source_of = {edge["target"]["node_id"]: edge["source"]["node_id"] for edge in graph.get("edges", [])}
    chain: list[str] = []
    current = source_of.get(node_id)
    while current is not None:
        chain.append(current)
        current = source_of.get(current)
    chain.reverse()
    return chain


def _host_refit_artifacts(
    node_results: list[dict[str, Any]],
    payloads: Mapping[str, bytes],
    graph: Mapping[str, Any],
    resolver: Any,
) -> list[dict[str, Any]]:
    """Rebuild each REFIT model as the host lane's captured ``Pipeline``.

    The fitted estimators are restored from their N4ME states; the entries
    keep the host artifact contract (``artifact_id`` naming, joblib backend,
    target decoder) so persistence, export and replay are unchanged.
    """

    from n4m.roles import NativeEstimator
    from sklearn.pipeline import make_pipeline

    from .node_runner import _artifact_id
    from .target_capture import captured_target_transform

    states: dict[tuple[str, str], Any] = {}
    models: list[tuple[str, str, str]] = []
    for frame in node_results:
        lineage = frame.get("lineage") or {}
        if lineage.get("phase") != "REFIT":
            continue
        variant = lineage.get("variant_id") or "base"
        for artifact in frame.get("artifacts", []):
            states[(frame["node_id"], variant)] = NativeEstimator.from_n4me(payloads[artifact["id"]])
            if frame.get("predictions"):
                models.append((frame["node_id"], variant, artifact["controller_id"]))
    captured = []
    for node_id, variant, controller_id in models:
        chain = [states[(upstream, variant)] for upstream in _x_chain(graph, node_id)]
        model = states[(node_id, variant)]
        estimator = make_pipeline(*chain, model) if chain else model
        captured.append({
            "artifact_id": _artifact_id(node_id, variant),
            "estimator": estimator,
            "y_transform": captured_target_transform(None, resolver.target_decoder(), estimator),
            "kind": "sklearn_estimator",
            "controller_id": controller_id,
            "backend": "joblib",
        })
    return captured


__all__ = [
    "HOST_CALLBACK_LANE",
    "MIXED_LANES",
    "NATIVE_METHODS_ENV_VAR",
    "NATIVE_METHODS_LANE",
    "lane_record",
    "merge_lane_records",
    "native_methods_refusal",
    "native_params",
    "record_execution_lane",
    "result_lane_record",
    "run_cv_refit_lane",
]
