"""Trusted Python sidecars retaining a complete native Torch predictor closure.

The signed DAG package owns scheduling and replay. This module transports fitted
objects, attests their owners and input declarations, and never constructs folds,
OOF predictions, scores or a selected recipe.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import numpy as np

TORCH_CLASS = "nirs4all.pipeline.dagml.torch_estimator.DagMLTorchEstimator"
TORCH_FACTORY = "nirs4all.operators.models.pytorch.mlp.structural_mlp"
TRAINING_POLICY = {"validation": "none", "shuffle": True, "early_stopping": False}


def operator_name(node: dict[str, Any]) -> str | None:
    """Accept the two existing import-reference spellings, no embedded controls."""
    value = node.get("operator")
    if isinstance(value, str):
        return value
    return value.get("class") if isinstance(value, dict) and set(value) == {"class"} else None


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(f"Python Torch topology: {message}")


def _task_seed(task: dict[str, Any], profile: dict[str, Any], node_id: str) -> int:
    from .tuning_contracts import tcv1_sha256

    return int(tcv1_sha256({"seed": profile["seed"], "variant": task.get("variant_id"), "fold": task.get("fold_id"),
                           "node": node_id, "phase": task["phase"]})[:8], 16)


def _array_identity(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    _require(array.dtype.kind in "fiu" and np.isfinite(array).all(), "finite learned numeric state required")
    canonical = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    return {"dtype": canonical.dtype.str, "shape": list(canonical.shape),
            "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest()}


def learned_state_sha256(estimator: Any) -> str:
    """Hash the fitted numerical state without serializing a replacement model."""
    from sklearn.linear_model import Ridge

    from .torch_estimator import DagMLTorchEstimator
    from .tuning_contracts import tcv1_sha256

    if type(estimator) is DagMLTorchEstimator:
        _require(hasattr(estimator, "model_"), "unfitted Torch artifact")
        state = {name: _array_identity(tensor.detach().cpu().numpy())
                 for name, tensor in sorted(estimator.model_.state_dict().items())}
    else:
        _require(type(estimator) is Ridge and hasattr(estimator, "coef_"), "unfitted or foreign Ridge artifact")
        state = {"coef_": _array_identity(estimator.coef_), "intercept_": _array_identity(estimator.intercept_)}
    return tcv1_sha256({"schema_version": 1, "n_features_in_": int(estimator.n_features_in_), "state": state})


def _selected_sources(node: dict[str, Any], node_lookup: Any, prediction_inputs: dict[str, Any]) -> list[str]:
    if operator_name(node) == TORCH_CLASS:
        return list(node["metadata"]["source_selection"])
    from .node_runner import _ordered_oof_specs

    specs = _ordered_oof_specs(prediction_inputs, suffix=None, source_order=node["metadata"].get("prediction_source_order"),
                               source_ports=node["metadata"].get("prediction_source_ports") or {})
    return [node_lookup(spec["producer_node"])["metadata"]["source_selection"][0] for spec in specs]


def emit_refit_attestation(task: dict[str, Any], node: dict[str, Any], bundle: dict[str, Any], artifact: dict[str, Any],
                           fit_ids: list[str], metadata: dict[str, Any], node_lookup: Any, edges: list[dict[str, Any]]) -> None:
    """Record the true completed owner FIT before its ArtifactRef is emitted.

    Capture and replay must never call this function or repair an unknown origin.
    The native ArtifactRef signs this attestation's digest; the trusted joblib ZIP
    member retains its separate byte integrity check.
    """
    from .node_runner import _variant_overrides
    from .tuning_contracts import tcv1_sha256

    del edges  # Native task prediction_inputs already carry the admitted signed edge identities.
    estimator = bundle["estimator"]
    _require(task["phase"] == "REFIT" and task.get("fold_id") is None and fit_ids and len(set(fit_ids)) == len(fit_ids), "genuine full-Train REFIT scope required")
    _require(not hasattr(estimator, "_nirs4all_torch_refit_attestation") and "refit_attestation" not in bundle, "cannot re-attest an existing learned artifact")
    _require(bundle["y_transform"] is None, "unsigned target transform")
    params = {**task["node_plan"]["params"], **_variant_overrides(task, node["id"])}
    _require(estimator.get_params(deep=False) == params, "effective fitted controls differ from the native task")
    profile = metadata["python_torch_profile"]
    inputs = task.get("prediction_inputs") or {}
    oof = {key: copy.deepcopy(value) for key, value in inputs.items()
           if not any(key.endswith(suffix) for suffix in (":outer", ":refit", ":predict", ":test"))}
    attestation = {
        "schema_version": 1, "artifact_id": artifact["id"], "node_id": node["id"],
        "controller_id": task["node_plan"]["controller_id"], "run_id": task["run_id"],
        "phase": "REFIT", "variant_id": task.get("variant_id"), "fold_id": None,
        "fit_sample_ids": list(fit_ids), "native_seed": task.get("seed"), "effective_seed": _task_seed(task, profile, node["id"]),
        "source_order": _selected_sources(node, node_lookup, inputs), "source_schemas": copy.deepcopy(metadata["source_schemas"]),
        "target_names": list(profile["target_names"]), "params": copy.deepcopy(params),
        "training_policy": copy.deepcopy(profile["training_policy"]), "upstream_oof": oof,
        "learned_state_sha256": learned_state_sha256(estimator),
    }
    # Independent copies bind the actual fitted object and the owner store entry.
    estimator._nirs4all_torch_refit_attestation = copy.deepcopy(attestation)
    bundle["refit_attestation"] = copy.deepcopy(attestation)
    artifact["content_fingerprint"] = tcv1_sha256(attestation)


def validate_refit_attestation(bundle: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    """Verify original owner provenance and current state, never regenerate it."""
    from .tuning_contracts import tcv1_sha256

    attestation = bundle.get("refit_attestation")
    _require(isinstance(attestation, dict) and artifact.get("content_fingerprint") == tcv1_sha256(attestation), "native REFIT attestation is missing or changed")
    attestation = cast(dict[str, Any], attestation)
    estimator = bundle["estimator"]
    _require(getattr(estimator, "_nirs4all_torch_refit_attestation", None) == attestation, "fitted object has a foreign or missing REFIT origin")
    _require(attestation["artifact_id"] == artifact["id"] and attestation["controller_id"] == artifact["controller_id"]
             and attestation["phase"] == "REFIT" and attestation["fold_id"] is None, "foreign artifact or REFIT scope")
    _require(attestation["params"] == estimator.get_params(deep=False)
             and attestation["learned_state_sha256"] == learned_state_sha256(estimator), "learned state or effective controls differ from the emitted REFIT artifact")
    return attestation


def _finite_float32(value: Any, message: str) -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(value, dtype=np.float32)
    _require(np.isfinite(converted).all(), message)


def _validate_sources(metadata: dict[str, Any], dataset: Any, ids: list[str], resolver: Any) -> None:
    from nirs4all.data.multimodal import MultimodalSpectroDataset

    profile = metadata["python_torch_profile"]
    _require(isinstance(dataset, MultimodalSpectroDataset), "named dense sources required")
    _require(list(dataset.source_names) == profile["source_order"], "source names/order differ from the signed profile")
    descriptors = dict(zip(dataset.source_names, dataset.cohort.schema_descriptors(), strict=True))
    actual = {
        name: {"representation_id": item["representation_id"], "input_shape": [item["shape"][1]], "dtype": item["dtype"],
               "identity": json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)}
        for name, item in descriptors.items()
    }
    _require(actual == metadata["source_schemas"], "source schema, axes, units, dtype or feature labels changed")
    if ids:
        resolved = resolver.resolve_feature_blocks(ids, include_augmented=False, source_names=tuple(profile["source_order"]))
        _require(not resolved.get("source_masks"), "complete modalities required")
        for name, block in zip(profile["source_order"], resolved["blocks"], strict=True):
            values = np.asarray(block)
            _require(values.shape == (len(ids), profile["source_widths"][name]) and values.dtype.kind == "f"
                     and values.size <= 16_777_216 and np.isfinite(values).all(), "finite bounded dense source buffer required")
            _finite_float32(values, "raw source is not finite after float32 conversion")


@contextmanager
def torch_task_scope(task: dict[str, Any], resolver: Any, node_lookup: Any, edges: list[dict[str, Any]], metadata: dict[str, Any] | None) -> Iterator[None]:
    """Validate the closed owner boundary and restore CPU RNG after each task."""
    from .node_runner import _train_predict_ids, _variant_overrides

    node = node_lookup(task["node_plan"]["node_id"])
    params = {**task["node_plan"].get("params", {}), **_variant_overrides(task, node["id"])}
    known = params.get("factory_path") == TORCH_FACTORY
    if not (metadata and "python_torch_profile" in metadata):
        _require(not known, "structural_mlp requires signed native graph metadata")
        yield
        return
    profile = metadata["python_torch_profile"]
    _require(profile.get("training_policy") == TRAINING_POLICY, "unsigned or unsupported training policy")
    _require(isinstance(profile.get("target_names"), list) and len(profile["target_names"]) == 1
             and isinstance(profile["target_names"][0], str) and profile["target_names"][0].strip(), "signed scalar target name required")
    _require((task.get("resources") or {}).get("cpu_threads") == 1 and not (task.get("resources") or {}).get("gpu_devices"), "serial CPU task resources required")
    phase = task["phase"]
    _require(phase in {"FIT_CV", "REFIT", "PREDICT"}, "unsupported phase")
    kind = task["node_plan"]["kind"]
    if kind != "model":
        yield
        return
    raw = operator_name(node) == TORCH_CLASS
    fit_ids: list[str] = []
    predict_ids: list[str] = []
    if raw:
        _require(known and params.get("device") == "cpu" and params.get("force_layout") == "2d", "exact real CPU Torch owner required")
        _require(task["node_plan"]["controller_id"] == "controller:nirs4all.model", "foreign Torch owner")
        fit_ids, predict_ids = _train_predict_ids(copy.deepcopy(task))
        keys = {"factory_path", "template_blob", "factory_params", "force_layout", "task_type", "num_classes", "epochs", "batch_size", "patience", "optimizer", "lr", "learning_rate", "loss", "device"}
        _require(set(params) == keys and params["template_blob"] is None and params["num_classes"] is None
                 and params["task_type"] == "regression" and params["optimizer"] == "Adam"
                 and params["loss"] == "MSELoss" and params["learning_rate"] is None, "unsigned training controls")
        _require(isinstance(params["factory_params"], dict) and set(params["factory_params"]) == {"hidden_units"}, "unsigned architecture parameters")
        for key, high in (("epochs", 100), ("batch_size", 1024), ("patience", 100)):
            _require(type(params[key]) is int and 1 <= params[key] <= high, "unbounded training controls")
        hidden = params["factory_params"]["hidden_units"]
        _require(type(hidden) is int and 1 <= hidden <= 128 and type(params["lr"]) in (int, float)
                 and np.isfinite(params["lr"]) and 1e-6 <= params["lr"] <= 0.1, "unbounded architecture or learning rate")

        selected = node["metadata"]["source_selection"]
        _require(1 <= len(selected) <= 4 and len(set(selected)) == len(selected) and set(selected) <= set(profile["source_order"]), "unsupported source selection")
        expected_index = profile["source_order"].index(selected[0]) if len(selected) == 1 else None
        _require(node["metadata"].get("source_index") == expected_index, "source selector differs from signed named source")
        width = sum(profile["source_widths"][name] for name in selected)
        _require((width + 2) * hidden + 1 <= 1_000_000 and max(len(fit_ids), len(predict_ids)) * width <= 16_777_216
                 and len(fit_ids) * params["epochs"] * hidden * (width + 1) <= 100_000_000, "actual owner work exceeds its declared closed budget")

    else:
        _require(task["node_plan"]["controller_id"] == "controller:nirs4all.meta_model", "foreign meta owner")
        incoming = [edge for edge in edges if edge["target"]["node_id"] == node["id"]]
        expected = {edge["source"]["node_id"]: edge for edge in incoming}
        _require(2 <= len(expected) <= 4 and all(edge["contract"].get("requires_oof") is True for edge in incoming), "exact signed OOF closure required")
        training_blocks = []
        for key, block in (task.get("prediction_inputs") or {}).items():
            producer = block["producer_node"]
            _require(producer in expected, "foreign prediction producer")
            edge = expected[producer]
            _require(block["source_port"] == edge["source"]["port_name"] and block["target_port"] == edge["target"]["port_name"], "foreign prediction port")
            samples = block["sample_ids"]
            values = np.asarray(block["values"], dtype=float)
            _require(samples and len(set(samples)) == len(samples) and values.shape == (len(samples), 1)
                     and np.isfinite(values).all(), "complete unique mono-y prediction identities required")
            operational = any(key.endswith(suffix) for suffix in (":outer", ":refit", ":predict", ":test"))
            if not operational:
                _require(phase != "PREDICT" and block["partition"] == "validation", "meta fit accepts Validation OOF only")
                _require(phase != "FIT_CV" or task["fold_id"] not in block.get("fold_ids", []), "outer Validation cannot fit its meta-model")
                training_blocks.append(block)
            else:
                suffix = key.rsplit(":", 1)[1]
                valid_scope = ((phase == "FIT_CV" and suffix in {"outer", "test"} and block["fold_id"] == task["fold_id"]
                                and block["partition"] == ("validation" if suffix == "outer" else "test"))
                               or (phase == "REFIT" and suffix in {"refit", "test"} and block["partition"] == "test" and block["fold_id"] is None)
                               or (phase == "PREDICT" and suffix == "predict" and block["partition"] == "final" and block["fold_id"] is None))
                _require(valid_scope, "off-fold prediction partition or phase differs from native scope")
                predict_ids.extend(samples)
        if phase == "PREDICT":
            supplied = list((task.get("prediction_inputs") or {}).values())
            _require(len(supplied) == len(expected) and {block["producer_node"] for block in supplied} == set(expected)
                     and all(block["sample_ids"] == supplied[0]["sample_ids"] for block in supplied), "incomplete or reordered native PREDICT closure")
        if phase != "PREDICT":
            _require({block["producer_node"] for block in training_blocks} == set(expected)
                     and len(training_blocks) == len(expected), "complete OOF branches required")
            fit_ids = training_blocks[0]["sample_ids"]
            _require(all(block["sample_ids"] == fit_ids for block in training_blocks), "OOF row order differs between branches")
    if fit_ids:
        train = {resolver._identity.to_wire(int(sample)) for sample in resolver._dataset.index_column("sample", {"partition": "train"})}
        _require(set(fit_ids) <= train, "foreign or Test sample in fit scope")
        target_block = resolver.resolve_targets(fit_ids)
        target = np.asarray(target_block["values"], dtype=float)
        # The resolver preserves the historical flat wire form for mono-y.
        # Only that exact form is normalized; multi-target axes remain strict.
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        _require(target_block.get("target_names") == profile["target_names"], "target names differ from the signed scalar target")
        _require(target.shape == (len(fit_ids), 1) and np.isfinite(target).all(), "finite mono-y training targets required")
        _finite_float32(target, "targets are not finite after float32 conversion")
    scope_ids = list(dict.fromkeys([*fit_ids, *predict_ids]))
    if raw and phase in {"FIT_CV", "REFIT"}:
        scope_ids = list(dict.fromkeys([*scope_ids, *resolver.partition_wire_ids("test")]))
    _validate_sources(metadata, resolver._dataset, scope_ids, resolver)
    seed = _task_seed(task, profile, node["id"])
    import torch

    # CPU only: no CUDA initialization or mutation. This profile's native
    # scheduler executes one node/fold at a time; no generic global lock is added.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        with torch.device("cpu"):
            yield


class CapturedTorchTopology:
    """One trusted joblib member holding all native-selected fitted sidecars."""

    def __init__(self, package: dict[str, Any], artifacts: dict[str, dict[str, Any]]) -> None:
        self.package = copy.deepcopy(package)
        self.artifacts = artifacts
        self.validate()

    def validate(self) -> None:
        from dag_ml import PortablePredictorPackage
        from sklearn.linear_model import Ridge

        from .node_runner import _variant_overrides
        from .torch_estimator import DagMLTorchEstimator

        PortablePredictorPackage(self.package)  # validates every native signature/closure
        package = self.package
        plan = package["effective_plan"]
        graph = plan["graph_plan"]["graph"]
        profile = graph["metadata"].get("python_torch_profile")
        _require(isinstance(profile, dict) and profile.get("training_policy") == TRAINING_POLICY,
                 "captured topology lacks the exact signed CPU training policy")
        _require(package["fitted_artifact_mode"] == "allow_host_sidecar", "trusted host-sidecar mode required")
        records = package["execution_bundle"]["refit_artifacts"]
        identifiers = [record["artifact"]["id"] for record in records]
        _require(len(set(identifiers)) == len(records) and set(self.artifacts) == set(identifiers), "sidecars differ from exact native REFIT inventory")
        _require({item["artifact_id"] for item in package["artifact_bindings"]} == set(identifiers)
                 and all(item["load_mode"] == "host_sidecar" for item in package["artifact_bindings"]), "foreign or portable-only artifact binding")
        nodes = {node["id"]: node for node in graph["nodes"] if node["kind"] == "model"}
        _require({record["node_id"] for record in records} == set(nodes) and len(records) == len(nodes), "incomplete selected model closure")
        bindings = package["output_bindings"]
        _require(len(bindings) == 1 and bindings[0]["node_id"] in nodes and bindings[0]["target_names"] == profile["target_names"], "exact terminal mono-y output required")
        selected = package["execution_bundle"]["selected_variant_id"]
        variants = [variant for variant in plan["variants"] if variant["variant_id"] == selected]
        _require(len(variants) == 1, "selected native variant is missing")
        for record in records:
            node = nodes[record["node_id"]]
            owner = "controller:nirs4all.model" if operator_name(node) == TORCH_CLASS else "controller:nirs4all.meta_model"
            artifact = record["artifact"]
            _require(record["controller_id"] == artifact["controller_id"] == owner
                     and artifact["backend"] == "joblib" and artifact["kind"] == "sklearn_estimator", "foreign fitted owner or serializer")
            bundle = self.artifacts[artifact["id"]]
            _require(set(bundle) == {"estimator", "y_transform", "refit_attestation"} and bundle["y_transform"] is None, "unsigned target transform or sidecar member")
            estimator = bundle["estimator"]
            _require(type(estimator) is (DagMLTorchEstimator if operator_name(node) == TORCH_CLASS else Ridge), "foreign fitted estimator class")
            params = {**plan["node_plans"][node["id"]]["params"], **_variant_overrides({"variant": variants[0]}, node["id"])}
            _require(estimator.get_params(deep=False) == params, "fitted estimator differs from selected signed parameters")
            attestation = validate_refit_attestation(bundle, artifact)
            train_ids = plan["fold_set"]["sample_ids"]
            _require(attestation["node_id"] == node["id"] and attestation["variant_id"] == selected
                     and attestation["params"] == params and attestation["source_schemas"] == graph["metadata"]["source_schemas"]
                     and attestation["target_names"] == profile["target_names"] and attestation["training_policy"] == profile["training_policy"]
                     and len(attestation["fit_sample_ids"]) == len(train_ids) and set(attestation["fit_sample_ids"]) == set(train_ids),
                     "emitted REFIT origin differs from selected signed full-Train recipe")
            expected_sources = list(node["metadata"]["source_selection"]) if operator_name(node) == TORCH_CLASS else [
                nodes[producer]["metadata"]["source_selection"][0] for producer in node["metadata"]["prediction_source_order"]]
            _require(attestation["source_order"] == expected_sources, "REFIT source order differs from the selected recipe")
            _require(attestation["effective_seed"] == _task_seed({"variant_id": selected, "fold_id": None, "phase": "REFIT"}, profile, node["id"]),
                     "REFIT effective seed differs from its signed task coordinates")
            if operator_name(node) == TORCH_CLASS:
                import torch

                _require(type(estimator) is DagMLTorchEstimator and hasattr(estimator, "model_"), "unfitted or foreign Torch sidecar")
                selected_sources = node["metadata"]["source_selection"]
                width = sum(profile["source_widths"][name] for name in selected_sources)
                hidden = params["factory_params"]["hidden_units"]
                layers = list(estimator.model_.children())
                _require(type(estimator.model_) is torch.nn.Sequential and len(layers) == 4
                         and [type(layer) for layer in layers] == [torch.nn.Flatten, torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear]
                         and (layers[1].in_features, layers[1].out_features, layers[3].in_features, layers[3].out_features) == (width, hidden, hidden, 1)
                         and estimator.n_features_in_ == width and layers[0].start_dim == 1 and layers[0].end_dim == -1
                         and not layers[2].inplace and layers[1].bias is not None and layers[3].bias is not None,
                         "fitted architecture differs from signed factory/width")
                _require(all(value.device.type == "cpu" and value.dtype == torch.float32 and torch.isfinite(value).all().item()
                             for value in estimator.model_.parameters()), "CPU float32 finite weights required")
            else:
                _require(operator_name(node) == "sklearn.linear_model._ridge.Ridge" and type(estimator) is Ridge,
                         "foreign learned-late meta sidecar")
                count = sum(edge["target"]["node_id"] == node["id"] for edge in graph["edges"])
                _require(estimator.n_features_in_ == count and np.asarray(estimator.coef_).size == count
                         and np.isfinite(estimator.coef_).all() and np.isfinite(estimator.intercept_).all(), "meta fitted width or state differs from selected OOF closure")
        self.multimodal_source_names = tuple(profile["source_order"])

    def predict(self, X: Any) -> np.ndarray:
        from .dataset import _materialize_dataset

        values, _evidence = replay_torch_topology(self, _materialize_dataset(X))
        return values


def capture_torch_topology(training: Any, frames: list[dict[str, Any]], store: dict[Any, Any]) -> dict[str, Any]:
    """Capture exactly the native REFIT artifact inventory before detach."""
    from .node_runner import _stable_handle

    package = training.export_portable_predictor_package("predictor:nirs4all.torch", fitted_artifact_mode="allow_host_sidecar", artifact_load_mode="host_sidecar").to_dict()
    records = package["execution_bundle"]["refit_artifacts"]
    emitted = {artifact["id"] for frame in frames for artifact in frame.get("artifacts", [])}
    expected = {record["artifact"]["id"] for record in records}
    _require(emitted == expected, "captured frames differ from complete native REFIT inventory")
    captured: dict[str, dict[str, Any]] = {}
    for record in records:
        identifier = record["artifact"]["id"]
        bundle = store.get(_stable_handle(identifier))
        _require(isinstance(bundle, dict) and "estimator" in bundle, "native REFIT owner artifact is missing")
        bundle = cast(dict[str, Any], bundle)
        validate_refit_attestation(bundle, record["artifact"])
        emissions = [frame for frame in frames if any(artifact["id"] == identifier for artifact in frame.get("artifacts", []))]
        _require(len(emissions) == 1, "REFIT artifact has no unique native emission")
        origin = bundle["refit_attestation"]
        lineage = emissions[0]["lineage"]
        emitted_ref = next(artifact for artifact in emissions[0]["artifacts"] if artifact["id"] == identifier)
        # Native serialization supplies optional null defaults (size_bytes).
        emitted_fields = {key: item for key, item in emitted_ref.items() if item is not None}
        native_fields = {key: item for key, item in record["artifact"].items() if item is not None}
        _require(all(origin[key] == lineage[key] for key in ("node_id", "controller_id", "run_id", "phase", "variant_id", "fold_id"))
                 and origin["native_seed"] == lineage.get("seed") and native_fields == emitted_fields,
                 "artifact provenance differs from its actual native REFIT emission")
        captured[identifier] = {"estimator": bundle["estimator"], "y_transform": bundle["y_transform"],
                                "refit_attestation": copy.deepcopy(bundle["refit_attestation"])}
    model = CapturedTorchTopology(package, captured)
    binding = package["output_bindings"][0]
    record = next(record for record in records if record["node_id"] == binding["node_id"])
    return {"artifact_id": record["artifact"]["id"], "estimator": model, "y_transform": None,
            "kind": "sklearn_estimator", "backend": "joblib", "controller_id": record["controller_id"]}


def replay_torch_topology(model: CapturedTorchTopology, spectro: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Invoke the existing loaded-package native replay for every signed node."""
    from .envelope import build_envelope
    from .identity import mint_identity
    from .node_runner import _stable_handle, run_node
    from .raw_replay_lowerer import _requirements, _source_outcome_fingerprint
    from .raw_training_lowerer import _core_relation_fingerprint
    from .resolver import MaterializationResolver

    model.validate()
    native = importlib.import_module("dag_ml")
    package = model.package
    graph = package["effective_plan"]["graph_plan"]["graph"]
    identity = mint_identity(spectro)
    current = build_envelope(spectro, identity)
    relations = current["coordinator_relations"]
    fingerprint = _core_relation_fingerprint(relations, native)
    requirements = _requirements(package["execution_bundle"])
    envelopes = {
        key: native.attach_predict_cohort_to_envelope({
            "schema_version": 1, "schema_fingerprint": requirement["schema_fingerprint"], "plan_fingerprint": requirement["plan_fingerprint"],
            "relation_fingerprint": fingerprint, "data_content_fingerprint": spectro.content_hash(), "target_content_fingerprint": None,
            "coordinator_relations": relations,
        }, {"role": "inference", "relations": relations, "target_names": graph["metadata"]["python_torch_profile"]["target_names"], "data_content_fingerprint": spectro.content_hash(), "target_content_fingerprint": None}).to_dict()
        for key, requirement in requirements.items()
    }
    binding = package["output_bindings"][0]
    request = native.sign_training_replay_request({
        "schema_version": 1, "request_id": "replay:nirs4all.torch", "source_outcome_fingerprint": _source_outcome_fingerprint(package),
        "phase": "PREDICT", "data_envelope_keys": sorted(envelopes), "output_binding_ids": [binding["binding_id"]], "request_fingerprint": "0" * 64,
    })
    store: dict[Any, Any] = {}
    handles = {}
    nodes = {node["id"]: node for node in graph["nodes"]}
    for record in package["execution_bundle"]["refit_artifacts"]:
        identifier = record["artifact"]["id"]
        handle = _stable_handle(identifier)
        store[handle] = model.artifacts[identifier]
        handles[identifier] = {"handle": handle, "kind": "model", "owner_controller": record["controller_id"]}
    resolver = MaterializationResolver(spectro, identity)
    phases: list[dict[str, Any]] = []

    def callback(task: dict[str, Any]) -> dict[str, Any]:
        _require(task["phase"] == "PREDICT" and task["node_plan"]["node_id"] in nodes, "replay cannot FIT, search or invoke a foreign node")
        phases.append({"phase": task["phase"], "node_id": task["node_plan"]["node_id"], "controller_id": task["node_plan"]["controller_id"]})
        return run_node(task, resolver, nodes.__getitem__, store, graph["edges"], graph_metadata=graph["metadata"])

    try:
        outcome = native.replay_loaded_predictor_package(package, request, envelopes, handles, callback,
                                                       outcome_id="outcome:nirs4all.torch_predict", run_id="run:nirs4all.torch_predict",
                                                       trusted_controller_manifests=list(package["effective_plan"]["controller_manifests"].values()))
        document = outcome.to_dict()
        expected_nodes = {record["node_id"] for record in package["execution_bundle"]["refit_artifacts"]}
        _require({event["node_id"] for event in phases} == expected_nodes and len(phases) == len(expected_nodes), "replay did not execute the exact complete selected closure")
        ids = identity.observation_ids()
        blocks = [block for output in document["outputs"] for block in output["predictions"]]
        _require(len(blocks) == 1 and blocks[0]["producer_node"] == binding["node_id"] and blocks[0]["partition"] == "final" and blocks[0]["fold_id"] is None, "native terminal PREDICT block required")
        block = blocks[0]
        rows = dict(zip(block["sample_ids"], block["values"], strict=True))
        _require(len(rows) == len(block["sample_ids"]) and set(rows) == set(ids), "native PREDICT cohort differs from current sample identities")
        values = np.asarray([rows[sample] for sample in ids], dtype=float)
        _require(values.shape == (len(ids), 1) and np.isfinite(values).all(), "finite native mono-y output required")
        return values, {"phase": "PREDICT", "training_performed": False, "package_fingerprint": package["package_fingerprint"],
                        "native_outcome": document, "native_tasks": phases, "sample_ids": ids, "target_names": graph["metadata"]["python_torch_profile"]["target_names"]}
    finally:
        store.clear()


def predict_torch_task(task: dict[str, Any], resolver: Any, node_lookup: Any, store: Any, *, target_names: list[str]) -> dict[str, Any]:
    """Apply a retained fitted owner on native PREDICT rows without target access."""
    from .node_runner import _artifact_id, _build_result, _meta_feature_matrix, _meta_prediction_block, _ordered_oof_specs, _stable_handle, _train_predict_ids

    node = node_lookup(task["node_plan"]["node_id"])
    variant = task.get("variant_id") or "base"
    bundle = store[_stable_handle(_artifact_id(node["id"], variant))]
    _require(bundle["y_transform"] is None, "target-free replay cannot hydrate a target transform")
    estimator = bundle["estimator"]
    if operator_name(node) == TORCH_CLASS:
        _fit, ids = _train_predict_ids(copy.deepcopy(task))
        index = node["metadata"].get("source_index")
        if index is None:
            features = selected_torch_features(resolver, ids, node["metadata"]["source_selection"], include_augmented=False)
        else:
            features = resolver.resolve_source_block(ids, index, include_augmented=False)["values"]
    else:
        specs = _ordered_oof_specs(task.get("prediction_inputs") or {}, suffix="predict",
                                   source_order=node["metadata"].get("prediction_source_order"),
                                   source_ports=node["metadata"].get("prediction_source_ports") or {})
        ids, features = _meta_feature_matrix(specs, node["id"])
    values = np.asarray(estimator.predict(features), dtype=float).reshape(len(ids), 1)
    _require(np.isfinite(values).all(), "finite target-free prediction required")
    block = _meta_prediction_block(node["id"], "PREDICT", variant, "nofold", "final", None, ids, values, target_names)
    return _build_result(task, [block], [], {}, [])


def selected_torch_features(resolver: Any, ids: list[str], selection: list[str], *, include_augmented: bool, fold_label: str | None = None) -> np.ndarray:
    """Project named feature columns in signed order without altering sample scope."""
    resolved = resolver.resolve_feature_blocks(ids, include_augmented=include_augmented, fold_label=fold_label)
    names = list(resolver._dataset.source_names)
    _require(len(selection) == len(set(selection)) and set(selection) <= set(names), "foreign or duplicate selected source")
    _require(not resolved.get("source_masks"), "complete selected sources required")
    by_name = dict(zip(names, resolved["blocks"], strict=True))
    return np.hstack([by_name[name] for name in selection])
