"""Declarative SDK bridge for the closed native four-source early-fusion profile.

IO owns alignment and source descriptors, DAG-ML owns the campaign, and Methods
owns every learned encoder and predictor. This module performs no encoding.
"""

from __future__ import annotations

import copy
import importlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nirs4all.api.result import RunResult
from nirs4all.operators.models.multimodal import MultimodalClassifier, MultimodalRegressor, TensorPCA

PROFILE = "dagml.methods.multimodal.v1"
SOURCE_ORDER = ("nir", "image", "series", "metadata")
TUNABLE_KEYS = frozenset({"model__alpha", "source_weights__image", "transformers__image__n_components"})
PYTHON_CONTROLLER = "controller:methods.python.multimodal"
REPLAY_CONTROLLERS = frozenset(f"controller:methods.{host}.multimodal" for host in ("python", "wasm", "r", "octave"))
SOURCE_CONTRACTS = {
    "nir": ("signal_1d", 2),
    "image": ("rgb_image", 4),
    "series": ("series_mv", 3),
    "metadata": ("tabular_mixed", 2),
}


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _source_encoder_recipe(model: Any, *, allow_source_selection: bool = False) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Validate the shared raw encoder declaration without constructing a model."""
    if model.fusion != "early" or getattr(model, "target_policy", "complete") != "complete" or model.missing_source_policy != "error":
        raise ValueError("Methods multimodal profile requires early fusion, complete targets and complete sources")
    if not allow_source_selection and (not isinstance(model.transformers, Mapping) or tuple(model.transformers) != SOURCE_ORDER):
        raise ValueError(f"Methods multimodal profile requires source order {SOURCE_ORDER}")
    if not isinstance(model.transformers, Mapping) or not model.transformers or set(model.transformers) - set(SOURCE_ORDER):
        raise ValueError(f"Methods multimodal profile requires a nonempty ordered subset of {SOURCE_ORDER}")
    selected = tuple(model.transformers)
    encoders: dict[str, Any] = {}
    if "nir" in selected:
        scaler = model.transformers["nir"]
        if type(scaler) is not StandardScaler or not scaler.with_mean or not scaler.with_std:
            raise ValueError("nir requires StandardScaler(with_mean=True, with_std=True)")
        encoders["nir"] = {"kind": "standard_scaler", "with_mean": True, "with_std": True}
    for name in ("image", "series"):
        if name not in selected:
            continue
        encoder = model.transformers[name]
        if (type(encoder) is not TensorPCA or type(encoder.n_components) is not int
                or encoder.n_components < 1 or encoder.whiten is not False
                or type(encoder.random_state) is not int or not 0 <= encoder.random_state <= 2**32 - 1):
            raise ValueError(f"{name} requires TensorPCA with positive integer n_components, whiten=False and integer random_state")
        encoders[name] = {"kind": "tensor_pca", "n_components": encoder.n_components, "whiten": False, "random_state": encoder.random_state}
    if "metadata" in selected:
        encoders["metadata"] = _metadata_encoder_recipe(model.transformers["metadata"])
    return selected, encoders


def recipe_from_estimator(model: MultimodalRegressor, *, allow_source_selection: bool = False) -> dict[str, Any]:
    """Translate the supported sklearn declarations without fitting them."""
    if type(model) is not MultimodalRegressor or model.backend != "methods":
        raise ValueError("native multimodal execution requires MultimodalRegressor(backend='methods')")
    selected, encoders = _source_encoder_recipe(model, allow_source_selection=allow_source_selection)
    ridge = model.model
    if (type(ridge) is not Ridge or ridge.fit_intercept is not True or ridge.positive is not False
            or ridge.solver != "auto" or ridge.max_iter is not None or ridge.random_state is not None
            or ridge.tol != 1e-4):
        raise ValueError("Methods multimodal profile requires ordinary Ridge(alpha=..., fit_intercept=True, solver='auto')")
    weights = {} if model.source_weights is None else model.source_weights
    if not isinstance(weights, Mapping) or set(weights) - set(selected):
        raise ValueError("source_weights must name only selected sources")
    return {
        "schema_version": 1, "fusion": "early", "source_order": list(selected), "encoders": encoders,
        "source_weights": {name: _finite_nonnegative(weights.get(name, 1.0), f"source_weights.{name}") for name in selected},
        "model": {"method_id": "models.regularized.ridge", "params": {
            "alpha": _finite_nonnegative(ridge.alpha, "Ridge.alpha"), "center_x": True, "center_y": True, "scale_x": False,
        }},
    }


def _metadata_encoder_recipe(mixed: Any) -> dict[str, Any]:
    """Keep the existing mixed-column encoder declaration for selected metadata."""
    if (type(mixed) is not ColumnTransformer or mixed.remainder != "drop"
            or mixed.transformer_weights is not None or mixed.n_jobs is not None
            or len(mixed.transformers) != 2):
        raise ValueError("metadata requires the numeric/category dense ColumnTransformer without remainder or weights")
    numeric_name, numeric, numeric_columns = mixed.transformers[0]
    category_name, category, category_columns = mixed.transformers[1]
    if (numeric_name != "numeric" or category_name != "category" or list(numeric_columns) != [0]
            or list(category_columns) != [1] or type(numeric) is not StandardScaler
            or not numeric.with_mean or not numeric.with_std or type(category) is not OneHotEncoder):
        raise ValueError("metadata requires numeric StandardScaler column 0 and category OneHotEncoder column 1")
    category_params = category.get_params(deep=False)
    if (not isinstance(category_params["categories"], str) or category_params["categories"] != "auto"
            or category_params["handle_unknown"] != "ignore"
            or category_params["sparse_output"] is not False or category_params["drop"] is not None
            or category_params["min_frequency"] is not None or category_params["max_categories"] is not None
            or np.dtype(category_params["dtype"]) != np.dtype("float64")
            or category_params.get("feature_name_combiner", "concat") != "concat"):
        raise ValueError("metadata requires learned categories, float64 dense one-hot, drop=None and handle_unknown='ignore'")
    return {
        "kind": "column_transformer", "numeric_columns": [0], "categorical_columns": [1],
        "with_mean": True, "with_std": True, "handle_unknown": "ignore", "sparse_output": False, "drop": None,
    }


def source_schemas_from_cohort(cohort: Any) -> dict[str, Any]:
    """Capture exact IO schema identities independently of learned native state."""
    if tuple(cohort.sources) != SOURCE_ORDER:
        raise ValueError(f"Methods multimodal input requires ordered sources {SOURCE_ORDER}")
    schemas = {}
    for name, descriptor in zip(SOURCE_ORDER, cohort.schema_descriptors(), strict=True):
        representation, rank = SOURCE_CONTRACTS[name]
        shape = descriptor["shape"][1:]
        native = descriptor["native_representation"]
        if (descriptor["source_id"] != name or descriptor["representation_id"] != representation
                or len(descriptor["shape"]) != rank or any(type(size) is not int or size < 1 for size in shape)
                or (name == "metadata" and shape != [2]) or descriptor["shape"][0] is not None
                or math.prod(shape) > 1_048_576 or native["ragged"] or native["sparse"]):
            raise ValueError(f"Methods multimodal input has unsupported representation or fixed shape for {name}")
        if not np.asarray(cohort.sources[name].presence_mask).all():
            raise ValueError("Methods multimodal input requires every source present")
        dtype = np.dtype(descriptor["dtype"])
        if name != "metadata" and dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError(f"Methods multimodal {name} requires a float32/float64 raw dtype")
        if name == "metadata":
            if dtype.kind not in "UO" or not descriptor["feature_names"] or len(descriptor["feature_names"]) != 2:
                raise ValueError("Methods metadata requires two declared raw Unicode/object numeric/category columns")
            # Category strings retain their exact UTF-8 identity. Numeric column
            # conversion belongs to the Methods binding, never this bridge.
            if any(not isinstance(cell, str) for cell in cohort.sources[name].values[:, 1]):
                raise ValueError("Methods metadata categories must be raw strings")
        identity = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        if not 0 < len(identity.encode("utf-8")) <= 1_048_576:
            raise ValueError(f"Methods multimodal {name} exceeds the source identity byte budget")
        schemas[name] = {
            "representation_id": representation, "input_shape": list(shape), "dtype": descriptor["dtype"],
            "identity": identity,
        }
    return schemas


def methods_model_in_pipeline(pipeline: Any) -> MultimodalRegressor | MultimodalClassifier | None:
    """Find an explicitly selected Methods backend, including generated recipes."""
    visited: set[int] = set()

    def visit(value: Any) -> MultimodalRegressor | MultimodalClassifier | None:
        if isinstance(value, (MultimodalRegressor, MultimodalClassifier)):
            if value.backend not in {"sklearn", "methods"}:
                raise ValueError("Multimodal estimator backend must be 'sklearn' or 'methods'")
            return value if value.backend == "methods" else None
        if isinstance(value, (Mapping, list, tuple)):
            if id(value) in visited:
                return None
            visited.add(id(value))
            children = value.values() if isinstance(value, Mapping) else value
            for child in children:
                found = visit(child)
                if found is not None:
                    return found
        return None

    return visit(pipeline)


def validate_training_profile(pipeline: Any, cohort: Any, *, refit: bool, allow_source_selection: bool = False) -> tuple[list[Any], Any, MultimodalRegressor]:
    """Refuse unsupported campaigns before either grid or global HPO fits."""
    from sklearn.model_selection import GroupKFold

    from .steps import _split_pipeline

    model = methods_model_in_pipeline(pipeline)
    if not isinstance(model, MultimodalRegressor):
        raise ValueError("Methods multimodal backend was not declared")
    recipe_from_estimator(model, allow_source_selection=allow_source_selection)
    source_schemas_from_cohort(cohort)
    steps, splitter = _split_pipeline(pipeline)
    if (len(steps) != 1 or not isinstance(steps[0], dict) or steps[0].get("model") is not model
            or type(splitter) is not GroupKFold or splitter.n_splits != 3 or cohort.groups is None):
        raise ValueError("Methods multimodal profile requires [GroupKFold(3), {'model': MultimodalRegressor(...)}]")
    if set(steps[0]) - {"model", "_grid_", "name"}:
        raise ValueError("Methods multimodal profile supports only its three public parameter keys through _grid_ or global tuning")
    grid = steps[0].get("_grid_", {})
    if not isinstance(grid, Mapping) or set(grid) - TUNABLE_KEYS:
        raise ValueError("Methods multimodal grid supports only alpha, image weight and image components")
    if getattr(cohort, "_generated_view_store", None) is not None:
        raise ValueError("Methods multimodal profile requires fixed complete source buffers")
    targets = np.asarray(cohort.y)
    if (not refit or cohort.y is None or tuple(cohort.target_names) != ("y",)
            or not np.asarray(cohort.target_mask).all() or targets.dtype.kind not in "fiu"
            or not np.isfinite(targets).all() or targets.ndim not in (1, 2)
            or (targets.ndim == 2 and targets.shape[1] != 1)):
        raise ValueError("Methods multimodal profile requires refit=True and one complete finite target named 'y'")
    if any(partition not in {"train", "test"} for partition in cohort.partitions):
        raise ValueError("Methods multimodal training accepts train/test partitions only")
    # The step identity comparison above widens mypy's local type to Any.
    # The explicit isinstance guard still guarantees the returned declaration.
    return steps, splitter, cast(MultimodalRegressor, model)


def fit_declared_methods_model(model: MultimodalRegressor, blocks: list[Any], y: Any, *, source_schemas: Any) -> None:
    """Fit a native candidate and replace the previous predictor only on success."""
    recipe = recipe_from_estimator(model, allow_source_selection=True)
    if source_schemas is None:
        raise ValueError("Methods fit requires explicit IO-derived source_schemas; public run() supplies them")
    selected = tuple(recipe["source_order"])
    values = model._validate_blocks(blocks, selected)
    targets = np.asarray(y)
    if targets.ndim not in (1, 2) or (targets.ndim == 2 and targets.shape[1] != 1):
        raise ValueError("Methods multimodal profile supports exactly one numeric target")
    if targets.dtype.kind not in "fiu" or not np.isfinite(targets).all():
        raise ValueError("Methods multimodal targets must be complete finite numbers")
    pipeline_type = getattr(importlib.import_module("n4m"), "MultimodalPipeline", None)
    if not callable(pipeline_type):
        raise ImportError("installed nirs4all-methods lacks MultimodalPipeline; install the matching native encoder build")
    previous = getattr(model, "native_pipeline_", None)
    selected_schemas = {name: source_schemas[name] for name in selected}
    native = pipeline_type(recipe, selected_schemas)
    try:
        native.fit(dict(zip(selected, values, strict=True)), targets)
        fitted = {
            "native_pipeline_": native,
            "source_schemas_": copy.deepcopy(selected_schemas),
            "source_names_": selected,
            "input_shapes_": {name: tuple(np.shape(block)[1:]) for name, block in zip(selected, values, strict=True)},
            "target_ndim_": targets.ndim,
            "n_outputs_": 1,
        }
        # Construct the complete replacement first. Failure in native fitting or
        # metadata preparation must leave every previous fitted attribute intact.
        model.__dict__ = {**model.__dict__, **fitted}
    except BaseException:
        native.close()
        raise
    if previous is not None:
        previous.close()


def bind_methods_dsl(dsl: dict[str, Any], model: MultimodalRegressor, cohort: Any) -> dict[str, Any]:
    """Replace a single host declaration with the independently attested native operator."""
    from dag_ml.multimodal_methods import MethodsMultimodalController

    recipe = recipe_from_estimator(model)
    schemas = source_schemas_from_cohort(cohort)
    steps = dsl["pipeline"]
    if len(steps) != 1:
        raise ValueError("Methods multimodal profile supports one model and no external transforms or branches")
    original = steps[0]
    bindings = dsl.get("data_bindings", [])
    if len(bindings) != 1 or not isinstance(bindings[0].get("node_id"), str) or not bindings[0]["node_id"]:
        raise ValueError("Methods multimodal declaration requires one compiler-resolved model binding")
    model_id = bindings[0]["node_id"]
    if "id" in original and original["id"] != model_id:
        raise ValueError("Methods multimodal explicit model identity differs from its native data binding")
    generator = original.get("generators")
    step: dict[str, Any] = {
        "id": original.get("id", model_id),
        "kind": "model", "operator": {"type": "N4mMultimodalPipeline", "recipe": recipe, "source_schemas": schemas},
        "params": {}, "metadata": {"controller_id": "controller:methods.python.multimodal"},
    }
    if generator:
        step["generators"] = copy.deepcopy(generator)
    bound = copy.deepcopy(dsl)
    bound["pipeline"] = [step]
    # This raw-source profile never fits generated or augmented observations.
    # Bind the policy explicitly because the native default permits them.
    binding = bound["data_bindings"][0]
    binding["view_policy"] = {**binding.get("view_policy", {}), "include_augmented_train": False, "include_refit_test_view": True}
    # Only metadata is needed to register the compiler. No native estimator is
    # constructed or fitted by this controller's declaration-only constructor.
    declaration = MethodsMultimodalController(
        operators={"declaration": step["operator"]},
        sources=controller_sources(cohort, schemas), targets=None, target_names=tuple(cohort.target_names), allow_fit=False,
    )
    try:
        return {"dsl": bound, "manifest": copy.deepcopy(declaration.manifest)}
    finally:
        declaration.close()


def controller_sources(cohort: Any, schemas: Mapping[str, Any]) -> dict[str, Any]:
    """Pass raw IO blocks and their exact independently derived descriptors."""
    return {name: {"sample_ids": list(cohort.sample_ids), "descriptor": schemas[name], "values": cohort.sources[name].values}
            for name in SOURCE_ORDER}


def _controller_id_for_node(node: Mapping[str, Any], *, allow_fit: bool, binding_controller_id: str | None) -> str:
    """Keep the signed producer owner for replay and Python ownership for fitting."""
    controller_id = PYTHON_CONTROLLER if binding_controller_id is None else binding_controller_id
    if controller_id not in REPLAY_CONTROLLERS or (allow_fit and controller_id != PYTHON_CONTROLLER):
        raise ValueError("Methods multimodal requires an exact closed producer owner; fitting requires Python ownership")
    declared = node.get("metadata", {}).get("controller_id")
    if declared is not None and declared != controller_id:
        raise ValueError("Methods multimodal selected owner differs from the signed graph declaration")
    return controller_id


def controller_for_graph(graph: Mapping[str, Any], cohort: Any, *, allow_fit: bool, node_params: Any = None,
                         binding_source_ids: Any = None, binding_controller_id: str | None = None) -> Any:
    """Create the installed DAG controller, with explicit native binding source IDs."""
    from dag_ml.multimodal_methods import MethodsMultimodalController

    from nirs4all.data.multimodal import MultimodalSpectroDataset

    from .envelope import source_ids

    models = [node for node in graph["nodes"] if node["kind"] == "model"]
    if any((node.get("operator") or {}).get("type") == "N4mMultimodalClassifierPipeline" for node in models):
        from .methods_classification import classifier_controller_for_graph

        return classifier_controller_for_graph(graph, cohort, allow_fit=allow_fit, node_params=node_params,
            binding_source_ids=binding_source_ids, binding_controller_id=binding_controller_id)
    if any(node["operator"].get("type") == "N4mRolePipeline" for node in models):
        from dag_ml.multimodal_topology import MethodsTopologyController

        schemas = source_schemas_from_cohort(cohort)
        current_source_ids = tuple(source_ids(MultimodalSpectroDataset(cohort)))
        if binding_source_ids is not None and tuple(binding_source_ids) != current_source_ids:
            raise ValueError("current source order differs from the signed native topology binding")
        raw_nodes = [node for node in models if node["operator"].get("type") == "N4mMultimodalPipeline"]
        if not raw_nodes or any(node["operator"].get("source_schemas") != schemas for node in raw_nodes):
            raise ValueError("Methods topology raw source schemas differ from the signed declarations")
        return MethodsTopologyController(
            operators={node["id"]: node["operator"] for node in models}, sources=controller_sources(cohort, schemas),
            targets={"sample_ids": list(cohort.sample_ids), "values": cohort.y} if allow_fit else None,
            target_names=tuple(cohort.target_names) if allow_fit else ("y",), allow_fit=allow_fit,
            source_ids=current_source_ids, node_params=node_params, edges=graph["edges"],
        )
    if len(models) != 1 or len(graph["nodes"]) != 1:
        raise ValueError("Methods multimodal profile requires exactly one native model node")
    controller_id = _controller_id_for_node(models[0], allow_fit=allow_fit, binding_controller_id=binding_controller_id)
    schemas = source_schemas_from_cohort(cohort)
    operator = models[0]["operator"]
    if operator.get("type") != "N4mMultimodalPipeline" or operator.get("source_schemas") != schemas:
        raise ValueError("Methods multimodal source schema differs from the signed model declaration")
    current_source_ids = tuple(source_ids(MultimodalSpectroDataset(cohort)))
    if binding_source_ids is not None and tuple(binding_source_ids) != current_source_ids:
        raise ValueError("current source order differs from the signed native data binding")
    return MethodsMultimodalController(
        operators={models[0]["id"]: operator}, sources=controller_sources(cohort, schemas),
        targets={"sample_ids": list(cohort.sample_ids), "values": cohort.y} if allow_fit else None,
        target_names=tuple(cohort.target_names) if allow_fit else ("y",), allow_fit=allow_fit,
        controller_id=controller_id,
        source_ids=current_source_ids,
        node_params=node_params,
    )


class MethodsMultimodalRunResult(RunResult):
    """Scored public result retaining complete native portable state only."""

    methods_multimodal_tuning_evidence: dict[str, Any]
    methods_multimodal_search_request: dict[str, Any]
    _dagml_graph: dict[str, Any]
    classes_: np.ndarray
    classification: dict[str, Any]
    classification_probability_blocks: list[dict[str, Any]]
    structural_tuning_training_request: dict[str, Any]
    structural_tuning_training_outcome: dict[str, Any]

    def __init__(self, projected: RunResult, *, outcome: Any, package: Any, audit: Any, request: Any, training_inputs: Any) -> None:
        super().__init__(predictions=projected.predictions, per_dataset=projected.per_dataset)
        self._dagml_score_set = copy.deepcopy(outcome.to_dict()["score_set"])
        self._methods_multimodal_outcome = outcome
        self._methods_multimodal_package = package
        self.methods_multimodal_audit = copy.deepcopy(audit)
        self.methods_multimodal_training_request = copy.deepcopy(request)
        self.methods_multimodal_training_inputs = copy.deepcopy(training_inputs)
        self.native_profile = PROFILE

    def export(self, output_path: str | Path, format: str = "n4a", source: dict[str, Any] | None = None,
               chain_id: str | None = None, *, compatibility: str | None = None) -> Path:
        """Publish the captured encoder/Ridge state without training again."""
        import os
        import tempfile

        from nirs4all.api.portable_archive import read_portable_predictor_archive_v2, write_portable_predictor_archive_v2

        if format != "n4a" or source is not None or chain_id is not None or compatibility is not None:
            raise ValueError("Methods multimodal export supports the captured Core Archive V2 only")
        path = Path(output_path)
        if path.suffix.lower() != ".n4a":
            raise ValueError("Methods multimodal export requires a .n4a destination")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".multimodal-publish-", dir=path.parent) as directory:
            candidate = Path(directory) / path.name
            write_portable_predictor_archive_v2(candidate, archive_id="archive:nirs4all.methods.multimodal",
                                                outcome=self._methods_multimodal_outcome, package=self._methods_multimodal_package)
            read_portable_predictor_archive_v2(candidate)
            os.replace(candidate, path)
        return path

    def close(self) -> None:
        """Training controllers have already released their native handles."""


def run_methods_multimodal(pipeline: Any, spectro: Any, *, name: str, random_state: int | None, refit: bool) -> MethodsMultimodalRunResult:
    """Execute the common native CV/SELECT/REFIT campaign with raw-source callbacks."""
    from nirs4all.data.multimodal import MultimodalSpectroDataset

    from .cli_runner import assemble_cv_refit_dsl
    from .envelope import build_envelope
    from .folds import _build_folds, _split_group_grain
    from .identity import mint_identity
    from .raw_training_lowerer import _array_content_fingerprint, _core_relation_fingerprint, _data_contracts_from_campaign, _output_request_for_node, _training_influence_manifest
    from .resources import current_execution_resources
    from .training_contracts import DagMLTrainingRequestSpec, assemble_training_request

    if isinstance(methods_model_in_pipeline(pipeline), MultimodalClassifier):
        from .structural_classification import run_fixed_classifier

        return cast(MethodsMultimodalRunResult, run_fixed_classifier(pipeline, spectro, name=name, random_state=random_state, refit=refit))
    if not isinstance(spectro, MultimodalSpectroDataset):
        raise TypeError("Methods multimodal run requires an IO MultimodalDataset")
    cohort = spectro.cohort
    steps, splitter, model = validate_training_profile(pipeline, cohort, refit=refit)
    native = importlib.import_module("dag_ml")
    if not callable(getattr(importlib.import_module("n4m"), "MultimodalPipeline", None)):
        raise ImportError("installed nirs4all-methods lacks MultimodalPipeline; install the matching native encoder build")
    pool = spectro.index_column("sample", {"partition": "train"})
    identity = mint_identity(spectro)
    folds = _build_folds(splitter, spectro, pool, set())
    groups = _split_group_grain(splitter, spectro, pool)
    if groups is None:
        raise ValueError("Methods multimodal training requires explicit sample groups")
    envelope = build_envelope(spectro, identity, sample_ints=pool, group_by_sample=groups)
    declaration = bind_methods_dsl(assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="methods-multimodal", n_splits=3), model, cohort)
    artifact = native.compile_pipeline_dsl_artifact_with_controllers(declaration["dsl"], [declaration["manifest"]])
    graph = artifact.graph.to_dict()
    campaign = artifact.campaign_template.to_dict()
    seed = random_state if random_state is not None else 17
    campaign["root_seed"] = seed
    envelope["relation_fingerprint"] = _core_relation_fingerprint(envelope["coordinator_relations"], native)
    envelope["data_content_fingerprint"] = spectro.content_hash(sample_rows=pool)
    envelope["target_content_fingerprint"] = _array_content_fingerprint("y", cohort.y[pool])
    test = spectro.index_column("sample", {"partition": "test"})
    if test:
        test_envelope = build_envelope(spectro, identity, sample_ints=test)
        envelope.update(native.attach_predict_cohort_to_envelope(envelope, {
            "role": "external_test", "relations": test_envelope["coordinator_relations"], "target_names": ["y"],
            "data_content_fingerprint": spectro.content_hash(sample_rows=test),
            "target_content_fingerprint": _array_content_fingerprint("y", cohort.y[test]),
        }).to_dict())
    data_envelopes, data_identities = _data_contracts_from_campaign(campaign, envelope)
    output = _output_request_for_node(graph, target_names=["y"])
    resources = current_execution_resources()
    request = assemble_training_request(DagMLTrainingRequestSpec(
        request_id="training:methods.multimodal", plan_id="plan:methods.multimodal", graph=graph, campaign=campaign,
        controller_manifests=[declaration["manifest"]], data_identities=data_identities, output_requests=[output],
        seed=seed, cv_artifacts="discard", fitted_artifacts="portable_required", cpu_threads=resources.cpu_threads,
        gpu_devices=resources.gpu_devices, selection_required_metric_level="sample", selection_evaluation_scope="oof",
    ))
    influence = _training_influence_manifest(graph, campaign, folds, identity, group_by_sample=groups, selection_metric="rmse")
    return execute_methods_training(
        spectro=spectro, identity=identity, envelope=envelope, request=request, data_envelopes=data_envelopes,
        influence=influence, output=output, name=name, binding_source_ids=declaration["dsl"]["data_bindings"][0]["source_ids"],
    )


def execute_methods_training(*, spectro: Any, identity: Any, envelope: dict[str, Any], request: dict[str, Any],
                             data_envelopes: dict[str, Any], influence: dict[str, Any], output: dict[str, Any],
                             name: str, binding_source_ids: list[str]) -> MethodsMultimodalRunResult:
    """Capture the same complete portable winner for fixed and structural campaigns."""
    from .result import _scores_to_run_result

    native = importlib.import_module("dag_ml")
    graph = request["graph"]
    cohort = spectro.cohort
    controller = controller_for_graph(graph, cohort, allow_fit=True,
                                      binding_source_ids=binding_source_ids)
    frames = []
    training = None

    def callback(task: dict[str, Any]) -> dict[str, Any]:
        result = cast(dict[str, Any], controller.operator(task))
        frames.append({**result, "variant_id": task.get("variant_id")})
        return result

    try:
        training = native.execute_training(request, data_envelopes, envelope["coordinator_relations"], influence, callback,
                                           artifact_callback=controller.artifact, outcome_id="outcome:methods.multimodal",
                                           run_id="run:methods.multimodal", bundle_id="bundle:methods.multimodal")
        outcome = training.outcome
        document = outcome.to_dict()
        package = training.export_portable_predictor_package("predictor:methods.multimodal", fitted_artifact_mode="portable_required", artifact_load_mode="native_portable")
        for average in document.get("oof_averages", []):
            frames.append({"aggregated_predictions": [average["predictions"]], "regression_targets": [average["y_true"]]})
        by_variant: dict[Any, list[Any]] = {}
        for frame in frames:
            variant = frame.get("variant_id", document["selected_variant_id"])
            by_variant.setdefault(variant, []).append(frame)
        classification = any((node.get("operator") or {}).get("type") == "N4mMultimodalClassifierPipeline" for node in graph["nodes"])
        vocabulary = None
        profile = PROFILE
        if classification:
            from .methods_classification import PROFILE as CLASSIFICATION_PROFILE
            from .methods_classification import graph_vocabulary

            vocabulary = graph_vocabulary(graph)
            profile = CLASSIFICATION_PROFILE
            # The probabilities port is auxiliary: it must never replace the
            # genuine y_hat labels in the public per-sample report projection.
            by_variant = {variant: [{**frame, "predictions": [block for block in frame.get("predictions", [])
                            if block.get("producer_port") == "y_hat"]} for frame in variant_frames]
                          for variant, variant_frames in by_variant.items()}
        projected = _scores_to_run_result(document["score_set"], spectro.name,
            "MethodsMultimodalPLSLogistic" if classification else "MethodsMultimodalRidge", producer=output["node_id"],
            metric=request["options"]["selection"]["metric"]["name"] if classification else "rmse",
            task_type=("binary_classification" if len(vocabulary["class_labels"]) == 2 else "multiclass_classification") if vocabulary else "regression",
            config_name=name, results_by_variant=by_variant, identity=identity, classification_vocabulary=vocabulary)
        for info in projected.per_dataset.values():
            info.update(native_profile=profile, engine="dag-ml", refit_enabled=True)
            if vocabulary is not None:
                info["classification"] = copy.deepcopy(vocabulary)
        result = MethodsMultimodalRunResult(projected, outcome=outcome, package=package, audit=controller.audit, request=request,
            training_inputs={"data_envelopes": data_envelopes, "relations": envelope["coordinator_relations"], "training_influence": influence})
        result.native_profile = profile
        if vocabulary is not None:
            # Preserve genuine native probability blocks and their signed
            # column labels; no probability calculation occurs in the SDK.
            result.classification_probability_blocks = [copy.deepcopy(block) for frame in frames
                for block in frame.get("predictions", []) if block.get("producer_port") == "probabilities"]
            expected_columns = [f"class:{label}" for label in vocabulary["class_labels"]]
            if any(block.get("target_names") != expected_columns for block in result.classification_probability_blocks):
                raise ValueError("native classifier probability columns differ from the signed class vocabulary")
            result.classes_ = np.asarray(vocabulary["label_names"])
            result.classification = copy.deepcopy(vocabulary)
        return result
    finally:
        try:
            if training is not None:
                training.detach()
        finally:
            controller.close()


def is_methods_multimodal_package(package: Mapping[str, Any]) -> bool:
    """Recognize the additive portable kind; native validation grants trust."""
    records = package.get("execution_bundle", {}).get("refit_artifacts", [])
    return isinstance(records, list) and any(isinstance(record, Mapping)
                                            and record.get("artifact", {}).get("kind") in {"methods_multimodal_pipeline", "methods_multimodal_classifier_pipeline"}
                                            for record in records)


def predict_methods_multimodal_archive(path: str | Path, data: Any, *, methods_library_path: str,
                                      outcome_id: str, run_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Replay current raw sources through the validated package without FIT/HPO."""
    from nirs4all.api.portable_archive import read_portable_predictor_archive_v2
    from nirs4all.data.multimodal import MultimodalSpectroDataset

    from .core_archive_replay import _decode_prediction, _resolve_methods_library_identity
    from .dataset import _materialize_dataset
    from .envelope import build_envelope
    from .identity import mint_identity
    from .raw_replay_lowerer import _requirements, _source_outcome_fingerprint
    from .raw_training_lowerer import _core_relation_fingerprint

    package = read_portable_predictor_archive_v2(path)
    document = package.to_dict()
    if not is_methods_multimodal_package(document):
        raise ValueError("archive is not a complete Methods multimodal package")
    # The ctypes binding owns one library. Explicit overrides must attest that
    # exact loaded library, rather than silently select a second native engine.
    actual_library = _resolve_methods_library_identity(None)
    if actual_library != _resolve_methods_library_identity(methods_library_path):
        raise ValueError("Methods multimodal replay requires the exact library selected by the installed n4m binding")
    spectro = _materialize_dataset(data)
    if not isinstance(spectro, MultimodalSpectroDataset):
        raise TypeError("Methods multimodal replay requires an IO MultimodalDataset or its serialized declaration")
    cohort = spectro.cohort
    schemas = source_schemas_from_cohort(cohort)
    plan = document["effective_plan"]
    graph = plan["graph_plan"]["graph"]
    model_nodes = [node for node in graph["nodes"] if node["kind"] == "model"]
    classification = any((node.get("operator") or {}).get("type") == "N4mMultimodalClassifierPipeline" for node in model_nodes)
    raw_type = "N4mMultimodalClassifierPipeline" if classification else "N4mMultimodalPipeline"
    meta_type = "N4mRoleClassifierPipeline" if classification else "N4mRolePipeline"
    raw_nodes = [node for node in model_nodes if node["operator"].get("type") == raw_type]
    topology = any(node["operator"].get("type") == meta_type for node in model_nodes)
    if (not raw_nodes or any(node["operator"].get("source_schemas") != schemas for node in raw_nodes)
            or (not topology and len(model_nodes) != 1)):
        raise ValueError("current source schema differs from the signed archived Methods multimodal declaration")
    bindings = document["output_bindings"]
    if len(bindings) != 1 or bindings[0]["target_names"] != ["y"]:
        raise ValueError("Methods multimodal replay requires one output target named 'y'")
    identity = mint_identity(spectro)
    current = build_envelope(spectro, identity)
    relations = current["coordinator_relations"]
    relation_fingerprint = _core_relation_fingerprint(relations, importlib.import_module("dag_ml"))
    requirements = _requirements(document["execution_bundle"])
    envelopes = {key: {
        "schema_version": 1, "schema_fingerprint": requirement["schema_fingerprint"],
        "plan_fingerprint": requirement["plan_fingerprint"], "relation_fingerprint": relation_fingerprint,
        "data_content_fingerprint": spectro.content_hash(), "target_content_fingerprint": None,
        "coordinator_relations": relations,
    } for key, requirement in requirements.items()}
    native = importlib.import_module("dag_ml")
    envelopes = {key: native.attach_predict_cohort_to_envelope(envelope, {
        "role": "inference", "relations": relations, "target_names": ["y"],
        "data_content_fingerprint": envelope["data_content_fingerprint"], "target_content_fingerprint": None,
    }).to_dict() for key, envelope in envelopes.items()}
    request = native.sign_training_replay_request({
        "schema_version": 1, "request_id": "replay:nirs4all.methods.multimodal",
        "source_outcome_fingerprint": _source_outcome_fingerprint(document), "phase": "PREDICT",
        "data_envelope_keys": sorted(envelopes), "output_binding_ids": [bindings[0]["binding_id"]],
        "request_fingerprint": "0" * 64,
    })
    params = {key: node.get("params", {}) for key, node in plan["node_plans"].items()}
    selected = plan["node_plans"][raw_nodes[0]["id"]]
    if len(selected["data_bindings"]) != 1:
        raise ValueError("Methods multimodal replay requires one signed raw data binding")
    if topology:
        for node in model_nodes:
            node_plan = plan["node_plans"][node["id"]]
            expected_owner = (("controller:methods.python.multimodal.classification" if node in raw_nodes else "controller:methods.python.classification")
                              if classification else ("controller:methods.python.multimodal" if node in raw_nodes else "controller:methods.python.regression"))
            if node_plan["controller_id"] != expected_owner or node.get("metadata", {}).get("controller_id") != expected_owner:
                raise ValueError("Methods topology replay requires each exact signed native producer owner")
            if node in raw_nodes and (len(node_plan["data_bindings"]) != 1
                                      or node_plan["data_bindings"][0]["source_ids"] != selected["data_bindings"][0]["source_ids"]):
                raise ValueError("Methods topology replay requires the complete shared signed raw source binding")
    controller = controller_for_graph(graph, cohort, allow_fit=False, node_params=params,
                                      binding_source_ids=selected["data_bindings"][0]["source_ids"],
                                      binding_controller_id=selected["controller_id"])
    try:
        outcome = native.replay_loaded_predictor_package(
            package, request, envelopes, {}, controller.operator, outcome_id=outcome_id, run_id=run_id,
            artifact_callback=controller.artifact, trusted_controller_manifests=getattr(controller, "manifests", [controller.manifest]),
        )
        evidence = outcome.to_dict()
        physical_ids = tuple(next(iter(envelopes.values()))["predict_cohort"]["physical_sample_ids"])
        values = _decode_prediction(evidence, physical_ids, target_names=("y",))
        producer = bindings[0]["node_id"]
        alignment = native.align_named_source_rows({
            "sample_ids": list(cohort.sample_ids), "required_source_ids": [producer],
            "sources": [{"source_id": producer, "sample_ids": list(physical_ids)}],
        })
        values = values[alignment["sources"][0]["row_indices"]]
        profile = PROFILE
        vocabulary = None
        if classification:
            from .methods_classification import PROFILE as CLASSIFICATION_PROFILE
            from .methods_classification import decode_labels, graph_vocabulary

            vocabulary = graph_vocabulary(graph)
            values = decode_labels(values, vocabulary)
            profile = CLASSIFICATION_PROFILE
        audit = copy.deepcopy(controller.audit)
        if any(event.get("operation") in {"fit", "FIT_CV", "REFIT"} for event in audit):
            raise RuntimeError("Methods multimodal inference unexpectedly performed training")
        return values, {"engine": "core-native", "native_profile": profile, "classification": vocabulary, "archive_path": str(path),
                        "archive_schema_version": 2, "sample_ids": list(cohort.sample_ids), "target_names": ["y"],
                        "outcome_id": outcome_id, "run_id": run_id, "training_performed": False, "methods_multimodal_audit": audit}
    finally:
        controller.close()
