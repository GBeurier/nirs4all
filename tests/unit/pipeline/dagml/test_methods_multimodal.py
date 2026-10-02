"""Declaration-only guards and unchanged sklearn defaults; no numerical substitutes."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nirs4all.operators.models.multimodal import MultimodalRegressor, TensorPCA
from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component
from nirs4all.pipeline.dagml.methods_multimodal import _controller_id_for_node, methods_model_in_pipeline, recipe_from_estimator, source_schemas_from_cohort, validate_training_profile


def _model(**params: Any) -> MultimodalRegressor:
    return MultimodalRegressor({
        "nir": StandardScaler(), "image": TensorPCA(3, random_state=23), "series": TensorPCA(1, random_state=0),
        "metadata": ColumnTransformer([("numeric", StandardScaler(), [0]),
                                       ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [1])]),
    }, Ridge(), backend="methods").set_params(**params)


def _cohort() -> Any:
    ids = ["a", "b", "c", "d"]
    return MultimodalDataset({
        "nir": TensorSource(np.zeros((4, 7)), ids, representation_id="signal_1d", axis_units={"wavelength": "nm"}),
        "image": TensorSource(np.zeros((4, 3, 5, 3)), ids, representation_id="rgb_image"),
        "series": TensorSource(np.zeros((4, 11, 4)), ids, representation_id="series_mv"),
        "metadata": TensorSource(np.asarray([["1.2", "A"], ["2.3", "B"], ["0.1", "A"], ["3.1", "B"]]), ids,
                                 representation_id="tabular_mixed", feature_names=["temperature", "instrument"]),
    }, sample_ids=ids, y=np.arange(4.0), groups=ids)


def test_backend_default_clone_serialization_need_no_native_import(monkeypatch: Any) -> None:
    import importlib

    original_import = importlib.import_module

    def guard(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"n4m", "nirs4all_core", "dag_ml.multimodal_methods"}:
            raise AssertionError("declaration loaded an optional native runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", guard)
    model = MultimodalRegressor({"nir": StandardScaler()}, Ridge())
    assert model.backend == "sklearn" and clone(model).backend == "sklearn"
    historical_state = model.__getstate__()
    historical_state.pop("backend")
    restored_historical = object.__new__(MultimodalRegressor)
    restored_historical.__setstate__(historical_state)
    assert restored_historical.backend == "sklearn" and clone(restored_historical).backend == "sklearn"
    for backend in ("sklearn", "methods"):
        declared = _model(backend=backend)
        restored = deserialize_component(json.loads(json.dumps(serialize_component(declared))))
        assert restored.backend == backend and clone(restored).backend == backend


def test_generic_fixed_shapes_counts_seeds_and_exact_descriptor_identity() -> None:
    model, cohort = _model(), _cohort()
    recipe = recipe_from_estimator(model)
    assert recipe["encoders"]["image"]["n_components"] == 3
    assert recipe["encoders"]["series"]["random_state"] == 0
    assert recipe["model"]["params"]["scale_x"] is False
    schemas = source_schemas_from_cohort(cohort)
    assert schemas["image"]["input_shape"] == [3, 5, 3]
    assert schemas["series"]["input_shape"] == [11, 4]
    assert schemas["metadata"]["dtype"] == str(cohort.sources["metadata"].values.dtype)
    for name, descriptor in zip(cohort.sources, cohort.schema_descriptors(), strict=True):
        assert json.loads(schemas[name]["identity"]) == descriptor
        assert schemas[name]["identity"] == json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        assert descriptor["shape"][0] is None


@pytest.mark.parametrize("params", [
    {"fusion": "intermediate"}, {"target_policy": "per_target"}, {"missing_source_policy": "zero_with_indicator"},
    {"transformers__image__whiten": True}, {"transformers__image__n_components": 0},
    {"transformers__image__n_components": 0.9}, {"transformers__image__n_components": True},
    {"transformers__series__random_state": None}, {"transformers__series__random_state": -1},
    {"transformers__series__random_state": 2**32}, {"model__fit_intercept": False},
    {"model__positive": True}, {"model__solver": "svd"}, {"model__alpha": float("nan")},
    {"source_weights__image": -1}, {"source_weights__image": True},
])
def test_unsupported_declarations_fail_without_fit(params: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        recipe_from_estimator(_model(**params))


@pytest.mark.parametrize("params", [
    {"handle_unknown": "error"}, {"sparse_output": True}, {"drop": "first"},
    {"min_frequency": 2}, {"max_categories": 2}, {"categories": [["A", "B"]]},
    {"categories": np.asarray([["A", "B"]])}, {"dtype": np.float32},
])
def test_unsupported_category_policy_is_refused(params: Any) -> None:
    model = _model()
    model.transformers["metadata"].transformers[1][1].set_params(**params)
    with pytest.raises(ValueError, match="metadata"):
        recipe_from_estimator(model)


def test_direct_methods_fit_requires_declared_io_schema_before_native_import() -> None:
    cohort = _cohort()
    with pytest.raises(ValueError, match="source_schemas"):
        _model().fit(cohort.source_values(), cohort.y)


def test_source_order_is_a_contract() -> None:
    model = _model()
    model.transformers = dict(reversed(list(model.transformers.items())))
    with pytest.raises(ValueError, match="source order"):
        recipe_from_estimator(model)


def test_backend_detection_handles_shared_and_cyclic_constructor_mappings() -> None:
    controls: dict[str, Any] = {"warm_start": True}
    controls["nested"] = controls
    assert methods_model_in_pipeline([{"model": Ridge(), "refit_params": controls}, controls]) is None
    model = _model()
    assert methods_model_in_pipeline([controls, {"model": model}]) is model


@pytest.mark.parametrize("splitter,controls,refit", [
    (KFold(3), {}, True), (GroupKFold(2), {}, True),
    (GroupKFold(3), {"train_params": {"alpha": 2.0}}, True),
    (GroupKFold(3), {"refit_params": {"alpha": 2.0}}, True),
    (GroupKFold(3), {"finetune_params": {}}, True),
    (GroupKFold(3), {"_grid_": {"model__solver": ["svd"]}}, True),
    (GroupKFold(3), {}, False),
])
def test_closed_campaign_guards_precede_global_hpo(splitter: Any, controls: Any, refit: bool) -> None:
    with pytest.raises(ValueError, match="Methods multimodal"):
        validate_training_profile([splitter, {"model": _model(), **controls}], _cohort(), refit=refit)


def test_closed_campaign_accepts_named_generic_declaration() -> None:
    model, cohort = _model(), _cohort()
    steps, splitter, found = validate_training_profile([GroupKFold(3), {"model": model, "name": "native"}], cohort, refit=True)
    assert found is model and len(steps) == 1 and splitter.n_splits == 3


@pytest.mark.parametrize("host", ["python", "wasm", "r", "octave"])
def test_replay_keeps_exact_signed_producer_owner_without_rewriting(host: str) -> None:
    owner = f"controller:methods.{host}.multimodal"
    node = {"metadata": {"controller_id": owner}}
    assert _controller_id_for_node(node, allow_fit=False, binding_controller_id=owner) == owner
    assert node == {"metadata": {"controller_id": owner}}


@pytest.mark.parametrize("owner", ["controller:methods.wasm.multimodal", "controller:methods.r.multimodal",
                                   "controller:methods.octave.multimodal", "controller:methods.python", "controller:methods.fake.multimodal"])
def test_fitting_refuses_foreign_or_unregistered_multimodal_owners(owner: str) -> None:
    with pytest.raises(ValueError, match="exact closed producer owner"):
        _controller_id_for_node({"metadata": {"controller_id": owner}}, allow_fit=True, binding_controller_id=owner)


def test_replay_refuses_selected_owner_mismatch_and_unregistered_owner() -> None:
    node = {"metadata": {"controller_id": "controller:methods.r.multimodal"}}
    with pytest.raises(ValueError, match="signed graph declaration"):
        _controller_id_for_node(node, allow_fit=False, binding_controller_id="controller:methods.python.multimodal")
    with pytest.raises(ValueError, match="exact closed producer owner"):
        _controller_id_for_node(node, allow_fit=False, binding_controller_id="controller:methods.fake.multimodal")
