"""Explicit identities and verbatim native fit weights; no host weight solver."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.operators.models.multimodal import MultimodalRegressor
from nirs4all.pipeline.dagml.envelope import build_envelope
from nirs4all.pipeline.dagml.experimental_units import (
    apply_experimental_unit_contract,
    experimental_unit_contract,
    experimental_unit_metadata,
    native_fit_weights,
    require_sample_weight_support,
    require_weighted_context,
    weighted_fit_kwargs,
)
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml_bridge import _SOURCE_RECIPE_CONTROLLER_ID, controller_manifests, pipeline_to_dsl


class _Identity:
    def to_wire(self, row: int) -> str:
        return ["sample:zeta", "sample:alpha", "sample:beta", "sample:eta"][row]

    def to_int(self, sample: str) -> int:
        return [self.to_wire(row) for row in range(4)].index(sample)


class KwargsOnly:
    def fit(self, X: Any, y: Any, **kwargs: Any) -> KwargsOnly:
        raise AssertionError("preflight must not fit a kwargs-only estimator")


def _dataset(*, partial: bool = False, units: list[str] | None = None, declared: bool = True) -> MultimodalSpectroDataset:
    ids = ["zeta", "alpha", "beta", "eta"]
    y = np.asarray([[2., 3.], [2., 3.], [8., 9.], [8., 9.]]) if partial else np.asarray([2., 2., 8., 8.])
    cohort = MultimodalDataset(
        {"nir": TensorSource(np.asarray([[1., 2.], [4., 7.], [8., 9.], [9., 6.]]), ids, representation_id="signal_1d")},
        sample_ids=ids, y=y, target_names=["a", "b"] if partial else ["a"], task_type="regression",
        target_mask=[[True, True], [True, False], [True, True], [False, True]] if partial else None,
        groups=["batch_0"] * 4, independent_unit_ids=(units or ["plant_a", "plant_a", "plant_b", "plant_b"]) if declared else None,
        repetition_ids=["scan_0", "scan_1", "scan_0", "scan_1"] if declared else None,
    )
    return MultimodalSpectroDataset(cohort)


def test_unit_descriptor_uses_explicit_task_and_native_target_vocabulary() -> None:
    dataset, identity = _dataset(partial=True), _Identity()
    native_values = np.asarray([[2., 3.], [2., 0.], [8., 9.], [0., 9.]])
    contract = experimental_unit_contract(dataset, identity, [1, 0, 3, 2],
                                         target_values=native_values[[1, 0, 3, 2]], target_names=["a", "b"])
    assert contract is not None
    assert contract["task_type"] == "regression"
    assert contract["sample_ids"] == [identity.to_wire(row) for row in [1, 0, 3, 2]]
    assert contract["target_values"] == [[2., None], [2., 3.], [None, 9.], [8., 9.]]
    assert experimental_unit_metadata(dataset)["independent_unit_id"] == {0: "plant_a", 1: "plant_a", 2: "plant_b", 3: "plant_b"}
    assert experimental_unit_metadata(dataset)["repetition_id"][1] == "scan_1"


def test_explicit_unit_relation_metadata_is_signed_without_changing_identity_grains() -> None:
    dataset = _dataset()
    identity = mint_identity(dataset)
    envelope = build_envelope(dataset, identity, group_by_sample=dict.fromkeys(range(4), "batch_0"))
    records = envelope["coordinator_relations"]["records"]
    # Native relations are canonicalized by ID; metadata must stay attached to
    # the same observation regardless of that order.
    assert len(records) == len(dataset.sample_ids)
    assert {
        record["observation_id"]: (
            record["sample_id"], record["group_id"],
            record["metadata"]["independent_unit_id"], record["metadata"]["repetition_id"],
        ) for record in records
    } == {
        sample: (sample, "batch_0", unit, repetition)
        for sample, unit, repetition in zip(
            dataset.sample_ids, dataset.independent_unit_ids, dataset.repetition_ids, strict=True,
        )
    }
    assert envelope["relation_fingerprint"]
    with pytest.raises(ValueError, match="disagrees"):
        build_envelope(dataset, identity, metadata_by_sample={"independent_unit_id": {0: "forged_unit"}})
    with pytest.raises(AttributeError):
        dataset.independent_unit_ids = ("forged",) * 4


def test_group_policy_is_explicit_and_compat_dsl_preflight_checks_real_components() -> None:
    dataset, identity = _dataset(), _Identity()
    dsl = pipeline_to_dsl([GroupKFold(2), StandardScaler(), {"model": Ridge(alpha=.3)}])
    apply_experimental_unit_contract(dsl, dataset, identity, [0, 1, 2, 3], target_values=dataset.cohort.y, target_names=["a"])
    assert dsl["aggregation_policy"]["grouping_key"] == {"kind": "relation_metadata", "key": "independent_unit_id"}
    assert dsl["aggregation_policy"]["selection_metric_level"] == "group"
    assert require_weighted_context(dsl)
    for replacement in (PCA(1), KwargsOnly()):
        bad = pipeline_to_dsl([replacement, {"model": Ridge()}])
        with pytest.raises(ValueError, match="sample_weight"):
            apply_experimental_unit_contract(bad, dataset, identity, [0, 1, 2, 3], target_values=dataset.cohort.y, target_names=["a"])


def test_implicit_split_groups_never_enable_weighted_policy() -> None:
    dataset = _dataset(declared=False)
    dsl = pipeline_to_dsl([PCA(1), {"model": Ridge()}])
    before = deepcopy(dsl)
    assert apply_experimental_unit_contract(dsl, dataset, _Identity(), [0, 1, 2, 3], target_values=dataset.cohort.y, target_names=["a"]) is None
    assert dsl == before and experimental_unit_metadata(dataset) == {}
    assert native_fit_weights({"phase": "FIT_CV"}, dataset, _Identity(), ["sample:zeta"]) is None


def _partial_recipe_dsl() -> dict[str, Any]:
    return {
        "metadata": {"experimental_unit": {"schema_version": 1},
                     "prediction_availability": {"source_presence": {"nir": [True, False]}}},
        "steps": [{"kind": "branch", "branches": [{"steps": [
            {"id": "recipe", "kind": "transform", "operator": {"class": "sklearn.preprocessing.StandardScaler"},
             "params": {}, "metadata": {"controller_id": _SOURCE_RECIPE_CONTROLLER_ID,
                "nirs4all_source_recipe": {"schema_version": 1, "model_node_id": "head", "source_name": "nir"}}},
            {"id": "head", "kind": "model", "operator": {"class": "sklearn.linear_model.Ridge"}, "params": {},
             "metadata": {"prediction_availability_source": "nir", "nirs4all_source_stacking": {
                 "source": {"source_name": "nir"}, "target_policy": "per_target"}}},
        ]}]}],
    }


def test_partial_source_recipes_leave_real_learning_in_weighted_model_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    import nirs4all.pipeline.dagml_bridge as bridge

    monkeypatch.setattr(bridge, "_derive_controller_manifests_from_dagml", lambda specs: None)
    manifests = {item["controller_id"]: item for item in controller_manifests(_partial_recipe_dsl())}
    assert manifests[_SOURCE_RECIPE_CONTROLLER_ID]["fit_scope"] == "stateless"
    assert manifests[_SOURCE_RECIPE_CONTROLLER_ID]["data_requirements"] is None
    assert manifests["controller:nirs4all.transform"]["fit_scope"] == "fold_train"
    assert manifests["controller:nirs4all.model"]["fit_scope"] == "fold_train"
    assert "supports_sample_weights" in manifests["controller:nirs4all.model"]["capabilities"]
    assert manifests["controller:nirs4all.prediction_feature_join"]["fit_scope"] == "stateless"


@pytest.mark.parametrize("mutation", ["owner", "source", "binding", "unweighted"])
def test_partial_source_recipe_cannot_hide_independent_fit_or_unweighted_encoder(
    mutation: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nirs4all.pipeline.dagml_bridge as bridge

    monkeypatch.setattr(bridge, "_derive_controller_manifests_from_dagml", lambda specs: None)
    dsl = _partial_recipe_dsl()
    recipe = dsl["steps"][0]["branches"][0]["steps"][0]
    if mutation == "owner":
        recipe["metadata"]["nirs4all_source_recipe"]["model_node_id"] = "absent"
    elif mutation == "source":
        recipe["metadata"]["nirs4all_source_recipe"]["source_name"] = "absent"
    elif mutation == "binding":
        dsl["data_bindings"] = [{"node_id": "recipe"}]
    else:
        recipe["operator"]["class"] = "sklearn.preprocessing.MinMaxScaler"
    with pytest.raises(ValueError, match="signed source model owner|independent data binding|sample_weight"):
        controller_manifests(dsl)


@pytest.mark.parametrize("unit", ["plant with spaces", "plante_é", "a" * 129])
def test_general_io_labels_have_clear_native_group_profile_preflight(unit: str) -> None:
    dataset = _dataset(units=[unit, unit, "b", "b"])
    with pytest.raises(ValueError, match="Group scoring requires ASCII"):
        experimental_unit_contract(dataset, _Identity(), [0, 1, 2, 3], target_values=dataset.cohort.y, target_names=["a"])


def _task(*, partial: bool = False) -> dict[str, Any]:
    return {"phase": "REFIT", "fit_influence": {
        "fit_sample_ids": [_Identity().to_wire(row) for row in range(4)],
        "independent_unit_ids": ["plant_a", "plant_a", "plant_b", "plant_b"],
        "row_weights": [.5, .5, .5, .5], "target_names": ["a", "b"],
        "target_row_weights": [[.5, 1.], [.5, 0.], [1., .5], [0., .5]] if partial else None,
    }}


@pytest.mark.parametrize("mutation", ["ids", "units", "shape", "nonfinite", "phase", "missing"])
def test_native_row_weights_refuse_scope_or_carrier_mutations(mutation: str) -> None:
    task = _task()
    if mutation == "ids":
        task["fit_influence"]["fit_sample_ids"].reverse()
    elif mutation == "units":
        task["fit_influence"]["independent_unit_ids"][0] = "foreign"
    elif mutation == "shape":
        task["fit_influence"]["row_weights"] = [1.]
    elif mutation == "nonfinite":
        task["fit_influence"]["row_weights"][0] = float("nan")
    elif mutation == "phase":
        task["phase"] = "PREDICT"
    else:
        task.pop("fit_influence")
    with pytest.raises(ValueError, match="native fit influence"):
        native_fit_weights(task, _dataset(), _Identity(), [_Identity().to_wire(row) for row in range(4)])


def test_per_target_native_vectors_are_consumed_without_renormalization() -> None:
    dataset, task = _dataset(partial=True), _task(partial=True)
    weights = native_fit_weights(task, dataset, _Identity(), task["fit_influence"]["fit_sample_ids"],
                                 target_names=["a", "b"], per_target=True)
    assert weights is not None and not weights.flags.writeable
    np.testing.assert_array_equal(weights, task["fit_influence"]["target_row_weights"])
    with pytest.raises(ValueError, match="target names"):
        native_fit_weights(task, dataset, _Identity(), task["fit_influence"]["fit_sample_ids"], target_names=["b", "a"], per_target=True)


@pytest.mark.parametrize("phase", ["FIT_CV", "REFIT"])
def test_complete_target_scope_consumes_explicit_native_target_matrix(phase: str) -> None:
    dataset, task = _dataset(), _task()
    task["phase"] = phase
    task["fit_influence"]["target_names"] = ["a"]
    task["fit_influence"]["target_row_weights"] = [[.5], [.5], [.5], [.5]]
    ids = task["fit_influence"]["fit_sample_ids"]
    weights = native_fit_weights(task, dataset, _Identity(), ids, target_names=["a"], per_target=True)
    assert weights is not None and not weights.flags.writeable
    np.testing.assert_array_equal(weights, [[.5], [.5], [.5], [.5]])
    # A missing matrix is a native-carrier violation, never a host broadcast.
    task["fit_influence"].pop("target_row_weights")
    with pytest.raises(ValueError, match="native fit influence"):
        native_fit_weights(task, dataset, _Identity(), ids, target_names=["a"], per_target=True)


def test_weighted_encoders_and_heads_match_independent_sklearn_per_target_fits() -> None:
    dataset, task = _dataset(partial=True), _task(partial=True)
    X, y = dataset.cohort.source_values(range(4)), dataset.cohort.y
    weights = np.asarray(task["fit_influence"]["target_row_weights"])
    model = MultimodalRegressor({"nir": StandardScaler()}, Ridge(alpha=.4), target_policy="per_target")
    model.fit(X, y, target_mask=dataset.cohort.target_mask, sample_weight=weights)
    for column, fitted in enumerate(model.target_models_):
        observed = dataset.cohort.target_mask[:, column]
        encoder = StandardScaler().fit(X[0][observed], sample_weight=weights[observed, column])
        head = Ridge(alpha=.4).fit(encoder.transform(X[0][observed]), y[observed, column], sample_weight=weights[observed, column])
        np.testing.assert_allclose(fitted.transformers_["nir"].mean_, encoder.mean_)
        np.testing.assert_allclose(model.predict(X)[:, column], head.predict(encoder.transform(X[0])), rtol=1e-12, atol=1e-12)


def test_pipeline_routes_same_native_vector_to_every_learned_component() -> None:
    pipeline, weights = make_pipeline(StandardScaler(), Ridge()), np.asarray([.25, .75])
    kwargs = weighted_fit_kwargs(pipeline, weights)
    assert kwargs["standardscaler__sample_weight"] is weights and kwargs["ridge__sample_weight"] is weights
    require_sample_weight_support(pipeline)
    for estimator in (KwargsOnly(), PCA(), make_pipeline(PCA(), Ridge())):
        with pytest.raises(ValueError, match="sample_weight"):
            require_sample_weight_support(estimator)
