"""Parity tests for the nirs4all → dag-ml(-data) data layer (migration phase 2b-i).

Identity + resolver are pure nirs4all (no dag-ml / dag-ml-data import), so these run
unconditionally — they verify the stable-id round-trip and that the resolver serves real
``SpectroDataset`` rows **in the caller's requested order** (the load-bearing
identity-keyed invariant), entirely in-process with no CLI / execution / FFI.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from nirs4all.data.config import DatasetConfigs
from nirs4all.pipeline.dagml.envelope import build_envelope, build_fold_set, sample_relations
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml.operator_routing import route_graph_node, route_operator
from nirs4all.pipeline.dagml.resolver import MaterializationResolver

from ._datasets import dataset_path
from ._registry import get

pytestmark = [pytest.mark.parity]


@pytest.fixture(scope="module")
def regression_dataset():
    return DatasetConfigs(dataset_path("regression")).get_dataset_at(0)


def test_identity_roundtrip_is_stable_and_legal(regression_dataset) -> None:
    identity = mint_identity(regression_dataset)
    sample_ints = regression_dataset.index_column("sample", {})
    # Bidirectional and total over every sample.
    assert len(identity.identities) == len(sample_ints)
    for sample_int in sample_ints:
        assert identity.to_int(identity.to_wire(sample_int)) == sample_int
    # Every wire id is a legal dag-ml-data id (alnum + '_.-', no ':').
    legal = re.compile(r"^[A-Za-z0-9_.\-]{1,128}\Z")
    for observation_id in identity.observation_ids():
        assert legal.match(observation_id), observation_id
        assert ":" not in observation_id
    # The fingerprint is content-derived and stable across mints (not position-derived).
    assert identity.fingerprint == mint_identity(regression_dataset).fingerprint


def test_resolver_features_restore_request_order(regression_dataset) -> None:
    identity = mint_identity(regression_dataset)
    resolver = MaterializationResolver(regression_dataset, identity)
    sample_ints = regression_dataset.index_column("sample", {})

    # A deliberately non-ascending request — SpectroDataset.x returns storage order, so
    # this only passes if the resolver re-orders back to the request.
    picked = [sample_ints[5], sample_ints[0], sample_ints[9], sample_ints[2]]
    wire = [identity.to_wire(s) for s in picked]
    out = resolver.resolve_features(wire)
    values = np.asarray(out["values"])

    ground_truth = np.stack([np.asarray(regression_dataset.x({"sample": [s]}, layout="2d"))[0] for s in picked])
    assert values.shape == ground_truth.shape
    assert np.array_equal(values, ground_truth)
    assert out["observation_ids"] == wire

    # Reversing the request reverses the rows — proves order tracks identity, not position.
    rev = resolver.resolve_features(list(reversed(wire)))
    assert np.array_equal(np.asarray(rev["values"]), values[::-1])


def test_resolver_targets_restore_request_order(regression_dataset) -> None:
    identity = mint_identity(regression_dataset)
    resolver = MaterializationResolver(regression_dataset, identity)
    sample_ints = regression_dataset.index_column("sample", {})

    picked = [sample_ints[7], sample_ints[1], sample_ints[4]]
    wire = [identity.to_wire(s) for s in picked]
    out = resolver.resolve_targets(wire)
    values = np.asarray(out["values"], dtype=float)

    ground_truth = np.asarray([float(np.asarray(regression_dataset.y({"sample": [s]})).ravel()[0]) for s in picked])
    assert np.allclose(values, ground_truth)
    assert out["sample_ids"] == wire


def test_resolver_serves_real_spectra_not_a_hash(regression_dataset) -> None:
    """Guards the closed gap: the shipped conformance adapters synthesize X from hashed
    sample ids; this resolver must return the actual spectra."""
    identity = mint_identity(regression_dataset)
    resolver = MaterializationResolver(regression_dataset, identity)
    sample_int = regression_dataset.index_column("sample", {})[0]
    wire = identity.to_wire(sample_int)

    row = np.asarray(resolver.resolve_features([wire])["values"])[0]
    real = np.asarray(regression_dataset.x({"sample": [sample_int]}, layout="2d"))[0]
    assert np.array_equal(row, real)
    # A hashed-id synthesis would collapse 2151 wavelengths to a constant; real spectra vary.
    assert row.shape[0] > 1 and float(np.ptp(row)) > 0.0


def test_route_operator_per_kind_and_overrides() -> None:
    """route_graph_node resolves each node-kind operator-ref shape; variants win; unknowns raise."""
    transform = route_graph_node({"kind": "transform", "operator": {"class": "nirs4all.operators.transforms.scalers.StandardNormalVariate"}, "params": {"with_std": True}})
    assert type(transform).__name__ == "StandardNormalVariate"
    # y_transform carries params nested under the operator, not on the node.
    y_transform = route_graph_node({"kind": "y_transform", "operator": {"class": "sklearn.preprocessing._data.MinMaxScaler", "params": {"feature_range": [0, 1]}}})
    assert type(y_transform).__name__ == "MinMaxScaler"
    # model carries a bare short name resolved through the allow-table.
    model = route_graph_node({"kind": "model", "operator": "PLSRegression", "params": {"n_components": 7}})
    assert type(model).__name__ == "PLSRegression" and model.get_params()["n_components"] == 7

    # Variant overrides win over node params.
    swept = route_operator("model", "PLSRegression", {"n_components": 3}, variant_overrides={"n_components": 9})
    assert swept.get_params()["n_components"] == 9
    # Unknown model name and unsupported kind both fail loudly. An unknown short name is not in the
    # allow-table, so it falls through to import-by-FQN, which rejects a bare (non-qualified) name.
    with pytest.raises(ValueError, match="fully-qualified class name"):
        route_operator("model", "NotARealModel")
    with pytest.raises(ValueError, match="operator_kind"):
        route_operator("branch", "whatever")


def test_route_real_compiled_vertical_slice_nodes() -> None:
    """Every node of the actually-compiled vertical slice routes to its real operator."""
    pytest.importorskip("dag_ml", reason="dag-ml not importable (core dependency; broken install?)")
    from nirs4all.pipeline.dagml_bridge import build_dagml_plan

    plan = build_dagml_plan(get("baseline_vertical_slice").pipeline, plan_id="p", dsl_id="vs").to_dict()
    routed = {node["kind"]: type(route_graph_node(node)).__name__ for node in plan["graph_plan"]["graph"]["nodes"]}
    assert routed == {"transform": "StandardNormalVariate", "y_transform": "MinMaxScaler", "model": "PLSRegression"}


def test_envelope_builds_and_validates_against_live_contract(regression_dataset) -> None:
    """build_envelope produces a contract-valid CoordinatorDataPlanEnvelope (the wheel
    derives relations + fingerprints; a successful build is itself the gate)."""
    dag_ml_data = pytest.importorskip("dag_ml_data", reason="dag-ml-data not importable (core dependency; broken install?)")
    identity = mint_identity(regression_dataset)

    envelope = build_envelope(regression_dataset, identity)
    assert envelope["schema_version"] == 1
    for fp in ("schema_fingerprint", "plan_fingerprint", "relation_fingerprint"):
        assert isinstance(envelope[fp], str) and len(envelope[fp]) == 64
    assert envelope["plan"]["output_representation"] == "tabular_numeric"
    step_kinds = [step["kind"] for step in envelope["plan"]["steps"]]
    assert "materialize" in step_kinds and "adapt" in step_kinds
    assert any(step.get("adapter_id") == "spectra.flatten" for step in envelope["plan"]["steps"])
    records = envelope["coordinator_relations"]["records"]
    assert len(records) == len(identity.identities)
    assert all(not record["is_augmented"] for record in records)
    # Re-validate through the same validator dag-ml-data uses (not a stale local schema).
    dag_ml_data.validate_coordinator_data_plan_envelope_json(json.dumps(envelope))


def test_fold_set_requires_an_oof_partition(regression_dataset) -> None:
    """A KFold partition validates against the CV-universe relations; ShuffleSplit (which
    does not validate every sample exactly once) is refused -- a real OOF-semantics gap."""
    dag_ml_data = pytest.importorskip("dag_ml_data", reason="dag-ml-data not importable (core dependency; broken install?)")
    from sklearn.model_selection import KFold, ShuffleSplit

    identity = mint_identity(regression_dataset)
    train = regression_dataset.index_column("sample", {"partition": "train"})
    cv_relations = sample_relations(identity, sample_ints=train)

    def fold_set(splitter):
        folds = [([train[j] for j in tr], [train[j] for j in va]) for tr, va in splitter.split(train)]
        return build_fold_set(identity, folds)

    # KFold partitions the pool: each sample validated exactly once -> valid.
    dag_ml_data.validate_fold_set_against_sample_relations(fold_set(KFold(n_splits=3, shuffle=True, random_state=42)), cv_relations)
    # ShuffleSplit's overlapping/incomplete validation sets break the OOF partition.
    with pytest.raises(dag_ml_data.DagMlDataContractError):
        dag_ml_data.validate_fold_set_against_sample_relations(fold_set(ShuffleSplit(n_splits=3, random_state=42)), cv_relations)


# ---------------------------------------------------------------------------
# Observation grain != sample grain: native augmentation data-plane (slice #13)
# ---------------------------------------------------------------------------


@pytest.fixture()
def augmented_dataset():
    """4 base train + 2 test samples; base 0 augmented twice ("noise").

    The augmented children get new ``sample`` ints (6, 7) whose ``origin`` is 0, so the
    observation grain (one stored row each) diverges from the sample grain (origin 0).
    """
    from nirs4all.data.dataset import SpectroDataset

    rng = np.random.default_rng(0)
    ds = SpectroDataset(name="aug_poc")
    base = rng.random((6, 5)).astype(np.float64)
    headers = [str(w) for w in range(5)]
    ds.add_samples(base[:4], {"partition": "train"}, headers=headers, header_unit="index")
    ds.add_samples(base[4:], {"partition": "test"}, headers=headers, header_unit="index")
    ds.add_targets(np.arange(6, dtype=np.float64).reshape(-1, 1))
    ds.add_samples_batch(
        rng.random((2, 1, 5)).astype(np.float64),
        [{"partition": "train", "origin": 0, "augmentation": "noise"}] * 2,
    )
    return ds


def _aug_observation_ids(identity):
    return [sample.observation_id for sample in identity.identities if sample.augmented]


def test_resolver_serves_real_augmented_row_features(augmented_dataset) -> None:
    """resolve_features for an augmented child's observation id returns that row's REAL
    features (its own stored row), not its origin's — the x_rows observation-grain read."""
    identity = mint_identity(augmented_dataset)
    resolver = MaterializationResolver(augmented_dataset, identity)
    aug_obs = _aug_observation_ids(identity)
    assert aug_obs, "fixture must contain augmented rows"

    aug_int = identity.to_int(aug_obs[0])
    origin_int = identity.identities[aug_int].origin_int
    out = resolver.resolve_features([aug_obs[0]])
    row = np.asarray(out["values"])[0]

    real = np.asarray(augmented_dataset.x_rows([aug_int], layout="2d"))[0]
    origin_row = np.asarray(augmented_dataset.x_rows([origin_int], layout="2d"))[0]
    assert np.array_equal(row, real)              # the augmented row's own features
    assert not np.array_equal(row, origin_row)    # distinct from the origin's features


def test_resolver_mixed_base_and_augmented_request_keeps_order(augmented_dataset) -> None:
    """A request mixing base and augmented observation ids returns one row per id, in request
    order — the old base-keyed path raised NotImplementedError on this; x_rows aligns it."""
    identity = mint_identity(augmented_dataset)
    resolver = MaterializationResolver(augmented_dataset, identity)
    aug_obs = _aug_observation_ids(identity)
    base_obs = [sample.observation_id for sample in identity.identities if not sample.augmented]

    request = [aug_obs[1], base_obs[0], aug_obs[0], base_obs[2]]
    out = resolver.resolve_features(request)
    values = np.asarray(out["values"])
    assert out["observation_ids"] == request
    assert values.shape == (len(request), 5)

    ground_truth = np.stack([
        np.asarray(augmented_dataset.x_rows([identity.to_int(oid)], layout="2d"))[0] for oid in request
    ])
    assert np.array_equal(values, ground_truth)


def test_resolver_refuses_augmented_in_non_augmented_view(augmented_dataset) -> None:
    """An augmented child must never reach a validation/predict view: resolve_features with
    include_augmented=False refuses an augmented observation id (origin-boundary guard)."""
    identity = mint_identity(augmented_dataset)
    resolver = MaterializationResolver(augmented_dataset, identity)
    aug_obs = _aug_observation_ids(identity)

    with pytest.raises(ValueError, match="origin boundary"):
        resolver.resolve_features([aug_obs[0]], include_augmented=False)


def test_sample_relations_emit_origin_and_augmentation_for_children(augmented_dataset) -> None:
    """Augmented rows carry origin_id == the origin's observation wire id + augmentation
    metadata; base rows keep origin_id None and no augmentation (the Rust contract rejects
    an origin/augmentation on a non-augmented row)."""
    dag_ml_data = pytest.importorskip("dag_ml_data", reason="dag-ml-data not importable (core dependency; broken install?)")
    identity = mint_identity(augmented_dataset)
    aug_by = {sample.sample_int: "noise" for sample in identity.identities if sample.augmented}

    table = sample_relations(identity, augmentation_by_sample=aug_by)
    by_obs = {row["observation_id"]: row for row in table["rows"]}
    for sample in identity.identities:
        row = by_obs[sample.observation_id]
        if sample.augmented:
            assert row["origin_id"] == identity.to_wire(sample.origin_int)
            assert row["augmented"] is True
            assert row["augmentation"]["transform_id"] == "noise"
            assert len(row["augmentation"]["params_fingerprint"]) == 64
        else:
            assert row["origin_id"] is None
            assert "augmentation" not in row
    # The whole table (incl. augmentation/origin fields) is contract-valid.
    dag_ml_data.validate_sample_relation_table_json(json.dumps(table))


def test_augmented_fold_set_passes_origin_boundary(augmented_dataset) -> None:
    """Folds at the SAMPLE grain (base sample ids only) with augmented children co-grouped
    under their origin pass validate_fold_set_against_sample_relations: the child shares the
    origin's sample_id, so it inherits the origin's partition (origin-boundary green)."""
    dag_ml_data = pytest.importorskip("dag_ml_data", reason="dag-ml-data not importable (core dependency; broken install?)")
    identity = mint_identity(augmented_dataset)
    aug_by = {sample.sample_int: "noise" for sample in identity.identities if sample.augmented}

    # The train pool from a partition query already includes the augmented children (direct
    # index, not the two-phase base-keyed path), so it carries base + aug ints.
    train_pool = augmented_dataset.index_column("sample", {"partition": "train"})  # [0,1,2,3,6,7]
    base_ints = [s for s in train_pool if not identity.identities[s].augmented]    # [0,1,2,3]
    cv_relations = sample_relations(identity, sample_ints=train_pool, augmentation_by_sample=aug_by)

    # 2-fold over the BASE samples only; augmented children are NOT listed in the fold — they
    # group under their origin's sample_id and inherit its partition.
    folds = [([base_ints[0], base_ints[1]], [base_ints[2], base_ints[3]]),
             ([base_ints[2], base_ints[3]], [base_ints[0], base_ints[1]])]
    fold_set = build_fold_set(identity, folds)
    # origin (sample 0) trains in fold0 -> its children inherit train; the check passes.
    dag_ml_data.validate_fold_set_against_sample_relations(fold_set, cv_relations)


def test_x_rows_equals_x_for_base_only_request(regression_dataset) -> None:
    """x_rows must equal the base-keyed x row-for-row when no augmentation is involved — the
    additive accessor changes nothing for existing base-only callers."""
    sample_ints = regression_dataset.index_column("sample", {})[:6]
    via_rows = np.asarray(regression_dataset.x_rows(sample_ints, layout="2d"))
    via_x = np.stack([np.asarray(regression_dataset.x({"sample": [s]}, layout="2d"))[0] for s in sample_ints])
    assert np.array_equal(via_rows, via_x)
