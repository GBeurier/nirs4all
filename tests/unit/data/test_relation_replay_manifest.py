"""Unit tests for N9 minimal relational replay manifests."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from nirs4all.data.fit_influence import FitInfluencePolicy
from nirs4all.data.raw_multisource import RawMultiSourceDataset, RepresentationPlan
from nirs4all.data.reduction import PredictionLevel, ReductionPlan
from nirs4all.data.relation_replay_manifest import (
    RelationReplayManifest,
    RelationReplayManifestError,
    build_relation_replay_manifest,
)
from nirs4all.data.relations import RepetitionSpec
from nirs4all.operators.data.merge import MetaFeaturePlan, StackingFitContract


def _fingerprint_payload(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _dataset() -> RawMultiSourceDataset:
    spec = RepetitionSpec(sample_id="sid", link_by="sid")
    return RawMultiSourceDataset.from_sources(
        spec,
        {
            "MIR": np.array([[1.0], [2.0], [3.0], [4.0]]),
            "RAMAN": np.array([[10.0], [11.0], [20.0], [21.0]]),
        },
        {"MIR": ["S1", "S1", "S2", "S2"], "RAMAN": ["S1", "S1", "S2", "S2"]},
    )


def test_build_relation_replay_manifest_round_trips_live_objects():
    ds = _dataset()
    materialization = ds.materialize(RepresentationPlan("cartesian_mc", max_combos_per_sample=2, random_state=3))
    reduction = ReductionPlan(input_level=PredictionLevel.COMBO, output_level=PredictionLevel.SAMPLE)
    fit_policy = FitInfluencePolicy(mode="equal_sample_influence")
    meta_plan = MetaFeaturePlan(missing_prediction_policy="drop_incomplete")
    stacking_contract = StackingFitContract(selection_protocol="holdout")

    manifest = build_relation_replay_manifest(
        staging=ds,
        materialization=materialization,
        reduction_plans=[reduction],
        fit_influence_policy=fit_policy,
        meta_feature_plan=meta_plan,
        stacking_fit_contract=stacking_contract,
        extra_fingerprints={"custom": "abc"},
    )
    payload = manifest.to_dict()
    restored = RelationReplayManifest.from_dict(json.loads(json.dumps(payload)))

    assert restored.fingerprint() == manifest.fingerprint()
    assert restored.representation_plan is not None
    assert restored.representation_plan.representation == "cartesian_mc"
    assert restored.representation_plan.combination_plan is not None
    assert restored.reduction_plans[0].output_level == PredictionLevel.SAMPLE
    assert restored.fit_influence_policy == fit_policy
    assert restored.meta_feature_plan == meta_plan
    assert restored.stacking_fit_contract == stacking_contract
    assert restored.extra_fingerprints == {"custom": "abc"}


def test_relation_replay_manifest_can_be_built_from_plain_manifests():
    ds = _dataset()
    materialization = ds.materialize("per_source_aggregate")

    manifest = build_relation_replay_manifest(
        staging=ds.to_manifest(),
        materialization=materialization.to_manifest(),
        reduction_plans=[ReductionPlan().to_dict()],
        fit_influence_policy=FitInfluencePolicy().to_dict(),
        meta_feature_plan=MetaFeaturePlan(missing_prediction_policy="mask").to_dict(),
        stacking_fit_contract=StackingFitContract().to_dict(),
    )

    assert manifest.representation_plan is not None
    assert manifest.representation_plan.representation == "per_source_aggregate"
    assert manifest.meta_feature_plan is not None
    assert manifest.meta_feature_plan.missing_prediction_policy == "mask"
    assert manifest.to_dict()["staging_manifest"]["relation_fingerprint"] == ds.relation_table.fingerprint()


def test_relation_replay_manifest_accepts_legacy_payload_without_meta_feature_plan():
    manifest = build_relation_replay_manifest(
        representation_plan=RepresentationPlan("sample_aggregate"),
        stacking_fit_contract=StackingFitContract().to_dict(),
    )
    payload = manifest.to_dict(include_fingerprint=False)
    payload.pop("meta_feature_plan")
    payload["fingerprint"] = _fingerprint_payload(payload)

    restored = RelationReplayManifest.from_dict(payload)

    assert restored.meta_feature_plan is None
    assert restored.stacking_fit_contract == StackingFitContract()


def test_relation_replay_manifest_rejects_tampered_legacy_payload_without_meta_feature_plan():
    manifest = build_relation_replay_manifest(
        representation_plan=RepresentationPlan("sample_aggregate"),
        extra_fingerprints={"source": "original"},
    )
    payload = manifest.to_dict(include_fingerprint=False)
    payload.pop("meta_feature_plan")
    payload["fingerprint"] = _fingerprint_payload(payload)
    payload["extra_fingerprints"]["source"] = "tampered"

    with pytest.raises(RelationReplayManifestError, match="fingerprint"):
        RelationReplayManifest.from_dict(payload)


def test_relation_replay_manifest_rejects_legacy_fingerprint_when_meta_feature_plan_key_is_explicit():
    manifest = build_relation_replay_manifest(representation_plan=RepresentationPlan("sample_aggregate"))
    payload = manifest.to_dict(include_fingerprint=False)
    legacy_payload = dict(payload)
    legacy_payload.pop("meta_feature_plan")
    payload["meta_feature_plan"] = None
    payload["fingerprint"] = _fingerprint_payload(legacy_payload)

    with pytest.raises(RelationReplayManifestError, match="fingerprint"):
        RelationReplayManifest.from_dict(payload)


def test_relation_replay_manifest_rejects_tampered_fingerprint():
    manifest = build_relation_replay_manifest(representation_plan=RepresentationPlan("sample_aggregate"))
    payload = manifest.to_dict()
    payload["fingerprint"] = "bad"

    with pytest.raises(RelationReplayManifestError, match="fingerprint"):
        RelationReplayManifest.from_dict(payload)


def test_relation_replay_manifest_requires_replay_data():
    with pytest.raises(RelationReplayManifestError, match="requires"):
        RelationReplayManifest()
