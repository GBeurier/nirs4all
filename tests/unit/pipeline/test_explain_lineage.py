"""Tests for relation explainability lineage helpers."""

import numpy as np

from nirs4all.data.relation_replay_manifest import build_relation_replay_manifest
from nirs4all.pipeline.explain_lineage import derive_relation_explain_lineage
from nirs4all.pipeline.explainer import Explainer
from nirs4all.pipeline.resolver import ResolvedPrediction


def _per_source_aggregate_manifest(headers: list[str] | None = None) -> dict:
    return {
        "representation": "per_source_aggregate",
        "fingerprint": "abc123",
        "shape": [2, 2],
        "headers": headers or ["MIR__1000", "NIRS__1000"],
        "source_ids": ["MIR", "NIRS"],
        "representation_plan": {
            "representation": "per_source_aggregate",
            "unit_level": "sample",
            "stage": "aggregate",
            "lineage": ["stage", "aggregate"],
        },
        "lineage": [
            {
                "unit_id": "sample-1",
                "source_observations": {
                    "MIR": ["mir-1", "mir-2"],
                    "NIRS": ["nirs-1", "nirs-2"],
                },
            }
        ],
    }


def test_derive_relation_explain_lineage_for_per_source_aggregate_manifest() -> None:
    """Relation helper exposes source aggregate feature provenance."""
    lineage = derive_relation_explain_lineage(_per_source_aggregate_manifest(), n_features=2)

    assert lineage is not None
    assert lineage.explanation_level == "source_aggregate"
    assert lineage.feature_names == ["MIR__1000", "NIRS__1000"]
    assert lineage.lineage_warning is not None
    assert "raw observation wavelengths" in lineage.lineage_warning

    mir_lineage = lineage.feature_lineage["MIR__1000"]
    assert mir_lineage["source_id"] == "MIR"
    assert mir_lineage["source_feature"] == "1000"
    assert mir_lineage["representation"] == "per_source_aggregate"
    assert mir_lineage["explanation_level"] == "source_aggregate"
    assert mir_lineage["materialization_fingerprint"] == "abc123"


def test_derive_relation_explain_lineage_unwraps_replay_manifest_and_colon_headers() -> None:
    """Replay manifests and live materializer headers use the same helper path."""
    manifest = {"materialization_manifest": _per_source_aggregate_manifest(["MIR:1000", "NIRS:1000"])}

    lineage = derive_relation_explain_lineage(manifest, n_features=2)

    assert lineage is not None
    assert lineage.feature_lineage["MIR:1000"]["source_id"] == "MIR"
    assert lineage.feature_lineage["NIRS:1000"]["source_feature"] == "1000"


def test_derive_relation_explain_lineage_prefers_model_headers_for_masks() -> None:
    """Masked materializations expose model-value and model-mask feature lineage."""
    materialization = _per_source_aggregate_manifest(["MIR:1000"])
    materialization["representation"] = "stack_padded_masked"
    materialization["shape"] = [1, 1]
    materialization["model_shape"] = [1, 2]
    materialization["model_headers"] = ["MIR:1000", "mask:MIR:1000"]

    lineage = derive_relation_explain_lineage(
        {"materialization_manifest": materialization},
        n_features=2,
    )

    assert lineage is not None
    assert lineage.feature_names == ["MIR:1000", "mask:MIR:1000"]
    assert lineage.feature_lineage["MIR:1000"]["feature_role"] == "signal"
    assert lineage.feature_lineage["mask:MIR:1000"]["source_id"] == "MIR"
    assert lineage.feature_lineage["mask:MIR:1000"]["feature_role"] == "presence_mask"


def test_derive_relation_explain_lineage_accepts_relation_replay_manifest_to_dict() -> None:
    """The helper unwraps the actual RelationReplayManifest serialisation shape."""
    replay_manifest = build_relation_replay_manifest(
        materialization=_per_source_aggregate_manifest(["MIR:1000", "NIRS:1000"]),
    ).to_dict()

    lineage = derive_relation_explain_lineage(replay_manifest, n_features=2)

    assert lineage is not None
    assert lineage.feature_lineage["MIR:1000"]["source_id"] == "MIR"
    assert lineage.explanation_level == "source_aggregate"


def test_sample_aggregate_explanation_level_is_not_source_aggregate() -> None:
    """sample_aggregate remains distinguishable from per-source aggregation."""
    manifest = _per_source_aggregate_manifest(["MIR:1000", "NIRS:1000"])
    manifest["representation"] = "sample_aggregate"
    manifest["representation_plan"] = {
        "representation": "sample_aggregate",
        "unit_level": "sample",
        "stage": "sample_aggregate",
        "lineage": ["raw_observation", "source_aggregate", "sample_aggregate"],
    }

    lineage = derive_relation_explain_lineage(manifest, n_features=2)

    assert lineage is not None
    assert lineage.explanation_level == "sample_aggregate"
    assert lineage.lineage_warning is not None
    assert "sample-level aggregates" in lineage.lineage_warning


def test_relation_lineage_skips_per_feature_payload_when_names_do_not_match_width() -> None:
    """SHAP-renamed or binned feature lists must not get positional lineage claims."""
    lineage = derive_relation_explain_lineage(
        _per_source_aggregate_manifest(["MIR:1000", "NIRS:1000"]),
        feature_names=["bin_0"],
        n_features=2,
    )

    assert lineage is not None
    assert lineage.explanation_level == "source_aggregate"
    assert lineage.feature_names is None
    assert lineage.feature_lineage == {}
    assert lineage.lineage_warning is not None


def test_explainer_attaches_relation_lineage_and_manifest_feature_names() -> None:
    """Explainer helper fills feature names and lineage without invoking SHAP."""
    manifest = _per_source_aggregate_manifest(["MIR:1000", "NIRS:1000"])
    shap_results = {"shap_values": np.array([[0.1, 0.2]])}

    Explainer._attach_relation_explain_lineage(shap_results, manifest, np.zeros((1, 2)), None)

    assert shap_results["feature_names"] == ["MIR:1000", "NIRS:1000"]
    assert shap_results["explanation_level"] == "source_aggregate"
    assert shap_results["feature_lineage"]["MIR:1000"]["source_id"] == "MIR"
    assert "raw observation wavelengths" in shap_results["lineage_warning"]


def test_explainer_extracts_relation_replay_manifest_from_resolved_prediction() -> None:
    """Explainer uses the same resolved manifest key emitted by BundleLoader."""
    replay_manifest = {"materialization_manifest": _per_source_aggregate_manifest(["MIR:1000"])}
    resolved = ResolvedPrediction(manifest={"relation_replay_manifest": replay_manifest})

    assert Explainer._extract_relation_replay_manifest(resolved) == replay_manifest
