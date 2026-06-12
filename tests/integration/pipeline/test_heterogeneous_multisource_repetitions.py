"""Integration smoke tests for heterogeneous multisource repetitions.

These tests cover the roadmap-level path that spans relation staging,
``rep_fusion`` materialisation, replay manifests, bundle replay and API-facing
provenance. They intentionally stay small so the global integration suite keeps
running quickly.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np

from nirs4all.api.result import PredictResult
from nirs4all.controllers.data.rep_fusion import RepFusionController
from nirs4all.data.raw_multisource import RawMultiSourceDataset, RepresentationPlan, replay_materialization
from nirs4all.data.relation_replay_manifest import build_relation_replay_manifest
from nirs4all.data.relations import RepetitionSpec
from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
from nirs4all.pipeline.predictor import Predictor
from nirs4all.pipeline.steps.parser import ParsedStep, StepType


class _LengthModel:
    """Picklable model returning one deterministic value per materialised row."""

    def predict(self, X):
        return np.arange(len(X), dtype=float)


def _step(value):
    return ParsedStep(
        operator=None,
        keyword="rep_fusion",
        step_type=StepType.WORKFLOW,
        original_step={"rep_fusion": value},
        metadata={},
    )


def _spec() -> RepetitionSpec:
    return RepetitionSpec.from_config(
        {
            "sample_id": "sample_id",
            "link_by": "sample_id",
            "target_level": "physical_sample",
            "strict_cardinality": True,
            "sources": {
                "A": {"expected": 2},
                "B": {"expected": 3},
                "C": {"expected": 2},
            },
        }
    )


def _raw_dataset(prefix: str = "S", offset: float = 0.0) -> RawMultiSourceDataset:
    """Build an A=2/B=3/C=2 source-aware dataset with two physical samples."""

    return RawMultiSourceDataset.from_sources(
        _spec(),
        {
            "A": np.array(
                [
                    [1.0 + offset, 101.0 + offset],
                    [3.0 + offset, 103.0 + offset],
                    [5.0 + offset, 105.0 + offset],
                    [7.0 + offset, 107.0 + offset],
                ]
            ),
            "B": np.array(
                [
                    [10.0 + offset],
                    [20.0 + offset],
                    [30.0 + offset],
                    [40.0 + offset],
                    [50.0 + offset],
                    [60.0 + offset],
                ]
            ),
            "C": np.array(
                [
                    [1000.0 + offset, 2000.0 + offset],
                    [1100.0 + offset, 2100.0 + offset],
                    [1200.0 + offset, 2200.0 + offset],
                    [1300.0 + offset, 2300.0 + offset],
                ]
            ),
        },
        {
            "A": [f"{prefix}1", f"{prefix}1", f"{prefix}2", f"{prefix}2"],
            "B": [f"{prefix}1", f"{prefix}1", f"{prefix}1", f"{prefix}2", f"{prefix}2", f"{prefix}2"],
            "C": [f"{prefix}1", f"{prefix}1", f"{prefix}2", f"{prefix}2"],
        },
        headers_by_source={
            "A": ["a_1", "a_2"],
            "B": ["b_1"],
            "C": ["c_1", "c_2"],
        },
        targets_by_source={"A": [10.0, 10.0, 20.0, 20.0]},
    )


def _relation_bundle(tmp_path: Path, training_dataset: RawMultiSourceDataset) -> Path:
    materialization = training_dataset.materialize("per_source_aggregate")
    relation_manifest = build_relation_replay_manifest(materialization=materialization).to_dict()
    bundle_path = tmp_path / "heterogeneous_relation_model.n4a"

    with zipfile.ZipFile(bundle_path, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "bundle_format_version": "1.0",
                    "pipeline_uid": "heterogeneous_relation_pipeline",
                    "model_step_index": 4,
                    "relation_replay_manifest": {
                        "path": "relation_replay_manifest.json",
                        "version": relation_manifest["version"],
                        "fingerprint": relation_manifest["fingerprint"],
                    },
                }
            ),
        )
        zf.writestr("pipeline.json", json.dumps({"steps": [{"model": "PLSRegression"}], "model_step_index": 4}))
        zf.writestr("relation_replay_manifest.json", json.dumps(relation_manifest))
        model_buffer = io.BytesIO()
        joblib.dump(_LengthModel(), model_buffer)
        zf.writestr("artifacts/step_4_fold0_PLSRegression.joblib", model_buffer.getvalue())

    return bundle_path


def test_rep_fusion_materializes_a2_b3_c2_and_preserves_manifest_lineage():
    """``rep_fusion`` is the pipeline boundary from ragged relation staging to Features."""

    raw = _raw_dataset()
    controller = RepFusionController()
    context, output = controller.execute(
        _step({"representation": "per_source_aggregate", "preserve_lineage": True}),
        raw,
        ExecutionContext(),
        RuntimeContext(),
        mode="train",
    )

    materialized_dataset = context.custom["dataset_override"]
    X = materialized_dataset.x({"partition": "train"})
    np.testing.assert_allclose(
        X,
        np.array(
            [
                [2.0, 102.0, 20.0, 1050.0, 2050.0],
                [6.0, 106.0, 50.0, 1250.0, 2250.0],
            ]
        ),
    )
    assert raw.cardinalities() == {
        ("S1", "A"): 2,
        ("S1", "B"): 3,
        ("S1", "C"): 2,
        ("S2", "A"): 2,
        ("S2", "B"): 3,
        ("S2", "C"): 2,
    }
    assert output.metadata["representation"] == "per_source_aggregate"
    assert output.metadata["materialization_manifest"]["sample_ids"] == ["S1", "S2"]
    assert output.metadata["materialization_manifest"]["model_headers"] == ["A:a_1", "A:a_2", "B:b_1", "C:c_1", "C:c_2"]
    assert materialized_dataset.metadata({"partition": "train"})["physical_sample_id"].to_list() == ["S1", "S2"]

    replayed = replay_materialization(raw, output.metadata["materialization_manifest"], validate_fingerprint=True)
    replayed_matrix, replayed_headers = replayed.to_feature_matrix()
    np.testing.assert_allclose(replayed_matrix, X)
    assert replayed_headers == ["A:a_1", "A:a_2", "B:b_1", "C:c_1", "C:c_2"]


def test_relation_bundle_replays_prediction_dataset_and_exposes_predict_result_provenance(tmp_path):
    """A relation-aware ``.n4a`` can replay RawMultiSourceDataset prediction input."""

    predictor = Predictor(MagicMock())
    bundle_path = _relation_bundle(tmp_path, _raw_dataset(prefix="T"))
    prediction_dataset = _raw_dataset(prefix="P", offset=100.0)

    y_pred, predictions = predictor._predict_from_bundle(
        str(bundle_path),
        prediction_dataset,
        dataset_name="heterogeneous_prediction",
        all_predictions=False,
        verbose=0,
    )

    np.testing.assert_allclose(y_pred, np.array([0.0, 1.0]))
    prediction_row = predictions.filter_predictions(dataset_name="heterogeneous_prediction")[0]
    relation_manifest = prediction_row["metadata"]["relation_replay_manifest"]
    assert relation_manifest["materialization_manifest"]["representation"] == "per_source_aggregate"
    assert relation_manifest["materialization_manifest"]["sample_ids"] == ["T1", "T2"]
    assert relation_manifest["materialization_manifest"]["model_headers"] == ["A:a_1", "A:a_2", "B:b_1", "C:c_1", "C:c_2"]

    result = PredictResult(y_pred=y_pred, metadata=prediction_row["metadata"])
    assert result.relation_replay_manifest == relation_manifest
    assert result.relation_materialization_manifest["model_headers"] == ["A:a_1", "A:a_2", "B:b_1", "C:c_1", "C:c_2"]
    assert result.get_feature_lineage("A:a_1")["source_id"] == "A"
    assert result.get_feature_lineage("C:c_2")["source_feature"] == "c_2"
    assert result.lineage_warning is not None


def test_cartesian_full_replay_keeps_combo_plan_and_sample_reduction_scope():
    """Cartesian materialisation remains bounded, replayable and sample-reducible."""

    raw = _raw_dataset()
    plan = RepresentationPlan("cartesian_full", max_combos_per_sample=12, max_total_combos=24)
    materialized = raw.materialize(plan)

    assert materialized.representation == "cartesian_full"
    assert materialized.X.shape[0] == 24
    assert materialized.unit_ids[0] == "S1::A0xB0xC0"
    assert materialized.lineage is not None
    assert materialized.lineage[0]["component_observation_ids"] == ["A:S1:0", "B:S1:0", "C:S1:0"]
    assert materialized.representation_plan is not None
    assert materialized.representation_plan.unit_level == "combo"
    assert materialized.representation_plan.stage == "combo"
    assert materialized.representation_plan.combination_plan is not None
    assert materialized.representation_plan.combination_plan.max_combos_per_sample == 12

    replayed = replay_materialization(raw, materialized.to_manifest(), validate_fingerprint=True)
    np.testing.assert_allclose(replayed.X, materialized.X)
    assert replayed.unit_ids == materialized.unit_ids
    replayed_plan = replayed.to_manifest()["representation_plan"]
    assert replayed_plan["unit_level"] == "combo"
    assert replayed_plan["combination_plan"]["max_combos_per_sample"] == 12
