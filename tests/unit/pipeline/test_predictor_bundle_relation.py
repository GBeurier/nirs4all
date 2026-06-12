"""Tests for relation-aware bundle prediction through Predictor."""

from __future__ import annotations

import io
import json
import zipfile
from unittest.mock import MagicMock

import numpy as np

from nirs4all.data.raw_multisource import RawMultiSourceDataset
from nirs4all.data.relation_replay_manifest import build_relation_replay_manifest
from nirs4all.data.relations import RepetitionSpec
from nirs4all.pipeline.predictor import Predictor


class _LengthModel:
    """Small picklable model returning one prediction per materialized row."""

    def predict(self, X):
        return np.arange(len(X), dtype=float)


def _raw_dataset(sample_prefix: str = "S") -> RawMultiSourceDataset:
    return RawMultiSourceDataset.from_sources(
        RepetitionSpec(sample_id="sid", link_by="sid"),
        {
            "MIR": np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]),
            "RAMAN": np.array([[10.0], [20.0]]),
        },
        {
            "MIR": [f"{sample_prefix}1", f"{sample_prefix}1", f"{sample_prefix}2", f"{sample_prefix}2"],
            "RAMAN": [f"{sample_prefix}1", f"{sample_prefix}2"],
        },
        targets_by_source={"MIR": [10.0, 10.0, 20.0, 20.0]},
    )


def _relation_bundle(tmp_path, training_dataset: RawMultiSourceDataset):
    import joblib

    materialization = training_dataset.materialize("per_source_aggregate")
    relation_manifest = build_relation_replay_manifest(materialization=materialization).to_dict()
    bundle_path = tmp_path / "relation_predictor.n4a"
    with zipfile.ZipFile(bundle_path, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({
                "bundle_format_version": "1.0",
                "pipeline_uid": "relation_pipeline",
                "model_step_index": 4,
                "relation_replay_manifest": {
                    "path": "relation_replay_manifest.json",
                    "version": relation_manifest["version"],
                    "fingerprint": relation_manifest["fingerprint"],
                },
            }),
        )
        zf.writestr("pipeline.json", json.dumps({"steps": [{"model": "PLSRegression"}], "model_step_index": 4}))
        zf.writestr("relation_replay_manifest.json", json.dumps(relation_manifest))
        buffer = io.BytesIO()
        joblib.dump(_LengthModel(), buffer)
        zf.writestr("artifacts/step_4_fold0_PLSRegression.joblib", buffer.getvalue())
    return bundle_path


def test_predictor_bundle_replays_raw_multisource_dataset(tmp_path):
    """Predictor passes RawMultiSourceDataset through relation-aware bundle replay."""
    predictor = Predictor(MagicMock())
    bundle_path = _relation_bundle(tmp_path, _raw_dataset("T"))
    prediction_dataset = _raw_dataset("P")

    y_pred, predictions = predictor._predict_from_bundle(
        str(bundle_path),
        prediction_dataset,
        dataset_name="relation_prediction",
        all_predictions=False,
        verbose=0,
    )

    assert y_pred.shape == (prediction_dataset.n_samples,)
    np.testing.assert_allclose(y_pred, np.arange(prediction_dataset.n_samples, dtype=float))
    assert predictions.filter_predictions(dataset_name="relation_prediction")[0]["y_pred"].shape == y_pred.shape
