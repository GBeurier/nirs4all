"""Tests for DAG-ML ScoreSet to RunResult projection metadata."""

from __future__ import annotations

from nirs4all.pipeline.dagml.result import _scores_to_run_result


def test_scores_to_run_result_carries_native_node_results_for_attestation_audit() -> None:
    scores = {
        "reports": [
            {
                "partition": "validation",
                "fold_id": "fold0",
                "variant_id": "variant:base",
                "producer_node": "model:compat.0",
                "metrics": {"rmse": 1.0},
            }
        ]
    }
    node_results = [
        {
            "lineage": {
                "node_id": "model:compat.0",
                "phase": "FIT_CV",
                "loss_attestations": [{"loss_id": "example.loss@1"}],
            }
        }
    ]

    result = _scores_to_run_result(
        scores,
        "dataset:test",
        "TinyModel",
        results=node_results,
    )

    assert result._dagml_node_results == node_results  # noqa: SLF001


def test_scores_to_run_result_carries_variant_node_results_for_attestation_audit() -> None:
    scores = {
        "reports": [
            {
                "partition": "validation",
                "fold_id": "fold0",
                "variant_id": "variant:base",
                "producer_node": "model:compat.0",
                "metrics": {"rmse": 1.0},
            }
        ]
    }
    variant_frames = {
        "variant:base": [
            {
                "lineage": {
                    "node_id": "model:compat.0",
                    "phase": "FIT_CV",
                    "loss_attestations": [{"loss_id": "example.loss@1"}],
                }
            }
        ],
        "variant:loser": [
            {
                "lineage": {
                    "node_id": "model:compat.0",
                    "phase": "FIT_CV",
                    "loss_attestations": [],
                }
            }
        ],
    }

    result = _scores_to_run_result(
        scores,
        "dataset:test",
        "TinyModel",
        results_by_variant=variant_frames,
    )

    assert result._dagml_node_results == [  # noqa: SLF001
        *variant_frames["variant:base"],
        *variant_frames["variant:loser"],
    ]
