"""Tests for DAG-ML ScoreSet to RunResult projection metadata."""

from __future__ import annotations

import numpy as np

from nirs4all.pipeline.dagml.identity import IdentityMap
from nirs4all.pipeline.dagml.result import _scores_to_run_result


def test_producer_projection_never_pairs_another_models_prediction_arrays() -> None:
    """Two equally sized producer outputs must retain their own arrays and scores."""
    reports = []
    frames = []
    for producer, values, rmse in [("branch:0", [[1.0], [2.0]], 0.0), ("branch:1", [[11.0], [12.0]], 10.0)]:
        reports.append({"producer_node": producer, "partition": "validation", "fold_id": "fold0", "variant_id": "variant:base", "metrics": {"rmse": rmse}})
        frames.append({
            "node_id": producer,
            "predictions": [{"producer_node": producer, "partition": "validation", "fold_id": "fold0", "sample_ids": ["s0", "s1"], "values": values, "target_names": ["y"]}],
            "regression_targets": [{"level": "sample", "unit_ids": [{"level": "sample", "id": "s0"}, {"level": "sample", "id": "s1"}], "values": [[1.0], [2.0]], "target_names": ["y"]}],
        })
    scores = {"reports": reports}
    identity = IdentityMap("test", (), {"s0": 0, "s1": 1}, {0: "s0", 1: "s1"})
    for producer in ("branch:0", "branch:1"):
        result = _scores_to_run_result(scores, "dataset", producer, producer=producer, results=frames, identity=identity)
        row = result.predictions.filter_predictions(fold_id="0", partition="val", load_arrays=True)[0]
        observed_rmse = np.sqrt(np.mean((row["y_true"] - row["y_pred"]) ** 2))
        assert observed_rmse == row["val_score"]
        assert result._dagml_score_set is scores
        result.close()


def test_outer_view_keeps_inner_reports_in_canonical_scores_without_public_fold_rows() -> None:
    scores = {"reports": [
        {"producer_node": "base", "partition": "validation", "fold_id": fold, "variant_id": "variant:base", "metrics": {"rmse": 1.0}}
        for fold in ("fold0", "fold0.inner.fold0", "fold0.inner.fold1")
    ]}
    result = _scores_to_run_result(scores, "dataset", "base", producer="base", report_fold_ids={"fold0"})
    assert {row["fold_id"] for row in result.predictions.filter_predictions()} == {"0"}
    assert result._dagml_score_set is scores
    assert len(result._dagml_score_set["reports"]) == 3
    result.close()


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


def test_scores_to_run_result_preserves_portable_methods_single_variant_oof_average() -> None:
    """Methods keeps its sole concrete variant id on the terminal avg report."""

    scores = {
        "reports": [
            {
                "partition": "validation",
                "fold_id": "fold0",
                "variant_id": "variant:base",
                "producer_node": "model:compat.0",
                "metrics": {"rmse": 1.0},
            },
            {
                "partition": "validation",
                "fold_id": "fold1",
                "variant_id": "variant:base",
                "producer_node": "model:compat.0",
                "metrics": {"rmse": 3.0},
            },
            {
                "partition": "validation",
                "fold_id": "avg",
                "variant_id": "variant:base",
                "producer_node": "model:compat.0",
                "metrics": {"rmse": 2.0},
            },
        ]
    }

    result = _scores_to_run_result(scores, "dataset:test", "MethodsPLS")

    avg_rows = [
        row
        for row in result.filter(load_arrays=False)
        if row.get("fold_id") in {"avg", "w_avg"} and row.get("partition") == "val"
    ]
    assert len(avg_rows) == 2
    assert {row["val_score"] for row in avg_rows} == {2.0}
    assert result.cv_best_score == 2.0


def test_terminal_group_score_wins_without_replacing_sample_oof_score() -> None:
    """Repetition vote projects group-grain terminal evidence, not row-grain fallback."""

    scores = {
        "reports": [
            {
                "partition": "validation",
                "fold_id": "fold0",
                "variant_id": "variant:base",
                "producer_node": "model:compat.0",
                "level": "sample",
                "metrics": {"accuracy": 0.75},
            },
            {
                "partition": "validation",
                "fold_id": "avg",
                "variant_id": "variant:base",
                "producer_node": "model:compat.0",
                "level": "sample",
                "metrics": {"accuracy": 0.75},
            },
            {
                "partition": "test",
                "fold_id": None,
                "variant_id": "variant:base",
                "producer_node": "model:compat.0",
                "level": "sample",
                "metrics": {"accuracy": 0.50},
            },
            {
                "partition": "test",
                "fold_id": None,
                "variant_id": "variant:base",
                "producer_node": "model:compat.0",
                "level": "group",
                "metrics": {"accuracy": 1.0},
            },
            {
                "partition": "final",
                "fold_id": None,
                "variant_id": "variant:base",
                "producer_node": "model:compat.0",
                "level": "group",
                "metrics": {"accuracy": 1.0},
            },
        ]
    }

    result = _scores_to_run_result(
        scores,
        "dataset:test",
        "VoteClassifier",
        metric="accuracy",
        task_type="classification",
    )

    assert result.cv_best_score == 0.75
    assert result.best_accuracy == 1.0
