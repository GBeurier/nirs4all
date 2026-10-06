"""Archive weights use the same scoped validation evidence as native stacking."""

from typing import Any

import numpy as np
import pytest

from nirs4all.pipeline.dagml.native_results import _stacking_replay_manifest


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("metric", ["rmse", "r2"])
def test_archive_stacking_weights_average_only_eligible_outer_folds(metric: str, reverse: bool) -> None:
    producers = ["branch:0.node:0", "branch:0.node:1"]
    artifacts = [{"artifact_id": f"artifact:{producer}:nirs4all:refit:0", "producer_node": producer}
                 for producer in producers]
    artifacts.append({"artifact_id": "artifact:merge:stack:nirs4all:refit:0",
                      "controller_id": "controller:nirs4all.meta_model"})
    reports: list[dict[str, Any]] = [{"producer_node": "merge:stack", "partition": "validation", "fold_id": "fold0",
                                    "level": "sample", "metrics": {metric: 1.0}}]
    candidate_scores = [(2.0, 6.0), (4.0, 4.0)] if metric == "rmse" else [(0.2, 0.6), (0.4, 0.4)]
    inner_score, target_score = (1000.0, 2000.0) if metric == "rmse" else (0.98, 0.99)
    for producer, scores in zip(producers, candidate_scores, strict=True):
        for fold, score, level in [("inner.fold0", inner_score, "sample"), ("fold0", scores[0], "sample"),
                                   ("fold1", scores[1], "sample"), ("fold0", target_score, "target"),
                                   ("fold1", np.nan, "sample")]:
            reports.append({"producer_node": producer, "partition": "validation", "fold_id": fold,
                            "level": level, "metrics": {metric: score}})
    if reverse:
        reports.reverse()
    manifest = _stacking_replay_manifest(
        {"reports": reports}, artifacts, [{"branch": "branch_0", "aggregate": "weighted_mean", "metric": metric}],
        outer_fold_ids=["fold0", "fold1"],
    )
    assert manifest is not None
    weights = manifest["reduction_groups"][0]["weights"]
    expected = [1.0 / (4.0 + 1e-10)] * 2 if metric == "rmse" else [0.4] * 2
    np.testing.assert_allclose(weights, expected, rtol=1e-12)
