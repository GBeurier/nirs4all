"""Reject repeated classification validation before encoded labels can be averaged."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.api.result import RunResult
from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.operators.models.multimodal import MultimodalClassifier, TensorPCA
from nirs4all.pipeline.dagml.folds import _build_folds
from nirs4all.pipeline.dagml.steps import FrozenDagMlSplitStep
from tests.integration.api.test_multimodal_late_fusion import _pipeline as late_pipeline
from tests.integration.api.test_multimodal_targets import _classification_cohort, _classifier


@pytest.mark.parametrize("route", ["early", "late", "tuning"])
@pytest.mark.parametrize("split_kind", ["repeated", "shuffle"])
def test_classification_overlapping_validation_fails_before_any_fit(
    route: str, split_kind: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    splitter = (
        RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=17)
        if split_kind == "repeated" else StratifiedShuffleSplit(n_splits=4, test_size=0.5, random_state=17)
    )
    pipeline: list[Any] = [splitter, {"model": _classifier()}]
    if route == "late":
        pipeline = late_pipeline()
        pipeline[0] = splitter
        for branch in pipeline[1]["branch"]["steps"].values():
            branch[-1] = LogisticRegression(max_iter=300)
        pipeline[-1] = LogisticRegression(max_iter=300)
    tuning = None
    if route == "tuning":
        tuning = {
            "engine": "n4m", "sampler": "random", "seed": 19, "n_trials": 2,
            "space": {"model__C": [0.1, 1.0]},
            "storage": (tmp_path / "study").as_uri(), "study_name": "overlap",
        }

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("invalid classification folds reached an encoder or estimator fit")

    for operator in (MultimodalClassifier, LogisticRegression, TensorPCA, StandardScaler, ColumnTransformer):
        monkeypatch.setattr(operator, "fit", forbidden)
    with pytest.raises(ValueError, match="classification requires non-overlapping validation folds"):
        nirs4all.run(
            pipeline, _classification_cohort(), tuning=tuning, engine="dag-ml",
            workspace_path=tmp_path / "workspace", verbose=0, save_charts=False, random_state=19,
        )
    assert not (tmp_path / "study" / "overlap.n4mopt.json").exists()


def test_stratified_classification_partition_remains_supported(tmp_path: Path) -> None:
    cohort = _classification_cohort()
    result = nirs4all.run(
        [StratifiedKFold(n_splits=3, shuffle=True, random_state=17), {"model": _classifier()}],
        cohort, engine="dag-ml", workspace_path=tmp_path / "workspace",
        verbose=0, save_charts=False, random_state=19,
    )
    assert isinstance(result, RunResult)
    try:
        sample_ids = [
            sample for node in result._dagml_node_results for block in node.get("predictions", [])
            if block["partition"] == "validation" for sample in block["sample_ids"]
        ]
        assert len(sample_ids) == len(set(sample_ids)) == 12
        assert np.isfinite(result.cv_best_score)
    finally:
        result.close()


def test_frozen_classification_folds_cannot_bypass_overlap_validation() -> None:
    dataset = MultimodalSpectroDataset(_classification_cohort())
    pool = list(range(12))
    frozen = FrozenDagMlSplitStep(
        splitter=StratifiedKFold(3), sample_pool=tuple(pool),
        folds=(((4, 5, 6, 7, 8, 9, 10, 11), (0, 1, 2, 3)), ((4, 5, 6, 7, 8, 9, 10, 11), (0, 1, 2, 3))),
    )
    with pytest.raises(ValueError, match="classification requires non-overlapping validation folds"):
        _build_folds(frozen, dataset, pool, set())
