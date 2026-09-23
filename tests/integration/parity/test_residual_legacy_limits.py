"""Witnesses for ResidualModel combinations that do not work in legacy."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.models.residual import ResidualModel

from ._datasets import dataset_path


@pytest.mark.parity
def test_legacy_residual_classification_has_no_continuous_prediction_metric(tmp_path):
    pipeline = [
        KFold(2),
        {"model": ResidualModel(base=Ridge(), learner=Ridge(), gate=False)},
    ]
    with pytest.raises(RuntimeError, match="Classification metrics can't handle a mix of multiclass and continuous targets"):
        nirs4all.run(
            pipeline, dataset_path("classification"), engine="legacy", refit=False,
            workspace_path=tmp_path / "classification", save_artifacts=False,
            save_charts=False, verbose=0,
        )


@pytest.mark.parity
def test_legacy_residual_multi_target_flattens_residual_rows(tmp_path):
    source = tmp_path / "multi_target"
    source.mkdir()
    rng = np.random.default_rng(7)
    for partition, count in (("train", 40), ("test", 10)):
        features = rng.normal(size=(count, 8))
        targets = np.column_stack((2 * features[:, 0] + features[:, 1], features[:, 2] - 3 * features[:, 3]))
        pd.DataFrame(features, columns=[f"x{index}" for index in range(8)]).to_csv(
            source / f"X{partition}.csv", index=False, sep=";",
        )
        pd.DataFrame(targets, columns=["y1", "y2"]).to_csv(
            source / f"Y{partition}.csv", index=False, sep=";",
        )
    pipeline = [
        KFold(2, shuffle=True, random_state=1),
        {"model": ResidualModel(base=Ridge(), learner=Ridge(), gate=False)},
    ]
    with pytest.raises(RuntimeError, match="Target data has 100 samples, expected 50"):
        nirs4all.run(
            pipeline, source, engine="legacy", refit=False,
            workspace_path=tmp_path / "multi_target_workspace", save_artifacts=False,
            save_charts=False, verbose=0,
        )


@pytest.mark.parity
def test_legacy_nested_residual_base_has_no_finite_outer_predictions(tmp_path):
    """A successful run can select an inner Ridge while its outer residual is empty."""
    rng = np.random.default_rng(917)
    features = rng.normal(size=(28, 7))
    targets = 2 * features[:, 0] - features[:, 1] + 0.1 * rng.normal(size=28)
    dataset = SpectroDataset("legacy_nested_residual_limit")
    dataset.add_samples(features[:24], {"partition": "train"}, headers=[str(index) for index in range(7)])
    dataset.add_samples(features[24:], {"partition": "test"})
    dataset.add_targets(targets.reshape(-1, 1))

    inner = ResidualModel(base=Ridge(alpha=1), learner=Ridge(alpha=1), gate=False)
    outer = ResidualModel(base=inner, learner=Ridge(alpha=1), gate=False)
    result = nirs4all.run(
        [KFold(2, shuffle=True, random_state=1), {"model": outer}],
        dataset, engine="legacy", refit=True,
        workspace_path=tmp_path / "nested_residual", save_artifacts=False,
        save_charts=False, verbose=0,
    )
    try:
        outer_rows = result.predictions.filter_predictions(model_name=outer.name, load_arrays=True)
        assert {row["partition"] for row in outer_rows} == {"val", "test"}
        assert all(row["fold_id"] == "final" for row in outer_rows)
        assert all(np.asarray(row["y_pred"]).size == 0 for row in outer_rows)
        assert any(np.isnan(row["val_score"]) for row in outer_rows)
        assert result.cv_best.model_name == "Ridge"
    finally:
        result.close()
