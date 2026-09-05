"""Successive model checkpoints share folds and retain their own fitted prefix."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


@pytest.mark.parametrize("with_split", [True, False])
def test_successive_models_preserve_prefix_and_captured_export(tmp_path, monkeypatch, with_split):
    import nirs4all

    rng = np.random.default_rng(126)
    X = rng.normal(size=(30, 5))
    y = X @ np.arange(1.0, 6.0)
    X32 = X.astype(np.float32)
    splitter = KFold(n_splits=3, shuffle=True)
    calls = []
    original_split = splitter.split

    def split(*args, **kwargs):
        calls.append(1)
        return original_split(*args, **kwargs)

    monkeypatch.setattr(splitter, "split", split)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("legacy execution"))
    pipeline = [StandardScaler(), *([splitter] if with_split else []), Ridge(0.1), RobustScaler(), Ridge(10)]
    result = nirs4all.run(pipeline, (X, y), save_artifacts=False)
    assert len(result.runs) == 2
    assert len(calls) == int(with_split)
    expected = [
        make_pipeline(StandardScaler(), Ridge(0.1)).fit(X32, y).predict(X32),
        make_pipeline(StandardScaler(), RobustScaler(), Ridge(10)).fit(X32, y).predict(X32),
    ]
    fold_memberships = []
    for index, child in enumerate(result.runs):
        assert child._dagml_score_set is not None
        assert child._dagml_refit_artifacts
        np.testing.assert_allclose(child._dagml_refit_artifacts[0]["estimator"].predict(X32), expected[index], atol=1e-6)
        rows = child.predictions.filter_predictions(load_arrays=True)
        fold_memberships.append({str(row["fold_id"]): tuple(row["sample_indices"]) for row in rows if row["partition"] == "val"})
        path = child.export(tmp_path / f"model-{index}.n4a")
        np.testing.assert_allclose(nirs4all.predict(path, X).y_pred, expected[index], atol=1e-6)
    assert fold_memberships[0] == fold_memberships[1]


def test_frozen_sequential_foldset_refuses_a_different_pool():
    from nirs4all.pipeline.dagml.steps import FrozenDagMlSplitStep

    frozen = FrozenDagMlSplitStep(KFold(2), sample_pool=(3, 1), folds=(((3,), (1,)),))
    assert frozen.materialized_folds([3, 1], set()) == [([3], [1])]
    with pytest.raises(ValueError, match="different sample pool"):
        frozen.materialized_folds([1, 3], set())


def test_successive_models_preserve_explicit_groups():
    import polars as pl
    from sklearn.model_selection import GroupKFold

    import nirs4all
    from nirs4all.data.dataset import SpectroDataset

    rng = np.random.default_rng(23)
    X = rng.normal(size=(24, 4))
    groups = np.repeat(np.arange(12), 2)
    dataset = SpectroDataset("grouped_checkpoints")
    dataset.add_samples(X, {"partition": "train"})
    dataset.add_targets(X @ np.arange(1.0, 5.0))
    dataset.add_metadata(pl.DataFrame({"specimen": groups}))
    result = nirs4all.run(
        [StandardScaler(), {"split": GroupKFold(3), "group_by": "specimen"}, Ridge(1), Ridge(2)],
        dataset, save_artifacts=False,
    )
    memberships = []
    for child in result.runs:
        rows = child.predictions.filter_predictions(load_arrays=True)
        by_fold = {str(row["fold_id"]): set(row["sample_indices"]) for row in rows if row["partition"] == "val"}
        for samples in by_fold.values():
            assert all((2 * group in samples) == (2 * group + 1 in samples) for group in range(12))
        memberships.append(by_fold)
    assert memberships[0] == memberships[1]
