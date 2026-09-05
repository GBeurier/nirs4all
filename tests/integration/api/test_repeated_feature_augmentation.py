"""Repeated feature augmentation retains CV and captured REFIT semantics."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


@pytest.mark.parametrize("action", ["extend", "add", "replace"])
def test_repeated_channels_cv_and_export_match_independent_oracle(tmp_path, monkeypatch, action):
    import nirs4all
    from nirs4all.data import SpectroDataset

    X = np.random.default_rng(314).normal(size=(39, 6)).astype(np.float32)
    y = 2 * X[:, 0] - X[:, 4] + np.arange(39, dtype=np.float32) * 0.007
    unseen = np.random.default_rng(315).normal(size=(7, 6)).astype(np.float32)
    dataset = SpectroDataset("repeated_channels")
    dataset.add_samples(X, {"partition": "train"})
    dataset.add_targets(y)

    def design(train, prediction):
        chains = [[], [StandardScaler()], [RobustScaler()]]
        if action != "extend":
            chains.append([StandardScaler(), RobustScaler()])
        train_blocks, predicted_blocks = [], []
        for index, chain in enumerate(chains):
            if action != "replace" or index >= 2:
                chain.append(MinMaxScaler())
            fitted = make_pipeline(*chain).fit(train) if chain else None
            train_blocks.append(fitted.transform(train) if fitted is not None else train)
            predicted_blocks.append(fitted.transform(prediction) if fitted is not None else prediction)
        return np.hstack(train_blocks), np.hstack(predicted_blocks)

    errors = []
    for train, val in KFold(3).split(X):
        train_X, val_X = design(X[train], X[val])
        predicted = Ridge().fit(train_X, y[train]).predict(val_X)
        errors.extend((predicted - y[val]) ** 2)
    train_X, unseen_X = design(X, unseen)
    expected = Ridge().fit(train_X, y).predict(unseen_X)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.__init__", lambda *a, **k: pytest.fail("implicit legacy execution"))
    result = nirs4all.run([
        {"feature_augmentation": [StandardScaler], "action": "extend"},
        {"feature_augmentation": [RobustScaler], "action": action},
        MinMaxScaler(), KFold(3), Ridge(),
    ], dataset, workspace_path=tmp_path / "workspace", verbose=0, save_charts=False)
    try:
        assert result.execution_engine == "dag-ml"
        np.testing.assert_allclose(result.cv_best_score, np.sqrt(np.mean(errors)), rtol=1e-5, atol=1e-6)
        archive = result.export(tmp_path / "channels.n4a")
        predicted = nirs4all.predict(archive, unseen, verbose=0)
        np.testing.assert_allclose(np.asarray(predicted.y_pred).ravel(), expected, rtol=1e-5, atol=1e-6)
    finally:
        result.close()
