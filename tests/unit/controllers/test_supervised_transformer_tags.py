"""Supervised transform dispatch supports both public sklearn and legacy tags."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSSVD
from sklearn.feature_selection import SelectFdr, SelectKBest, SequentialFeatureSelector
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler, TargetEncoder

from nirs4all.controllers.transforms.transformer import TransformerMixinController


@pytest.mark.parametrize("operator", [
    SelectFdr(), SelectKBest(), SequentialFeatureSelector(Ridge()), PLSSVD(),
    NeighborhoodComponentsAnalysis(), TargetEncoder(),
])
def test_structured_tags_supply_y_to_supervised_operators(operator):
    assert TransformerMixinController._uses_y(operator)


def test_unsupervised_transform_stays_unsupervised():
    assert not TransformerMixinController._requires_y(StandardScaler())


def test_custom_legacy_tags_remain_supported():
    class Legacy:
        def _more_tags(self):
            return {"requires_y": True}

    assert TransformerMixinController._requires_y(Legacy())


@pytest.mark.parametrize("selector", [{"class": "sklearn.feature_selection.SelectFdr"}, SequentialFeatureSelector(Ridge(), n_features_to_select=2, cv=2)])
def test_real_pipeline_passes_training_targets_to_selector(tmp_path, selector):
    import nirs4all

    rng = np.random.default_rng(88)
    X = rng.normal(size=(40, 5))
    y = (X[:, 0] > 0).astype(int)
    with nirs4all.run([selector, RidgeClassifier()], (X, y, {"train": 32}), engine="legacy",
                     workspace_path=tmp_path, verbose=0, save_charts=False, refit=False) as result:
        predictions = result.predictions.filter_predictions(partition="test", load_arrays=True)
        assert predictions
        assert np.isfinite(np.asarray(predictions[0]["y_pred"])).all()
        assert len(predictions[0]["y_pred"]) == 8
