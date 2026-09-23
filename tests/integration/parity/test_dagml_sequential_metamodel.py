"""Public oracle for a sequential classifier followed by a MetaModel."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from nirs4all.operators.models import MetaModel


@pytest.mark.parametrize("use_proba", [False, True])
def test_sequential_classification_metamodel_legacy_contract_is_not_yet_lowered(use_proba):
    import nirs4all

    rng = np.random.default_rng(79)
    features = rng.normal(size=(30, 6))
    targets = (features[:, 0] + features[:, 1] > 0).astype(int)
    pipeline = [
        StratifiedKFold(2, shuffle=True, random_state=42),
        LogisticRegression(max_iter=300),
        {"model": MetaModel(model=LogisticRegression(max_iter=300), use_proba=use_proba)},
    ]
    legacy = nirs4all.run(pipeline, (features, targets), engine="legacy", refit=False,
                          save_artifacts=False, save_charts=False, verbose=0)
    assert legacy.cv_best_score == pytest.approx(0.9333333333333333)

    with pytest.raises(Exception, match="MetaModel.*fit"):
        nirs4all.run(pipeline, (features, targets), engine="dag-ml", refit=False,
                     save_artifacts=False, save_charts=False, verbose=0)
