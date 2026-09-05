"""Shared preprocessing is a fold-local prefix of each stacked base learner."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def test_shared_stacking_prefix_equals_explicit_branch_prefixes(monkeypatch):
    import nirs4all

    rng = np.random.default_rng(371)
    X = rng.normal(size=(36, 6))
    X[-12:] += 30
    y = X @ np.arange(1.0, 7.0)
    split = KFold(n_splits=3)
    explicit = nirs4all.run(
        [split, {"branch": [[StandardScaler(), Ridge(1)], [StandardScaler(), Ridge(2)]]}, {"merge": "predictions"}, Ridge(0.1)],
        (X, y), save_artifacts=False,
    )
    fitted_rows = []
    original_fit = StandardScaler.fit

    def fit(estimator, values, *args, **kwargs):
        fitted_rows.append(len(values))
        return original_fit(estimator, values, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", fit)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("legacy execution"))
    shared = nirs4all.run(
        [StandardScaler(), split, {"branch": [[Ridge(1)], [Ridge(2)]]}, {"merge": "predictions"}, Ridge(0.1)],
        (X, y), save_artifacts=False,
    )
    assert shared._dagml_score_set is not None
    assert shared._dagml_refit_artifacts
    assert fitted_rows.count(36) == 2  # only the two final base refits, not a global CV prefit
    assert any(size < 24 for size in fitted_rows)  # nested inner training views actually ran
    assert shared.cv_best_score == explicit.cv_best_score


def test_outer_validation_targets_cannot_influence_stacked_validation_predictions():
    import nirs4all

    rng = np.random.default_rng(842)
    X = rng.normal(size=(36, 5))
    y = X @ np.arange(1.0, 6.0)

    def run(targets):
        return nirs4all.run(
            [StandardScaler(), KFold(n_splits=3), {"branch": [[Ridge(1)], [Ridge(2)]]}, {"merge": "predictions"}, Ridge(0.1)],
            (X, targets), save_artifacts=False,
        )

    before = run(y)
    poisoned = y.copy()
    poisoned[:12] += 10000
    after = run(poisoned)

    def held_out(result):
        candidates = [
            row for row in result.predictions.filter_predictions(load_arrays=True)
            if row["partition"] == "val" and set(row.get("sample_indices", [])) == set(range(12))
        ]
        assert len(candidates) == 1
        return candidates[0]["y_pred"]

    np.testing.assert_array_equal(held_out(before), held_out(after))
