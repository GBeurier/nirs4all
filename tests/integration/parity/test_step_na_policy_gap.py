"""Public oracle for step-level missing-value replacement."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import nirs4all


@pytest.mark.parity
def test_model_na_replacement_matches_filled_legacy_run_and_exposes_native_gap() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(24, 6))
    y = 2 * x[:, 0] - x[:, 1]
    x[[2, 8, 16], 0] = np.nan
    fill_value = -2.0
    pipeline = [KFold(3), {"model": Ridge(alpha=1.0), "na_policy": "replace", "fill_value": fill_value}]

    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", save_artifacts=False, verbose=0)
    filled = nirs4all.run(
        [KFold(3), Ridge(alpha=1.0)],
        (np.where(np.isnan(x), fill_value, x), y),
        engine="legacy", save_artifacts=False, verbose=0,
    )
    assert legacy.cv_best_score == pytest.approx(filled.cv_best_score, abs=1e-10)
    legacy.close()
    filled.close()

    with pytest.raises(Exception, match="Input X contains NaN"):
        nirs4all.run(pipeline, (x, y), engine="dag-ml", save_artifacts=False, verbose=0)
