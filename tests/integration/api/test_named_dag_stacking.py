"""Named base/meta models retain native ownership, nested validation and export identity."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def _run(X, y, monkeypatch):
    import nirs4all

    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._named_dict_stacking_legacy_projection", lambda **kwargs: pytest.fail("local stacking projection executed"))
    return nirs4all.run(
        [StandardScaler(), KFold(3), {"branch": {"low": [Ridge(0.1)], "high": [Ridge(100)]}}, {"merge": "predictions"}, Ridge(1)],
        (X, y), save_artifacts=False,
    )


def test_named_stacking_preserves_model_views_and_selected_exports(tmp_path, monkeypatch):
    import nirs4all

    rng = np.random.default_rng(618)
    X = rng.normal(size=(36, 5))
    y = X @ np.arange(1.0, 6.0)
    result = _run(X, y, monkeypatch)
    assert result._dagml_score_set is not None
    assert len(result.runs) == 3
    assert len(result._dagml_refit_artifacts) == 3
    assert {row["branch_name"] for row in result.predictions.filter_predictions()} == {"low", "high", None}
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("export retrained"))
    for index, child in enumerate(result.runs):
        archive = result.export(tmp_path / f"producer-{index}.n4a", source=child.best)
        predicted = nirs4all.predict(archive, X).y_pred
        if index < 2:
            final_rows = child.predictions.filter_predictions(fold_id="final", partition="train", load_arrays=True)
            assert len(final_rows) == 1
            expected = final_rows[0]["y_pred"]
        else:
            # No held-out test was supplied. The meta REFIT artifact exists,
            # but no raw-input training score should be fabricated for it.
            base = [view._dagml_refit_artifacts[0]["estimator"].predict(X.astype(np.float32)) for view in result.runs[:2]]
            meta = next(artifact["estimator"] for artifact in child._dagml_refit_artifacts if artifact["controller_id"] == "controller:nirs4all.meta_model")
            expected = meta.predict(np.column_stack(base))
        np.testing.assert_array_equal(predicted, expected)
    result.close()


def test_named_stacking_outer_targets_do_not_leak(monkeypatch):
    rng = np.random.default_rng(619)
    X = rng.normal(size=(36, 5))
    y = X @ np.arange(1.0, 6.0)
    before = _run(X, y, monkeypatch)
    changed = y.copy()
    changed[:12] += 10000
    after = _run(X, changed, monkeypatch)

    def outer_meta(result):
        rows = result.runs[-1].predictions.filter_predictions(partition="val", load_arrays=True)
        return next(row["y_pred"] for row in rows if set(row["sample_indices"]) == set(range(12)))

    np.testing.assert_array_equal(outer_meta(before), outer_meta(after))
    before.close()
    after.close()


def test_named_stacking_double_digit_branches_keep_numeric_identity(monkeypatch):
    import nirs4all

    rng = np.random.default_rng(620)
    X = rng.normal(size=(24, 3))
    y = X @ np.array([1.0, 2.0, 3.0])
    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._named_dict_stacking_legacy_projection", lambda **kwargs: pytest.fail("local stacking projection executed"))
    branches = {f"alias-{index}": [Ridge(alpha=index + 0.1)] for index in range(11)}
    result = nirs4all.run([KFold(3), {"branch": branches}, {"merge": "predictions"}, Ridge()], (X, y), save_artifacts=False)
    for index, child in enumerate(result.runs[:-1]):
        capture = child._dagml_refit_artifacts[0]
        assert capture["estimator"].alpha == index + 0.1
        assert {row["branch_name"] for row in child.predictions.filter_predictions()} == {f"alias-{index}"}
        assert {row["branch_id"] for row in child.predictions.filter_predictions()} == {index}
    result.close()
