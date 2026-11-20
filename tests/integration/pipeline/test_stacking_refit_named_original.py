"""Named stacking refit preserves the training-only augmented feature matrix."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


@pytest.mark.parametrize("direct_models", [False, True])
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("include_original", [False, True])
def test_stacking_refit_preserves_names_order_parameters_and_original_features(tmp_path, named, include_original, direct_models):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(40, 6))
    x[32:] += 1000  # Held-out rows must never influence either fitted scaler.
    y = x @ np.arange(1., 7.) + rng.normal(scale=.05, size=40)
    data = {"train_x": x[:32], "train_y": y[:32], "test_x": x[32:], "test_y": y[32:]}
    branches = {"second_named_first": [StandardScaler(), {"model": Ridge(alpha=.5), "name": "BASE_A"}],
                "first_named_second": [StandardScaler(), {"model": Ridge(alpha=2.), "name": "BASE_B"}]}
    if direct_models:
        branches = {name: [branch[0], branch[1]["model"]] for name, branch in branches.items()}
    branch_spec = branches if named else [{"name": key, "steps": value} for key, value in branches.items()]
    steps = [StandardScaler(), KFold(3, shuffle=True, random_state=42), {"branch": branch_spec},
             {"merge": {"predictions": [0, 1], "include_original": include_original}},
             Ridge(alpha=.1) if direct_models else {"model": Ridge(alpha=.1), "name": "META"}]
    workspace = tmp_path / "workspace"
    with nirs4all.run(steps, data, engine="legacy", refit=True, workspace_path=workspace,
                      save_charts=False, save_artifacts=True, verbose=0) as result:
        rows = list(result.predictions.iter_entries())
        final = [row for row in rows if str(row["fold_id"]) == "final"]
        meta_rows = [row for row in final if (row.get("metadata") or {}).get("stacking_role") == "meta_model"]
        meta_name = "Ridge" if direct_models else "META"
        assert meta_rows and all(row["model_name"] == meta_name for row in meta_rows)
        assert {row["n_features"] for row in meta_rows} == {8 if include_original else 2}
        cv_meta = [row for row in rows if row["model_name"] == meta_name and row.get("branch_id") is None and str(row["fold_id"]) != "final"]
        assert cv_meta and {row["n_features"] for row in cv_meta} == {8 if include_original else 2}
        bases = [row for row in final if (row.get("metadata") or {}).get("stacking_role") == "base_model"]
        assert {(row["model_name"], row["branch_id"], row["branch_name"]) for row in bases} == {
            ("Ridge" if direct_models else "BASE_A", 0, "second_named_first"),
            ("Ridge" if direct_models else "BASE_B", 1, "first_named_second")}
        # Use the actual dataset storage precision at the SDK boundary.
        train_x = np.asarray(x[:32], dtype=np.float32)
        train_y = np.asarray(y[:32], dtype=np.float32)
        prefix = StandardScaler().fit(train_x)
        original = prefix.transform(train_x)
        branch_x = StandardScaler().fit_transform(original)
        expected_bases = [Ridge(alpha=alpha).fit(branch_x, train_y).predict(branch_x) for alpha in (.5, 2.)]
        features = np.column_stack(([original] if include_original else []) + expected_bases)
        expected_meta = Ridge(alpha=.1).fit(features, train_y)
        with WorkspaceStore(workspace) as store:
            fitted_meta = store.load_artifact(meta_rows[0]["model_artifact_id"])
            assert fitted_meta.n_features_in_ == features.shape[1]
            np.testing.assert_allclose(fitted_meta.coef_, expected_meta.coef_, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(fitted_meta.predict(features), expected_meta.predict(features), rtol=1e-5, atol=1e-5)
            for name, alpha in (("BASE_A", .5), ("BASE_B", 2.)):
                selected = next(row for row in bases if row["branch_id"] == (0 if name == "BASE_A" else 1))
                fitted = store.load_artifact(selected["model_artifact_id"])
                assert fitted.alpha == alpha
                np.testing.assert_allclose(fitted.predict(branch_x), expected_bases[0 if name == "BASE_A" else 1], rtol=1e-5, atol=1e-5)


def test_direct_unfitted_forest_branch_refits_without_truth_value_inspection(tmp_path):
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(19)
    x = rng.normal(size=(36, 4))
    y = x @ np.arange(1., 5.)
    steps = [KFold(3, shuffle=True, random_state=42),
             {"branch": {"forest": [RandomForestRegressor(n_estimators=5, random_state=42)],
                         "ridge": [Ridge(alpha=2.)]}},
             {"merge": "predictions"}, Ridge(alpha=.1)]
    with nirs4all.run(steps, {"train_x": x[:28], "train_y": y[:28], "test_x": x[28:], "test_y": y[28:]},
                      engine="legacy", refit=True, workspace_path=tmp_path / "workspace", save_charts=False, verbose=0) as result:
        final = [row for row in result.predictions.iter_entries() if str(row["fold_id"]) == "final"]
        assert any((row.get("metadata") or {}).get("stacking_role") == "meta_model" for row in final)
        assert any(row["model_name"] == "RandomForestRegressor" and row.get("branch_name") == "forest" for row in final)


@pytest.mark.parametrize("refs,reason", [([1, 0], "cannot preserve declared merge configuration"),
                                        ([1], "cannot preserve declared merge configuration"),
                                        (["right"], "Prediction branch indices must be integers")])
def test_stacking_refit_refuses_merge_selection_it_cannot_preserve(tmp_path, refs, reason):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(24, 3))
    y = x @ np.arange(1., 4.)
    steps = [KFold(3, shuffle=True, random_state=42),
             {"branch": {"left": [Ridge(alpha=.5)], "right": [Ridge(alpha=2.)]}},
             {"merge": {"predictions": refs}}, Ridge(alpha=.1)]
    workspace = tmp_path / "workspace"
    with pytest.raises((NotImplementedError, RuntimeError), match=reason):
        nirs4all.run(steps, {"train_x": x[:18], "train_y": y[:18], "test_x": x[18:], "test_y": y[18:]},
                      engine="legacy", refit=True, workspace_path=workspace, save_charts=False, verbose=0)
    with WorkspaceStore(workspace) as store:
        assert store.query_predictions(fold_id="final").is_empty()
