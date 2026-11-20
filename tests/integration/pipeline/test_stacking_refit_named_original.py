"""Named stacking refit preserves the training-only augmented feature matrix."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


class _RecordedRidge(Ridge):
    """Record the actual fit boundary while retaining unchanged Ridge numerics."""

    def fit(self, X, y, sample_weight=None):
        self.fit_features_ = np.asarray(X).copy()
        self.fit_targets_ = np.asarray(y).copy()
        return super().fit(X, y, sample_weight=sample_weight)


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


@pytest.mark.parametrize("refs", [[1, 0], [1], ["right"]])
def test_stacking_refit_preserves_declared_selection_order(tmp_path, refs):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(24, 3))
    y = x @ np.arange(1., 4.)
    steps = [KFold(3, shuffle=True, random_state=42),
             {"branch": {"left": [Ridge(alpha=.5)], "right": [Ridge(alpha=2.)]}},
             {"merge": {"predictions": refs}}, Ridge(alpha=.1)]
    workspace = tmp_path / "workspace"
    data = {"train_x": x[:18], "train_y": y[:18], "test_x": x[18:], "test_y": y[18:]}
    if refs == ["right"]:
        with pytest.raises(RuntimeError, match="Prediction branch indices must be integers"):
            nirs4all.run(steps, data, engine="legacy", refit=True, workspace_path=workspace, save_charts=False, verbose=0)
        with WorkspaceStore(workspace) as store:
            assert store.query_predictions(fold_id="final").is_empty()
        return
    with nirs4all.run(steps, data, engine="legacy", refit=True, workspace_path=workspace, save_charts=False, verbose=0) as result:
        rows = [row for row in result.predictions.iter_entries() if str(row["fold_id"]) == "final" and row.get("branch_id") is None]
        assert rows and {row["n_features"] for row in rows} == {len(refs)}
        train_x = x[:18].astype(np.float32)
        train_y = y[:18].astype(np.float32)
        predictions = [Ridge(alpha=alpha).fit(train_x, train_y).predict(train_x) for alpha in (.5, 2.)]
        features = np.column_stack([predictions[index] for index in refs])
        expected = Ridge(alpha=.1).fit(features, train_y)
        with WorkspaceStore(workspace) as store:
            model = store.load_artifact(rows[0]["model_artifact_id"])
            np.testing.assert_allclose(model.coef_, expected.coef_, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(model.predict(features), expected.predict(features), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("branch_meta", [False, True])
def test_mixed_selected_refit_preserves_cv_choice_feature_paths_and_training_scope(tmp_path, branch_meta):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import PolynomialFeatures

    from nirs4all.operators.models import MetaModel

    rng = np.random.default_rng(92)
    x = rng.normal(size=(44, 6))
    x[36:] += 1000
    y = x @ np.arange(1., 7.) + x[:, 0] ** 2 + rng.normal(scale=3., size=44)
    branches = {
        "late_features": [StandardScaler(), {"model": _RecordedRidge(alpha=.1), "name": "A0"},
                          {"model": _RecordedRidge(alpha=20.), "name": "B0"}, PCA(n_components=2),
                          {"model": _RecordedRidge(alpha=1.), "name": "C0"}],
        "polynomial": [PolynomialFeatures(degree=2, include_bias=False), {"model": _RecordedRidge(alpha=.1), "name": "A1"},
                       {"model": _RecordedRidge(alpha=50.), "name": "B1"}],
        "feature_only": [PCA(n_components=3), {"model": _RecordedRidge(alpha=1.), "name": "A2"}],
    }
    configs = [{"branch": 1, "select": "best", "metric": "rmse"},
               {"branch": 0, "select": {"top_k": 2}, "metric": "r2"}]
    steps = [StandardScaler(), KFold(3, shuffle=True, random_state=42), {"branch": branches}]
    if branch_meta:
        steps.append({"model": MetaModel(_RecordedRidge(alpha=.3)), "name": "BranchMeta"})
    steps += [{"merge": {"features": [2], "predictions": configs, "include_original": True}},
              {"model": _RecordedRidge(alpha=.7), "name": "FINAL"}]
    train_x = x[:36].astype(np.float32)
    train_y = y[:36].astype(np.float32)
    original = StandardScaler().fit_transform(train_x)
    path0 = StandardScaler().fit_transform(original)
    path0_late = PCA(n_components=2).fit_transform(path0)
    path1 = PolynomialFeatures(degree=2, include_bias=False).fit_transform(original)
    feature_path = PCA(n_components=3).fit_transform(original)
    expected_inputs = {"A0": path0, "B0": path0, "C0": path0_late, "A1": path1, "B1": path1, "A2": feature_path}
    expected_alphas = {"A0": .1, "B0": 20., "C0": 1., "A1": .1, "B1": 50., "A2": 1.}
    observed = []
    for run_index, test_target_shift in enumerate((0., 100000.)):
        workspace = tmp_path / f"run-{run_index}"
        data = {"train_x": x[:36], "train_y": y[:36], "test_x": x[36:], "test_y": y[36:] + test_target_shift}
        with nirs4all.run(steps, data, engine="legacy", refit=True, workspace_path=workspace,
                          save_charts=False, save_artifacts=True, verbose=0) as result:
            rows = list(result.predictions.iter_entries())
            cv = [row for row in rows if row.get("partition") == "val" and str(row["fold_id"]) not in ("final", "avg", "w_avg")]
            selected = []
            for branch_id, metric, count in ((1, "rmse", 1), (0, "r2", 2)):
                scores = {}
                for row in cv:
                    if row.get("branch_id") == branch_id:
                        scores.setdefault(row["model_name"], row["scores"]["val"][metric])
                names = sorted(scores)
                names.sort(key=scores.__getitem__, reverse=metric == "r2")
                selected.extend((branch_id, name) for name in names[:count])
            assert len(selected) == 3
            finals = [row for row in rows if str(row["fold_id"]) == "final" and row["model_name"] == "FINAL"]
            assert finals and {row["n_features"] for row in finals} == {12}
            assert all((row.get("metadata") or {}).get("stacking_role") == "meta_model" for row in finals)
            with WorkspaceStore(workspace) as store:
                # Host integration tests the fitted feature/identity boundary;
                # numerical prediction remains delegated to the fitted owner.
                fitted_predictions = {}
                for name, values in expected_inputs.items():
                    row = next(row for row in rows if row["model_name"] == name and str(row["fold_id"]) == "final")
                    fitted = store.load_artifact(row["model_artifact_id"])
                    assert fitted.alpha == expected_alphas[name]
                    np.testing.assert_allclose(fitted.fit_features_, values, rtol=1e-5, atol=1e-5)
                    np.testing.assert_array_equal(fitted.fit_targets_, train_y)
                    expected_fit = Ridge(alpha=expected_alphas[name]).fit(fitted.fit_features_, train_y)
                    np.testing.assert_allclose(fitted.coef_, expected_fit.coef_, rtol=1e-5, atol=1e-5)
                    fitted_predictions[name] = fitted.predict(fitted.fit_features_)
                if branch_meta:
                    for branch_id, names in ((0, ["A0", "B0", "C0"]), (1, ["A1", "B1"])):
                        row = next(row for row in rows if row["model_name"] == "BranchMeta"
                                   and str(row["fold_id"]) == "final" and row.get("branch_id") == branch_id)
                        fitted = store.load_artifact(row["model_artifact_id"])
                        expected_columns = np.column_stack([fitted_predictions[name] for name in names])
                        np.testing.assert_allclose(fitted.fit_features_, expected_columns, rtol=1e-5, atol=1e-5)
                        np.testing.assert_array_equal(fitted.fit_targets_, train_y)
                        fitted_predictions[f"BranchMeta:{branch_id}"] = fitted.predict(fitted.fit_features_)
                columns = [fitted_predictions[f"BranchMeta:{branch_id}" if name == "BranchMeta" else name]
                           for branch_id, name in selected]
                features = np.column_stack([original, feature_path, *columns])
                model = store.load_artifact(finals[0]["model_artifact_id"])
                np.testing.assert_allclose(model.fit_features_, features, rtol=1e-5, atol=1e-5)
                np.testing.assert_array_equal(model.fit_targets_, train_y)
                expected_final = Ridge(alpha=.7).fit(model.fit_features_, train_y)
                np.testing.assert_allclose(model.coef_, expected_final.coef_, rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(model.predict(features), expected_final.predict(features), rtol=1e-5, atol=1e-5)
                observed.append((selected, model.coef_.copy()))
    assert observed[0][0] == observed[1][0]
    np.testing.assert_array_equal(observed[0][1], observed[1][1])
