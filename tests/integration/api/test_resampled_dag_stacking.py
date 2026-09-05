"""Resampled outer CV and independently partitioned final OOF remain distinct."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _run(X, y, monkeypatch, named=True):
    import nirs4all

    monkeypatch.setattr("nirs4all.pipeline.dagml.run_paths._named_dict_stacking_legacy_projection", lambda **kwargs: pytest.fail("local stacking executed"))
    branches = {"low": [Ridge(.1)], "high": [Ridge(100)]}
    return nirs4all.run([
        StandardScaler(), ShuffleSplit(n_splits=3, test_size=.25, random_state=42),
        {"branch": branches if named else list(branches.values())}, {"merge": "predictions"}, Ridge(),
    ], (X, y), save_artifacts=False)


def _independent_nested_prediction(X, y, training, predicted):
    """Fit sklearn independently under the native identity-sorted inner partition."""
    outer_training = list(training)
    training = sorted(training, key=lambda index: f"s{index}")
    X = X.astype(np.float32)
    y = y.astype(np.float32).astype(float)
    inner_features, external_features = [], []
    for alpha in (.1, 100):
        oof = {}
        for fold in range(2):
            fit_ids, val_ids = training[1 - fold::2], training[fold::2]
            model = make_pipeline(StandardScaler(), Ridge(alpha)).fit(X[fit_ids], y[fit_ids])
            oof.update(zip(val_ids, model.predict(X[val_ids]), strict=True))
        inner_features.append([oof[sample] for sample in training])
        # Outer model fitting preserves the explicit splitter order (ShuffleSplit
        # is deliberately shuffled); only the native inner KFold sorts IDs.
        full = make_pipeline(StandardScaler(), Ridge(alpha)).fit(X[outer_training], y[outer_training])
        external_features.append(full.predict(X[predicted]))
    meta = Ridge().fit(np.column_stack(inner_features).astype(float), y[training])
    return meta.predict(np.column_stack(external_features).astype(float))


@pytest.mark.parametrize("named", [True, False])
def test_resampled_stacking_refit_coverage_is_separate_and_exportable(tmp_path, monkeypatch, named):
    import nirs4all
    from nirs4all.pipeline.dagml.general_archive import load_general_archive

    rng = np.random.default_rng(641)
    X = rng.normal(size=(36, 5))
    y = X @ np.arange(1., 6.)
    result = _run(X, y, monkeypatch, named)
    meta = result.runs[-1] if named else result
    evidence = next(iter(result.per_dataset.values()))["stacking_evaluation"]
    assert evidence["outer_partition_mode"] == "resampled"
    assert evidence["outer_validation_sample_count"] < 36
    assert evidence["outer_validation_occurrences"] == 27
    assert evidence["refit_oof_is_selection_evidence"] is False
    outer = meta.predictions.filter_predictions(partition="val", load_arrays=True)
    assert {row["fold_id"] for row in outer} == {"0", "1", "2", "avg", "w_avg"}
    # Native preparation produces one held-out contribution per training sample
    # for each base, and no meta validation report for those preparation folds.
    blocks_by_producer = {}
    for frame in result._dagml_node_results:
        for block in frame.get("predictions", []):
            if str(block.get("fold_id", "")).startswith("stacking.refit.inner.") and block["partition"] == "validation":
                blocks_by_producer.setdefault(block["producer_node"], []).extend(block["sample_ids"])
    assert len(blocks_by_producer) == 2
    assert all(len(ids) == len(set(ids)) == 36 for ids in blocks_by_producer.values())
    assert not any(str(report.get("fold_id", "")).startswith("stacking.refit.inner.") for report in result._dagml_score_set["reports"] if report["producer_node"] == "merge:stack")
    occurrences = {}
    for fold, (training, validation) in enumerate(ShuffleSplit(n_splits=3, test_size=.25, random_state=42).split(X)):
        expected = _independent_nested_prediction(X, y, training, validation)
        actual = meta.predictions.filter_predictions(fold_id=str(fold), partition="val", load_arrays=True)[0]
        by_id = dict(zip(validation, expected, strict=True))
        np.testing.assert_allclose(actual["y_pred"], [by_id[index] for index in actual["sample_indices"]], rtol=1e-6, atol=1e-6)
        for index, value in by_id.items():
            occurrences.setdefault(index, []).append(value)
    expected_rmse = np.sqrt(np.mean([(np.mean(values) - y.astype(np.float32)[index]) ** 2 for index, values in occurrences.items()]))
    assert meta.cv_best_score == pytest.approx(expected_rmse, rel=1e-6, abs=1e-6)
    expected_final = _independent_nested_prediction(X, y, list(range(36)), list(range(36)))
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("export or predict fitted"))
    archive = result.export(tmp_path / "resampled.n4a", **({"source": meta.best} if named else {}))
    predicted = nirs4all.predict(archive, X).y_pred
    loaded = load_general_archive(archive)
    assert loaded["manifest"]["stacking_evaluation"] == evidence
    assert predicted.shape == (36,)
    assert np.isfinite(predicted).all()
    np.testing.assert_allclose(predicted, expected_final, rtol=1e-6, atol=1e-6)
    result.close()


def test_resampled_stacking_outer_target_poison_does_not_enter_its_model(monkeypatch):
    rng = np.random.default_rng(642)
    X = rng.normal(size=(36, 5))
    y = X @ np.arange(1., 6.)
    first_validation = next(ShuffleSplit(n_splits=3, test_size=.25, random_state=42).split(X))[1]
    before = _run(X, y, monkeypatch)
    poisoned = y.copy()
    poisoned[first_validation] += 10000
    after = _run(X, poisoned, monkeypatch)
    left = before.runs[-1].predictions.filter_predictions(fold_id="0", partition="val", load_arrays=True)[0]
    right = after.runs[-1].predictions.filter_predictions(fold_id="0", partition="val", load_arrays=True)[0]
    assert set(left["sample_indices"]) == set(first_validation)
    np.testing.assert_array_equal(left["y_pred"], right["y_pred"])
    before.close()
    after.close()
