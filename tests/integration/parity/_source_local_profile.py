"""Independent source_local_v160 science; the old concatenation gold is historical."""

from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

from nirs4all.operators.transforms import StandardNormalVariate


def assert_source_local_stacking_profile(result: Any, dataset_config: Any) -> None:
    """Qualify source IDs, effective CV folds, independent predictions and genuine FINALs."""
    dataset = dataset_config.get_dataset_at(0)
    sources = dataset.x({"partition": "train"}, "2d", concat_source=False)
    tests = dataset.x({"partition": "test"}, "2d", concat_source=False)
    y = np.asarray(dataset.y({"partition": "train"})).ravel()
    train_ids = np.asarray(dataset.index_column("sample", {"partition": "train"}))
    test_ids = np.asarray(dataset.index_column("sample", {"partition": "test"}))
    positions = {int(sample): index for index, sample in enumerate(train_ids)}
    rows = result.predictions.filter_predictions(load_arrays=True)
    cv = [row for row in rows if str(row["fold_id"]) != "final"]
    finals = [row for row in rows if str(row["fold_id"]) == "final"]
    assert len(sources) == len(tests) == 3
    assert len(cv) == 4 * (3 * 3 + 2 * 3) == 60
    assert len(finals) == 4 * 2 == 8
    assert len(rows) == result.num_predictions == 68
    assert {row["branch_id"] for row in cv if row["model_name"] == "PLSRegression"} == {0, 1, 2}
    assert {row["branch_id"] for row in cv if row["model_name"] == "Ridge"} == {None}

    def selected(name: str, branch: int | None, fold: str, partition: str) -> dict[str, Any]:
        matches = [row for row in rows if row["model_name"] == name and row["branch_id"] == branch
                   and str(row["fold_id"]) == fold and row["partition"] == partition]
        assert len(matches) == 1
        return matches[0]

    # Read actual effective fold identities; additionally verify the declared
    # seeded splitter. No retained TEST row enters fitting or selection.
    folds = []
    for fold, (declared_train, declared_val) in enumerate(KFold(3, shuffle=True, random_state=42).split(sources[0])):
        fit_ids = selected("PLSRegression", 0, str(fold), "train")["sample_indices"]
        val_ids = selected("PLSRegression", 0, str(fold), "val")["sample_indices"]
        fit = np.asarray([positions[int(sample)] for sample in fit_ids])
        val = np.asarray([positions[int(sample)] for sample in val_ids])
        np.testing.assert_array_equal(fit, declared_train)
        np.testing.assert_array_equal(val, declared_val)
        assert not set(fit_ids) & set(val_ids)
        folds.append((fit, val))

    def compare(name: str, branch: int | None, fold: str, predictions: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        for partition, (sample_ids, expected) in predictions.items():
            row = selected(name, branch, fold, partition)
            np.testing.assert_array_equal(row["sample_indices"], sample_ids)
            np.testing.assert_allclose(np.asarray(row["y_pred"]).ravel(), np.asarray(expected).ravel(), rtol=1e-5, atol=1e-5)

    oof_sources, test_sources, final_train, final_test, validation_scores = [], [], [], [], []
    for branch, (raw_train, raw_test) in enumerate(zip(sources, tests, strict=True)):
        transform = StandardNormalVariate().fit(raw_train)
        # Methods owns SNV; the existing feature owner stores processing blocks
        # as float32. This reference never calls SDK run/model controllers.
        x = np.asarray(transform.transform(raw_train), dtype=np.float32)
        test_x = np.asarray(transform.transform(raw_test), dtype=np.float32)
        oof = np.empty(len(y))
        heldout = []
        for fold, (fit, val) in enumerate(folds):
            model = PLSRegression(n_components=10).fit(x[fit], y[fit])
            oof[val] = np.asarray(model.predict(x[val])).ravel()
            heldout.append(np.asarray(model.predict(test_x)).ravel())
            compare("PLSRegression", branch, str(fold), {
                "train": (train_ids[fit], model.predict(x[fit])), "val": (train_ids[val], oof[val]), "test": (test_ids, heldout[-1]),
            })
        oof_sources.append(oof)
        test_sources.append(np.mean(heldout, axis=0))
        validation_scores.append(root_mean_squared_error(y, oof))
        fitted = PLSRegression(n_components=10).fit(x, y)
        final_train.append(np.asarray(fitted.predict(x)).ravel())
        final_test.append(np.asarray(fitted.predict(test_x)).ravel())
        compare("PLSRegression", branch, "final", {"train": (train_ids, final_train[-1]), "test": (test_ids, final_test[-1])})

    meta_x = np.column_stack(oof_sources).astype(np.float32)
    meta_test = np.column_stack(test_sources).astype(np.float32)
    meta_oof = np.empty(len(y))
    for fold, (fit, val) in enumerate(folds):
        model = Ridge(alpha=1.0, random_state=42).fit(meta_x[fit], y[fit])
        meta_oof[val] = np.asarray(model.predict(meta_x[val])).ravel()
        compare("Ridge", None, str(fold), {
            "train": (train_ids[fit], model.predict(meta_x[fit])), "val": (train_ids[val], meta_oof[val]), "test": (test_ids, model.predict(meta_test)),
        })
    validation_scores.append(root_mean_squared_error(y, meta_oof))
    winner = int(np.argmin(validation_scores))
    assert result.cv_best["model_name"] == ("Ridge" if winner == 3 else "PLSRegression")
    np.testing.assert_allclose(result.cv_best_score, validation_scores[winner], rtol=1e-5, atol=1e-5)
    # Retain the established FINAL algorithm, distinct from CV: full-TRAIN
    # bases followed by META on their in-sample TRAIN predictions.
    final_x = np.column_stack(final_train).astype(np.float32)
    final_test_x = np.column_stack(final_test).astype(np.float32)
    model = Ridge(alpha=1.0, random_state=42).fit(final_x, y)
    compare("Ridge", None, "final", {"train": (train_ids, model.predict(final_x)), "test": (test_ids, model.predict(final_test_x))})
