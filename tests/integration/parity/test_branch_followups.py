"""Public parity cases for independent branch models and repetition sources."""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.pipeline.dagml.folds import _build_folds
from nirs4all.pipeline.dagml.run_paths import _reshape_for_rep_fusion
from tests.integration.pipeline.test_separation_branch_generators import create_dataset_with_metadata

from ._datasets import dataset_path
from .test_dagml_cli_runner import _equal_rep_dataset, _two_source_distinct_dataset


def test_metadata_model_branches_without_merge_keep_partition_predictions() -> None:
    pipeline = [
        ShuffleSplit(n_splits=2, random_state=42),
        {"branch": {"by_metadata": "site", "steps": [{"model": Ridge(alpha=1.0)}]}},
    ]
    legacy = nirs4all.run(pipeline, create_dataset_with_metadata(), engine="legacy", save_artifacts=False, verbose=0)
    dataset = create_dataset_with_metadata()
    native = nirs4all.run(pipeline, dataset, engine="dag-ml", save_artifacts=False, verbose=0)
    assert legacy.num_predictions > 0
    labels = dict(zip(dataset.index_column("sample", {}), dataset.metadata_column("site", {}), strict=True))
    validation = [row for row in native.predictions.filter_predictions(load_arrays=True) if row["partition"] == "val" and str(row["fold_id"]).isdigit()]
    assert {row["branch_name"] for row in validation} == {"site_A", "site_B"}
    for row in validation:
        assert row["sample_indices"]
        assert all(labels[sample] == row["branch_name"] for sample in row["sample_indices"])


def test_metadata_operator_product_select_archive_replays_fitted_chains(tmp_path) -> None:
    pipeline = [
        ShuffleSplit(n_splits=2, random_state=42),
        {"branch": {
            "by_metadata": "site",
            "steps": [{"_or_": [StandardScaler(), MinMaxScaler()]}, Ridge(alpha=1.0)],
        }},
    ]
    legacy = nirs4all.run(pipeline, create_dataset_with_metadata(), engine="legacy", save_artifacts=False, verbose=0)
    dataset = create_dataset_with_metadata()
    native = nirs4all.run(pipeline, dataset, engine="dag-ml", save_artifacts=False, verbose=0)
    assert legacy.num_predictions > 0
    average_rows = [
        row for row in native.predictions.filter_predictions(load_arrays=True)
        if row["partition"] == "val" and row["fold_id"] == "avg"
    ]
    by_variant: dict[str, set[str]] = {}
    for row in average_rows:
        variant_id = row["result_metadata"]["dagml_projection"]["variant_id"]
        by_variant.setdefault(variant_id, set()).add(row["branch_name"])
    assert len(by_variant) == 4  # Two independent choices at each of two metadata sites.
    assert all(branches == {"site_A", "site_B"} for branches in by_variant.values())

    archive = native.export(tmp_path / "metadata_operator_product.n4a")
    x = np.asarray(dataset.x({"partition": "train"}, layout="2d"))
    groups = dataset.metadata_column("site", {"partition": "train"})
    with pytest.raises(ValueError, match="metadata.*site"):
        nirs4all.predict(archive, x, engine="legacy")
    replayed = nirs4all.predict(archive, {"X": x, "metadata": {"site": groups}}, engine="legacy")
    final_rows = [
        row for row in native.predictions.filter_predictions(load_arrays=True)
        if row["partition"] == "train" and row["fold_id"] == "final"
    ]
    assert {row["branch_name"] for row in final_rows} == {"site_A", "site_B"}
    expected = np.full(len(x), np.nan)
    for row in final_rows:
        expected[np.asarray(row["sample_indices"], dtype=int)] = np.asarray(row["y_pred"]).ravel()
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(np.asarray(replayed.y_pred).ravel(), expected, atol=1e-9)


def test_branch_only_archive_replays_native_selected_refit(tmp_path) -> None:
    pipeline = [
        KFold(n_splits=3, shuffle=True, random_state=42),
        {"branch": {
            "pls": [{"model": PLSRegression(n_components=3)}],
            "ridge": [{"model": Ridge(alpha=1.0)}],
        }},
    ]
    native = nirs4all.run(pipeline, dataset_path("regression"), engine="dag-ml", save_artifacts=False, verbose=0)
    selected = native.per_dataset["regression"]["selected_branch"]
    assert selected in {"pls", "ridge"}
    scored = [row for row in native.predictions.filter_predictions(load_arrays=True) if row["branch_name"] == selected and row["partition"] == "test" and row["fold_id"] == "final"]
    assert len(scored) == 1
    archive = native.export(tmp_path / "branch_selected.n4a")
    dataset = DatasetConfigs(dataset_path("regression")).get_dataset_at(0)
    features = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        replayed = nirs4all.predict(archive, features, engine="legacy")
    np.testing.assert_allclose(np.asarray(replayed.y_pred).ravel(), np.asarray(scored[0]["y_pred"]).ravel(), atol=1e-6)


def test_stacking_meta_sibling_param_matches_explicit_estimator() -> None:
    prefix = [
        KFold(n_splits=3, shuffle=True, random_state=42),
        {"branch": [[{"model": PLSRegression(n_components=3)}], [{"model": Ridge(alpha=1.0)}]]},
        {"merge": "predictions"},
    ]
    sibling = [*prefix, {"model": Ridge(), "alpha": 0.2}]
    explicit = [*prefix, {"model": Ridge(alpha=0.2)}]
    legacy = nirs4all.run(sibling, dataset_path("regression"), engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(sibling, dataset_path("regression"), engine="dag-ml", save_artifacts=False, verbose=0)
    direct = nirs4all.run(explicit, dataset_path("regression"), engine="dag-ml", save_artifacts=False, verbose=0)
    assert legacy.num_predictions > 0
    assert native.execution_engine == "dag-ml"
    assert native.best_rmse == pytest.approx(direct.best_rmse, abs=1e-9)


def test_rep_to_sources_by_source_matches_per_source_oracle() -> None:
    splitter = KFold(n_splits=3, shuffle=True, random_state=42)
    pipeline = [
        {"rep_to_sources": "sample_id"}, splitter,
        {"branch": {"by_source": True, "steps": [{"model": PLSRegression(n_components=3)}]}},
        {"merge": {"sources": "concat"}},
    ]
    legacy = nirs4all.run(pipeline, _equal_rep_dataset(), engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, _equal_rep_dataset(), engine="dag-ml", save_artifacts=False, verbose=0)
    assert legacy.num_predictions > 0
    reshaped = _equal_rep_dataset()
    _reshape_for_rep_fusion({"rep_to_sources": "sample_id"}, reshaped)
    pool = reshaped.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, reshaped, pool, set())
    native_avg = {row["branch_name"]: row for row in native.predictions.filter_predictions(load_arrays=True) if row["partition"] == "val" and row["fold_id"] == "avg"}
    assert set(native_avg) == {"source_0", "source_1", "source_2"}
    for source_index in range(3):
        predictions: dict[int, float] = {}
        targets: dict[int, float] = {}
        for train, validation in folds:
            train_x = np.asarray(reshaped.x_rows(train, layout="2d", concat_source=False)[source_index])
            validation_x = np.asarray(reshaped.x_rows(validation, layout="2d", concat_source=False)[source_index])
            train_y = np.asarray(reshaped.y({"sample": train})).ravel()
            validation_y = np.asarray(reshaped.y({"sample": validation})).ravel()
            model = PLSRegression(n_components=3).fit(train_x, train_y)
            for sample, prediction, target in zip(validation, model.predict(validation_x).ravel(), validation_y, strict=True):
                predictions[sample], targets[sample] = float(prediction), float(target)
        ids = sorted(predictions)
        oracle = float(np.sqrt(mean_squared_error([targets[sample] for sample in ids], [predictions[sample] for sample in ids])))
        assert native_avg[f"source_{source_index}"]["val_score"] == pytest.approx(oracle, abs=1e-9)


@pytest.mark.parametrize("generate_second_source", [False, True])
def test_by_source_operator_generator_matches_per_source_oracle(generate_second_source: bool) -> None:
    """Independent operator choices must score the intended source-local fits."""
    splitter = ShuffleSplit(n_splits=2, random_state=42)
    dataset = _two_source_distinct_dataset()
    pipeline = [
        {"y_processing": MinMaxScaler()},
        splitter,
        {"branch": {"by_source": True, "steps": {
            "source_0": [{"_or_": [StandardScaler(), MinMaxScaler()]}, PLSRegression(5)],
            "source_1": (
                [{"_or_": [StandardScaler(), MinMaxScaler()]}, PLSRegression(5)]
                if generate_second_source else [StandardScaler(), PLSRegression(5)]
            ),
        }}},
        {"merge": {"sources": "concat"}},
    ]
    legacy = nirs4all.run(pipeline, _two_source_distinct_dataset(), engine="legacy", save_artifacts=False, verbose=0)
    native = nirs4all.run(pipeline, dataset, engine="dag-ml", save_artifacts=False, verbose=0)
    assert legacy.num_predictions > 0

    pool = dataset.index_column("sample", {"partition": "train"})
    targets = dict(zip(pool, np.asarray(dataset.y({"partition": "train"})).ravel(), strict=True))
    folds = _build_folds(splitter, dataset, pool, set())

    def oracle(source_index: int, scaler: StandardScaler | MinMaxScaler) -> float:
        predictions: dict[int, list[float]] = defaultdict(list)
        for train, validation in folds:
            train_x = np.asarray(dataset.x_rows(train, layout="2d", concat_source=False)[source_index])
            validation_x = np.asarray(dataset.x_rows(validation, layout="2d", concat_source=False)[source_index])
            model = make_pipeline(scaler, PLSRegression(5)).fit(train_x, [targets[sample] for sample in train])
            for sample, prediction in zip(validation, model.predict(validation_x).ravel(), strict=True):
                predictions[sample].append(float(prediction))
        samples = sorted(predictions)
        return float(np.sqrt(mean_squared_error(
            [targets[sample] for sample in samples],
            [np.mean(predictions[sample]) for sample in samples],
        )))

    expected_source_0 = sorted([oracle(0, StandardScaler()), oracle(0, MinMaxScaler())])
    expected_source_1 = sorted([oracle(1, StandardScaler()), oracle(1, MinMaxScaler())]) if generate_second_source else [oracle(1, StandardScaler())]
    averages: dict[str, dict[str, float]] = defaultdict(dict)
    for row in native.predictions.filter_predictions(load_arrays=True):
        if row["partition"] != "val" or row["fold_id"] != "avg":
            continue
        assert len(row["y_pred"]) > 0, "every variant must retain every source's OOF predictions"
        variant_id = row["result_metadata"]["dagml_projection"]["variant_id"]
        averages[variant_id][row["branch_name"]] = row["val_score"]
    assert len(averages) == (4 if generate_second_source else 2)
    assert all(set(branches) == {"source_0", "source_1"} for branches in averages.values())
    assert sorted(branches["source_0"] for branches in averages.values()) == pytest.approx(sorted(expected_source_0 * len(expected_source_1)), abs=1e-9)
    assert sorted(branches["source_1"] for branches in averages.values()) == pytest.approx(sorted(expected_source_1 * 2), abs=1e-9)
    expected_selected = min(
        np.sqrt((left * left + right * right) / 2)
        for left in expected_source_0 for right in expected_source_1
    )
    candidate_scores = {
        variant_id: float(np.sqrt((branches["source_0"] ** 2 + branches["source_1"] ** 2) / 2))
        for variant_id, branches in averages.items()
    }
    final_variants = {
        row["result_metadata"]["dagml_projection"]["variant_id"]
        for row in native.predictions.filter_predictions(load_arrays=True)
        if row["fold_id"] == "final"
    }
    assert len(final_variants) == 1
    assert candidate_scores[final_variants.pop()] == pytest.approx(expected_selected, abs=1e-9)
    assert native.cv_best_score == pytest.approx(min(min(branches.values()) for branches in averages.values()), abs=1e-9)
