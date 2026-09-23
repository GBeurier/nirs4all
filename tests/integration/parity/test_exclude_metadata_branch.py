"""Exclusion followed by a metadata model branch, with a direct per-group oracle."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.filters.y_outlier import YOutlierFilter

from ._dagml_cli import dagml_cli_path

pytestmark = pytest.mark.parity


def _case() -> tuple[SpectroDataset, np.ndarray, np.ndarray, np.ndarray]:
    """Balanced groups in train and test, with one clear training outlier."""
    rng = np.random.default_rng(42)
    features = rng.normal(size=(120, 20))
    targets = features[:, :5].sum(axis=1) + rng.normal(size=120) * 0.05
    targets[3] = 25.0
    groups = np.asarray(["A", "B"] * 60)
    dataset = SpectroDataset("balanced_metadata")
    dataset.add_samples(features[:96], indexes={"partition": "train"})
    dataset.add_samples(features[96:], indexes={"partition": "test"})
    dataset.add_targets(targets[:96])
    dataset.add_targets(targets[96:])
    dataset.add_metadata(groups[:96].reshape(-1, 1), headers=["group"])
    dataset.add_metadata(groups[96:].reshape(-1, 1), headers=["group"])
    return dataset, features, targets, groups


def _pipeline(*, keep_in_oof: bool) -> list[object]:
    return [
        {"exclude": YOutlierFilter(method="iqr", threshold=1.0), "keep_in_oof": keep_in_oof},
        KFold(n_splits=3, shuffle=True, random_state=42),
        {"branch": {"by_metadata": "group", "steps": [{"model": PLSRegression(n_components=3)}]}},
        {"merge": "concat"},
    ]


def _oracle(features: np.ndarray, targets: np.ndarray, groups: np.ndarray, *, keep_in_oof: bool) -> tuple[float, float, int]:
    """Fit the filter globally, then each fold's model only on kept rows of its group."""
    exclusion = YOutlierFilter(method="iqr", threshold=1.0)
    exclusion.fit(features[:96], targets[:96])
    kept_mask = exclusion.get_mask(features[:96], targets[:96])
    kept = set(np.flatnonzero(kept_mask).tolist())
    assert len(kept) == 95 and 3 not in kept
    pool = list(range(96)) if keep_in_oof else sorted(kept)
    predictions: dict[int, float] = {}
    for train_positions, validation_positions in KFold(n_splits=3, shuffle=True, random_state=42).split(pool):
        train = [pool[position] for position in train_positions if pool[position] in kept]
        validation = [pool[position] for position in validation_positions]
        for group in ("A", "B"):
            local_train = [sample for sample in train if groups[sample] == group]
            local_validation = [sample for sample in validation if groups[sample] == group]
            model = PLSRegression(n_components=3).fit(features[local_train], targets[local_train])
            for sample, value in zip(local_validation, model.predict(features[local_validation]).ravel(), strict=True):
                predictions[sample] = float(value)
    assert set(predictions) == set(pool)
    cv_score = float(np.sqrt(mean_squared_error(targets[pool], [predictions[sample] for sample in pool])))

    final_predictions = np.empty(24)
    for group in ("A", "B"):
        local_train = [sample for sample in sorted(kept) if groups[sample] == group]
        local_test = [sample for sample in range(96, 120) if groups[sample] == group]
        model = PLSRegression(n_components=3).fit(features[local_train], targets[local_train])
        final_predictions[np.asarray(local_test) - 96] = model.predict(features[local_test]).ravel()
    test_score = float(np.sqrt(mean_squared_error(targets[96:], final_predictions)))
    return cv_score, test_score, len(pool)


@pytest.mark.parametrize("mechanism", ["inprocess", "cli"])
@pytest.mark.parametrize("keep_in_oof", [False, True])
def test_exclude_then_metadata_branch_matches_per_group_oracle(
    mechanism: str, keep_in_oof: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = dagml_cli_path()
    if mechanism == "cli" and not cli.exists():
        pytest.skip(f"dag-ml-cli binary not built at {cli}")
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "0" if mechanism == "cli" else "1")
    if mechanism == "cli":
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))

    dataset, features, targets, groups = _case()
    expected_cv, expected_test, expected_oof_rows = _oracle(features, targets, groups, keep_in_oof=keep_in_oof)
    if mechanism == "inprocess" and not keep_in_oof:
        # The historical grammar executes, but cv_best_score selects its best
        # group-local row rather than scoring the reassembled OOF universe.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            legacy = nirs4all.run(_pipeline(keep_in_oof=False), _case()[0], engine="legacy", save_artifacts=False, verbose=0)
        assert legacy.num_predictions > 0
        legacy_averages = [
            row for row in legacy.predictions.filter_predictions(load_arrays=True)
            if row["partition"] == "val" and row["fold_id"] == "avg"
        ]
        assert {row["branch_name"] for row in legacy_averages} == {"A", "B"}
        assert legacy.cv_best_score == min(row["val_score"] for row in legacy_averages)
        assert legacy.cv_best_score < expected_cv

    native = nirs4all.run(_pipeline(keep_in_oof=keep_in_oof), dataset, engine="dag-ml", save_artifacts=False, verbose=0)
    assert native.execution_engine == "dag-ml"
    assert native.cv_best_score == pytest.approx(expected_cv, abs=1e-5)
    assert native.best_rmse == pytest.approx(expected_test, abs=1e-5)
    averages = [
        report for report in native._dagml_score_set["reports"]  # noqa: SLF001 - native score evidence
        if report["producer_node"] == "merge:concat" and report["partition"] == "validation" and report["fold_id"] == "avg"
    ]
    assert len(averages) == 1
    assert averages[0]["row_count"] == expected_oof_rows
