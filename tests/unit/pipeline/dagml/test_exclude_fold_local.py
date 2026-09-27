"""``exclude(keep_in_oof=True)`` fits its filters on each fold's train rows only.

Validation targets must never decide which rows a fold trains on; the full-train fit marks the
envelope and drives the refit. Filter errors and multi-target Y follow ``ExcludeController``.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from nirs4all.data import SpectroDataset
from nirs4all.operators.filters.base import SampleFilter
from nirs4all.operators.filters.y_outlier import YOutlierFilter
from nirs4all.pipeline.dagml.envelope import build_fold_set
from nirs4all.pipeline.dagml.exclude import FoldLocalExclusion, _resolve_exclude, _resolve_tags
from nirs4all.pipeline.dagml.folds import FoldLocalFolds, _build_folds
from nirs4all.pipeline.dagml.identity import mint_identity


def _dataset(y: np.ndarray) -> SpectroDataset:
    rng = np.random.default_rng(4)
    dataset = SpectroDataset("fold-local")
    dataset.add_samples(rng.normal(size=(len(y), 8)), {"partition": "train"})
    dataset.add_targets(y)
    return dataset


def _targets() -> np.ndarray:
    rng = np.random.default_rng(4)
    return rng.normal(size=60)


def _fold_trains(y: np.ndarray) -> list[list[int]]:
    dataset = _dataset(y)
    steps, pool, excluded = _resolve_exclude([{"exclude": YOutlierFilter(threshold=0.5), "keep_in_oof": True}], dataset)
    assert steps == []
    assert isinstance(excluded, FoldLocalExclusion)
    return [train for train, _ in _build_folds(KFold(3), dataset, pool, excluded)]


def test_fold_train_ignores_validation_targets():
    y = _targets()
    shifted = y.copy()
    shifted[:20] += 100.0  # KFold(3) without shuffle: rows 0-19 are fold 0's validation

    assert _fold_trains(y)[0] == _fold_trains(shifted)[0]


def test_fold_train_drops_what_its_own_fit_flags():
    y = _targets()
    dataset = _dataset(y)
    _, pool, excluded = _resolve_exclude([{"exclude": YOutlierFilter(threshold=0.5), "keep_in_oof": True}], dataset)
    folds = _build_folds(KFold(3), dataset, pool, excluded)

    for (train, validation), (raw_train, raw_validation) in zip(folds, KFold(3).split(pool), strict=True):
        assert validation == [pool[i] for i in raw_validation]
        raw = [pool[i] for i in raw_train]
        x = np.asarray(dataset.x({"sample": raw}, layout="2d"))
        keep = YOutlierFilter(threshold=0.5).fit(x, y[raw]).get_mask(x, y[raw])
        assert train == [sample for sample, kept in zip(raw, keep, strict=True) if kept]


def test_full_train_fit_marks_the_refit_exclusion():
    y = _targets()
    dataset = _dataset(y)
    _, pool, excluded = _resolve_exclude([{"exclude": YOutlierFilter(threshold=0.5), "keep_in_oof": True}], dataset)
    x = np.asarray(dataset.x({"partition": "train"}, layout="2d"))
    keep = YOutlierFilter(threshold=0.5).fit(x, y).get_mask(x, y)

    assert pool == list(range(60))
    assert set(excluded) == {sample for sample in range(60) if not keep[sample]}


def test_empty_full_train_exclusion_still_applies_per_fold():
    exclusion = FoldLocalExclusion(set(), [], True, {})

    assert not len(exclusion)
    assert exclusion


def test_fold_set_declares_fold_local_train_lists():
    identity = mint_identity(_dataset(_targets()))
    folds = [([0, 1], [2]), ([1, 2], [0]), ([0, 2], [1])]

    assert build_fold_set(identity, FoldLocalFolds(folds))["train_exclusion"] == "fold_local"
    assert "train_exclusion" not in build_fold_set(identity, folds)


def test_default_mode_removes_excluded_from_the_cv_universe():
    y = _targets()
    _, pool, excluded = _resolve_exclude([{"exclude": YOutlierFilter(threshold=0.5)}], _dataset(y))

    assert excluded == set()
    assert len(pool) < 60


class _Refuses(SampleFilter):
    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        raise ValueError("insufficient data")

    def get_mask(self, X, y=None):  # noqa: ANN001, ANN201
        return np.ones(len(X), dtype=bool)


@pytest.mark.parametrize("keyword", ["exclude", "tag"])
def test_filter_errors_fail_the_step(keyword):
    dataset = _dataset(_targets())
    step = {keyword: _Refuses()}

    with pytest.raises(ValueError, match="_Refuses could not be applied: insufficient data"):
        if keyword == "exclude":
            _resolve_exclude([step], dataset)
        else:
            _resolve_tags([step], dataset, list(range(60)))


class _TargetShape(SampleFilter):
    shapes: list[tuple[int, ...]] = []

    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        type(self).shapes.append(np.shape(y))
        return self

    def get_mask(self, X, y=None):  # noqa: ANN001, ANN201
        return np.arange(len(X)) % 2 == 0


def test_multi_target_y_reaches_filters_unflattened():
    rng = np.random.default_rng(4)
    dataset = SpectroDataset("multi-target")
    dataset.add_samples(rng.normal(size=(30, 8)), {"partition": "train"})
    dataset.add_targets(rng.normal(size=(30, 2)))
    _TargetShape.shapes = []

    _, pool, _ = _resolve_exclude([{"exclude": _TargetShape()}], dataset)
    _, tags = _resolve_tags([{"tag": _TargetShape()}], dataset, list(range(30)))

    assert _TargetShape.shapes == [(30, 2), (30, 2)]
    assert len(pool) == 15
    assert tags is not None and len(tags) == 15
