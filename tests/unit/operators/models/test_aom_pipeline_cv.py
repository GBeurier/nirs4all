import numpy as np

from nirs4all.controllers.models.pipeline_cv import PrecomputedFoldSplitter
from nirs4all.operators.models._aom_nirs.ridge.auto_selector import _dispatch_candidate, _subset_inner_cv


def test_auto_selector_dispatch_passes_splitter_to_aom_pls_candidate():
    splitter = PrecomputedFoldSplitter.from_folds(
        [([0, 1, 2], [3]), ([0, 2, 3], [1])],
        n_samples=4,
    )
    spec = {
        "selection": "aom_pls",
        "operator_bank": "compact",
        "extra": {"cv": 3, "max_components": 4},
    }

    estimator, _ = _dispatch_candidate(spec, seed=0, inner_cv=splitter)

    assert estimator.cv_splitter is splitter
    assert estimator.cv == splitter.get_n_splits()


def test_subset_inner_cv_remaps_precomputed_splitter_for_candidate_train_rows():
    splitter = PrecomputedFoldSplitter.from_folds(
        [
            ([0, 1, 2, 3], [4, 5]),
            ([0, 1, 4, 5], [2, 3]),
            ([2, 3, 4, 5], [0, 1]),
        ],
        n_samples=6,
    )

    subset = _subset_inner_cv(splitter, np.array([0, 1, 2, 3]))
    folds = list(subset.split(np.zeros((4, 2))))

    assert subset.get_n_splits() == 2
    np.testing.assert_array_equal(folds[0][1], [2, 3])
    np.testing.assert_array_equal(folds[1][1], [0, 1])


def test_subset_inner_cv_allows_single_nested_split():
    splitter = PrecomputedFoldSplitter.from_folds(
        [
            ([0, 1, 4, 5], [2, 3]),
            ([0, 1, 2, 3], [4, 5]),
        ],
        n_samples=6,
    )

    subset = _subset_inner_cv(splitter, np.array([0, 1, 2, 3]))
    folds = list(subset.split(np.zeros((4, 2))))

    assert subset.get_n_splits() == 1
    np.testing.assert_array_equal(folds[0][1], [2, 3])
