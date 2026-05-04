"""Phase H3 tests: split-aware inner CV detection and Y-blocked K-Fold."""

from __future__ import annotations

import numpy as np
import pytest
from aomridge.selection import (
    _trimmed_rmse_from_residuals,
    cv_score_alphas,
)
from aomridge.split_aware_cv import (
    YBlockedKFold,
    detect_split_kind,
    make_inner_cv,
)
from sklearn.model_selection import GroupKFold

# ----------------------------------------------------------------------
# Detection heuristics
# ----------------------------------------------------------------------


def test_detect_y_based():
    assert detect_split_kind("Rice_Amylose_313_YbasedSplit") == "y_based"
    assert detect_split_kind("Beer_OriginalExtract_60_YbaseSplit") == "y_based"
    assert (
        detect_split_kind("Corn_Oil_80_ZhengChenPelegYbaseSplit") == "y_based"
    )


def test_detect_grouped():
    assert (
        detect_split_kind("An_spxyG70_30_byCultivar_ASD") == "grouped"
    )
    assert (
        detect_split_kind("brix_groupSampleID_stratDateVar_balRows") == "grouped"
    )
    assert (
        detect_split_kind("Field_byManure_type_v1") == "grouped"
    )


def test_detect_spxy_fallback():
    assert detect_split_kind("Random_Foo_Bar") == "spxy"
    assert detect_split_kind("ALPINE_P_291_KS") == "spxy"
    assert detect_split_kind("TIC_spxy70") == "spxy"
    assert detect_split_kind("") == "spxy"


def test_detect_grouped_via_metadata():
    """Even without a name signal, providing groups forces 'grouped'."""
    groups = np.arange(10) % 3
    assert detect_split_kind("anon", groups=groups) == "grouped"


# ----------------------------------------------------------------------
# YBlockedKFold splitter
# ----------------------------------------------------------------------


def test_y_blocked_kfold_no_overlap():
    """train/val indices must be disjoint and union to all rows exactly once."""
    rng = np.random.default_rng(0)
    n = 30
    y = rng.normal(size=n)
    cv = YBlockedKFold(n_splits=3, random_state=0)
    seen_val = np.zeros(n, dtype=int)
    for train_idx, val_idx in cv.split(np.zeros((n, 4)), y):
        assert np.intersect1d(train_idx, val_idx).size == 0
        seen_val[val_idx] += 1
        assert train_idx.size + val_idx.size == n
    assert np.all(seen_val == 1)


def test_y_blocked_kfold_extrapolates():
    """fold 0 holds the lowest-y stratum; fold n-1 holds the highest."""
    n = 30
    y = np.linspace(0.0, 10.0, n)
    cv = YBlockedKFold(n_splits=3, random_state=0)
    folds = list(cv.split(np.zeros((n, 1)), y))
    val_low = y[folds[0][1]]
    val_high = y[folds[-1][1]]
    assert val_low.max() < val_high.min(), (
        f"fold 0 max ({val_low.max():.3f}) should be < fold 2 min "
        f"({val_high.min():.3f})"
    )


def test_y_blocked_kfold_balanced_sizes():
    n = 31  # not divisible by 3
    y = np.linspace(0.0, 1.0, n)
    cv = YBlockedKFold(n_splits=3, random_state=0)
    sizes = [val_idx.size for _, val_idx in cv.split(np.zeros((n, 1)), y)]
    # Bins are equal-rank, so fold sizes must each fall within ±1 of n/k
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == n


def test_y_blocked_kfold_rejects_bad_inputs():
    with pytest.raises(ValueError):
        YBlockedKFold(n_splits=1)
    cv = YBlockedKFold(n_splits=3)
    with pytest.raises(ValueError):
        list(cv.split(np.zeros((10, 1))))   # missing y
    with pytest.raises(ValueError):
        list(cv.split(np.zeros((2, 1)), np.array([1.0, 2.0])))  # n < n_splits


# ----------------------------------------------------------------------
# make_inner_cv factory
# ----------------------------------------------------------------------


def test_make_inner_cv_y_based_returns_yblocked():
    cv = make_inner_cv("y_based", n_splits=3, random_state=7)
    assert isinstance(cv, YBlockedKFold)
    assert cv.n_splits == 3


def test_make_inner_cv_grouped_with_groups_returns_groupkfold():
    groups = np.arange(20) % 4
    cv = make_inner_cv("grouped", n_splits=4, groups=groups)
    # Adapter wraps GroupKFold so it can be called as a plain splitter
    n_seen = 0
    for tr, va in cv.split(np.zeros((20, 1)), np.arange(20)):
        assert np.intersect1d(tr, va).size == 0
        # No group should straddle train/val
        assert set(groups[tr]).isdisjoint(set(groups[va]))
        n_seen += 1
    assert n_seen == 4
    assert cv.get_n_splits() == 4


def test_make_inner_cv_grouped_without_groups_falls_back_to_spxy():
    # Without groups, must fall through to SPXY (or raise if SPXY is missing).
    pytest.importorskip("nirs4all.operators.splitters")
    from nirs4all.operators.splitters import SPXYFold

    cv = make_inner_cv("grouped", n_splits=3, random_state=0)
    assert isinstance(cv, SPXYFold)


def test_make_inner_cv_spxy_returns_spxy():
    pytest.importorskip("nirs4all.operators.splitters")
    from nirs4all.operators.splitters import SPXYFold

    cv = make_inner_cv("spxy", n_splits=3, random_state=0)
    assert isinstance(cv, SPXYFold)


def test_make_inner_cv_unknown_raises():
    with pytest.raises(ValueError):
        make_inner_cv("bogus", n_splits=3)


def test_groupkfold_adapter_validates_lengths():
    groups = np.arange(20) % 4
    cv = make_inner_cv("grouped", n_splits=4, groups=groups)
    with pytest.raises(ValueError):
        list(cv.split(np.zeros((10, 1)), np.arange(10)))


# ----------------------------------------------------------------------
# Pooled trimmed RMSE
# ----------------------------------------------------------------------


def test_pooled_trimmed_helper_basic():
    """Trimming reduces magnitude when extreme outliers exist."""
    rng = np.random.default_rng(42)
    base = rng.normal(scale=0.1, size=200)
    outliers = np.concatenate([base, np.array([100.0, -100.0])])
    rmse_no_trim = _trimmed_rmse_from_residuals(outliers, trim=0.0)
    rmse_trim = _trimmed_rmse_from_residuals(outliers, trim=0.05)
    assert rmse_trim < rmse_no_trim
    # Untrimmed RMSE is dominated by the two outliers
    assert rmse_no_trim > 5.0
    assert rmse_trim < 1.0


def test_pooled_trimmed_helper_invalid():
    with pytest.raises(ValueError):
        _trimmed_rmse_from_residuals(np.array([1.0, 2.0]), trim=0.6)
    with pytest.raises(ValueError):
        _trimmed_rmse_from_residuals(np.array([]), trim=0.0)


def test_pooled_trimmed_robust():
    """Synthetic CV with one fold outlier: trimmed RMSE should be smaller
    than (and finite alongside) the mean RMSE."""
    pytest.importorskip("aompls.operators")
    from aompls.operators import IdentityOperator

    rng = np.random.default_rng(2026)
    n, p = 60, 8
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=p)
    y = X @ coef + 0.05 * rng.normal(size=n)
    # Inject a small group of extreme y outliers; with 3 folds only one fold
    # hosts most of them, biasing per-fold-mean RMSE.
    y[:6] += 50.0

    class FixedFolds:
        def split(self, X, Y=None):
            yield np.arange(20, n), np.arange(0, 20)
            yield np.concatenate([np.arange(20), np.arange(40, n)]), np.arange(20, 40)
            yield np.arange(0, 40), np.arange(40, n)

    cv = FixedFolds()
    alphas = np.array([0.1, 1.0, 10.0])
    bank = [IdentityOperator()]
    rmse_mean = cv_score_alphas(
        X, y.reshape(-1, 1), bank, alphas, cv,
        block_scaling="none", center=True, scoring="rmse",
    )
    rmse_trim = cv_score_alphas(
        X, y.reshape(-1, 1), bank, alphas, cv,
        block_scaling="none", center=True,
        scoring="rmse_pooled_trimmed", trim=0.1,
    )
    # The fold containing the injected outliers dominates per-fold mean,
    # but trimming the worst residuals reduces the score below the mean.
    assert np.all(np.isfinite(rmse_mean))
    assert np.all(np.isfinite(rmse_trim))
    assert rmse_trim.min() < rmse_mean.min()


# ----------------------------------------------------------------------
# Leakage safety on YBlockedKFold (anti-leakage parity with SPXYFold tests)
# ----------------------------------------------------------------------


def test_y_blocked_kfold_does_not_observe_validation_in_fit():
    """Replicates the fold-locality property: each fold's train indices
    contain no validation rows. Combined with the disjointness test above,
    this proves the splitter is leakage-safe under the same contract as
    SPXYFold."""
    rng = np.random.default_rng(0)
    n = 24
    y = rng.normal(size=n)
    cv = YBlockedKFold(n_splits=3, random_state=0)
    for train_idx, val_idx in cv.split(np.zeros((n, 1)), y):
        assert set(train_idx.tolist()).isdisjoint(val_idx.tolist())
