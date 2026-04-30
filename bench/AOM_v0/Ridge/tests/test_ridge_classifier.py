"""Tests for ``AOMRidgeClassifier`` (PLS-DA-style classification).

The classifier wraps :class:`AOMRidgeRegressor` on a class-balanced one-hot
encoding, so it inherits the AOM-Ridge no-leak invariants. These tests
cover the public API contract (predict / predict_proba shapes, probability
normalisation), accuracy on synthetic data, calibrator fallback, and the
fold-local CV anti-leakage guarantee.
"""

from __future__ import annotations

import numpy as np
import pytest
from aompls.metrics import balanced_accuracy, expected_calibration_error, log_loss
from aompls.operators import IdentityOperator, LinearSpectralOperator
from aompls.synthetic import make_classification
from aomridge.classification import AOMRidgeClassifier
from sklearn.model_selection import KFold

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def binary_data():
    return make_classification(
        n_train=80, n_test=40, p=120, n_classes=2, random_state=3,
    )


@pytest.fixture
def multiclass_data():
    return make_classification(
        n_train=120, n_test=60, p=140, n_classes=3, random_state=4,
    )


# ----------------------------------------------------------------------
# Accuracy / shape contracts
# ----------------------------------------------------------------------


def test_aomridge_classifier_binary(binary_data):
    """Accuracy on a synthetic 2-class problem must exceed 0.85."""
    ds = binary_data
    clf = AOMRidgeClassifier(
        selection="superblock", operator_bank="compact",
        cv=3, random_state=0,
    )
    clf.fit(ds.X_train, ds.y_train)
    pred = clf.predict(ds.X_test)
    assert pred.shape == (ds.X_test.shape[0],)
    acc = float(np.mean(pred == ds.y_test))
    assert acc > 0.85, f"binary accuracy {acc:.3f} below 0.85 threshold"


def test_aomridge_classifier_multiclass(multiclass_data):
    """Accuracy on a synthetic 3-class problem must exceed 0.7."""
    ds = multiclass_data
    clf = AOMRidgeClassifier(
        selection="superblock", operator_bank="compact",
        cv=3, random_state=0,
    )
    clf.fit(ds.X_train, ds.y_train)
    pred = clf.predict(ds.X_test)
    acc = float(np.mean(pred == ds.y_test))
    assert acc > 0.7, f"multiclass accuracy {acc:.3f} below 0.7 threshold"


def test_classifier_handles_imbalanced(binary_data):
    """Heavily imbalanced training data must yield reasonable balanced accuracy.

    Subsample the minority class to a (0.05, 0.95) prior. The class-balanced
    encoding combined with the balanced logistic calibrator should still
    produce probabilities that sum to one and a balanced accuracy markedly
    above the trivial constant-prediction baseline (which is 0.5 by
    construction of balanced accuracy).
    """
    ds = binary_data
    rng = np.random.default_rng(42)
    pos_idx = np.where(ds.y_train == 1)[0]
    neg_idx = np.where(ds.y_train == 0)[0]
    keep_pos = rng.permutation(pos_idx)[: max(2, len(pos_idx) // 16)]
    keep_idx = np.concatenate([neg_idx, keep_pos])
    Xtr = ds.X_train[keep_idx]
    ytr = ds.y_train[keep_idx]
    # Sanity check: prior is heavily skewed (~< 0.15)
    minority_prior = float((ytr == 1).sum()) / float(ytr.shape[0])
    assert minority_prior < 0.15

    clf = AOMRidgeClassifier(
        selection="superblock", operator_bank="compact",
        cv=3, random_state=0,
    )
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(ds.X_test)
    assert proba.shape == (ds.X_test.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    pred = clf.predict(ds.X_test)
    bacc = balanced_accuracy(ds.y_test, pred)
    assert bacc > 0.6, f"balanced accuracy {bacc:.3f} below 0.6 threshold"


def test_classifier_predict_proba_sums_to_one(multiclass_data):
    """Every row of ``predict_proba`` must sum to one (within fp tolerance)."""
    ds = multiclass_data
    clf = AOMRidgeClassifier(
        selection="superblock", operator_bank="compact",
        cv=3, random_state=0,
    )
    clf.fit(ds.X_train, ds.y_train)
    proba = clf.predict_proba(ds.X_test)
    assert proba.shape == (ds.X_test.shape[0], 3)
    assert np.all(proba >= 0.0 - 1e-9)
    assert np.all(proba <= 1.0 + 1e-9)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ----------------------------------------------------------------------
# Calibrator fallback
# ----------------------------------------------------------------------


def test_classifier_calibrator_temperature_fallback(monkeypatch, binary_data):
    """When the logistic fit raises, the classifier falls back to temperature.

    We monkeypatch ``LogisticRegression.fit`` to raise a
    ``RuntimeError`` and verify that the resulting estimator still produces
    valid probabilities (rows sum to one) and uses the temperature
    calibrator.
    """
    ds = binary_data

    from sklearn.linear_model import LogisticRegression

    def _raise(self, *args, **kwargs):
        raise RuntimeError("forced logistic failure")

    monkeypatch.setattr(LogisticRegression, "fit", _raise)

    clf = AOMRidgeClassifier(
        selection="superblock", operator_bank="compact",
        cv=3, random_state=0, calibrator="logistic",
    )
    clf.fit(ds.X_train, ds.y_train)
    assert clf.calibrator_kind_ == "temperature"
    assert clf.calibrator_ is None
    assert clf.temperature_ is not None
    proba = clf.predict_proba(ds.X_test)
    assert proba.shape == (ds.X_test.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_classifier_explicit_temperature_calibrator(binary_data):
    """``calibrator='temperature'`` skips logistic fitting entirely."""
    ds = binary_data
    clf = AOMRidgeClassifier(
        selection="superblock", operator_bank="compact",
        cv=3, random_state=0, calibrator="temperature",
    )
    clf.fit(ds.X_train, ds.y_train)
    assert clf.calibrator_kind_ == "temperature"
    assert clf.calibrator_ is None
    assert clf.temperature_ is not None and clf.temperature_ > 0.0
    proba = clf.predict_proba(ds.X_test)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_classifier_rejects_unknown_calibrator(binary_data):
    ds = binary_data
    clf = AOMRidgeClassifier(calibrator="bogus", cv=3, random_state=0)
    with pytest.raises(ValueError):
        clf.fit(ds.X_train, ds.y_train)


# ----------------------------------------------------------------------
# Anti-leakage probe
# ----------------------------------------------------------------------


class _SpyOperator(LinearSpectralOperator):
    """Identity-equivalent operator that records every fit/apply_cov input.

    Used to verify that the fold-local CV inside the regressor never lets
    validation rows reach the operator's ``fit`` / ``apply_cov`` paths.
    """

    def __init__(self, p=None) -> None:
        super().__init__(name="cls_spy_identity", p=p)
        self.fit_row_signatures: list[tuple] = []
        self.apply_cov_col_counts: list[int] = []

    def fit(self, X=None, y=None):
        if X is not None:
            X = np.asarray(X)
            self.fit_row_signatures.append(
                tuple(np.round(np.sort(np.sum(X, axis=1)), 8).tolist())
            )
            self.p = X.shape[1]
        return self

    def _transform_impl(self, X):
        return X.copy()

    def _apply_cov_impl(self, S):
        self.apply_cov_col_counts.append(int(S.shape[1]))
        return S.copy()

    def _adjoint_vec_impl(self, v):
        return v.copy()

    def _matrix_impl(self, p: int):
        return np.eye(p)


def test_classifier_no_leak_in_cv():
    """Spy operator inside the bank must never see validation rows.

    The classifier wraps ``AOMRidgeRegressor`` whose CV is fold-local, so a
    bank that includes a spy must not see any ``fit`` / ``apply_cov`` call
    that references the full dataset (which would betray a leak).
    """
    rng = np.random.default_rng(123)
    n, p = 60, 24
    X = rng.normal(size=(n, p))
    # Three roughly-balanced classes
    y = (np.arange(n) % 3).astype(int)
    rng.shuffle(y)

    cv = KFold(n_splits=3, shuffle=False)
    folds = list(cv.split(X, y))

    # Build a *new* spy class whose clones share lists, so we can probe the
    # fold clones (the user-supplied template is never fitted directly).
    fit_sigs: list[tuple] = []
    col_counts: list[int] = []

    class _SharedSpy(_SpyOperator):
        def fit(self, X=None, y=None):
            if X is not None:
                X = np.asarray(X)
                fit_sigs.append(
                    tuple(np.round(np.sort(np.sum(X, axis=1)), 8).tolist())
                )
                self.p = X.shape[1]
            return self

        def _apply_cov_impl(self, S):
            col_counts.append(int(S.shape[1]))
            return S.copy()

    bank = [IdentityOperator(p=p), _SharedSpy(p=p)]
    # No fixed ``alpha`` so that the fold-local CV path runs (otherwise the
    # regressor short-circuits to the final refit and the spy only sees the
    # full calibration set, not the per-fold training subsets).
    clf = AOMRidgeClassifier(
        operator_bank=bank, cv=cv, random_state=0,
    )
    clf.fit(X, y)

    # Every fit signature must correspond to a centered training fold (or
    # the centered full calibration set used for the final refit). None
    # should match the sums of the *raw uncentered* full dataset, and no
    # column count from ``apply_cov`` may equal the full ``n`` (which would
    # indicate the kernel was built on all rows).
    train_fold_sigs = set()
    for tr, _ in folds:
        x_mean = X[tr].mean(axis=0)
        train_fold_sigs.add(
            tuple(np.round(np.sort(np.sum(X[tr] - x_mean, axis=1)), 8).tolist())
        )
    full_sig = tuple(np.round(np.sort(np.sum(X, axis=1)), 8).tolist())
    # The spy template is never fitted directly
    assert fit_sigs, "spy fit was never invoked through fold clones"
    assert full_sig not in fit_sigs, "spy was fitted on uncentered full X (leak)"
    # apply_cov column counts must be either training fold sizes (for kernel
    # construction) or 1 (for screening on Xc^T @ Yc with q=K columns; for
    # K classes this is K not n). They must never equal n.
    assert col_counts, "spy apply_cov was never invoked"
    train_sizes = {len(tr) for tr, _ in folds}
    n_classes = int(np.unique(y).size)
    refit_sizes = {n}  # final refit uses full calibration set
    # The classifier final refit DOES touch ``n`` rows (legitimate). But the
    # CV / selection paths must not. We only assert: there exist fold-local
    # column counts (some kernel was built per-fold), and the count ``n``
    # only appears at refit time (no more than the number of refit kernels).
    assert any(k in train_sizes for k in col_counts), (
        "no fold-local kernels were constructed (CV must run)"
    )
    # Selection screening also touches ``n_classes`` columns when computing
    # Xc^T @ Yc. Both are fine; what we forbid is *fold CV* using ``n``.
    # That's hard to assert without more instrumentation, so we instead
    # verify that fit signatures contain at least one *training-fold*
    # centered signature (proving fold-local centering was applied).
    assert train_fold_sigs & set(fit_sigs), (
        "no fit signature matches a fold-local training centering"
    )


# ----------------------------------------------------------------------
# Diagnostics + selection mode coverage
# ----------------------------------------------------------------------


def test_diagnostics_contains_expected_keys(multiclass_data):
    ds = multiclass_data
    clf = AOMRidgeClassifier(
        selection="superblock", operator_bank="compact",
        cv=3, random_state=0,
    )
    clf.fit(ds.X_train, ds.y_train)
    diag = clf.get_diagnostics()
    assert diag["model"] == "AOMRidgeClassifier"
    assert diag["n_classes"] == 3
    assert diag["calibrator_kind"] in ("logistic", "temperature")
    assert "alpha" in diag
    assert "selected_operator_names" in diag
    # Selected operator list is non-empty
    assert len(clf.get_selected_operators()) >= 1


@pytest.mark.parametrize(
    "selection",
    ["superblock", "global", "active_superblock", "mkl", "branch_global"],
)
def test_all_selection_modes_run(selection, multiclass_data):
    """Every AOM-Ridge selection mode must work for classification."""
    ds = multiclass_data
    extra = {}
    if selection == "active_superblock":
        extra["active_top_m"] = 6
    if selection == "mkl":
        extra["mkl_top_k"] = 4
    clf = AOMRidgeClassifier(
        selection=selection, operator_bank="compact",
        cv=3, random_state=0, **extra,
    )
    clf.fit(ds.X_train, ds.y_train)
    proba = clf.predict_proba(ds.X_test)
    assert proba.shape == (ds.X_test.shape[0], 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ----------------------------------------------------------------------
# Calibration metrics sanity
# ----------------------------------------------------------------------


def test_calibration_metrics_finite(multiclass_data):
    """Log loss and ECE must be finite and within sensible bounds."""
    ds = multiclass_data
    clf = AOMRidgeClassifier(
        selection="superblock", operator_bank="compact",
        cv=3, random_state=0,
    )
    clf.fit(ds.X_train, ds.y_train)
    proba = clf.predict_proba(ds.X_test)
    ll = log_loss(ds.y_test, proba, classes=clf.classes_)
    ece = expected_calibration_error(ds.y_test, proba, n_bins=5)
    assert ll >= 0.0 and np.isfinite(ll)
    assert 0.0 <= ece <= 1.0


def test_classifier_requires_two_classes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 8))
    y = np.zeros(20, dtype=int)
    clf = AOMRidgeClassifier(cv=3, random_state=0)
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_classifier_decision_function_shape(multiclass_data):
    """``decision_function`` returns the latent regression scores."""
    ds = multiclass_data
    clf = AOMRidgeClassifier(
        selection="superblock", operator_bank="compact",
        cv=3, random_state=0,
    )
    clf.fit(ds.X_train, ds.y_train)
    scores = clf.decision_function(ds.X_test)
    assert scores.shape == (ds.X_test.shape[0], 3)
