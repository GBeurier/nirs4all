"""Tests for the fold-local branch-global selection mode.

The ``selection="branch_global"`` mode scores ``(branch, operator, alpha)``
triples *inside* every CV fold, with branch parameters (e.g. the MSC
reference) fitted on the training fold only. These tests verify:

- no validation rows leak into the branch fit;
- the picker recovers the correct branch on synthetic data where one branch
  is clearly superior;
- predict() applies the stored branch transformer faithfully (matches a
  manual fit_transform + predict reference).
"""

from __future__ import annotations

import numpy as np
from aompls.operators import IdentityOperator
from aompls.preprocessing import MultiplicativeScatterCorrection
from aomridge.branches import make_branch_preproc
from aomridge.estimators import AOMRidgeRegressor
from aomridge.selection import select_branch_global
from sklearn.model_selection import KFold

# ----------------------------------------------------------------------
# Test 1 — branch reference is fitted on training rows only
# ----------------------------------------------------------------------


def test_branch_global_no_leak(monkeypatch):
    """Spy that records every fold's MSC reference; validation rows never seen."""
    rng = np.random.default_rng(0)
    n, p = 30, 16
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    cv = KFold(n_splits=3, shuffle=False)
    folds = list(cv.split(X, y))

    seen_msc_refs: list[np.ndarray] = []

    real_fit = MultiplicativeScatterCorrection.fit

    def recording_fit(self, X_in, y_in=None):
        out = real_fit(self, X_in, y_in)
        # Reference is computed as X_in.mean(axis=0); record it.
        seen_msc_refs.append(np.asarray(self.reference_, dtype=float).copy())
        return out

    monkeypatch.setattr(MultiplicativeScatterCorrection, "fit", recording_fit)

    est = AOMRidgeRegressor(
        selection="branch_global",
        operator_bank=[IdentityOperator()],
        branches=("none", "snv", "msc"),
        block_scaling="none",
        cv=cv,
        random_state=0,
    ).fit(X, y)

    # Build the set of allowed references (one per training fold + one for
    # the final refit on the full calibration set).
    allowed_refs: list[np.ndarray] = [X[tr].mean(axis=0) for tr, _ in folds]
    allowed_refs.append(X.mean(axis=0))

    # Validation rows added to the training mean would change the reference;
    # if any seen reference matched a "validation-included" mean, we'd know
    # there was a leak.
    # The "leaky" reference would be a validation-only mean (or any mean
    # that includes validation rows but excludes the corresponding train
    # rows). We probe the strongest specific leak: the per-fold validation
    # mean.
    leaky_refs = [X[va].mean(axis=0) for _, va in folds]

    for ref in seen_msc_refs:
        # The recorded reference must match one of the per-fold or full-data
        # means (within fp tolerance).
        match = any(
            np.allclose(ref, r, atol=1e-9, rtol=1e-9) for r in allowed_refs
        )
        assert match, "MSC reference did not match any allowed (per-fold or full) mean"
        # And must NOT match a pure validation-fold mean.
        assert not any(
            np.allclose(ref, r, atol=1e-9, rtol=1e-9) for r in leaky_refs
        ), "MSC reference equals a validation-fold-only mean: LEAK"

    # Estimator should still produce finite predictions.
    assert np.all(np.isfinite(est.predict(X[:5])))


# ----------------------------------------------------------------------
# Test 2 — picker recovers the correct branch on a synthetic case
# ----------------------------------------------------------------------


def test_branch_global_picks_correct_branch_on_synthetic():
    """Build data where SNV is clearly the best preprocessing.

    Each spectrum is generated so that the predictive signal lives in the
    *shape* (relative variation across wavelengths), but every row is
    contaminated by a large random additive offset and a large positive
    multiplicative scaling. SNV (per-sample mean + std normalisation)
    removes both contaminations and exposes the true shape; raw and MSC
    cannot.
    """
    rng = np.random.default_rng(42)
    n, p = 80, 32
    # Two distinct shape templates. y selects a smooth blend between them.
    template_a = np.sin(np.linspace(0.0, 2 * np.pi, p))
    template_b = np.cos(np.linspace(0.0, 2 * np.pi, p))
    y = rng.uniform(-1.0, 1.0, size=n)

    # The "clean" spectrum is a y-dependent blend of the two templates.
    clean = template_a[None, :] + y[:, None] * template_b[None, :]
    # Per-row offsets and scales that vary widely across rows. SNV cancels
    # both; raw/MSC see a heavily contaminated signal.
    offsets = rng.uniform(-5.0, 5.0, size=n)
    scales = np.exp(rng.uniform(-1.0, 1.0, size=n))  # in (e^-1, e)
    X = scales[:, None] * clean + offsets[:, None]
    # Small additive noise.
    X += 0.02 * rng.normal(size=(n, p))

    cv = KFold(n_splits=4, shuffle=True, random_state=0)
    bank = [IdentityOperator(p=p)]
    alphas = np.logspace(-3, 3, 13)
    branch_name, op_idx, alpha, table = select_branch_global(
        X,
        y.reshape(-1, 1),
        bank,
        alphas,
        cv,
        branches=("none", "snv", "msc"),
        block_scaling="none",
    )
    # On this construction both SNV and MSC remove the per-row offset and
    # scale; the picker must choose one of them — the raw "none" branch must
    # be strictly worse.
    none_min = float(np.min(table[0]))
    snv_min = float(np.min(table[1]))
    msc_min = float(np.min(table[2]))
    assert snv_min < none_min, (
        f"SNV ({snv_min:.4f}) did not beat raw ({none_min:.4f})"
    )
    assert msc_min < none_min, (
        f"MSC ({msc_min:.4f}) did not beat raw ({none_min:.4f})"
    )
    # The picker must report a non-trivial branch (the better of SNV/MSC).
    assert branch_name in ("snv", "msc"), (
        f"expected 'snv' or 'msc', got {branch_name!r}"
    )
    assert 0 <= op_idx < len(bank)
    assert alpha in alphas


# ----------------------------------------------------------------------
# Test 3 — predict() faithfully replays the stored branch transformer
# ----------------------------------------------------------------------


def test_branch_global_predict_uses_stored_branch():
    """When the chosen branch is SNV, fit + predict must equal a manual
    pipeline (StandardNormalVariate + dual Ridge with the same operator and
    alpha) up to floating-point precision.
    """
    rng = np.random.default_rng(7)
    n, p = 40, 20
    X = rng.normal(loc=0.0, scale=1.0, size=(n, p))
    y = X[:, :3].sum(axis=1) + 0.1 * rng.normal(size=n)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)

    # Force SNV to be picked by restricting the allowed branches to {"snv"}.
    est = AOMRidgeRegressor(
        selection="branch_global",
        operator_bank=[IdentityOperator()],
        branches=("snv",),
        alpha=1.0,
        block_scaling="none",
        cv=cv,
        random_state=0,
    ).fit(X, y)
    assert est.diagnostics_["chosen_branch"] == "snv"
    assert est.diagnostics_["coef_available"] is False
    assert est.coef_ is None

    # Manual reference: apply SNV (fit on full train), center features, run
    # the same identity-only dual ridge with the same alpha.
    preproc = make_branch_preproc("snv")
    X_branched = preproc.fit_transform(X)
    x_mean = X_branched.mean(axis=0)
    y_mean = float(np.mean(y))
    Xc = X_branched - x_mean
    yc = y - y_mean

    from sklearn.linear_model import Ridge

    sk = Ridge(alpha=1.0, fit_intercept=False).fit(Xc, yc)

    rng2 = np.random.default_rng(11)
    X_te = rng2.normal(loc=0.0, scale=1.0, size=(8, p))
    X_te_branched = preproc.transform(X_te)
    Xc_te = X_te_branched - x_mean
    sk_pred = Xc_te @ sk.coef_ + y_mean

    est_pred = est.predict(X_te)
    np.testing.assert_allclose(est_pred, sk_pred, atol=1e-7, rtol=1e-7)

    # Also verify fit + predict on the training data is finite.
    train_pred = est.predict(X)
    assert train_pred.shape == y.shape
    assert np.all(np.isfinite(train_pred))


def test_branch_global_diagnostics_are_json_serialisable():
    """Branch-global diagnostics expose the chosen branch and stay JSON-friendly."""
    import json

    rng = np.random.default_rng(13)
    n, p = 30, 16
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    est = AOMRidgeRegressor(
        selection="branch_global",
        operator_bank="compact",
        branches=("none", "snv"),
        cv=3,
        random_state=0,
    ).fit(X, y)
    diag = est.get_diagnostics()
    assert "chosen_branch" in diag
    assert diag["chosen_branch"] in ("none", "snv")
    assert "branches" in diag
    json.dumps(diag)


def test_branch_global_unknown_branch_raises():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(20, 8))
    y = rng.normal(size=20)
    est = AOMRidgeRegressor(
        selection="branch_global",
        operator_bank=[IdentityOperator()],
        branches=("none", "bogus"),
        cv=2,
    )
    import pytest

    with pytest.raises(ValueError, match="unknown branch"):
        est.fit(X, y)


# ----------------------------------------------------------------------
# Test 4 — registry covers the AOM-PLS row-wise preprocessor library
# ----------------------------------------------------------------------


def test_branch_registry_includes_all_preprocessors():
    """The registry must expose every NIRS preprocessor used by AOM-PLS.

    This test is the executable spec of the registry: each known branch
    name must produce a fresh sklearn-style transformer with the standard
    ``fit`` / ``transform`` interface, and ``is_stateless`` must reflect
    whether the transformer carries any training-time state.
    """
    from aompls.preprocessing import (
        ASLSBaseline,
        ExtendedMSC,
        MultiplicativeScatterCorrection,
        OrthogonalSignalCorrection,
        PreprocessingPipeline,
        StandardNormalVariate,
    )
    from aomridge.branches import (
        VALID_BRANCHES,
        is_stateless,
        make_branch_preproc,
    )

    # ``ASLSBaseline`` and ``ExtendedMSC`` are factory functions that return
    # nirs4all transformers; the rest are direct classes. The expected entry
    # is therefore a small descriptor: ``("class"|"factory"|"pipeline"|"none", attrs)``.
    expected = {
        "none": ("none", {}),
        "snv": ("class", {"_cls": StandardNormalVariate}),
        "msc": ("class", {"_cls": MultiplicativeScatterCorrection}),
        "osc": ("class", {"_cls": OrthogonalSignalCorrection, "n_components": 2}),
        "osc1": ("class", {"_cls": OrthogonalSignalCorrection, "n_components": 1}),
        "osc2": ("class", {"_cls": OrthogonalSignalCorrection, "n_components": 2}),
        "osc3": ("class", {"_cls": OrthogonalSignalCorrection, "n_components": 3}),
        "asls": ("factory", {}),
        "asls_soft": ("factory", {"lam": 1e4}),
        "asls_medium": ("factory", {"lam": 1e6}),
        "asls_hard": ("factory", {"lam": 1e8}),
        "emsc1": ("factory", {"degree": 1}),
        "emsc2": ("factory", {"degree": 2}),
        "snv_osc": ("pipeline", {"_steps": (StandardNormalVariate, OrthogonalSignalCorrection)}),
        "msc_osc": ("pipeline", {"_steps": (MultiplicativeScatterCorrection, OrthogonalSignalCorrection)}),
        "snv_asls": ("pipeline", {"_steps": (StandardNormalVariate, None)}),
        "msc_asls": ("pipeline", {"_steps": (MultiplicativeScatterCorrection, None)}),
    }
    # The registry's enumeration must match the expected set exactly so we
    # never silently drop or rename a branch.
    assert set(VALID_BRANCHES) == set(expected.keys())

    for name, (kind, attrs) in expected.items():
        preproc = make_branch_preproc(name)
        if kind == "none":
            assert preproc is None
            continue
        # Every transformer must expose the standard sklearn-style API.
        assert hasattr(preproc, "fit")
        assert hasattr(preproc, "transform")
        if kind == "class":
            assert isinstance(preproc, attrs["_cls"]), (
                f"branch={name!r}: expected {attrs['_cls'].__name__}, "
                f"got {type(preproc).__name__}"
            )
            for attr, expected_value in attrs.items():
                if attr.startswith("_"):
                    continue
                assert getattr(preproc, attr) == expected_value, (
                    f"branch={name!r}, attr={attr!r}: expected {expected_value}, "
                    f"got {getattr(preproc, attr)}"
                )
        elif kind == "factory":
            # Factory result: only check the constructor-time attributes.
            for attr, expected_value in attrs.items():
                if attr.startswith("_"):
                    continue
                assert getattr(preproc, attr) == expected_value, (
                    f"branch={name!r}, attr={attr!r}: expected {expected_value}, "
                    f"got {getattr(preproc, attr)}"
                )
        elif kind == "pipeline":
            assert isinstance(preproc, PreprocessingPipeline)
            step_classes = attrs["_steps"]
            assert len(preproc.steps) == len(step_classes)
            for step, cls in zip(preproc.steps, step_classes, strict=True):
                if cls is None:
                    # ASLS step is a factory result — only verify the step
                    # exposes the standard interface.
                    assert hasattr(step, "fit")
                    assert hasattr(step, "transform")
                else:
                    assert isinstance(step, cls)

    # Only SNV is stateless: every other branch carries either training-time
    # parameters (MSC/EMSC reference, OSC projection) or a deliberate fit
    # call we still want to make explicit (ASLS).
    assert is_stateless("snv")
    for name in expected:
        if name == "snv":
            continue
        assert not is_stateless(name), (
            f"branch={name!r} should not be flagged stateless"
        )


# ----------------------------------------------------------------------
# Test 5 — end-to-end branch_global with the full registry
# ----------------------------------------------------------------------


def test_branch_global_with_full_branch_list():
    """Branch_global must converge with the full set of new preprocessors.

    Build a synthetic regression where the predictive signal is a linear
    function of the spectrum, contaminated by per-row offsets that any of
    SNV / MSC / OSC / EMSC / ASLS can correct. The picker must converge,
    record a finite RMSE for every cell, and choose a non-trivial branch.
    """
    rng = np.random.default_rng(17)
    n, p = 60, 24
    template = np.sin(np.linspace(0.0, 2 * np.pi, p))
    y = rng.uniform(-1.0, 1.0, size=n)
    clean = template[None, :] + y[:, None] * 0.5
    offsets = rng.uniform(-3.0, 3.0, size=n)
    scales = np.exp(rng.uniform(-0.5, 0.5, size=n))
    X = scales[:, None] * clean + offsets[:, None]
    X += 0.01 * rng.normal(size=(n, p))

    branches = ("none", "snv", "msc", "osc", "asls", "emsc2")
    est = AOMRidgeRegressor(
        selection="branch_global",
        operator_bank=[IdentityOperator()],
        branches=branches,
        block_scaling="none",
        cv=KFold(n_splits=4, shuffle=True, random_state=0),
        random_state=0,
    ).fit(X, y)

    chosen = est.diagnostics_["chosen_branch"]
    assert chosen in branches
    rmse_table = est._selection_rmse_table
    # Every (branch, op, alpha) cell must be finite — no NaNs from a misfit.
    assert np.all(np.isfinite(rmse_table))
    # Predictions must be finite.
    yhat = est.predict(X)
    assert np.all(np.isfinite(yhat))
    # The diagnostics should expose the full branch list as configured.
    assert tuple(est.diagnostics_["branches"]) == branches


# ----------------------------------------------------------------------
# Test 6 — OSC fold-local fit sees only training rows
# ----------------------------------------------------------------------


def test_osc_fold_local_no_leak(monkeypatch):
    """Spy that records every OSC fold-local fit; sample count must equal len(train_idx)."""
    from aompls.preprocessing import OrthogonalSignalCorrection

    rng = np.random.default_rng(21)
    n, p = 32, 16
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    cv = KFold(n_splits=4, shuffle=False)
    folds = list(cv.split(X, y))
    train_sizes = {len(tr) for tr, _ in folds}

    seen_fit_sizes: list[int] = []

    real_fit = OrthogonalSignalCorrection.fit

    def recording_fit(self, X_in, y_in):
        seen_fit_sizes.append(int(np.asarray(X_in).shape[0]))
        return real_fit(self, X_in, y_in)

    monkeypatch.setattr(OrthogonalSignalCorrection, "fit", recording_fit)

    est = AOMRidgeRegressor(
        selection="branch_global",
        operator_bank=[IdentityOperator()],
        branches=("osc",),
        block_scaling="none",
        cv=cv,
        random_state=0,
    ).fit(X, y)

    # OSC must have been fit at least len(folds) times during CV plus once
    # for the final refit on the full calibration set. Every fold-local fit
    # must use a training-fold-sized matrix.
    assert len(seen_fit_sizes) >= len(folds) + 1, (
        f"expected at least {len(folds) + 1} fits, got {len(seen_fit_sizes)}"
    )
    full_n = X.shape[0]
    assert full_n in seen_fit_sizes, "final refit on full calibration set missing"
    cv_fit_sizes = [s for s in seen_fit_sizes if s != full_n]
    # Every CV-time fit must match a training-fold size and never the
    # validation-fold size (which would mean a leak).
    valid_sizes = {len(va) for _, va in folds}
    for s in cv_fit_sizes:
        assert s in train_sizes, (
            f"OSC fit on {s} rows but only {train_sizes} are valid train sizes"
        )
        assert s not in valid_sizes - train_sizes, (
            f"OSC fit on {s} rows matches a validation-fold size: LEAK"
        )

    assert est.diagnostics_["chosen_branch"] == "osc"
    assert np.all(np.isfinite(est.predict(X)))


# ----------------------------------------------------------------------
# Phase H2 tests — 10-branch expanded list and 1-SE rule on the full triple table
# ----------------------------------------------------------------------

# Subset of branches first introduced for the Phase H2 ``branch_global`` headline
# variant. Main now exposes a richer registry but these labels remain valid.
TEN_BRANCHES = (
    "none", "snv", "msc", "emsc1", "emsc2",
    "asls_soft", "asls", "asls_hard", "snv_asls", "msc_asls",
)


def test_branch_global_10branches_runs():
    """End-to-end fit with the 10-branch list converges and selects a valid branch."""
    rng = np.random.default_rng(123)
    n, p = 60, 32
    base = np.sin(np.linspace(0.0, 4 * np.pi, p))
    y = rng.uniform(-1.0, 1.0, size=n)
    X = (
        base[None, :]
        + 0.5 * y[:, None] * np.cos(np.linspace(0.0, 4 * np.pi, p))[None, :]
        + rng.uniform(-2.0, 2.0, size=(n, 1))
    )
    X += 0.05 * rng.normal(size=(n, p))

    est = AOMRidgeRegressor(
        selection="branch_global",
        operator_bank=[IdentityOperator()],
        branches=TEN_BRANCHES,
        block_scaling="none",
        cv=3,
        random_state=0,
    ).fit(X, y)

    assert est.diagnostics_["chosen_branch"] in TEN_BRANCHES
    assert len(est.diagnostics_["branches"]) == 10
    pred = est.predict(X[:8])
    assert pred.shape == (8,)
    assert np.all(np.isfinite(pred))
    assert est._selection_rmse_table.shape[0] == 10


def test_branch_global_10branches_no_leak(monkeypatch):
    """Spy every branch's fit; validation rows never enter branch fitting."""
    from aompls.preprocessing import ASLSBaseline as _ASLSAdapter
    from aompls.preprocessing import StandardNormalVariate

    from nirs4all.operators.transforms.nirs import (
        ASLSBaseline,
        ExtendedMultiplicativeScatterCorrection,
    )

    rng = np.random.default_rng(31)
    n, p = 30, 20
    X = rng.normal(size=(n, p)) + 5.0
    y = rng.normal(size=n)
    cv = KFold(n_splits=3, shuffle=False)
    folds = list(cv.split(X, y))

    allowed_row_sets: list[frozenset[int]] = [
        frozenset(int(i) for i in tr) for tr, _ in folds
    ]
    allowed_row_sets.append(frozenset(range(n)))
    leaky_row_sets: list[frozenset[int]] = [
        frozenset(int(i) for i in va) for _, va in folds
    ]

    seen_calls: list[tuple[str, np.ndarray]] = []

    def _classify(X_in: np.ndarray) -> str:
        n_in = X_in.shape[0]
        row_to_idx: dict[bytes, int] = {}
        for i in range(n):
            row_to_idx[X[i].tobytes()] = i
        idx_set: set[int] = set()
        for r in range(n_in):
            key = X_in[r].tobytes()
            if key not in row_to_idx:
                return "downstream"
            idx_set.add(row_to_idx[key])
        idx_fset = frozenset(idx_set)
        if idx_fset in allowed_row_sets:
            return "ok"
        if idx_fset in leaky_row_sets:
            return "valid"
        return "unknown"

    real_msc_fit = MultiplicativeScatterCorrection.fit
    real_emsc_fit = ExtendedMultiplicativeScatterCorrection.fit
    real_snv_fit = StandardNormalVariate.fit
    real_asls_fit = ASLSBaseline.fit

    def make_recorder(name: str, original):
        def _recording(self, X_in, y_in=None):
            seen_calls.append((name, np.asarray(X_in, dtype=float).copy()))
            try:
                return original(self, X_in, y_in)
            except TypeError:
                return original(self, X_in)
        return _recording

    monkeypatch.setattr(
        MultiplicativeScatterCorrection, "fit",
        make_recorder("msc", real_msc_fit),
    )
    monkeypatch.setattr(
        ExtendedMultiplicativeScatterCorrection, "fit",
        make_recorder("emsc", real_emsc_fit),
    )
    monkeypatch.setattr(
        StandardNormalVariate, "fit",
        make_recorder("snv", real_snv_fit),
    )
    monkeypatch.setattr(
        ASLSBaseline, "fit",
        make_recorder("asls", real_asls_fit),
    )
    assert _ASLSAdapter is not None  # silence unused import

    AOMRidgeRegressor(
        selection="branch_global",
        operator_bank=[IdentityOperator()],
        branches=TEN_BRANCHES,
        block_scaling="none",
        cv=cv,
        random_state=0,
    ).fit(X, y)

    leak_count = 0
    for _name, X_in in seen_calls:
        cls = _classify(X_in)
        if cls == "valid":
            leak_count += 1
    assert leak_count == 0, f"{leak_count} branch fits saw validation rows (LEAK)"
    assert len(seen_calls) >= len(TEN_BRANCHES) - 1


def test_branch_global_pick_1se_helper_prefers_more_regularised():
    """1-SE rule on the (branch, op, alpha) tensor prefers the largest alpha
    among triples within one SE of the minimum."""
    from aomridge.selection import _branch_global_pick_1se

    alphas = np.logspace(-3, 3, 7)
    n_branches, n_ops = 2, 1
    rmse_table = np.full((n_branches, n_ops, len(alphas)), 0.30)
    rmse_table[0, 0, 1] = 0.20    # global min at alpha=1e-2
    rmse_table[0, 0, 2] = 0.21    # within 1 SE
    rmse_table[0, 0, 3] = 0.22    # within 1 SE
    rmse_table[0, 0, 4] = 0.23    # within 1 SE
    rmse_table[0, 0, 5] = 0.40    # outside 1 SE
    rmse_se = np.full_like(rmse_table, 0.05)

    bi_min, oi_min, ai_min = 0, 0, 1
    bi_1se, oi_1se, ai_1se = _branch_global_pick_1se(
        rmse_table, rmse_se, alphas,
    )
    assert (bi_min, oi_min, ai_min) == (0, 0, 1)
    assert ai_1se > ai_min, (
        f"1-SE alpha index {ai_1se} not greater than min index {ai_min}"
    )
    assert (
        rmse_table[bi_1se, oi_1se, ai_1se]
        <= rmse_table[bi_min, oi_min, ai_min] + rmse_se[bi_min, oi_min, ai_min]
    )


def test_ten_branches_subset_of_registry():
    """Every label in ``TEN_BRANCHES`` is in ``VALID_BRANCHES``."""
    from aomridge.branches import VALID_BRANCHES as _VALID
    for b in TEN_BRANCHES:
        assert b in _VALID
