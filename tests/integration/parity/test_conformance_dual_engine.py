"""DUAL-ENGINE CONFORMANCE: the same case on legacy AND dag-ml, asserted equal.

The single contract reference for the nirs4all-core → dag-ml migration. Every
:class:`PipelineCase` in the registry is run on BOTH engines via
:func:`_conformance_helpers.dual_engine_runner` and dispositioned by what the
dag-ml engine actually did:

* **NATIVE** (``dagml_native is True``) — the dag-ml backend ran the pipeline
  itself. Assert FULL parity: score (within the case's tolerances), exact
  ``num_predictions``, the public ``RunResult`` contract, and per-sample
  ``y_pred``. This is the real "both engines agree" claim.
* **FALLBACK** (``dagml_native is False``) — the dag-ml path rejected the shape
  and re-ran legacy. A parity assertion here would be legacy-vs-legacy and
  trivially true, so we instead assert ``dagml_native is False`` as a DOCUMENTED
  coverage-boundary: this case is not yet dag-ml-native. The day dag-ml gains
  native coverage, this assertion flips to a failure and forces the case into
  the NATIVE branch — the boundary can never silently widen.

Known cross-engine NON-EQUALITIES are marked ``xfail(strict=True)`` AT
PARAMETRIZE TIME (NOT loosened tolerances). The case still RUNS on both engines
and the parity assertions still execute — they are merely EXPECTED to fail. A
strict xfail XPASS-flips the moment the engines converge, so a fixed divergence
never goes silent. The ``legacy_bug`` registry cases are likewise strict-xfail
(the legacy engine itself is broken, so no legacy oracle exists), and the
``fixture`` / ``unknown_semantics`` cases ``skip`` — mirroring
``test_parity_smoke``.

Slow: each case runs twice (legacy + dag-ml). Gated by the ``slow`` marker.

    pytest tests/integration/parity/test_conformance_dual_engine.py -q
"""

from __future__ import annotations

import pytest

from . import _conformance_helpers as H
from ._registry import PipelineCase, all_cases

pytestmark = [pytest.mark.parity, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Documented cross-engine divergences (NATIVE on dag-ml, but the engines do
# NOT agree within tolerance). Each entry is a measured, triaged finding — the
# value is the human-readable cause that the xfail reason surfaces. Marked
# xfail(strict=True) at parametrize time: the case still RUNS on both engines
# (proving native coverage) and the parity assertions still execute, but are
# EXPECTED to fail; an XPASS means the engines converged and the entry must be
# removed (the suite goes RED until it is).
#
# Measured legacy↔dag-ml best_rmse deltas (regression sample_data) at scope time:
#   sample_augmentation_gaussian              13.28643 vs 13.38345  (Δ≈9.7e-2)
#   sample_augmentation_chained               13.28643 vs 12.19792  (Δ≈1.1e0)
#   sample_augmentation_after_savgol          16.76066 vs 15.83810  (Δ≈9.2e-1)
#   feature_augmentation_replace_three_views  12.50608 vs 12.62726  (Δ≈1.2e-1)
#   concat_transform_pca_svd_plsr             14.13327 vs 15.53153  (Δ≈1.4e0)
#   baseline_savgol_rf_kfold                  24.29178 vs 24.36652  (Δ≈7.5e-2)
#   baseline_detrend_firstderiv_gbr          22.15498 vs 21.88759   (Δ≈2.7e-1)
#   generator_finetune_params_optuna          19.46609 vs 21.12623  (Δ≈1.7e0)
#   generator_log_range_alpha                 12.13799 vs 12.14032  (Δ≈2.3e-3, SAME winner)
#   generator_sample_log_uniform_alpha        13.26828 vs 13.79983  (Δ≈5.3e-1, DIFFERENT winner)
# ---------------------------------------------------------------------------
KNOWN_DIVERGENCES: dict[str, str] = {
    # Augmentation expands the train set; legacy vs dag-ml apply the augmentation
    # ops in a different order / with a different per-op RNG draw, so the fitted
    # model differs. A real numerical-parity item for the augmentation kernels.
    "sample_augmentation_gaussian": "augmentation RNG/order differs (Δrmse≈9.7e-2)",
    "sample_augmentation_chained": "chained augmentation RNG/order differs (Δrmse≈1.1e0)",
    "sample_augmentation_after_savgol": "augmentation-after-preproc RNG/order differs (Δrmse≈9.2e-1)",
    # Feature-view fan-out: the three replace-views are built in a different order
    # across engines, so the concatenated feature matrix (and the model) differs.
    "feature_augmentation_replace_three_views": "feature-view build order differs (Δrmse≈1.2e-1)",
    "concat_transform_pca_svd_plsr": "concat_transform view order/decomposition differs (Δrmse≈1.4e0)",
    # Tree ensembles with random_state=42: sklearn RF/GBR are DETERMINISTIC given
    # identical fit-data ROW ORDER, so this is NOT inherent RNG noise. The cause is
    # a fold-materialization / row-order divergence — legacy and dag-ml feed the
    # estimator the training rows in a different order, so the fitted trees differ.
    # Parity debt in dag-ml's fold materialization, not a stochastic estimator.
    "baseline_savgol_rf_kfold": "RandomForest fold-materialization/row-order differs "
    "legacy↔dag-ml (random_state=42 → deterministic given identical order; Δrmse≈7.5e-2)",
    "baseline_detrend_firstderiv_gbr": "GradientBoosting fold-materialization/row-order differs "
    "legacy↔dag-ml (random_state=42 → deterministic given identical order; Δrmse≈2.7e-1)",
    # RF CLASSIFICATION sibling (SAME row-order cause, evidenced): both engines pass
    # random_state=42 + select balanced_accuracy + see the SAME 10-class space [0..9]
    # (class 0 is real, not a label-encode bug). But the SELECTED best_score
    # (balanced_accuracy) DIFFERS — legacy 0.160317 vs dag-ml 0.166667 — and 11/30
    # per-sample class labels differ, both consequences of the differing training
    # row order fed to the deterministic RF. Not a dag-ml RF-classification bug
    # (n_classes + vote aggregation are correct; plain best_accuracy is even equal).
    "baseline_classification_rf_stratified": "RandomForest fold-materialization/row-order differs "
    "legacy↔dag-ml (selected balanced_accuracy 0.1603 vs 0.1667; 11/30 sample labels)",
    # Optuna drives its own search; the two engines explore a different trial
    # sequence, so the selected hyperparameters (and final score) differ.
    "generator_finetune_params_optuna": "Optuna trial sequence differs across engines (Δrmse≈1.7e0)",
    # _log_range_ alpha sweep: the SAME winning alpha is selected on BOTH engines
    # (config_e2bc756e, stable across runs), but that winner is an ILL-CONDITIONED
    # Ridge solve (rcond≈2e-8) that amplifies the cross-engine input float noise —
    # per-sample y_pred wobbles ~8.4e-2 and best_rmse by 2.33e-3 (just over the 1e-3
    # score tol). A numerical-conditioning divergence (NOT a selection tip — winner
    # is identical), kept loud until the engines' Ridge solves converge.
    "generator_log_range_alpha": "ill-conditioned Ridge (rcond≈2e-8) amplifies cross-engine "
    "float noise; same winner, Δrmse≈2.3e-3 / Δy_pred≈8.4e-2",
    # _sample_ random-alpha sweep that does NOT set _seed_ → genuinely UNSEEDED
    # stochastic. The random variant set (and therefore the winner) is not
    # reproducible across the two engines' samplers, so best_rmse differs run to
    # run (observed Δ up to ≈5.3e-1, different winning config). Honest disposition:
    # unseeded _sample_ is nondeterministic across engines (pin _seed_ in the case
    # to make it deterministic and re-evaluate); not a proven fixed divergence.
    "generator_sample_log_uniform_alpha": "unseeded _sample_ (_seed_ not set) → "
    "nondeterministic variant set/winner across engines (Δrmse up to ≈5.3e-1)",
    # No-test-set rep shapes: the only score they produce is cv_best_score, a SCALAR
    # that DIVERGES (rep_to_sources 6.6735 vs 6.1906; rep_to_pp 6.1427 vs 6.1906).
    # The cause is the rep OOF-aggregation difference — legacy DOUBLE-COUNTS the
    # overlapping rep folds (ShuffleSplit reps appear in several folds; legacy
    # concatenates them, so each rep pipeline scores a different cv), while dag-ml
    # aggregates each sample's OOF exactly ONCE (6.1906 for both). dag-ml is the
    # CORRECT value; this is a PERMANENT semantic divergence, not a fixable bug.
    # A2 (2a-iii) surfaced the per-sample OOF avg y_pred but does NOT touch this
    # scalar — the divergence is in the score's aggregation semantics, not the
    # per-sample values — so these stay xfailed (measured: still XFAIL after A2).
    "rep_to_sources_basic": "PERMANENT semantic divergence in cv_best_score: legacy double-counts "
    "overlapping rep folds (6.6735); dag-ml aggregates each OOF sample once (6.1906, the correct value)",
    "rep_to_pp_basic": "PERMANENT semantic divergence in cv_best_score: legacy double-counts "
    "overlapping rep folds (6.1427); dag-ml aggregates each OOF sample once (6.1906, the correct value)",
    # Sample-level aggregation (mean/median/outlier-exclude) now flows the final-(test) y_pred across the
    # bridge at parity (Gap 2 / A1): the repetition concrete path threads the node results + identity into
    # the projection, so the refit's already-aggregated `(test, None)` sample block fills the final-(test)
    # row (12 vs 12 preds, max |Δy_pred| ≈ 3.6e-5). Their entries were removed from KNOWN_DIVERGENCES — they
    # are LIVE parity assertions now.
}


# Per-case per-sample y_pred tolerance overrides. The default y_pred tolerance is
# 1e-3 (the SNV-PLS noise ceiling). A FEW cases run a SNV>1stDer-PLS variant whose
# winner is IDENTICAL on both engines but whose per-sample y_pred carries more PLS
# Rust-vs-sklearn float noise than the SNV-only baselines — the FirstDerivative
# preprocessing amplifies it. These are NOT selection tips (proven below) and NOT a
# real divergence, so the y_pred tolerance is relaxed to a justified value rather
# than xfailed. Score parity for these still holds at the default 1e-3 (Δrmse≈9e-6).
#
# EVIDENCE (legacy vs dag-ml, regression sample_data):
#   generator_or_with_pick + generator_cartesian_stages both select the SAME winner
#   on both engines — config_e57dcd52_refit / PLSRegression / preprocessings
#   'SNV>1stDer' — with best_rmse 11.972629 vs 11.972638 (Δ≈9.3e-6, pure PLS noise).
#   Per-sample y_pred maxΔ = 3.45e-3 (31/59 over 1e-3), meanΔ = 1.1e-3. 5e-3 sits
#   above the observed 3.45e-3 ceiling while still catching any real divergence.
Y_PRED_TOL_OVERRIDES: dict[str, float] = {
    "generator_or_with_pick": 5e-3,
    "generator_cartesian_stages": 5e-3,
}


# EXPECTED-FALLBACK allowlist: the cases the dag-ml path LEGITIMATELY rejects
# today (branch+merge, by-source multi-source, the preprocessing-keyword shapes),
# so engine="dag-ml" transparently re-runs legacy. A case that falls back but is
# NOT on this allowlist is a native-coverage REGRESSION (a shape that used to run
# native now rejects) and MUST FAIL — never silently pass as a boundary. When
# dag-ml gains native coverage for one of these, it leaves the allowlist (the test
# then demands native parity). Measured at scope time; see the probe in the PR.
EXPECTED_FALLBACK: frozenset[str] = frozenset({
    # branch (duplication) + merge → multi-model; dag-ml bridge spike does not yet
    # serialize the branch/merge step keywords, so the whole shape falls back.
    "branch_dup_three_way_merge_predictions",
    "branch_dup_two_way_merge_features",
    "branch_dup_named_with_metamodel",
    "branch_dup_merge_all",
    # by-source separation / per-source models / source-concat multi-source shapes.
    "multi_source_by_source_branch_shared_preproc",
    "multi_source_by_source_branch_distinct_preproc",
    "multi_source_per_source_models_stacking",
    "multi_source_sources_concat_then_rf",
    # the explicit `preprocessing` keyword + fit_on_all + force_layout shapes.
    "preprocessing_explicit_keyword",
    "preprocessing_fit_on_all",
    "preprocessing_force_layout_2d",
})


def _params() -> list:
    """Build the parametrize list, attaching the right xfail/skip marks per case.

    Marks applied at collection time so pytest tracks XPASS (strict xfail flip)
    and so the disposition is visible in ``-v`` output without per-test branching:

    * ``legacy_bug`` registry cases → ``xfail(strict=True)`` (no legacy oracle).
    * ``fixture`` / ``unknown_semantics`` registry cases → ``skip``.
    * ``KNOWN_DIVERGENCES`` cases → ``xfail(strict=True)`` with the measured cause.
    """
    params = []
    for case in all_cases():
        marks = []
        if case.skip_reason:
            if case.skip_kind == "legacy_bug":
                marks.append(pytest.mark.xfail(reason=f"[legacy_bug] {case.skip_reason}", strict=True))
            else:
                marks.append(pytest.mark.skip(reason=f"[{case.skip_kind or 'unknown'}] {case.skip_reason}"))
        elif case.name in KNOWN_DIVERGENCES:
            marks.append(
                pytest.mark.xfail(
                    reason=f"documented cross-engine divergence — {KNOWN_DIVERGENCES[case.name]}",
                    strict=True,
                )
            )
        params.append(pytest.param(case, id=case.name, marks=marks))
    return params


def _runnable_cases() -> list:
    """All cases that can actually be exercised (no registry ``skip_reason``).

    The ``fixture`` / ``unknown_semantics`` / ``legacy_bug`` registry skips can't
    construct or can't run (missing fixture, unconfirmed semantics, a legacy
    crash), so they are excluded from the boundary test entirely — there is no
    dag-ml leg to observe for them.
    """
    return [pytest.param(c, id=c.name) for c in all_cases() if not c.skip_reason]


@pytest.mark.parametrize("case", _runnable_cases())
def test_native_fallback_boundary(case: PipelineCase) -> None:
    """The dag-ml native/fallback status EXACTLY matches the EXPECTED_FALLBACK allowlist.

    NEVER xfailed (and NOT wrapped by any KNOWN_DIVERGENCES marker): this test
    runs for EVERY runnable case — INCLUDING the KNOWN_DIVERGENCES ones — so the
    boundary can never be masked by the parity test's strict-xfail marker. It is
    the single source of truth for native-vs-fallback:

    * a case that fell back but is NOT on the allowlist → native-coverage
      REGRESSION → FAIL;
    * a case that ran NATIVE but IS on the allowlist → STALE allowlist entry
      (the shape gained native coverage) → FAIL (drop it from EXPECTED_FALLBACK).

    Runs only the dag-ml leg (no legacy run) via :func:`dagml_native_status`.
    """
    dataset = H.make_dataset(case)
    native = H.dagml_native_status(case, dataset)
    on_allowlist = case.name in EXPECTED_FALLBACK

    if native:
        assert not on_allowlist, (
            f"{case.name}: ran NATIVE on dag-ml but is on the EXPECTED_FALLBACK allowlist — "
            "STALE entry; the shape gained native coverage, remove it from EXPECTED_FALLBACK "
            "(the parity test will then demand native parity)"
        )
    else:
        assert on_allowlist, (
            f"{case.name}: dag-ml fell back to legacy but is NOT on the EXPECTED_FALLBACK allowlist "
            "— native-coverage regression (a shape that used to run native now rejects), or a new "
            "fallback that must be triaged + allowlisted with a reason"
        )


@pytest.mark.parametrize("case", _params())
def test_dual_engine_conformance(case: PipelineCase) -> None:
    """Run ``case`` on both engines and assert PARITY for its disposition.

    The native/fallback BOUNDARY is owned by the never-xfailed
    :func:`test_native_fallback_boundary`; here a fallback case simply returns
    (its boundary is asserted there), and a native case asserts full parity.
    Keeping the boundary out of this strict-xfail-marked body is what prevents a
    KNOWN_DIVERGENCES/legacy_bug marker from masking a boundary regression.
    """
    dataset = H.make_dataset(case)
    run = H.dual_engine_runner(case, dataset)
    legacy, dagml, native = run["legacy"], run["dag-ml"], run["dagml_native"]

    if not native:
        # FALLBACK: parity is N/A (legacy-vs-legacy). The boundary (allowlist
        # membership) is enforced by test_native_fallback_boundary, NEVER here —
        # so this test's strict-xfail marker cannot mask a boundary regression.
        return

    # NATIVE: the real both-engines-agree contract. For a KNOWN_DIVERGENCES case
    # these run under the collection-time strict-xfail and are expected to fail.
    H.assert_score_parity(legacy, dagml, case)
    H.assert_num_predictions_parity(legacy, dagml)
    H.assert_runresult_contract(legacy, dagml, case)

    y_pred_tol = Y_PRED_TOL_OVERRIDES.get(case.name, H._DEFAULT_YPRED_TOL)  # noqa: SLF001
    if case.name in Y_PRED_TOL_OVERRIDES:
        # A relaxed y_pred tolerance is only valid when the engines picked the SAME
        # winning variant — make that claim LOAD-BEARING, not a comment.
        H.assert_same_winner(legacy, dagml, case)
    H.assert_y_pred_parity(legacy, dagml, case, tol=y_pred_tol)
