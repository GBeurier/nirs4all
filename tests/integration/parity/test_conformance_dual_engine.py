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

from typing import TypedDict

import pytest

from . import _conformance_helpers as H
from ._registry import PipelineCase, all_cases

pytestmark = [pytest.mark.parity, pytest.mark.slow]


class _NumPredDivergence(TypedDict):
    """The pinned EXACT num_predictions counts (+ cause) for an intentional native-vs-legacy divergence."""

    legacy: int
    dagml: int
    reason: str


class _CvBestScoreDivergence(TypedDict):
    """The pinned EXACT cv_best_score values (+ cause) for a semantic native-vs-legacy note."""

    legacy: float
    dagml: float
    num_predictions: int
    reason: str


class _RunResultScoreDivergence(TypedDict):
    """The pinned public RunResult score scalars (+ cause) for a semantic native-vs-legacy note."""

    legacy_best_score: float
    dagml_best_score: float
    legacy_best_rmse: float
    dagml_best_rmse: float
    legacy_best_r2: float
    dagml_best_r2: float
    num_predictions: int
    reason: str


# ---------------------------------------------------------------------------
# Documented cross-engine divergences (NATIVE on dag-ml, but the engines do
# NOT agree within tolerance). Each entry is a measured, triaged finding — the
# value is the human-readable cause that the xfail reason surfaces. Marked
# xfail(strict=True) at parametrize time: the case still RUNS on both engines
# (proving native coverage) and the parity assertions still execute, but are
# EXPECTED to fail; an XPASS means the engines converged and the entry must be
# removed (the suite goes RED until it is).
#
# The last remaining strict-xfail (`concat_transform_pca_svd_plsr`) was moved to
# EXPECTED_FALLBACK: the dag-ml FeatureConcat path fits stateful PCA/SVD inside
# each fold, while the Python oracle materializes concat_transform before CV.
# That is a coverage boundary, not a native parity claim.
# (generator_or_models_pls_ridge was here too — it is NOT a divergence in score/winner/winner-y_pred
#  (all equal: best_rmse Δ≈2e-15, winner PLSRegression, winner y_pred Δ=0.0); its ONLY delta is
#  num_predictions 34-legacy vs 32-native, an INTENTIONAL native-vs-legacy refit-policy divergence —
#  moved to NUM_PREDICTIONS_DIVERGENCE and asserted as a parity-note, not strict-xfailed. See ADR-17 1c.)
# (baseline_savgol_rf_kfold, baseline_detrend_firstderiv_gbr, generator_log_range_alpha were
#  here too — the dtype-pin + refit-row-order fix converged them to Δ=0.0 / Δ≈1.3e-7; entries
#  removed from KNOWN_DIVERGENCES, see the notes there.)
# ---------------------------------------------------------------------------
KNOWN_DIVERGENCES: dict[str, str] = {
    # NOTE: sample_augmentation_gaussian / chained / after_savgol were REMOVED
    # from this dict after the native path started materializing the preprocessing
    # prefix before augmentation and threading final-test results back through the
    # RunResult projection. They are live parity assertions now.
    # NOTE: feature_augmentation_replace_three_views was REMOVED from this dict.
    # Legacy includes raw in the model matrix despite the replace-mode context
    # excluding it. dag-ml follows replace semantics. The semantic delta now
    # passes as a RUNRESULT_SCORE_DIVERGENCE parity-note with exact score scalars.
    # NOTE: concat_transform_pca_svd_plsr was REMOVED from this dict. The native
    # path is not equivalent to the Python oracle for stateful concat transforms:
    # legacy materializes the concat_transform output before CV, while dag-ml's
    # FeatureConcat lowering fits PCA/SVD fold-locally. It is therefore an
    # explicit EXPECTED_FALLBACK boundary until backlog #27 can preserve the
    # pre-CV materialized semantics natively.
    # NOTE: the three tree-ensemble cases (baseline_savgol_rf_kfold,
    # baseline_detrend_firstderiv_gbr, baseline_classification_rf_stratified) were REMOVED
    # from this dict — they reach Δ=0.0 now. Their old "fold-materialization/row-order"
    # label was wrong about FOLDS (per-fold train_sample_ids were already byte-identical).
    # The two real divergences were at the MODEL-FIT DATA BOUNDARY (host-side): (1) the
    # resolver/node_runner WIDENED X float32→float64 (.tolist() + dtype=float), shifting
    # SavGol/Detrend inputs ~1e-7 and tipping fixed-seed RF/GBR split thresholds; (2)
    # build_fold_set ordered the REFIT full-train pool by fold-first-seen, not storage
    # order, so the fixed-seed bootstrap drew different rows than legacy's storage-order
    # refit. Pinning the host to the dataset's native dtype + ordering the refit pool by
    # storage order converged both engines. These are LIVE parity assertions now.
    # NOTE: generator_log_range_alpha was REMOVED from this dict — it reaches parity now
    # (Δrmse 2.3e-3 → 1.3e-7). It was the ill-conditioned Ridge (rcond≈2e-8) case whose
    # ONLY divergence was the engine AMPLIFYING the host's float32→float64 widening noise;
    # the same winning alpha was always selected on both engines. Pinning X to the dataset's
    # native dtype removed that ~1e-6 input noise, so the two Ridge solves now converge well
    # under the 1e-3 score tol and the per-sample y_pred tol. A LIVE parity assertion now.
    # NOTE: generator_sample_log_uniform_alpha was REMOVED from this dict after
    # pinning `_seed_` in the case. The unseeded generator sweep remains invalid
    # as a native comparison shape; this conformance case is now deliberately
    # seeded and must pass as a live parity assertion.
    # NOTE: rep_to_sources_basic / rep_to_pp_basic were REMOVED from this dict.
    # They are native dag-ml cases whose only score is cv_best_score. The legacy
    # reshape leaves a stale processing selector and scores a different scalar;
    # dag-ml should not emulate that legacy bug. They now pass as pinned
    # CV_BEST_SCORE_DIVERGENCE parity-notes instead of strict xfails.
    # Sample-level aggregation (mean/median/outlier-exclude) now flows the final-(test) y_pred across the
    # bridge at parity (Gap 2 / A1): the repetition concrete path threads the node results + identity into
    # the projection, so the refit's already-aggregated `(test, None)` sample block fills the final-(test)
    # row (12 vs 12 preds, max |Δy_pred| ≈ 3.6e-5). Their entries were removed from KNOWN_DIVERGENCES — they
    # are LIVE parity assertions now.
    # NOTE: generator_or_models_pls_ridge was REMOVED from this dict (ADR-17 item 1c). It is NOT a
    # strict-xfail anymore — it is a documented INTENTIONAL num_predictions divergence (see
    # NUM_PREDICTIONS_DIVERGENCE below). The engines agree on the winner (PLSRegression), best_score,
    # best_rmse, and the WINNER's per-sample y_pred (Δ=0.0); ONLY num_predictions diverges — legacy refits
    # EVERY model and stores a loser (Ridge) `(test, final)` + `(train, final)` row (34), while dag-ml refits
    # the WINNER ONLY (32, the correct operator-SELECT semantic). dag-ml is RIGHT, so this is a permanent
    # native-vs-legacy delta, not a fixable bug — asserted as a parity-note (winner/score/winner-y_pred), with
    # num_predictions exempted, rather than a strict-xfail that would wrongly chase the legacy refit-all count.
    # NOTE: generator_or_count_seed / generator_or_weights_count_seed are NOT here — they are registry
    # SKIPs (skip_kind="unknown_semantics"), not strict-xfails. Measured across 3 fresh processes, the `_or_`
    # count/`_weights_` subsample is NONDETERMINISTIC even with `_seed_` (varies run-to-run within ONE engine —
    # `_seed_` is not threaded into OrStrategy's sample_with_seed), so a strict-xfail would FLIP to XPASS
    # whenever the two unseeded draws coincide. A skip-with-evidence makes NO parity claim (not a force-pass);
    # the deterministic `_cartesian_` count path (generator_cartesian_count_seed) IS a live GREEN parity case.
}


# ---------------------------------------------------------------------------
# INTENTIONAL native-vs-legacy num_predictions divergences (ADR-17 item 1c).
#
# A case here runs NATIVE on dag-ml and AGREES with legacy on everything that is a
# correctness claim — the SELECTED winner (config_name), best_score/best_rmse/best_r2,
# the selected-metric name, the top-n model set, AND the WINNER's per-sample y_pred —
# but emits a DIFFERENT num_predictions BY DESIGN, because dag-ml's operator-SELECT
# refits the WINNER ONLY (the correct SELECT semantic) while the legacy engine refits
# EVERY model variant and stores the losers' final rows too. This is NOT a bug to be
# fixed (dag-ml is right) and NOT a strict-xfail (a strict-xfail would assert the WRONG
# thing — that the engines should converge on the legacy refit-all count — and would
# XPASS-flip the moment a fix accidentally chased the 34). Instead the case is a
# documented PARITY-NOTE: the conformance body asserts winner identity + FULL metric /
# RunResult-contract parity + WINNER-scoped y_pred parity, and pins the num_predictions
# to the EXACT documented `legacy`/`dagml` counts (NOT merely "exempt", which would let
# an unrelated 31/32 or 34/40 drift pass) — only the one measured +2 loser-final-row
# delta is allowed; any other count FAILS.
#
# generator_or_models_pls_ridge: `{"model": {"_or_": [PLSRegression(10), Ridge(1.0)]}}`
#   over distinct model classes. Both engines select PLSRegression (config_53e81da4_refit),
#   best_rmse 13.286431643519423 vs …425 (Δ≈1.8e-15), winner per-sample y_pred Δ=0.0. The
#   ONLY delta: legacy num_predictions 34 (winner PLS final + loser Ridge final rows) vs
#   dag-ml 32 (winner PLS final only). The 2-entry gap is exactly the one loser Ridge's
#   stored `(test, final)` + `(train, final)` refit rows — losers dag-ml never refits.
#
# generator_chain_model_configs (ADR-17 item 5 slice F): the operator-level `_chain_` analogue —
#   `{"_chain_": [{model: PLSR(5)}, {model: PLSR(10)}, {model: Ridge(1.0)}]}` over distinct model
#   classes. Both engines select PLSRegression (config_53e81da4_refit, the SAME PLS(10) winner as
#   the `_or_` case — content-identical config), best_rmse 13.286431643519423 vs …425 (Δ≈2e-15),
#   winner per-sample y_pred Δ=0.0. The ONLY delta: legacy num_predictions 49 (winner PLS final +
#   the two loser models' final rows + the chain's extra OOF rows) vs dag-ml 47 (winner-only refit);
#   the 2-entry gap is exactly one loser model's stored `(test,final)`+`(train,final)` refit rows —
#   losers dag-ml never refits. Identical mechanism to `generator_or_models_pls_ridge`; `_chain_`
#   sequential ordering does not affect the SELECTED winner (SELECT picks the single CV-best).
#
# Shape: {case_name: {"legacy": <int>, "dagml": <int>, "reason": <str>}}.
NUM_PREDICTIONS_DIVERGENCE: dict[str, _NumPredDivergence] = {
    "generator_or_models_pls_ridge": {
        "legacy": 34,
        "dagml": 32,
        "reason": "multi-model `_or_` operator-SELECT refits the WINNER only (32) — legacy refits every "
        "loser model and stores its (train,final)+(test,final) rows (34); winner/best_score/winner-y_pred all match",
    },
    "generator_chain_model_configs": {
        "legacy": 49,
        "dagml": 47,
        "reason": "multi-model `_chain_` operator-SELECT refits the WINNER only (47) — legacy refits every "
        "loser model and stores its (train,final)+(test,final) rows (49); winner/best_score/winner-y_pred all match (slice F)",
    },
}


# INTENTIONAL native-vs-legacy cv_best_score semantic divergences.
#
# These no-test-set repetition-fusion shapes expose only `cv_best_score` as a
# scalar score. The legacy reshaper mutates dataset axes but returns a stale
# ExecutionContext selector, so downstream preprocessing is applied to a stale
# view. dag-ml follows the corrected native semantics and should not chase that
# legacy scalar. They pass as parity-notes by pinning both exact scalars plus the
# public no-test-set RunResult contract; any drift on either engine fails.
CV_BEST_SCORE_DIVERGENCE: dict[str, _CvBestScoreDivergence] = {
    "rep_to_sources_basic": {
        "legacy": 6.673486795441247,
        "dagml": 6.190624012206827,
        "num_predictions": 11,
        "reason": "legacy rep_to_sources keeps a stale processing selector after reshape; dag-ml scores each OOF sample once",
    },
    "rep_to_pp_basic": {
        "legacy": 6.1427143240770405,
        "dagml": 6.190624012206827,
        "num_predictions": 21,
        "reason": "legacy rep_to_pp keeps a stale processing selector after reshape; dag-ml scores each OOF sample once",
    },
}


# INTENTIONAL native-vs-legacy public-score semantic divergences.
#
# These run native on dag-ml and keep the same model/config/count contract, but
# a documented legacy semantic bug changes the public selected-score scalars.
# They pass by pinning both engines' exact scalars, not by relaxing tolerances.
RUNRESULT_SCORE_DIVERGENCE: dict[str, _RunResultScoreDivergence] = {
    "feature_augmentation_replace_three_views": {
        "legacy_best_score": 12.506075123140343,
        "dagml_best_score": 12.627259100347944,
        "legacy_best_rmse": 12.506075123140343,
        "dagml_best_rmse": 12.627259100347944,
        "legacy_best_r2": 0.601245573618067,
        "dagml_best_r2": 0.5934802643140145,
        "num_predictions": 17,
        "reason": "legacy feature_augmentation replace keeps raw in the model matrix; dag-ml follows replacement semantics",
    },
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
#
#   generator_cartesian_with_param_range: SAME winner config_419ee35e_refit on both
#   engines (best_rmse 10.943759 vs 10.943951, Δ≈1.9e-4 — well under the 1e-3 score
#   tol), 182 vs 182 preds. The winning pipeline is a SNV>1stDer cartesian stage, so
#   the FirstDerivative-amplified PLS noise pushes per-sample y_pred maxΔ = 1.837e-3
#   (14/59 over 1e-3) — same family as above, NOT a selection tip.
#   generator_or_pick_requires: SAME winner config_cc693660_refit (best_rmse 11.970209
#   vs 11.970252, Δ≈4.3e-5), 62 vs 62 preds. The SNV+MSC pick pair carries y_pred maxΔ
#   = 1.109e-3 on a single sample (1/59 over 1e-3) — borderline PLS Rust-vs-sklearn
#   noise. Both relaxed to 5e-3 (above their observed ceilings) under the SAME-winner
#   guard (assert_same_winner runs for every override case before the relaxed compare).
#
#   generator_cartesian_pick: SAME winner config_8ee9444f_refit on both engines (best_rmse
#   11.53146 vs 11.53135, Δ≈1.1e-4 — well under the 1e-3 score tol), 92 vs 92 preds. The
#   pick selects PAIRS of complete pipelines including FirstDerivative branches, so y_pred
#   maxΔ = 2.107e-3 (28/59 over 1e-3) — same FirstDerivative-amplified PLS noise family,
#   NOT a selection tip (in SAME_WINNER_CASES). Relaxed to 5e-3, above the observed ceiling.
#
#   generator_or_pick_mutex3: the SIZE-3 mutex [SNV,MSC,Detrend] forbids only the all-three combo,
#   so EVERY surviving pick-3 variant necessarily carries FirstDerivative (the one op outside the
#   mutex group). The engines select the SAME winner (asserted in SAME_WINNER_CASES) at score parity
#   (well under the 1e-3 score tol) and identical num_predictions; the only gap is the same
#   FirstDerivative-amplified PLS Rust-vs-sklearn per-sample noise — measured maxΔ = 1.455e-3 (14/59
#   over 1e-3). NOT a selection tip, NOT a real divergence. Relaxed to 5e-3 (the family ceiling),
#   under the SAME-winner guard, mirroring generator_or_pick_requires / generator_cartesian_pick.
Y_PRED_TOL_OVERRIDES: dict[str, float] = {
    "generator_or_with_pick": 5e-3,
    "generator_cartesian_stages": 5e-3,
    "generator_cartesian_with_param_range": 5e-3,
    "generator_or_pick_requires": 5e-3,
    "generator_cartesian_pick": 5e-3,
    "generator_or_pick_mutex3": 5e-3,
}


# Generator/constraint cases that MUST select the IDENTICAL winning variant on both
# engines (asserted via config_name). This is the engine-level companion to the
# DSL-level EXACT-survivor lock in test_generators_conformance_extra: the survivor
# SET is locked there, and here we assert the engines AGREE on which survivor wins —
# so a constraint that pruned the right set but tipped the winner is caught.
#
# Scoped to MULTI-variant cases. A SINGLE-variant generator (generator_or_single_variant,
# generator_constraint_prunes_to_one) yields an EMPTY config_name on the dag-ml side
# (no selection among multiple), so assert_same_winner is intentionally NOT applied to
# those — their parity is fully covered by score + num_predictions + y_pred. The
# KNOWN_DIVERGENCES cases (which by definition pick a different winner) are excluded.
SAME_WINNER_CASES: frozenset[str] = frozenset({
    "generator_or_pick_mutex",
    "generator_or_pick_mutex3",
    "generator_or_pick_exclude",
    "generator_cartesian_exclude",
    "generator_combined_constraints",
    # ADR-17 item 5 slice C — UNCONSTRAINED operator generators now route NATIVE operator-SELECT
    # (`_is_unconstrained_operator_generator`): a multi-variant pick/arrange/cartesian survivor set selects
    # a REAL winner config_name natively, so lock that the native winner matches legacy. (`generator_or_then_*`
    # / `generator_cartesian_pick` / `generator_cartesian_count_seed` stay DEMOTED to Python-expand — they are
    # listed here for the SAME-winner lock on the Python-expand survivor set, not for native routing.)
    "generator_or_with_pick",
    "generator_or_arrange_ordered",
    "generator_cartesian_stages",
    "generator_or_then_pick",
    "generator_or_then_arrange",
    "generator_cartesian_pick",
    "generator_cartesian_count_seed",
    # NATIVE param-sweep `_grid_` (routes `_run_native_generation`): a non-degenerate grid selects the
    # TRUE CV-best, so the winner's config_name must be the WINNING variant's name (content-recovered from
    # the winner's refit model params), NOT names[0]. Locks that the native param path winner matches legacy.
    "generator_grid_twelve_variants",
    "generator_grid_n_components_scale",
    # ADR-17 item 5 slice D — MULTI-STEP `_or_` choices route NATIVE operator-SELECT (each survivor = one
    # choice's flat op list); the DEMOTED nested / multistep-pick shapes select a real winner via
    # Python-expand. All three are multi-variant, so lock that the selected winner matches legacy.
    "generator_or_multistep_choices",
    "generator_or_multistep_pick",
    "generator_or_nested",
    # ADR-17 item 5 slice F — param-level `_zip_` (with-model) and operator-level `_chain_` of multi-step
    # sub-pipelines run NATIVE-via-expand: both are multi-variant FULL-parity cases with a deterministic
    # winner, so lock that the selected winner config_name matches legacy (the `_chain_` of distinct MODEL
    # configs is a num_predictions-divergence note, NOT here — its winner is asserted in that branch).
    "generator_zip_single_param",
    "generator_chain_multistep_configs",
})


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
    # by-tag/by-filter separation branches use the same branch/merge keywords and
    # currently fall back until dag-ml serializes separation branch DAGs natively.
    # The metadata-auto fixture is now runnable on legacy, but its shape is
    # preproc-in-branch → feature concat → downstream model; the native path only
    # supports model-in-branch concat today, so this remains a documented boundary.
    "branch_separation_by_metadata_auto",
    "branch_separation_by_tag",
    "branch_separation_by_filter",
    # Classification repetition + vote aggregation requires the final-test surface
    # to be scored at the legacy sample-vote grain; the native repetition path still
    # scores classification final-test at the repetition-row grain.
    "aggregation_classification_vote",
    # Legacy Optuna finetuning mutates model parameters before the final run. The
    # native dag-ml path does not serialize/execute `finetune_params` yet, so it must
    # fall back rather than silently run the untuned model.
    "generator_finetune_params_optuna",
    # Stateful concat_transform sub-operations (PCA/SVD/scalers) are materialized
    # pre-CV by the Python oracle; dag-ml's current FeatureConcat lowering fits
    # them fold-locally, so native execution would compare the wrong semantics.
    "concat_transform_pca_svd_plsr",
    # by-source separation / per-source models / source-concat multi-source shapes.
    "multi_source_by_source_branch_shared_preproc",
    "multi_source_per_source_models_stacking",
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
    * ``NUM_PREDICTIONS_DIVERGENCE`` cases get NO mark — they PASS as a documented
      parity-note (winner + metric + winner-y_pred parity, num_predictions exempt);
      the conformance body branches on the allowlist (ADR-17 1c).
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

    if case.name in NUM_PREDICTIONS_DIVERGENCE:
        # INTENTIONAL native-vs-legacy num_predictions divergence (ADR-17 1c): dag-ml's
        # operator-SELECT refits the WINNER ONLY (the correct SELECT semantic) while legacy
        # refits every loser too. Assert the FULL correctness surface — SAME winner, METRIC +
        # structure parity, the whole RunResult contract (best_score / best_rmse / best_r2 /
        # the selected-metric name / the top-n model set), and the WINNER's per-sample y_pred —
        # and pin num_predictions to the EXACT documented legacy/dag-ml counts (NOT merely
        # exempt): only the one measured +2 loser-final-row delta passes, any other count is a
        # regression and FAILS. A PASSING parity-note, NOT a strict-xfail, so it never
        # XPASS-flips on a spurious convergence to the wrong (34) count.
        expected = NUM_PREDICTIONS_DIVERGENCE[case.name]
        H.assert_same_winner(legacy, dagml, case)
        H.assert_num_predictions_divergence(legacy, dagml, case, expected["legacy"], expected["dagml"])
        H.assert_score_parity_metrics_only(legacy, dagml, case)
        H.assert_runresult_contract(legacy, dagml, case, num_predictions_exempt=True)
        H.assert_winner_y_pred_parity(legacy, dagml, case)
        return

    if case.name in CV_BEST_SCORE_DIVERGENCE:
        # INTENTIONAL native-vs-legacy cv_best_score divergence for no-test-set
        # repetition-fusion shapes. dag-ml follows the corrected reshape semantics;
        # legacy exposes a stale-context scalar. Pin the exact scalars while still
        # asserting the rest of the public no-test-set RunResult contract.
        expected = CV_BEST_SCORE_DIVERGENCE[case.name]
        H.assert_num_predictions_parity(legacy, dagml)
        assert legacy.num_predictions == expected["num_predictions"], (
            f"{case.name}: documented num_predictions changed — expected {expected['num_predictions']}, "
            f"got legacy={legacy.num_predictions} dag-ml={dagml.num_predictions}"
        )
        H.assert_runresult_contract(legacy, dagml, case)
        H.assert_cv_best_score_divergence(legacy, dagml, case, expected["legacy"], expected["dagml"])
        return

    if case.name in RUNRESULT_SCORE_DIVERGENCE:
        # INTENTIONAL native-vs-legacy public-score divergence. The engines agree
        # on the selected model/config and prediction-row count; only the public
        # score scalars are documented as different and pinned exactly.
        expected = RUNRESULT_SCORE_DIVERGENCE[case.name]
        H.assert_num_predictions_parity(legacy, dagml)
        assert legacy.num_predictions == expected["num_predictions"], (
            f"{case.name}: documented num_predictions changed — expected {expected['num_predictions']}, "
            f"got legacy={legacy.num_predictions} dag-ml={dagml.num_predictions}"
        )
        assert legacy.best.get("metric") == dagml.best.get("metric"), (
            f"{case.name}: selected metric legacy={legacy.best.get('metric')!r} dag-ml={dagml.best.get('metric')!r}"
        )
        H.assert_same_winner(legacy, dagml, case)
        H.assert_runresult_score_divergence(
            legacy,
            dagml,
            case,
            legacy_best_score=expected["legacy_best_score"],
            dagml_best_score=expected["dagml_best_score"],
            legacy_best_rmse=expected["legacy_best_rmse"],
            dagml_best_rmse=expected["dagml_best_rmse"],
            legacy_best_r2=expected["legacy_best_r2"],
            dagml_best_r2=expected["dagml_best_r2"],
        )
        return

    # NATIVE: the real both-engines-agree contract. For a KNOWN_DIVERGENCES case
    # these run under the collection-time strict-xfail and are expected to fail.
    H.assert_score_parity(legacy, dagml, case)
    H.assert_num_predictions_parity(legacy, dagml)
    H.assert_runresult_contract(legacy, dagml, case)

    # Engine-level WINNER-IDENTITY lock for the multi-variant generator/constraint
    # cases: the DSL-level survivor SET is locked in test_generators_conformance_extra;
    # here both engines must agree on WHICH survivor wins (a wrong-prune that tipped
    # the winner is caught). Single-variant cases are excluded (empty dag-ml config_name).
    if case.name in SAME_WINNER_CASES:
        H.assert_same_winner(legacy, dagml, case)

    y_pred_tol = Y_PRED_TOL_OVERRIDES.get(case.name, H._DEFAULT_YPRED_TOL)  # noqa: SLF001
    if case.name in Y_PRED_TOL_OVERRIDES:
        # A relaxed y_pred tolerance is only valid when the engines picked the SAME
        # winning variant — make that claim LOAD-BEARING, not a comment.
        H.assert_same_winner(legacy, dagml, case)
    H.assert_y_pred_parity(legacy, dagml, case, tol=y_pred_tol)
