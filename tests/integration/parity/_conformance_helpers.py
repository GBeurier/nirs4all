"""Shared helpers for the DUAL-ENGINE CONFORMANCE PACK (tasks #50/#58).

This is the single contract reference both nirs4all engines must meet: the same
``PipelineCase`` is run on the legacy orchestrator (``engine="legacy"``) and on
the dag-ml backend (``engine="dag-ml"``, selected EXPLICITLY), and the two
``RunResult`` objects are asserted EQUAL within the case's recorded tolerances.

Prior parity tests captured a legacy gold baseline (``_oracle``) or ran the
dag-ml bridge in isolation (``test_dagml_*``). None ran the SAME case on BOTH
engines and asserted equality — these helpers fill that gap and the
``test_conformance_*`` modules consume them.

LOAD-BEARING fallback detection
-------------------------------
``run(engine="dag-ml")`` transparently re-runs on
the LEGACY engine for any pipeline shape the dag-ml path cannot honor yet (the
P1b/P0 reject→fallback), emitting a ``"falling back to the legacy engine"``
warning (:mod:`nirs4all.api.run`). A fallback run is legacy-under-the-hood, so
asserting "dag-ml == legacy" on it would be a trivially-true legacy-vs-legacy
claim. :func:`dual_engine_runner` therefore records ``dagml_native`` from TWO
independent signals — the fallback warning text AND the ``per_dataset`` engine
marker (:meth:`RunResult._is_dagml_engine`) — and the conformance tests only
make a real parity claim when ``dagml_native`` is ``True``.

Tolerances
----------
Cross-engine float noise is real: the dag-ml PLS kernel is Rust, legacy is
sklearn, so a deterministic PLS test ``y_pred`` differs by ~1e-4 (measured) and
a ``y_processing``-inverse pipeline by ~6e-4. Score parity reuses the case's
recorded :attr:`PipelineCase.metric_tolerances` (or :data:`_DEFAULT_SCORE_TOL`,
the registry's established ``1e-3`` engine-parity standard, when a case declares
none). Per-sample ``y_pred`` parity defaults to :data:`_DEFAULT_YPRED_TOL`
(``1e-3``, safely above the observed ~6e-4 PLS+inverse noise) — NOT a blanket
``1e-5``, which would spuriously fail every PLS case.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

import nirs4all
from nirs4all.data import DatasetConfigs

from ._datasets import dataset_path
from ._oracle import compare, observe
from ._registry import PipelineCase

# The dag-ml→legacy fallback warning fragment, shared by both fallback paths
# (DagMlUnavailable + DagMlUnsupported) in nirs4all/api/run.py. Matching this
# fragment is the primary fallback signal.
_FALLBACK_WARNING_FRAGMENT = "falling back to the legacy engine"

# Engine-parity standard used across the parity registry (e.g.
# baseline_vertical_slice records rmse/r2 tol 1e-3): absorbs sklearn-vs-Rust PLS
# float noise (~7e-6 on rmse/r2) while still catching real divergence. Applied
# to a case's enforced metrics when it declares no explicit metric_tolerances.
_DEFAULT_SCORE_TOL = 1e-3

# Per-sample y_pred tolerance. Measured cross-engine PLS noise is ~1.1e-4, and a
# y_processing-inverse pipeline reaches ~6e-4; 1e-3 sits safely above both while
# still catching the ~1e-1..1e0 divergences the stochastic/augmentation cases show.
_DEFAULT_YPRED_TOL = 1e-3

_DualResult = dict[str, Any]


def make_dataset(case: PipelineCase) -> DatasetConfigs:
    """Resolve a case's ``dataset_key`` + ``dataset_kwargs`` into a ``DatasetConfigs``.

    Mirrors ``test_parity_smoke._make_dataset`` so the conformance pack feeds the
    two engines byte-identical dataset configuration.
    """
    return DatasetConfigs(dataset_path(case.dataset_key), **case.dataset_kwargs)


def dual_engine_runner(case: PipelineCase, dataset: DatasetConfigs) -> _DualResult:
    """Run ``case`` on BOTH engines from the SAME dataset config; report native-ness.

    Runs ``engine="legacy"`` then ``engine="dag-ml"`` EXPLICITLY (NOT the
    default ``engine=None``) on a freshly materialized pipeline per engine (the
    factory yields fresh operator instances, so the two runs never share mutable
    state). The explicit engine is load-bearing: ``resolve_engine`` honors
    ``$N4A_ENGINE``, so under ``N4A_ENGINE=legacy`` an ``engine=None`` dag-ml leg
    would silently run LEGACY and pass as a fake fallback boundary. The dag-ml
    run is wrapped in a warning capture: if the ``"falling back to the legacy
    engine"`` warning fires, the dag-ml run actually executed LEGACY
    (reject→fallback), so ``dagml_native`` is ``False`` regardless of any other
    signal.

    ``dagml_native`` is the AND of two independent signals so a future change to
    either path cannot silently turn a legacy fallback into a fake parity pass:

    * no fallback warning fired during the dag-ml run, AND
    * :meth:`RunResult._is_dagml_engine` is ``True`` (the dag-ml backend tags
      ``per_dataset[name]["engine"] == "dag-ml"``).

    Args:
        case: the pipeline contract to exercise.
        dataset: resolved dataset config (built once via :func:`make_dataset`).

    Returns:
        ``{"legacy": RunResult, "dag-ml": RunResult, "dagml_native": bool}``.
    """
    legacy = nirs4all.run(pipeline=case.pipeline, dataset=dataset, verbose=0, engine="legacy")
    dagml, dagml_native = _run_dagml_leg(case, dataset)
    return {"legacy": legacy, "dag-ml": dagml, "dagml_native": dagml_native}


def _run_dagml_leg(case: PipelineCase, dataset: DatasetConfigs) -> tuple[Any, bool]:
    """Run ONLY the dag-ml leg (explicit ``engine="dag-ml"``); return ``(result, native)``.

    Shared by :func:`dual_engine_runner` and :func:`dagml_native_status` so the
    fallback detection (warning fragment AND the per_dataset engine marker) lives
    in one place. The explicit engine is load-bearing — see
    :func:`dual_engine_runner`.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dagml = nirs4all.run(pipeline=case.pipeline, dataset=dataset, verbose=0, engine="dag-ml")
        fell_back = any(_FALLBACK_WARNING_FRAGMENT in str(w.message) for w in caught)
    dagml_native = (not fell_back) and bool(dagml._is_dagml_engine())  # noqa: SLF001
    return dagml, dagml_native


def dagml_native_status(case: PipelineCase, dataset: DatasetConfigs) -> bool:
    """Whether the dag-ml engine ran ``case`` NATIVELY (no legacy fallback).

    Runs ONLY the dag-ml leg (no legacy run) so the never-xfailed boundary test
    can assert native/fallback status against the allowlist cheaply, without the
    full parity double-run.
    """
    _result, native = _run_dagml_leg(case, dataset)
    return native


def _enforced_score_tolerances(case: PipelineCase, gold: dict[str, Any], obs: dict[str, Any]) -> dict[str, float]:
    """The metric→tolerance map ACTUALLY enforced for ``case`` against this pair of observations.

    Layers:

    * A case's own :attr:`PipelineCase.metric_tolerances` wins verbatim — it
      encodes case-author intent (e.g. enforce r2 AND rmse at 1e-3). Those cases
      always have a test set, so the metrics exist.
    * Otherwise the DEFAULT primary metric (``rmse`` / ``accuracy``) is enforced
      at :data:`_DEFAULT_SCORE_TOL` when present in BOTH observations.
    * A no-test-set shape (``rep_to_sources`` / ``rep_to_pp``) produces no final
      ``rmse`` (``best_score`` is NaN) on EITHER engine, so the primary drops out.
      Rather than fall to a STRUCTURE-ONLY pass (which masks a real score
      divergence), enforce ``cv_best_score`` — the only score these shapes produce
      — when it is present in both. This SURFACES the rep OOF-aggregation
      divergence (legacy concatenates overlapping rep folds; dag-ml aggregates the
      OOF differently) instead of hiding it. Cases where dag-ml is authoritative
      can then be moved to an explicit passing non-equivalence assertion instead
      of silently passing structure-only.
    """
    if case.metric_tolerances:
        return dict(case.metric_tolerances)
    primary = "accuracy" if case.task == "classification" else "rmse"
    gold_metrics = gold.get("metrics", {})
    obs_metrics = obs.get("metrics", {})
    if primary in gold_metrics and primary in obs_metrics:
        return {primary: _DEFAULT_SCORE_TOL}
    if "cv_best_score" in gold_metrics and "cv_best_score" in obs_metrics:
        return {"cv_best_score": _DEFAULT_SCORE_TOL}
    return {}


def assert_score_parity(legacy: Any, dagml: Any, case: PipelineCase) -> None:
    """Assert dag-ml scores match legacy within the case's tolerances (reuses ``_oracle``).

    The legacy ``RunResult`` is the oracle (ADR-01). ``observe`` extracts the
    same JSON-serializable record from both, and ``compare`` enforces exact
    structural equality (models / datasets / num_predictions) plus per-metric
    absolute tolerance from :func:`_enforced_score_tolerances` — which only
    enforces metrics that ACTUALLY EXIST in both results (so a no-test-set rep
    case compares structurally rather than demanding an absent ``rmse``).
    """
    gold = observe(legacy, case.task)
    obs = observe(dagml, case.task)
    violations = compare(gold, obs, _enforced_score_tolerances(case, gold, obs))
    assert not violations, f"{case.name}: legacy↔dag-ml score parity violated:\n  " + "\n  ".join(violations)


def assert_score_parity_metrics_only(legacy: Any, dagml: Any, case: PipelineCase) -> None:
    """Assert dag-ml METRIC scores match legacy, EXEMPTING the structural ``num_predictions`` equality.

    The :func:`assert_score_parity` comparison is structural-AND-metric: ``compare``
    flags a ``num_predictions`` mismatch even when every enforced metric is within
    tolerance. For a documented :data:`NUM_PREDICTIONS_DIVERGENCE` case (a multi-model
    ``_or_`` where legacy refits every loser and dag-ml refits the winner only, the
    correct SELECT semantic) that structural check is EXPECTED to diverge, so this
    variant drops ONLY ``num_predictions`` from ``gold``/``observed`` and enforces the
    metric tolerances PLUS the remaining structural fields (``models``/``datasets``,
    which still MATCH — both engines RUN both models; only the refit COUNT differs).
    The enforced metrics are the SELECTED winner's ``best_score``/``rmse``/``r2``, which
    DO match (measured Δ≈2e-15). The winner identity is locked separately by
    :func:`assert_same_winner`, and the winner's per-sample y_pred by
    :func:`assert_winner_y_pred_parity`, so the only thing exempted is the prediction
    COUNT, never a score, model set, or winner.
    """
    gold = observe(legacy, case.task)
    obs = observe(dagml, case.task)
    gold_exempt = {k: v for k, v in gold.items() if k != "num_predictions"}
    obs_exempt = {k: v for k, v in obs.items() if k != "num_predictions"}
    violations = compare(gold_exempt, obs_exempt, _enforced_score_tolerances(case, gold, obs))
    assert not violations, (
        f"{case.name}: legacy<->dag-ml METRIC/structure parity violated (num_predictions exempt):\n  " + "\n  ".join(violations)
    )


def assert_num_predictions_parity(legacy: Any, dagml: Any) -> None:
    """Assert the two engines emit EXACTLY the same number of prediction entries."""
    assert legacy.num_predictions == dagml.num_predictions, (
        f"num_predictions diverged: legacy={legacy.num_predictions} != dag-ml={dagml.num_predictions}"
    )


def assert_num_predictions_divergence(legacy: Any, dagml: Any, case: PipelineCase, expected_legacy: int, expected_dagml: int) -> None:
    """Assert the DOCUMENTED EXACT num_predictions for an intentional native-vs-legacy divergence.

    The :data:`NUM_PREDICTIONS_DIVERGENCE` companion to :func:`assert_num_predictions_parity`.
    Rather than merely EXEMPTING ``num_predictions`` from parity (which would let ANY future
    count drift — 31/32, 34/40, a stray extra row — pass silently as long as the winner score
    and y_pred are unchanged), this pins BOTH engines to the EXACT measured counts: legacy must
    emit ``expected_legacy`` (it refits every model and stores the losers' final rows) and dag-ml
    must emit ``expected_dagml`` (operator-SELECT refits the winner only). For
    ``generator_or_models_pls_ridge`` that is legacy 34 / dag-ml 32 — the 2-entry gap being
    EXACTLY the one loser model's ``(test, final)`` + ``(train, final)`` refit rows. ANY other
    pair (a regression that adds/drops a row on EITHER engine) FAILS — so only the one documented
    +2 loser-final-row delta is allowed through.
    """
    assert legacy.num_predictions == expected_legacy, (
        f"{case.name}: documented LEGACY num_predictions changed — expected {expected_legacy}, "
        f"got {legacy.num_predictions} (the intentional divergence is pinned to the EXACT measured count; "
        "a new value is an unrelated regression, not the documented loser-refit-row delta)"
    )
    assert dagml.num_predictions == expected_dagml, (
        f"{case.name}: documented dag-ml num_predictions changed — expected {expected_dagml}, "
        f"got {dagml.num_predictions} (operator-SELECT must refit the WINNER ONLY; a new value is an "
        "unrelated regression, not the documented winner-only count)"
    )


def _close_or_both_nan(a: float, b: float, tol: float) -> bool:
    """Floats are equal for parity if both NaN (a no-score metric) or within ``tol``."""
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(abs(a - b) <= tol)


def assert_runresult_contract(legacy: Any, dagml: Any, case: PipelineCase, num_predictions_exempt: bool = False) -> None:
    """Assert the public ``RunResult`` surface matches across engines.

    Pins the contract both engines expose to the webapp / public API:

    * ``best_score`` — the SELECTED model's score under its OWN selection metric
      (``best["metric"]``: ``balanced_accuracy`` for classification, ``rmse`` for
      regression). This is THE load-bearing contract: a classification run can
      have an equal plain ``best_accuracy`` while the selected ``balanced_accuracy``
      differs (RF row-order divergence), so ``best_score`` is what must match.
    * ``best_rmse`` / ``best_r2`` (regression) for completeness.
    * the selected metric NAME, ``num_predictions``, and the top-n model set.

    Float scalars compare within the case's score tolerance (the same cross-engine
    noise ``assert_score_parity`` absorbs); ``_close_or_both_nan`` handles a
    no-score (NaN-on-both) run; the metric name + top-n model set match exactly.

    ``num_predictions_exempt`` (default ``False``) drops ONLY the ``num_predictions``
    equality check — every OTHER field (best_score / best_rmse / best_r2 / the
    selected-metric name / the top-n model set) is STILL asserted. Set it for a
    documented :data:`NUM_PREDICTIONS_DIVERGENCE` case whose num_predictions diverges
    BY DESIGN (operator-SELECT refits the winner only); the EXACT documented counts
    are asserted separately by :func:`assert_num_predictions_divergence`, so the count
    is never simply unchecked.
    """
    if not num_predictions_exempt:
        assert_num_predictions_parity(legacy, dagml)

    # Scalar float tolerance: the case's declared metric tolerances if any, else
    # the registry engine-parity standard. Independent of which metrics exist in
    # a given run — _close_or_both_nan absorbs the NaN-on-both no-score case.
    tol = max(case.metric_tolerances.values()) if case.metric_tolerances else _DEFAULT_SCORE_TOL

    # THE selection contract: best_score is the SELECTED model's score under its
    # own selection metric (balanced_accuracy for classification). Asserted for
    # every task — this is what surfaces a classification selection divergence the
    # plain best_accuracy would mask.
    assert _close_or_both_nan(legacy.best_score, dagml.best_score, tol), (
        f"{case.name}: SELECTED best_score ({legacy.best.get('metric')!r}) "
        f"legacy={legacy.best_score} dag-ml={dagml.best_score} (tol {tol})"
    )
    if case.task != "classification":
        assert _close_or_both_nan(legacy.best_rmse, dagml.best_rmse, tol), (
            f"{case.name}: best_rmse legacy={legacy.best_rmse} dag-ml={dagml.best_rmse} (tol {tol})"
        )
        assert _close_or_both_nan(legacy.best_r2, dagml.best_r2, tol), (
            f"{case.name}: best_r2 legacy={legacy.best_r2} dag-ml={dagml.best_r2} (tol {tol})"
        )

    # The selected metric name is identity, not a float — it must match exactly.
    assert legacy.best.get("metric") == dagml.best.get("metric"), (
        f"{case.name}: selected metric legacy={legacy.best.get('metric')!r} dag-ml={dagml.best.get('metric')!r}"
    )

    # top(n) model identities (the leaderboard the public API exposes) must agree
    # as a SET — both engines must surface the same model classes, order aside.
    n = 5
    legacy_models = {r.get("model_name") for r in legacy.top(n)}
    dagml_models = {r.get("model_name") for r in dagml.top(n)}
    assert legacy_models == dagml_models, (
        f"{case.name}: top({n}) models diverged: legacy={sorted(legacy_models)} dag-ml={sorted(dagml_models)}"
    )


def _final_test_pred_by_sample(result: Any, config_name: str | None = None) -> dict[int, np.ndarray]:
    """Map sample-id → final-(test) y_pred for a result, keyed by sample (not row).

    Matching by ``sample_indices`` (not row position) is the dag-ml invariant:
    joins are keyed by identity. Rows with no ``y_pred`` array or no
    ``sample_indices`` (score-only refit rows for shapes that collapse samples,
    e.g. aggregation) contribute nothing — the caller then sees no common
    samples and skips, which is correct.

    ``config_name`` scopes the map to ONE variant's final/test row: a multi-model
    ``{"model": {"_or_": [...]}}`` run that refits the winner ONLY on dag-ml but
    refits EVERY model on legacy has TWO final/test rows on legacy (winner +
    loser), and the loser's row would overwrite the winner's per sample (the map
    is keyed by sample-id, last-row-wins). Filtering to the winner's
    ``config_name`` compares the WINNER on both engines instead of legacy's loser
    vs dag-ml's winner. ``None`` (the default) keeps the all-rows behavior the
    standard parity path uses.
    """
    by_sample: dict[int, np.ndarray] = {}
    for row in result.predictions.filter_predictions(partition="test", fold_id="final"):
        if config_name is not None and row.get("config_name") != config_name:
            continue
        y_pred = row.get("y_pred")
        sample_indices = row.get("sample_indices")
        if y_pred is None or sample_indices is None:
            continue
        arr = np.asarray(y_pred, dtype=float)
        if arr.size == 0 or len(sample_indices) == 0:
            continue
        # Per-sample rows: y_pred is (n_samples,) or (n_samples, n_targets); index by row.
        flat = arr.reshape(len(sample_indices), -1) if arr.ndim > 1 else arr.reshape(-1, 1)
        if flat.shape[0] != len(sample_indices):
            continue
        for sid, vec in zip(sample_indices, flat):
            by_sample[int(sid)] = vec
    return by_sample


def assert_y_pred_parity(legacy: Any, dagml: Any, case: PipelineCase, tol: float = _DEFAULT_YPRED_TOL) -> None:
    """Assert per-sample final-(test) ``y_pred`` matches across engines, by sample id.

    Both engines' final/test predictions are mapped sample-id → vector. The
    sample-id SETS must be EQUAL when either engine emitted per-sample
    predictions: comparing only the intersection would let an engine that drops
    final/test predictions for some samples pass silently, so a set mismatch is a
    FAILURE (the engines disagree on which samples they predict). Only when
    BOTH maps are empty (a sample-collapsing aggregation refit with no per-sample
    y_pred on either side) is y_pred parity not applicable — then score parity
    covers it. Default ``tol`` is the measured cross-engine PLS+inverse noise
    ceiling (~6e-4) rounded to ``1e-3``.
    """
    legacy_map = _final_test_pred_by_sample(legacy)
    dagml_map = _final_test_pred_by_sample(dagml)
    if not legacy_map and not dagml_map:
        # No per-sample y_pred on EITHER side (sample-collapsing aggregation refit).
        # y_pred parity is not applicable here — score parity covers it.
        return

    legacy_ids, dagml_ids = set(legacy_map), set(dagml_map)
    assert legacy_ids == dagml_ids, (
        f"{case.name}: final-(test) y_pred sample-id SETS diverge — "
        f"legacy-only={sorted(legacy_ids - dagml_ids)[:10]} dag-ml-only={sorted(dagml_ids - legacy_ids)[:10]} "
        f"(|legacy|={len(legacy_ids)} |dag-ml|={len(dagml_ids)})"
    )

    common = sorted(legacy_ids)
    deltas = np.array([np.max(np.abs(legacy_map[s] - dagml_map[s])) for s in common])
    max_delta = float(deltas.max())
    assert max_delta <= tol, (
        f"{case.name}: per-sample y_pred parity violated on {int((deltas > tol).sum())}/{len(common)} samples; "
        f"max |Δy_pred| = {max_delta:.3e} > tol {tol:.3e}"
    )


def assert_winner_y_pred_parity(legacy: Any, dagml: Any, case: PipelineCase, tol: float = _DEFAULT_YPRED_TOL) -> None:
    """Assert the WINNER's per-sample final-(test) ``y_pred`` matches across engines, by sample id.

    The :data:`NUM_PREDICTIONS_DIVERGENCE` companion to :func:`assert_y_pred_parity`.
    A multi-model ``{"model": {"_or_": [...]}}`` run refits the WINNER ONLY on dag-ml
    (the correct SELECT semantic) but refits EVERY model on legacy, so legacy carries a
    SECOND final/test row (the loser) that, keyed by sample-id (last-row-wins),
    overwrites the winner's per-sample prediction in the all-rows map — making the plain
    :func:`assert_y_pred_parity` compare legacy's LOSER against dag-ml's WINNER (a
    spurious Δ≈2e1, PLS vs Ridge). Scoping BOTH engines' maps to the WINNER's
    ``config_name`` (identical on both, locked by :func:`assert_same_winner`) compares
    winner-vs-winner — which DOES match (measured Δ=0.0). Only the prediction COUNT and
    the loser's stored refit row diverge, never the winner's predictions.
    """
    winner_cfg = dagml.best.get("config_name")
    assert winner_cfg, f"{case.name}: dag-ml winner has no config_name to scope the winner y_pred comparison"
    legacy_map = _final_test_pred_by_sample(legacy, config_name=winner_cfg)
    dagml_map = _final_test_pred_by_sample(dagml, config_name=winner_cfg)

    legacy_ids, dagml_ids = set(legacy_map), set(dagml_map)
    assert legacy_ids and legacy_ids == dagml_ids, (
        f"{case.name}: WINNER ({winner_cfg!r}) final-(test) y_pred sample-id SETS diverge — "
        f"legacy-only={sorted(legacy_ids - dagml_ids)[:10]} dag-ml-only={sorted(dagml_ids - legacy_ids)[:10]} "
        f"(|legacy|={len(legacy_ids)} |dag-ml|={len(dagml_ids)})"
    )

    common = sorted(legacy_ids)
    deltas = np.array([np.max(np.abs(legacy_map[s] - dagml_map[s])) for s in common])
    max_delta = float(deltas.max())
    assert max_delta <= tol, (
        f"{case.name}: WINNER per-sample y_pred parity violated on {int((deltas > tol).sum())}/{len(common)} samples; "
        f"max |Δy_pred| = {max_delta:.3e} > tol {tol:.3e}"
    )


def assert_same_winner(legacy: Any, dagml: Any, case: PipelineCase) -> None:
    """Assert both engines selected the SAME winning variant (by ``config_name``).

    Load-bearing guard for a relaxed-tolerance case: a relaxed per-sample y_pred
    tolerance is only justified when the engines agree on WHICH variant won (same
    config → same model shape, the delta is pure cross-engine float noise). If the
    winners differ, the relaxed tolerance would mask a real SELECTION divergence,
    so this must fail first. ``config_name`` is the stable cross-engine winner id
    (workspace-only fields like ``config_path`` / ``pipeline_uid`` are empty on a
    dag-ml run and must not be used).
    """
    legacy_cfg = legacy.best.get("config_name")
    dagml_cfg = dagml.best.get("config_name")
    assert legacy_cfg and legacy_cfg == dagml_cfg, (
        f"{case.name}: engines selected DIFFERENT winners — legacy config_name={legacy_cfg!r} "
        f"dag-ml config_name={dagml_cfg!r}; a relaxed y_pred tolerance is not justified across a selection split"
    )


def round_trip_export_reload_predict(result: Any, x_predict: np.ndarray, output_path: Any) -> np.ndarray:
    """Export the run's model, reload it, and predict on ``x_predict``.

    Uses ``RunResult.export_model`` (the lightweight single-artifact export) →
    ``joblib.load`` → ``predict``. For a native single-model dag-ml run this
    exports the captured REFIT artifact directly (P3); for a multi-model dag-ml
    run or a legacy run it round-trips via the legacy export path. Returns the
    reloaded model's predictions, ravelled.
    """
    import joblib

    path = result.export_model(output_path)
    loaded = joblib.load(path)
    return np.asarray(loaded.predict(x_predict), dtype=float).ravel()
