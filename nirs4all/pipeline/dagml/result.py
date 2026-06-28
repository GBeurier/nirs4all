"""Map a dag-ml ScoreSet into a nirs4all ``RunResult`` for the dag-ml backend.

dag-ml computes per-fold validation / cross-fold OOF / final-test scores natively (in Rust); this
turns its ``bundle.scores`` into an in-memory :class:`~nirs4all.data.predictions.Predictions`
wrapped in a :class:`~nirs4all.api.result.RunResult`.

The native ScoreSet is deliberately COMPACT — one report per ``(partition, fold_id)``: the per-fold
``(validation, foldN)`` OOF scores, the cross-fold ``(validation, avg)`` OOF average, and the refit
``(final, None)`` / ``(test, None)`` train/test scores. The legacy nirs4all ``Predictions`` table the
0.9.x webapp consumes is WIDER: per fold it stores a ``train``/``val``/``test`` row, it carries BOTH a
fold-ensemble ``avg`` and a weighted-average ``w_avg`` block (each train/val/test), and a refit
``final`` block (train + test). For a KFold(3) run that is 3·3 + 3 + 3 + 2 = 17 entries vs the native
2 this module used to surface.

:func:`_scores_to_run_result` therefore emits the FULL legacy table as a **labeled compatibility
PROJECTION** over the native reports — not a hardcoded count. Each emitted row carries an explicit
ROLE (``fold_id`` ∈ {a fold key, ``"avg"``, ``"w_avg"``, ``"final"``} × ``partition`` ∈
{train, val, test} + ``refit_context``), so it generalizes by row-role to ``n_folds`` (per-fold rows
scale with the native validation folds), single-split CV (no native ``avg`` ⇒ no avg/w_avg block, the
legacy KennardStone/SPXY = 5-row shape), classification (the native metric block already carries
``accuracy``), repeated-CV / multi-target (more validation-fold reports / wider metric blocks flow
through unchanged), and a missing-test partition (no native ``(test, None)`` ⇒ no test rows, the final
block degrades to train-only).

Native reports only score the partition each role natively owns (validation per fold, train/test for
the refit). Each emitted row carries a full ``scores`` dict whose partitions are FILLED so the
``Predictions`` ranking (``get_best`` / ``top`` / ``score_scope``) resolves the SAME entries as legacy
and every partition-keyed accessor reads a true native value:

* the ``val`` (and ``train``) keys carry the role's OWN block — a fold's OOF for a fold row, the
  cross-fold OOF average for an ``avg`` / ``w_avg`` / ``final`` row — so ranking on ``val`` selects the
  best-val fold / the ensemble exactly as legacy does;
* the ``test`` key carries THAT VARIANT's OWN held-out ``(test, None)`` block (looked up by its
  ``variant_id``), never another variant's. A single pipeline / the selected winner carries the held-out
  test scored at refit (or the reassembled-merge test); in a generator SWEEP each variant carries its own
  held-out test — operator-expanded variants ran fully so they have a real one, while a NATIVE loser that
  never refit gets a NULL test (the row is still emitted for count parity). Each row's ``test`` key holds
  that row's own native test, so ``top`` / ``get_best`` / ``score_scope`` rank the full per-variant table
  correctly; the partition-keyed scalar shortcuts ``best_rmse`` / ``best_r2`` / ``best_accuracy`` all read
  the SELECTED model's test metric via :meth:`RunResult._selected_metric` — the same model ``best_score``
  describes — so the four scalars stay mutually consistent.

``cv_best_score`` stays the native cross-fold OOF average (the winner ``avg`` row's ``val`` score), preserved
exactly; ``best`` / ``best_final`` resolve to the refit ``final`` through the refit-only
``score_scope="refit"`` ranking (winner-only). A merge producer (separation / fusion / stacking) emits no
refit-train report, so it has no ``final`` row — its ``test`` block is the reassembled merge test.
"""

from __future__ import annotations

from typing import Any

from nirs4all.api.result import RunResult
from nirs4all.data.predictions import Predictions

# The three CV partitions every fold-grain / ensemble role stores in the legacy table.
_CV_PARTITIONS = ("train", "val", "test")

# Sentinel distinguishing "no refit report at all" (a merge producer never refits) from a refit report
# whose variant_id is legitimately ``None`` — both would otherwise read as ``None``.
_MISSING = object()


def _legacy_fold_id(native_fold_id: str) -> str:
    """Map a native ``foldN`` validation-fold id to the legacy integer-string ``N``.

    dag-ml labels its OOF folds ``fold0`` / ``fold1`` / …; the legacy ``Predictions`` table (and the
    webapp / ``PredictionAggregator`` that key on it) uses the bare zero-based index ``"0"`` / ``"1"`` /
    …. Any non-``foldN`` id (defensive) is returned unchanged.
    """
    return native_fold_id[len("fold"):] if native_fold_id.startswith("fold") and native_fold_id[len("fold"):].isdigit() else native_fold_id


def _native_variant_config_map(scores: dict[str, Any] | None, ordered_config_names: list[str]) -> dict[Any, str]:
    """Pair a NATIVE-generation ScoreSet's CV variant ids with the ordered legacy expand config names.

    Native generation emits OPAQUE ``variant_id`` hashes with NO params in the reports, so a variant
    cannot be matched to its param value (hence its specific legacy ``config_name``) by content. dag-ml
    lays the reports out WINNER-first (the SELECTED variant's reports lead, from the real winner-only
    FIT_CV), then the losers in generation/expand order. ``ordered_config_names`` is
    ``PipelineConfigs.names`` in expand order, where index 0 is the config legacy refits for a native
    param sweep (legacy's native variants are degenerate — same score — so it selects the first). We
    therefore pair the winner with ``names[0]`` (so the dag-ml winner's ``_refit`` matches legacy's
    index-0 refit) and the losers with the rest in order. This gives the exact legacy SET + count of
    config names with a legacy-aligned winner label; the per-loser hash↔variant pairing is positional
    (param recovery is impossible from the native reports). Returns ``{fold_variant_id: config_name}``
    keyed by the variant's own (non-``None``) fold-level id — the avg's native ``None`` tag is not a key.
    """
    cv_variant_ids = list(
        dict.fromkeys(
            report.get("variant_id")
            for report in (scores or {}).get("reports", [])
            if report["partition"] == "validation" and report.get("fold_id") != "avg"
        )
    )
    return dict(zip(cv_variant_ids, ordered_config_names, strict=False))


def _scores_to_run_result(
    scores: dict[str, Any] | None,
    dataset_name: str,
    model_name: str,
    metric: str = "rmse",
    task_type: str = "regression",
    producer: str | None = None,
    config_name: str = "",
    variant_config_names: dict[Any, str] | None = None,
    variant_model_names: dict[Any, str] | None = None,
    skip_refit: bool = False,
) -> RunResult:
    """Project a dag-ml ScoreSet into the full legacy ``Predictions`` table (a labeled compat projection).

    ``producer`` filters to one ``producer_node`` — e.g. a separation/duplication/stacking merge node,
    whose reports carry the full-universe OOF average (``cv_best_score``) and — since dag-ml routes the
    base branches' REFIT-test predictions to the merge node (``reassemble_branch_merge_off_fold`` for
    concat/fusion; the ``:refit`` off-fold input for stacking) — a reassembled ``(test, fold_id=None)``
    block (``best_rmse``); ``None`` keeps all reports (the single-model path, where exactly one producer
    scores).

    PER-VARIANT projection (#55). Since dag-ml native generation surfaces EVERY variant's validation
    reports (each stamped with a distinct ``variant_id``; the winner's are re-tagged ``None`` on the
    cross-fold ``avg`` and carry the winner's id on the per-fold rows + the refit ``(final/test, None)``),
    this groups by ``(variant_id, partition, fold_id)`` and emits the legacy 15-row CV block (per-fold
    {train,val[,test]} + avg + w_avg) FOR EACH variant — so a generator sweep's ``num_predictions``
    matches legacy (N·15 + 2). Only the WINNER (the variant owning the refit ``(final, None)``; the only
    variant dag-ml refits) additionally gets the ``(final,train)`` + ``(final,test)`` standalone-refit
    rows. A single (non-sweep) pipeline has exactly ONE CV variant (``variant:base``) and its output is
    byte-identical to the pre-#55 winner-only projection.

    ``config_name`` is the CANONICAL legacy config name already DERIVED upstream
    (:func:`~nirs4all.pipeline.dagml.run_backend._derive_config_name` via ``PipelineConfigs``):
    ``"config_{hash}"`` for an unnamed run, ``"{name}_p0_{hash}"`` for a named one, or ``""`` for a
    generator pipeline. For a SWEEP, ``variant_config_names`` / ``variant_model_names`` map each CV
    variant's fold-level ``variant_id`` → its legacy ``config_name`` / ``model_name``, so a variant is
    labeled by its OWN identity (NOT its loop / iteration position) — the WINNER (the variant owning the
    refit ``(final, None)``) is looked up by its own id and gets the ``"_refit"`` suffix on its refit
    rows. The caller builds the maps from the SAME ``PipelineConfigs`` mechanism legacy uses, so a sweep
    carries the legacy config names AND model names with a legacy-correct winner label. A variant absent
    from the map falls back to the scalar ``config_name`` / ``model_name``. A single pipeline (no maps)
    applies ``config_name`` / ``model_name`` verbatim, byte-identical to legacy.

    The emitted rows are keyed on ROLE (see the module docstring), so the projection adapts to the
    pipeline shape rather than hardcoding a count: ``n_folds`` per-fold rows, an ``avg``/``w_avg``
    ensemble block only when dag-ml concatenated multiple folds, and a refit ``final`` block whose
    partitions follow the native refit reports. The ``test`` partition is emitted on EVERY role only
    when the run has a held-out test partition (a native ``(test, None)`` report) — a no-test dataset
    drops every ``test`` row, the legacy train+val-only shape.

    ``skip_refit`` REPLICATES the legacy refit gate (see
    :func:`~nirs4all.pipeline.dagml.steps._legacy_skips_refit`): when the pipeline's splitter serializes
    to a bare class string (all-default params), legacy's ``execute_simple_refit`` does not recognize it
    and SKIPS the refit, emitting NO ``(final, train)`` / ``(final, test)`` rows (e.g. ``KFold(n_splits=5)``
    → 21 rows, not 23). dag-ml's native bundle ALWAYS refits, so when ``skip_refit`` is set we suppress
    JUST the standalone-refit ``final`` rows from the projection (the per-fold / avg / w_avg test rows are
    unaffected — those come from the dataset's own held-out test partition, which legacy still scores).
    ``cv_best_score`` (the OOF avg) is unchanged; with no ``final`` row, ``best`` / ``best_score`` (and the
    ``best_rmse`` / ``best_r2`` / ``best_accuracy`` shortcuts, which all read the selected entry via
    :meth:`RunResult._selected_metric`) fall back to ``cv_best`` exactly as legacy does with no refit.
    """
    reports = [report for report in (scores or {}).get("reports", []) if producer is None or report.get("producer_node") == producer]
    # Key on (variant_id, partition, fold_id): native generation surfaces every variant's reports and
    # ScoreSet.validate guarantees this triple (per producer) is unique, so distinct variants never
    # collide (the pre-#55 (partition, fold_id) key let later variants overwrite earlier ones).
    by_key = {(report.get("variant_id"), report["partition"], report.get("fold_id")): {name: float(value) for name, value in report["metrics"].items()} for report in reports}

    predictions = Predictions()

    def add(fold_id: str, partition: str, partition_blocks: dict[str, dict[str, float] | None], *, row_config_name: str, row_model_name: str, refit_context: str | None = None) -> None:
        """Emit one legacy row for a (role, partition) with its full train/val/test ``scores`` dict.

        ``partition_blocks`` maps each of train/val/test to its native metric block (``None`` skips that
        partition key). The legacy scalar ``{val,test,train}_score`` fields mirror the matching block's
        ``metric`` value so the partition-keyed accessors resolve them. The native blocks are shared
        verbatim — no metric is perturbed — so every reported value (and every accessor that reads it)
        is the true native score.

        ``row_config_name`` is the variant's config name (already ``"_refit"``-suffixed by the caller for
        the standalone-refit rows, matching legacy ``"{cv_config_name}_refit"`` — see
        ``execution.refit.executor``); ``row_model_name`` is the variant's model label (a multi-MODEL
        sweep carries a different model per variant — the winner's final/best rows must show ITS model).
        """
        score_dict: dict[str, dict[str, float]] = {}
        kwargs: dict[str, Any] = {}
        for part in _CV_PARTITIONS:
            block = partition_blocks.get(part)
            if block is None:
                continue
            score_dict[part] = block
            kwargs[f"{part}_score"] = block.get(metric)
        predictions.add_prediction(
            dataset_name=dataset_name,
            config_name=row_config_name,
            model_name=row_model_name,
            fold_id=fold_id,
            partition=partition,
            metric=metric,
            task_type=task_type,
            scores=score_dict,
            refit_context=refit_context,
            **kwargs,
        )

    # The refit-train / held-out test blocks are looked up PER-VARIANT below (`(variant_id, final/test,
    # None)`), NOT once globally — a LOSER must NOT borrow the winner's test/train under its own
    # config_name. The winner owns the refit `(final, None)`; only the winner refits, so it is the sole
    # variant with a `(final, None)` / `(test, None)` report in a native sweep. (In an operator sweep each
    # variant ran fully and HAS its own test report, kept and re-tagged by `_project_operator_sweep`.)
    final_variant_id = next((variant_id for (variant_id, partition, fold_id) in by_key if partition == "final" and fold_id is None), _MISSING)

    # Whether the RUN has a held-out test partition at all — true when ANY variant reported a `(test,
    # None)` block. This decides whether test ROWS are emitted per role (count parity); a variant WITHOUT
    # its own test block still emits the test rows but with a NULL test score (it never borrows another
    # variant's). A no-test dataset drops every test row (the legacy train+val shape).
    run_has_test = any(partition == "test" and fold_id is None for (variant_id, partition, fold_id) in by_key)
    cv_partitions = _CV_PARTITIONS if run_has_test else ("train", "val")

    # Order the CV variants WINNER-first (the variant owning the refit `(final, None)`; its cross-fold
    # avg is the native `None`-tagged row), then the losers in the order dag-ml emitted them. A single
    # (non-sweep) pipeline has exactly one CV variant — `variant:base`, which is also the winner — so the
    # loop runs once and reproduces the pre-#55 winner-only shape. The winner's avg key is the `None`
    # variant_id; each loser's avg key is its own variant_id.
    cv_variant_ids = list(dict.fromkeys(variant_id for (variant_id, partition, fold_id) in by_key if partition == "validation" and fold_id != "avg"))
    if final_variant_id is not _MISSING and final_variant_id in cv_variant_ids:
        cv_variant_ids = [final_variant_id] + [variant_id for variant_id in cv_variant_ids if variant_id != final_variant_id]

    for variant_id in cv_variant_ids:
        is_winner = variant_id == final_variant_id
        # Per-variant config name / model name, looked up by the variant's OWN id (NOT its loop position),
        # so a sweep labels each variant by its own identity and the winner is labeled correctly even when
        # it is not the first expanded variant. A variant absent from the map (or a single non-sweep run)
        # falls back to the scalar `config_name` / `model_name`. The winner's standalone-refit rows get the
        # `_refit` suffix (skipped when the name is empty so we never emit a bare `"_refit"`).
        variant_config_name = variant_config_names.get(variant_id, config_name) if variant_config_names is not None else config_name
        variant_model_name = variant_model_names.get(variant_id, model_name) if variant_model_names is not None else model_name

        # The native validation folds for THIS variant, in dag-ml's emitted order (foldN, excluding avg).
        fold_keys = [fold_id for (other_variant_id, partition, fold_id) in by_key if other_variant_id == variant_id and partition == "validation" and fold_id != "avg"]
        # The cross-fold OOF average for THIS variant. dag-ml emits the avg with `variant_id = None` for
        # the SOLE producer (a single concrete pipeline or a merge node) and for the SWEEP WINNER; a sweep
        # LOSER's avg carries its own variant_id. So the `None`-tagged avg belongs to the winner, or to the
        # lone variant when there is only one. A single-fold splitter (KennardStone/SPXY, n_splits=1) emits
        # NO "avg" — just the one validation fold — so there is no ensemble block (the legacy 5-row
        # single-split shape) and the lone fold's OOF is the CV score.
        avg_variant_id = None if (is_winner or len(cv_variant_ids) == 1) else variant_id
        has_avg = by_key.get((avg_variant_id, "validation", "avg")) is not None
        avg = by_key.get((avg_variant_id, "validation", "avg"))
        if avg is None and len(fold_keys) == 1:
            avg = by_key[(variant_id, "validation", fold_keys[0])]

        # THIS variant's OWN held-out test + refit-train blocks (never another variant's). The winner /
        # sole producer has both; a native sweep LOSER never refits/tests, so both are None → its test &
        # ensemble-train rows carry a NULL score (the row is still emitted for count parity). An operator
        # sweep LOSER kept its own `(test, None)` / `(final, None)` reports (re-tagged by
        # `_project_operator_sweep`), so it carries its REAL test value.
        variant_test = by_key.get((variant_id, "test", None))
        variant_final_train = by_key.get((variant_id, "final", None))
        is_final_owner = is_winner or len(cv_variant_ids) == 1

        # --- Per-fold rows: (foldN, {train, val[, test]}). dag-ml scores only VALIDATION per fold, so the
        #     fold's own OOF block sits under train+val; the `test` key carries THIS variant's own held-out
        #     `(test, None)` block (None for a native loser → null test score, never the winner's). The
        #     native `foldN` id is normalized to the legacy integer-string `N` the webapp /
        #     PredictionAggregator key on. ---
        for fold_id in fold_keys:
            fold_block = by_key[(variant_id, "validation", fold_id)]
            fold_blocks: dict[str, dict[str, float] | None] = {"train": fold_block, "val": fold_block, "test": variant_test}
            for partition in cv_partitions:
                add(_legacy_fold_id(fold_id), partition, fold_blocks, row_config_name=variant_config_name, row_model_name=variant_model_name)

        # --- Ensemble rows: avg + w_avg, each over {train, val[, test]}. Emitted only when dag-ml produced
        #     an OOF average over multiple folds (the legacy avg/w_avg blocks). val carries THIS variant's
        #     true OOF average; train carries THIS variant's own refit-train (winner / sole producer) else
        #     its own avg (a loser has no refit, so the ensemble-train proxy is its own OOF average, NOT the
        #     winner's final-train); test carries THIS variant's own held-out test (None → null for a native
        #     loser). Matching legacy, where avg.val == w_avg.val == cv_best_score. ---
        if has_avg and avg is not None:
            ensemble_blocks: dict[str, dict[str, float] | None] = {"train": variant_final_train or avg, "val": avg, "test": variant_test}
            for fold_id in ("avg", "w_avg"):
                for partition in cv_partitions:
                    add(fold_id, partition, ensemble_blocks, row_config_name=variant_config_name, row_model_name=variant_model_name)

        # --- Refit final rows: (final, train) + (final, test), refit_context="standalone". Emitted only
        #     for the FINAL OWNER — the sweep WINNER (the one variant dag-ml refits), or the sole producer
        #     (a single concrete pipeline / a merge node) when there is just one variant. A sweep LOSER
        #     never refits, so it gets no final block. Each row carries THIS variant's refit train/test
        #     blocks plus its OOF average under val (so it ranks on the same CV axis as the avg).
        #     `best`/`best_final` resolve to a final via score_scope="refit" (refit-only ranking). Degrades
        #     to train-only with no test, or (a merge node, no refit-train report) to test-only. The
        #     `_refit` suffix is skipped for an empty config name. SUPPRESSED entirely when `skip_refit` is
        #     set — the legacy gate did not refit (bare-string splitter), so NO `(final, *)` rows. ---
        if is_final_owner and not skip_refit:
            refit_config_name = variant_config_name + "_refit" if variant_config_name else variant_config_name
            final_blocks: dict[str, dict[str, float] | None] = {"train": variant_final_train, "val": avg, "test": variant_test}
            if variant_final_train is not None:
                add("final", "train", final_blocks, row_config_name=refit_config_name, row_model_name=variant_model_name, refit_context="standalone")
            if variant_test is not None:
                add("final", "test", final_blocks, row_config_name=refit_config_name, row_model_name=variant_model_name, refit_context="standalone")

    predictions.flush()
    return RunResult(predictions=predictions, per_dataset={dataset_name: {"engine": "dag-ml"}})


def _variant_cv_score(scores: dict[str, Any], metric: str) -> float:
    """The variant's cross-fold OOF ``metric`` — for SELECT, in the metric's OWN units / direction.

    Reads the REQUESTED ``metric`` (``accuracy`` for classification, ``rmse`` for regression) — NOT a
    hardcoded ``rmse`` (which is absent / meaningless for a classification sweep, so every variant would
    tie NaN and the first would win regardless). The score is the cross-fold ``(validation, avg)`` report
    (the single concrete path emits exactly one, ``variant_id`` ``None``); when there is NO avg (a
    single-split splitter — KennardStone / SPXY ``n_splits=1`` — emits just the one validation fold), it
    falls back to that SOLE ``(validation, fold)`` report, so a no-avg sweep still ranks on a real CV
    score instead of defaulting to variant 0. Returns NaN only when the run produced no validation score.
    """
    avg = next((report for report in scores.get("reports", []) if report.get("partition") == "validation" and report.get("fold_id") == "avg"), None)
    if avg is None:
        avg = next((report for report in scores.get("reports", []) if report.get("partition") == "validation" and report.get("fold_id") != "avg"), None)
    return float(avg["metrics"].get(metric, float("nan"))) if avg is not None else float("nan")


def _project_operator_sweep(
    variant_scores: list[tuple[dict[str, Any], str, bool]],
    dataset_name: str,
    metric: str,
    task_type: str,
    is_classification: bool,
    variant_config_names: list[str],
) -> RunResult:
    """Combine each operator-expanded variant's single-variant ScoreSet into ONE per-variant projection.

    Each variant ran as its own concrete dag-ml run, so its ScoreSet carries the native single-producer
    tagging (``variant:base`` on the per-fold / refit reports, ``variant_id = None`` on the cross-fold
    ``avg``). To feed the shared per-variant projection (:func:`_scores_to_run_result`), we synthesize the
    multi-variant ScoreSet that native generation would have emitted: the CV WINNER (best ``metric`` —
    accuracy MAXIMIZED for classification, rmse MINIMIZED for regression, matching nirs4all's get_best /
    the native SELECT direction) keeps ALL its reports (incl. the refit ``(final/test, None)`` and the
    ``None``-tagged avg — only the winner refits); every LOSER keeps only its VALIDATION reports, re-tagged
    with a distinct ``variant_id`` (its avg re-tagged too, so it no longer claims the winner's ``None`` avg).

    Each variant's rows are labeled by its OWN ``variant_id`` via the ``variant_config_names`` /
    ``variant_model_names`` maps — the winner gets ITS expansion ``config_name`` (+ ``_refit``) and ITS
    ``model_name`` (a multi-MODEL ``_or_`` sweep carries a different model per variant; the winner's
    final/best rows must show the WINNING model, not the first variant's).
    """
    scores_by_variant = [scores for scores, _, _ in variant_scores]
    model_names = [name for _, name, _ in variant_scores]
    cv_scores = [_variant_cv_score(scores, metric) for scores in scores_by_variant]

    def _rank(index: int) -> float:
        score = cv_scores[index]
        if score != score:  # NaN ranks last
            return float("inf")
        return -score if is_classification else score  # maximize accuracy, minimize RMSE

    winner_index = min(range(len(scores_by_variant)), key=_rank)
    # The legacy refit gate is the WINNER's, not `any`: a splitter GENERATOR (e.g.
    # ``{"_or_": [KFold(n_splits=5), KFold(n_splits=3)]}``) yields per-variant gates — the all-default
    # variant serializes to a bare string (legacy skips its refit) while the non-default one refits — so
    # ONLY the selected variant's gate decides whether the standalone ``(final, *)`` rows are emitted.
    skip_refit = variant_scores[winner_index][2]
    # Winner first, then the losers in expand order — the projection iterates / labels by variant_id, so
    # the order here only sets which reports lead, not the labels.
    ordered_indices = [winner_index] + [index for index in range(len(scores_by_variant)) if index != winner_index]

    combined_reports: list[dict[str, Any]] = []
    variant_config_map: dict[Any, str] = {}
    variant_model_map: dict[Any, str] = {}
    for position, index in enumerate(ordered_indices):
        is_winner = position == 0
        variant_tag = "variant:base" if is_winner else f"variant:v{index}"
        # Label this variant's rows with ITS expansion config_name + model_name (by the fold-level id the
        # projection keys on). The winner's `(final, None)` carries `variant:base`, so the winner's refit
        # rows resolve to the winner's config_name (+ `_refit`) and model_name too.
        if variant_config_names:
            variant_config_map[variant_tag] = variant_config_names[index]
        variant_model_map[variant_tag] = model_names[index]
        for report in scores_by_variant[index].get("reports", []):
            partition, fold_id = report.get("partition"), report.get("fold_id")
            entry = dict(report)
            # Re-tag by variant_id so each variant's rows are distinct AND each variant carries its OWN
            # test / refit-train (a LOSER must show ITS real held-out test + refit-train under ITS
            # config_name, never the winner's). Every variant ran fully via _run_concrete here, so each has
            # its own `(final, None)` / `(test, None)`; we KEEP them, re-tagged, so the projection scopes
            # them per-variant. The winner keeps the native single-producer tagging (`variant:base` on
            # folds/final/test, the avg's native `None`); a loser is re-tagged on ALL its reports (its avg
            # too, so it no longer claims the winner's `None` avg). The projection still emits the
            # standalone `_refit` rows for the WINNER only — a loser's `(final, *)` feeds only its
            # ensemble-train block, not standalone-refit rows.
            if is_winner:
                entry["variant_id"] = None if (partition == "validation" and fold_id == "avg") else variant_tag
            else:
                entry["variant_id"] = variant_tag
            combined_reports.append(entry)

    combined_scores = {**scores_by_variant[winner_index], "reports": combined_reports}
    return _scores_to_run_result(
        combined_scores,
        dataset_name,
        model_names[winner_index],
        metric,
        task_type,
        variant_config_names=variant_config_map or None,
        variant_model_names=variant_model_map,
        skip_refit=skip_refit,
    )
