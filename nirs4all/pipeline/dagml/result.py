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
* the ``test`` key on EVERY role carries the SAME shared held-out ``(test, None)`` block. dag-ml scores
  the held-out test only once (the refit / reassembled-merge test), so every role shares it — which makes
  the partition-keyed test accessors (``best_rmse`` / ``best_r2`` / ``best_accuracy`` all read
  ``scores["test"][metric]``) return the true held-out value REGARDLESS of which ``val`` row wins the
  rank. That mirrors legacy: ranking picks the best-val row, the displayed test metric is the held-out test.

``cv_best_score`` stays the native cross-fold OOF average (the ``avg`` row's ``val`` score) and
``best_rmse`` / ``best_r2`` / ``best_accuracy`` the native held-out test — both preserved exactly, no
metric is perturbed; ``best`` / ``best_final`` resolve to the refit ``final`` through the refit-only
``score_scope="refit"`` ranking. A merge producer (separation / fusion / stacking) emits no refit-train
report, so it has no ``final`` row — its ``test`` block is the reassembled merge test, shared identically.
"""

from __future__ import annotations

from typing import Any

from nirs4all.api.result import RunResult
from nirs4all.data.predictions import Predictions

# The three CV partitions every fold-grain / ensemble role stores in the legacy table.
_CV_PARTITIONS = ("train", "val", "test")


def _legacy_fold_id(native_fold_id: str) -> str:
    """Map a native ``foldN`` validation-fold id to the legacy integer-string ``N``.

    dag-ml labels its OOF folds ``fold0`` / ``fold1`` / …; the legacy ``Predictions`` table (and the
    webapp / ``PredictionAggregator`` that key on it) uses the bare zero-based index ``"0"`` / ``"1"`` /
    …. Any non-``foldN`` id (defensive) is returned unchanged.
    """
    return native_fold_id[len("fold"):] if native_fold_id.startswith("fold") and native_fold_id[len("fold"):].isdigit() else native_fold_id


def _scores_to_run_result(scores: dict[str, Any] | None, dataset_name: str, model_name: str, metric: str = "rmse", task_type: str = "regression", producer: str | None = None) -> RunResult:
    """Project a dag-ml ScoreSet into the full legacy ``Predictions`` table (a labeled compat projection).

    ``producer`` filters to one ``producer_node`` — e.g. a separation/duplication/stacking merge node,
    whose reports carry the full-universe OOF average (``cv_best_score``) and — since dag-ml routes the
    base branches' REFIT-test predictions to the merge node (``reassemble_branch_merge_off_fold`` for
    concat/fusion; the ``:refit`` off-fold input for stacking) — a reassembled ``(test, fold_id=None)``
    block (``best_rmse``); ``None`` keeps all reports (the single-model path, where exactly one producer
    scores).

    The emitted rows are keyed on ROLE (see the module docstring), so the projection adapts to the
    pipeline shape rather than hardcoding a count: ``n_folds`` per-fold rows, an ``avg``/``w_avg``
    ensemble block only when dag-ml concatenated multiple folds, and a refit ``final`` block whose
    partitions follow the native refit reports. The ``test`` partition is emitted on EVERY role only
    when the run has a held-out test partition (a native ``(test, None)`` report) — a no-test dataset
    drops every ``test`` row, the legacy train+val-only shape.
    """
    reports = [report for report in (scores or {}).get("reports", []) if producer is None or report.get("producer_node") == producer]
    by_key = {(report["partition"], report.get("fold_id")): {name: float(value) for name, value in report["metrics"].items()} for report in reports}

    predictions = Predictions()

    def add(fold_id: str, partition: str, partition_blocks: dict[str, dict[str, float] | None], *, refit_context: str | None = None) -> None:
        """Emit one legacy row for a (role, partition) with its full train/val/test ``scores`` dict.

        ``partition_blocks`` maps each of train/val/test to its native metric block (``None`` skips that
        partition key). The legacy scalar ``{val,test,train}_score`` fields mirror the matching block's
        ``metric`` value so the partition-keyed accessors resolve them. The native blocks are shared
        verbatim — no metric is perturbed — so every reported value (and every accessor that reads it)
        is the true native score.
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
            model_name=model_name,
            fold_id=fold_id,
            partition=partition,
            metric=metric,
            task_type=task_type,
            scores=score_dict,
            refit_context=refit_context,
            **kwargs,
        )

    # Native validation folds, in the order dag-ml emitted them (foldN keys, excluding the avg).
    fold_keys = [fold_id for (partition, fold_id) in by_key if partition == "validation" and fold_id != "avg"]
    # The cross-fold OOF average ("avg") when dag-ml concatenated multiple folds. A single-fold splitter
    # (KennardStone/SPXY, n_splits=1) emits NO "avg" — just the one validation fold — so there is no
    # ensemble block (the legacy 5-row single-split shape), and the lone fold's OOF is the CV score.
    has_avg = by_key.get(("validation", "avg")) is not None
    avg = by_key.get(("validation", "avg"))
    if avg is None and len(fold_keys) == 1:
        avg = by_key[("validation", fold_keys[0])]
    # The refit train / held-out test blocks: the producer's `(final, None)` and `(test, None)` reports
    # (the off-fold convention the node runner emits for the REFIT, and that the native
    # concat/fusion/stacking merge handlers reassemble under the merge producer). `producer` already
    # scopes `by_key` to the right node.
    final_train = by_key.get(("final", None))
    test = by_key.get(("test", None))

    # The partitions every CV/ensemble role emits: legacy stores a test row per role only when the run
    # has a held-out test partition. dag-ml reports a `(test, None)` block exactly then, so its presence
    # is the test-partition signal — a no-test dataset drops every test row (the legacy train+val shape).
    cv_partitions = _CV_PARTITIONS if test is not None else ("train", "val")

    # --- Per-fold rows: (foldN, {train, val[, test]}). dag-ml scores only VALIDATION per fold, so the
    #     fold's own OOF block sits under train+val; the `test` key carries the SHARED held-out `(test,
    #     None)` block — every role's `test` is the same held-out test, so a partition-keyed `test`
    #     accessor (best_rmse / best_r2 / best_accuracy all read `scores[test][metric]`) returns the true
    #     held-out value no matter which val-row wins the rank, while ranking on `val` still selects the
    #     best-val fold exactly as legacy does. The native `foldN` id is normalized to the legacy
    #     integer-string `N` the webapp / PredictionAggregator key on. ---
    for fold_id in fold_keys:
        fold_block = by_key[("validation", fold_id)]
        fold_blocks: dict[str, dict[str, float] | None] = {"train": fold_block, "val": fold_block, "test": test}
        for partition in cv_partitions:
            add(_legacy_fold_id(fold_id), partition, fold_blocks)

    # --- Ensemble rows: avg + w_avg, each over {train, val[, test]}. Emitted only when dag-ml produced an
    #     OOF average over multiple folds (the legacy avg/w_avg blocks). val carries the true OOF average;
    #     train reuses the refit-train block, test the refit-test block (the ensemble's held-out proxy) —
    #     matching legacy, where avg.val == w_avg.val == cv_best_score. ---
    if has_avg and avg is not None:
        ensemble_blocks: dict[str, dict[str, float] | None] = {"train": final_train or avg, "val": avg, "test": test}
        for fold_id in ("avg", "w_avg"):
            for partition in cv_partitions:
                add(fold_id, partition, ensemble_blocks)

    # --- Refit final rows: (final, train) + (final, test), refit_context="standalone". Each carries the
    #     refit train/test blocks plus the OOF average under val (so it ranks on the same CV axis as the
    #     avg). `best`/`best_final` resolve to a final via score_scope="refit" (refit-only ranking), so no
    #     ranking perturbation is needed for the final to win. Degrades to train-only with no test. ---
    final_blocks: dict[str, dict[str, float] | None] = {"train": final_train, "val": avg, "test": test}
    if final_train is not None:
        add("final", "train", final_blocks, refit_context="standalone")
    if test is not None:
        add("final", "test", final_blocks, refit_context="standalone")

    predictions.flush()
    return RunResult(predictions=predictions, per_dataset={dataset_name: {"engine": "dag-ml"}})
