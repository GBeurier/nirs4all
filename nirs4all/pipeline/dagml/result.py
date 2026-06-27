"""Map a dag-ml ScoreSet into a nirs4all ``RunResult`` for the dag-ml backend.

dag-ml computes per-fold validation / cross-fold OOF / final-test scores natively (in Rust); this
turns its ``bundle.scores`` into an in-memory :class:`~nirs4all.data.predictions.Predictions`
wrapped in a :class:`~nirs4all.api.result.RunResult`, mirroring nirs4all's entry shape (an ``avg``
CV entry + one combined ``final`` refit entry carrying val/test/train together).
"""

from __future__ import annotations

from typing import Any

from nirs4all.api.result import RunResult
from nirs4all.data.predictions import Predictions


def _scores_to_run_result(scores: dict[str, Any] | None, dataset_name: str, model_name: str, metric: str = "rmse", task_type: str = "regression", producer: str | None = None) -> RunResult:
    """Map a dag-ml ScoreSet into a RunResult, mirroring nirs4all's entry shape.

    ``producer`` filters to one ``producer_node`` — e.g. a separation/duplication/stacking merge node,
    whose reports carry the full-universe OOF average (``cv_best_score``) and — since dag-ml routes the
    base branches' REFIT-test predictions to the merge node (``reassemble_branch_merge_off_fold`` for
    concat/fusion; the ``:refit`` off-fold input for stacking) — a reassembled ``(test, fold_id=None)``
    block (``best_rmse``); ``None`` keeps all reports (the single-model path, where exactly one producer
    scores).

    dag-ml emits one report per (partition, fold). nirs4all's RunResult expects per-fold validation
    entries + a single combined **refit/final** entry that carries val (the CV score), test and train
    scores together — that combined entry is what `best`/`best_rmse`/`best_final` resolve. We build
    exactly that: an `avg` CV entry (`cv_best_score`), and one `final` entry (`fold_id="final"`,
    `partition="val"`) with val_score=OOF-average, test_score from the producer's `(test, None)` block,
    train_score=final-train, and a combined `scores` dict.
    """
    reports = [report for report in (scores or {}).get("reports", []) if producer is None or report.get("producer_node") == producer]
    by_key = {(report["partition"], report.get("fold_id")): {name: float(value) for name, value in report["metrics"].items()} for report in reports}

    def add(fold_id: str | None, partition: str, scores_map: dict[str, dict[str, float]], *, val: float | None = None, test: float | None = None, train: float | None = None) -> None:
        predictions.add_prediction(dataset_name=dataset_name, model_name=model_name, fold_id=fold_id, partition=partition, metric=metric, task_type=task_type, scores=scores_map, val_score=val, test_score=test, train_score=train)

    # Two entries: the `avg` CV entry (cv_best_score) and the combined refit `final` entry that holds
    # val + test + train scores (best/best_rmse/best_accuracy resolve from it). Per-fold val entries
    # are omitted — they would let get_best(score_scope="all") rank a single fold first.
    #
    # _rank_candidates ranks an is_final entry by its OWN partition's metric, so the final entry uses
    # partition="val" (ranks on the CV metric, same axis as the avg) with the ranking value nudged a
    # negligible epsilon in the better direction (maximize accuracy, minimize rmse). That makes the
    # refit entry win the get_best tie over the avg for BOTH task directions; reported scores come
    # from the unmodified `scores` dict, so best_rmse/best_accuracy read the true final-test value.
    maximize = metric in ("accuracy", "r2")
    predictions = Predictions()
    # The CV score is the cross-fold OOF average ("avg") when dag-ml concatenated multiple folds. A
    # single-fold splitter (a train/validation split, e.g. KennardStone/SPXY with n_splits=1) emits no
    # "avg" — just the one validation fold — so fall back to that single fold's report as the CV score.
    avg = by_key.get(("validation", "avg"))
    if avg is None:
        validation_folds = [value for (partition, fold_id), value in by_key.items() if partition == "validation"]
        if len(validation_folds) == 1:
            avg = validation_folds[0]
    # The held-out TEST score (best_rmse) is the producer's `(test, fold_id=None)` block: the off-fold
    # convention the node runner emits for a model's REFIT test prediction AND that dag-ml's native
    # concat/fusion/stacking merge handlers reassemble under the merge producer. `producer` already
    # scopes `by_key` to the right node (the merge node for a merge pipeline; the single model otherwise).
    test = by_key.get(("test", None))
    train = by_key.get(("final", None))
    if avg is not None:
        add("avg", "val", {"val": avg}, val=avg.get(metric))
    if test is not None or train is not None:
        rank_val = dict(avg) if avg is not None else {}
        if metric in rank_val:
            rank_val[metric] += 1e-9 if maximize else -1e-9
        combined: dict[str, dict[str, float]] = {"val": rank_val}
        if train is not None:
            combined["train"] = train
        if test is not None:
            combined["test"] = test
        add("final", "val", combined, val=(avg or {}).get(metric), test=(test or {}).get(metric), train=(train or {}).get(metric))

    predictions.flush()
    return RunResult(predictions=predictions, per_dataset={dataset_name: {"engine": "dag-ml"}})
