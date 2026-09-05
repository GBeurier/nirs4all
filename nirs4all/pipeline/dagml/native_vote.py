"""Classification presentation evidence captured inside native scientific tasks.

The native ScoreSet remains the selection authority. These sidecars retain real
fold probabilities for the library's existing ensemble/repetition presentation;
they are neither extra native score reports nor fabricated one-hot probabilities.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

import numpy as np
from sklearn.base import is_classifier

from nirs4all.core.metrics import eval_list
from nirs4all.data.ensemble_utils import EnsembleUtils

EVIDENCE_KEY = "classification_evidence"


def capture_vote_evidence(
    task: dict[str, Any], resolver: Any, estimator: Any,
    predictions: list[dict[str, Any]], train_ids: list[str],
    features: Callable[[list[str], bool], Any],
    predict: Callable[[list[str], bool], list[list[float]]],
    model_store: MutableMapping[Any, Any],
) -> None:
    """Capture labelled fold/refit outputs; never fit or replace a native block."""
    dataset = getattr(resolver, "_dataset", None)
    if not (getattr(dataset, "repetition", None) and getattr(dataset, "aggregate_method", None) == "vote" and is_classifier(estimator)):
        return
    phase = task["phase"]
    if phase not in {"FIT_CV", "REFIT"}:
        return
    node = task["node_plan"]["node_id"]
    fold = task.get("fold_id")
    variant = task.get("variant_id")
    specs = [(block["sample_ids"], "train" if block["partition"] == "final" else block["partition"], block["values"]) for block in predictions]
    if phase == "FIT_CV":
        specs.extend((ids, part, None) for ids, part in (
            (train_ids, "train"),
            (resolver.partition_wire_ids("train"), "ensemble_train"),
            (resolver.partition_wire_ids("test"), "test"),
        ) if ids)
    classes = np.asarray(estimator.classes_, dtype=float)
    if classes.ndim != 1 or not np.isfinite(classes).all() or len(np.unique(classes)) != len(classes):
        raise ValueError("classification evidence requires distinct finite class identities")
    records = []
    for ids, partition, values in specs:
        if len(ids) != len(set(ids)):
            raise ValueError("classification evidence requires unique row identities")
        y_pred = np.asarray(values if values is not None else predict(ids, False), dtype=float).reshape(len(ids), -1)
        y_true = np.asarray(resolver.resolve_targets(ids)["values"], dtype=float).reshape(len(ids), -1)
        if y_pred.shape != y_true.shape or y_true.shape[1] != 1:
            raise ValueError("classification vote evidence requires one aligned target")
        probabilities = None
        if callable(getattr(estimator, "predict_proba", None)):
            proba = np.asarray(estimator.predict_proba(features(ids, False)), dtype=float)
            if (proba.shape != (len(ids), len(classes)) or not np.isfinite(proba).all()
                    or np.any(proba < 0) or np.any(proba > 1) or not np.allclose(proba.sum(axis=1), 1, atol=1e-7, rtol=1e-7)):
                raise ValueError("classification evidence has invalid class probabilities")
            probabilities = proba.tolist()
        records.append({
            "node_id": node, "variant_id": variant, "phase": phase, "fold_id": fold,
            "run_id": task["run_id"], "params_fingerprint": task["node_plan"]["params_fingerprint"],
            "partition": partition, "sample_ids": list(ids), "classes": classes.tolist(),
            "y_true": y_true.ravel().tolist(), "y_pred": y_pred.ravel().tolist(), "y_proba": probabilities,
            "training_performed_for_evidence": False,
        })
    model_store[(EVIDENCE_KEY, node, variant, phase, fold)] = records


def collect_vote_evidence(model_store: dict[Any, Any]) -> list[dict[str, Any]]:
    """Extract only our typed sidecars; fitted artifact handles stay untouched."""
    return [record for key, records in model_store.items()
            if isinstance(key, tuple) and len(key) == 5 and key[0] == EVIDENCE_KEY
            for record in records]


def _fold_label(record: dict[str, Any]) -> str:
    if record["phase"] == "REFIT":
        return "final"
    fold = str(record["fold_id"])
    return fold[4:] if fold.startswith("fold") and fold[4:].isdigit() else fold


def _aligned_probabilities(record: dict[str, Any], classes: np.ndarray) -> np.ndarray | None:
    if record["y_proba"] is None:
        return None
    values = np.asarray(record["y_proba"], dtype=float)
    aligned = np.zeros((len(record["sample_ids"]), len(classes)))
    for source, label in enumerate(record["classes"]):
        target = np.flatnonzero(classes == label)
        if len(target) != 1:
            raise ValueError("probability class is outside the declared classifier universe")
        aligned[:, target[0]] = values[:, source]
    return aligned


def _ensemble_record(records: list[dict[str, Any]], weights: np.ndarray | None, classes: np.ndarray, *, oof: bool) -> dict[str, Any]:
    first = records[0]
    probabilities = [_aligned_probabilities(record, classes) for record in records]
    if oof:
        return {**first,
                "sample_ids": [sample for record in records for sample in record["sample_ids"]],
                "y_true": [value for record in records for value in record["y_true"]],
                "y_pred": [value for record in records for value in record["y_pred"]],
                "y_proba": np.concatenate([np.asarray(p) for p in probabilities]).tolist() if all(p is not None for p in probabilities) else None,
                "classes": classes.tolist()}
    if any(record["sample_ids"] != first["sample_ids"] or record["y_true"] != first["y_true"] for record in records):
        raise ValueError("fold ensemble evidence must have identical ordered sample and target identities")
    if all(p is not None for p in probabilities):
        labels, proba = EnsembleUtils.compute_soft_voting_average(
            [np.asarray(p) for p in probabilities], weights=weights, use_confidence_weighting=weights is not None,
        )
        labels = classes[labels.astype(int).ravel()]
    else:
        labels = EnsembleUtils.compute_hard_voting([np.asarray(record["y_pred"]) for record in records], weights=weights)
        proba = None
    return {**first, "y_pred": labels.ravel().tolist(), "y_proba": None if proba is None else proba.tolist(), "classes": classes.tolist()}


def project_vote_evidence(result: Any, records: list[dict[str, Any]], identity: Any) -> None:
    """Fill real row arrays/probabilities before the existing aggregate-twin owner.

    Per-fold validation and final reports remain native. Additional fold train/
    test and ensemble presentation scores are explicitly marked as library
    evidence, not inserted into the native ScoreSet or used for selection.
    """
    if not records:
        return
    classes = np.unique([label for record in records for label in record["classes"]])
    indexed = {(_fold_label(record), "val" if record["partition"] == "validation" else record["partition"]): record for record in records}
    if len(indexed) != len(records):
        raise ValueError("ambiguous multi-producer classification evidence")
    folds = list(dict.fromkeys(_fold_label(record) for record in records if record["phase"] == "FIT_CV"))
    entries = result.predictions.iter_entries()
    scores = [next(entry["val_score"] for entry in entries if str(entry["fold_id"]) == fold and entry["partition"] == "val") for fold in folds]
    weights = EnsembleUtils._scores_to_weights(np.asarray(scores), higher_is_better=True) if scores else None  # noqa: SLF001 - same owner as historical fold ensembles
    for label in ("avg", "w_avg"):
        for part, evidence_part in (("val", "val"), ("train", "ensemble_train"), ("test", "test")):
            matching = [indexed[(fold, evidence_part)] for fold in folds if (fold, evidence_part) in indexed]
            if matching:
                indexed[(label, part)] = _ensemble_record(matching, weights if label == "w_avg" else None, classes, oof=part == "val")
    for entry in entries:
        key = (str(entry["fold_id"]), entry["partition"])
        record = indexed.get(key)
        if record is None:
            continue
        sample_ids = record["sample_ids"]
        sample_indices = [identity.to_int(sample) for sample in sample_ids]
        y_true, y_pred = np.asarray(record["y_true"]), np.asarray(record["y_pred"])
        # Existing native blocks provide an independent, identity-aligned witness.
        prior_indices = entry.get("sample_indices") or []
        if prior_indices:
            positions = {sample: index for index, sample in enumerate(sample_indices)}
            order = [positions[sample] for sample in prior_indices]
            if not np.array_equal(np.asarray(entry["y_pred"]).ravel(), y_pred[order]) or not np.array_equal(np.asarray(entry["y_true"]).ravel(), y_true[order]):
                raise ValueError("classification host evidence disagrees with native predictions")
        entry.update(sample_indices=sample_indices, y_true=y_true, y_pred=y_pred, n_samples=len(sample_ids))
        entry["metadata"] = {**(entry.get("metadata") or {}), "physical_sample_id": list(sample_ids)}
        entry["y_proba"] = _aligned_probabilities(record, classes)
        metadata = dict(entry.get("result_metadata") or {})
        metadata["classification_evidence"] = {
            "source": "native_task_python_classifier", "classes": classes.tolist(),
            "sample_ids": sample_ids, "phase": record["phase"], "fold_id": record["fold_id"],
            "training_performed_for_evidence": False,
        }
        entry["result_metadata"] = metadata
        if key[0] != "final" and key[1] != "val":
            names = list((entry.get("scores") or {}).get(key[1], {})) or [entry["metric"]]
            names = [name for name in names if ":" not in name]
            values = eval_list(y_true, y_pred, names)
            display_scores = {name: float(value) for name, value in zip(names, values, strict=True) if value is not None}
            entry.setdefault("scores", {})[key[1]] = display_scores
            entry[f"{key[1]}_score"] = display_scores[entry["metric"]]
            metadata["score_origin"] = "library_classifier_presentation"
