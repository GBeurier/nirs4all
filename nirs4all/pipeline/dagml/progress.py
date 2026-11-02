"""Invocation-scoped observation of real DAG operator execution.

Observers receive only bounded execution metadata, never spectra or targets.
Progress measures completed cross-validation fits, not elapsed fit time.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass
class ExecutionProgress:
    callback: Callable[[dict[str, Any]], None]
    dataset: str = ""
    total_variants: int = 1
    completed: set[tuple[Any, ...]] = field(default_factory=set)
    variants: dict[tuple[Any, ...], int] = field(default_factory=dict)
    active_models: dict[tuple[Any, ...], str] = field(default_factory=dict)
    total_folds: int = 0
    bundle: int = 0
    reliable_total: bool = True
    lock: RLock = field(default_factory=RLock)

    def emit(self, message: str, **fields: Any) -> None:
        total = self.total_variants * self.total_folds
        progress = min(99.0, 100 * len(self.completed) / total) if total and self.reliable_total else None
        text = " ".join(str(message).split())[:1200]
        try:
            self.callback({"message": text, "progress": progress, "completed_cv_fits": len(self.completed),
                           "total_cv_fits": total if self.reliable_total else None, **fields})
        except Exception:
            # Telemetry must never change the scientific outcome.
            pass

    def configure(self, variants: int, dataset: str) -> None:
        with self.lock:
            self.completed.clear()
            self.variants.clear()
            self.active_models.clear()
            self.total_folds = 0
            # A subsequent dispatch has no reliable invocation-wide denominator.
            self.reliable_total = self.bundle == 0
            self.dataset = str(dataset)[:200]
            self.total_variants = max(1, variants)
            self.emit(f"{self.dataset}: preparing {self.total_variants} pipeline variants")

    def open_bundle(self, dsl: dict[str, Any], graph: dict[str, Any]) -> tuple[int, dict[str, int]]:
        with self.lock:
            self.bundle += 1
            fold_set = (dsl.get("split_invocation") or {}).get("fold_set") or {}
            folds = fold_set.get("folds") or []
            labels = {str(fold["fold_id"]): index + 1 for index, fold in enumerate(folds)}
            if self.total_folds and self.total_folds != len(labels):
                self.reliable_total = False
            self.total_folds = len(labels)
            models = [node for node in graph.get("nodes", []) if node.get("kind") in {"model", "tuner"}]
            if any(node.get("kind") == "tuner" for node in models) or (len(models) > 1 and self.total_variants == 1):
                self.reliable_total = False
            self.emit(f"{self.dataset}: cross-validation, {self.total_variants} variants, {len(labels)} folds")
            return self.bundle, labels

    def task(self, bundle: int, folds: dict[str, int], task: dict[str, Any], node: dict[str, Any], *, completed: bool) -> None:
        with self.lock:
            phase = str(task.get("phase", ""))
            variant_key = (bundle, str(task.get("variant_id") or "base"))
            variant = self.variants.setdefault(variant_key, len(self.variants) + 1)
            node_id = str(task["node_plan"]["node_id"])
            operator = node.get("operator", node_id)
            if isinstance(operator, dict):
                operator = operator.get("class", node_id)
            operator = str(operator).rsplit(".", 1)[-1][:150]
            fold_id = str(task.get("fold_id") or "")
            model = task["node_plan"].get("kind") == "model"
            if model and phase == "FIT_CV":
                previous = self.active_models.setdefault(variant_key, node_id)
                if previous != node_id or len(self.completed) > 100_000:
                    self.reliable_total = False
                if completed and len(self.completed) <= 100_000:
                    self.completed.add((bundle, node_id, variant_key[1], fold_id))
            total = self.total_variants * self.total_folds
            if len(self.completed) > total:
                self.reliable_total = False
            context = f"{self.dataset} · Variant {variant}/{self.total_variants}"
            if phase == "FIT_CV":
                context += f" · Fold {folds.get(fold_id, fold_id)}/{len(folds)}"
            action = "Refitting selected model" if phase == "REFIT" else "Predicting" if phase == "PREDICT" else "Fitting" if model else "Preprocessing"
            suffix = "completed" if completed else "running"
            counter = f" · CV fits {len(self.completed)}/{total}" if total and self.reliable_total else f" · {len(self.completed)} CV fits completed"
            self.emit(f"{context} · {action}: {operator} ({suffix}){counter}", phase=phase,
                      current_variant=variant, total_variants=self.total_variants, current_fold=folds.get(fold_id),
                      total_folds=len(folds), operator=operator)


_PROGRESS: ContextVar[ExecutionProgress | None] = ContextVar("nirs4all_execution_progress", default=None)


def execution_progress() -> ExecutionProgress | None:
    """Return the current invocation observer, if explicitly installed."""
    return _PROGRESS.get()


@contextmanager
def observe_progress(callback: Callable[[dict[str, Any]], None]) -> Iterator[ExecutionProgress]:
    """Observe the actual synchronous execution without changing scientific inputs."""
    if not callable(callback):
        raise TypeError("progress observer must be callable")
    observer = ExecutionProgress(callback)
    token = _PROGRESS.set(observer)
    try:
        yield observer
    finally:
        _PROGRESS.reset(token)
