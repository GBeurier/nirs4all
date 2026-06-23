"""Gold-baseline capture + parity comparison for the parity oracle.

Implements Layers 0–2 of the migration parity plan
(``dag-ml/docs/migration-nirs4all/PARITY_AND_PERF_HARNESS.md``):

- **Layer 0 (determinism).** Every parity case pins seeds in its
  ``pipeline_factory``, so re-running the legacy backend yields an identical
  observation. This module assumes that contract.
- **Layer 1 (capture).** ``observe()`` extracts a stable, JSON-serializable
  record from a ``RunResult``; ``save_baseline()`` / ``load_baseline()`` persist
  it per case under ``baselines/``.
- **Layer 2 (enforce).** ``compare()`` diffs a fresh observation against the
  captured gold baseline within the case's recorded ``metric_tolerances``.

The legacy backend is the oracle of record (ADR-01). When the dag-ml backend is
wired in (Layer 3), the same ``compare()`` runs dag-ml observations against
these baselines, within the ADR-01 per-model-class tolerance table.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

BASELINE_DIR = Path(__file__).resolve().parent / "baselines"

# RunResult metric accessor per logical metric name.
_METRIC_ACCESSOR: dict[str, str] = {
    "best_score": "best_score",
    "rmse": "best_rmse",
    "r2": "best_r2",
    "accuracy": "best_accuracy",
    "cv_best_score": "cv_best_score",
}

# Metrics captured per task. Only the subset listed in a case's
# ``metric_tolerances`` is *enforced*; the rest are recorded for context.
_REGRESSION_METRICS = ("best_score", "rmse", "r2", "cv_best_score")
_CLASSIFICATION_METRICS = ("best_score", "accuracy", "cv_best_score")


def pipeline_fingerprint(pipeline: list[Any]) -> str:
    """Stable short hash of a materialized pipeline (baseline-staleness guard)."""
    return hashlib.sha256(repr(pipeline).encode("utf-8")).hexdigest()[:16]


def observe(result: Any, task: str) -> dict[str, Any]:
    """Extract a deterministic, JSON-serializable observation from a ``RunResult``."""
    metric_names = _CLASSIFICATION_METRICS if task == "classification" else _REGRESSION_METRICS
    metrics: dict[str, float] = {}
    for name in metric_names:
        value = getattr(result, _METRIC_ACCESSOR[name])
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            metrics[name] = float(value)
    return {
        "num_predictions": int(result.num_predictions),
        "models": sorted(result.get_models()),
        "datasets": sorted(result.get_datasets()),
        "metrics": metrics,
    }


def compare(gold: dict[str, Any], observed: dict[str, Any], tolerances: dict[str, float]) -> list[str]:
    """Return the list of parity violations (empty list == parity holds).

    Structural fields (``num_predictions``, ``models``, ``datasets``) must match
    exactly. Each metric named in ``tolerances`` must be within its absolute
    tolerance. Metrics not listed in ``tolerances`` are not enforced.
    """
    violations: list[str] = []
    for key in ("num_predictions", "models", "datasets"):
        if gold.get(key) != observed.get(key):
            violations.append(f"{key}: gold={gold.get(key)!r} != observed={observed.get(key)!r}")

    gold_metrics = gold.get("metrics", {})
    obs_metrics = observed.get("metrics", {})
    for metric, tol in tolerances.items():
        if metric not in gold_metrics:
            violations.append(f"metric {metric!r}: absent from gold baseline (recapture needed)")
            continue
        if metric not in obs_metrics:
            violations.append(f"metric {metric!r}: absent from observed run")
            continue
        delta = abs(gold_metrics[metric] - obs_metrics[metric])
        if delta > tol:
            violations.append(
                f"metric {metric!r}: |{gold_metrics[metric]} - {obs_metrics[metric]}| = "
                f"{delta:.3e} > tol {tol:.3e}"
            )
    return violations


def baseline_file(case_name: str) -> Path:
    """Path to the gold-baseline JSON for ``case_name``."""
    return BASELINE_DIR / f"{case_name}.json"


def load_baseline(case_name: str) -> dict[str, Any] | None:
    """Load a captured gold baseline, or ``None`` if none has been captured."""
    path = baseline_file(case_name)
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def save_baseline(case_name: str, fingerprint: str, observation: dict[str, Any], backend: str = "legacy") -> Path:
    """Persist an observation as the gold baseline for ``case_name``."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = baseline_file(case_name)
    payload = {"case": case_name, "backend": backend, "pipeline_fingerprint": fingerprint, **observation}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
