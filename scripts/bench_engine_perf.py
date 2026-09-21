#!/usr/bin/env python3
"""Engine usability gate: real training and predictions, legacy vs DAG-ML.

Measures the SAME seeded pipeline+dataset case on both engines and reports wall time, peak RSS,
and the dag-ml/legacy overhead ratio the cutover decision needs (see
``dag-ml/docs/migration-nirs4all/PARITY_AND_PERF_HARNESS.md`` Layer 4).

Each (case, engine, repeat) measurement runs in a FRESH subprocess so:

* native OS peak RSS is per-engine, not polluted by the other engine's allocations;
* module import / JIT / cache state cannot leak between engines or repeats;
* the engine is selected per-child via ``$N4A_ENGINE`` — no in-process engine switching.

The reported ``wall_s`` times ONLY the ``nirs4all.run()`` call (post-import), which is the engine
comparison that matters; ``total_s`` (interpreter start → exit) and ``peak_rss_mb`` are recorded for
context. Each child has an isolated temporary workspace and a hard timeout.
Twenty percent of samples are held out: every final prediction must match the
reference. OOF scores are compared after averaging repeated validation predictions
per unique sample, since legacy concatenation and native sample averaging have
different public score semantics. Prediction-row counts are diagnostic only:
DAG-ML exposes actual score evidence and does not create legacy-shaped filler rows.

Usage (from the nirs4all repo root, with the venv + the dag-ml/dag-ml-data bindings you want to
measure on ``PYTHONPATH``)::

    python scripts/bench_engine_perf.py                       # all cases, 3 repeats
    python scripts/bench_engine_perf.py --cases pls_small     # one case
    python scripts/bench_engine_perf.py --repeats 5 --json out.json
    python scripts/bench_engine_perf.py --max-wall-ratio 1.25 --max-rss-ratio 1.50
    python scripts/bench_engine_perf.py --cases pls_small --repeats 1 --max-wall-ratio 3 --max-rss-ratio 2  # smoke gate

The harness inherits the parent environment, so RC worktree bindings are selected exactly like a
test run, e.g.::

    PYTHONPATH=…/RC-v1-dagml/crates/dag-ml-py/python:…/RC-v1-dmd/crates/dag-ml-data-py/python \
        python scripts/bench_engine_perf.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Cases: name -> child-process source that defines `pipeline` and `dataset`.
# Every case is fully seeded; sizes are chosen so a full sweep stays in minutes.
# ---------------------------------------------------------------------------

_CASE_COMMON = """
import numpy as np
rng = np.random.default_rng(2026)
def synth(n, p):
    X = rng.normal(0.5, 0.1, size=(n, p)).astype(np.float64)
    y = X[:, :5].sum(axis=1) + rng.normal(0, 0.05, size=n)
    return X, y
"""

CASES: dict[str, str] = {
    # The vertical-slice shape: tiny PLS + 2-fold CV. Measures fixed per-run engine overhead.
    "pls_small": _CASE_COMMON
    + """
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
dataset = synth(80, 50)
pipeline = [MinMaxScaler(), ShuffleSplit(n_splits=2, test_size=0.25, random_state=0), {"model": PLSRegression(n_components=3)}]
""",
    # Spectra-sized matrix, more folds: measures per-fold amortization on one variant.
    "pls_medium": _CASE_COMMON
    + """
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
dataset = synth(500, 500)
pipeline = [MinMaxScaler(), ShuffleSplit(n_splits=5, test_size=0.25, random_state=0), {"model": PLSRegression(n_components=10)}]
""",
    # NIRS preprocessing chain (host operators) + PLS: measures operator-callback overhead.
    "preproc_chain": _CASE_COMMON
    + """
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from nirs4all.operators.transforms import SavitzkyGolay, StandardNormalVariate
dataset = synth(300, 300)
pipeline = [StandardNormalVariate(), SavitzkyGolay(window_length=11, polyorder=2), ShuffleSplit(n_splits=3, test_size=0.25, random_state=0), {"model": PLSRegression(n_components=8)}]
""",
    # Generator sweep (4 variants x 3 folds): measures variant scheduling/amortization.
    "sweep_or": _CASE_COMMON
    + """
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
dataset = synth(200, 200)
pipeline = [MinMaxScaler(), ShuffleSplit(n_splits=3, test_size=0.25, random_state=0), {"model": {"_or_": [PLSRegression(n_components=k) for k in (2, 4, 8, 12)]}}]
""",
}

def _peak_rss_mb() -> float:
    """Read process peak resident memory with the native OS accounting API."""
    if sys.platform != "win32":
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(peak) / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            *[(field, ctypes.c_size_t) for field in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
            )],
        ]

    current_process = ctypes.windll.kernel32.GetCurrentProcess
    current_process.restype = wintypes.HANDLE
    get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    get_memory_info.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not get_memory_info(current_process(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0)


def _canonical_oof_rmse(rows: list[dict]) -> float:
    """Score unique validation samples after averaging their fold predictions.

    ShuffleSplit can validate a sample repeatedly. Legacy's public aggregate
    concatenates fold observations while DAG-ML averages predictions per sample;
    comparing those scalar scores would compare different estimands. This gate
    rebuilds the same sample-level estimand from each engine's actual fold rows.
    """
    import numpy as np

    samples: dict[int, tuple[float, list[float]]] = {}
    for row in rows:
        if row.get("partition") != "val" or str(row.get("fold_id")) in {"avg", "w_avg", "final"}:
            continue
        indices = row["sample_indices"]
        targets = np.asarray(row["y_true"], dtype=float).ravel()
        predictions = np.asarray(row["y_pred"], dtype=float).ravel()
        if not len(indices) or len(indices) != len(targets) or len(indices) != len(predictions):
            raise ValueError("OOF fold arrays are missing or misaligned")
        for index, target, prediction in zip(indices, targets, predictions, strict=True):
            index, target, prediction = int(index), float(target), float(prediction)
            if not math.isfinite(target) or not math.isfinite(prediction):
                raise ValueError("OOF sample values must be finite")
            if index in samples:
                if samples[index][0] != target:
                    raise ValueError("OOF sample has inconsistent targets across folds")
                samples[index][1].append(prediction)
            else:
                samples[index] = (target, [prediction])
    if not samples:
        raise ValueError("run did not produce OOF sample evidence")
    return math.sqrt(sum((target - sum(values) / len(values)) ** 2 for target, values in samples.values()) / len(samples))


_CHILD_TEMPLATE = """
import json, math, os, sys, tempfile, time
from scripts.bench_engine_perf import _canonical_oof_rmse, _peak_rss_mb
t0 = time.perf_counter()
{case_source}
dataset = (*dataset, {{"train": int(len(dataset[0]) * 0.8)}})
import nirs4all
requested_engine = os.environ["N4A_ENGINE"]
t_import = time.perf_counter() - t0
t1 = time.perf_counter()
result = None
workspace = tempfile.TemporaryDirectory(prefix="nirs4all-perf-")
try:
    result = nirs4all.run(
        pipeline=pipeline,
        dataset=dataset,
        workspace_path=workspace.name,
        verbose=0,
        random_state=0,
        engine=requested_engine,
        allow_fallback=False,
        save_artifacts=False,
        save_charts=False,
        plots_visible=False,
    )
    wall = time.perf_counter() - t1
    reported_cv_score = float(result.cv_best_score)
    if not math.isfinite(reported_cv_score):
        raise RuntimeError("run did not produce a finite CV score")
    selected_cv = result.cv_best
    cv_rows = result.predictions.filter_predictions(config_name=selected_cv["config_name"], partition="val", load_arrays=True)
    best = _canonical_oof_rmse(cv_rows)
    if requested_engine == "dag-ml" and abs(best - reported_cv_score) > 1e-5:
        raise RuntimeError("native CV score disagrees with sample-level OOF predictions")
    final_rows = result.predictions.filter_predictions(
        partition="test", fold_id="final", config_name=result.best["config_name"],
        model_name=result.best["model_name"], load_arrays=True,
    )
    if len(final_rows) != 1:
        raise RuntimeError("run did not produce one selected refit prediction on held-out data")
    test_predictions = np.asarray(final_rows[0]["y_pred"], dtype=float).ravel()
    if test_predictions.size != len(dataset[0]) - dataset[2]["train"] or not np.isfinite(test_predictions).all():
        raise RuntimeError("held-out refit predictions are missing or non-finite")
    per_dataset = getattr(result, "per_dataset", {{}})
    engine_tags = sorted(
        {{str(info.get("engine")) for info in per_dataset.values() if isinstance(info, dict) and info.get("engine") is not None}}
    )
    is_dagml_result = bool(getattr(result, "_is_dagml_engine", lambda: False)())
    fallback_diagnostics = [str(item) for item in getattr(result, "_rt_diagnostics", [])]
    if requested_engine == "dag-ml" and (not is_dagml_result or fallback_diagnostics):
        raise RuntimeError(
            "requested dag-ml but result was not verified as native dag-ml "
            f"(engine_tags={{engine_tags!r}}, is_dagml_result={{is_dagml_result!r}}, fallback_diagnostics={{fallback_diagnostics!r}})"
        )
    engine_observed = "dag-ml" if is_dagml_result else ("legacy/fallback" if fallback_diagnostics else "unknown")
    payload = {{
        "engine_requested": requested_engine,
        "engine_observed": engine_observed,
        "engine_verified": requested_engine != "dag-ml" or is_dagml_result,
        "engine_evidence": {{
            "per_dataset_engine_tags": engine_tags,
            "is_dagml_result": is_dagml_result,
            "fallback_diagnostics": fallback_diagnostics,
        }},
        "wall_s": wall,
        "import_s": t_import,
        "peak_rss_mb": _peak_rss_mb(),
        "num_predictions": result.num_predictions,
        "best_score": best,
        "reported_cv_score": reported_cv_score,
        "cv_score_semantics": "rmse_after_mean_prediction_per_unique_sample",
        "test_predictions": test_predictions.tolist(),
    }}
finally:
    close = getattr(result, "close", None)
    if callable(close):
        close()
    workspace.cleanup()
print("@@RESULT@@" + json.dumps(payload))
"""


def _run_child(case: str, engine: str, python: str, *, timeout: float = 90.0) -> dict[str, object]:
    source = _CHILD_TEMPLATE.format(case_source=CASES[case])
    env = dict(os.environ)
    env["N4A_ENGINE"] = engine
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [python, "-c", source], capture_output=True, text=True, env=env,
            cwd=Path(__file__).resolve().parent.parent, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"child exceeded {timeout:g} seconds", "total_s": time.perf_counter() - t0}
    total = time.perf_counter() - t0
    if proc.returncode != 0:
        return {"error": (proc.stderr.strip().splitlines() or ["child failed with no stderr"])[-1], "total_s": total}
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            raw_payload: object = json.loads(line[len("@@RESULT@@"):])
            if not isinstance(raw_payload, dict):
                return {"error": "child @@RESULT@@ payload was not a JSON object", "total_s": total}
            payload = {str(key): value for key, value in raw_payload.items()}
            payload["total_s"] = total
            return payload
    return {"error": "child produced no @@RESULT@@ line", "total_s": total}


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return None
    return numerator / denominator


def _float_field(row: dict[str, object], key: str) -> float:
    value = row[key]
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"child payload field {key!r} must be numeric, got {type(value).__name__}")


def _engine_evidence_from_run(run: dict) -> dict[str, object]:
    evidence: object = run.get("engine_evidence")
    if not isinstance(evidence, dict):
        return {}
    return {str(key): value for key, value in evidence.items()}


def _dagml_run_is_verified(run: dict) -> bool:
    """Return whether a child row proves that requested dag-ml did not measure legacy fallback.

    The strongest available signal is the RunResult/per_dataset engine tag populated by the dag-ml
    result projection and exposed in the child as ``engine_evidence.is_dagml_result``. A fallback run
    carries ``_rt_diagnostics`` instead; any such diagnostic makes the row unverified.
    """
    if run.get("engine_requested") != "dag-ml":
        return True
    evidence = _engine_evidence_from_run(run)
    return bool(run.get("engine_verified")) and bool(evidence.get("is_dagml_result")) and not evidence.get("fallback_diagnostics")


def _case_ratios(engines: dict[str, dict]) -> dict[str, float | None]:
    legacy = engines.get("legacy")
    dagml = engines.get("dag-ml")
    if not legacy or not dagml or "error" in legacy or "error" in dagml:
        return {"wall": None, "rss": None, "score_delta_abs": None, "predictions_delta_abs": None}
    legacy_score = legacy.get("best_score")
    dagml_score = dagml.get("best_score")
    score_delta_abs = None
    if legacy_score is not None and dagml_score is not None:
        score_delta_abs = abs(float(dagml_score) - float(legacy_score))
        if not math.isfinite(score_delta_abs):
            score_delta_abs = None
    legacy_predictions = legacy.get("num_predictions")
    dagml_predictions = dagml.get("num_predictions")
    predictions_delta_abs = None
    if legacy_predictions is not None and dagml_predictions is not None:
        predictions_delta_abs = abs(int(dagml_predictions) - int(legacy_predictions))
    return {
        "wall": _safe_ratio(float(dagml["wall_s_median"]), float(legacy["wall_s_median"])),
        "rss": _safe_ratio(float(dagml["peak_rss_mb_median"]), float(legacy["peak_rss_mb_median"])),
        "score_delta_abs": score_delta_abs,
        "predictions_delta_abs": predictions_delta_abs,
    }


def _ratio_summary(results: dict[str, dict[str, dict]]) -> dict[str, dict[str, float | None]]:
    return {case: _case_ratios(engines) for case, engines in results.items()}


def _check_ratio_gates(
    ratios: dict[str, dict[str, float | None]],
    *,
    max_wall_ratio: float | None,
    max_rss_ratio: float | None,
    max_score_delta: float | None,
) -> list[str]:
    failures: list[str] = []
    for case, case_ratios in ratios.items():
        checks = (
            ("wall", max_wall_ratio, "dag-ml/legacy wall ratio"),
            ("rss", max_rss_ratio, "dag-ml/legacy RSS ratio"),
            ("score_delta_abs", max_score_delta, "absolute canonical OOF RMSE delta"),
        )
        for key, limit, label in checks:
            if limit is None:
                continue
            actual = case_ratios.get(key)
            if actual is None:
                failures.append(f"{case}: {label} unavailable")
            elif actual > limit:
                failures.append(f"{case}: {label} {actual:.6g} > {limit:.6g}")
    return failures


def _check_engine_verification(results: dict[str, dict[str, dict]]) -> list[str]:
    failures: list[str] = []
    for case, engines in results.items():
        dagml = engines.get("dag-ml")
        if not dagml or "error" in dagml:
            continue
        runs = dagml.get("runs")
        if not isinstance(runs, list) or not runs:
            failures.append(f"{case}: dag-ml engine verification unavailable")
            continue
        unverified = [index for index, run in enumerate(runs, start=1) if not isinstance(run, dict) or not _dagml_run_is_verified(run)]
        if unverified:
            failures.append(f"{case}: dag-ml engine verification failed for repeats {unverified}")
    return failures


def _check_heldout_predictions(results: dict[str, dict[str, dict]], tolerance: float) -> list[str]:
    """Compare every native repeat to real legacy held-out predictions."""
    failures = []
    for case, engines in results.items():
        legacy = engines.get("legacy", {})
        native = engines.get("dag-ml", {})
        if not legacy or not native or "error" in legacy or "error" in native:
            continue
        expected = legacy["runs"][0].get("test_predictions")
        for engine, summary in engines.items():
            for index, row in enumerate(summary.get("runs", []), 1):
                actual = row.get("test_predictions")
                if not expected or not isinstance(actual, list) or len(actual) != len(expected):
                    failures.append(f"{case}/{engine} repeat {index}: held-out prediction evidence missing")
                elif any(not math.isfinite(value) or abs(value - target) > tolerance for value, target in zip(actual, expected, strict=True)):
                    failures.append(f"{case}/{engine} repeat {index}: held-out prediction difference exceeds {tolerance:g}")
    return failures


def _json_payload(
    *,
    cases: dict[str, dict[str, dict]],
    ratios: dict[str, dict[str, float | None]],
    args: argparse.Namespace,
) -> dict:
    return {
        "metadata": {
            "cases": args.cases,
            "engines": args.engines,
            "repeats": args.repeats,
            "python": args.python,
            "max_wall_ratio": args.max_wall_ratio,
            "max_rss_ratio": args.max_rss_ratio,
            "max_score_delta": args.max_score_delta,
        },
        "cases": cases,
        "ratios": ratios,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", nargs="*", default=list(CASES), choices=list(CASES), help="cases to run")
    parser.add_argument("--engines", nargs="*", default=["legacy", "dag-ml"], help="engines to compare")
    parser.add_argument("--repeats", type=int, default=3, help="repeats per (case, engine); median is reported")
    parser.add_argument("--json", type=Path, default=None, help="write full results as JSON to this path")
    parser.add_argument("--python", default=sys.executable, help="interpreter for measurement children")
    parser.add_argument("--max-wall-ratio", type=float, default=None, help="fail if any dag-ml/legacy wall-time ratio exceeds this limit")
    parser.add_argument("--max-rss-ratio", type=float, default=None, help="fail if any dag-ml/legacy peak-RSS ratio exceeds this limit")
    parser.add_argument("--max-score-delta", type=float, default=1e-5, help="maximum absolute CV score difference")
    parser.add_argument("--max-prediction-delta", type=float, default=1e-5, help="maximum absolute held-out prediction difference")
    parser.add_argument("--timeout", type=float, default=90.0, help="maximum seconds per fresh child")
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0 or not math.isfinite(args.timeout):
        parser.error("repeats and timeout must be positive")

    results: dict[str, dict[str, dict]] = {}
    for case in args.cases:
        results[case] = {}
        for engine in args.engines:
            runs = [_run_child(case, engine, args.python, timeout=args.timeout) for _ in range(args.repeats)]
            errors = [r["error"] for r in runs if "error" in r]
            if errors:
                results[case][engine] = {"error": errors[0], "runs": runs}
                print(f"[{case} / {engine}] FAILED: {errors[0]}", file=sys.stderr)
                continue
            summary = {
                "wall_s_median": statistics.median(_float_field(r, "wall_s") for r in runs),
                "total_s_median": statistics.median(_float_field(r, "total_s") for r in runs),
                "peak_rss_mb_median": statistics.median(_float_field(r, "peak_rss_mb") for r in runs),
                "num_predictions": runs[0]["num_predictions"],
                "best_score": runs[0]["best_score"],
                "engine_observed": runs[0].get("engine_observed"),
                "engine_verified": all(_dagml_run_is_verified(r) for r in runs),
                "runs": runs,
            }
            results[case][engine] = summary
            verified = " verified" if engine == "dag-ml" and summary["engine_verified"] else ""
            print(f"[{case} / {engine}] wall={summary['wall_s_median']:.3f}s rss={summary['peak_rss_mb_median']:.0f}MB preds={summary['num_predictions']}{verified}", file=sys.stderr)

    # Markdown summary with the dag-ml/legacy ratio (the cutover-decision number).
    print("\n| case | engine | engine proof | run wall (median s) | peak RSS (MB) | preds | CV score |")
    print("|---|---|---|---|---|---|---|")
    for case, engines in results.items():
        for engine, summary in engines.items():
            if "error" in summary:
                print(f"| {case} | {engine} | ERROR: {summary['error']} | | | | |")
                continue
            best_score = summary["best_score"]
            best_score_text = "" if best_score is None else f"{best_score:.6f}"
            engine_proof = "verified dag-ml" if engine == "dag-ml" and summary.get("engine_verified") else "n/a"
            print(
                f"| {case} | {engine} | {engine_proof} | {summary['wall_s_median']:.3f} | {summary['peak_rss_mb_median']:.0f} "
                f"| {summary['num_predictions']} | {best_score_text} |"
            )
    ratios = _ratio_summary(results)
    print("\n| case | dag-ml/legacy wall ratio | dag-ml/legacy RSS ratio | abs CV score delta | prediction count delta |")
    print("|---|---|---|---|---|")
    for case, case_ratios in ratios.items():
        wall_ratio = case_ratios["wall"]
        rss_ratio = case_ratios["rss"]
        score_delta_abs = case_ratios["score_delta_abs"]
        predictions_delta_abs = case_ratios["predictions_delta_abs"]
        wall_text = "n/a" if wall_ratio is None else f"{wall_ratio:.2f}x"
        rss_text = "n/a" if rss_ratio is None else f"{rss_ratio:.2f}x"
        score_text = "n/a" if score_delta_abs is None else f"{score_delta_abs:.6f}"
        predictions_text = "n/a" if predictions_delta_abs is None else str(int(predictions_delta_abs))
        print(f"| {case} | {wall_text} | {rss_text} | {score_text} | {predictions_text} |")

    if args.json is not None:
        args.json.write_text(json.dumps(_json_payload(cases=results, ratios=ratios, args=args), allow_nan=False, indent=2) + "\n")
        print(f"\nwrote {args.json}", file=sys.stderr)

    failed = any("error" in summary for engines in results.values() for summary in engines.values())
    ratio_failures = _check_ratio_gates(
        ratios,
        max_wall_ratio=args.max_wall_ratio,
        max_rss_ratio=args.max_rss_ratio,
        max_score_delta=args.max_score_delta,
    )
    for failure in ratio_failures:
        print(f"[gate] FAILED: {failure}", file=sys.stderr)
    engine_failures = _check_engine_verification(results)
    for failure in engine_failures:
        print(f"[gate] FAILED: {failure}", file=sys.stderr)
    prediction_failures = _check_heldout_predictions(results, args.max_prediction_delta)
    for failure in prediction_failures:
        print(f"[gate] FAILED: {failure}", file=sys.stderr)
    failed = failed or bool(ratio_failures) or bool(engine_failures) or bool(prediction_failures)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
