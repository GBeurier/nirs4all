#!/usr/bin/env python3
"""Engine performance comparison: nirs4all legacy vs dag-ml (RC-D flip-gate probe).

Measures the SAME seeded pipeline+dataset case on both engines and reports wall time, peak RSS,
and the dag-ml/legacy overhead ratio the cutover decision needs (see
``dag-ml/docs/migration-nirs4all/PARITY_AND_PERF_HARNESS.md`` Layer 4).

Each (case, engine, repeat) measurement runs in a FRESH subprocess so:

* peak RSS (``ru_maxrss``) is per-engine, not polluted by the other engine's allocations;
* module import / JIT / cache state cannot leak between engines or repeats;
* the engine is selected per-child via ``$N4A_ENGINE`` — no in-process engine switching.

The reported ``wall_s`` times ONLY the ``nirs4all.run()`` call (post-import), which is the engine
comparison that matters; ``total_s`` (interpreter start → exit) and ``peak_rss_mb`` are recorded for
context. Scores are captured per engine so a perf row can never silently hide a broken run.

Usage (from the nirs4all repo root, with the venv + the dag-ml/dag-ml-data bindings you want to
measure on ``PYTHONPATH``)::

    python scripts/bench_engine_perf.py                       # all cases, 3 repeats
    python scripts/bench_engine_perf.py --cases pls_small     # one case
    python scripts/bench_engine_perf.py --repeats 5 --json out.json
    python scripts/bench_engine_perf.py --max-wall-ratio 1.25 --max-rss-ratio 1.50

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

_CHILD_TEMPLATE = """
import json, math, os, resource, sys, time
t0 = time.perf_counter()
{case_source}
import nirs4all
requested_engine = os.environ["N4A_ENGINE"]
t_import = time.perf_counter() - t0
t1 = time.perf_counter()
result = None
try:
    result = nirs4all.run(
        pipeline=pipeline,
        dataset=dataset,
        verbose=0,
        random_state=0,
        engine=requested_engine,
        allow_fallback=False,
        save_artifacts=False,
        save_charts=False,
        plots_visible=False,
    )
    wall = time.perf_counter() - t1
    raw_best = result.best_score
    best = None if raw_best is None else float(raw_best)
    if best is not None and not math.isfinite(best):
        best = None
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB on Linux
    payload = {{
        "engine_requested": requested_engine,
        "wall_s": wall,
        "import_s": t_import,
        "peak_rss_mb": peak_kb / 1024.0,
        "num_predictions": result.num_predictions,
        "best_score": best,
    }}
finally:
    close = getattr(result, "close", None)
    if callable(close):
        close()
print("@@RESULT@@" + json.dumps(payload))
"""


def _run_child(case: str, engine: str, python: str) -> dict:
    source = _CHILD_TEMPLATE.format(case_source=CASES[case])
    env = dict(os.environ)
    env["N4A_ENGINE"] = engine
    t0 = time.perf_counter()
    proc = subprocess.run([python, "-c", source], capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parent.parent)
    total = time.perf_counter() - t0
    if proc.returncode != 0:
        return {"error": (proc.stderr.strip().splitlines() or ["child failed with no stderr"])[-1], "total_s": total}
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            payload = json.loads(line[len("@@RESULT@@"):])
            payload["total_s"] = total
            return payload
    return {"error": "child produced no @@RESULT@@ line", "total_s": total}


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return None
    return numerator / denominator


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
        comparison_available = case_ratios.get("wall") is not None or case_ratios.get("rss") is not None
        prediction_delta = case_ratios.get("predictions_delta_abs")
        if comparison_available and prediction_delta is None:
            failures.append(f"{case}: prediction count comparison unavailable")
        elif prediction_delta is not None and prediction_delta != 0:
            failures.append(f"{case}: prediction count delta {prediction_delta:g} != 0")
        checks = (
            ("wall", max_wall_ratio, "dag-ml/legacy wall ratio"),
            ("rss", max_rss_ratio, "dag-ml/legacy RSS ratio"),
            ("score_delta_abs", max_score_delta, "absolute best_score delta"),
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
    parser.add_argument("--max-score-delta", type=float, default=None, help="fail if any absolute best_score difference exceeds this limit")
    args = parser.parse_args()

    results: dict[str, dict[str, dict]] = {}
    for case in args.cases:
        results[case] = {}
        for engine in args.engines:
            runs = [_run_child(case, engine, args.python) for _ in range(args.repeats)]
            errors = [r["error"] for r in runs if "error" in r]
            if errors:
                results[case][engine] = {"error": errors[0], "runs": runs}
                print(f"[{case} / {engine}] FAILED: {errors[0]}", file=sys.stderr)
                continue
            summary = {
                "wall_s_median": statistics.median(r["wall_s"] for r in runs),
                "total_s_median": statistics.median(r["total_s"] for r in runs),
                "peak_rss_mb_median": statistics.median(r["peak_rss_mb"] for r in runs),
                "num_predictions": runs[0]["num_predictions"],
                "best_score": runs[0]["best_score"],
                "runs": runs,
            }
            results[case][engine] = summary
            print(f"[{case} / {engine}] wall={summary['wall_s_median']:.3f}s rss={summary['peak_rss_mb_median']:.0f}MB preds={summary['num_predictions']}", file=sys.stderr)

    # Markdown summary with the dag-ml/legacy ratio (the cutover-decision number).
    print("\n| case | engine | run wall (median s) | peak RSS (MB) | preds | best_score |")
    print("|---|---|---|---|---|---|")
    for case, engines in results.items():
        for engine, summary in engines.items():
            if "error" in summary:
                print(f"| {case} | {engine} | ERROR: {summary['error']} | | | |")
                continue
            best_score = summary["best_score"]
            best_score_text = "" if best_score is None else f"{best_score:.6f}"
            print(
                f"| {case} | {engine} | {summary['wall_s_median']:.3f} | {summary['peak_rss_mb_median']:.0f} "
                f"| {summary['num_predictions']} | {best_score_text} |"
            )
    ratios = _ratio_summary(results)
    print("\n| case | dag-ml/legacy wall ratio | dag-ml/legacy RSS ratio | abs best_score delta | prediction count delta |")
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
    failed = failed or bool(ratio_failures)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
