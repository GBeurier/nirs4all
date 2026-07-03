"""Unit coverage for the legacy-vs-dag-ml performance gate helpers."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_bench_module() -> ModuleType:
    script = Path(__file__).resolve().parents[2] / "scripts" / "bench_engine_perf.py"
    spec = importlib.util.spec_from_file_location("bench_engine_perf", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ratio_summary_and_gates_detect_perf_regression() -> None:
    module = _load_bench_module()
    results = {
        "pls_small": {
            "legacy": {
                "wall_s_median": 2.0,
                "peak_rss_mb_median": 100.0,
                "best_score": 0.12,
                "num_predictions": 9,
            },
            "dag-ml": {
                "wall_s_median": 3.0,
                "peak_rss_mb_median": 130.0,
                "best_score": 0.20,
                "num_predictions": 9,
            },
        }
    }

    ratios = module._ratio_summary(results)

    assert ratios == {
        "pls_small": {
            "wall": 1.5,
            "rss": 1.3,
            "score_delta_abs": 0.08000000000000002,
            "predictions_delta_abs": 0,
        }
    }
    assert module._check_ratio_gates(
        ratios,
        max_wall_ratio=1.25,
        max_rss_ratio=1.5,
        max_score_delta=0.1,
    ) == ["pls_small: dag-ml/legacy wall ratio 1.5 > 1.25"]


def test_ratio_summary_marks_unavailable_comparisons() -> None:
    module = _load_bench_module()
    results = {
        "pls_small": {
            "legacy": {"error": "failed", "runs": []},
            "dag-ml": {"wall_s_median": 1.0, "peak_rss_mb_median": 1.0, "best_score": None},
        }
    }

    ratios = module._ratio_summary(results)

    assert ratios["pls_small"] == {
        "wall": None,
        "rss": None,
        "score_delta_abs": None,
        "predictions_delta_abs": None,
    }
    assert module._check_ratio_gates(
        ratios,
        max_wall_ratio=1.0,
        max_rss_ratio=None,
        max_score_delta=None,
    ) == ["pls_small: dag-ml/legacy wall ratio unavailable"]


def test_json_payload_is_strict_json_serializable() -> None:
    module = _load_bench_module()
    args = argparse.Namespace(
        cases=["pls_small"],
        engines=["legacy", "dag-ml"],
        repeats=1,
        python="/usr/bin/python3",
        max_wall_ratio=1.25,
        max_rss_ratio=1.5,
        max_score_delta=None,
    )
    payload = module._json_payload(
        cases={"pls_small": {"legacy": {"best_score": None}, "dag-ml": {"best_score": None}}},
        ratios={"pls_small": {"wall": 0.75, "rss": 1.0, "score_delta_abs": None, "predictions_delta_abs": 0}},
        args=args,
    )

    rendered = json.dumps(payload, allow_nan=False)

    assert '"ratios"' in rendered
    assert payload["metadata"]["max_wall_ratio"] == 1.25


def test_dagml_engine_verification_accepts_native_result() -> None:
    module = _load_bench_module()
    results = {
        "pls_small": {
            "dag-ml": {
                "runs": [
                    {
                        "engine_requested": "dag-ml",
                        "engine_verified": True,
                        "engine_evidence": {
                            "is_dagml_result": True,
                            "per_dataset_engine_tags": ["dag-ml"],
                            "fallback_diagnostics": [],
                        },
                    }
                ]
            }
        }
    }

    assert module._check_engine_verification(results) == []


def test_dagml_engine_verification_rejects_fallback_diagnostic() -> None:
    module = _load_bench_module()
    results = {
        "pls_small": {
            "dag-ml": {
                "runs": [
                    {
                        "engine_requested": "dag-ml",
                        "engine_verified": True,
                        "engine_evidence": {
                            "is_dagml_result": False,
                            "per_dataset_engine_tags": [],
                            "fallback_diagnostics": ["unsupported shape"],
                        },
                    }
                ]
            }
        }
    }

    assert module._check_engine_verification(results) == ["pls_small: dag-ml engine verification failed for repeats [1]"]


def test_dagml_engine_verification_rejects_missing_child_signal() -> None:
    module = _load_bench_module()
    results = {"pls_small": {"dag-ml": {"runs": [{"engine_requested": "dag-ml"}]}}}

    assert module._check_engine_verification(results) == ["pls_small: dag-ml engine verification failed for repeats [1]"]
