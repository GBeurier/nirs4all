"""
D03 - Cache Performance: Measuring Step Cache and CoW Impact
=============================================================

Demonstrates the caching subsystem's impact on pipeline execution time
and memory usage. Runs the same pipeline with different cache settings
and compares wall-clock time, peak memory, and cache hit statistics.

This tutorial covers:

* Enabling step-level caching via CacheConfig
* Copy-on-Write (CoW) branch snapshots and CoW step-cache restore
* Observing cache statistics (including timing breakdown) after a run
* Side-by-side performance comparison
* **Strict reproducibility test**: cached vs. uncached results must match

Prerequisites
-------------
- 01_quickstart/U02_basic_regression for pipeline basics
- 05_advanced_features/D01_generators for generator sweeps

Duration: ~2 minutes
Difficulty: ★★★★☆
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import nirs4all
from nirs4all.config.cache_config import CacheConfig
from nirs4all.utils.memory import get_process_rss_mb

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _measure_run(pipeline, dataset_path, cache_config, label, verbose=0):
    """Run a pipeline and measure wall-clock time and peak RSS.

    Returns:
        Tuple of (elapsed_seconds, rss_delta_mb, result).
    """
    rss_before = get_process_rss_mb()
    t0 = time.perf_counter()

    result = nirs4all.run(
        pipeline=pipeline,
        dataset=dataset_path,
        verbose=verbose,
        cache=cache_config,
    )

    elapsed = time.perf_counter() - t0
    rss_after = get_process_rss_mb()
    rss_delta = rss_after - rss_before

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Wall-clock time : {elapsed:.2f}s")
    print(f"  RSS before/after: {rss_before:.1f} / {rss_after:.1f} MB  (delta: {rss_delta:+.1f} MB)")
    print(f"  Best score      : {result.best_score:.4f}")
    print(f"  Predictions     : {result.num_predictions}")
    print(f"{'=' * 60}\n")
    return elapsed, rss_delta, result


def _extract_sorted_scores(result):
    """Extract all prediction test scores sorted for deterministic comparison."""
    scores = []
    for pred in result.predictions.to_dicts():
        scores.append(pred.get("test_score", 0.0) or 0.0)
    return sorted(scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(dataset_path: str, verbose: int = 0):
    # Import sklearn components
    from sklearn.cross_decomposition import PLSRegression
    from nirs4all.operators.transforms import (
        StandardNormalVariate,
        ExtendedMultiplicativeScatterCorrection as EMSC,
        Detrend,
        SavitzkyGolay,
        Gaussian,
    )

    # -----------------------------------------------------------------------
    # 1. Build a pipeline with a generator sweep (multiple variants)
    #    The _cartesian_ keyword generates all combinations:
    #    4 scatter x 4 smooth x 3 derivatives = 48 preprocessing chains
    #    x 4 PLS components = 192 total variants.
    #    With CoW step caching, shared preprocessing prefixes are computed
    #    once and restored via zero-copy SharedBlocks references.
    # -----------------------------------------------------------------------
    pipeline = [
        {"_cartesian_": [
            # Stage 1: Scatter correction
            {"_or_": [None, StandardNormalVariate(), EMSC(), Detrend()]},
            # Stage 2: Smoothing/filtering
            {"_or_": [None, EMSC(), SavitzkyGolay(window_length=15), Gaussian(order=1, sigma=2)]},
            # Stage 3: Derivatives
            # {"_or_": [None, SavitzkyGolay(window_length=15, deriv=1), SavitzkyGolay(window_length=15, deriv=2)]},
        ]},
        {"_range_": [5, 20, 5], "param": "n_components", "model": PLSRegression},
    ]

    # -----------------------------------------------------------------------
    # 2. Run WITHOUT caching (baseline)
    # -----------------------------------------------------------------------
    no_cache = CacheConfig(
        step_cache_enabled=False,
        use_cow_snapshots=False,
        log_cache_stats=False,
        log_step_memory=False,
    )

    time_no_cache, mem_no_cache, result_no_cache = _measure_run(
        pipeline, dataset_path, no_cache,
        label="NO CACHE (baseline)",
        verbose=verbose,
    )

    # -----------------------------------------------------------------------
    # 3. Run WITH CoW snapshots only (branch memory reduction)
    # -----------------------------------------------------------------------
    cow_only = CacheConfig(
        step_cache_enabled=False,
        use_cow_snapshots=True,
        log_cache_stats=False,
        log_step_memory=False,
    )

    time_cow, mem_cow, result_cow = _measure_run(
        pipeline, dataset_path, cow_only,
        label="CoW SNAPSHOTS ONLY",
        verbose=verbose,
    )

    # -----------------------------------------------------------------------
    # 4. Run WITH full caching (CoW step cache + CoW branch snapshots)
    # -----------------------------------------------------------------------
    full_cache = CacheConfig(
        step_cache_enabled=True,
        use_cow_snapshots=True,
        log_cache_stats=True,
        log_step_memory=False,
    )

    time_full, mem_full, result_full = _measure_run(
        pipeline, dataset_path, full_cache,
        label="FULL CACHE (CoW step cache + CoW branches)",
        verbose=verbose,
    )

    # -----------------------------------------------------------------------
    # 5. Summary comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Mode':<30} {'Time':>8} {'RSS delta':>12}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 12}")
    print(f"  {'No cache (baseline)':<30} {time_no_cache:>7.2f}s {mem_no_cache:>+10.1f} MB")
    print(f"  {'CoW snapshots only':<30} {time_cow:>7.2f}s {mem_cow:>+10.1f} MB")
    print(f"  {'Full cache (CoW step + CoW)':<30} {time_full:>7.2f}s {mem_full:>+10.1f} MB")

    if time_no_cache > 0:
        speedup_cow = time_no_cache / time_cow if time_cow > 0 else float('inf')
        speedup_full = time_no_cache / time_full if time_full > 0 else float('inf')
        print(f"\n  CoW speedup   : {speedup_cow:.2f}x")
        print(f"  Full speedup  : {speedup_full:.2f}x")

    print("=" * 60)

    # -----------------------------------------------------------------------
    # 6. Strict reproducibility test
    #    Cached and uncached runs MUST produce identical prediction scores.
    #    This validates that caching does not alter results.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  REPRODUCIBILITY TEST")
    print("=" * 60)

    scores_baseline = _extract_sorted_scores(result_no_cache)
    scores_cow = _extract_sorted_scores(result_cow)
    scores_full = _extract_sorted_scores(result_full)

    all_passed = True

    # Check prediction counts match and are non-zero
    if len(scores_baseline) == 0:
        print("  FAIL: No predictions produced (pipeline may not match dataset)")
        all_passed = False
    elif len(scores_baseline) != len(scores_cow):
        print(f"  FAIL: Count mismatch: baseline={len(scores_baseline)}, cow={len(scores_cow)}")
        all_passed = False
    elif len(scores_baseline) != len(scores_full):
        print(f"  FAIL: Count mismatch: baseline={len(scores_baseline)}, full={len(scores_full)}")
        all_passed = False
    else:
        print(f"  Prediction count: {len(scores_baseline)} (all modes match)")

        # Check CoW-only vs. baseline
        try:
            np.testing.assert_allclose(
                scores_baseline, scores_cow, atol=1e-10,
                err_msg="CoW-only scores differ from baseline",
            )
            print("  CoW-only vs baseline   : PASS (atol=1e-10)")
        except AssertionError:
            max_diff = max(abs(a - b) for a, b in zip(scores_baseline, scores_cow))
            print(f"  CoW-only vs baseline   : FAIL (max diff: {max_diff:.2e})")
            all_passed = False

        # Check full cache vs. baseline
        try:
            np.testing.assert_allclose(
                scores_baseline, scores_full, atol=1e-10,
                err_msg="Full cache scores differ from baseline",
            )
            print("  Full cache vs baseline : PASS (atol=1e-10)")
        except AssertionError:
            max_diff = max(abs(a - b) for a, b in zip(scores_baseline, scores_full))
            print(f"  Full cache vs baseline : FAIL (max diff: {max_diff:.2e})")
            all_passed = False

    if all_passed:
        print("\n  ALL REPRODUCIBILITY CHECKS PASSED")
    else:
        print("\n  SOME CHECKS FAILED")

    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache performance comparison")
    parser.add_argument(
        "--dataset-root",
        default=str(Path(__file__).resolve().parents[2] / "sample_data"),
        help="Path to dataset root directory (default: sample_data)",
    )
    parser.add_argument("-v", "--verbose", type=int, default=0)
    parser.add_argument("--plots", action="store_true", help="Generate plots (not used in this example)")
    parser.add_argument("--show", action="store_true", help="Display plots (not used in this example)")
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)

    all_ok = True
    for dataset_name in ("regression", "multi"):
        dataset_path = str(dataset_root / dataset_name)
        print(f"\n\n=== DATASET: {dataset_name} ===")
        ok = main(dataset_path, args.verbose)
        if not ok:
            all_ok = False

    if not all_ok:
        print("\nWARNING: Some reproducibility checks failed!")
        sys.exit(1)
    else:
        print("\nAll datasets passed reproducibility checks.")
