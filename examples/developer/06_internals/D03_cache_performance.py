"""
D03 - Cache Performance: Measuring Step Cache and CoW Impact
=============================================================

Demonstrates the caching subsystem's impact on pipeline execution time
and memory usage. Runs the same pipeline with different cache settings
and compares wall-clock time, peak memory, and cache hit statistics.

This tutorial covers:

* Enabling step-level caching via CacheConfig
* Copy-on-Write (CoW) branch snapshots
* Block-based feature storage internals
* Observing cache statistics after a run
* Side-by-side performance comparison

Prerequisites
-------------
- 01_quickstart/U02_basic_regression for pipeline basics
- 05_advanced_features/D01_generators for generator sweeps

Duration: ~2 minutes
Difficulty: ★★★★☆
"""

import argparse
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
    """Run a pipeline and measure wall-clock time and peak RSS."""
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
    print(f"{'=' * 60}\n")
    return elapsed, rss_delta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(dataset_path: str, verbose: int = 0):
    # Import sklearn components
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from nirs4all.operators.transforms import (
        StandardNormalVariate,
        ExtendedMultiplicativeScatterCorrection as EMSC,
        Detrend,
        SavitzkyGolay,
        Gaussian,
        Haar,
        AreaNormalization,
        Wavelet,
    )

    # -----------------------------------------------------------------------
    # 1. Build a pipeline with a generator sweep (multiple variants)
    #    The _or_ keyword generates 2 variants, and _range_ generates 3 PLS
    #    component counts. Total = 2 preprocessors x 3 components = 6 variants.
    #    With step caching, shared preprocessing prefixes are computed once.
    # -----------------------------------------------------------------------
    pipeline = [
        {"_cartesian_": [
            # Stage 1: Scatter correction
            {"_or_": [None, StandardNormalVariate(), EMSC(), Detrend()]},
            # Stage 2: Smoothing/filtering
            {"_or_": [None, EMSC(), SavitzkyGolay(window_length=15), Gaussian(order=1, sigma=2)]},
            # Stage 3: Derivatives
            {"_or_": [None, SavitzkyGolay(window_length=15, deriv=1), SavitzkyGolay(window_length=15, deriv=2)]},
            # Stage 4: Additional transforms
            # {"_or_": [None, Haar(), Detrend(), AreaNormalization(), Wavelet("coif3")]},
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

    time_no_cache, mem_no_cache = _measure_run(
        pipeline, dataset_path, no_cache,
        label="NO CACHE (baseline)",
        verbose=verbose,
    )

    # -----------------------------------------------------------------------
    # 3. Run WITH CoW snapshots only (Phase 3)
    # -----------------------------------------------------------------------
    cow_only = CacheConfig(
        step_cache_enabled=False,
        use_cow_snapshots=True,
        log_cache_stats=False,
        log_step_memory=False,
    )

    time_cow, mem_cow = _measure_run(
        pipeline, dataset_path, cow_only,
        label="CoW SNAPSHOTS ONLY",
        verbose=verbose,
    )

    # -----------------------------------------------------------------------
    # 4. Run WITH full caching (step cache + CoW)
    # -----------------------------------------------------------------------
    full_cache = CacheConfig(
        step_cache_enabled=True,
        use_cow_snapshots=True,
        log_cache_stats=True,
        log_step_memory=False,
    )

    time_full, mem_full = _measure_run(
        pipeline, dataset_path, full_cache,
        label="FULL CACHE (step cache + CoW)",
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
    print(f"  {'Full cache (step + CoW)':<30} {time_full:>7.2f}s {mem_full:>+10.1f} MB")

    if time_no_cache > 0:
        speedup_cow = time_no_cache / time_cow if time_cow > 0 else float('inf')
        speedup_full = time_no_cache / time_full if time_full > 0 else float('inf')
        print(f"\n  CoW speedup   : {speedup_cow:.2f}x")
        print(f"  Full speedup  : {speedup_full:.2f}x")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache performance comparison")
    parser.add_argument(
        "--dataset-root",
        default=str(Path(__file__).resolve().parents[2] / "sample_data"),
        help="Path to dataset root directory (default: sample_data)",
    )
    parser.add_argument("-v", "--verbose", type=int, default=0)
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)
    for dataset_name in ("regression", "regression2", "multi"):
        dataset_path = str(dataset_root / dataset_name)
        print(f"\n\n=== DATASET: {dataset_name} ===")
        main(dataset_path, args.verbose)
