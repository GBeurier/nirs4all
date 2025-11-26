"""Benchmark iPLS NumPy vs JAX performance."""

import time
import numpy as np

from nirs4all.operators.models.sklearn.ipls import IntervalPLS


def benchmark_ipls(n_samples: int, n_features: int, n_intervals: int,
                   n_components: int, cv: int, mode: str, n_runs: int = 3):
    """Benchmark iPLS with NumPy and JAX backends.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features (wavelengths).
    n_intervals : int
        Number of intervals for iPLS.
    n_components : int
        Number of PLS components.
    cv : int
        Number of CV folds.
    mode : str
        Selection mode ('single', 'forward', 'backward').
    n_runs : int
        Number of benchmark runs (first is warmup for JAX).
    """
    print(f"\n{'='*70}")
    print(f"Benchmark: {n_samples} samples x {n_features} features")
    print(f"Config: {n_intervals} intervals, {n_components} components, {cv}-fold CV, mode='{mode}'")
    print('='*70)

    # Generate data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    # Create target with signal in middle intervals
    signal_start = n_features // 3
    signal_end = 2 * n_features // 3
    y = X[:, signal_start:signal_end].sum(axis=1) + 0.1 * np.random.randn(n_samples)

    # NumPy benchmark
    numpy_times = []
    for i in range(n_runs):
        model_np = IntervalPLS(
            n_components=n_components,
            n_intervals=n_intervals,
            cv=cv,
            mode=mode,
            backend='numpy'
        )
        start = time.perf_counter()
        model_np.fit(X, y)
        elapsed = time.perf_counter() - start
        numpy_times.append(elapsed)
        if i == 0:
            print(f"NumPy run {i+1}: {elapsed:.3f}s (selected: {model_np.selected_intervals_})")
        else:
            print(f"NumPy run {i+1}: {elapsed:.3f}s")

    numpy_avg = np.mean(numpy_times[1:]) if len(numpy_times) > 1 else numpy_times[0]
    print(f"NumPy average (excl. first): {numpy_avg:.3f}s")

    # JAX benchmark
    print()
    jax_times = []
    for i in range(n_runs + 1):  # Extra run for warmup
        model_jax = IntervalPLS(
            n_components=n_components,
            n_intervals=n_intervals,
            cv=cv,
            mode=mode,
            backend='jax'
        )
        start = time.perf_counter()
        model_jax.fit(X, y)
        elapsed = time.perf_counter() - start
        jax_times.append(elapsed)
        if i == 0:
            print(f"JAX run {i+1} (JIT compile): {elapsed:.3f}s")
        elif i == 1:
            print(f"JAX run {i+1}: {elapsed:.3f}s (selected: {model_jax.selected_intervals_})")
        else:
            print(f"JAX run {i+1}: {elapsed:.3f}s")

    jax_avg = np.mean(jax_times[2:]) if len(jax_times) > 2 else jax_times[-1]
    print(f"JAX average (excl. compile + first): {jax_avg:.3f}s")

    # Summary
    print()
    speedup = numpy_avg / jax_avg if jax_avg > 0 else float('inf')
    if speedup > 1:
        print(f"JAX is {speedup:.2f}x FASTER than NumPy")
    else:
        print(f"JAX is {1/speedup:.2f}x SLOWER than NumPy")

    return numpy_avg, jax_avg


if __name__ == "__main__":
    print("iPLS Benchmark: NumPy vs JAX")
    print("="*70)

    # Test configurations
    configs = [
        # Small: quick test
        {"n_samples": 50, "n_features": 100, "n_intervals": 5,
         "n_components": 3, "cv": 3, "mode": "single"},

        # Medium: typical NIR spectra
        {"n_samples": 100, "n_features": 500, "n_intervals": 10,
         "n_components": 5, "cv": 5, "mode": "single"},

        # Larger: more intervals
        {"n_samples": 100, "n_features": 1000, "n_intervals": 20,
         "n_components": 5, "cv": 5, "mode": "single"},

        # Forward selection (more computation)
        {"n_samples": 100, "n_features": 500, "n_intervals": 10,
         "n_components": 5, "cv": 5, "mode": "forward"},
    ]

    results = []
    for cfg in configs:
        np_time, jax_time = benchmark_ipls(**cfg, n_runs=3)
        results.append({**cfg, "numpy": np_time, "jax": jax_time})

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'Config':<50} {'NumPy':>8} {'JAX':>8} {'Speedup':>10}")
    print("-"*70)
    for r in results:
        cfg_str = f"{r['n_samples']}x{r['n_features']}, {r['n_intervals']} int, {r['mode']}"
        speedup = r['numpy'] / r['jax'] if r['jax'] > 0 else 0
        print(f"{cfg_str:<50} {r['numpy']:>7.3f}s {r['jax']:>7.3f}s {speedup:>9.2f}x")
