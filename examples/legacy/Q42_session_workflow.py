"""
Q42 Example - Session Workflow for Multiple Pipeline Runs
==========================================================
Demonstrates the session context manager for efficient resource sharing
across multiple pipeline runs.

This example shows:
1. Basic session usage with shared configuration
2. Comparing multiple preprocessing approaches
3. Hyperparameter sweep within a session
4. Session best practices and tips

Sessions are useful when:
- Running multiple related experiments
- Comparing different pipelines on the same data
- Systematically exploring hyperparameters
- Maintaining consistent logging/workspace across runs

Phase 5 Implementation - Session Workflow Example
"""

# Standard library imports
import argparse
import time

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q42 Session Workflow Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots at end')
args = parser.parse_args()

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NIRS4All imports
import nirs4all
from nirs4all.operators.transforms import (
    StandardNormalVariate, SavitzkyGolay, Detrend, FirstDerivative, Gaussian
)


def example_1_basic_session():
    """Example 1: Basic session usage.

    Shows the fundamental pattern of using sessions for multiple runs.
    """
    print("\n" + "="*60)
    print("Example 1: Basic Session Usage")
    print("="*60)

    # Without session - each run creates its own runner
    print("\nWithout session (for comparison):")
    result1_no_session = nirs4all.run(
        pipeline=[MinMaxScaler(), PLSRegression(n_components=10)],
        dataset="sample_data/regression",
        name="NoSession_PLS",
        verbose=0,
        plots_visible=False
    )
    print(f"  Run 1: RMSE = {result1_no_session.best_rmse:.4f}")

    # With session - runs share configuration and workspace
    print("\nWith session (recommended for multiple runs):")
    with nirs4all.session(verbose=1, save_artifacts=True, plots_visible=False) as s:
        # All runs in this block share the session's configuration
        result1 = nirs4all.run(
            pipeline=[MinMaxScaler(), PLSRegression(n_components=10)],
            dataset="sample_data/regression",
            name="Session_PLS",
            session=s  # Pass session to reuse resources
        )
        print(f"  PLS: RMSE = {result1.best_rmse:.4f}")

        result2 = nirs4all.run(
            pipeline=[MinMaxScaler(), Ridge(alpha=1.0)],
            dataset="sample_data/regression",
            name="Session_Ridge",
            session=s
        )
        print(f"  Ridge: RMSE = {result2.best_rmse:.4f}")

    print("\nSession closed - resources cleaned up")

    return result1, result2


def example_2_preprocessing_comparison():
    """Example 2: Compare preprocessing methods.

    Uses session to systematically compare different preprocessing chains.
    """
    print("\n" + "="*60)
    print("Example 2: Preprocessing Method Comparison")
    print("="*60)

    # Define preprocessing methods to compare
    preprocessing_configs = {
        "Baseline": [],
        "MinMax": [MinMaxScaler()],
        "SNV": [MinMaxScaler(), StandardNormalVariate()],
        "SavGol": [MinMaxScaler(), SavitzkyGolay(window_length=11, polyorder=2)],
        "Detrend": [MinMaxScaler(), Detrend()],
        "1st_Derivative": [MinMaxScaler(), FirstDerivative()],
        "Gaussian": [MinMaxScaler(), Gaussian(sigma=2)],
        "Combined": [MinMaxScaler(), StandardNormalVariate(), SavitzkyGolay(window_length=11, polyorder=2)],
    }

    results = {}

    # Use session for efficient comparison
    with nirs4all.session(verbose=0, save_artifacts=False, plots_visible=False) as s:
        print("\nRunning preprocessing comparison...")

        for name, preprocess_steps in preprocessing_configs.items():
            # Build pipeline with current preprocessing
            pipeline = preprocess_steps + [
                ShuffleSplit(n_splits=3, test_size=0.25),
                {"model": PLSRegression(n_components=10)}
            ]

            result = nirs4all.run(
                pipeline=pipeline,
                dataset="sample_data/regression",
                name=f"Preproc_{name}",
                session=s
            )

            results[name] = {
                'rmse': result.best_rmse,
                'r2': result.best_r2
            }
            print(f"  {name:20s} - RMSE: {result.best_rmse:.4f}, R²: {result.best_r2:.4f}")

    # Summary
    print("\n" + "-"*50)
    print("Summary (sorted by RMSE):")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['rmse'])
    for i, (name, metrics) in enumerate(sorted_results, 1):
        marker = "★" if i == 1 else " "
        print(f"  {marker} {i}. {name:20s} RMSE: {metrics['rmse']:.4f}")

    best_name = sorted_results[0][0]
    print(f"\nBest preprocessing: {best_name}")

    return results


def example_3_hyperparameter_sweep():
    """Example 3: Hyperparameter sweep within session.

    Uses session to efficiently explore hyperparameter space.
    """
    print("\n" + "="*60)
    print("Example 3: Hyperparameter Sweep")
    print("="*60)

    # Parameter ranges to explore
    n_components_range = [3, 5, 8, 10, 15, 20, 25]

    results = {}

    print("\nPLS n_components sweep:")
    start_time = time.time()

    with nirs4all.session(verbose=0, save_artifacts=False, plots_visible=False) as s:
        for n_comp in n_components_range:
            pipeline = [
                MinMaxScaler(),
                StandardNormalVariate(),
                ShuffleSplit(n_splits=3, test_size=0.25),
                {"model": PLSRegression(n_components=n_comp)}
            ]

            result = nirs4all.run(
                pipeline=pipeline,
                dataset="sample_data/regression",
                name=f"PLS_{n_comp}",
                session=s
            )

            results[n_comp] = result.best_rmse
            print(f"  n_components={n_comp:2d}: RMSE = {result.best_rmse:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f}s ({elapsed/len(n_components_range):.2f}s per run)")

    # Find optimal
    best_n = min(results, key=results.get)
    print(f"\nOptimal n_components: {best_n} (RMSE = {results[best_n]:.4f})")

    return results


def example_4_model_comparison():
    """Example 4: Compare multiple model types.

    Uses session to compare different model architectures fairly.
    """
    print("\n" + "="*60)
    print("Example 4: Multi-Model Comparison")
    print("="*60)

    # Models to compare
    models = {
        "PLS_5": PLSRegression(n_components=5),
        "PLS_10": PLSRegression(n_components=10),
        "PLS_15": PLSRegression(n_components=15),
        "Ridge": Ridge(alpha=1.0),
        "RF_50": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
        "GBR": GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42),
    }

    # Shared preprocessing
    base_preprocessing = [
        MinMaxScaler(),
        StandardNormalVariate(),
        ShuffleSplit(n_splits=3, test_size=0.25),
    ]

    results = {}

    print("\nComparing models with shared preprocessing...")

    with nirs4all.session(verbose=0, save_artifacts=False, plots_visible=False) as s:
        for name, model in models.items():
            pipeline = base_preprocessing + [{"model": model, "name": name}]

            result = nirs4all.run(
                pipeline=pipeline,
                dataset="sample_data/regression",
                name=name,
                session=s
            )

            results[name] = {
                'rmse': result.best_rmse,
                'r2': result.best_r2,
                'score': result.best_score
            }
            print(f"  {name:12s} - RMSE: {result.best_rmse:.4f}, R²: {result.best_r2:.4f}")

    # Ranking
    print("\n" + "-"*50)
    print("Model Ranking (by RMSE):")
    sorted_models = sorted(results.items(), key=lambda x: x[1]['rmse'])
    for i, (name, metrics) in enumerate(sorted_models, 1):
        marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"  {marker} {i}. {name:12s} RMSE: {metrics['rmse']:.4f}")

    return results


def example_5_session_patterns():
    """Example 5: Advanced session patterns.

    Shows best practices and common patterns for session usage.
    """
    print("\n" + "="*60)
    print("Example 5: Session Best Practices")
    print("="*60)

    # Pattern 1: Session with custom workspace
    print("\nPattern 1: Custom workspace path")
    print("-" * 40)
    with nirs4all.session(
        verbose=0,
        save_artifacts=True,
        workspace_path="workspace/session_example"
    ) as s:
        result = nirs4all.run(
            pipeline=[MinMaxScaler(), PLSRegression(10)],
            dataset="sample_data/regression",
            name="CustomWorkspace",
            session=s
        )
        print(f"  Run saved to custom workspace")
        print(f"  RMSE: {result.best_rmse:.4f}")

    # Pattern 2: Verbose levels within session
    print("\nPattern 2: Override verbose per run")
    print("-" * 40)
    with nirs4all.session(verbose=0) as s:
        # Quiet run
        result1 = nirs4all.run(
            pipeline=[MinMaxScaler(), PLSRegression(5)],
            dataset="sample_data/regression",
            name="Quiet",
            session=s,
            verbose=0  # Override session verbose
        )
        print(f"  Quiet run completed")

        # Verbose run (override session setting)
        result2 = nirs4all.run(
            pipeline=[MinMaxScaler(), PLSRegression(10)],
            dataset="sample_data/regression",
            name="Verbose",
            session=s,
            verbose=1  # More verbose for this specific run
        )

    # Pattern 3: Error handling
    print("\nPattern 3: Error handling in session")
    print("-" * 40)
    with nirs4all.session(verbose=0, continue_on_error=True) as s:
        try:
            # Even if one run fails, session continues
            result = nirs4all.run(
                pipeline=[MinMaxScaler(), PLSRegression(10)],
                dataset="sample_data/regression",
                name="SafeRun",
                session=s
            )
            print(f"  Run completed: RMSE = {result.best_rmse:.4f}")
        except Exception as e:
            print(f"  Run failed but session is still valid: {e}")

    # Pattern 4: Collecting results
    print("\nPattern 4: Systematic result collection")
    print("-" * 40)

    all_results = []
    with nirs4all.session(verbose=0, save_artifacts=False) as s:
        for n in [5, 10, 15]:
            result = nirs4all.run(
                pipeline=[MinMaxScaler(), PLSRegression(n)],
                dataset="sample_data/regression",
                name=f"Collect_PLS_{n}",
                session=s
            )
            all_results.append({
                'n_components': n,
                'rmse': result.best_rmse,
                'r2': result.best_r2,
                'result': result  # Keep full result if needed
            })

    # Process collected results
    print("  Collected results:")
    for r in all_results:
        print(f"    n={r['n_components']:2d}: RMSE={r['rmse']:.4f}")

    best = min(all_results, key=lambda x: x['rmse'])
    print(f"  Best: n={best['n_components']} with RMSE={best['rmse']:.4f}")

    return all_results


def example_6_session_vs_no_session():
    """Example 6: Timing comparison - session vs no session.

    Demonstrates efficiency benefits of session usage.
    """
    print("\n" + "="*60)
    print("Example 6: Session vs No-Session Performance")
    print("="*60)

    n_runs = 5

    # Without session
    print(f"\nTiming {n_runs} runs WITHOUT session...")
    start = time.time()
    for i in range(n_runs):
        nirs4all.run(
            pipeline=[MinMaxScaler(), PLSRegression(10)],
            dataset="sample_data/regression",
            name=f"NoSession_{i}",
            verbose=0,
            save_artifacts=False,
            plots_visible=False
        )
    time_no_session = time.time() - start
    print(f"  Time: {time_no_session:.2f}s ({time_no_session/n_runs:.2f}s per run)")

    # With session
    print(f"\nTiming {n_runs} runs WITH session...")
    start = time.time()
    with nirs4all.session(verbose=0, save_artifacts=False, plots_visible=False) as s:
        for i in range(n_runs):
            nirs4all.run(
                pipeline=[MinMaxScaler(), PLSRegression(10)],
                dataset="sample_data/regression",
                name=f"Session_{i}",
                session=s
            )
    time_with_session = time.time() - start
    print(f"  Time: {time_with_session:.2f}s ({time_with_session/n_runs:.2f}s per run)")

    # Comparison
    speedup = time_no_session / time_with_session if time_with_session > 0 else 1.0
    print(f"\nResult: Session is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

    return time_no_session, time_with_session


def main():
    """Run all session examples."""
    print("\n" + "#"*60)
    print("# Q42: Session Workflow for Multiple Pipeline Runs")
    print("#"*60)

    # Run examples
    example_1_basic_session()
    example_2_preprocessing_comparison()
    example_3_hyperparameter_sweep()
    example_4_model_comparison()
    example_5_session_patterns()
    example_6_session_vs_no_session()

    print("\n" + "#"*60)
    print("# All Session Examples Complete!")
    print("#"*60)

    print("\nKey takeaways:")
    print("  1. Use 'with nirs4all.session(...) as s:' for multiple related runs")
    print("  2. Pass 'session=s' to nirs4all.run() to reuse resources")
    print("  3. Session configuration applies to all runs unless overridden")
    print("  4. Sessions help organize experiments with shared workspace")
    print("  5. Collect results in a list for post-processing")
    print("  6. Sessions can provide minor performance benefits")

    print("\nWhen to use sessions:")
    print("  ✓ Comparing multiple preprocessing methods")
    print("  ✓ Hyperparameter sweeps")
    print("  ✓ Model architecture comparisons")
    print("  ✓ Systematic experiments with shared configuration")
    print("  ✗ Single one-off runs (just use nirs4all.run() directly)")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
