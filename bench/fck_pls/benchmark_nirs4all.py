"""
FCK-PLS Benchmark using nirs4all Pipelines
===========================================

Reproducible benchmark comparing multiple model approaches using pure nirs4all pipelines.
This replaces the manual benchmark with native nirs4all branching and finetuning.

MODELS (via branching):
1. PLS-Baseline: PLSRegression with SNV preprocessing (no tuning)
2. PLS-Tuned: PLSRegression with cartesian preprocessing search + n_components tuning
3. FCK-PLS-PP: nirs4all FCKPLS with preprocessing (SNV) + finetuning
4. FCK-PLS-Raw: nirs4all FCKPLS without preprocessing + finetuning

STRUCTURE:
- Pipeline 1: For datasets WITH explicit train/test split (uses path directly)
- Pipeline 2: For datasets WITHOUT explicit test split (uses tuple with partition)

The benchmark uses:
- Branch syntax to run all models in parallel
- finetune_params for hyperparameter tuning
- feature_augmentation for preprocessing search (PLS-Tuned branch)

Usage:
    python benchmark_nirs4all.py                    # Run all datasets
    python benchmark_nirs4all.py --quick            # Quick mode (reduced trials)
    python benchmark_nirs4all.py --dataset hiba     # Run specific dataset
    python benchmark_nirs4all.py --plots            # Generate plots

Author: nirs4all Team
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# Path setup
# =============================================================================
BENCH_DIR = Path(__file__).parent
PROJECT_ROOT = BENCH_DIR.parent.parent
SYNTHETIC_DIR = BENCH_DIR.parent / 'synthetic'

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SYNTHETIC_DIR))

# =============================================================================
# NIRS4ALL imports
# =============================================================================
# Synthetic data generator
from generator import SyntheticNIRSGenerator

# sklearn imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all
from nirs4all.data import DatasetConfigs, SpectroDataset
from nirs4all.operators.models import FCKPLS
from nirs4all.operators.transforms import (
    Detrend,
    FirstDerivative,
    Gaussian,
    Haar,
    SavitzkyGolay,
    SecondDerivative,
    Wavelet,
)
from nirs4all.operators.transforms import (
    MultiplicativeScatterCorrection as MSC,
)

# nirs4all transforms
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
)
from nirs4all.operators.transforms.nirs import (
    AreaNormalization,
)
from nirs4all.operators.transforms.nirs import (
    ExtendedMultiplicativeScatterCorrection as EMSC,
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    # General
    quick_mode: bool = False
    verbose: int = 1
    seed: int = 42

    # Finetuning
    finetune_trials: int = 30  # Full: 30 trials per model
    cartesian_count: int = 50   # Full: 50 preprocessing variants for PLS-Tuned

    # CV configuration
    n_splits: int = 3
    test_size: float = 0.20

# =============================================================================
# Dataset definitions
# =============================================================================

@dataclass
class DatasetDef:
    """Dataset definition for benchmarking."""
    name: str
    # Path for nirs4all (can be relative to examples/ folder)
    nirs4all_path: str | None = None
    # Pre-loaded arrays (for datasets without standard path format)
    X_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    # Info
    n_features: int = 0
    n_train: int = 0
    n_test: int = 0
    # Whether the dataset originally had a separate test set
    has_explicit_test: bool = False

# =============================================================================
# Dataset loading
# =============================================================================

def generate_synthetic_dataset(
    n_features: int,
    n_samples: int,
    seed: int = 42,
) -> DatasetDef:
    """Generate synthetic NIRS dataset (no explicit test split)."""
    wl_start = 1000
    wl_end = wl_start + n_features * 2

    generator = SyntheticNIRSGenerator(
        wavelength_start=wl_start,
        wavelength_end=wl_end,
        wavelength_step=(wl_end - wl_start) / n_features,
        complexity="complex",
        random_state=seed,
    )

    X, C, E = generator.generate(n_samples=n_samples, concentration_method="lognormal")
    y = C[:, 0]

    return DatasetDef(
        name=f"Synthetic_{n_features}x{n_samples}",
        nirs4all_path=None,
        X_train=X.astype(np.float32),
        y_train=y.astype(np.float32),
        X_test=None,
        y_test=None,
        n_features=n_features,
        n_train=n_samples,
        n_test=0,
        has_explicit_test=False,
    )

def load_csv_dataset(base_path: Path, name: str, sep: str = ";") -> DatasetDef:
    """Load dataset from CSV files (Xtrain, Ytrain, Xtest, Ytest)."""
    X_train = pd.read_csv(base_path / "Xtrain.csv", sep=sep, header=0).values
    y_train = pd.read_csv(base_path / "Ytrain.csv", sep=sep, header=0).values.ravel()
    X_test = pd.read_csv(base_path / "Xtest.csv", sep=sep, header=0).values
    y_test = pd.read_csv(base_path / "Ytest.csv", sep=sep, header=0).values.ravel()

    return DatasetDef(
        name=name,
        nirs4all_path=str(base_path),
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.float32),
        n_features=X_train.shape[1],
        n_train=X_train.shape[0],
        n_test=X_test.shape[0],
        has_explicit_test=True,
    )

def load_train_only_csv_dataset(base_path: Path, name: str, sep: str = ";") -> DatasetDef:
    """Load dataset from CSV files (Xtrain, Ytrain only - no explicit test)."""
    X_train = pd.read_csv(base_path / "Xtrain.csv", sep=sep, header=0).values
    y_train = pd.read_csv(base_path / "Ytrain.csv", sep=sep, header=0).values.ravel()

    return DatasetDef(
        name=name,
        nirs4all_path=None,
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.float32),
        X_test=None,
        y_test=None,
        n_features=X_train.shape[1],
        n_train=X_train.shape[0],
        n_test=0,
        has_explicit_test=False,
    )

def load_sample_data_regression() -> DatasetDef:
    """Load nirs4all sample regression data (has explicit train/test)."""
    return DatasetDef(
        name="Sample_Regression",
        nirs4all_path="sample_data/regression",  # Relative path works for sample_data
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        n_features=0,  # Will be determined at runtime
        n_train=0,
        n_test=0,
        has_explicit_test=True,
    )

def load_all_datasets(seed: int = 42, dataset_filter: str | None = None) -> list[DatasetDef]:
    """Load all benchmark datasets."""
    datasets = []

    # Synthetic datasets
    if not dataset_filter or "synthetic" in dataset_filter.lower():
        print("Generating synthetic datasets...")
        datasets.append(generate_synthetic_dataset(500, 150, seed))
        datasets.append(generate_synthetic_dataset(2500, 150, seed))
        datasets.append(generate_synthetic_dataset(2500, 1500, seed))

    # Real datasets
    print("Loading real datasets...")

    if not dataset_filter or "hiba" in dataset_filter.lower():
        ldmc_path = PROJECT_ROOT / "bench" / "_datasets" / "hiba" / "LDMC_hiba"
        if ldmc_path.exists():
            datasets.append(load_train_only_csv_dataset(ldmc_path, "LDMC_Hiba", sep=";"))
        else:
            print(f"  WARNING: LDMC dataset not found at {ldmc_path}")

    if not dataset_filter or "redox" in dataset_filter.lower():
        redox_path = PROJECT_ROOT / "bench" / "_datasets" / "redox" / "1700_Brix_StratGroupedKfold"
        if redox_path.exists():
            datasets.append(load_csv_dataset(redox_path, "Redox_Brix", sep=";"))
        else:
            print(f"  WARNING: Redox dataset not found at {redox_path}")

    if not dataset_filter or "sample" in dataset_filter.lower():
        try:
            datasets.append(load_sample_data_regression())
        except Exception as e:
            print(f"  WARNING: Sample regression data not found: {e}")

    return datasets

# =============================================================================
# Pipeline Definition for Datasets WITH Explicit Test Split
# =============================================================================

def build_pipeline_with_test(config: BenchmarkConfig, max_components: int) -> list:
    """
    Build benchmark pipeline for datasets with explicit train/test split.

    Uses branching to run all model variants in parallel:
    - PLS-Baseline: Simple baseline with SNV
    - PLS-Tuned: Cartesian preprocessing + tuning
    - FCK-PLS-PP: FCKPLS with preprocessing + tuning
    - FCK-PLS-Raw: FCKPLS without preprocessing + tuning

    Args:
        config: Benchmark configuration
        max_components: Maximum PLS components (based on dataset size)

    Returns:
        Pipeline list for nirs4all.run()
    """
    n_trials = 5 if config.quick_mode else config.finetune_trials
    n_pp_variants = 10 if config.quick_mode else config.cartesian_count

    # Common cross-validation
    cv_step = ShuffleSplit(
        n_splits=config.n_splits,
        test_size=config.test_size,
        random_state=config.seed
    )

    # FCK-PLS hyperparameter search space
    fckpls_params = {
        "n_components": ('int', 3, max_components),
        "alphas": [
            (0.0, 0.5, 1.0, 1.5, 2.0),
            (0.0, 1.0, 2.0),
            (0.5, 1.0, 1.5),
            (0.0, 0.5, 1.0),
            (1.0, 1.5, 2.0),
        ],
        "kernel_size": [11, 15, 21, 31],
    }

    # PLS hyperparameter search space
    pls_params = {
        "n_components": ('int', 1, max_components),
    }

    pipeline = [
        # Initial scaling (shared across all branches)
        MinMaxScaler(),

        # Cross-validation splitter
        cv_step,

        # Branch: Run different model configurations in parallel
        {"branch": {

            # =================================================================
            # Branch 1: PLS-Baseline (no tuning, fixed preprocessing)
            # =================================================================
            "PLS-Baseline": [
                SNV(),
                {"model": PLSRegression(n_components=min(10, max_components)), "name": "PLS-Baseline"},
            ],

            # =================================================================
            # Branch 2: PLS-Tuned (cartesian preprocessing + tuning)
            # =================================================================
            "PLS-Tuned": [
                # Cartesian preprocessing exploration
                {"feature_augmentation": {
                    "_cartesian_": [
                        {"_or_": [None, SNV(), EMSC(), Detrend()]},
                        {"_or_": [None, EMSC(), SavitzkyGolay(window_length=15), Gaussian(order=1, sigma=2)]},
                        {"_or_": [None, SavitzkyGolay(window_length=15, deriv=1), SavitzkyGolay(window_length=15, deriv=2)]},
                        {"_or_": [None, Haar(), Detrend(), AreaNormalization(), Wavelet("coif3")]},
                    ],
                    "count": n_pp_variants,
                }, "action": "extend"},

                # PLS with tuning
                {
                    "model": PLSRegression(),
                    "name": "PLS-Tuned",
                    "finetune_params": {
                        "n_trials": n_trials,
                        "sample": "tpe",
                        "verbose": 0,
                        "approach": "grouped",  # Tune per preprocessing variant
                        "model_params": pls_params,
                    },
                },
            ],

            # =================================================================
            # Branch 3: FCK-PLS with preprocessing (SNV)
            # =================================================================
            "FCK-PLS-PP": [
                SNV(),
                {
                    "model": FCKPLS(),
                    "name": "FCK-PLS-PP",
                    "finetune_params": {
                        "n_trials": n_trials,
                        "sample": "tpe",
                        "verbose": 0,
                        "approach": "single",
                        "model_params": fckpls_params,
                    },
                },
            ],

            # =================================================================
            # Branch 4: FCK-PLS Raw (no preprocessing)
            # =================================================================
            "FCK-PLS-Raw": [
                {
                    "model": FCKPLS(),
                    "name": "FCK-PLS-Raw",
                    "finetune_params": {
                        "n_trials": n_trials,
                        "sample": "tpe",
                        "verbose": 0,
                        "approach": "single",
                        "model_params": fckpls_params,
                    },
                },
            ],
        }},
    ]

    return pipeline

# =============================================================================
# Pipeline Definition for Datasets WITHOUT Explicit Test Split
# =============================================================================

def build_pipeline_without_test(config: BenchmarkConfig, max_components: int) -> list:
    """
    Build benchmark pipeline for datasets WITHOUT explicit train/test split.

    Same structure as build_pipeline_with_test but the dataset will be
    provided as a tuple (X, y, {"train": n_train, "test": n_test}) to
    create the holdout split.

    Args:
        config: Benchmark configuration
        max_components: Maximum PLS components (based on dataset size)

    Returns:
        Pipeline list for nirs4all.run()
    """
    # Same pipeline structure - the difference is how the dataset is provided
    return build_pipeline_with_test(config, max_components)

# =============================================================================
# Result Extraction
# =============================================================================

def extract_branch_results(result) -> pd.DataFrame:
    """
    Extract results from nirs4all RunResult, grouped by branch.

    Returns DataFrame with columns:
    - branch_name, model_name, r2, rmse, mae, best_params
    """
    records = []

    try:
        df = result.predictions.to_dataframe()

        # Get weighted average predictions (aggregated across folds)
        w_avg_rows = df.filter(df['fold_id'] == 'w_avg')
        if len(w_avg_rows) == 0:
            w_avg_rows = df.filter(df['fold_id'] == 'avg')
        if len(w_avg_rows) == 0:
            w_avg_rows = df  # Fallback to all rows

        for row in w_avg_rows.iter_rows(named=True):
            branch_name = row.get('branch_name', 'unknown')
            model_name = row.get('model_name', 'unknown')

            # Extract scores
            scores_str = row.get('scores', '{}')
            scores = json.loads(scores_str) if scores_str else {}

            # Prefer test metrics
            if 'test' in scores and scores['test'].get('r2') is not None:
                r2 = scores['test'].get('r2', float('nan'))
                rmse = scores['test'].get('rmse', float('nan'))
                mae = scores['test'].get('mae', float('nan'))
            elif 'val' in scores:
                r2 = scores['val'].get('r2', float('nan'))
                rmse = scores['val'].get('rmse', float('nan'))
                mae = scores['val'].get('mae', float('nan'))
            else:
                r2 = float('nan')
                rmse = float('nan')
                mae = float('nan')

            # Extract best params
            best_params_str = row.get('best_params', '{}')
            try:
                best_params = json.loads(best_params_str) if best_params_str else {}
            except (json.JSONDecodeError, TypeError):
                best_params = {}

            records.append({
                'branch_name': branch_name,
                'model_name': model_name,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'best_params': best_params,
            })

    except Exception as e:
        print(f"Warning: Could not extract results: {e}")

    return pd.DataFrame(records)

# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark_on_dataset(
    dataset: DatasetDef,
    config: BenchmarkConfig,
    show_plots: bool = False,
) -> pd.DataFrame:
    """
    Run benchmark on a single dataset using nirs4all pipeline.

    Args:
        dataset: Dataset definition
        config: Benchmark configuration
        show_plots: Whether to show plots

    Returns:
        DataFrame with results for this dataset
    """
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset.name}")
    print(f"{'=' * 60}")

    # Determine max_components based on dataset size
    if dataset.has_explicit_test:
        effective_train_size = dataset.n_train if dataset.n_train > 0 else 100
    else:
        n_total = dataset.n_train
        n_test = int(n_total * config.holdout_size)
        effective_train_size = n_total - n_test

    fold_train_fraction = 1 - config.test_size
    fold_train_size = int(effective_train_size * fold_train_fraction)
    max_components = min(25, fold_train_size - 2)
    max_components = max(3, max_components)

    print(f"  Effective train size per fold: ~{fold_train_size}")
    print(f"  Max components: {max_components}")

    # Build appropriate pipeline
    if dataset.has_explicit_test:
        pipeline = build_pipeline_with_test(config, max_components)
        dataset_arg = dataset.nirs4all_path
    else:
        pipeline = build_pipeline_without_test(config, max_components)
        # Create holdout split via partition
        n_total = dataset.n_train
        n_test = int(n_total * config.holdout_size)
        n_train = n_total - n_test
        dataset_arg = (
            dataset.X_train,
            dataset.y_train,
            {"train": n_train, "test": n_test}
        )

    # Run the pipeline
    start = time.time()

    result = nirs4all.run(
        pipeline=pipeline,
        dataset=dataset_arg,
        name=f"Benchmark_{dataset.name}",
        verbose=config.verbose,
        save_artifacts=True,
        plots_visible=show_plots,
    )

    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed:.1f}s")

    # Extract results
    results_df = extract_branch_results(result)
    results_df['dataset'] = dataset.name
    results_df['time_s'] = elapsed

    # Print summary
    print(f"\n📊 Results for {dataset.name}:")
    if not results_df.empty:
        for _, row in results_df.iterrows():
            print(f"  {row['branch_name']}: R²={row['r2']:.4f}, RMSE={row['rmse']:.4f}")

    return results_df

def run_full_benchmark(
    config: BenchmarkConfig,
    dataset_filter: str | None = None,
    show_plots: bool = False,
) -> pd.DataFrame:
    """
    Run the complete benchmark across all datasets.

    Args:
        config: Benchmark configuration
        dataset_filter: Optional filter for dataset names
        show_plots: Whether to show plots

    Returns:
        DataFrame with all results
    """
    print("=" * 70)
    print("FCK-PLS Benchmark (nirs4all Pipeline)")
    print("=" * 70)

    if config.quick_mode:
        print("🚀 QUICK MODE: Reduced trials for faster testing")
    print(f"CV: {config.n_splits} folds, {config.test_size:.0%} test size")
    print(f"Tuning: {config.finetune_trials} trials per model")
    print()

    # Load datasets
    datasets = load_all_datasets(config.seed, dataset_filter)
    print(f"\nLoaded {len(datasets)} datasets:")
    for ds in datasets:
        test_info = f"(train/test: {ds.n_train}/{ds.n_test})" if ds.has_explicit_test else "(no test split)"
        print(f"  - {ds.name}: {ds.n_features} features, {ds.n_train} samples {test_info}")
    print()

    # Run benchmark on each dataset
    all_results = []
    for dataset in datasets:
        try:
            results_df = run_benchmark_on_dataset(dataset, config, show_plots)
            all_results.append(results_df)
        except Exception as e:
            print(f"ERROR on {dataset.name}: {e}")
            import traceback
            traceback.print_exc()

    # Combine results
    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    return combined

def generate_report(results_df: pd.DataFrame) -> str:
    """Generate a markdown report from benchmark results."""
    lines = [
        "# FCK-PLS Benchmark Results (nirs4all Pipeline)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    if results_df.empty:
        lines.append("No results to report.")
        return "\n".join(lines)

    # Summary table
    lines.append("## Summary Table")
    lines.append("")

    # Pivot: Dataset × Branch → R²
    pivot = results_df.pivot_table(
        index="dataset",
        columns="branch_name",
        values="r2",
        aggfunc="first"
    )

    # Sort columns by average R²
    col_order = pivot.mean().sort_values(ascending=False).index
    pivot = pivot[col_order]

    lines.append("### R² Scores by Dataset and Model")
    lines.append("")
    lines.append(pivot.round(4).to_markdown())
    lines.append("")

    # Best model per dataset
    lines.append("### Best Model per Dataset")
    lines.append("")
    lines.append("| Dataset | Best Model | R² |")
    lines.append("|---------|------------|-----|")

    for dataset in results_df["dataset"].unique():
        ds_df = results_df[results_df["dataset"] == dataset]
        best_idx = ds_df["r2"].idxmax()
        best_row = ds_df.loc[best_idx]
        lines.append(f"| {dataset} | {best_row['branch_name']} | {best_row['r2']:.4f} |")

    lines.append("")

    # Model rankings
    lines.append("### Model Rankings (Average R² across datasets)")
    lines.append("")

    avg_r2 = results_df.groupby("branch_name")["r2"].mean().sort_values(ascending=False)
    lines.append("| Rank | Model | Avg R² |")
    lines.append("|------|-------|--------|")
    for rank, (model, r2) in enumerate(avg_r2.items(), 1):
        lines.append(f"| {rank} | {model} | {r2:.4f} |")

    lines.append("")

    return "\n".join(lines)

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FCK-PLS Benchmark using nirs4all Pipelines")
    parser.add_argument("--quick", action="store_true", help="Quick mode (reduced trials)")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset name")
    parser.add_argument("--plots", action="store_true", help="Show plots")
    parser.add_argument("--output", type=str, default=None, help="Save report to file")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0-2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = BenchmarkConfig(
        quick_mode=args.quick,
        verbose=args.verbose,
        seed=args.seed,
    )

    if args.quick:
        config.finetune_trials = 5
        config.cartesian_count = 10

    # Run benchmark
    results_df = run_full_benchmark(config, dataset_filter=args.dataset, show_plots=args.plots)

    # Generate and print report
    report = generate_report(results_df)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print()
    print(report)

    # Save report
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report)
        print(f"\n📄 Report saved to: {output_path}")

    # Save raw results
    if not results_df.empty:
        results_path = BENCH_DIR / "reports" / "benchmark_nirs4all_results.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"📊 Raw results saved to: {results_path}")

if __name__ == "__main__":
    main()
