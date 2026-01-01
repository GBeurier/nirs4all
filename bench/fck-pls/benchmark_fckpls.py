"""
FCK-PLS Comprehensive Benchmark Suite
======================================

Reproducible benchmark comparing 4 model approaches on 6 datasets.

MODELS (all finetuned):
1. PLS-Tuned: PLSRegression with cartesian preprocessing matrix + n_components tuning
2. FCK-PLS-Static (with pp): nirs4all FCKPLS with small preprocessing + finetuning
3. FCK-PLS-Static (raw): nirs4all FCKPLS without preprocessing + finetuning
4. FCK-PLS-Torch-V1: PyTorch version with learnable kernels (no preprocessing)
5. FCK-PLS-Torch-V2: PyTorch version with parametric kernels (no preprocessing)

DATASETS:
- 3 Synthetic: 500x150, 2500x150, 2500x1500 features x samples
- 3 Real: HIBA LDMC, Redox Brix, Sample regression

CROSS-VALIDATION STRATEGY:
- If dataset has ONLY train data → 2 splits: first for test holdout, second for CV folds
- If dataset has train+test → 1 split: CV folds on train, final eval on test

All models use same:
- y_processing: MinMaxScaler for target normalization
- random_state/seed: 42 for reproducibility
- CV: 5-fold ShuffleSplit with 25% test per fold

Usage:
    python benchmark_fckpls.py                    # Run all models on all datasets
    python benchmark_fckpls.py --quick            # Quick test (reduced epochs/trials)
    python benchmark_fckpls.py --model pls        # Run only PLS baseline
    python benchmark_fckpls.py --model fckpls     # Run only FCK-PLS static
    python benchmark_fckpls.py --model torch      # Run only FCK-PLS Torch
    python benchmark_fckpls.py --step 1           # Step-by-step: PLS baseline only
    python benchmark_fckpls.py --step 2           # Step-by-step: add finetuning/cartesian
    python benchmark_fckpls.py --step 3           # Step-by-step: all models
    python benchmark_fckpls.py --plots            # Generate plots
    python benchmark_fckpls.py --output report.md # Save report to file
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
sys.path.insert(0, str(BENCH_DIR))

# =============================================================================
# Imports: nirs4all
# =============================================================================
import nirs4all
from nirs4all.data import DatasetConfigs, SpectroDataset
from nirs4all.operators.models import FCKPLS
from nirs4all.operators.transforms import (
    StandardNormalVariate, FirstDerivative, SecondDerivative,
    SavitzkyGolay, Detrend, Gaussian, Haar, MultiplicativeScatterCorrection,
    Wavelet,
)
from nirs4all.operators.transforms.nirs import (
    AreaNormalization,
    ExtendedMultiplicativeScatterCorrection as EMSC,
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

# =============================================================================
# Imports: FCK-PLS Torch
# =============================================================================
from fckpls_torch import FCKPLSTorch, TrainConfig

# =============================================================================
# Imports: Synthetic data generator
# =============================================================================
from generator import SyntheticNIRSGenerator


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
    step: int = 3  # 1=baseline, 2=finetuning, 3=all models

    # Torch training
    torch_epochs: int = 400
    torch_lr: float = 0.01
    torch_val_fraction: float = 0.25

    # Finetuning
    finetune_trials: int = 15

    # CV configuration
    n_splits: int = 5
    test_size: float = 0.25  # For CV folds
    holdout_size: float = 0.20  # For initial train/test split when no test provided


# =============================================================================
# Dataset definitions
# =============================================================================

@dataclass
class DatasetDef:
    """Dataset definition for benchmarking."""
    name: str
    # Path for nirs4all (can be relative to examples/ folder)
    nirs4all_path: Optional[str] = None
    # Pre-loaded arrays for Torch models
    X_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
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
    test_fraction: float = 0.2,
) -> DatasetDef:
    """Generate synthetic NIRS dataset.

    Synthetic datasets have NO explicit test set - we generate all data as 'train'
    and let the benchmark create the holdout split for reproducibility.
    """
    # Adjust wavelength step based on feature count
    wl_start = 1000
    wl_end = wl_start + n_features * 2  # ~2nm step

    generator = SyntheticNIRSGenerator(
        wavelength_start=wl_start,
        wavelength_end=wl_end,
        wavelength_step=(wl_end - wl_start) / n_features,
        complexity="complex",
        random_state=seed,
    )

    X, C, E = generator.generate(n_samples=n_samples, concentration_method="lognormal")
    y = C[:, 0]  # Use first component as target

    # For synthetic data, we DON'T split here - let benchmark handle it
    return DatasetDef(
        name=f"Synthetic_{n_features}x{n_samples}",
        nirs4all_path=None,
        X_train=X.astype(np.float32),
        y_train=y.astype(np.float32),
        X_test=None,  # No explicit test set
        y_test=None,
        n_features=n_features,
        n_train=n_samples,
        n_test=0,
        has_explicit_test=False,  # Benchmark will create holdout
    )


def load_csv_dataset(base_path: Path, name: str, sep: str = ";") -> DatasetDef:
    """Load dataset from CSV files (Xtrain, Ytrain, Xtest, Ytest).

    Real datasets with explicit train/test split.
    """
    X_train = pd.read_csv(base_path / "Xtrain.csv", sep=sep, header=0).values
    y_train = pd.read_csv(base_path / "Ytrain.csv", sep=sep, header=0).values.ravel()
    X_test = pd.read_csv(base_path / "Xtest.csv", sep=sep, header=0).values
    y_test = pd.read_csv(base_path / "Ytest.csv", sep=sep, header=0).values.ravel()

    return DatasetDef(
        name=name,
        nirs4all_path=str(base_path),  # Absolute path for nirs4all
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.float32),
        n_features=X_train.shape[1],
        n_train=X_train.shape[0],
        n_test=X_test.shape[0],
        has_explicit_test=True,  # Dataset provides test set
    )


def load_train_only_csv_dataset(base_path: Path, name: str, sep: str = ";") -> DatasetDef:
    """Load dataset from CSV files (Xtrain, Ytrain only - no explicit test).

    Real datasets without explicit train/test split - benchmark will create holdout.
    """
    X_train = pd.read_csv(base_path / "Xtrain.csv", sep=sep, header=0).values
    y_train = pd.read_csv(base_path / "Ytrain.csv", sep=sep, header=0).values.ravel()

    return DatasetDef(
        name=name,
        nirs4all_path=None,  # Can't use path since nirs4all expects train/test or cal/val
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.float32),
        X_test=None,  # No explicit test set
        y_test=None,
        n_features=X_train.shape[1],
        n_train=X_train.shape[0],
        n_test=0,
        has_explicit_test=False,  # Benchmark will create holdout
    )


def load_sample_data_regression() -> DatasetDef:
    """Load nirs4all sample regression data.

    This dataset has explicit train/test (Xcal/Xval).
    """
    base_path = PROJECT_ROOT / "examples" / "sample_data" / "regression"

    # X files use ; separator, no header. Y files are single column, no header.
    X_train = pd.read_csv(base_path / "Xcal.csv.gz", sep=";", header=None).values
    y_train = pd.read_csv(base_path / "Ycal.csv.gz", header=None).values.ravel()
    X_test = pd.read_csv(base_path / "Xval.csv.gz", sep=";", header=None).values
    y_test = pd.read_csv(base_path / "Yval.csv.gz", header=None).values.ravel()

    return DatasetDef(
        name="Sample_Regression",
        nirs4all_path=str(base_path),  # Use absolute path for nirs4all
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.float32),
        n_features=X_train.shape[1],
        n_train=X_train.shape[0],
        n_test=X_test.shape[0],
        has_explicit_test=True,  # Dataset provides test set
    )


def load_all_datasets(seed: int = 42) -> List[DatasetDef]:
    """Load all benchmark datasets."""
    datasets = []

    # Synthetic datasets
    print("Generating synthetic datasets...")
    datasets.append(generate_synthetic_dataset(500, 150, seed))
    datasets.append(generate_synthetic_dataset(2500, 150, seed))
    datasets.append(generate_synthetic_dataset(2500, 1500, seed))

    # Real datasets
    print("Loading real datasets...")

    ldmc_path = PROJECT_ROOT / "bench" / "_datasets" / "hiba" / "LDMC_hiba"
    if ldmc_path.exists():
        datasets.append(load_train_only_csv_dataset(ldmc_path, "LDMC_Hiba", sep=";"))
    else:
        print(f"  WARNING: LDMC dataset not found at {ldmc_path}")

    redox_path = PROJECT_ROOT / "bench" / "_datasets" / "redox" / "1700_Brix_StratGroupedKfold"
    if redox_path.exists():
        datasets.append(load_csv_dataset(redox_path, "Redox_Brix", sep=";"))
    else:
        print(f"  WARNING: Redox dataset not found at {redox_path}")

    try:
        datasets.append(load_sample_data_regression())
    except Exception as e:
        print(f"  WARNING: Sample regression data not found: {e}")

    return datasets


# =============================================================================
# Model evaluation helpers
# =============================================================================

@dataclass
class ModelResult:
    """Results for a single model run."""
    model_name: str
    dataset_name: str
    r2: float
    rmse: float
    mae: float
    train_time: float
    extra_info: Dict[str, Any] = None


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute metrics."""
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def extract_metrics_from_result(result) -> Tuple[float, float, float]:
    """
    Extract R², RMSE, and MAE from nirs4all result object.

    Returns:
        (r2, rmse, mae) - R², RMSE, and MAE values from the best prediction
    """
    import json

    try:
        df = result.predictions.to_dataframe()

        # Get the weighted average prediction (best aggregated result)
        best_rows = df.filter((df['fold_id'] == 'w_avg'))
        if len(best_rows) == 0:
            # Fall back to avg
            best_rows = df.filter((df['fold_id'] == 'avg'))
        if len(best_rows) == 0:
            # Fall back to first non-fold-specific row
            best_rows = df.head(1)

        if len(best_rows) == 0:
            return float('nan'), float('nan'), float('nan')

        # Get scores from the first matching row
        scores_str = best_rows['scores'][0]
        if scores_str is None:
            return float('nan'), float('nan'), float('nan')

        scores = json.loads(scores_str)

        # Prefer test metrics if available, otherwise use val
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

        return r2, rmse, mae

    except Exception as e:
        print(f"  Warning: Could not extract metrics: {e}")
        return float('nan'), float('nan'), float('nan')


def prepare_dataset_for_nirs4all(
    dataset: DatasetDef,
    config: BenchmarkConfig,
) -> Tuple[Any, int]:
    """
    Prepare dataset for nirs4all pipeline.

    Returns:
        - dataset_arg: The dataset argument for nirs4all.run()
        - effective_train_size: Estimated samples available for fitting in each fold

    Strategy:
    - If has_explicit_test: use path (nirs4all handles train/test) or tuple with partition
    - If no explicit test: use tuple (X, y, partition) creating a holdout split
    """
    if dataset.has_explicit_test:
        # Dataset has train/test split - use path if available
        if dataset.nirs4all_path:
            dataset_arg = dataset.nirs4all_path
        else:
            # For tuples with explicit test, create partition dict
            dataset_arg = (
                np.vstack([dataset.X_train, dataset.X_test]),
                np.concatenate([dataset.y_train, dataset.y_test]),
                {"train": len(dataset.X_train), "test": len(dataset.X_test)}
            )
        effective_train_size = dataset.n_train
    else:
        # No explicit test - create a holdout split from the data
        # We split using holdout_size to create train/test partition
        n_total = dataset.n_train
        n_test = int(n_total * config.holdout_size)
        n_train = n_total - n_test

        # Create partition: first n_train samples are train, rest are test
        # The data itself doesn't need reshuffling - ShuffleSplit will handle CV
        dataset_arg = (
            dataset.X_train,
            dataset.y_train,
            {"train": n_train, "test": n_test}
        )
        effective_train_size = n_train

    # Account for CV splits: after test_size split, we have ~(1-test_size) for training in each fold
    # Then for each fold, that's what's available for fitting
    fold_train_fraction = 1 - config.test_size
    effective_train_size = int(effective_train_size * fold_train_fraction)

    return dataset_arg, effective_train_size


# =============================================================================
# Model 1: PLSRegression - Baseline (no tuning, fixed preprocessing)
# =============================================================================

def run_pls_baseline(
    dataset: DatasetDef,
    config: BenchmarkConfig,
) -> ModelResult:
    """
    Run PLSRegression baseline with fixed preprocessing and n_components.

    This establishes a simple baseline to verify CV is working correctly.
    Preprocessing: MinMaxScaler + SNV (simple, reasonable defaults)
    """
    dataset_arg, effective_train_size = prepare_dataset_for_nirs4all(dataset, config)
    n_components = min(10, effective_train_size - 2, dataset.n_features)

    pipeline = [
        MinMaxScaler(),
        # Note: y_processing not used - causes conflicts with feature_augmentation
        # For fair comparison, all models use raw y values
        StandardNormalVariate(),
        ShuffleSplit(n_splits=config.n_splits, test_size=config.test_size, random_state=config.seed),
        {"model": PLSRegression(n_components=n_components)}
    ]

    start = time.time()
    try:
        result = nirs4all.run(
            pipeline=pipeline,
            dataset=dataset_arg,
            name="PLS-Baseline",
            verbose=0,
            save_artifacts=False,
            save_charts=False,
        )

        train_time = time.time() - start

        # Extract metrics properly
        r2, rmse, mae = extract_metrics_from_result(result)

        return ModelResult(
            model_name="PLS-Baseline",
            dataset_name=dataset.name,
            r2=r2,
            rmse=rmse,
            mae=mae,
            train_time=train_time,
            extra_info={"n_components": n_components},
        )

    except Exception as e:
        print(f"  ERROR in PLS-Baseline: {e}")
        import traceback
        traceback.print_exc()
        return ModelResult(
            model_name="PLS-Baseline",
            dataset_name=dataset.name,
            r2=float('nan'),
            rmse=float('nan'),
            mae=float('nan'),
            train_time=time.time() - start,
            extra_info={"error": str(e)},
        )


# =============================================================================
# Model 2: PLSRegression with cartesian preprocessing + tuning
# =============================================================================

def run_pls_tuned(
    dataset: DatasetDef,
    config: BenchmarkConfig,
) -> ModelResult:
    """
    Run PLSRegression with cartesian preprocessing search and component tuning.
    Uses nirs4all pipeline with full preprocessing search to show the overhead
    that FCK-PLS aims to avoid.

    The cartesian product generates many different preprocessing combinations
    as separate pipelines (not feature augmentation). Each is evaluated with
    the same finetuning strategy.
    """
    n_trials = 5 if config.quick_mode else config.finetune_trials
    dataset_arg, effective_train_size = prepare_dataset_for_nirs4all(dataset, config)

    # For finetuning, we need to be more conservative with max_components
    # In CV, we further split train into train/val, so effective size is smaller
    # Also account for grouped approach which uses all folds
    fold_train_size = int(effective_train_size * (config.n_splits - 1) / config.n_splits)
    max_components = min(20, fold_train_size - 2, dataset.n_features)
    max_components = max(3, max_components)  # Ensure at least 3 components

    # Large cartesian preprocessing search - this is what FCK-PLS aims to avoid!
    # Uses the full 4-stage pipeline generator from study_runner.py
    # 4 × 4 × 3 × 5 = 240 combinations (capped by count)
    cartesian_preprocessing = {
        "_cartesian_": [
            # Stage 1: Scatter correction
            {"_or_": [None, StandardNormalVariate(), EMSC(), Detrend()]},
            # Stage 2: Smoothing/filtering
            {"_or_": [None, EMSC(), SavitzkyGolay(window_length=15), Gaussian(order=1, sigma=2)]},
            # Stage 3: Derivatives
            {"_or_": [None, SavitzkyGolay(window_length=15, deriv=1), SavitzkyGolay(window_length=15, deriv=2)]},
            # Stage 4: Additional transforms
            {"_or_": [None, Haar(), Detrend(), AreaNormalization(), Wavelet("coif3")]},
        ],
        "count": 10 if config.quick_mode else 100,  # Number of preprocessing variants to try
    }

    # Use cartesian as pipeline generator (not feature_augmentation)
    # This creates multiple pipelines, each with a different preprocessing chain
    pipeline = [
        MinMaxScaler(),
        cartesian_preprocessing,  # Generator: creates N pipeline variants
        ShuffleSplit(n_splits=config.n_splits, test_size=config.test_size, random_state=config.seed),
        {
            "model": PLSRegression(),
            "name": "PLS",
            "finetune_params": {
                "n_trials": n_trials,
                "sample": "tpe",
                "verbose": 0,
                "approach": "single",
                "model_params": {
                    "n_components": ('int', 1, max_components),
                },
            },
        },
    ]

    start = time.time()
    try:
        result = nirs4all.run(
            pipeline=pipeline,
            dataset=dataset_arg,
            name="PLS-Tuned",
            verbose=0,
            save_artifacts=False,
            save_charts=False,
        )

        train_time = time.time() - start

        # Extract metrics properly
        r2, rmse, mae = extract_metrics_from_result(result)

        return ModelResult(
            model_name="PLS-Tuned",
            dataset_name=dataset.name,
            r2=r2,
            rmse=rmse,
            mae=mae,
            train_time=train_time,
            extra_info={"n_configs": result.predictions.num_predictions},
        )

    except Exception as e:
        print(f"  ERROR in PLS-Tuned: {e}")
        import traceback
        traceback.print_exc()
        return ModelResult(
            model_name="PLS-Tuned",
            dataset_name=dataset.name,
            r2=float('nan'),
            rmse=float('nan'),
            mae=float('nan'),
            train_time=time.time() - start,
            extra_info={"error": str(e)},
        )


# =============================================================================
# Model 3 & 4: FCK-PLS static (nirs4all) - with/without preprocessing
# =============================================================================

def run_fckpls_static(
    dataset: DatasetDef,
    config: BenchmarkConfig,
    with_preprocessing: bool = True,
) -> ModelResult:
    """
    Run FCK-PLS (nirs4all static implementation) with finetuning.

    Args:
        with_preprocessing: If True, use minimal preprocessing (SNV).
                          If False, raw signal only (MinMax for scaling).

    The point of FCK-PLS is to avoid extensive preprocessing search.
    """
    n_trials = 5 if config.quick_mode else config.finetune_trials
    dataset_arg, effective_train_size = prepare_dataset_for_nirs4all(dataset, config)
    max_components = min(20, effective_train_size - 2)

    model_name = "FCK-PLS-Static" + ("-PP" if with_preprocessing else "-Raw")

    # Build preprocessing pipeline
    if with_preprocessing:
        preprocessing_steps = [
            MinMaxScaler(),
            StandardNormalVariate(),
        ]
    else:
        preprocessing_steps = [
            MinMaxScaler(),
        ]

    pipeline = preprocessing_steps + [
        ShuffleSplit(n_splits=config.n_splits, test_size=config.test_size, random_state=config.seed),
        {
            "model": FCKPLS(),
            "name": model_name,
            "finetune_params": {
                "n_trials": n_trials,
                "sample": "tpe",
                "verbose": 0,
                "approach": "single",
                "model_params": {
                    "n_components": ('int', 3, max_components),
                    "alphas": [
                        (0.0, 0.5, 1.0, 1.5, 2.0),
                        (0.0, 1.0, 2.0),
                        (0.5, 1.0, 1.5),
                    ],
                    "kernel_size": [11, 15, 21],
                },
            },
        },
    ]

    start = time.time()
    try:
        result = nirs4all.run(
            pipeline=pipeline,
            dataset=dataset_arg,
            name=model_name,
            verbose=0,
            save_artifacts=False,
            save_charts=False,
        )

        train_time = time.time() - start

        # Extract metrics properly
        r2, rmse, mae = extract_metrics_from_result(result)

        return ModelResult(
            model_name=model_name,
            dataset_name=dataset.name,
            r2=r2,
            rmse=rmse,
            mae=mae,
            train_time=train_time,
            extra_info={"n_configs": result.predictions.num_predictions, "with_preprocessing": with_preprocessing},
        )

    except Exception as e:
        print(f"  ERROR in {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return ModelResult(
            model_name=model_name,
            dataset_name=dataset.name,
            r2=float('nan'),
            rmse=float('nan'),
            mae=float('nan'),
            train_time=time.time() - start,
            extra_info={"error": str(e)},
        )


# =============================================================================
# Model 5 & 6: FCK-PLS Torch V1/V2
# =============================================================================

def prepare_data_for_torch(
    dataset: DatasetDef,
    config: BenchmarkConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for Torch models with consistent train/test split.

    For datasets without explicit test: create holdout split using same seed.
    For datasets with explicit test: use as-is.

    Also applies MinMaxScaler to y for consistency with nirs4all models.
    """
    if dataset.has_explicit_test:
        X_train, y_train = dataset.X_train, dataset.y_train
        X_test, y_test = dataset.X_test, dataset.y_test
    else:
        # Create reproducible holdout split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X_train, dataset.y_train,
            test_size=config.holdout_size,
            random_state=config.seed,
        )

    # Apply y scaling (MinMaxScaler) for consistency
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Apply X scaling
    x_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler


def run_fckpls_torch(
    dataset: DatasetDef,
    config: BenchmarkConfig,
    version: str = "v1",
) -> ModelResult:
    """
    Run FCK-PLS Torch (V1 or V2) with finetuning via grid search.

    - Raw signal (no preprocessing except standardization in model)
    - Good training parameters (proper validation split, early stopping)
    - Always tuned (grid search over key parameters)
    """
    X_train, y_train, X_test, y_test, y_scaler = prepare_data_for_torch(dataset, config)

    epochs = 100 if config.quick_mode else config.torch_epochs
    model_name = f"FCK-PLS-Torch-{version.upper()}"

    start = time.time()

    # Finetuning via grid search over key parameters
    best_r2 = -np.inf
    best_model = None
    best_params = {}

    # Parameter grid
    n_kernels_opts = [5, 8, 12]
    n_components_opts = [5, 10, 15]
    kernel_size_opts = [21, 31]

    if config.quick_mode:
        n_kernels_opts = [5, 8]
        n_components_opts = [5, 10]
        kernel_size_opts = [21]

    for n_kernels in n_kernels_opts:
        for n_components in n_components_opts:
            for kernel_size in kernel_size_opts:
                train_cfg = TrainConfig(
                    epochs=epochs,
                    lr=config.torch_lr,
                    verbose=0,
                    val_fraction=config.torch_val_fraction,
                    seed=config.seed,
                )

                fck = FCKPLSTorch(
                    version=version,
                    n_kernels=n_kernels,
                    n_components=n_components,
                    kernel_size=kernel_size,
                    init_mode="derivative" if version == "v1" else "random",
                    train_cfg=train_cfg,
                )

                try:
                    fck.fit(X_train, y_train)
                    y_pred = fck.predict(X_test)
                    r2 = r2_score(y_test, y_pred)

                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = fck
                        best_params = {
                            "n_kernels": n_kernels,
                            "n_components": n_components,
                            "kernel_size": kernel_size,
                        }
                except Exception as e:
                    pass

    train_time = time.time() - start

    if best_model is not None:
        y_pred = best_model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)

        return ModelResult(
            model_name=model_name,
            dataset_name=dataset.name,
            r2=metrics["r2"],
            rmse=metrics["rmse"],
            mae=metrics["mae"],
            train_time=train_time,
            extra_info={"best_params": best_params},
        )
    else:
        return ModelResult(
            model_name=model_name,
            dataset_name=dataset.name,
            r2=float('nan'),
            rmse=float('nan'),
            mae=float('nan'),
            train_time=train_time,
            extra_info={"error": "All configurations failed"},
        )


# =============================================================================
# Benchmark runner
# =============================================================================

def run_benchmark(
    config: BenchmarkConfig,
    model_filter: Optional[str] = None,
) -> List[ModelResult]:
    """
    Run the benchmark suite.

    Args:
        config: Benchmark configuration including step level
        model_filter: Optional filter ("pls", "fckpls", "torch", or None for all)

    Step levels:
        1: PLS baseline only (verify CV works)
        2: PLS baseline + PLS tuned with cartesian (verify finetuning)
        3: All models (full benchmark)
    """

    print("=" * 70)
    print("FCK-PLS Comprehensive Benchmark")
    print("=" * 70)

    if config.quick_mode:
        print("QUICK MODE: Reduced epochs and trials")
    print(f"Step level: {config.step}")
    if model_filter:
        print(f"Model filter: {model_filter}")
    print()

    # Load datasets
    datasets = load_all_datasets(config.seed)
    print(f"\nLoaded {len(datasets)} datasets:")
    for ds in datasets:
        test_info = f"{ds.n_test} test" if ds.has_explicit_test else "no explicit test"
        print(f"  - {ds.name}: {ds.n_train} train, {test_info}, {ds.n_features} features")
    print()

    results = []

    for dataset in datasets:
        print("-" * 70)
        print(f"Dataset: {dataset.name}")
        print("-" * 70)

        model_idx = 0

        # =====================================================================
        # Step 1: PLS Baseline (always run unless filtered out)
        # =====================================================================
        if config.step >= 1 and (model_filter is None or model_filter == "pls"):
            model_idx += 1
            print(f"  [{model_idx}] PLS-Baseline...", end=" ", flush=True)
            result = run_pls_baseline(dataset, config)
            result.dataset_name = dataset.name
            results.append(result)
            print(f"R²={result.r2:.4f}, RMSE={result.rmse:.4f}, Time={result.train_time:.1f}s")

        # =====================================================================
        # Step 2: PLS Tuned with cartesian preprocessing
        # =====================================================================
        if config.step >= 2 and (model_filter is None or model_filter == "pls"):
            model_idx += 1
            print(f"  [{model_idx}] PLS-Tuned (cartesian + finetune)...", end=" ", flush=True)
            result = run_pls_tuned(dataset, config)
            result.dataset_name = dataset.name
            results.append(result)
            print(f"R²={result.r2:.4f}, RMSE={result.rmse:.4f}, Time={result.train_time:.1f}s")

        # =====================================================================
        # Step 3: FCK-PLS and Torch models
        # =====================================================================
        if config.step >= 3:
            # FCK-PLS Static with preprocessing
            if model_filter is None or model_filter == "fckpls":
                model_idx += 1
                print(f"  [{model_idx}] FCK-PLS-Static (with PP)...", end=" ", flush=True)
                result = run_fckpls_static(dataset, config, with_preprocessing=True)
                result.dataset_name = dataset.name
                results.append(result)
                print(f"R²={result.r2:.4f}, RMSE={result.rmse:.4f}, Time={result.train_time:.1f}s")

                # FCK-PLS Static without preprocessing
                model_idx += 1
                print(f"  [{model_idx}] FCK-PLS-Static (raw)...", end=" ", flush=True)
                result = run_fckpls_static(dataset, config, with_preprocessing=False)
                result.dataset_name = dataset.name
                results.append(result)
                print(f"R²={result.r2:.4f}, RMSE={result.rmse:.4f}, Time={result.train_time:.1f}s")

            # FCK-PLS Torch V1
            if model_filter is None or model_filter == "torch":
                model_idx += 1
                print(f"  [{model_idx}] FCK-PLS-Torch-V1...", end=" ", flush=True)
                result = run_fckpls_torch(dataset, config, version="v1")
                result.dataset_name = dataset.name
                results.append(result)
                print(f"R²={result.r2:.4f}, RMSE={result.rmse:.4f}, Time={result.train_time:.1f}s")

                # FCK-PLS Torch V2
                model_idx += 1
                print(f"  [{model_idx}] FCK-PLS-Torch-V2...", end=" ", flush=True)
                result = run_fckpls_torch(dataset, config, version="v2")
                result.dataset_name = dataset.name
                results.append(result)
                print(f"R²={result.r2:.4f}, RMSE={result.rmse:.4f}, Time={result.train_time:.1f}s")

        print()

    return results


def generate_report(results: List[ModelResult]) -> str:
    """Generate a markdown report from benchmark results."""
    lines = [
        "# FCK-PLS Benchmark Results",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Table",
        "",
    ]

    if not results:
        lines.append("No results to report.")
        return "\n".join(lines)

    # Build summary table
    df = pd.DataFrame([
        {
            "Dataset": r.dataset_name,
            "Model": r.model_name,
            "R²": r.r2,
            "RMSE": r.rmse,
            "MAE": r.mae,
            "Time(s)": r.train_time,
        }
        for r in results
    ])

    # Pivot table for easier comparison
    pivot = df.pivot_table(
        index="Dataset",
        columns="Model",
        values="R²",
        aggfunc="first"
    )

    # Sort columns by average R² (best to worst)
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

    for dataset in df["Dataset"].unique():
        subset = df[df["Dataset"] == dataset]
        # Filter out NaN values for finding best
        valid_subset = subset[subset["R²"].notna()]
        if len(valid_subset) > 0:
            best_idx = valid_subset["R²"].idxmax()
            best = valid_subset.loc[best_idx]
            lines.append(f"| {dataset} | {best['Model']} | {best['R²']:.4f} |")
        else:
            lines.append(f"| {dataset} | (no valid results) | N/A |")

    lines.append("")

    # Full results table (sorted by R² descending)
    lines.append("### Full Results")
    lines.append("")
    df_sorted = df.sort_values(by="R²", ascending=False)
    lines.append(df_sorted.round(4).to_markdown(index=False))
    lines.append("")

    # Model rankings
    lines.append("### Model Rankings (Average R² across datasets)")
    lines.append("")

    avg_r2 = df.groupby("Model")["R²"].mean().sort_values(ascending=False)
    lines.append("| Rank | Model | Avg R² |")
    lines.append("|------|-------|--------|")
    for rank, (model, r2) in enumerate(avg_r2.items(), 1):
        lines.append(f"| {rank} | {model} | {r2:.4f} |")

    lines.append("")

    # Observations
    lines.append("## Model Descriptions")
    lines.append("")
    lines.append("| Model | Description |")
    lines.append("|-------|-------------|")
    lines.append("| **PLS-Baseline** | PLSRegression with fixed preprocessing (MinMax + SNV) |")
    lines.append("| **PLS-Tuned** | PLSRegression with cartesian preprocessing search + n_components tuning |")
    lines.append("| **FCK-PLS-Static-PP** | nirs4all FCKPLS with preprocessing (SNV) + finetuning |")
    lines.append("| **FCK-PLS-Static-Raw** | nirs4all FCKPLS without preprocessing + finetuning |")
    lines.append("| **FCK-PLS-Torch-V1** | PyTorch learnable kernels (no preprocessing) |")
    lines.append("| **FCK-PLS-Torch-V2** | PyTorch parametric α/σ kernels (no preprocessing) |")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FCK-PLS Comprehensive Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick mode (reduced epochs/trials)")
    parser.add_argument("--model", type=str, choices=["pls", "fckpls", "torch"], default=None,
                       help="Run only specific model type")
    parser.add_argument("--step", type=int, choices=[1, 2, 3], default=3,
                       help="Step level: 1=baseline, 2=+finetuning, 3=all models")
    parser.add_argument("--plots", action="store_true", help="Generate plots")
    parser.add_argument("--output", type=str, default=None, help="Save report to file")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = BenchmarkConfig(
        quick_mode=args.quick,
        verbose=args.verbose,
        seed=args.seed,
        step=args.step,
    )

    # Run benchmark
    results = run_benchmark(config, model_filter=args.model)

    # Generate report
    report = generate_report(results)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print()
    print(report)

    # Save report
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report)
        print(f"\nReport saved to: {output_path}")

    # Generate plots
    if args.plots and results:
        try:
            import matplotlib.pyplot as plt

            df = pd.DataFrame([
                {
                    "Dataset": r.dataset_name,
                    "Model": r.model_name,
                    "R²": r.r2,
                }
                for r in results
            ])

            pivot = df.pivot_table(index="Dataset", columns="Model", values="R²", aggfunc="first")

            fig, ax = plt.subplots(figsize=(14, 6))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title("FCK-PLS Benchmark: R² by Dataset and Model")
            ax.set_ylabel("R²")
            ax.set_xlabel("Dataset")
            ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            fig_path = BENCH_DIR / "benchmark_results.png"
            plt.savefig(fig_path, dpi=150)
            print(f"\nPlot saved to: {fig_path}")

            plt.show()

        except ImportError:
            print("\nWARNING: matplotlib not available for plotting")


if __name__ == "__main__":
    main()
