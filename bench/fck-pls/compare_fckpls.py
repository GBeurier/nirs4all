"""
FCK-PLS Comparison Study
========================

Compares different PLS variants on regression tasks:
- Standard PLS (sklearn)
- OPLS
- Original FCK-PLS (scipy/sklearn based)
- New FCK-PLS Torch V1 (learnable kernels)
- New FCK-PLS Torch V2 (learnable alpha/sigma)

Also compares with and without preprocessing.

Usage:
    python compare_fckpls.py
    python compare_fckpls.py --quick  # fast mode with fewer epochs
    python compare_fckpls.py --plot   # show plots
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# nirs4all imports
from nirs4all.operators.models.sklearn import OPLS, FCKPLS
from nirs4all.operators.transforms import StandardNormalVariate as SNV, MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import SavitzkyGolay

# Local imports
from fckpls_torch import FCKPLSTorch, TrainConfig, create_fckpls_v1, create_fckpls_v2

# Synthetic data generator
sys.path.insert(0, str(ROOT / "bench" / "synthetic"))
from generator import SyntheticNIRSGenerator, ComponentLibrary


# =============================================================================
# Data Loading
# =============================================================================

def load_sample_regression_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load sample regression data from examples."""
    import gzip

    data_dir = ROOT / "examples" / "sample_data" / "regression"

    # CSV files use semicolon as delimiter
    with gzip.open(data_dir / "Xcal.csv.gz", "rt") as f:
        X_train = pd.read_csv(f, header=None, sep=";").values.astype(float)
    with gzip.open(data_dir / "Ycal.csv.gz", "rt") as f:
        y_train = pd.read_csv(f, header=None, sep=";").values.ravel().astype(float)
    with gzip.open(data_dir / "Xval.csv.gz", "rt") as f:
        X_test = pd.read_csv(f, header=None, sep=";").values.astype(float)
    with gzip.open(data_dir / "Yval.csv.gz", "rt") as f:
        y_test = pd.read_csv(f, header=None, sep=";").values.ravel().astype(float)

    return X_train, y_train, X_test, y_test


def generate_synthetic_simple(n_train: int = 200, n_test: int = 50, seed: int = 42) -> dict:
    """Generate simple synthetic data with low noise."""
    gen = SyntheticNIRSGenerator(
        wavelength_start=1000,
        wavelength_end=2500,
        wavelength_step=4,
        complexity="simple",
        random_state=seed,
    )
    X, C, E = gen.generate(n_samples=n_train + n_test, concentration_method="dirichlet")
    y = C[:, 0]  # First component as target

    return {
        "name": "synthetic_simple",
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test": X[n_train:],
        "y_test": y[n_train:],
        "wavelengths": gen.wavelengths,
    }


def generate_synthetic_realistic(n_train: int = 200, n_test: int = 50, seed: int = 42) -> dict:
    """Generate realistic synthetic data with moderate effects."""
    gen = SyntheticNIRSGenerator(
        wavelength_start=1000,
        wavelength_end=2500,
        wavelength_step=4,
        complexity="realistic",
        random_state=seed,
    )
    X, C, E = gen.generate(n_samples=n_train + n_test, concentration_method="correlated")
    y = C[:, 1]  # Protein as target

    return {
        "name": "synthetic_realistic",
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test": X[n_train:],
        "y_test": y[n_train:],
        "wavelengths": gen.wavelengths,
    }


def generate_synthetic_complex(n_train: int = 200, n_test: int = 50, seed: int = 42) -> dict:
    """Generate complex synthetic data with heavy effects and noise."""
    gen = SyntheticNIRSGenerator(
        wavelength_start=1000,
        wavelength_end=2500,
        wavelength_step=4,
        complexity="complex",
        random_state=seed,
    )
    X, C, E = gen.generate(
        n_samples=n_train + n_test,
        concentration_method="correlated",
        include_batch_effects=True,
        n_batches=3,
    )
    y = C[:, 2]  # Lipid as target

    return {
        "name": "synthetic_complex",
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test": X[n_train:],
        "y_test": y[n_train:],
        "wavelengths": gen.wavelengths,
    }


# =============================================================================
# Model Factories
# =============================================================================

def get_models(quick: bool = False, n_components: int = 10) -> Dict[str, Any]:
    """Get all models to compare."""
    epochs = 100 if quick else 300

    models = {}

    # 1. Standard PLS
    models["PLS"] = PLSRegression(n_components=n_components)

    # 2. OPLS
    try:
        models["OPLS"] = OPLS(n_components=2, pls_components=n_components)
    except Exception as e:
        print(f"Warning: OPLS not available: {e}")

    # 3. Original FCK-PLS (sklearn-based)
    models["FCK-PLS (original)"] = FCKPLS(
        n_components=n_components,
        alphas=(0.0, 0.5, 1.0, 1.5, 2.0),
        sigmas=(2.0,),
        kernel_size=15,
    )

    # 4. FCK-PLS Torch V1 (learnable kernels)
    cfg_v1 = TrainConfig(
        epochs=epochs,
        lr=1e-3,
        early_stopping_patience=30,
        verbose=0,
    )
    models["FCK-PLS Torch V1"] = FCKPLSTorch(
        version="v1",
        n_kernels=16,
        kernel_size=31,
        n_components=n_components,
        init_mode="fractional",  # Initialize with fractional-like kernels
        train_cfg=cfg_v1,
    )

    # 5. FCK-PLS Torch V2 (alpha/sigma)
    cfg_v2 = TrainConfig(
        epochs=epochs,
        lr=1e-3,
        early_stopping_patience=30,
        verbose=0,
    )
    models["FCK-PLS Torch V2"] = FCKPLSTorch(
        version="v2",
        n_kernels=16,
        kernel_size=31,
        n_components=n_components,
        alpha_max=2.0,
        tau=1.0,
        train_cfg=cfg_v2,
    )

    return models


def get_preprocessings() -> Dict[str, Any]:
    """Get preprocessing transforms to compare."""
    return {
        "None": None,
        "SNV": SNV(),
        "SG1": SavitzkyGolay(window_length=11, polyorder=2, deriv=1),
        "SG2": SavitzkyGolay(window_length=11, polyorder=2, deriv=2),
        "MSC": MSC(),
    }


# =============================================================================
# Evaluation
# =============================================================================

@dataclass
class Result:
    """Single evaluation result."""
    model_name: str
    dataset_name: str
    preprocessing: str
    rmse_train: float
    rmse_test: float
    r2_train: float
    r2_test: float
    fit_time: float
    n_features: int


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    dataset_name: str,
    preprocessing_name: str = "None",
) -> Result:
    """Evaluate a single model."""
    n_features = X_train.shape[1]

    # Fit
    t0 = time.time()
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"  Error fitting {model_name}: {e}")
        return Result(
            model_name=model_name,
            dataset_name=dataset_name,
            preprocessing=preprocessing_name,
            rmse_train=np.nan,
            rmse_test=np.nan,
            r2_train=np.nan,
            r2_test=np.nan,
            fit_time=np.nan,
            n_features=n_features,
        )
    fit_time = time.time() - t0

    # Predict
    try:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    except Exception as e:
        print(f"  Error predicting {model_name}: {e}")
        return Result(
            model_name=model_name,
            dataset_name=dataset_name,
            preprocessing=preprocessing_name,
            rmse_train=np.nan,
            rmse_test=np.nan,
            r2_train=np.nan,
            r2_test=np.nan,
            fit_time=fit_time,
            n_features=n_features,
        )

    # Metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return Result(
        model_name=model_name,
        dataset_name=dataset_name,
        preprocessing=preprocessing_name,
        rmse_train=rmse_train,
        rmse_test=rmse_test,
        r2_train=r2_train,
        r2_test=r2_test,
        fit_time=fit_time,
        n_features=n_features,
    )


def apply_preprocessing(
    X_train: np.ndarray,
    X_test: np.ndarray,
    preprocessing: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply preprocessing to train and test data."""
    if preprocessing is None:
        return X_train, X_test

    # Fit on train, transform both
    X_train_pp = preprocessing.fit_transform(X_train)
    X_test_pp = preprocessing.transform(X_test)

    return X_train_pp, X_test_pp


def run_comparison(
    datasets: List[dict],
    quick: bool = False,
    with_preprocessing: bool = True,
    n_components: int = 10,
) -> pd.DataFrame:
    """Run full comparison study."""
    results: List[Result] = []

    for data in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {data['name']}")
        print(f"  Train: {data['X_train'].shape}, Test: {data['X_test'].shape}")
        print(f"{'='*60}")

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        # Get preprocessing options
        if with_preprocessing:
            preprocessings = get_preprocessings()
        else:
            preprocessings = {"None": None}

        for pp_name, pp in preprocessings.items():
            print(f"\n  Preprocessing: {pp_name}")

            # Apply preprocessing
            try:
                X_train_pp, X_test_pp = apply_preprocessing(X_train, X_test, pp)
            except Exception as e:
                print(f"    Error in preprocessing: {e}")
                continue

            # Get fresh models for each preprocessing
            models = get_models(quick=quick, n_components=n_components)

            for model_name, model in models.items():
                print(f"    {model_name}...", end=" ", flush=True)

                # Clone model if sklearn-compatible
                if hasattr(model, "get_params"):
                    try:
                        model = model.__class__(**model.get_params())
                    except:
                        pass

                result = evaluate_model(
                    model=model,
                    X_train=X_train_pp,
                    y_train=y_train,
                    X_test=X_test_pp,
                    y_test=y_test,
                    model_name=model_name,
                    dataset_name=data["name"],
                    preprocessing_name=pp_name,
                )
                results.append(result)

                print(f"RMSE={result.rmse_test:.4f}, R²={result.r2_test:.4f}, time={result.fit_time:.2f}s")

    # Convert to DataFrame
    df = pd.DataFrame([vars(r) for r in results])
    return df


# =============================================================================
# Visualization
# =============================================================================

def plot_results(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot comparison results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Matplotlib/seaborn not available for plotting")
        return

    # Filter out NaN results
    df = df.dropna(subset=["rmse_test", "r2_test"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. RMSE by model and dataset
    ax = axes[0, 0]
    pivot = df.pivot_table(
        values="rmse_test",
        index="model_name",
        columns="dataset_name",
        aggfunc="mean",
    )
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Test RMSE by Model and Dataset")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.legend(title="Dataset", loc="upper right")
    ax.tick_params(axis='x', rotation=45)

    # 2. R² by model and dataset
    ax = axes[0, 1]
    pivot = df.pivot_table(
        values="r2_test",
        index="model_name",
        columns="dataset_name",
        aggfunc="mean",
    )
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Test R² by Model and Dataset")
    ax.set_xlabel("Model")
    ax.set_ylabel("R²")
    ax.legend(title="Dataset", loc="lower right")
    ax.tick_params(axis='x', rotation=45)

    # 3. Effect of preprocessing
    ax = axes[1, 0]
    pivot = df.pivot_table(
        values="r2_test",
        index="preprocessing",
        columns="model_name",
        aggfunc="mean",
    )
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Test R² by Preprocessing")
    ax.set_xlabel("Preprocessing")
    ax.set_ylabel("R²")
    ax.legend(title="Model", loc="lower right", fontsize=8)
    ax.tick_params(axis='x', rotation=45)

    # 4. Fit time comparison
    ax = axes[1, 1]
    pivot = df.pivot_table(
        values="fit_time",
        index="model_name",
        columns="dataset_name",
        aggfunc="mean",
    )
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Fit Time by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Time (s)")
    ax.legend(title="Dataset", loc="upper right")
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_kernels(model: FCKPLSTorch, save_path: Optional[Path] = None):
    """Plot learned kernels from FCK-PLS Torch."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available")
        return

    kernels = model.get_kernels()
    n_kernels = kernels.shape[0]

    n_cols = 4
    n_rows = (n_kernels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    for i, kernel in enumerate(kernels):
        ax = axes[i]
        x = np.arange(len(kernel)) - len(kernel) // 2
        ax.plot(x, kernel, "b-", linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Kernel {i+1}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Value")

    # Hide unused axes
    for i in range(n_kernels, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"Learned Kernels ({model.version})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FCK-PLS Comparison Study")
    parser.add_argument("--quick", action="store_true", help="Fast mode with fewer epochs")
    parser.add_argument("--plot", action="store_true", help="Show plots")
    parser.add_argument("--save", type=str, default=None, help="Save results to CSV")
    parser.add_argument("--no-preprocessing", action="store_true", help="Skip preprocessing comparison")
    parser.add_argument("--n-components", type=int, default=10, help="Number of PLS components")
    parser.add_argument("--n-train", type=int, default=200, help="Number of training samples for synthetic")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("="*60)
    print("FCK-PLS Comparison Study")
    print("="*60)
    print(f"Quick mode: {args.quick}")
    print(f"With preprocessing: {not args.no_preprocessing}")
    print(f"N components: {args.n_components}")
    print()

    # Prepare datasets
    print("Loading/generating datasets...")
    datasets = []

    # 1. Sample regression data
    try:
        X_train, y_train, X_test, y_test = load_sample_regression_data()
        datasets.append({
            "name": "sample_regression",
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        })
        print(f"  Loaded sample_regression: {X_train.shape}")
    except Exception as e:
        print(f"  Could not load sample_regression: {e}")

    # 2. Synthetic datasets
    datasets.append(generate_synthetic_simple(n_train=args.n_train, seed=args.seed))
    print(f"  Generated synthetic_simple")

    datasets.append(generate_synthetic_realistic(n_train=args.n_train, seed=args.seed))
    print(f"  Generated synthetic_realistic")

    datasets.append(generate_synthetic_complex(n_train=args.n_train, seed=args.seed))
    print(f"  Generated synthetic_complex")

    # Run comparison
    df = run_comparison(
        datasets=datasets,
        quick=args.quick,
        with_preprocessing=not args.no_preprocessing,
        n_components=args.n_components,
    )

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Filter out NaN results for summary
    df_valid = df.dropna(subset=["r2_test"])

    if df_valid.empty:
        print("\nNo valid results to summarize.")
    else:
        # Best model per dataset
        print("\nBest model per dataset (by test R²):")
        for dataset in df_valid["dataset_name"].unique():
            sub = df_valid[df_valid["dataset_name"] == dataset]
            if not sub.empty:
                best = sub.loc[sub["r2_test"].idxmax()]
                print(f"  {dataset}: {best['model_name']} (R²={best['r2_test']:.4f}, pp={best['preprocessing']})")

        # Average performance
        print("\nAverage test R² across all datasets:")
        avg = df_valid.groupby("model_name")["r2_test"].mean().sort_values(ascending=False)
        for model, r2 in avg.items():
            print(f"  {model}: {r2:.4f}")

    # Save results
    if args.save:
        df.to_csv(args.save, index=False)
        print(f"\nResults saved to {args.save}")

    # Plots
    if args.plot:
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)

        plot_results(df_valid, save_path=output_dir / "comparison_results.png")

        # Plot kernels from a fitted Torch model
        print("\nTraining a FCK-PLS Torch V1 for kernel visualization...")
        data = datasets[1]  # Use simple synthetic
        model_v1 = create_fckpls_v1(
            n_kernels=8,
            n_components=10,
            epochs=100 if args.quick else 200,
            verbose=1,
        )
        model_v1.fit(data["X_train"], data["y_train"])
        plot_kernels(model_v1, save_path=output_dir / "kernels_v1.png")


if __name__ == "__main__":
    main()
