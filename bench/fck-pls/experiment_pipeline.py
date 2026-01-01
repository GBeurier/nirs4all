"""
Experiment: FCK-PLS in nirs4all Pipelines
=========================================

This script tests the new FCK-PLS Torch implementation within nirs4all pipelines,
comparing it to standard approaches with various preprocessing combinations.

Usage:
    python experiment_pipeline.py
    python experiment_pipeline.py --quick
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# nirs4all imports
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.data import DatasetConfigs

# Preprocessing
from nirs4all.operators.transforms import StandardNormalVariate as SNV, MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import SavitzkyGolay
from nirs4all.operators.transforms import ASLSBaseline as ALSBaseline, Detrend

# Models
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from nirs4all.operators.models.sklearn import OPLS, FCKPLS

# Splitters
from sklearn.model_selection import ShuffleSplit

# Local
sys.path.insert(0, str(Path(__file__).parent))
from fckpls_torch import FCKPLSTorch, TrainConfig

# Synthetic generator
sys.path.insert(0, str(ROOT / "bench" / "synthetic"))
from generator import SyntheticNIRSGenerator


# =============================================================================
# Custom Wrapper for FCK-PLS Torch (sklearn-compatible)
# =============================================================================

class FCKPLSTorchWrapper:
    """
    Wrapper to make FCK-PLS Torch work in nirs4all pipelines.

    Since FCKPLSTorch is already sklearn-compatible, this is mainly
    for easy configuration in pipeline definitions.
    """

    def __init__(
        self,
        version: str = "v1",
        n_kernels: int = 16,
        kernel_size: int = 31,
        n_components: int = 10,
        epochs: int = 200,
        lr: float = 1e-3,
        verbose: int = 0,
        **kwargs,
    ):
        self.version = version
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.n_components = n_components
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.kwargs = kwargs

        self._model = None

    def fit(self, X, y):
        cfg = TrainConfig(
            epochs=self.epochs,
            lr=self.lr,
            verbose=self.verbose,
        )
        self._model = FCKPLSTorch(
            version=self.version,
            n_kernels=self.n_kernels,
            kernel_size=self.kernel_size,
            n_components=self.n_components,
            train_cfg=cfg,
            **self.kwargs,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def transform(self, X):
        return self._model.transform(X)

    def get_params(self, deep=True):
        return {
            "version": self.version,
            "n_kernels": self.n_kernels,
            "kernel_size": self.kernel_size,
            "n_components": self.n_components,
            "epochs": self.epochs,
            "lr": self.lr,
            "verbose": self.verbose,
            **self.kwargs,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# =============================================================================
# Pipeline Definitions
# =============================================================================

def get_pipelines(quick: bool = False, n_components: int = 10) -> Dict[str, list]:
    """Define pipelines to compare."""
    epochs = 100 if quick else 250

    pipelines = {}

    # 1. Simple PLS
    pipelines["PLS"] = [
        StandardScaler(),
        ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        {"model": PLSRegression(n_components=n_components)},
    ]

    # 2. PLS + SNV
    pipelines["SNV + PLS"] = [
        SNV(),
        ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        {"model": PLSRegression(n_components=n_components)},
    ]

    # 3. PLS + Derivative
    pipelines["SG1 + PLS"] = [
        SavitzkyGolay(window_length=11, polyorder=2, deriv=1),
        StandardScaler(),
        ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        {"model": PLSRegression(n_components=n_components)},
    ]

    # 4. OPLS
    try:
        pipelines["OPLS"] = [
            StandardScaler(),
            ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
            {"model": OPLS(n_components=2, pls_components=n_components)},
        ]
    except:
        pass

    # 5. Original FCK-PLS
    pipelines["FCK-PLS (orig)"] = [
        StandardScaler(),
        ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        {"model": FCKPLS(n_components=n_components, kernel_size=15)},
    ]

    # 6. FCK-PLS Torch V1
    pipelines["FCK-PLS Torch V1"] = [
        ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        {"model": FCKPLSTorchWrapper(
            version="v1",
            n_kernels=16,
            kernel_size=31,
            n_components=n_components,
            epochs=epochs,
        )},
    ]

    # 7. FCK-PLS Torch V2
    pipelines["FCK-PLS Torch V2"] = [
        ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        {"model": FCKPLSTorchWrapper(
            version="v2",
            n_kernels=16,
            kernel_size=31,
            n_components=n_components,
            epochs=epochs,
        )},
    ]

    # 8. SNV + FCK-PLS Torch (test if preprocessing helps the new method)
    pipelines["SNV + FCK-PLS Torch V1"] = [
        SNV(),
        ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        {"model": FCKPLSTorchWrapper(
            version="v1",
            n_kernels=16,
            kernel_size=31,
            n_components=n_components,
            epochs=epochs,
        )},
    ]

    # 9. Baseline + FCK-PLS Torch
    pipelines["ALS + FCK-PLS Torch V1"] = [
        ALSBaseline(lam=1e5, p=0.01),
        ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        {"model": FCKPLSTorchWrapper(
            version="v1",
            n_kernels=16,
            kernel_size=31,
            n_components=n_components,
            epochs=epochs,
        )},
    ]

    return pipelines


# =============================================================================
# Manual Evaluation (without nirs4all runner for flexibility)
# =============================================================================

def evaluate_pipeline_manual(
    pipeline_steps: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
) -> Dict[str, Any]:
    """Manually evaluate a pipeline."""

    X_tr = X_train.copy()
    X_te = X_test.copy()
    model = None

    t0 = time.time()

    for step in pipeline_steps:
        # Skip splitters for manual evaluation
        if isinstance(step, ShuffleSplit):
            continue

        # Handle model step
        if isinstance(step, dict) and "model" in step:
            model = step["model"]
            continue

        # Handle transformer
        if hasattr(step, "fit_transform"):
            X_tr = step.fit_transform(X_tr)
            X_te = step.transform(X_te)
        elif hasattr(step, "transform"):
            X_tr = step.transform(X_tr)
            X_te = step.transform(X_te)

    # Fit model
    if model is None:
        return {"name": name, "error": "No model found"}

    try:
        model.fit(X_tr, y_train)
        y_train_pred = model.predict(X_tr)
        y_test_pred = model.predict(X_te)
    except Exception as e:
        return {"name": name, "error": str(e)}

    fit_time = time.time() - t0

    # Flatten predictions if needed
    y_train_pred = np.asarray(y_train_pred).ravel()
    y_test_pred = np.asarray(y_test_pred).ravel()

    return {
        "name": name,
        "rmse_train": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "r2_train": r2_score(y_train, y_train_pred),
        "r2_test": r2_score(y_test, y_test_pred),
        "fit_time": fit_time,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(quick: bool = False):
    """Run the full experiment."""

    print("="*60)
    print("FCK-PLS Pipeline Experiment")
    print("="*60)
    print(f"Quick mode: {quick}")

    # Generate synthetic data
    print("\nGenerating synthetic data...")

    gen = SyntheticNIRSGenerator(
        wavelength_start=1000,
        wavelength_end=2500,
        wavelength_step=4,
        complexity="realistic",
        random_state=42,
    )
    X, C, E = gen.generate(n_samples=300, concentration_method="correlated")
    y = C[:, 1]  # Protein

    X_train, X_test = X[:250], X[250:]
    y_train, y_test = y[:250], y[250:]

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Get pipelines
    pipelines = get_pipelines(quick=quick, n_components=10)

    # Evaluate
    results = []

    for name, steps in pipelines.items():
        print(f"\nEvaluating: {name}...")
        result = evaluate_pipeline_manual(
            pipeline_steps=steps,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            name=name,
        )
        results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  RMSE: train={result['rmse_train']:.4f}, test={result['rmse_test']:.4f}")
            print(f"  R²: train={result['r2_train']:.4f}, test={result['r2_test']:.4f}")
            print(f"  Time: {result['fit_time']:.2f}s")

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    df = pd.DataFrame(results)
    if "error" in df.columns:
        df = df[df["error"].isna()].drop(columns=["error"])

    df = df.sort_values("r2_test", ascending=False)
    print(df.to_string(index=False))

    # Save
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "pipeline_experiment_results.csv", index=False)
    print(f"\nResults saved to {output_dir / 'pipeline_experiment_results.csv'}")

    return df


def run_preprocessing_analysis(quick: bool = False):
    """
    Analyze which preprocessing is best for FCK-PLS Torch.

    Hypothesis: FCK-PLS might need different preprocessing than PLS
    since it learns its own spectral transformations.
    """

    print("\n" + "="*60)
    print("Preprocessing Analysis for FCK-PLS Torch")
    print("="*60)

    # Generate data with strong baseline/scatter
    gen = SyntheticNIRSGenerator(
        wavelength_start=1000,
        wavelength_end=2500,
        wavelength_step=4,
        complexity="complex",  # More challenging
        random_state=42,
    )
    X, C, E = gen.generate(n_samples=300, concentration_method="correlated")
    y = C[:, 2]  # Lipid

    X_train, X_test = X[:250], X[250:]
    y_train, y_test = y[:250], y[250:]

    # Preprocessing options
    preprocessings = {
        "None": None,
        "StandardScaler": StandardScaler(),
        "SNV": SNV(),
        "MSC": MSC(),
        "SG1": SavitzkyGolay(window_length=11, polyorder=2, deriv=1),
        "SG2": SavitzkyGolay(window_length=11, polyorder=2, deriv=2),
        "ALS": ALSBaseline(lam=1e5, p=0.01),
        "Detrend": Detrend(order=1),
    }

    epochs = 100 if quick else 200

    results = []

    for pp_name, pp in preprocessings.items():
        print(f"\n{pp_name}:")

        # Apply preprocessing
        if pp is None:
            X_tr = X_train
            X_te = X_test
        else:
            try:
                X_tr = pp.fit_transform(X_train)
                X_te = pp.transform(X_test)
            except Exception as e:
                print(f"  Preprocessing failed: {e}")
                continue

        # Test both PLS and FCK-PLS Torch
        for model_name, model in [
            ("PLS", PLSRegression(n_components=10)),
            ("FCK-PLS Torch V1", FCKPLSTorchWrapper(
                version="v1", n_kernels=16, n_components=10, epochs=epochs
            )),
        ]:
            try:
                model.fit(X_tr, y_train)
                y_pred = model.predict(X_te).ravel()
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                results.append({
                    "preprocessing": pp_name,
                    "model": model_name,
                    "r2_test": r2,
                    "rmse_test": rmse,
                })

                print(f"  {model_name}: R²={r2:.4f}")

            except Exception as e:
                print(f"  {model_name}: ERROR - {e}")

    # Summary
    df = pd.DataFrame(results)
    pivot = df.pivot(index="preprocessing", columns="model", values="r2_test")
    print("\n" + "="*40)
    print("R² by Preprocessing and Model")
    print("="*40)
    print(pivot.to_string())

    # Save
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "preprocessing_analysis.csv", index=False)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--preprocessing", action="store_true",
                       help="Run preprocessing analysis")
    args = parser.parse_args()

    # Main experiment
    run_experiment(quick=args.quick)

    # Preprocessing analysis
    if args.preprocessing:
        run_preprocessing_analysis(quick=args.quick)


if __name__ == "__main__":
    main()
