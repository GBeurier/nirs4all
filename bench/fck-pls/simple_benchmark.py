from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cross_decomposition import PLSRegression


import nirs4all
from nirs4all.data.synthetic import SyntheticNIRSGenerator, ComponentLibrary, SyntheticDatasetBuilder
from nirs4all.operators.transforms import (
    StandardNormalVariate as SNV,
    MultiplicativeScatterCorrection as MSC,
    FirstDerivative,
    SecondDerivative,
    SavitzkyGolay,
    Detrend,
    Gaussian,
    Haar,
    Wavelet,
    ASLSBaseline
)
from nirs4all.operators.transforms.nirs import (
    AreaNormalization,
    ExtendedMultiplicativeScatterCorrection as EMSC,
)
from nirs4all.operators.splitters import SPXYGFold

from nirs4all.operators.models import FCKPLS
from fckpls_torch import FCKPLSTorch, TrainConfig, fckpls_v1, fckpls_v2

def get_synthetic_dataset(n_samples=100, n_features=50, random_state=42, name="Synthetic Dataset"):
    """Create reproducible test dataset."""
    return (
        SyntheticDatasetBuilder(n_samples=n_samples, random_state=random_state, name=name)
        .with_features(
            wavelength_range=(1000, 2500),
            complexity="realistic",
            components=["water", "protein", "lipid", "starch"]
        )
        .with_targets(
            distribution="lognormal",
            range=(5, 50),
            component="protein"
        )
        .with_nonlinear_targets(
            interactions="polynomial",
            interaction_strength=0.3,
            hidden_factors=3
        )
        .with_partitions(train_ratio=0.8)
        .build()
    )

synthetic_dataset_small = get_synthetic_dataset(n_samples=150, n_features=500, name="Small Synthetic Dataset")
synthetic_dataset_mid = get_synthetic_dataset(n_samples=150, n_features=2500, name="Mid Synthetic Dataset")
synthetic_dataset_large = get_synthetic_dataset(n_samples=1500, n_features=2500, name="Large Synthetic Dataset")


print("Datasets created:")
print(" - SMALL:\n", synthetic_dataset_small)
print(" - MID:\n", synthetic_dataset_mid)
print(" - LARGE:\n", synthetic_dataset_large)


pls_max_components = 25

# PLS hyperparameter search space
pls_params = {
    "n_components": ('int', 3, pls_max_components),
}

pipeline_pls_pp = [
    {"_or_": [None, ASLSBaseline()]},
    {"_or_": [None, StandardScaler(), MinMaxScaler()]},
    {"y_processing": MinMaxScaler},
    SPXYGFold(n_splits=3),
    {
        "_cartesian_": [
            {"_or_": [None, SNV(), EMSC(), Detrend()]},
            {"_or_": [None, EMSC(), SavitzkyGolay(window_length=15), Gaussian(order=1, sigma=2)]},
            {"_or_": [None, SavitzkyGolay(window_length=15, deriv=1), SavitzkyGolay(window_length=15, deriv=2)]},
            {"_or_": [None, Haar(), Detrend(), AreaNormalization(), Wavelet("coif3")]},
        ],
        # "count": 100,
    },
    {
        "model": PLSRegression,
        "name": "PLS_pp_tuned",
        "finetune_params": {
            "n_trials": 25,
            "sample": "tpe",
            "verbose": 2,
            "approach": "grouped",
            "model_params": pls_params,
        },
    },
]




# ==============================================================================
# FCK-PLS V1 Configuration (Learnable Free Kernels)
# ==============================================================================
# V1 uses fully learnable kernels - more stable optimization, recommended for
# production use. Focus tuning on: n_kernels, kernel_size, n_components,
# learning rate, and regularization.

# All tunable parameters go in model_params (they're passed to the model constructor)
fckpls_v1_model_params = {
    # Architecture
    'n_kernels': ('int', 2, 16),              # Number of conv kernels
    'kernel_size': {'type': 'int', 'min': 5, 'max': 51, 'step': 2},  # Kernel size (odd, larger = smoother)
    'n_components': ('int', 3, pls_max_components),  # PLS components

    # PLS configuration
    'ridge_lambda': ('float_log', 1e-5, 1e-1),  # Ridge regularization
    'pls_mode': ['deflation', 'svd'],           # PLS mode

    # Kernel initialization (v1 specific)
    'init_mode': ['random', 'derivative', 'fractional'],  # How to initialize kernels

    # Regularization (passed to model params, used in kernel_regularization)
    'reg_smooth': ('float_log', 1e-5, 1e-1),   # Kernel smoothness
    'reg_zeromean': ('float_log', 1e-5, 1e-1), # Zero-mean prior (derivative-like)
    'reg_l2': ('float_log', 1e-6, 1e-2),       # Weight decay on kernels
}

# Fixed training params for search (epochs reduced for speed)
fckpls_v1_train_params_search = {
    'epochs': 100,
    'lr': 1e-3,                               # Fixed learning rate during search
    'patience': 20,                           # Early stopping patience
    'batch_size': 2048,
}

fckpls_v1_train_params_final = {
    # Final training with best params (thorough)
    'epochs': 800,
    'lr': 1e-4,
    'patience': 100,
    'batch_size': 2048,
}


# ==============================================================================
# FCK-PLS V2 Configuration (Alpha/Sigma Parametric Kernels)
# ==============================================================================
# V2 uses learnable alpha/sigma parameters to build fractional kernels.
# More interpretable but potentially less stable.
# Focus tuning on: n_kernels, alpha_max, tau, and learning rate.

fckpls_v2_model_params = {
    # Architecture
    'n_kernels': ('int', 2, 16),              # Number of conv kernels
    'kernel_size': {'type': 'int', 'min': 5, 'max': 51, 'step': 2},  # Kernel size (odd)
    'n_components': ('int', 3, pls_max_components),  # PLS components

    # PLS configuration
    'ridge_lambda': ('float_log', 1e-5, 1e-1),  # Ridge regularization
    'pls_mode': ['deflation', 'svd'],           # PLS mode

    # Fractional kernel params (v2 specific)
    'alpha_max': ('float', 1.5, 3.0),   # Maximum fractional order (key param!)
    'tau': ('float', 0.5, 2.0),         # Smoothness for sign approximation

    # Regularization for v2 (alpha/sigma specific)
    'alpha_w': ('float_log', 1e-5, 1e-2),     # Regularization on alpha params
    'sigma_w': ('float_log', 1e-5, 1e-2),     # Regularization on sigma params
}

fckpls_v2_train_params_search = {
    # Training during hyperparameter search (fast)
    'epochs': 150,  # V2 may need more epochs for parametric learning
    'lr': 1e-3,                               # Fixed learning rate during search
    'patience': 25,
    'batch_size': 2048,
}

fckpls_v2_train_params_final = {
    # Final training with best params (thorough)
    'epochs': 800,  # More epochs for parametric convergence
    'lr': 1e-4,
    'patience': 100,
    'batch_size': 2048,
}

# FCK-PLS hyperparameter search space
fckpls_params = {
    "n_components": ('int', 3, pls_max_components),
    "alphas": {'type': 'sorted_tuple', 'length': ('int', 2, 16), 'min': 0.0, 'max': 3.0},
    "kernel_size": {'type': 'int', 'min': 5, 'max': 51, 'step': 2},
}

fck_pls_n_trials = 500

pipeline_fck_pls = [
    {"_or_": [None, ASLSBaseline()]},
    {"_or_": [None, StandardScaler(), MinMaxScaler()]},
    {"y_processing": MinMaxScaler},
    SPXYGFold(n_splits=3),
    {
        "model": fckpls_v1,
        "name": "FCK-PLS-v1-tuned",
        "finetune_params": {
            "n_trials": fck_pls_n_trials,
            "sample": "tpe",
            "verbose": 1,
            "approach": "grouped",
            "model_params": fckpls_v1_model_params,
            "train_params": fckpls_v1_train_params_search,
        },
        "train_params": fckpls_v1_train_params_final,
    },
    {
        "model": fckpls_v2,
        "name": "FCK-PLS-v2-tuned",
        "finetune_params": {
            "n_trials": fck_pls_n_trials,
            "sample": "tpe",
            "verbose": 1,
            "approach": "grouped",
            "model_params": fckpls_v2_model_params,
            "train_params": fckpls_v2_train_params_search,
        },
        "train_params": fckpls_v2_train_params_final,
    },
    {
        "model": FCKPLS(),
        "name": "FCK-PLS-default",
        "finetune_params": {
            "n_trials": 200,
            "sample": "tpe",
            "verbose": 0,
            "approach": "single",
            "model_params": fckpls_params,
        },
    },
]


# ==============================================================================
# Select which pipeline to run (v1 or v2)
# ==============================================================================

result = nirs4all.run(
    pipeline=[pipeline_pls_pp, pipeline_fck_pls],
    dataset=[synthetic_dataset_small, synthetic_dataset_mid, synthetic_dataset_large],
    verbose=1,
    random_state=42
)

print(f"\nNumber of model configurations: {result.num_predictions}")
print(f"Best RMSE: {result.best_score:.4f}")


# Alternative: Use return_grouped=True for dict output
print("\nGrouped results (dict format):")
grouped = result.top(2, display_metrics=['rmse', 'r2'], group_by=['dataset_name', 'model_name'], return_grouped=True)
for group_key, preds in grouped.items():
    print(f"  {group_key[0]}:")
    for i, pred in enumerate(preds, 1):
        pp_chain = pred.get('preprocessings', 'N/A')
        print(f"    {i}. {pred.get('model_name', 'Unknown')}: RMSE={pred.get('rmse', 0):.4f}, {pp_chain}")