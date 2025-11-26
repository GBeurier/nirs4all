"""
Q19 PLS Test - PLS Methods Integration Test
============================================
Minimal example to test PLS method operators as they are integrated.
Add new PLS operators here after implementation.

Usage:
    python Q19_pls_test.py --plots --show
"""

# Standard library imports
import argparse
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q19 PLS Test')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()


###############
### IMPORTS ###
###############

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit, GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.operators.transforms import FirstDerivative, StandardNormalVariate
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer

# PLS operators
from nirs4all.operators.models.sklearn import (
    PLSDA, IKPLS, OPLS, OPLSDA, MBPLS, DiPLS, SparsePLS, SIMPLS
)
from nirs4all.operators.models.sklearn.lwpls import LWPLS
from nirs4all.operators.models.sklearn.ipls import IntervalPLS
from nirs4all.operators.models.sklearn.robust_pls import RobustPLS
from nirs4all.operators.models.sklearn.recursive_pls import RecursivePLS
from nirs4all.operators.models.sklearn.kopls import KOPLS
from nirs4all.operators.models.sklearn.nlpls import KernelPLS
from nirs4all.operators.models.sklearn.oklmpls import OKLMPLS, PolynomialFeaturizer
from nirs4all.operators.models.sklearn.fckpls import FCKPLS
from nirs4all.operators.transforms import (
    Detrend, FirstDerivative as FstDer, SecondDerivative as SndDer,
    Gaussian as Gauss, StandardNormalVariate as SNV, SavitzkyGolay as SavGol,
    Haar, MultiplicativeScatterCorrection as MSC, Derivate,
    RobustStandardNormalVariate as RSNV, LocalStandardNormalVariate as LSNV, Wavelet
)
from nirs4all.operators.transforms.nirs import (
    AreaNormalization, ExtendedMultiplicativeScatterCorrection as EMSC
)

import jax

# Check if JAX is available for GPU-accelerated models

# TODO: Add variable selection operators as they are implemented
# from nirs4all.operators.transforms.feature_selection import VIPSelector, CARSSelector


###############
### REGRESSION PIPELINE ##
###############
# Build regression pipeline
pipeline = [
    {"y_processing": MinMaxScaler()},

    {"feature_augmentation": {
        "_or_": [Detrend, FstDer, SndDer, Gauss, SNV, SavGol, Haar, MSC, Derivate, RSNV, LSNV, Wavelet, AreaNormalization, EMSC],
        "size": [(1, 3), (1, 2)],
        "count": 50}
     },  # Generate combinations of preprocessing techniques
    MinMaxScaler(),

    {"split": GroupKFold(n_splits=3, shuffle=True, random_state=42), "group": "ID"},

    {
        "model": PLSRegression(n_components=10),
        "name": "PLS-Finetuned",
        "finetune_params": {
            "n_trials": 30,
            "verbose": 2,                           # 0=silent, 1=basic, 2=detailed
            "approach": "single",                                  # "grouped", "individual", or "single"
            "eval_mode": "best",                    # "best" or "avg" (for grouped approach)
            "sample": "grid",                       # "random", "grid", "bayes", "hyperband", "skopt", "tpe", "cmaes"
            "model_params": {
                'n_components': ('int', 1, 30),
                'algorithm': 1,
                "backend": 'jax'
            },
        }
    },

    {
        "model": IKPLS(n_components=10, backend='jax', algorithm=1),
        "name": "PLS-Finetuned",
        "finetune_params": {
            "n_trials": 30,
            "verbose": 2,                           # 0=silent, 1=basic, 2=detailed
            "approach": "single",                                  # "grouped", "individual", or "single"
            "eval_mode": "best",                    # "best" or "avg" (for grouped approach)
            "sample": "grid",                       # "random", "grid", "bayes", "hyperband", "skopt", "tpe", "cmaes"
            "model_params": {
                'n_components': ('int', 1, 30),
                'algorithm': 1,
                "backend": 'jax'
            },
        }
    },
    {
        "model": IKPLS(n_components=10, backend='jax', algorithm=1),
        "name": "PLS-Finetuned",
        "finetune_params": {
            "n_trials": 30,
            "verbose": 2,                           # 0=silent, 1=basic, 2=detailed
            "approach": "single",                                  # "grouped", "individual", or "single"
            "eval_mode": "best",                    # "best" or "avg" (for grouped approach)
            "sample": "grid",                       # "random", "grid", "bayes", "hyperband", "skopt", "tpe", "cmaes"
            "model_params": {
                'n_components': ('int', 1, 30),
                'algorithm': 2,
                "backend": 'jax'
            },
        }
    },
    {
        "model": LWPLS(n_components=10, lambda_in_similarity=0.5, backend='jax'),
        "name": "LWPLS-Finetuned",
        "finetune_params": {
            "n_trials": 60,
            "verbose": 2,
            "sample": "tpe",
            "approach": "single",
            "model_params": {
                "n_components": ('int', 1, 30),
                "lambda_in_similarity": (0.0, 1.0),
                "backend": 'jax'
            }
        }
    },
    {
        "model": MBPLS(n_components=10, backend='jax'),
        "name": "MBPLS-Finetuned",
        "finetune_params": {
            "n_trials": 30,
            "verbose": 2,
            "sample": "grid",
            "approach": "single",
            "model_params": {
                "n_components": ('int', 1, 30),
                "backend": 'jax'
            }
        }
    },
    {
        "model": SIMPLS(n_components=15, backend='jax'),
        "name": "SIMPLS-Finetuned",
        "finetune_params": {
            "n_trials": 30,
            "verbose": 2,
            "sample": "hyperband",
            "approach": "single",
            "model_params": {
                "n_components": ('int', 1, 30),
                "backend": 'jax'
            }
        }
    },
    {
        "model": IntervalPLS(n_components=5, n_intervals=10, mode='forward', backend='jax'),
        "name": "iPLS-Finetuned",
        "finetune_params": {
            "n_trials": 50,
            "verbose": 2,
            "sample": "tpe",
            "approach": "single",
            "model_params": {
                "n_components": ('int', 1, 30),
                "n_intervals": ('int', 5, 20),
                "mode": ['forward', 'backward'],
                "backend": 'jax'
            }
        }
    },
    {
        "model": FCKPLS(n_components=5, alphas=(0.0, 1.0, 2.0), sigmas=(2.0,), kernel_size=15, backend='jax'),
        "name": "FCKPLS-Finetuned",
        "finetune_params": {
            "n_trials": 50,
            "verbose": 2,
            "sample": "grid",
            "approach": "single",
            "model_params": {
                "n_components": ('int', 1, 30),
                "alphas": [(0.0, 1.0, 2.0)],
                "sigmas": [(2.0,)],
                "kernel_size": ('int', 5, 30),
                "backend": 'jax'
            }
        }
    }
]


regression_data = ['nitro_regression/Digestibility_0.8', 'nitro_regression/Hardness_0.8', 'nitro_regression/Tannin_0.8']
regression_dataset = DatasetConfigs(regression_data)


###############
### REGRESSION RUN #######
###############

regression_config = PipelineConfigs(pipeline, "Q19_regression")
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
reg_predictions, _ = runner.run(regression_config, regression_dataset)

print("\nTop 20 regression models:")
for idx, pred in enumerate(reg_predictions.top(20, 'rmse', rank_partition='test')):
    print(f"{idx+1}. {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])}")

reg_analyzer = PredictionAnalyzer(reg_predictions)
reg_analyzer.plot_top_k(k=10, rank_metric='rmse')
reg_analyzer.plot_candlestick(variable="model_name")
reg_analyzer.plot_heatmap(x_var="model_name", y_var="preprocessings")
