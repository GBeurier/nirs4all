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
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.operators.transforms import FirstDerivative, StandardNormalVariate
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer

# PLS operators
from nirs4all.operators.models.sklearn.pls import (
    PLSDA, IKPLS, OPLS, OPLSDA, MBPLS, DiPLS, SparsePLS, SIMPLS
)
from nirs4all.operators.models.sklearn.lwpls import LWPLS
from nirs4all.operators.models.sklearn.ipls import IntervalPLS
from nirs4all.operators.models.sklearn.robust_pls import RobustPLS
from nirs4all.operators.models.sklearn.recursive_pls import RecursivePLS
from nirs4all.operators.models.sklearn.kopls import KOPLS
from nirs4all.operators.models.sklearn.nlpls import KernelPLS

# Check if JAX is available for GPU-accelerated models
try:
    import jax
    JAX_AVAILABLE = True
    print("JAX detected - GPU-accelerated PLS models available")
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not installed - using NumPy backend only")

# TODO: Add variable selection operators as they are implemented
# from nirs4all.operators.transforms.feature_selection import VIPSelector, CARSSelector


###############
### REGRESSION PIPELINE ##
###############

print("=" * 60)
print("REGRESSION TEST - PLSRegression + IKPLS + OPLS + MBPLS + SparsePLS + LWPLS + SIMPLS + IntervalPLS + RobustPLS + RecursivePLS + KOPLS + KernelPLS")
print("=" * 60)

# Build regression pipeline
regression_models = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    {"feature_augmentation": [FirstDerivative, StandardNormalVariate]},
    ShuffleSplit(n_splits=3, test_size=0.25),

    # Tier 1: sklearn native PLSRegression
    PLSRegression(n_components=10),
    PLSRegression(n_components=15),

    # Tier 2: IKPLS (Improved Kernel PLS - faster implementation)
    IKPLS(n_components=10, backend='numpy'),
    IKPLS(n_components=15, backend='numpy'),

    # Tier 2: OPLS (Orthogonal PLS - removes Y-orthogonal variation)
    OPLS(n_components=1, pls_components=1, backend='numpy'),
    OPLS(n_components=2, pls_components=1, backend='numpy'),

    # Tier 2: MBPLS (Multiblock PLS)
    MBPLS(n_components=5, backend='numpy'),
    MBPLS(n_components=10, backend='numpy'),

    # Tier 2: SparsePLS (Sparse PLS - variable selection)
    SparsePLS(n_components=5, alpha=0.5, backend='numpy'),
    SparsePLS(n_components=5, alpha=1.0, backend='numpy'),

    # Tier 5: SIMPLS (de Jong 1993 algorithm)
    SIMPLS(n_components=10, backend='numpy'),
    SIMPLS(n_components=15, backend='numpy'),

    # Tier 5: IntervalPLS (iPLS - wavelength interval selection)
    IntervalPLS(n_components=5, n_intervals=10, mode='single', backend='numpy'),
    IntervalPLS(n_components=5, n_intervals=10, mode='forward', backend='numpy'),

    # Tier 5: RobustPLS (Robust PLS - outlier-resistant)
    RobustPLS(n_components=10, weighting='huber', max_iter=50, backend='numpy'),
    RobustPLS(n_components=10, weighting='tukey', max_iter=50, backend='numpy'),

    # Tier 5: RecursivePLS (Recursive PLS - online learning for drifting processes)
    RecursivePLS(n_components=10, forgetting_factor=0.99, backend='numpy'),
    RecursivePLS(n_components=10, forgetting_factor=0.95, backend='numpy'),

    # Tier 6: KOPLS (Kernel OPLS - nonlinear OPLS using kernel methods)
    KOPLS(n_components=5, n_ortho_components=1, kernel='linear', backend='numpy'),
    KOPLS(n_components=5, n_ortho_components=2, kernel='rbf', backend='numpy'),
    KOPLS(n_components=5, n_ortho_components=1, kernel='poly', degree=2, backend='numpy'),

    # Tier 6: KernelPLS (Nonlinear PLS using kernel methods - NL-PLS)
    KernelPLS(n_components=5, kernel='rbf', gamma=0.1, backend='numpy'),
    KernelPLS(n_components=5, kernel='rbf', gamma=1.0, backend='numpy'),
    KernelPLS(n_components=5, kernel='linear', backend='numpy'),

    # Tier 3: LWPLS (Locally-Weighted PLS - local models for nonlinearity) # COMMENTED because very slow
    # LWPLS(n_components=5, lambda_in_similarity=0.5, backend='numpy'),
    # LWPLS(n_components=10, lambda_in_similarity=1.0, backend='numpy'),
]

# Add JAX-accelerated models if available
if JAX_AVAILABLE:
    regression_models.extend([
        # IKPLS with JAX backend for GPU/TPU acceleration
        {"model": IKPLS(n_components=10, backend='jax', algorithm=1), "name": "IKPLS_JAX_10"},
        {"model": IKPLS(n_components=15, backend='jax', algorithm=1), "name": "IKPLS_JAX_15"},

        # OPLS with JAX backend
        {"model": OPLS(n_components=1, pls_components=1, backend='jax'), "name": "OPLS_JAX_1"},
        {"model": OPLS(n_components=2, pls_components=1, backend='jax'), "name": "OPLS_JAX_2"},

        # MBPLS with JAX backend
        {"model": MBPLS(n_components=5, backend='jax'), "name": "MBPLS_JAX_5"},
        {"model": MBPLS(n_components=10, backend='jax'), "name": "MBPLS_JAX_10"},

        # SparsePLS with JAX backend
        {"model": SparsePLS(n_components=5, alpha=0.5, backend='jax'), "name": "SparsePLS_JAX_5_a05"},
        {"model": SparsePLS(n_components=5, alpha=1.0, backend='jax'), "name": "SparsePLS_JAX_5_a10"},

        # SIMPLS with JAX backend
        {"model": SIMPLS(n_components=10, backend='jax'), "name": "SIMPLS_JAX_10"},
        {"model": SIMPLS(n_components=15, backend='jax'), "name": "SIMPLS_JAX_15"},

        # IntervalPLS with JAX backend
        {"model": IntervalPLS(n_components=5, n_intervals=10, mode='single', backend='jax'), "name": "iPLS_JAX_single"},
        {"model": IntervalPLS(n_components=5, n_intervals=10, mode='forward', backend='jax'), "name": "iPLS_JAX_forward"},

        # RobustPLS with JAX backend
        {"model": RobustPLS(n_components=10, weighting='huber', max_iter=50, backend='jax'), "name": "RobustPLS_JAX_huber"},
        {"model": RobustPLS(n_components=10, weighting='tukey', max_iter=50, backend='jax'), "name": "RobustPLS_JAX_tukey"},

        # RecursivePLS with JAX backend
        {"model": RecursivePLS(n_components=10, forgetting_factor=0.99, backend='jax'), "name": "RecursivePLS_JAX_ff099"},
        {"model": RecursivePLS(n_components=10, forgetting_factor=0.95, backend='jax'), "name": "RecursivePLS_JAX_ff095"},

        # KOPLS with JAX backend
        {"model": KOPLS(n_components=5, n_ortho_components=1, kernel='linear', backend='jax'), "name": "KOPLS_JAX_linear"},
        {"model": KOPLS(n_components=5, n_ortho_components=2, kernel='rbf', backend='jax'), "name": "KOPLS_JAX_rbf"},
        {"model": KOPLS(n_components=5, n_ortho_components=1, kernel='poly', degree=2, backend='jax'), "name": "KOPLS_JAX_poly"},

        # KernelPLS with JAX backend (Nonlinear PLS / NL-PLS)
        {"model": KernelPLS(n_components=5, kernel='rbf', gamma=0.1, backend='jax'), "name": "KernelPLS_JAX_rbf_g01"},
        {"model": KernelPLS(n_components=5, kernel='rbf', gamma=1.0, backend='jax'), "name": "KernelPLS_JAX_rbf_g10"},
        {"model": KernelPLS(n_components=5, kernel='linear', backend='jax'), "name": "KernelPLS_JAX_linear"},

        # LWPLS with JAX backend
        {"model": LWPLS(n_components=5, lambda_in_similarity=0.5, backend='jax'), "name": "LWPLS_JAX_5"},
        {"model": LWPLS(n_components=10, lambda_in_similarity=1.0, backend='jax'), "name": "LWPLS_JAX_10"},
    ])

# Note: DiPLS is excluded from pipeline examples because it uses
# Hankelization which returns fewer predictions than input samples.
# Use DiPLS directly for time-series/process data applications.

regression_pipeline = regression_models


###############
### REGRESSION DATA ######
###############

regression_data = 'sample_data/regression'
regression_dataset = DatasetConfigs(regression_data)


###############
### REGRESSION RUN #######
###############

regression_config = PipelineConfigs(regression_pipeline, "Q19_regression")
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
reg_predictions, _ = runner.run(regression_config, regression_dataset)

print("\nTop 5 regression models:")
for idx, pred in enumerate(reg_predictions.top(5, 'rmse')):
    print(f"{idx+1}. {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])}")


###############
### CLASSIFICATION PIPELINE ##
###############

# print("\n" + "=" * 60)
# print("CLASSIFICATION TEST - PLSDA + OPLSDA")
# print("=" * 60)

# classification_pipeline = [
#     StandardScaler(),
#     {"feature_augmentation": [FirstDerivative, StandardNormalVariate]},
#     ShuffleSplit(n_splits=3, test_size=0.25),

#     # Tier 1: PLSDA (PLS Discriminant Analysis)
#     PLSDA(n_components=5),
#     PLSDA(n_components=10),

#     # Tier 2: OPLSDA (Orthogonal PLS-DA)
#     OPLSDA(n_components=1, pls_components=5),
#     OPLSDA(n_components=2, pls_components=5),
# ]


# ###############
# ### CLASSIFICATION DATA ######
# ###############

# classification_data = {
#     'folder': 'sample_data/binary/',
#     'params': {
#         'has_header': False,
#         'delimiter': ';',
#         'decimal_separator': '.'
#     }
# }
# classification_dataset = DatasetConfigs([classification_data])


# ###############
# ### CLASSIFICATION RUN #######
# ###############

# classification_config = PipelineConfigs(classification_pipeline, "Q19_classification")
# runner_cls = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
# cls_predictions, _ = runner_cls.run(classification_config, classification_dataset)

# print("\nTop 5 classification models:")
# for idx, pred in enumerate(cls_predictions.top(5, 'accuracy')):
#     print(f"{idx+1}. {Predictions.pred_short_string(pred, metrics=['accuracy', 'balanced_accuracy'])}")


# ###############
# ### SUMMARY ###
# ###############

# print("\n" + "=" * 60)
# print("SUMMARY")
# print("=" * 60)
# print(f"Regression models evaluated:     {len(reg_predictions)}")
# print(f"Classification models evaluated: {len(cls_predictions)}")


# ###############
# ### VISUALIZATION ###
# ###############

# if args.show:
#     print("\nGenerating plots...")

#     # Regression plots
#     reg_analyzer = PredictionAnalyzer(reg_predictions)
#     reg_analyzer.plot_top_k(k=3, rank_metric='rmse')
#     reg_analyzer.plot_candlestick(variable="model_name")

#     # Classification plots
#     cls_analyzer = PredictionAnalyzer(cls_predictions)
#     cls_analyzer.plot_top_k(k=3, rank_metric='accuracy')
#     cls_analyzer.plot_candlestick(variable="model_name", display_metric='accuracy')

#     plt.show()
