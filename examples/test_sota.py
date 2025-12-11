"""
Q1 Example - Basic Regression Pipeline with PLS Models
=====================================================
Demonstrates NIRS regression analysis using PLS models with various preprocessing techniques.
Features automated hyperparameter tuning for n_components and comprehensive result visualization.
"""

# Standard library imports
import argparse
import time
import matplotlib.pyplot as plt

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.operators.transforms import (
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from catboost import CatBoostRegressor

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.analysis import TransferPreprocessingSelector
from nirs4all.operators.models.sklearn import OPLS, IKPLS
from nirs4all.operators.models.sklearn.lwpls import LWPLS
from nirs4all.operators.splitters import SPXYGFold
from nirs4all.operators.transforms import (
    Wavelet, WaveletFeatures, WaveletPCA, WaveletSVD,
    StandardNormalVariate, FirstDerivative, SecondDerivative, SavitzkyGolay,
    MultiplicativeScatterCorrection, Detrend, Gaussian, Haar,
    RobustStandardNormalVariate,
)
from nirs4all.operators.transforms import (
    AirPLS,      # Adaptive Iteratively Reweighted PLS
    ArPLS,       # Asymmetrically Reweighted PLS
    ASLSBaseline, # Asymmetric Least Squares
    IASLS,       # Improved ASLS
    IModPoly,    # Improved Modified Polynomial
    ModPoly,     # Modified Polynomial
    SNIP,        # Statistics-sensitive Non-linear Iterative Peak-clipping
    RollingBall, # Rolling ball algorithm
    BEADS,       # Baseline Estimation And Denoising with Sparsity
)

from nirs4all.operators.transforms.nirs import (
    AreaNormalization,
    ExtendedMultiplicativeScatterCorrection as EMSC, ReflectanceToAbsorbance, ASLSBaseline,
)
from nirs4all.operators.models.pytorch.nicon import nicon, customizable_nicon, thin_nicon, nicon_VG
from autogluon.tabular import TabularDataset, TabularPredictor
from nirs4all.operators.splitters import SPXYGFold, BinnedStratifiedGroupKFold

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q1 Regression Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()

import os
from pathlib import Path
from dotenv import load_dotenv
from tabpfn import TabPFNClassifier, TabPFNRegressor
from huggingface_hub import login

# Load .env file from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# Hugging Face login for TabPFN
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found in .env or environment. TabPFN may not work properly.")


# Configuration variables
data_path = 'sample_data/regression'


pipeline = [
    SPXYGFold(n_splits=5, random_state=42),
    ASLSBaseline(),
    StandardScaler(),
    SavitzkyGolay(),
    PCA(n_components=50, random_state=42, whiten=True),
    StandardScaler(),
    # ShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
    {"y_processing": StandardScaler()},
    {
        'model': {
            'framework': 'autogluon',
            'params': {
                'presets': 'extreme_quality',
                'time_limit': 3600,
                'num_bag_folds': 5,
            }
        },
        "name": "AutoGluon",
    },
]

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "test")
dataset_config = DatasetConfigs(data_path)


# Run the pipeline
runner = PipelineRunner(save_files=True, verbose=0, plots_visible=args.plots)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)


# Analysis and visualization
start_time = time.time()
best_model_count = 5
ranking_metric = 'rmse'  # Options: 'rmse', 'mae', 'r2'

# Display top performing models
top_models = predictions.top(best_model_count, ranking_metric)
print(f"Top models display took: {time.time() - start_time:.2f}s")
start_time = time.time()
for idx, prediction in enumerate(top_models):
    print(f"{idx+1}. {Predictions.pred_short_string(prediction, metrics=[ranking_metric])} - {prediction['preprocessings']}")
print(f"Print display took: {time.time() - start_time:.2f}s")
start_time = time.time()


if args.show:
    plt.show()