"""
Integration tests for processing pipeline.
These tests verify that the processing pipeline works correctly without model training.
"""

import pytest
import os
import sys
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

parent_dir = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(parent_dir)

from nirs4all.presets.ref_models import decon, nicon, customizable_nicon, nicon_classification
from nirs4all.presets.preprocessings import decon_set, nicon_set
from nirs4all.data_splitters import KennardStoneSplitter
from nirs4all.transformations import StandardNormalVariate as SNV, SavitzkyGolay as SG, Gaussian as GS, Derivate as Dv
from nirs4all.transformations import Rotate_Translate as RT, Spline_X_Simplification as SXS, Random_X_Operation as RXO
from nirs4all.transformations import CropTransformer
from nirs4all.core.runner import ExperimentRunner
from nirs4all.core.config import Config

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, ShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Define pipelines for processing
x_pipeline_full = [
    RobustScaler(),
    {"samples": [None, None, None, None, SXS, RXO]},
    {"split": RepeatedKFold(n_splits=3, n_repeats=1)},
    {"features": [None, GS(2,1), SG, SNV, Dv, [GS, SNV], [GS, GS], [GS, SG], [SG, SNV], [GS, Dv], [SG, Dv]]},
    MinMaxScaler()
]

x_pipeline_with_augmentation = [
    RobustScaler(),
    {"samples": [None, SXS(), RXO()]},
    {"features": [None, GS(), SG(), SNV(), [GS(), SNV()]]},
    MinMaxScaler()
]

y_pipeline = MinMaxScaler()
seed = 123459456

@pytest.mark.preprocessing
def test_processing_pipeline():
    """Test data processing pipeline without model training."""
    config = Config("sample_data/binary", x_pipeline_full, y_pipeline, None, None, seed)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    datasets, predictions, scores, best_params = runner.run()
    end = time.time()
    print(f"Time elapsed: {end-start} seconds")
    
    # Since we're using a list of configs, get the first dataset
    dataset = datasets[0]
    assert dataset is not None, "Dataset should not be None"
    x_train = dataset.x_train
    y_train = dataset.y_train
    assert x_train is not None, "X_train should not be None"
    assert y_train is not None, "Y_train should not be None"
    assert len(x_train) > 0, "X_train should not be empty"
    assert len(y_train) > 0, "Y_train should not be empty"
    folds = dataset.folds
    assert folds is not None, "Folds should not be None"
    assert len(folds) > 0, "Folds should not be empty"
    assert len(folds) == 3, "Number of folds should be 3"

@pytest.mark.preprocessing
def test_basic_processing_pipeline():
    """Test basic processing pipeline without model training."""
    config = Config("sample_data/classification", x_pipeline_with_augmentation, y_pipeline, None, None, seed)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    datasets, predictions, scores, best_params = runner.run()
    end = time.time()
    print(f"Time elapsed: {end-start} seconds")
    
    # Since we're using a list of configs, get the first dataset
    dataset = datasets[0]
    assert dataset is not None, "Dataset should not be None"
    x_train = dataset.x_train
    y_train = dataset.y_train
    assert x_train is not None, "X_train should not be None"
    assert y_train is not None, "Y_train should not be None"
    assert len(x_train) > 0, "X_train should not be empty"
    assert len(y_train) > 0, "Y_train should not be empty"