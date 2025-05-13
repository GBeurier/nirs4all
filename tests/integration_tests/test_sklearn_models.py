"""
Integration tests for scikit-learn models using Nirs4all API.
"""

import pytest
import os
import sys
import time
import warnings
from sklearn.exceptions import ConvergenceWarning

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

parent_dir = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(parent_dir)

from nirs4all.core.runner import ExperimentRunner
from nirs4all.core.config import Config
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Define models
sklearn_reg_model = {
    "class": "sklearn.cross_decomposition.PLSRegression",
    "model_params": {
        "n_components": 10,
    }
}

sklearn_rf_reg_model = {
    "class": "sklearn.ensemble.RandomForestRegressor",
    "model_params": {
        "n_estimators": 50,
        "max_depth": 10,
    }
}

sklearn_class_model = {
    "class": "sklearn.ensemble.RandomForestClassifier",
    "model_params": {
        "n_estimators": 50,
        "max_depth": 10,
    }
}

# Define pipelines
x_pipeline = [
    RobustScaler(), 
    {"split": RepeatedKFold(n_splits=3, n_repeats=1)}, 
    MinMaxScaler()
]

# Dataset and seed configuration
seed = 123459456
y_pipeline = MinMaxScaler()


@pytest.mark.sklearn
def test_sklearn_regression():
    """Test running a scikit-learn regression model."""
    config = Config("sample_data/regression", x_pipeline, y_pipeline, sklearn_reg_model, None, seed)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    datasets, predictions, scores, best_params = runner.run()
    end = time.time()
    
    # Since we're using a list of configs, get the first dataset
    dataset = datasets[0]
    assert dataset is not None, "Dataset should not be None"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.sklearn
def test_sklearn_rf_regression():
    """Test running a scikit-learn RandomForestRegressor model."""
    config = Config("sample_data/regression", x_pipeline, y_pipeline, sklearn_rf_reg_model, None, seed)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    datasets, predictions, scores, best_params = runner.run()
    end = time.time()
    
    # Since we're using a list of configs, get the first dataset
    dataset = datasets[0]
    assert dataset is not None, "Dataset should not be None"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.sklearn
@pytest.mark.classification
def test_sklearn_classification():
    """Test running a scikit-learn classification model."""
    config = Config("sample_data/classification", x_pipeline, None, sklearn_class_model, {"task": "classification"}, seed)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    datasets, predictions, scores, best_params = runner.run()
    end = time.time()
    
    # Since we're using a list of configs, get the first dataset
    dataset = datasets[0]
    assert dataset is not None, "Dataset should not be None"
    print(f"Time elapsed: {end-start} seconds")


@pytest.mark.sklearn
@pytest.mark.classification
def test_sklearn_binary():
    """Test running a scikit-learn classification model."""
    config = Config("sample_data/binary", x_pipeline, None, sklearn_class_model, {"task": "classification"}, seed)
    
    start = time.time()
    runner = ExperimentRunner([config], resume_mode="restart")
    datasets, predictions, scores, best_params = runner.run()
    end = time.time()
    
    # Since we're using a list of configs, get the first dataset
    dataset = datasets[0]
    assert dataset is not None, "Dataset should not be None"
    print(f"Time elapsed: {end-start} seconds")