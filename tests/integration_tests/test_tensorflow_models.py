"""
Integration tests for TensorFlow models using Nirs4all API.
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

parent_dir = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(parent_dir)

from nirs4all.core.runner import ExperimentRunner
from nirs4all.core.config import Config
from nirs4all.core.utils import framework
from nirs4all.presets.ref_models import nicon, nicon_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Define models

@framework('tensorflow')
def custom_tf_regression(input_shape, params={}):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input, Flatten
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(32, activation="relu"))
    model.add(Flatten())
    model.add(Dense(1, activation="linear"))
    return model

@framework('tensorflow')
def custom_tf_classification(input_shape, num_classes=2, params={}):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input, Flatten
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(32, activation="relu"))
    model.add(Flatten())
    if num_classes == 2:
        model.add(Dense(1, activation="sigmoid"))
    else:
        model.add(Dense(num_classes, activation="softmax"))
    return model

# Training parameters
train_params = {
    "action": "train", 
    "training_params": {
        "epochs": 10, 
        "batch_size": 32, 
        "patience": 5
    }
}

class_train_params = {
    "action": "train",
    "task": "classification", 
    "training_params": {
        "epochs": 10, 
        "batch_size": 32, 
        "patience": 5
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

# ---------------------------------------------------------------------------
# Tests

@pytest.mark.tensorflow
def test_tensorflow_regression():
    """Test running a TensorFlow regression model using both preset and custom model."""
    try:
        import tensorflow as tf
    except (ImportError, ModuleNotFoundError):
        pytest.skip("TensorFlow not available")
    
    # Test with preset model (nicon)
    for model in [nicon, custom_tf_regression]:
        config = Config("sample_data/regression", x_pipeline, y_pipeline, model, train_params, seed)
        start = time.time()
        runner = ExperimentRunner([config], resume_mode="restart")
        datasets, predictions, scores, best_params = runner.run()
        end = time.time()
        print(f"Time elapsed: {end-start} seconds")
        
        # Since we're using a list of configs, get the first dataset
        dataset = datasets[0]
        assert dataset is not None, "Dataset should not be None"


@pytest.mark.tensorflow
@pytest.mark.classification
def test_tensorflow_classification():
    """Test running a TensorFlow classification model using both preset and custom model."""
    try:
        import tensorflow as tf
    except (ImportError, ModuleNotFoundError):
        pytest.skip("TensorFlow not available")
    
    for model_config in [nicon_classification, custom_tf_classification]:
        config = Config("sample_data/classification", x_pipeline, None, model_config, class_train_params, seed)
        start = time.time()
        runner = ExperimentRunner([config], resume_mode="restart")
        datasets, predictions, scores, best_params = runner.run()
        end = time.time()
        print(f"Time elapsed: {end-start} seconds")
        
        # Since we're using a list of configs, get the first dataset
        dataset = datasets[0]
        assert dataset is not None, "Dataset should not be None"


@pytest.mark.tensorflow
@pytest.mark.classification
def test_tensorflow_binary():
    """Test running a TensorFlow classification model using both preset and custom model."""
    try:
        import tensorflow as tf
    except (ImportError, ModuleNotFoundError):
        pytest.skip("TensorFlow not available")
    
    for model_config in [nicon_classification, custom_tf_classification]:
        config = Config("sample_data/binary", x_pipeline, None, model_config, class_train_params, seed)
        start = time.time()
        runner = ExperimentRunner([config], resume_mode="restart")
        datasets, predictions, scores, best_params = runner.run()
        end = time.time()
        print(f"Time elapsed: {end-start} seconds")
        
        # Since we're using a list of configs, get the first dataset
        dataset = datasets[0]
        assert dataset is not None, "Dataset should not be None"