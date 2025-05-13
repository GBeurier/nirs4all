"""
Integration tests for TensorFlow model finetuning using Nirs4all API.
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
from nirs4all.core.utils import framework
from nirs4all.presets.ref_models import nicon, nicon_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

@framework('tensorflow')
def custom_tf_regression(input_shape, params={}):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input, Flatten, Dropout
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(params.get('units1', 32), activation=params.get('activation1', "relu")))
    model.add(Dropout(params.get('dropout1', 0.1)))
    model.add(Dense(params.get('units2', 64), activation=params.get('activation2', "relu")))
    model.add(Dropout(params.get('dropout2', 0.2)))
    model.add(Flatten())
    model.add(Dense(params.get('units3', 16), activation=params.get('activation3', "relu")))
    model.add(Dense(1, activation="linear"))
    return model

@framework('tensorflow')
def custom_tf_classification(input_shape, num_classes=2, params={}):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input, Flatten, Dropout
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(params.get('units1', 32), activation=params.get('activation1', "relu")))
    model.add(Dropout(params.get('dropout1', 0.1)))
    model.add(Dense(params.get('units2', 64), activation=params.get('activation2', "relu")))
    model.add(Dropout(params.get('dropout2', 0.2)))
    model.add(Flatten())
    model.add(Dense(params.get('units3', 16), activation=params.get('activation3', "relu")))
    if num_classes == 2:
        model.add(Dense(1, activation="sigmoid"))
    else:
        model.add(Dense(num_classes, activation="softmax"))
    return model


nicon_finetune = {
    "action": "finetune",
    "finetune_params": {
        "n_trials": 3,
        "model_params": {
            "filters_1": [8, 16, 32, 64],
            "filters_2": [8, 16, 32, 64],
            "filters_3": [8, 16, 32, 64]
        }
    },
    "training_params": {
        "epochs": 3,
        "verbose": 0
    }
}

custom_tf_finetune_regression = {
    "action": "finetune",
    "finetune_params": {
        "n_trials": 3,
        "model_params": {
            "units1": [16, 32, 64, 128],
            "units2": [32, 64, 128, 256],
            "units3": [8, 16, 32, 64],
            "dropout1": [0.1, 0.2, 0.3, 0.5],
            "dropout2": [0.1, 0.2, 0.3, 0.5],
            "activation1": ["relu", "elu", "selu", "tanh"],
            "activation2": ["relu", "elu", "selu", "tanh"],
            "activation3": ["relu", "elu", "selu", "tanh"]
        }
    },
    "training_params": {
        "epochs": 3,
        "verbose": 0
    }
}

custom_tf_finetune_classification = {
    "action": "finetune",
    "task": "classification",
    "finetune_params": {
        "n_trials": 3,
        "model_params": {
            "units1": [16, 32, 64, 128],
            "units2": [32, 64, 128, 256],
            "units3": [8, 16, 32, 64],
            "dropout1": [0.1, 0.2, 0.3, 0.5],
            "dropout2": [0.1, 0.2, 0.3, 0.5],
            "activation1": ["relu", "elu", "selu", "tanh"],
            "activation2": ["relu", "elu", "selu", "tanh"],
            "activation3": ["relu", "elu", "selu", "tanh"]
        }
    },
    "training_params": {
        "epochs": 3,
        "verbose": 0
    }
}

nicon_finetune_classif = {
    "action": "finetune",
    "task": "classification",
    "finetune_params": {
        "n_trials": 3,
        "model_params": {
            "filters_1": [8, 16, 32, 64],
            "filters_2": [8, 16, 32, 64],
            "filters_3": [8, 16, 32, 64]
        }
    },
    "training_params": {
        "epochs": 3,
        "verbose":0
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


@pytest.mark.tensorflow
@pytest.mark.finetune
def test_tensorflow_regression_finetuning():
    """Test finetuning a TensorFlow regression model."""
    try:
        import tensorflow as tf
        
        config1 = Config("sample_data/regression", x_pipeline, y_pipeline, nicon, nicon_finetune, seed)
        
        start = time.time()
        runner = ExperimentRunner([config1], resume_mode="restart")
        datasets, predictions, scores, best_params = runner.run()
        end = time.time()
        print(f"Time elapsed: {end-start} seconds")
        
        # Since we're using a list of configs, get the first dataset
        dataset = datasets[0]
        assert dataset is not None, "Dataset should not be None"
        
        # Check best parameters from finetuning
        best_params_first = best_params[0]
        assert best_params_first is not None, "Best parameters should not be None"
    except (ImportError, ModuleNotFoundError):
        pytest.skip("TensorFlow not available")


@pytest.mark.tensorflow
@pytest.mark.finetune
@pytest.mark.classification
def test_tensorflow_classification_finetuning():
    """Test finetuning a TensorFlow classification model."""
    try:
        import tensorflow as tf
        
        config = Config("sample_data/classification", x_pipeline, None, nicon_classification, nicon_finetune_classif, seed)
        
        start = time.time()
        runner = ExperimentRunner([config], resume_mode="restart")
        datasets, predictions, scores, best_params = runner.run()
        end = time.time()
        print(f"Time elapsed: {end-start} seconds")
        
        # Since we're using a list of configs, get the first dataset
        dataset = datasets[0]
        assert dataset is not None, "Dataset should not be None"
        
        # Check best parameters from finetuning
        best_params_first = best_params[0]
        assert best_params_first is not None, "Best parameters should not be None"
    except (ImportError, ModuleNotFoundError):
        pytest.skip("TensorFlow not available")


@pytest.mark.tensorflow
@pytest.mark.finetune
@pytest.mark.classification
def test_tensorflow_binary_finetuning():
    """Test finetuning a TensorFlow binary classification model using preset model."""
    try:
        import tensorflow as tf
        config = Config("sample_data/binary", x_pipeline, None, nicon_classification, nicon_finetune_classif, seed)
        runner = ExperimentRunner([config], resume_mode="restart")
        datasets, _, _, best_params = runner.run()
        dataset = datasets[0]
        assert dataset is not None, "Dataset should not be None"
        best_params_first = best_params[0]
        assert best_params_first is not None, "Best parameters should not be None"
    except (ImportError, ModuleNotFoundError):
        pytest.skip("TensorFlow not available")


@pytest.mark.tensorflow
@pytest.mark.finetune
def test_custom_tf_regression_finetuning():
    """Test finetuning a custom TensorFlow regression model."""
    try:
        import tensorflow as tf
        
        config = Config("sample_data/regression", x_pipeline, y_pipeline, custom_tf_regression, custom_tf_finetune_regression, seed)
        
        start = time.time()
        runner = ExperimentRunner([config], resume_mode="restart")
        datasets, predictions, scores, best_params = runner.run()
        end = time.time()
        print(f"Time elapsed: {end-start} seconds")
        
        # Since we're using a list of configs, get the first dataset
        dataset = datasets[0]
        assert dataset is not None, "Dataset should not be None"
        
        # Check best parameters from finetuning
        best_params_first = best_params[0]
        assert best_params_first is not None, "Best parameters should not be None"
    except (ImportError, ModuleNotFoundError):
        pytest.skip("TensorFlow not available")


@pytest.mark.tensorflow
@pytest.mark.finetune
@pytest.mark.classification
def test_custom_tf_classification_finetuning():
    """Test finetuning a custom TensorFlow classification model."""
    try:
        import tensorflow as tf
        
        config = Config("sample_data/classification", x_pipeline, None, custom_tf_classification, custom_tf_finetune_classification, seed)
        
        start = time.time()
        runner = ExperimentRunner([config], resume_mode="restart")
        datasets, predictions, scores, best_params = runner.run()
        end = time.time()
        print(f"Time elapsed: {end-start} seconds")
        
        # Since we're using a list of configs, get the first dataset
        dataset = datasets[0]
        assert dataset is not None, "Dataset should not be None"
        
        # Check best parameters from finetuning
        best_params_first = best_params[0]
        assert best_params_first is not None, "Best parameters should not be None"
    except (ImportError, ModuleNotFoundError):
        pytest.skip("TensorFlow not available")


@pytest.mark.tensorflow
@pytest.mark.finetune
@pytest.mark.classification
def test_custom_tf_binary_finetuning():
    """Test finetuning a TensorFlow binary classification model using custom model."""
    try:
        import tensorflow as tf
        config = Config("sample_data/binary", x_pipeline, None, custom_tf_classification, custom_tf_finetune_classification, seed)
        runner = ExperimentRunner([config], resume_mode="restart")
        datasets, _, _, best_params = runner.run()
        dataset = datasets[0]
        assert dataset is not None, "Dataset should not be None"
        best_params_first = best_params[0]
        assert best_params_first is not None, "Best parameters should not be None"
    except (ImportError, ModuleNotFoundError):
        pytest.skip("TensorFlow not available")
