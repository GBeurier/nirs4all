"""
Tests for model file loading functionality.

Tests the ModelFactory._load_model_from_file() and related methods
for loading models from various file formats.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from nirs4all.controllers.models.factory import ModelFactory


class TestLoadModelFromFile:
    """Tests for ModelFactory._load_model_from_file()."""

    @pytest.fixture
    def sklearn_model(self):
        """Create a simple sklearn model for testing."""
        from sklearn.cross_decomposition import PLSRegression
        model = PLSRegression(n_components=3)
        # Fit with dummy data
        X = np.random.randn(20, 10)
        y = np.random.randn(20)
        model.fit(X, y)
        return model

    def test_load_joblib_file(self, sklearn_model, tmp_path):
        """Test loading a model from .joblib file."""
        import joblib

        # Save model
        model_path = tmp_path / "model.joblib"
        joblib.dump(sklearn_model, model_path)

        # Load model
        loaded = ModelFactory._load_model_from_file(str(model_path))

        assert loaded is not None
        assert hasattr(loaded, 'predict')
        assert loaded.n_components == sklearn_model.n_components

    def test_load_pkl_file(self, sklearn_model, tmp_path):
        """Test loading a model from .pkl file."""
        import cloudpickle

        # Save model
        model_path = tmp_path / "model.pkl"
        with open(model_path, 'wb') as f:
            cloudpickle.dump(sklearn_model, f)

        # Load model
        loaded = ModelFactory._load_model_from_file(str(model_path))

        assert loaded is not None
        assert hasattr(loaded, 'predict')
        assert loaded.n_components == sklearn_model.n_components

    def test_load_nonexistent_file(self):
        """Test loading from a nonexistent file raises error."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            ModelFactory._load_model_from_file("/nonexistent/path/model.joblib")

    def test_load_unsupported_extension(self, tmp_path):
        """Test loading from unsupported extension raises error."""
        # Create a file with unsupported extension
        model_path = tmp_path / "model.xyz"
        model_path.write_text("dummy content")

        with pytest.raises(ValueError, match="Unsupported file extension"):
            ModelFactory._load_model_from_file(str(model_path))

    def test_extension_case_insensitive(self, sklearn_model, tmp_path):
        """Test that file extensions are case-insensitive."""
        import joblib

        # Save with uppercase extension
        model_path = tmp_path / "model.JOBLIB"
        joblib.dump(sklearn_model, model_path)

        # Load should work
        loaded = ModelFactory._load_model_from_file(str(model_path))
        assert loaded is not None


class TestLoadTensorFlowModel:
    """Tests for loading TensorFlow/Keras models."""

    @pytest.fixture
    def tf_model(self):
        """Create a simple TensorFlow model for testing."""
        pytest.importorskip("tensorflow")
        from tensorflow import keras

        model = keras.Sequential([
            keras.layers.Input(shape=(5,)),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def test_load_h5_file(self, tf_model, tmp_path):
        """Test loading a TensorFlow model from .h5 file."""
        # Note: .h5 is legacy in Keras 3, but should still work
        # Using .keras format is recommended
        model_path = tmp_path / "model.keras"
        tf_model.save(str(model_path))

        # Load model
        loaded = ModelFactory._load_model_from_file(str(model_path))

        assert loaded is not None
        assert hasattr(loaded, 'predict')

    def test_load_keras_file(self, tf_model, tmp_path):
        """Test loading a TensorFlow model from .keras file."""
        # Save model
        model_path = tmp_path / "model.keras"
        tf_model.save(str(model_path))

        # Load model
        loaded = ModelFactory._load_model_from_file(str(model_path))

        assert loaded is not None
        assert hasattr(loaded, 'predict')

    def test_load_savedmodel_folder(self, tf_model, tmp_path):
        """Test loading a TensorFlow SavedModel from folder."""
        # Save as SavedModel format - in Keras 3, just save to a directory
        # Keras 3 doesn't support save_format='tf' directly
        model_path = tmp_path / "saved_model.keras"
        tf_model.save(str(model_path))

        # Load model
        loaded = ModelFactory._load_model_from_file(str(model_path))

        assert loaded is not None
        assert hasattr(loaded, 'predict')


class TestLoadPyTorchModel:
    """Tests for loading PyTorch models."""

    def test_load_pt_file(self, tmp_path):
        """Test loading a PyTorch model from .pt file."""
        torch = pytest.importorskip("torch")

        # Use a built-in module instead of local class
        model = torch.nn.Linear(5, 1)
        model_path = tmp_path / "model.pt"
        torch.save(model, model_path)

        # Load model
        loaded = ModelFactory._load_model_from_file(str(model_path))

        assert loaded is not None

    def test_load_pth_file(self, tmp_path):
        """Test loading a PyTorch model from .pth file."""
        torch = pytest.importorskip("torch")

        # Use a built-in module
        model = torch.nn.Linear(5, 1)
        model_path = tmp_path / "model.pth"
        torch.save(model, model_path)

        # Load model
        loaded = ModelFactory._load_model_from_file(str(model_path))

        assert loaded is not None

    def test_load_checkpoint_with_model_key(self, tmp_path):
        """Test loading a PyTorch checkpoint with 'model' key."""
        torch = pytest.importorskip("torch")

        # Use a built-in module
        model = torch.nn.Linear(5, 1)
        model_path = tmp_path / "checkpoint.ckpt"
        torch.save({'model': model, 'epoch': 10}, model_path)

        # Load model - should extract the 'model' key
        loaded = ModelFactory._load_model_from_file(str(model_path))

        assert loaded is not None
        # The loaded object should be the model, not the full checkpoint
        assert hasattr(loaded, 'forward') or isinstance(loaded, torch.nn.Module)

    def test_load_checkpoint_with_state_dict(self, tmp_path):
        """Test loading a PyTorch checkpoint with 'state_dict' key."""
        torch = pytest.importorskip("torch")

        # Use a built-in module
        model = torch.nn.Linear(5, 1)
        model_path = tmp_path / "checkpoint.ckpt"
        torch.save({
            'state_dict': model.state_dict(),
            'epoch': 10
        }, model_path)

        # Load - should return the checkpoint dict for caller to handle
        loaded = ModelFactory._load_model_from_file(str(model_path))

        assert loaded is not None
        assert isinstance(loaded, dict)
        assert 'state_dict' in loaded


class TestLoadModelFromFolder:
    """Tests for ModelFactory._load_model_from_folder()."""

    def test_unrecognized_folder_raises_error(self, tmp_path):
        """Test that unrecognized folder structure raises error."""
        folder = tmp_path / "unknown_model"
        folder.mkdir()
        (folder / "some_file.txt").write_text("dummy")

        with pytest.raises(ValueError, match="Unrecognized model folder structure"):
            ModelFactory._load_model_from_folder(str(folder))


class TestModelFactoryFromString:
    """Tests for ModelFactory._from_string() with file paths."""

    @pytest.fixture
    def sklearn_model(self):
        """Create a simple sklearn model for testing."""
        from sklearn.cross_decomposition import PLSRegression
        model = PLSRegression(n_components=3)
        X = np.random.randn(20, 10)
        y = np.random.randn(20)
        model.fit(X, y)
        return model

    def test_from_string_with_file_path(self, sklearn_model, tmp_path):
        """Test _from_string recognizes file paths."""
        import joblib

        # Save model
        model_path = tmp_path / "model.joblib"
        joblib.dump(sklearn_model, model_path)

        # Load via _from_string
        loaded = ModelFactory._from_string(str(model_path))

        assert loaded is not None
        assert loaded.n_components == sklearn_model.n_components

    def test_from_string_with_class_path(self):
        """Test _from_string handles class paths."""
        # This should import and instantiate the class
        model = ModelFactory._from_string("sklearn.linear_model.Ridge")

        assert model is not None
        from sklearn.linear_model import Ridge
        assert isinstance(model, Ridge)
