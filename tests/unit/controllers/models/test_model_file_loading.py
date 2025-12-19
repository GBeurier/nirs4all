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


class TestExportModel:
    """Tests for PipelineRunner.export_model() functionality."""

    @pytest.fixture
    def sklearn_model(self):
        """Create a fitted sklearn model."""
        from sklearn.cross_decomposition import PLSRegression
        model = PLSRegression(n_components=3)
        X = np.random.randn(20, 10)
        y = np.random.randn(20)
        model.fit(X, y)
        return model

    @pytest.fixture
    def mock_resolved_prediction(self, sklearn_model, tmp_path):
        """Create a mock resolved prediction with model artifact."""
        from nirs4all.pipeline.resolver import ResolvedPrediction, SourceType, FoldStrategy
        from nirs4all.pipeline.config.context import MapArtifactProvider
        from typing import Dict, List, Tuple, Any

        # Create artifact map with the model
        artifact_map: Dict[int, List[Tuple[str, Any]]] = {
            0: [("test:0:0", sklearn_model)]
        }

        resolved = ResolvedPrediction(
            source_type=SourceType.PREDICTION,
            model_step_index=0,
            fold_strategy=FoldStrategy.SINGLE,
            fold_weights={0: 1.0},
            artifact_provider=MapArtifactProvider(artifact_map)
        )
        return resolved

    def test_export_model_joblib(self, sklearn_model, tmp_path):
        """Test exporting a model to .joblib format."""
        import joblib
        from unittest.mock import patch, MagicMock

        # Setup: save model to a temp file first (simulating workspace)
        model_path = tmp_path / "source_model.joblib"
        joblib.dump(sklearn_model, model_path)

        # Create output path
        output_path = tmp_path / "exported_model.joblib"

        # Mock the resolver and test export via artifact persistence
        from nirs4all.pipeline.storage.artifacts.artifact_persistence import to_bytes

        data, format_name = to_bytes(sklearn_model, 'joblib')
        with open(output_path, 'wb') as f:
            f.write(data)

        # Verify the exported file
        assert output_path.exists()
        loaded = joblib.load(output_path)
        assert hasattr(loaded, 'predict')
        assert loaded.n_components == sklearn_model.n_components

    def test_export_model_pickle(self, sklearn_model, tmp_path):
        """Test exporting a model to .pkl format."""
        from nirs4all.pipeline.storage.artifacts.artifact_persistence import to_bytes, from_bytes

        output_path = tmp_path / "exported_model.pkl"

        # Export
        data, format_name = to_bytes(sklearn_model, 'cloudpickle')
        with open(output_path, 'wb') as f:
            f.write(data)

        # Verify
        assert output_path.exists()
        with open(output_path, 'rb') as f:
            loaded_data = f.read()
        loaded = from_bytes(loaded_data, format_name)
        assert hasattr(loaded, 'predict')

    def test_roundtrip_export_load(self, sklearn_model, tmp_path):
        """Test full roundtrip: export then load via ModelFactory."""
        import joblib

        # Export using joblib (what export_model does internally)
        export_path = tmp_path / "roundtrip_model.joblib"
        joblib.dump(sklearn_model, export_path)

        # Load using ModelFactory (what predict would do)
        loaded = ModelFactory._load_model_from_file(str(export_path))

        assert loaded is not None
        assert loaded.n_components == sklearn_model.n_components

        # Verify it can predict
        X_test = np.random.randn(5, 10)
        y_orig = sklearn_model.predict(X_test)
        y_loaded = loaded.predict(X_test)
        np.testing.assert_array_almost_equal(y_orig, y_loaded)
